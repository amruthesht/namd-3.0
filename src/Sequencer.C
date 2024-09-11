/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/*****************************************************************************
 * $Source: /home/cvs/namd/cvsroot/namd2/src/Sequencer.C,v $
 * $Author: jim $
 * $Date: 2016/08/26 19:40:32 $
 * $Revision: 1.1230 $
 *****************************************************************************/

// The UPPER_BOUND macro is used to eliminate all of the per atom
// computation done for the numerical integration in Sequencer::integrate()
// other than the actual force computation and atom migration.
// The idea is to "turn off" the integration for doing performance
// profiling in order to get an upper bound on the speedup available
// by moving the integration parts to the GPU.
//
// Define it in the Make.config file, i.e. CXXOPTS += -DUPPER_BOUND
// or simply uncomment the line below.
//
//#define UPPER_BOUND

//for gbis debugging; print net force on each atom
#include "common.h"
#define PRINT_FORCES 0

#include "InfoStream.h"
#include "Node.h"
#include "SimParameters.h"
#include "Sequencer.h"
#include "HomePatch.h"
#include "ReductionMgr.h"
#include "CollectionMgr.h"
#include "BroadcastObject.h"
#include "Output.h"
#include "Controller.h"
#include "Broadcasts.h"
#include "Molecule.h"
#include "NamdOneTools.h"
#include "LdbCoordinator.h"
#include "Thread.h"
#include "Random.h"
#include "PatchMap.inl"
#include "ComputeMgr.h"
#include "ComputeGlobal.h"
#include "NamdEventsProfiling.h"
#include <iomanip>
#include "ComputeCUDAMgr.h"
#include "CollectionMaster.h"
#include "IMDOutput.h"

#include "TestArray.h"

#include <algorithm> // Used for sorting

#define MIN_DEBUG_LEVEL 1
//#define DEBUGM
//
// Define NL_DEBUG below to activate D_*() macros in integrate_SOA()
// for debugging.
//
//#define NL_DEBUG
#include "Debug.h"

#if USE_HPM
#define START_HPM_STEP  1000
#define STOP_HPM_STEP   1500
#endif

#include "DeviceCUDA.h"
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#ifdef WIN32
#define __thread __declspec(thread)
#endif
extern __thread DeviceCUDA *deviceCUDA;
#ifdef __IBMCPP__
// IBM compiler requires separate definition for static members
constexpr int CudaLocalRecord::num_inline_peer;
#endif
#endif

#define SPECIAL_PATCH_ID  91

//
// BEGIN
// print_* routines
// assist in debugging SOA integration code
//
static void print_vel_AOS(
    const FullAtom *a,
    int ilo=0, int ihip1=1
    ) {
  printf("AOS Velocities:\n");
  for (int i=ilo;  i < ihip1;  i++) {
    printf("%d     %g  %g  %g\n", i,
        a[i].velocity.x, a[i].velocity.y, a[i].velocity.z);
  }
}


static void print_vel_SOA(
    const double *vel_x,
    const double *vel_y,
    const double *vel_z,
    int ilo=0, int ihip1=1
    ) {
  printf("SOA Velocities:\n");
  for (int i=ilo;  i < ihip1;  i++) {
    printf("%d     %g  %g  %g\n", i, vel_x[i], vel_y[i], vel_z[i]);
  }
}


static void print_tensor(const Tensor& t) {
  printf("%g %g %g  %g %g %g  %g %g %g\n",
      t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz);
}
//
// END
// print_* routines
// assist in debugging SOA integration code
//


///
/// Check the current step number for some periodic event.
/// The idea is to store extra integers to avoid mod operations.
/// Specifically, replace calls like:  ! (step % stepsPerCycle)
/// with:  stepsPerCycle.check(step)
/// or:    stepsPerCycle.init(firstTimestep, simParams->stepsPerCycle)
/// when initializing.
///
/// Set up the struct by calling init with initial step and period.
/// Then call once per time step loop in place of mod operation.
///
struct CheckStep {
  int period;   ///< period for some step dependent event (e.g. stepsPerCycle)
  int nextstep; ///< next step value

  /// Check step to see if current step is special for this event.
  /// Returns nonzero for true, zero for false.
  /// Side effect: might increment nextstep to the next event step.
  inline int check(int step) {
    if (step == nextstep)  return( nextstep += period, 1 );
    else return 0;
  }

  /// Initialize the step checking, e.g., with
  /// simParams->firsttimestep and simParams->stepsPerCycle
  /// Special case set delta for off boundary shift,
  /// needed for half-step-cycle Langevin piston.
  /// Returns nonzero if initial step is divisible by period.
  inline int init(int initstep, int initperiod, int delta=0) {
    period = initperiod;
    nextstep = initstep - (initstep % period) - (delta % period);
    while (nextstep <= initstep) nextstep += period;
    // returns true if initstep is divisible by period
    return (initstep + period == nextstep);
  }

  CheckStep() : period(0), nextstep(0) { }
};


Sequencer::Sequencer(HomePatch *p) :
	simParams(Node::Object()->simParameters),
	patch(p),
	collection(CollectionMgr::Object()),
	ldbSteps(0),
        pairlistsAreValid(0),
        pairlistsAge(0),
        pairlistsAgeLimit(0)
{
    pairlistsAgeLimit =
      (simParams->stepsPerCycle - 1) / simParams->pairlistsPerCycle;
    broadcast = new ControllerBroadcasts(& patch->ldObjHandle);
    reduction = ReductionMgr::Object()->willSubmit(
                  simParams->accelMDOn ? REDUCTIONS_AMD : REDUCTIONS_BASIC );
    min_reduction = ReductionMgr::Object()->willSubmit(REDUCTIONS_MINIMIZER,1);
    if (simParams->pressureProfileOn) {
      int ntypes = simParams->pressureProfileAtomTypes;
      int nslabs = simParams->pressureProfileSlabs;
      pressureProfileReduction =
        ReductionMgr::Object()->willSubmit(
		REDUCTIONS_PPROF_INTERNAL, 3*nslabs*ntypes);
    } else {
      pressureProfileReduction = NULL;
    }
    if (simParams->multigratorOn) {
      multigratorReduction = ReductionMgr::Object()->willSubmit(REDUCTIONS_MULTIGRATOR,MULTIGRATOR_REDUCTION_MAX_RESERVED);
    } else {
      multigratorReduction = NULL;
    }
    ldbCoordinator = (LdbCoordinator::Object());
    random = new Random(simParams->randomSeed);
    random->split(patch->getPatchID()+1,PatchMap::Object()->numPatches()+1);

    // Is soluteScaling enabled?
    if (simParams->soluteScalingOn) {
      // If so, we must "manually" perform charge scaling on startup because
      // Sequencer will not get a scripting task for initial charge scaling.
      // Subsequent rescalings will take place through a scripting task.
      rescaleSoluteCharges(simParams->soluteScalingFactorCharge);
    }

    rescaleVelocities_numTemps = 0;
    stochRescale_count = 0;
    berendsenPressure_count = 0;
    masterThread = true;
//    patch->write_tip4_props();
#if (defined(NAMD_CUDA) || defined(NAMD_HIP)) && defined(SEQUENCER_SOA) && defined(NODEGROUP_FORCE_REGISTER)
    if ( simParams->CUDASOAintegrateMode ){
#if 0
      CUDASequencer = new SequencerCUDA(deviceCUDA->getDeviceID(),
                                        simParams);
#else
      CUDASequencer = SequencerCUDA::InstanceInit(deviceCUDA->getDeviceID(),
                                        simParams);
       
      constructDevicePatchMap();
#endif
      CUDASequencer->patchData->reduction->zero();
    }
#endif
}

Sequencer::~Sequencer(void)
{
    delete broadcast;
    delete reduction;
    delete min_reduction;
    if (pressureProfileReduction) delete pressureProfileReduction;
    delete random;
    if (multigratorReduction) delete multigratorReduction;
#if (defined(NAMD_CUDA) || defined(NAMD_HIP)) && defined(SEQUENCER_SOA) && defined(NODEGROUP_FORCE_REGISTER)
    if ( simParams->CUDASOAintegrateMode ){
      delete CUDASequencer;
      clearDevicePatchMap();
    }
#endif
}

// Invoked by thread
void Sequencer::threadRun(Sequencer* arg)
{
    LdbCoordinator::Object()->startWork(arg->patch->ldObjHandle);
    arg->algorithm();
}

// Invoked by Node::run() via HomePatch::runSequencer()
void Sequencer::run(void)
{
    // create a Thread and invoke it
    DebugM(4, "::run() - this = " << this << "\n" );
    thread = CthCreate((CthVoidFn)&(threadRun),(void*)(this),SEQ_STK_SZ);
    CthSetStrategyDefault(thread);
    priority = PATCH_PRIORITY(patch->getPatchID());
    awaken();
}

void Sequencer::suspend(void)
{
    LdbCoordinator::Object()->pauseWork(patch->ldObjHandle);
    CthSuspend();
    LdbCoordinator::Object()->startWork(patch->ldObjHandle);
}

// Defines sequence of operations on a patch.  e.g. when
// to push out information for Compute objects to consume
// when to migrate atoms, when to add forces to velocity update.
void Sequencer::algorithm(void)
{
  int scriptTask;
  int scriptSeq = 0;
  // Blocking receive for the script barrier.
  while ( (scriptTask = broadcast->scriptBarrier.get(scriptSeq++)) != SCRIPT_END ) {
    switch ( scriptTask ) {
      case SCRIPT_OUTPUT:
	submitCollections(FILE_OUTPUT);
	break;
      case SCRIPT_FORCEOUTPUT:
	submitCollections(FORCE_OUTPUT);
	break;
      case SCRIPT_MEASURE:
	submitCollections(EVAL_MEASURE);
	break;
      case SCRIPT_REINITVELS:
	reinitVelocities();
	break;
      case SCRIPT_RESCALEVELS:
	rescaleVelocitiesByFactor(simParams->scriptArg1);
	break;
      case SCRIPT_RESCALESOLUTECHARGES:
        rescaleSoluteCharges(simParams->soluteScalingFactorCharge);
        break;
      case SCRIPT_RELOADCHARGES:
	reloadCharges();
	break;
      case SCRIPT_CHECKPOINT:
        patch->checkpoint();
        checkpoint_berendsenPressure_count = berendsenPressure_count;
	break;
      case SCRIPT_REVERT:
        patch->revert();
        berendsenPressure_count = checkpoint_berendsenPressure_count;
        pairlistsAreValid = 0;
	break;
      case SCRIPT_CHECKPOINT_STORE:
      case SCRIPT_CHECKPOINT_LOAD:
      case SCRIPT_CHECKPOINT_SWAP:
      case SCRIPT_CHECKPOINT_FREE:
        patch->exchangeCheckpoint(scriptTask,berendsenPressure_count);
	break;
      case SCRIPT_ATOMSENDRECV:
      case SCRIPT_ATOMSEND:
      case SCRIPT_ATOMRECV:
        patch->exchangeAtoms(scriptTask);
        break;
      case SCRIPT_MINIMIZE:
#if 0
        if (simParams->CUDASOAintegrate){
          NAMD_die("Minimization is currently not supported on the GPU integrator\n");
        }
#endif
	      minimize();
	      break;
      case SCRIPT_RUN:
      case SCRIPT_CONTINUE:
  //
  // DJH: Call a cleaned up version of integrate().
  //
  // We could test for simulation options and call a more basic version
  // of integrate() where we can avoid performing most tests.
  //
#ifdef SEQUENCER_SOA
        if ( simParams->SOAintegrateOn ) {
#ifdef NODEGROUP_FORCE_REGISTER

          if(simParams->CUDASOAintegrate) integrate_CUDA_SOA(scriptTask);
          else {
#endif
            integrate_SOA(scriptTask);
#ifdef NODEGROUP_FORCE_REGISTER
          }
#endif
        }
        else
#endif
	integrate(scriptTask);
	break;
      default:
        NAMD_bug("Unknown task in Sequencer::algorithm");
    }
  }
  submitCollections(END_OF_RUN);
  terminate();
}


#ifdef SEQUENCER_SOA

//////////////////////////////////////////////////////////////////////////
//
// begin SOA code
//

#if defined(NODEGROUP_FORCE_REGISTER)

void Sequencer::suspendULTs(){
  PatchMap* patchMap = PatchMap::Object();
  CUDASequencer->numPatchesCheckedIn += 1;
  if (CUDASequencer->numPatchesCheckedIn < patchMap->numPatchesOnNode(CkMyPe())) {
    masterThread = false;
    CUDASequencer->waitingThreads.push_back(CthSelf());
    NAMD_EVENT_STOP(patch->flags.event_on, NamdProfileEvent::INTEGRATE_SOA_1);
    CthSuspend();

    // JM: if a thread get here, it will be for migrating atoms until the end of the simulation
    while(true){
      // read global flags
      int lastStep = CUDASequencer->patchData->flags.step;
      int startup = (CUDASequencer->patchData->flags.step == simParams->firstTimestep);
      if (CUDASequencer->breakSuspends) break;
      if (simParams->useDeviceMigration) {
        this->patch->positionsReady_GPU(true, startup);
      } else {
        this->patch->positionsReady_SOA(true);
      }
      CUDASequencer->numPatchesCheckedIn += 1;
      CUDASequencer->waitingThreads.push_back(CthSelf());
      if(CUDASequencer->numPatchesCheckedIn == patchMap->numPatchesOnNode(CkMyPe()) - 1 &&
          CUDASequencer->masterThreadSleeping){
          CUDASequencer->masterThreadSleeping = false;
          CthAwaken(CUDASequencer->masterThread);
      }
      CthSuspend();
    }
  }
}
void Sequencer::wakeULTs(){
  CUDASequencer->numPatchesCheckedIn = 0;
  for (CthThread t : CUDASequencer->waitingThreads) {
      CthAwaken(t);
  }
  CUDASequencer->waitingThreads.clear();
}

void Sequencer::runComputeObjectsCUDA(int doMigration, int doGlobal, int pairlists, int nstep, int startup) {

  PatchMap* map = PatchMap::Object();
  
  bool isMaster = deviceCUDA->getMasterPe() == CkMyPe();
  CmiNodeBarrier();

  // Sync after the node barrier. This is making sure that the position buffers have been
  // populated. However, this doesn't need to happen at the node level. I.e. the non-pme
  // nonbonded calculations can begin before the PME device is finished setting it's positions.
  // There is a node barrier after the forces are done, so we don't have to worry about
  // the positions being updated before the positions have been set
  if (isMaster) {
    CUDASequencer->sync();
  }
  

  // JM: Each masterPE owns a particular copy of the compute object we need to launch
  //     work on. The goal is to launch work on everyone, but for migration steps, sometimes
  //     there are a few operation that need to be launched on computes owned by different PEs.
  //     ComputeBondedCUDA::openBoxesOnPe() is an example: There is a list of PEs on each compute
  //     which holds information on which proxy object it should also invoke openBoxesOnPe();

  //     We need to be mindful of that and, since we want to launch methods on different computes.
  //     A data structure that holds all nonbonded Computes from all masterPEs is necessary
  ComputeCUDAMgr*       cudaMgr    = ComputeCUDAMgr::getComputeCUDAMgr();
  ComputeBondedCUDA*    cudaBond   = cudaMgr->getComputeBondedCUDA();
  CudaComputeNonbonded* cudaNbond  = cudaMgr->getCudaComputeNonbonded();
  CudaPmeOneDevice*     cudaPme    = ( (patch->flags.doFullElectrostatics && deviceCUDA->getIsPmeDevice()) ?
      cudaMgr->getCudaPmeOneDevice() : NULL );
  int reducePme = ( cudaPme && patch->flags.doVirial );
  // fprintf(stderr, "Patch %d invoking computes\n", this->patch->patchID);


  // JM NOTE: I don't think the scheme below holds for nMasterPes > 1, check it out laters

  // Invoking computes on the GPU //
  if(doMigration){
    // JM: if we're on a migration step, we call the setup functions manually
    // which means:
    //  0.  masterPe->doWork();
    //  1.  openBoxesOnPe();
    //      loadTuplesOnPe();
    //  2.  masterPe->launchWork();
    //  3.  finishPatchesOnPe();
    //  4.  masterPe->finishReductions();

    if(isMaster){
      NAMD_EVENT_START(1, NamdProfileEvent::MIG_ATOMUPDATE);
      cudaNbond->atomUpdate();
      cudaBond->atomUpdate();
      cudaNbond->doWork();
      cudaBond->doWork();
      NAMD_EVENT_STOP(1, NamdProfileEvent::MIG_ATOMUPDATE);

      if (cudaPme && !simParams->useDeviceMigration) CUDASequencer->atomUpdatePme();
    }

    CmiNodeBarrier();

    if (simParams->useDeviceMigration) {
      if(isMaster){
        CUDASequencer->launch_set_compute_positions();
        CUDASequencer->sync(); // TODO move this to tuple migration
      }
      CmiNodeBarrier();
    }

    NAMD_EVENT_START(1, NamdProfileEvent::MIG_OPENBOXESONPE);
    
    // Here we need to do the following, for each Comput
    for(int i = 0 ; i < CkNumPes(); i++){
      // Here I need to find if the PE is on the bonded PE list
      // XXX NOTE: This might be inefficient. Check the overhead later
      ComputeBondedCUDA* b     = CUDASequencer->patchData->cudaBondedList[i];
      CudaComputeNonbonded* nb = CUDASequencer->patchData->cudaNonbondedList[i];
      if (b == NULL) continue;
      auto list = std::find(std::begin(b->getBondedPes()), std::end(b->getBondedPes()), CkMyPe());
      if( list != std::end(b->getBondedPes()) ){
        b->openBoxesOnPe(startup);

        // XXX NOTE: nb has a differente PE list!!! We need a different loop for nb
        nb->openBoxesOnPe();

      }
      CmiNodeBarrier();
    }
    NAMD_EVENT_STOP(1, NamdProfileEvent::MIG_OPENBOXESONPE);
    // for the bonded kernels, there's an additional step here, loadTuplesOnPe
    // JM NOTE: Those are major hotspots, they account for 50% of the migration time.
    CmiNodeBarrier();    
    NAMD_EVENT_START(1, NamdProfileEvent::MIG_LOADTUPLESONPE);
    
    // NOTE: problem here: One of the CompAtomExt structures is turning to null, why?
    cudaBond->loadTuplesOnPe(startup);
    NAMD_EVENT_STOP(1, NamdProfileEvent::MIG_LOADTUPLESONPE);
    CmiNodeBarrier();
    NAMD_EVENT_START(1, NamdProfileEvent::MIG_COPYTUPLEDATA);

    if (simParams->useDeviceMigration) {
      cudaBond->copyTupleDataGPU(startup);
    } else {
      cudaBond->copyTupleDataSN();
    }

    NAMD_EVENT_STOP(1, NamdProfileEvent::MIG_COPYTUPLEDATA);
    // waits until everyone has finished to open their respective boxes
    // node barrier actually prevents the error that is happening.
    CmiNodeBarrier();
    if(isMaster){
      // launches work on the masterPe
      NAMD_EVENT_START(1, NamdProfileEvent::MIG_LAUNCHWORK);
      cudaBond->launchWork();
      cudaNbond->launchWork();
      if (cudaPme) {
        cudaPme->compute(*(CUDASequencer->patchData->lat), reducePme, this->patch->flags.step);
      }
      cudaNbond->reSortTileLists();
      NAMD_EVENT_STOP(1, NamdProfileEvent::MIG_LAUNCHWORK);
    }

    CmiNodeBarrier();
    //global master force calculation

    if(doGlobal) {
      NAMD_EVENT_START(1, NamdProfileEvent::GM_CALCULATE);
      ComputeGlobal *computeGlobal = Node::Object()->computeMgr->computeGlobalObject; 
      // Zero all SOA global forces before computing next global force
      NAMD_EVENT_START(1, NamdProfileEvent::GM_ZERO);
      int numhp = PatchMap::Object()->numHomePatches();
      HomePatchList *hpList = PatchMap::Object()->homePatchList();
      for(int i = 0; i < numhp; ++i) {
        HomePatch *hp = hpList->item(i).patch;
        hp->zero_global_forces_SOA();
      }
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_ZERO);
      NAMD_EVENT_START(1, NamdProfileEvent::GM_DOWORK);
      // call globalmaster to calculate the force from client. 
      computeGlobal->doWork();
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_DOWORK);
      NAMD_EVENT_START(1, NamdProfileEvent::GM_BARRIER);
      CmiNodeBarrier();
      //      CkPrintf("post doWork step %d \n",this->patch->flags.step);
      //      CUDASequencer->printSOAPositionsAndVelocities();
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_BARRIER);
      if(isMaster) {
        // aggregate and copy the global forces to d_f_global device buffer
        NAMD_EVENT_START(1, NamdProfileEvent::GM_CPY_FORCE);
        CUDASequencer->copyGlobalForcesToDevice();
        NAMD_EVENT_STOP(1, NamdProfileEvent::GM_CPY_FORCE);
      }
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_CALCULATE);
    }
    NAMD_EVENT_START(1, NamdProfileEvent::MIG_FINISHPATCHES);
    cudaNbond->finishPatches();
    cudaBond->finishPatches();
    NAMD_EVENT_STOP(1, NamdProfileEvent::MIG_FINISHPATCHES);
    CmiNodeBarrier();

    // finishes reduction with masterPe!
    if(isMaster){
      cudaNbond->finishReductions();
      if (cudaPme) cudaPme->finishReduction(reducePme);
      cudaBond->finishReductions();
    }
    CmiNodeBarrier();
  }
  // if we're not on a migration step, do the work only on masterPE, except globalmaster work
  else {
    int doNbond = patch->flags.doNonbonded;
    if(isMaster) {
      // JM NOTE: We issue the nonbonded work first and sync it last
      if (cudaPme) {
	// cudaPme->compute(this->patch->lattice, reducePme, this->patch->flags.step);
	cudaPme->compute(*(CUDASequencer->patchData->lat), reducePme, this->patch->flags.step);
      }
      if (doNbond) cudaNbond->doWork();
      cudaBond->doWork();
    }
    //global master force calculation
    if(doGlobal) {
      NAMD_EVENT_START(1, NamdProfileEvent::GM_CALCULATE);
      NAMD_EVENT_START(1, NamdProfileEvent::GM_ZERO);
      ComputeGlobal *computeGlobal = Node::Object()->computeMgr->computeGlobalObject;
      // Zero all SOA global forces before computing next global force 
      int numhp = PatchMap::Object()->numHomePatches();
      HomePatchList *hpList = PatchMap::Object()->homePatchList();
      for(int i = 0; i < numhp; ++i) {
        HomePatch *hp = hpList->item(i).patch;
        hp->zero_global_forces_SOA();
      }
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_ZERO);
      // call globalmaster to calculate the force from client.
      NAMD_EVENT_START(1, NamdProfileEvent::GM_DOWORK);
      computeGlobal->doWork();
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_DOWORK);
      NAMD_EVENT_START(1, NamdProfileEvent::GM_BARRIER);
      CmiNodeBarrier();
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_BARRIER);      
      //      CkPrintf("post doWork 2 step %d \n",this->patch->flags.step);
      //      CUDASequencer->printSOAPositionsAndVelocities();
      if(isMaster) {
        // aggregate and copy the global forces to d_f_global device buffer
        NAMD_EVENT_START(1, NamdProfileEvent::GM_CPY_FORCE);
        CUDASequencer->copyGlobalForcesToDevice();
        NAMD_EVENT_STOP(1, NamdProfileEvent::GM_CPY_FORCE);
      }
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_CALCULATE);
    }
    if(isMaster) {
      cudaBond->finishPatches();
      if (cudaPme) {
        cudaPme->finishReduction(reducePme);
      }
      if (doNbond) cudaNbond->finishPatches();
    }
  }

#if 0
  // for migrations, I need to call OpenBoxesOnPe and finishPatches for every Pe
  if ( patch->flags.savePairlists && patch->flags.doNonbonded ) {
    pairlistsAreValid = 1;
    pairlistsAge = 0;
  }
  if ( pairlistsAreValid /* && !pressureStep */ ) ++pairlistsAge;
#endif
  // CmiNodeBarrier();
}

//apply MC pressure control
void Sequencer::monteCarloPressureControl(
  const int step,
  const int doMigration,
  const int doEnergy,
  const int doVirial,
  const int maxForceNumber)
{
  bool isMasterPe = (deviceCUDA->getMasterPe() == CkMyPe() );
  NodeReduction *reduction = CUDASequencer->patchData->reduction;
  Controller *c_out = CUDASequencer->patchData->c_out;
  bool mGpuOn = CUDASequencer->mGpuOn;
  Lattice oldLattice = this->patch->lattice;
  Vector origin = this->patch->lattice.origin();
  Tensor factor; 
  int accepted = 0; // status of MC volume fluctuation trial

  if(isMasterPe){ 
    // Backup the reduction values for rejected move
    CUDASequencer->patchData->nodeReductionSave->setVal(reduction);
 
    if(deviceCUDA->getIsMasterDevice()){
      // Send the rescale factor for Monte Carlo Volume change from controller
      c_out->mcPressure_prepare(step);
      // receive the factor
      factor = c_out->getPositionRescaleFactor(step);
    }
    
    // Backup positions and forces, scale the coordinates and lattice
    // Setup positions for energy and force calculation
    CUDASequencer->monteCarloPressure_part1(factor, origin, oldLattice);
    if(deviceCUDA->getIsMasterDevice()) {
      // Scale the lattice with factor
      // patch.lattice is pointing to patch.flags.lattice
      this->patch->lattice.rescale(factor);
      CUDASequencer->patchData->lat    = &(this->patch->lattice);
      CUDASequencer->patchData->factor = &(factor); 
      // Copy scaled lattic flags to all patches
      CUDASequencer->patchData->flags.copyIntFlags(this->patch->flags);

      // Zero all reduction values. We will add halfStep values, if
      // the move is accepted.
      reduction->zero();
    }
  }

  CmiNodeBarrier();
  if(isMasterPe){ 
    // copy global flags
    CUDASequencer->update_patch_flags();
  }
  int doTcl = simParams->tclForcesOn;
  int doColvars = simParams->colvarsOn;
  const int doGlobal = (doTcl || doColvars);
  CmiNodeBarrier();
  // Calculate the new force and energy after rescaling the coordinates
  // Migration happend before calling this function
  this->runComputeObjectsCUDA(0, doGlobal, 1, step, 0 /* startup */);
  CmiNodeBarrier();

  if(isMasterPe){
    // Accumulate force to SOA, calculate External energy/force
    // reduce energy and virial
    CUDASequencer->monteCarloPressure_part2(reduction, step, maxForceNumber, 
      doEnergy, doVirial);

    if(deviceCUDA->getIsMasterDevice()){
      // Check to see if the move is accepted or not
      c_out->mcPressure_accept(step);
      accepted = c_out->getMCAcceptance(step);
      //accepted = broadcast->monteCarloBarostatAcceptance.get(step);
      //printf("Sequencer (accept): step: %d, Pe: %d, ACC status: %d\n", step, CkMyPe(), accepted);
    }

    if (accepted) { // Move accepted
      CUDASequencer->monteCarloPressure_accept(reduction, doMigration);
    } else { // Move rejected
      if(deviceCUDA->getIsMasterDevice()) {
        // Set the lattice to the original value, before scaling
        this->patch->lattice = oldLattice;
        CUDASequencer->patchData->lat    = &(this->patch->lattice);
        // Copy scaled lattic flags to all patches
        CUDASequencer->patchData->flags.copyIntFlags(this->patch->flags);
      }

      // Restore all positions and forces and cuLattice
      CUDASequencer->monteCarloPressure_reject(this->patch->lattice);
      // Restore the reduction values
      reduction->setVal(CUDASequencer->patchData->nodeReductionSave);
    }
  }

  CmiNodeBarrier();
  //continue the rejection step. Need to update lattice in all patches
  if(isMasterPe && !accepted){ 
    // copy global flags
    CUDASequencer->update_patch_flags();
  }
}

void Sequencer::doMigrationGPU(const int startup, const int doGlobal, 
  const int updatePatchMap) {

  const bool isMasterPe = deviceCUDA->getMasterPe() == CkMyPe();
  const bool updatePatchData = startup || doGlobal || updatePatchMap;
  PatchMap* patchMap = PatchMap::Object();

  bool realloc = false;

  // This will check if a reallocation was done on the previous migration
  // We use the scratch buffers to store the atomic data during reallocation
  // However, the migrationDestination data much be maintained throughout
  // migration (and tuple migration so beyond the scope of this function)
  // We probably should add a function to do this at the end of migration
  // But for now, DMC thought it was easier to just do at the begining
  for (int i = 0; i < deviceCUDA->getNumDevice(); i++) {
    if(CUDASequencer->patchData->atomReallocationFlagPerDevice[i] != 0) {
      realloc = true;
      break;
    }
  }
  if (realloc) {
    if (isMasterPe) {
      CUDASequencer->reallocateMigrationDestination();
      CUDASequencer->registerSOAPointersToHost();
    }
    CmiNodeBarrier(); 
    if (isMasterPe) {
      CUDASequencer->copySOAHostRegisterToDevice();
    } 
  }

  // Proceed with migration
  //
  // Starts GPU migration
  // 
  if (isMasterPe) {
    CUDASequencer->migrationLocalInit();
    // Hidden stream sync
  }
  CmiNodeBarrier();

  if (isMasterPe) {
    CUDASequencer->migrationPerform();
    // Hidden stream sync
  }
  CmiNodeBarrier();

  if (isMasterPe) {
    CUDASequencer->migrationUpdateAtomCounts();
    // Hidden stream sync
  }
  CmiNodeBarrier();

  if (isMasterPe) {
    CUDASequencer->migrationUpdateAtomOffsets();
    // Hidden stream sync
  }
  CmiNodeBarrier();

  if (isMasterPe) {
    CUDASequencer->copyPatchDataToHost();
    // Hidden stream sync
  }
  CmiNodeBarrier();
 
  // Update device buffer allocations
  realloc = false;
  if (isMasterPe) {
    realloc = CUDASequencer->copyPatchData(true, false);
    CUDASequencer->patchData->atomReallocationFlagPerDevice[deviceCUDA->getDeviceIndex()] = realloc;
  }
  CmiNodeBarrier();

  // If any of the devices have reallocated, we need to re-register the p2p buffers
  for (int i = 0; i < deviceCUDA->getNumDevice(); i++) {
    if(CUDASequencer->patchData->atomReallocationFlagPerDevice[i] != 0) {
      realloc = true;
      break;
    }
  }
  if (realloc) {
    if (isMasterPe) {
      CUDASequencer->registerSOAPointersToHost();
    }
    CmiNodeBarrier(); 
    if (isMasterPe) {
      CUDASequencer->copySOAHostRegisterToDevice();
    } 
  }

  // Performs various post processing like Solute/Solvent sorting and copies back to host
  if (isMasterPe) {
    CUDASequencer->migrationLocalPost(0);
    CUDASequencer->migrationSortAtomsNonbonded();
  }

  // If this is startup, we need to delay this until after AoS has been copied back to host
  // Because we do need the atomIDs for the atom map initially
  if (!updatePatchData) {
    wakeULTs(); // Wakes everyone back up for migration
    this->patch->positionsReady_GPU(1, startup);
    if(CUDASequencer->numPatchesCheckedIn < patchMap->numPatchesOnNode(CkMyPe()) -1 ) {
      CUDASequencer->masterThreadSleeping = true;
      CUDASequencer->masterThread = CthSelf();
      CthSuspend();
    }
  }

  if (isMasterPe) {
    CUDASequencer->sync();
  }
  CmiNodeBarrier(); 
  if (isMasterPe) {
    CUDASequencer->migrationUpdateDestination();
  }
  CmiNodeBarrier();

  if (isMasterPe) {
    CUDASequencer->migrationUpdateProxyDestination();
  }
  CmiNodeBarrier();

  if (isMasterPe) {
    CUDASequencer->migrationUpdateRemoteOffsets();
  }
  CmiNodeBarrier();

  if (isMasterPe) {
    CUDASequencer->copyDataToPeers(true);
  }
  CmiNodeBarrier();

  if (updatePatchData) {
    // The atom maps need to be cleared the HomePatch atom arrays have been updated
    int numhp = PatchMap::Object()->numHomePatches();
    HomePatchList *hpList = PatchMap::Object()->homePatchList();
    for(int i = 0; i < numhp; ++i) {
      HomePatch *hp = hpList->item(i).patch;
      hp->clearAtomMap();
    }
    CmiNodeBarrier();
    if (isMasterPe) {
        // We need the atom ordering to be correct within each 
        // patch to setup the atom map. The vdwType of each atom
        // is also used for exclusion tuple generation
        CUDASequencer->copyAoSDataToHost();
    }
    CmiNodeBarrier();
    wakeULTs(); // Wakes everyone back up for migration
    this->patch->positionsReady_GPU(1, startup);
    if(CUDASequencer->numPatchesCheckedIn < patchMap->numPatchesOnNode(CkMyPe()) -1 ) {
      CUDASequencer->masterThreadSleeping = true;
      CUDASequencer->masterThread = CthSelf();
      CthSuspend();
    }
    CmiNodeBarrier();
  }
  if (isMasterPe) {
    if (doGlobal) {
      CUDASequencer->updateHostPatchDataSOA();  // Needs to be called after HomePatch updates
    }
  }
  CmiNodeBarrier();
  if (isMasterPe) { 
    // This needs to be called after positionsReady_GPU to that the atom maps have been updated
    // This will be called in updateDeviceData during with startup=true, but we need to call it
    // with startup=false to make sure the atoms are updated
    CUDASequencer->migrationUpdateAdvancedFeatures(false);
  }
  CmiNodeBarrier();
}

// JM: Single-node integration scheme
void Sequencer::integrate_CUDA_SOA(int scriptTask){

  #ifdef TIMER_COLLECTION
    TimerSet& t = patch->timerSet;
  #endif
    TIMER_INIT_WIDTH(t, KICK, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, MAXMOVE, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, DRIFT, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, PISTON, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, SUBMITHALF, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, VELBBK1, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, VELBBK2, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, RATTLE1, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, SUBMITFULL, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, SUBMITCOLLECT, simParams->timerBinWidth);

    // Keep track of the step number.
    //int &step = patch->flags.step;
    int &step = patch->flags.step;
    step = simParams->firstTimestep;
    Controller *c_out = CUDASequencer->patchData->c_out;
    PatchMap* patchMap = PatchMap::Object();

    // For multiple time stepping, which force boxes are used?
    int &maxForceUsed = patch->flags.maxForceUsed;
    int &maxForceMerged = patch->flags.maxForceMerged;
    maxForceUsed = Results::normal;
    maxForceMerged = Results::normal;

    // Keep track of total steps and steps per cycle.
    const int numberOfSteps = simParams->N;
    //const int stepsPerCycle = simParams->stepsPerCycle;
    CheckStep stepsPerCycle;
    stepsPerCycle.init(step, simParams->stepsPerCycle);
    // The fundamental time step, get the scaling right for velocity units.
    const BigReal timestep = simParams->dt * RECIP_TIMEFACTOR;

    //const int nonbondedFrequency = simParams->nonbondedFrequency;
    //slowFreq = nonbondedFrequency;
    CheckStep nonbondedFrequency;
    slowFreq = simParams->nonbondedFrequency;
    // The step size for short-range nonbonded forces.
    const BigReal nbondstep = timestep * simParams->nonbondedFrequency;
    int &doNonbonded = patch->flags.doNonbonded;
    //doNonbonded = (step >= numberOfSteps) || !(step%nonbondedFrequency);
    doNonbonded = (step >= numberOfSteps) ||
      nonbondedFrequency.init(step, simParams->nonbondedFrequency);
    //if ( nonbondedFrequency == 1 ) maxForceMerged = Results::nbond;
    if ( nonbondedFrequency.period == 1 ) maxForceMerged = Results::nbond;
    if ( doNonbonded ) maxForceUsed = Results::nbond;

    // Do we do full electrostatics?
    const int dofull = ( simParams->fullElectFrequency ? 1 : 0 );
    //const int fullElectFrequency = simParams->fullElectFrequency;
    //if ( dofull ) slowFreq = fullElectFrequency;
    CheckStep fullElectFrequency;
    if ( dofull ) slowFreq = simParams->fullElectFrequency;
    // The step size for long-range electrostatics.
    const BigReal slowstep = timestep * simParams->fullElectFrequency;
    int &doFullElectrostatics = patch->flags.doFullElectrostatics;
    //doFullElectrostatics = (dofull &&
    //    ((step >= numberOfSteps) || !(step%fullElectFrequency)));
    doFullElectrostatics = (dofull &&
        ((step >= numberOfSteps) ||
         fullElectFrequency.init(step, simParams->fullElectFrequency)));
    //if ( dofull && fullElectFrequency == 1 ) maxForceMerged = Results::slow;
    if ( dofull && fullElectFrequency.period == 1 ) maxForceMerged = Results::slow;
    if ( doFullElectrostatics ) maxForceUsed = Results::slow;

    // Bother to calculate energies?
    int &doEnergy = patch->flags.doEnergy;
    //int energyFrequency = simParams->outputEnergies;
    CheckStep energyFrequency;
    int newComputeEnergies = simParams->computeEnergies;
    if(simParams->alchOn) newComputeEnergies = NAMD_gcd(newComputeEnergies, simParams->alchOutFreq);
    doEnergy = energyFrequency.init(step, newComputeEnergies);

    // check for Monte Carlo pressure control. 
    CheckStep monteCarloPressureFrequency; 
    doEnergy += monteCarloPressureFrequency.init(step, (simParams->monteCarloPressureOn ? 
        simParams->monteCarloPressureFreq : numberOfSteps + 1) );

    int &doVirial = patch->flags.doVirial;
    doVirial = 1;
    // Do we need to return forces to TCL script or Colvar module?
    int doTcl = simParams->tclForcesOn;
    int doColvars = simParams->colvarsOn;
    const int doGlobal = (doTcl || doColvars);
    ComputeGlobal *computeGlobal = Node::Object()->computeMgr->computeGlobalObject;

    // The following flags have to be explicitly disabled in Patch object.
    patch->flags.doMolly = 0;
    patch->flags.doLoweAndersen = 0;
    patch->flags.doGBIS = 0;
    patch->flags.doLCPO = 0;

    // Square of maximum velocity for simulation safety check
    const BigReal maxvel2 =
      (simParams->cutoff * simParams->cutoff) / (timestep * timestep);

    // check for Langevin piston
    // set period beyond numberOfSteps to disable
    // fprintf(stderr, " Patch %d Pinging in from integrate_cuda!\n", this->patch->getPatchID());
    CheckStep langevinPistonFrequency;
    langevinPistonFrequency.init(step,
        (simParams->langevinPistonOn ? slowFreq : numberOfSteps+1 ),
        (simParams->langevinPistonOn ? -1-slowFreq/2 : 0) /* = delta */);

    // check for velocity rescaling
    // set period beyond numberOfSteps to disable
    CheckStep stochRescaleFrequency;
    stochRescaleFrequency.init(step, (simParams->stochRescaleOn ?
          simParams->stochRescaleFreq : numberOfSteps+1 ) );

    CheckStep reassignVelocityFrequency;
    reassignVelocityFrequency.init(step, ((simParams->reassignFreq>0) ?
	  simParams->reassignFreq : numberOfSteps+1 ) );

    // check for output
    // set period beyond numberOfSteps to disable
    CheckStep restartFrequency;
    restartFrequency.init(step, (simParams->restartFrequency > 0 ?
          simParams->restartFrequency : numberOfSteps+1) );
    CheckStep dcdFrequency;
    dcdFrequency.init(step, (simParams->dcdFrequency > 0 ?
          simParams->dcdFrequency : numberOfSteps+1) );
    CheckStep velDcdFrequency;
    velDcdFrequency.init(step, (simParams->velDcdFrequency > 0 ?
          simParams->velDcdFrequency : numberOfSteps+1) );
    CheckStep forceDcdFrequency;
    forceDcdFrequency.init(step, (simParams->forceDcdFrequency > 0 ?
          simParams->forceDcdFrequency : numberOfSteps+1) );
    CheckStep imdFrequency;
    imdFrequency.init(step, (simParams->IMDfreq > 0 ?
          simParams->IMDfreq : numberOfSteps+1) );

  patch->copy_atoms_to_SOA(); // do this whether or not useDeviceMigration

  if (simParams->rigidBonds != RIGID_NONE && ! patch->settle_initialized) {
    patch->buildRattleList_SOA();
    patch->rattleListValid_SOA = true;
  }
  
  this->suspendULTs();
  // for "run 0", numberOfSteps is zero, but we want to have at least a single energy evaluation
  if(!masterThread) {
    return;
  }
  bool isMasterPe = (deviceCUDA->getMasterPe() == CkMyPe() );
  CmiNodeBarrier();

  CUDASequencer->breakSuspends = false;

  // XXX this is ugly!
  // one thread will have the CollectionMaster and Output defined
  // use it to set the node group so that any thread can access
  if (CUDASequencer->patchData->ptrCollectionMaster == NULL) {
    CollectionMaster *pcm = CkpvAccess(CollectionMaster_instance)->Object();
    if (pcm) {
      CUDASequencer->patchData->ptrCollectionMaster = pcm;
    }
  }
  if (CUDASequencer->patchData->ptrOutput == NULL) {
    Output *pout = Node::Object()->output;
    if (pout) {
      CUDASequencer->patchData->ptrOutput = pout;
    }
  }
  if (CUDASequencer->patchData->pdb == NULL) {
    PDB *pdb = Node::Object()->pdb;
    if (pdb) {
      CUDASequencer->patchData->pdb = pdb;
    }
  }
  if (CUDASequencer->patchData->imd == NULL) {
    IMDOutput *imd = Node::Object()->imd;
    if (imd->getIMD()) {
      CUDASequencer->patchData->imd = imd;
    }
  }

  // Register ComputeCUDAMgrs from each PE into a list for later usage
  if(isMasterPe){
    // Each masterPE registers its own computeCUDAMgr
    CUDASequencer->patchData->cudaBondedList[CkMyPe()]    = ComputeCUDAMgr::getComputeCUDAMgr()->getComputeBondedCUDA();
    CUDASequencer->patchData->cudaNonbondedList[CkMyPe()] = ComputeCUDAMgr::getComputeCUDAMgr()->getCudaComputeNonbonded();
  }else{
    CUDASequencer->patchData->cudaBondedList[CkMyPe()]    = NULL;
    CUDASequencer->patchData->cudaNonbondedList[CkMyPe()] = NULL;
  }
 
  if (isMasterPe) {
    if(dofull && deviceCUDA->getIsPmeDevice()){
      ComputeCUDAMgr* cudaMgr = ComputeCUDAMgr::getComputeCUDAMgr();
      CudaPmeOneDevice* cudaPme = 0;
      cudaPme = cudaMgr->createCudaPmeOneDevice();
    }
    CUDASequencer->patchData->reduction->zero();
  }

  CmiNodeBarrier();
  
/* JM NOTE: This Will Contains the first calls to the integration loop. The order is:
 *          1 - Rattle (0,0)
 *          2 - runComputeObjects
 *          3 - addForceToMomentum(-0.5, tstep)
 *          4 - Rattle (-timestep, 0);
 *          5 - submitHalfstep();
 *          6 - addForceToMomentum(1.0 , tstep)
 *          7 - Rattle (tstep, 1)
 *          8 - SubmitHalf()
 *          9 - addForceToMomentum(-0.5, tstep)
 *         10 - submitReductions()
 */
  
  if(scriptTask == SCRIPT_RUN){
    updateDeviceData(1, maxForceUsed, doGlobal);

    if(isMasterPe) {
      // warm_up1 is basically rattle1_SOA(0,0)
      CUDASequencer->startRun1(maxForceUsed, this->patch->lattice);
      (this->patch->flags.sequence)++;
      if (deviceCUDA->getIsMasterDevice()){
        CUDASequencer->patchData->lat = &(this->patch->lattice);
        CUDASequencer->patchData->flags.copyIntFlags(this->patch->flags);
      }
    }

    CmiNodeBarrier();

    if (!simParams->useDeviceMigration) { 
      wakeULTs(); // Wakes everyone back up for migration
      this->patch->positionsReady_SOA(1);
      if(CUDASequencer->numPatchesCheckedIn < patchMap->numPatchesOnNode(CkMyPe()) -1 ) {
        CUDASequencer->masterThreadSleeping = true;
        CUDASequencer->masterThread = CthSelf();
        CthSuspend();
      }
      CmiNodeBarrier();
      updateDeviceData(0, maxForceUsed, doGlobal);
    } else {
      doMigrationGPU(1, doGlobal, simParams->updateAtomMap);
    }
    CmiNodeBarrier();
    // I've migrated everything. Now run computes
    runComputeObjectsCUDA(/*isMigration = */ 1 , 
      doGlobal,
      /* step < numberofSteps */ 1, 
      /* step = */ 0,
      /* startup = */ 1);

    if(isMasterPe){
      CUDASequencer->finish_patch_flags(true);
      CUDASequencer->startRun2(timestep,
        nbondstep, slowstep, this->patch->lattice.origin(),  
	CUDASequencer->patchData->reduction, doGlobal, maxForceUsed);
    }
    CmiNodeBarrier();
    if(isMasterPe){
      CUDASequencer->startRun3(timestep,
        nbondstep, slowstep, this->patch->lattice.origin(),  
        CUDASequencer->patchData->reduction,
        doTcl || doColvars, maxForceUsed);
    }

    // save total force in computeGlobal, forces are copied from device
    // to host in startRun3
    if (doTcl || doColvars) {
      CmiNodeBarrier();
      // store the total force for compute global clients
      int numhp = PatchMap::Object()->numHomePatches();
      HomePatchList *hpList = PatchMap::Object()->homePatchList();
      for(int i = 0; i < numhp; ++i) {
        HomePatch *hp = hpList->item(i).patch;
        computeGlobal->saveTotalForces(hp);
      }
    }
  }
  
  CmiNodeBarrier();
  // Called everything, now I can go ahead and print the step
  // PE 0 needs to handle IO as it owns the controller object
  // JM: What happens if PE 0 does not own a GPU here? XXX Check
  if(deviceCUDA->getIsMasterDevice()) {
    CUDASequencer->patchData->flags.copyIntFlags(this->patch->flags);
    c_out->resetMovingAverage();
    c_out->printStep(step, CUDASequencer->patchData->reduction);
    CUDASequencer->patchData->reduction->zero();
  }
  CmiNodeBarrier();

  // XXX Should we promote velrescaling into Sequencer in order to save
  // the velocity rescaling coefficient between script run commands?
  double velrescaling = 1;
  // --------- Start of the MD loop ------- //
  for( ++step; step <= numberOfSteps; ++step ){
    const int isCollection = restartFrequency.check(step) +
      dcdFrequency.check(step) + velDcdFrequency.check(step) +
      forceDcdFrequency.check(step) + imdFrequency.check(step);

     int isMigration = false;
     const int doVelocityRescale = stochRescaleFrequency.check(step);
     const int doMCPressure = monteCarloPressureFrequency.check(step);
     // XXX doVelRescale should instead set a "doTemperature" flag
     doEnergy = energyFrequency.check(step) || doVelocityRescale || doMCPressure;
     int langevinPistonStep = langevinPistonFrequency.check(step);

     int reassignVelocityStep = reassignVelocityFrequency.check(step); 

     // berendsen pressure control
    int berendsenPressureStep = 0;
    if(simParams->berendsenPressureOn) {
      ++berendsenPressure_count;
      if (berendsenPressure_count == simParams->berendsenPressureFreq) {
        berendsenPressure_count = 0;
        berendsenPressureStep = 1;
      }
    }


#if defined(NAMD_NVTX_ENABLED) || defined(NAMD_CMK_TRACE_ENABLED)  || defined(NAMD_ROCTX_ENABLED)
    //eon = epid && (beginStep < step && step <= endStep);
    // int eon = epid && (beginStep < step && step <= endStep);
    // if (controlProfiling && step == beginStep) {
    //  NAMD_PROFILE_START();
    // }
    //if (controlProfiling && step == endStep) {
    //  NAMD_PROFILE_STOP();
    //}
#endif

    Vector origin = this->patch->lattice.origin();
    Tensor factor;
    if (deviceCUDA->getIsMasterDevice()) {    
      if (simParams->langevinPistonOn) {
        c_out->piston1(step);
      }
      // Get the rescale factor for berendsen from controller
      if (simParams->berendsenPressureOn) {
        c_out->berendsenPressureController(step);
      }
    }
    if (langevinPistonStep || berendsenPressureStep) {
      if(deviceCUDA->getIsMasterDevice()){
        factor = c_out->getPositionRescaleFactor(step);
        this->patch->lattice.rescale(factor);
        CUDASequencer->patchData->lat    = &(this->patch->lattice);
        CUDASequencer->patchData->factor = &(factor); // This looks unsafe, but as devices run on lockstep, it's fine
      }
    }
    
    CmiNodeBarrier(); 
    NAMD_EVENT_START(1, NamdProfileEvent::CUDASOA_LAUNCHPT1);
    int previousMaxForceUsed; 
    if(isMasterPe){
      // need to remember number of buffers for previous force calculation
      previousMaxForceUsed = maxForceUsed;
      // update local flags
      //doNonbonded = !(step%nonbondedFrequency);
      // no need to include doMCPressure since it's common factor of nonbondedFrequency
      doNonbonded = nonbondedFrequency.check(step);
      // no need to include doMCPressure since it's common factor of fullElectFrequency
      doFullElectrostatics = (dofull && fullElectFrequency.check(step));
      maxForceUsed = Results::normal;
      if ( doNonbonded ) maxForceUsed = Results::nbond;
      if ( doFullElectrostatics ) maxForceUsed = Results::slow;

      (this->patch->flags.sequence)++;
      // JM: Pressures needed for every timestep if the piston is on
      (this->patch->flags.doVirial) = doEnergy || simParams->langevinPistonOn || simParams->berendsenPressureOn; 

      // copy local flags to global
      if(deviceCUDA->getIsMasterDevice()) CUDASequencer->patchData->flags.copyIntFlags(this->patch->flags);
    }
    
    CmiNodeBarrier();

    if(isMasterPe){
      CUDASequencer->launch_part1(
	step,
        timestep, nbondstep, slowstep, velrescaling, maxvel2,
        CUDASequencer->patchData->reduction,
        *(CUDASequencer->patchData->factor),
        origin,
        // this->patch->lattice, // need to use the lattice from PE 0 right now
        (langevinPistonStep || berendsenPressureStep) ? *(CUDASequencer->patchData->lat) : this->patch->lattice,
	reassignVelocityStep,
        langevinPistonStep,
        berendsenPressureStep,
        previousMaxForceUsed,  // call with previous maxForceUsed
        (const int)(step == simParams->firstTimestep + 1),
        this->patch->flags.savePairlists,  // XXX how to initialize?
        this->patch->flags.usePairlists,   // XXX how to initialize?
        doEnergy);
      // reset velocity rescaling coefficient after applying it
      velrescaling = 1;
    }
    if (reassignVelocityStep)
      {
	//	CkPrintf("dump after launch_part1\n");
	//	CUDASequencer->printSOAPositionsAndVelocities(2,10);
      }
    NAMD_EVENT_STOP(1, NamdProfileEvent::CUDASOA_LAUNCHPT1);
 
    CmiNodeBarrier();

    if(isMasterPe){
      CUDASequencer->launch_part11(
        timestep, nbondstep, slowstep, velrescaling, maxvel2,
        CUDASequencer->patchData->reduction,
        *(CUDASequencer->patchData->factor),
        origin,
        // this->patch->lattice, // need to use the lattice from PE 0 right now
        (langevinPistonStep || berendsenPressureStep) ? *(CUDASequencer->patchData->lat) : this->patch->lattice,
        langevinPistonStep,
        previousMaxForceUsed,  // call with previous maxForceUsed
        (const int)(step == simParams->firstTimestep + 1),
        this->patch->flags.savePairlists,  // XXX how to initialize?
        this->patch->flags.usePairlists,   // XXX how to initialize?
        doEnergy);
      // reset velocity rescaling coefficient after applying it
      velrescaling = 1;
    }
    NAMD_EVENT_STOP(1, NamdProfileEvent::CUDASOA_LAUNCHPT1);
 
    CmiNodeBarrier();

    
    for(int i = 0; i < deviceCUDA->getNumDevice(); i++){
      if(CUDASequencer->patchData->migrationFlagPerDevice[i] != 0) {
        isMigration = true;
        break;
      }
    }

    if(isMasterPe){
      // If this is a Device Migration step we'll do it later
      if (!simParams->useDeviceMigration || !isMigration) {
        CUDASequencer->launch_set_compute_positions();
      }
    }
    
    // isMigration = (CUDASequencer->patchData->migrationFlagPerDevice.end() != t) ? 1:0;
    
    if(isMasterPe) {
      // if(CkMyPe() == 0) CUDASequencer->updatePairlistFlags(isMigration);
      CUDASequencer->updatePairlistFlags(isMigration);
      if (!simParams->useDeviceMigration) {
        CUDASequencer->copyPositionsAndVelocitiesToHost(isMigration, doGlobal);
      }
    }


    if(isMigration) {
      if (!simParams->useDeviceMigration) {
        CmiNodeBarrier();
        wakeULTs(); // sets the number of patches 
        this->patch->positionsReady_SOA(isMigration);
        if(CUDASequencer->numPatchesCheckedIn < patchMap->numPatchesOnNode(CkMyPe()) -1 ) {
          CUDASequencer->masterThreadSleeping = true;
          CUDASequencer->masterThread = CthSelf();
          CthSuspend(); // suspends until everyone else has pinged back. :]
        }
        CmiNodeBarrier();
        updateDeviceData(0, maxForceUsed, doGlobal);
      } else {
        doMigrationGPU(false, doGlobal, simParams->updateAtomMap);
        CmiNodeBarrier();
      }
    }

    // Calculate force/energy for bond, nonBond, pme.
    this->runComputeObjectsCUDA(isMigration, doGlobal, step<numberOfSteps, step, 0 /* startup */);
    if (isMasterPe) {
      // if(CkMyPe() ==  0) CUDASequencer->finish_patch_flags(isMigration);
      CUDASequencer->finish_patch_flags(isMigration);
      CUDASequencer->patchData->migrationFlagPerDevice[deviceCUDA->getDeviceIndex()] = 0; // flags it back to zero
    }
    CmiNodeBarrier();

    NAMD_EVENT_START(1, NamdProfileEvent::CUDASOA_LAUNCHPT2);
    if(isMasterPe){
      CUDASequencer->launch_part2(doMCPressure,
        timestep, nbondstep, slowstep,
        CUDASequencer->patchData->reduction,
        origin,
        step,
        maxForceUsed,
        langevinPistonStep, 
        isMigration && (!simParams->useDeviceMigration),
        isCollection,
        doGlobal, 
        doEnergy);
    }
    CmiNodeBarrier();
    NAMD_EVENT_STOP(1, NamdProfileEvent::CUDASOA_LAUNCHPT2);

    // Apply MC pressure control
    if(doMCPressure){
      monteCarloPressureControl(step, isMigration, 1, 1, maxForceUsed);
      CmiNodeBarrier();
    }

    // continue launch_part2, after cellBasis fluctuation in MC barostat
    if(isMasterPe){
      CUDASequencer->launch_part3(doMCPressure,
        timestep, nbondstep, slowstep,
        CUDASequencer->patchData->reduction,
        origin,
	step,			  
        maxForceUsed,
        doTcl || doColvars, // requested Force
        isMigration,
        isCollection,
        doEnergy);
    }
    CmiNodeBarrier();
    NAMD_EVENT_STOP(1, NamdProfileEvent::CUDASOA_LAUNCHPT2);

    // save total force in computeGlobal, forces are copied from device
    // to host in launch_part3
    if (doTcl || doColvars) {
      CmiNodeBarrier();
      // store the total force for compute global clients
      int numhp = PatchMap::Object()->numHomePatches();
      HomePatchList *hpList = PatchMap::Object()->homePatchList();
      for(int i = 0; i < numhp; ++i) {
        HomePatch *hp = hpList->item(i).patch;
        computeGlobal->saveTotalForces(hp);
      }
    }


    NAMD_EVENT_START(1, NamdProfileEvent::CUDASOA_PRTSTEP);
    CmiNodeBarrier();

    if (deviceCUDA->getIsMasterDevice()) {
      // even though you're not on a printstep, calling this still takes 15us approx!!!
      c_out->printStep(step, CUDASequencer->patchData->reduction);
      // stochastic velocity rescaling
      // get coefficient from current temperature
      // to be applied on NEXT loop iteration
      if (doVelocityRescale) {
        // calculate coefficient based on current temperature
        velrescaling = c_out->stochRescaleCoefficient();
      }
    }
    NAMD_EVENT_STOP(1, NamdProfileEvent::CUDASOA_PRTSTEP);

    NAMD_EVENT_START(1, NamdProfileEvent::RESET_REDUCTIONS);
    if(deviceCUDA->getIsMasterDevice()) CUDASequencer->patchData->reduction->zero();
    NAMD_EVENT_STOP(1, NamdProfileEvent::RESET_REDUCTIONS);

    NAMD_EVENT_START(1, NamdProfileEvent::CUDASOA_SUBCOL);
    if (isCollection) {
      CmiNodeBarrier();
      if (simParams->useDeviceMigration) {
        if (isMasterPe) {
          CUDASequencer->copyAoSDataToHost();
        }
        // Make sure the data has been copied to all home patches. All PEs
        // participate in outputting
        CmiNodeBarrier(); 
      }
      HomePatchList *hplist = patchMap->homePatchList();
      for (auto i= hplist->begin(); i != hplist->end(); i++) {
        HomePatch *hp = i->patch;
        hp->sequencer->submitCollections_SOA(step);
      }
      CmiNodeBarrier();
    }
    NAMD_EVENT_STOP(1, NamdProfileEvent::CUDASOA_SUBCOL);
  }

  if (simParams->useDeviceMigration) {
    CmiNodeBarrier();
    if (isMasterPe) {
      CUDASequencer->copyAoSDataToHost();
    }
  } else {
    if(isMasterPe) CUDASequencer->copyPositionsAndVelocitiesToHost(true,doGlobal);
    CmiNodeBarrier();
    HomePatchList *hplist = patchMap->homePatchList();
    for (auto i= hplist->begin(); i != hplist->end(); i++) {
      HomePatch *hp = i->patch;
      hp->copy_updates_to_AOS();
    }
  }
  CmiNodeBarrier(); // Make sure the data has been copied to all home patches

  //CmiNodeBarrier();
  CUDASequencer->breakSuspends = true;
  wakeULTs();
  if(deviceCUDA->getIsMasterDevice()) c_out->awaken();
}


/*
 * Updates device data after a migration
 *
 */
void Sequencer::updateDeviceData(const int startup, const int maxForceUsed, const int doGlobal) {
  bool isMaster = deviceCUDA->getMasterPe() == CkMyPe();
  updateDevicePatchMap(1);
  if (isMaster) {
    CUDASequencer->copyPatchData(true, startup);
    if (simParams->useDeviceMigration) {
      CUDASequencer->reallocateMigrationDestination();
      CUDASequencer->copyAtomDataToDeviceAoS();
    } else {
      CUDASequencer->copyAtomDataToDevice(startup, maxForceUsed);
    }
    CUDASequencer->migrationLocalPost(startup);
    CUDASequencer->migrationUpdateAdvancedFeatures(startup);
    // XXX This is only necessary if reallocation happens
    CUDASequencer->registerSOAPointersToHost(); 
  }
  CmiNodeBarrier();
  if (isMaster) {
    CUDASequencer->copySOAHostRegisterToDevice();
    if (simParams->useDeviceMigration) {
      CUDASequencer->patchData->atomReallocationFlagPerDevice[deviceCUDA->getDeviceIndex()] = 0;
    }

    if (doGlobal) {
      CUDASequencer->updateHostPatchDataSOA();  // Needs to be called after HomePatch::domigration
    }
  }
  CmiNodeBarrier();
}

/*
 * Constructs the meta data structures storing the patch data for GPU resident code path
 *
 * This is called once during startup
 *
 */
void Sequencer::constructDevicePatchMap() {
  ComputeCUDAMgr*       cudaMgr    = ComputeCUDAMgr::getComputeCUDAMgr();
  ComputeBondedCUDA*    cudaBond   = cudaMgr->getComputeBondedCUDA();
  CudaComputeNonbonded* cudaNbond  = cudaMgr->getCudaComputeNonbonded();

  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();

  const bool isMasterPe = deviceCUDA->getMasterPe() == CkMyPe();

  // constructDevicePatchMap should only be called once per PE
  if (patchData->devicePatchMapFlag[CkMyPe()]) return;
  patchData->devicePatchMapFlag[CkMyPe()] = 1;

  // One thread per GPU will execute this block
  if (isMasterPe) {
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    
    // Nonbonded patches are computed by CudaComputeNonbonded and contain all the patches and proxy
    // patches on this device. HomePatches is computed by SequencerCUDA and only contains the 
    // home patches. localPatches will be generated by this function
    using NBPatchRecord = CudaComputeNonbonded::PatchRecord;
    std::vector<NBPatchRecord>&    nonbondedPatches   = cudaNbond->getPatches();
    std::vector<HomePatch*>&       homePatches = patchData->devData[deviceIndex].patches;
    std::vector<CudaLocalRecord>&  localPatches = patchData->devData[deviceIndex].h_localPatches;

    // The home patches are not necessarily ordered by their patchID. This can happen if there
    // are multiple PEs assigned to the same GPU. Sorting the home patches by their patch ID
    // makes it easy to have a consistent ordering 
    std::stable_sort(
      homePatches.begin(), 
      homePatches.end(), 
      [](HomePatch* a, HomePatch* b) {
        return (a->getPatchID() < b->getPatchID());
      });

    // Iterates over all the patches on this device and adds them to h_localPatches
    // and determine if they are a home or proxy patch
    for (int i = 0; i < nonbondedPatches.size(); i++) {
      CudaLocalRecord record;
      record.patchID = nonbondedPatches[i].patchID;

      // TODO DMC the patchmap should be able to do this
      const int targetPatchID = record.patchID;
      auto result = std::find_if(
        homePatches.begin(),
        homePatches.end(), 
        [targetPatchID](HomePatch* p) {
          return (p->getPatchID() == targetPatchID);
        });

      record.isProxy = (result == homePatches.end());
      localPatches.push_back(record);
    }

    // The home patches should be at the begining of the patch list
    // This makes integration easier since we can ignore the patches and operate on a
    // contiguous chunk of home atoms
    std::stable_sort(
      localPatches.begin(), 
      localPatches.end(), 
      [](CudaLocalRecord a, CudaLocalRecord b) {
        return (a.isProxy < b.isProxy);
      });

    // Now the ordering is fixed we can update the bonded and nonbonded orders. Since we have
    // moved the home patches to the begining the ordering has changed
    cudaBond->updatePatchOrder(localPatches);
    cudaNbond->updatePatchOrder(localPatches);
    patchData->devData[deviceIndex].numPatchesHome = homePatches.size();
    patchData->devData[deviceIndex].numPatchesHomeAndProxy = localPatches.size();
  }
  CmiNodeBarrier();

  // Iterates over all patches again, and generates the mapping between GPUs. For each patch,
  // it checks the other devices to see if the patch is on that device. 
  //   - For HomePatches, there will be a peer record for all of its proxies
  //   - For ProxyPatches, there will only be a peer record for its home patch
  // There is a single array of peer records per device. Each patch stores an offset into this
  // array as well as its number of peer records
  if (isMasterPe) {
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    std::vector<CudaPeerRecord>& myPeerPatches = patchData->devData[deviceIndex].h_peerPatches;
    std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;

    for (int i = 0; i < localPatches.size(); i++) {
      std::vector<CudaPeerRecord> tempPeers;
      const int targetPatchID = localPatches[i].patchID;
      const int targetIsProxy = localPatches[i].isProxy;

      for (int devIdx = 0; devIdx < deviceCUDA->getNumDevice(); devIdx++) {
        if (devIdx == deviceIndex) continue;
        std::vector<CudaLocalRecord>& peerPatches = patchData->devData[devIdx].h_localPatches;

        // Searches peerPatches for patchID. If it is not being integrated on this device
        // then ignore other non-integration patches
        for (int j = 0; j < patchData->devData[devIdx].numPatchesHomeAndProxy; j++) {
          const CudaLocalRecord peer = peerPatches[j];
          if (peer.patchID == targetPatchID && peer.isProxy != targetIsProxy) {
            CudaPeerRecord peerRecord;
            peerRecord.deviceIndex = devIdx;
            peerRecord.patchIndex = j;
            tempPeers.push_back(peerRecord);
            break;
          }
        }
      }

      // Once we have the list of peer records, add them to the single device-width vector
      // and record the offset and count
      localPatches[i].numPeerRecord = tempPeers.size();
      if (!tempPeers.empty()) {
        localPatches[i].peerRecordStartIndex = myPeerPatches.size();
        myPeerPatches.insert(myPeerPatches.end(), tempPeers.begin(), tempPeers.end());
      }
    }
  }
  CmiNodeBarrier();
}

void Sequencer::printDevicePatchMap() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();

  const bool isMasterPe = deviceCUDA->getMasterPe() == CkMyPe();

  if (isMasterPe) {
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    const int numPatchesHome = patchData->devData[deviceIndex].numPatchesHome;
    std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;

    CmiLock(patchData->printlock);
    CkPrintf("PE: %d\n", CkMyPe());

    CkPrintf("[%d] Home patches %d Local patches %d\n", CkMyPe(), numPatchesHome, localPatches.size());

    CkPrintf("Home Patches: ");
    for (int i = 0; i < numPatchesHome; i++) {
      CkPrintf("%d ", localPatches[i].patchID);
    }
    CkPrintf("\n");

    CkPrintf("Proxy Patches: ");
    for (int i = numPatchesHome; i < localPatches.size(); i++) {
      CkPrintf("%d ", localPatches[i].patchID);
    }
    CkPrintf("\n");

    CmiUnlock(patchData->printlock);
  }
  CmiNodeBarrier();
}

void Sequencer::clearDevicePatchMap() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();

  const bool isMasterPe = deviceCUDA->getMasterPe() == CkMyPe();

  // clearDevicePatchMap should only be called once per PE
  if (!patchData->devicePatchMapFlag[CkMyPe()]) return;
  patchData->devicePatchMapFlag[CkMyPe()] = 0;

  // One thread per GPU will execute this block
  if (isMasterPe) {
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    
    using NBPatchRecord = CudaComputeNonbonded::PatchRecord;
    std::vector<HomePatch*>&       homePatches = patchData->devData[deviceIndex].patches;
    std::vector<CudaLocalRecord>&  localPatches = patchData->devData[deviceIndex].h_localPatches;
    std::vector<CudaPeerRecord>&   peerPatches = patchData->devData[deviceIndex].h_peerPatches;

    homePatches.clear();
    localPatches.clear();
    peerPatches.clear();
  }
}

/*
 * Updates the meta data structures storing the patch data for GPU resident code path
 *
 * This is called every migration step. The actual mapping stays the same, 
 * but the atom counts per patch change
 *
 */
void Sequencer::updateDevicePatchMap(int startup) {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();

  const bool isMasterPe = deviceCUDA->getMasterPe() == CkMyPe();

  if (isMasterPe) {
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    const int numPatchesHome = patchData->devData[deviceIndex].numPatchesHome;
    std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;

    ComputeCUDAMgr*       cudaMgr    = ComputeCUDAMgr::getComputeCUDAMgr();
    CudaComputeNonbonded* cudaNbond  = cudaMgr->getCudaComputeNonbonded();

    int max_atom_count = 0;
    int total_atom_count = 0;

    // Update the atom count of home patches
    for (int i = 0; i < numPatchesHome; i++) {
      Patch* patch = NULL;
      for(int j = 0; j < deviceCUDA->getNumPesSharingDevice(); j++){
        PatchMap* pm = PatchMap::ObjectOnPe(deviceCUDA->getPesSharingDevice(j));
        patch = pm->patch(localPatches[i].patchID);
        if (patch != NULL) break;
      }
      if (patch == NULL) NAMD_die("Sequencer: Failed to find patch in updateDevicePatchMap");

      localPatches[i].numAtoms = patch->getNumAtoms();
      localPatches[i].numAtomsNBPad = CudaComputeNonbondedKernel::computeAtomPad(localPatches[i].numAtoms);

      if (localPatches[i].numAtoms > max_atom_count) max_atom_count = localPatches[i].numAtoms;
      total_atom_count += localPatches[i].numAtoms;
    }
  }
  CmiNodeBarrier();

  // Update the proxy patches next, using the home patch atom counts of other devices
  if (isMasterPe) {
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    const int numPatchesHome = patchData->devData[deviceIndex].numPatchesHome;
    const int numPatchesHomeAndProxy = patchData->devData[deviceIndex].numPatchesHomeAndProxy;
    std::vector<CudaPeerRecord>& peerPatches = patchData->devData[deviceIndex].h_peerPatches;
    std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;

    for (int i = numPatchesHome; i < numPatchesHomeAndProxy; i++) {
      const int index = localPatches[i].peerRecordStartIndex;
      const int devIdx = peerPatches[index].deviceIndex;
      const int peerIdx = peerPatches[index].patchIndex;
      const CudaLocalRecord peer = patchData->devData[devIdx].h_localPatches[peerIdx];
      
      localPatches[i].numAtoms = peer.numAtoms;
      localPatches[i].numAtomsNBPad = peer.numAtomsNBPad;
    }
  }
  CmiNodeBarrier();

  // Computes the offset for each patch using the atom counts
  if (isMasterPe) {
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    const int numPatchesHomeAndProxy = patchData->devData[deviceIndex].numPatchesHomeAndProxy;
    std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
  
    int runningOffset = 0;
    int runningOffsetNBPad = 0;
    // TODO Change to a C++ prefix sum
    for (int i = 0; i < numPatchesHomeAndProxy; i++) {
      localPatches[i].bufferOffset   = runningOffset;
      localPatches[i].bufferOffsetNBPad = runningOffsetNBPad;
      runningOffset   += localPatches[i].numAtoms;
      runningOffsetNBPad += localPatches[i].numAtomsNBPad;
    }
  }
  CmiNodeBarrier();

  // Update the peer records using the local record data
  if (isMasterPe) {
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    const int numPatchesHomeAndProxy = patchData->devData[deviceIndex].numPatchesHomeAndProxy;
    std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
    std::vector<CudaPeerRecord>& peerPatches = patchData->devData[deviceIndex].h_peerPatches;


    for (int i = 0; i < peerPatches.size(); i++) {
      const int devIdx = peerPatches[i].deviceIndex;
      const int peerIdx = peerPatches[i].patchIndex;
      const CudaLocalRecord peer = patchData->devData[devIdx].h_localPatches[peerIdx];

      peerPatches[i].bufferOffset = peer.bufferOffset;
      peerPatches[i].bufferOffsetNBPad = peer.bufferOffsetNBPad;
    }

    // Update inline copy of peer data
    for (int i = 0; i < numPatchesHomeAndProxy; i++) {
      const int numPeerRecord = localPatches[i].numPeerRecord;
      const int peerOffset = localPatches[i].peerRecordStartIndex;

      for (int j = 0; j < std::min(numPeerRecord, CudaLocalRecord::num_inline_peer); j++) {
        localPatches[i].inline_peers[j] = peerPatches[peerOffset+j];
      }
    }
  }
  CmiNodeBarrier();
}

#endif


void Sequencer::integrate_SOA(int scriptTask) {
  //
  // Below when accessing the array buffers for position, velocity, force,
  // note that we don't want to set up pointers directly to the buffers
  // because the allocations might get resized after atom migration.
  //

#ifdef TIMER_COLLECTION
  TimerSet& t = patch->timerSet;
#endif
  TIMER_INIT_WIDTH(t, KICK, simParams->timerBinWidth);
  TIMER_INIT_WIDTH(t, MAXMOVE, simParams->timerBinWidth);
  TIMER_INIT_WIDTH(t, DRIFT, simParams->timerBinWidth);
  TIMER_INIT_WIDTH(t, PISTON, simParams->timerBinWidth);
  TIMER_INIT_WIDTH(t, SUBMITHALF, simParams->timerBinWidth);
  TIMER_INIT_WIDTH(t, VELBBK1, simParams->timerBinWidth);
  TIMER_INIT_WIDTH(t, VELBBK2, simParams->timerBinWidth);
  TIMER_INIT_WIDTH(t, RATTLE1, simParams->timerBinWidth);
  TIMER_INIT_WIDTH(t, SUBMITFULL, simParams->timerBinWidth);
  TIMER_INIT_WIDTH(t, SUBMITCOLLECT, simParams->timerBinWidth);

  // Keep track of the step number.
  int &step = patch->flags.step;
  step = simParams->firstTimestep;

  // For multiple time stepping, which force boxes are used?
  int &maxForceUsed = patch->flags.maxForceUsed;
  int &maxForceMerged = patch->flags.maxForceMerged;
  maxForceUsed = Results::normal;
  maxForceMerged = Results::normal;

  // Keep track of total steps and steps per cycle.
  const int numberOfSteps = simParams->N;
  //const int stepsPerCycle = simParams->stepsPerCycle;
  CheckStep stepsPerCycle;
  stepsPerCycle.init(step, simParams->stepsPerCycle);
  // The fundamental time step, get the scaling right for velocity units.
  const BigReal timestep = simParams->dt * RECIP_TIMEFACTOR;

  //const int nonbondedFrequency = simParams->nonbondedFrequency;
  //slowFreq = nonbondedFrequency;
  CheckStep nonbondedFrequency;
  slowFreq = simParams->nonbondedFrequency;
  // The step size for short-range nonbonded forces.
  const BigReal nbondstep = timestep * simParams->nonbondedFrequency;
  int &doNonbonded = patch->flags.doNonbonded;
  //doNonbonded = (step >= numberOfSteps) || !(step%nonbondedFrequency);
  doNonbonded = (step >= numberOfSteps) ||
    nonbondedFrequency.init(step, simParams->nonbondedFrequency);
  //if ( nonbondedFrequency == 1 ) maxForceMerged = Results::nbond;
  if ( nonbondedFrequency.period == 1 ) maxForceMerged = Results::nbond;
  if ( doNonbonded ) maxForceUsed = Results::nbond;

  // Do we do full electrostatics?
  const int dofull = ( simParams->fullElectFrequency ? 1 : 0 );
  //const int fullElectFrequency = simParams->fullElectFrequency;
  //if ( dofull ) slowFreq = fullElectFrequency;
  CheckStep fullElectFrequency;
  if ( dofull ) slowFreq = simParams->fullElectFrequency;
  // The step size for long-range electrostatics.
  const BigReal slowstep = timestep * simParams->fullElectFrequency;
  int &doFullElectrostatics = patch->flags.doFullElectrostatics;
  //doFullElectrostatics = (dofull &&
  //    ((step >= numberOfSteps) || !(step%fullElectFrequency)));
  doFullElectrostatics = (dofull &&
      ((step >= numberOfSteps) ||
       fullElectFrequency.init(step, simParams->fullElectFrequency)));
  //if ( dofull && fullElectFrequency == 1 ) maxForceMerged = Results::slow;
  if ( dofull && fullElectFrequency.period == 1 ) maxForceMerged = Results::slow;
  if ( doFullElectrostatics ) maxForceUsed = Results::slow;

  // Bother to calculate energies?
  int &doEnergy = patch->flags.doEnergy;
  //int energyFrequency = simParams->outputEnergies;
  CheckStep energyFrequency;
  int newComputeEnergies = simParams->computeEnergies;
  if(simParams->alchOn) newComputeEnergies = NAMD_gcd(newComputeEnergies, simParams->alchOutFreq);
  doEnergy = energyFrequency.init(step, newComputeEnergies);

  // Do we need to return forces to TCL script or Colvar module?
  int doTcl = simParams->tclForcesOn;
	int doColvars = simParams->colvarsOn;
  ComputeGlobal *computeGlobal = Node::Object()->computeMgr->computeGlobalObject;

  int &doVirial = patch->flags.doVirial;
  doVirial = 1;

  // The following flags have to be explicitly disabled in Patch object.
  patch->flags.doMolly = 0;
  patch->flags.doLoweAndersen = 0;
  patch->flags.doGBIS = 0;
  patch->flags.doLCPO = 0;

  // Square of maximum velocity for simulation safety check
  const BigReal maxvel2 =
    (simParams->cutoff * simParams->cutoff) / (timestep * timestep);

  // check for Langevin piston
  // set period beyond numberOfSteps to disable
  CheckStep langevinPistonFrequency;
  langevinPistonFrequency.init(step,
      (simParams->langevinPistonOn ? slowFreq : numberOfSteps+1 ),
      (simParams->langevinPistonOn ? -1-slowFreq/2 : 0) /* = delta */);

  // check for output
  // set period beyond numberOfSteps to disable
  CheckStep restartFrequency;
  restartFrequency.init(step, (simParams->restartFrequency ?
        simParams->restartFrequency : numberOfSteps+1) );
  CheckStep dcdFrequency;
  dcdFrequency.init(step, (simParams->dcdFrequency ?
        simParams->dcdFrequency : numberOfSteps+1) );
  CheckStep velDcdFrequency;
  velDcdFrequency.init(step, (simParams->velDcdFrequency ?
        simParams->velDcdFrequency : numberOfSteps+1) );
  CheckStep forceDcdFrequency;
  forceDcdFrequency.init(step, (simParams->forceDcdFrequency ?
        simParams->forceDcdFrequency : numberOfSteps+1) );
  CheckStep imdFrequency;
  imdFrequency.init(step, (simParams->IMDfreq ?
        simParams->IMDfreq : numberOfSteps+1) );

  if ( scriptTask == SCRIPT_RUN ) {
    // enforce rigid bond constraints on initial positions
    TIMER_START(t, RATTLE1);
    rattle1_SOA(0., 0);
    TIMER_STOP(t, RATTLE1);

    // must migrate here!
    int natoms = patch->patchDataSOA.numAtoms;
    runComputeObjects_SOA(1, step<numberOfSteps, step);
    // kick -0.5
    TIMER_START(t, KICK);
    addForceToMomentum_SOA(-0.5, timestep, nbondstep, slowstep,
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.recipMass,
        patch->patchDataSOA.f_normal_x,
        patch->patchDataSOA.f_normal_y,
        patch->patchDataSOA.f_normal_z,
        patch->patchDataSOA.f_nbond_x,
        patch->patchDataSOA.f_nbond_y,
        patch->patchDataSOA.f_nbond_z,
        patch->patchDataSOA.f_slow_x,
        patch->patchDataSOA.f_slow_y,
        patch->patchDataSOA.f_slow_z,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms,
#endif
        maxForceUsed
        );
    TIMER_STOP(t, KICK);

    TIMER_START(t, RATTLE1);
    rattle1_SOA(-timestep, 0);
    TIMER_STOP(t, RATTLE1);

    TIMER_START(t, SUBMITHALF);
    submitHalfstep_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.hydrogenGroupSize,
        patch->patchDataSOA.mass,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms
#endif
        );
    TIMER_STOP(t, SUBMITHALF);

    // kick 1.0
    TIMER_START(t, KICK);
    addForceToMomentum_SOA(1.0, timestep, nbondstep, slowstep,
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.recipMass,
        patch->patchDataSOA.f_normal_x,
        patch->patchDataSOA.f_normal_y,
        patch->patchDataSOA.f_normal_z,
        patch->patchDataSOA.f_nbond_x,
        patch->patchDataSOA.f_nbond_y,
        patch->patchDataSOA.f_nbond_z,
        patch->patchDataSOA.f_slow_x,
        patch->patchDataSOA.f_slow_y,
        patch->patchDataSOA.f_slow_z,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms,
#endif
        maxForceUsed
        );
    TIMER_STOP(t, KICK);

    TIMER_START(t, RATTLE1);
    rattle1_SOA(timestep, 1);
    TIMER_STOP(t, RATTLE1);

    // save total force in computeGlobal
    if (doTcl || doColvars) {
      computeGlobal->saveTotalForces(patch);
    }

    TIMER_START(t, SUBMITHALF);
    submitHalfstep_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.hydrogenGroupSize,
        patch->patchDataSOA.mass,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms
#endif
        );
    TIMER_STOP(t, SUBMITHALF);

    // kick -0.5
    TIMER_START(t, KICK);
    addForceToMomentum_SOA(-0.5, timestep, nbondstep, slowstep,
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.recipMass,
        patch->patchDataSOA.f_normal_x,
        patch->patchDataSOA.f_normal_y,
        patch->patchDataSOA.f_normal_z,
        patch->patchDataSOA.f_nbond_x,
        patch->patchDataSOA.f_nbond_y,
        patch->patchDataSOA.f_nbond_z,
        patch->patchDataSOA.f_slow_x,
        patch->patchDataSOA.f_slow_y,
        patch->patchDataSOA.f_slow_z,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms,
#endif
        maxForceUsed
        );
    TIMER_STOP(t, KICK);

    TIMER_START(t, SUBMITFULL);
    submitReductions_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.hydrogenGroupSize,
        patch->patchDataSOA.mass,
        patch->patchDataSOA.pos_x,
        patch->patchDataSOA.pos_y,
        patch->patchDataSOA.pos_z,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.f_normal_x,
        patch->patchDataSOA.f_normal_y,
        patch->patchDataSOA.f_normal_z,
        patch->patchDataSOA.f_nbond_x,
        patch->patchDataSOA.f_nbond_y,
        patch->patchDataSOA.f_nbond_z,
        patch->patchDataSOA.f_slow_x,
        patch->patchDataSOA.f_slow_y,
        patch->patchDataSOA.f_slow_z,
        patch->patchDataSOA.numAtoms
#endif
        );
    TIMER_STOP(t, SUBMITFULL);

    rebalanceLoad(step);
  } // scriptTask == SCRIPT_RUN

#if defined(NAMD_NVTX_ENABLED) || defined(NAMD_CMK_TRACE_ENABLED) || defined(NAMD_ROCTX_ENABLED)
  int& eon = patch->flags.event_on;
  int epid = (simParams->beginEventPatchID <= patch->getPatchID()
      && patch->getPatchID() <= simParams->endEventPatchID);
  int beginStep = simParams->beginEventStep;
  int endStep = simParams->endEventStep;
  bool controlProfiling = patch->getPatchID() == 0;
#endif

  for ( ++step; step <= numberOfSteps; ++step ) {
    const int isCollection = restartFrequency.check(step) +
      dcdFrequency.check(step) + velDcdFrequency.check(step) +
      forceDcdFrequency.check(step) + imdFrequency.check(step);
    const int isMigration = stepsPerCycle.check(step);
    doEnergy = energyFrequency.check(step);

#if defined(NAMD_NVTX_ENABLED) || defined(NAMD_CMK_TRACE_ENABLED) || defined(NAMD_ROCTX_ENABLED)
    eon = epid && (beginStep < step && step <= endStep);

    if (controlProfiling && step == beginStep) {
      NAMD_PROFILE_START();
    }
    if (controlProfiling && step == endStep) {
      NAMD_PROFILE_STOP();
    }
//    NAMD_EVENT_START(eon, NamdProfileEvent::INTEGRATE_SOA_1);
    char buf[32];
    sprintf(buf, "%s: %d", NamdProfileEventStr[NamdProfileEvent::INTEGRATE_SOA_1], patch->getPatchID());
    NAMD_EVENT_START_EX(eon, NamdProfileEvent::INTEGRATE_SOA_1, buf);
#endif

    if ( simParams->stochRescaleOn ) {
      stochRescaleVelocities_SOA(step);
    }

    if ( simParams->berendsenPressureOn ) {
      berendsenPressure_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.hydrogenGroupSize,
        patch->patchDataSOA.mass,
        patch->patchDataSOA.pos_x,
        patch->patchDataSOA.pos_y,
        patch->patchDataSOA.pos_z,
        patch->patchDataSOA.numAtoms,
#endif
        step);
    }

    // kick 0.5
    TIMER_START(t, KICK);
    addForceToMomentum_SOA(0.5, timestep, nbondstep, slowstep,
#ifndef SOA_SIMPLIFY_PARAMS
                           patch->patchDataSOA.recipMass,
                           patch->patchDataSOA.f_normal_x,
                           patch->patchDataSOA.f_normal_y,
                           patch->patchDataSOA.f_normal_z,
                           patch->patchDataSOA.f_nbond_x,
                           patch->patchDataSOA.f_nbond_y,
                           patch->patchDataSOA.f_nbond_z,
                           patch->patchDataSOA.f_slow_x,
                           patch->patchDataSOA.f_slow_y,
                           patch->patchDataSOA.f_slow_z,
                           patch->patchDataSOA.vel_x,
                           patch->patchDataSOA.vel_y,
                           patch->patchDataSOA.vel_z,
                           patch->patchDataSOA.numAtoms,
#endif
                           maxForceUsed
      );
    TIMER_STOP(t, KICK);

    // maximumMove checks velocity bound on atoms
    TIMER_START(t, MAXMOVE);
    maximumMove_SOA(timestep, maxvel2
#ifndef SOA_SIMPLIFY_PARAMS
                    ,
                    patch->patchDataSOA.vel_x,
                    patch->patchDataSOA.vel_y,
                    patch->patchDataSOA.vel_z,
                    patch->patchDataSOA.numAtoms
#endif
      );
    TIMER_STOP(t, MAXMOVE);


    NAMD_EVENT_STOP(eon, NamdProfileEvent::INTEGRATE_SOA_1);

    // Check to see if Langevin piston is enabled this step:
    //   ! ((step-1-slowFreq/2) % slowFreq)
    if ( langevinPistonFrequency.check(step) ) {
      // if (langevinPistonStep) {
      // drift 0.5
      TIMER_START(t, DRIFT);
      addVelocityToPosition_SOA(0.5*timestep
#ifndef SOA_SIMPLIFY_PARAMS
                                ,
                                patch->patchDataSOA.vel_x,
                                patch->patchDataSOA.vel_y,
                                patch->patchDataSOA.vel_z,
                                patch->patchDataSOA.pos_x,
                                patch->patchDataSOA.pos_y,
                                patch->patchDataSOA.pos_z,
                                patch->patchDataSOA.numAtoms
#endif
        );
      TIMER_STOP(t, DRIFT);
      // There is a blocking receive inside of langevinPiston()
      // that might suspend the current thread of execution,
      // so split profiling around this conditional block.
      langevinPiston_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.hydrogenGroupSize,
        patch->patchDataSOA.mass,
        patch->patchDataSOA.pos_x,
        patch->patchDataSOA.pos_y,
        patch->patchDataSOA.pos_z,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms,
#endif
        step
        );

      // drift 0.5
      TIMER_START(t, DRIFT);
      addVelocityToPosition_SOA(0.5*timestep
#ifndef SOA_SIMPLIFY_PARAMS
                                ,
                                patch->patchDataSOA.vel_x,
                                patch->patchDataSOA.vel_y,
                                patch->patchDataSOA.vel_z,
                                patch->patchDataSOA.pos_x,
                                patch->patchDataSOA.pos_y,
                                patch->patchDataSOA.pos_z,
                                patch->patchDataSOA.numAtoms
#endif
        );
      TIMER_STOP(t, DRIFT);
    }
    else {
      // drift 1.0
      TIMER_START(t, DRIFT);
      addVelocityToPosition_SOA(timestep
#ifndef SOA_SIMPLIFY_PARAMS
                                ,
                                patch->patchDataSOA.vel_x,
                                patch->patchDataSOA.vel_y,
                                patch->patchDataSOA.vel_z,
                                patch->patchDataSOA.pos_x,
                                patch->patchDataSOA.pos_y,
                                patch->patchDataSOA.pos_z,
                                patch->patchDataSOA.numAtoms
#endif
        );
      TIMER_STOP(t, DRIFT);
    }

    //NAMD_EVENT_START(eon, NamdProfileEvent::INTEGRATE_SOA_2);

    // There are NO sends in submitHalfstep() just local summation
    // into the Reduction struct.
    TIMER_START(t, SUBMITHALF);
    submitHalfstep_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
      patch->patchDataSOA.hydrogenGroupSize,
      patch->patchDataSOA.mass,
      patch->patchDataSOA.vel_x,
      patch->patchDataSOA.vel_y,
      patch->patchDataSOA.vel_z,
      patch->patchDataSOA.numAtoms
#endif
      );
    TIMER_STOP(t, SUBMITHALF);

    //doNonbonded = !(step%nonbondedFrequency);
    doNonbonded = nonbondedFrequency.check(step);
    //doFullElectrostatics = (dofull && !(step%fullElectFrequency));
    doFullElectrostatics = (dofull && fullElectFrequency.check(step));

    maxForceUsed = Results::normal;
    if ( doNonbonded ) maxForceUsed = Results::nbond;
    if ( doFullElectrostatics ) maxForceUsed = Results::slow;

    // Migrate Atoms on stepsPerCycle
    // Check to see if this is energy evaluation step:
    //   doEnergy = ! ( step % energyFrequency );
    doVirial = 1;
    doKineticEnergy = 1;
    doMomenta = 1;

    //NAMD_EVENT_STOP(eon, NamdProfileEvent::INTEGRATE_SOA_2); // integrate_SOA 2

    // The current thread of execution will suspend in runComputeObjects().
    // Check to see if we are at a migration step:
    //   runComputeObjects_SOA(!(step%stepsPerCycle), step<numberOfSteps);
    runComputeObjects_SOA(isMigration, step<numberOfSteps, step);

    NAMD_EVENT_START(eon, NamdProfileEvent::INTEGRATE_SOA_3);

    TIMER_START(t, VELBBK1);
    langevinVelocitiesBBK1_SOA(
        timestep
#ifndef SOA_SIMPLIFY_PARAMS
        ,
        patch->patchDataSOA.langevinParam,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms
#endif
        );
    TIMER_STOP(t, VELBBK1);

    // kick 1.0
    TIMER_START(t, KICK);
    addForceToMomentum_SOA(1.0, timestep, nbondstep, slowstep,
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.recipMass,
        patch->patchDataSOA.f_normal_x,
        patch->patchDataSOA.f_normal_y,
        patch->patchDataSOA.f_normal_z,
        patch->patchDataSOA.f_nbond_x,
        patch->patchDataSOA.f_nbond_y,
        patch->patchDataSOA.f_nbond_z,
        patch->patchDataSOA.f_slow_x,
        patch->patchDataSOA.f_slow_y,
        patch->patchDataSOA.f_slow_z,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms,
#endif
        maxForceUsed
        );
    TIMER_STOP(t, KICK);

    TIMER_START(t, VELBBK2);
    langevinVelocitiesBBK2_SOA(
        timestep
#ifndef SOA_SIMPLIFY_PARAMS
        ,
        patch->patchDataSOA.langevinParam,
        patch->patchDataSOA.langScalVelBBK2,
        patch->patchDataSOA.langScalRandBBK2,
        patch->patchDataSOA.gaussrand_x,
        patch->patchDataSOA.gaussrand_y,
        patch->patchDataSOA.gaussrand_z,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms
#endif
        );
    TIMER_STOP(t, VELBBK2);
    
    TIMER_START(t, RATTLE1);
    rattle1_SOA(timestep, 1);
    TIMER_STOP(t, RATTLE1);

    // save total force in computeGlobal
    if (doTcl || doColvars) {
      computeGlobal->saveTotalForces(patch);
    }

    TIMER_START(t, SUBMITHALF);
    submitHalfstep_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.hydrogenGroupSize,
        patch->patchDataSOA.mass,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms
#endif
        );
    TIMER_STOP(t, SUBMITHALF);

    // kick -0.5
    TIMER_START(t, KICK);
    addForceToMomentum_SOA(-0.5, timestep, nbondstep, slowstep,
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.recipMass,
        patch->patchDataSOA.f_normal_x,
        patch->patchDataSOA.f_normal_y,
        patch->patchDataSOA.f_normal_z,
        patch->patchDataSOA.f_nbond_x,
        patch->patchDataSOA.f_nbond_y,
        patch->patchDataSOA.f_nbond_z,
        patch->patchDataSOA.f_slow_x,
        patch->patchDataSOA.f_slow_y,
        patch->patchDataSOA.f_slow_z,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.numAtoms,
#endif
        maxForceUsed
        );
    TIMER_STOP(t, KICK);

    // XXX rattle2_SOA(timestep,step);

    TIMER_START(t, SUBMITFULL);
    submitReductions_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        patch->patchDataSOA.hydrogenGroupSize,
        patch->patchDataSOA.mass,
        patch->patchDataSOA.pos_x,
        patch->patchDataSOA.pos_y,
        patch->patchDataSOA.pos_z,
        patch->patchDataSOA.vel_x,
        patch->patchDataSOA.vel_y,
        patch->patchDataSOA.vel_z,
        patch->patchDataSOA.f_normal_x,
        patch->patchDataSOA.f_normal_y,
        patch->patchDataSOA.f_normal_z,
        patch->patchDataSOA.f_nbond_x,
        patch->patchDataSOA.f_nbond_y,
        patch->patchDataSOA.f_nbond_z,
        patch->patchDataSOA.f_slow_x,
        patch->patchDataSOA.f_slow_y,
        patch->patchDataSOA.f_slow_z,
        patch->patchDataSOA.numAtoms
#endif
        );
    TIMER_STOP(t, SUBMITFULL);
#ifdef TESTPID
    if (1) {
      int pid = TESTPID;
      if (patch->patchID == pid) {
        const PatchDataSOA& p = patch->patchDataSOA;
        int n = p.numAtoms;
#if 0
        fprintf(stderr, "Patch %d has %d atoms\n", pid, n);
        fprintf(stderr, "%3s  %8s  %12s  %12s  %12s\n",
            "", "id", "fnormal_x", "fnbond_x", "fslow_x");
        for (int i=0;  i < n;  i++) {
          int index = p.id[i];
          fprintf(stderr, "%3d  %8d  %12.8f  %12.8f  %12.8f\n",
              i, index, p.f_normal_x[i], p.f_nbond_x[i], p.f_slow_x[i]);
        }
#else
        Vector *f_normal = new Vector[n];
        Vector *f_nbond = new Vector[n];
        Vector *f_slow = new Vector[n];
        for (int i=0;  i < n;  i++) {
          f_normal[i].x = p.f_normal_x[i];
          f_normal[i].y = p.f_normal_y[i];
          f_normal[i].z = p.f_normal_z[i];
          f_nbond[i].x = p.f_nbond_x[i];
          f_nbond[i].y = p.f_nbond_y[i];
          f_nbond[i].z = p.f_nbond_z[i];
          f_slow[i].x = p.f_slow_x[i];
          f_slow[i].y = p.f_slow_y[i];
          f_slow[i].z = p.f_slow_z[i];
        }
        TestArray_write<double>(
            "f_normal_good.bin", "f_normal good", (double*)f_normal, 3*n);
        TestArray_write<double>(
            "f_nbond_good.bin", "f_nbond good", (double*)f_nbond, 3*n);
        TestArray_write<double>(
            "f_slow_good.bin", "f_slow good", (double*)f_slow, 3*n);
        delete [] f_normal;
        delete [] f_nbond;
        delete [] f_slow;
#endif
      }
    }
#endif

    // Do collections if any checks below are "on."
    // We add because we can't short-circuit.
    TIMER_START(t, SUBMITCOLLECT);
    if (isCollection) {
      submitCollections_SOA(step);
    }
    TIMER_STOP(t, SUBMITCOLLECT);

    NAMD_EVENT_STOP(eon, NamdProfileEvent::INTEGRATE_SOA_3); // integrate_SOA 3

    rebalanceLoad(step);
  }

  patch->copy_updates_to_AOS();

  TIMER_DONE(t);
  if (patch->patchID == SPECIAL_PATCH_ID) {
    printf("Timer collection reporting in microseconds for "
        "Patch %d\n", patch->patchID);
    TIMER_REPORT(t);
  }
}


// XXX inline it?
// XXX does not handle fixed atoms
// Each timestep:  dt = scaling * (timestep / TIMEFACTOR);
void Sequencer::addForceToMomentum_SOA(
    const double scaling,
    double       dt_normal,               // timestep Results::normal = 0
    double       dt_nbond,                // timestep Results::nbond  = 1
    double       dt_slow,                 // timestep Results::slow   = 2
#ifndef SOA_SIMPLIFY_PARAMS
    const double * __restrict recipMass,
    const double * __restrict f_normal_x, // force    Results::normal = 0
    const double * __restrict f_normal_y,
    const double * __restrict f_normal_z,
    const double * __restrict f_nbond_x,  // force    Results::nbond  = 1
    const double * __restrict f_nbond_y,
    const double * __restrict f_nbond_z,
    const double * __restrict f_slow_x,   // force    Results::slow   = 2
    const double * __restrict f_slow_y,
    const double * __restrict f_slow_z,
    double       * __restrict vel_x,
    double       * __restrict vel_y,
    double       * __restrict vel_z,
    int numAtoms,
#endif
    int maxForceNumber
    ) {
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::ADD_FORCE_TO_MOMENTUM_SOA);

#ifdef SOA_SIMPLIFY_PARAMS
  const double * __restrict recipMass = patch->patchDataSOA.recipMass;
  // force    Results::normal = 0
  const double * __restrict f_normal_x = patch->patchDataSOA.f_normal_x;
  const double * __restrict f_normal_y = patch->patchDataSOA.f_normal_y;
  const double * __restrict f_normal_z = patch->patchDataSOA.f_normal_z;
  // force    Results::nbond  = 1
  const double * __restrict f_nbond_x = patch->patchDataSOA.f_nbond_x;
  const double * __restrict f_nbond_y = patch->patchDataSOA.f_nbond_y;
  const double * __restrict f_nbond_z = patch->patchDataSOA.f_nbond_z;
  // force    Results::slow   = 2
  const double * __restrict f_slow_x = patch->patchDataSOA.f_slow_x;
  const double * __restrict f_slow_y = patch->patchDataSOA.f_slow_y;
  const double * __restrict f_slow_z = patch->patchDataSOA.f_slow_z;
  double       * __restrict vel_x = patch->patchDataSOA.vel_x;
  double       * __restrict vel_y = patch->patchDataSOA.vel_y;
  double       * __restrict vel_z = patch->patchDataSOA.vel_z;
  int numAtoms = patch->patchDataSOA.numAtoms;
#endif
  //
  // We could combine each case into a single loop with breaks,
  // with all faster forces also summed, like addForceToMomentum3().
  //
  // Things to consider:
  // - Do we always use acceleration (f/m) instead of just plain force?
  //   Then we could instead buffer accel_slow, accel_nbond, etc.
  // - We will always need one multiply, since each dt includes
  //   also a scaling factor.
  //
  
#if 0
    if(this->patch->getPatchID() == 538){
      // fprintf(stderr, "Old Positions %lf %lf %lf\n", patch->patchDataSOA.pos_x[43], patch->patchDataSOA.pos_y[43], patch->patchDataSOA.pos_z[43]);
      // fprintf(stderr, "Old Velocities %lf %lf %lf\n", vel_x[43], vel_y[43], vel_z[ 43]);
      // fprintf(stderr, "Adding forces %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", 
      //   f_slow_x[43],   f_slow_y[43],   f_slow_z[43], 
      //   f_nbond_x[43],  f_nbond_y[43],  f_nbond_z[43], 
      //   f_normal_x[43], f_normal_y[43], f_normal_z[43]);
      fprintf(stderr, "Old Positions %lf %lf %lf\n", patch->patchDataSOA.pos_x[0], patch->patchDataSOA.pos_y[0], patch->patchDataSOA.pos_z[0]);
      fprintf(stderr, "Old Velocities %lf %lf %lf\n", vel_x[0], vel_y[0], vel_z[ 0]);
      fprintf(stderr, "Adding forces %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", 
        f_slow_x[43],   f_slow_y[43],   f_slow_z[43], 
        f_nbond_x[43],  f_nbond_y[43],  f_nbond_z[43], 
        f_normal_x[43], f_normal_y[43], f_normal_z[43]);
    }
#endif
  switch (maxForceNumber) {
    case Results::slow:
      dt_slow *= scaling;
      for (int i=0;  i < numAtoms;  i++) {
        vel_x[i] += f_slow_x[i] * recipMass[i] * dt_slow;
        vel_y[i] += f_slow_y[i] * recipMass[i] * dt_slow;
        vel_z[i] += f_slow_z[i] * recipMass[i] * dt_slow;
      }
      // fall through because we will always have the "faster" forces
    case Results::nbond:
      dt_nbond *= scaling;
      for (int i=0;  i < numAtoms;  i++) {
        vel_x[i] += f_nbond_x[i] * recipMass[i] * dt_nbond;
        vel_y[i] += f_nbond_y[i] * recipMass[i] * dt_nbond;
        vel_z[i] += f_nbond_z[i] * recipMass[i] * dt_nbond;
      }
      // fall through because we will always have the "faster" forces
    case Results::normal:
      dt_normal *= scaling;
      for (int i=0;  i < numAtoms;  i++) {
        vel_x[i] += f_normal_x[i] * recipMass[i] * dt_normal;
        vel_y[i] += f_normal_y[i] * recipMass[i] * dt_normal;
        vel_z[i] += f_normal_z[i] * recipMass[i] * dt_normal;
      }
  }
}


// XXX inline it?
// XXX does not handle fixed atoms
// Timestep:  dt = scaling * (timestep / TIMEFACTOR);
void Sequencer::addVelocityToPosition_SOA(
    const double dt   ///< scaled timestep
#ifndef SOA_SIMPLIFY_PARAMS
    ,
    const double * __restrict vel_x,
    const double * __restrict vel_y,
    const double * __restrict vel_z,
    double *       __restrict pos_x,
    double *       __restrict pos_y,
    double *       __restrict pos_z,
    int numAtoms      ///< number of atoms
#endif
    ) {
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::ADD_VELOCITY_TO_POSITION_SOA);
#ifdef SOA_SIMPLIFY_PARAMS
  const double * __restrict vel_x = patch->patchDataSOA.vel_x;
  const double * __restrict vel_y = patch->patchDataSOA.vel_y;
  const double * __restrict vel_z = patch->patchDataSOA.vel_z;
  double *       __restrict pos_x = patch->patchDataSOA.pos_x;
  double *       __restrict pos_y = patch->patchDataSOA.pos_y;
  double *       __restrict pos_z = patch->patchDataSOA.pos_z;
  int numAtoms = patch->patchDataSOA.numAtoms;
#endif
  for (int i=0;  i < numAtoms;  i++) {
    pos_x[i] += vel_x[i] * dt;
    pos_y[i] += vel_y[i] * dt;
    pos_z[i] += vel_z[i] * dt;
  }
#if 0
    if(this->patch->getPatchID() == 538){
      fprintf(stderr, "New Positions %lf %lf %lf\n",  pos_x[43], pos_y[43], pos_z[43]);
      fprintf(stderr, "New Velocities %lf %lf %lf\n", vel_x[43], vel_y[43], vel_z[43]);
    }
#endif

}


void Sequencer::submitHalfstep_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
    const int    * __restrict hydrogenGroupSize,
    const float  * __restrict mass,
    const double * __restrict vel_x,
    const double * __restrict vel_y,
    const double * __restrict vel_z,
    int numAtoms
#endif
    ) {
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::SUBMIT_HALFSTEP_SOA);
#ifdef SOA_SIMPLIFY_PARAMS
  const int    * __restrict hydrogenGroupSize = patch->patchDataSOA.hydrogenGroupSize;
  const float  * __restrict mass = patch->patchDataSOA.mass;
  const double * __restrict vel_x = patch->patchDataSOA.vel_x;
  const double * __restrict vel_y = patch->patchDataSOA.vel_y;
  const double * __restrict vel_z = patch->patchDataSOA.vel_z;
  int numAtoms = patch->patchDataSOA.numAtoms;
#endif
  if ( 1 /* doKineticEnergy || patch->flags.doVirial */ ) {
    BigReal kineticEnergy = 0;
    Tensor virial;
    for (int i=0;  i < numAtoms;  i++) {
      // scalar kineticEnergy += mass[i] * vel[i]^2
      kineticEnergy += mass[i] *
        (vel_x[i]*vel_x[i] + vel_y[i]*vel_y[i] + vel_z[i]*vel_z[i]);
      // tensor virial += mass[i] * outer_product(vel[i], vel[i])
      virial.xx += mass[i] * vel_x[i] * vel_x[i];
      virial.xy += mass[i] * vel_x[i] * vel_y[i];
      virial.xz += mass[i] * vel_x[i] * vel_z[i];
      virial.yx += mass[i] * vel_y[i] * vel_x[i];
      virial.yy += mass[i] * vel_y[i] * vel_y[i];
      virial.yz += mass[i] * vel_y[i] * vel_z[i];
      virial.zx += mass[i] * vel_z[i] * vel_x[i];
      virial.zy += mass[i] * vel_z[i] * vel_y[i];
      virial.zz += mass[i] * vel_z[i] * vel_z[i];
    }
    kineticEnergy *= 0.5 * 0.5;
    virial *= 0.5;

#ifdef NODEGROUP_FORCE_REGISTER
    if(simParams->CUDASOAintegrate){
      CUDASequencer->patchData->reduction->item(REDUCTION_HALFSTEP_KINETIC_ENERGY) += kineticEnergy;
      ADD_TENSOR_OBJECT(CUDASequencer->patchData->reduction,REDUCTION_VIRIAL_NORMAL,virial);
    }
    else{
#endif
      reduction->item(REDUCTION_HALFSTEP_KINETIC_ENERGY) += kineticEnergy;
      ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,virial);
#ifdef NODEGROUP_FORCE_REGISTER
    }
#endif
  }

  if ( 1 /* doKineticEnergy || patch->flags.doVirial */ ) {
    BigReal intKineticEnergy = 0;
    Tensor intVirialNormal;
    int hgs;
    for (int i=0;  i < numAtoms;  i += hgs) {
      // find velocity of center-of-mass of hydrogen group
      // calculate mass-weighted velocity
      hgs = hydrogenGroupSize[i];
      BigReal m_cm = 0;
      BigReal v_cm_x = 0;
      BigReal v_cm_y = 0;
      BigReal v_cm_z = 0;
      for (int j = i;  j < (i+hgs);  j++) {
        m_cm += mass[j];
        v_cm_x += mass[j] * vel_x[j];
        v_cm_y += mass[j] * vel_y[j];
        v_cm_z += mass[j] * vel_z[j];
      }
      BigReal recip_m_cm = 1.0 / m_cm;
      v_cm_x *= recip_m_cm;
      v_cm_y *= recip_m_cm;
      v_cm_z *= recip_m_cm;
      // sum virial contributions wrt vel center-of-mass
      for (int j = i;  j < (i+hgs);  j++) {
        BigReal dv_x = vel_x[j] - v_cm_x;
        BigReal dv_y = vel_y[j] - v_cm_y;
        BigReal dv_z = vel_z[j] - v_cm_z;
        // scalar intKineticEnergy += mass[j] * dot_product(vel[j], dv)
        intKineticEnergy += mass[j] *
          (vel_x[j] * dv_x + vel_y[j] * dv_y + vel_z[j] * dv_z);
        // tensor intVirialNormal += mass[j] * outer_product(vel[j], dv)
        intVirialNormal.xx += mass[j] * vel_x[j] * dv_x;
        intVirialNormal.xy += mass[j] * vel_x[j] * dv_y;
        intVirialNormal.xz += mass[j] * vel_x[j] * dv_z;
        intVirialNormal.yx += mass[j] * vel_y[j] * dv_x;
        intVirialNormal.yy += mass[j] * vel_y[j] * dv_y;
        intVirialNormal.yz += mass[j] * vel_y[j] * dv_z;
        intVirialNormal.zx += mass[j] * vel_z[j] * dv_x;
        intVirialNormal.zy += mass[j] * vel_z[j] * dv_y;
        intVirialNormal.zz += mass[j] * vel_z[j] * dv_z;
      }
    }
    intKineticEnergy *= 0.5 * 0.5;
    intVirialNormal *= 0.5;
#ifdef NODEGROUP_FORCE_REGISTER
    if(simParams->CUDASOAintegrate){
      CUDASequencer->patchData->reduction->item(REDUCTION_INT_HALFSTEP_KINETIC_ENERGY) += intKineticEnergy;
      ADD_TENSOR_OBJECT(CUDASequencer->patchData->reduction,REDUCTION_INT_VIRIAL_NORMAL, intVirialNormal);
    }
    else{
#endif
      reduction->item(REDUCTION_INT_HALFSTEP_KINETIC_ENERGY)
        += intKineticEnergy;
      ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NORMAL,
        intVirialNormal);
#ifdef NODEGROUP_FORCE_REGISTER
    }
#endif
  }
}


//
// XXX
//
void Sequencer::submitReductions_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
    const int    * __restrict hydrogenGroupSize,
    const float  * __restrict mass,
    const double * __restrict pos_x,
    const double * __restrict pos_y,
    const double * __restrict pos_z,
    const double * __restrict vel_x,
    const double * __restrict vel_y,
    const double * __restrict vel_z,
    const double * __restrict f_normal_x,
    const double * __restrict f_normal_y,
    const double * __restrict f_normal_z,
    const double * __restrict f_nbond_x,
    const double * __restrict f_nbond_y,
    const double * __restrict f_nbond_z,
    const double * __restrict f_slow_x,
    const double * __restrict f_slow_y,
    const double * __restrict f_slow_z,
    int numAtoms
#endif
    ) {
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::SUBMIT_REDUCTIONS_SOA);
#ifdef SOA_SIMPLIFY_PARAMS
  const int    * __restrict hydrogenGroupSize = patch->patchDataSOA.hydrogenGroupSize;
  const float  * __restrict mass = patch->patchDataSOA.mass;
  const double * __restrict pos_x = patch->patchDataSOA.pos_x;
  const double * __restrict pos_y = patch->patchDataSOA.pos_y;
  const double * __restrict pos_z = patch->patchDataSOA.pos_z;
  const double * __restrict vel_x = patch->patchDataSOA.vel_x;
  const double * __restrict vel_y = patch->patchDataSOA.vel_y;
  const double * __restrict vel_z = patch->patchDataSOA.vel_z;
  const double * __restrict f_normal_x = patch->patchDataSOA.f_normal_x;
  const double * __restrict f_normal_y = patch->patchDataSOA.f_normal_y;
  const double * __restrict f_normal_z = patch->patchDataSOA.f_normal_z;
  const double * __restrict f_nbond_x = patch->patchDataSOA.f_nbond_x;
  const double * __restrict f_nbond_y = patch->patchDataSOA.f_nbond_y;
  const double * __restrict f_nbond_z = patch->patchDataSOA.f_nbond_z;
  const double * __restrict f_slow_x = patch->patchDataSOA.f_slow_x;
  const double * __restrict f_slow_y = patch->patchDataSOA.f_slow_y;
  const double * __restrict f_slow_z = patch->patchDataSOA.f_slow_z;
  int numAtoms = patch->patchDataSOA.numAtoms;
#endif

#ifdef NODEGROUP_FORCE_REGISTER
  if(simParams->CUDASOAintegrate){
    CUDASequencer->patchData->reduction->item(REDUCTION_ATOM_CHECKSUM) += numAtoms;
    CUDASequencer->patchData->reduction->item(REDUCTION_MARGIN_VIOLATIONS) += patch->marginViolations;
  }else{
#endif
    reduction->item(REDUCTION_ATOM_CHECKSUM) += numAtoms;
    reduction->item(REDUCTION_MARGIN_VIOLATIONS) += patch->marginViolations;
#ifdef NODEGROUP_FORCE_REGISTER
  }
#endif

  if ( 1 /* doKineticEnergy || doMomenta || patch->flags.doVirial */ ) {
    BigReal kineticEnergy = 0;
    BigReal momentum_x = 0;
    BigReal momentum_y = 0;
    BigReal momentum_z = 0;
    BigReal angularMomentum_x = 0;
    BigReal angularMomentum_y = 0;
    BigReal angularMomentum_z = 0;
    BigReal origin_x = patch->lattice.origin().x;
    BigReal origin_y = patch->lattice.origin().y;
    BigReal origin_z = patch->lattice.origin().z;

    // XXX pairInteraction

    for (int i=0;  i < numAtoms;  i++) {

      // scalar kineticEnergy += mass[i] * dot_product(vel[i], vel[i])
      kineticEnergy += mass[i] *
        (vel_x[i]*vel_x[i] + vel_y[i]*vel_y[i] + vel_z[i]*vel_z[i]);

      // vector momentum += mass[i] * vel[i]
      momentum_x += mass[i] * vel_x[i];
      momentum_y += mass[i] * vel_y[i];
      momentum_z += mass[i] * vel_z[i];

      // vector dpos = pos[i] - origin
      BigReal dpos_x = pos_x[i] - origin_x;
      BigReal dpos_y = pos_y[i] - origin_y;
      BigReal dpos_z = pos_z[i] - origin_z;

      // vector angularMomentum += mass[i] * cross_product(dpos, vel[i])
      angularMomentum_x += mass[i] * (dpos_y*vel_z[i] - dpos_z*vel_y[i]);
      angularMomentum_y += mass[i] * (dpos_z*vel_x[i] - dpos_x*vel_z[i]);
      angularMomentum_z += mass[i] * (dpos_x*vel_y[i] - dpos_y*vel_x[i]);
    }

    // XXX missing Drude

    kineticEnergy *= 0.5;
    Vector momentum(momentum_x, momentum_y, momentum_z);
    Vector angularMomentum(angularMomentum_x, angularMomentum_y,
        angularMomentum_z);

#ifdef NODEGROUP_FORCE_REGISTER
    if(simParams->CUDASOAintegrate){
      CUDASequencer->patchData->reduction->item(REDUCTION_CENTERED_KINETIC_ENERGY) += kineticEnergy;
      ADD_VECTOR_OBJECT(CUDASequencer->patchData->reduction,REDUCTION_MOMENTUM,momentum);
      ADD_VECTOR_OBJECT(CUDASequencer->patchData->reduction,REDUCTION_ANGULAR_MOMENTUM,angularMomentum);
    }else{
#endif
      reduction->item(REDUCTION_CENTERED_KINETIC_ENERGY) += kineticEnergy;
      ADD_VECTOR_OBJECT(reduction,REDUCTION_MOMENTUM,momentum);
      ADD_VECTOR_OBJECT(reduction,REDUCTION_ANGULAR_MOMENTUM,angularMomentum);
#ifdef NODEGROUP_FORCE_REGISTER
    }
#endif
  }
  // For non-Multigrator doKineticEnergy = 1 always
  if ( 1 /* doKineticEnergy || patch->flags.doVirial */ ) {
    BigReal intKineticEnergy = 0;
    Tensor intVirialNormal;
    Tensor intVirialNbond;
    Tensor intVirialSlow;

    int hgs = 1;  // hydrogen group size
    for (int i=0;  i < numAtoms;  i += hgs) {
      hgs = hydrogenGroupSize[i];
      int j;
      BigReal m_cm = 0;
      BigReal r_cm_x = 0;
      BigReal r_cm_y = 0;
      BigReal r_cm_z = 0;
      BigReal v_cm_x = 0;
      BigReal v_cm_y = 0;
      BigReal v_cm_z = 0;
      for ( j = i; j < (i+hgs); ++j ) {
        m_cm += mass[j];
        r_cm_x += mass[j] * pos_x[j];
        r_cm_y += mass[j] * pos_y[j];
        r_cm_z += mass[j] * pos_z[j];
        v_cm_x += mass[j] * vel_x[j];
        v_cm_y += mass[j] * vel_y[j];
        v_cm_z += mass[j] * vel_z[j];
      }
      BigReal inv_m_cm = namd_reciprocal(m_cm);
      r_cm_x *= inv_m_cm;
      r_cm_y *= inv_m_cm;
      r_cm_z *= inv_m_cm;
      v_cm_x *= inv_m_cm;
      v_cm_y *= inv_m_cm;
      v_cm_z *= inv_m_cm;

      // XXX removed pairInteraction
      for ( j = i; j < (i+hgs); ++j ) {
        // XXX removed fixed atoms

        // vector vel[j] used twice below
        BigReal v_x = vel_x[j];
        BigReal v_y = vel_y[j];
        BigReal v_z = vel_z[j];

        // vector dv = vel[j] - v_cm
        BigReal dv_x = v_x - v_cm_x;
        BigReal dv_y = v_y - v_cm_y;
        BigReal dv_z = v_z - v_cm_z;

        // scalar intKineticEnergy += mass[j] * dot_product(v, dv)
        intKineticEnergy += mass[j] *
          (v_x * dv_x + v_y * dv_y + v_z * dv_z);

        // vector dr = pos[j] - r_cm
        BigReal dr_x = pos_x[j] - r_cm_x;
        BigReal dr_y = pos_y[j] - r_cm_y;
        BigReal dr_z = pos_z[j] - r_cm_z;

        // tensor intVirialNormal += outer_product(f_normal[j], dr)
        intVirialNormal.xx += f_normal_x[j] * dr_x;
        intVirialNormal.xy += f_normal_x[j] * dr_y;
        intVirialNormal.xz += f_normal_x[j] * dr_z;
        intVirialNormal.yx += f_normal_y[j] * dr_x;
        intVirialNormal.yy += f_normal_y[j] * dr_y;
        intVirialNormal.yz += f_normal_y[j] * dr_z;
        intVirialNormal.zx += f_normal_z[j] * dr_x;
        intVirialNormal.zy += f_normal_z[j] * dr_y;
        intVirialNormal.zz += f_normal_z[j] * dr_z;

        // tensor intVirialNbond += outer_product(f_nbond[j], dr)
        intVirialNbond.xx += f_nbond_x[j] * dr_x;
        intVirialNbond.xy += f_nbond_x[j] * dr_y;
        intVirialNbond.xz += f_nbond_x[j] * dr_z;
        intVirialNbond.yx += f_nbond_y[j] * dr_x;
        intVirialNbond.yy += f_nbond_y[j] * dr_y;
        intVirialNbond.yz += f_nbond_y[j] * dr_z;
        intVirialNbond.zx += f_nbond_z[j] * dr_x;
        intVirialNbond.zy += f_nbond_z[j] * dr_y;
        intVirialNbond.zz += f_nbond_z[j] * dr_z;

        // tensor intVirialSlow += outer_product(f_slow[j], dr)
        intVirialSlow.xx += f_slow_x[j] * dr_x;
        intVirialSlow.xy += f_slow_x[j] * dr_y;
        intVirialSlow.xz += f_slow_x[j] * dr_z;
        intVirialSlow.yx += f_slow_y[j] * dr_x;
        intVirialSlow.yy += f_slow_y[j] * dr_y;
        intVirialSlow.yz += f_slow_y[j] * dr_z;
        intVirialSlow.zx += f_slow_z[j] * dr_x;
        intVirialSlow.zy += f_slow_z[j] * dr_y;
        intVirialSlow.zz += f_slow_z[j] * dr_z;
      }
    }

    intKineticEnergy *= 0.5;

#ifdef NODEGROUP_FORCE_REGISTER
    if(simParams->CUDASOAintegrate){
      // JM: Every PE will have its own copy of CUDASequencer, since it's a 
      //     group. However, they all share the same nodegroup pointer, 
      //     which means the reduction object is the same across all PEs
      CUDASequencer->patchData->reduction->item(REDUCTION_INT_CENTERED_KINETIC_ENERGY) += intKineticEnergy;
      ADD_TENSOR_OBJECT(CUDASequencer->patchData->reduction,REDUCTION_INT_VIRIAL_NORMAL,intVirialNormal);
      ADD_TENSOR_OBJECT(CUDASequencer->patchData->reduction,REDUCTION_INT_VIRIAL_NBOND,intVirialNbond);
      ADD_TENSOR_OBJECT(CUDASequencer->patchData->reduction,REDUCTION_INT_VIRIAL_SLOW,intVirialSlow);
    }else{
#endif
      reduction->item(REDUCTION_INT_CENTERED_KINETIC_ENERGY) += intKineticEnergy;
      ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_NORMAL,intVirialNormal);
      ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_NBOND,intVirialNbond);
      ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_SLOW,intVirialSlow);
#ifdef NODEGROUP_FORCE_REGISTER
     }
#endif
  }
  // XXX removed pressure profile

  // XXX removed fixed atoms

  if(!simParams->CUDASOAintegrate) reduction->submit();

  // XXX removed pressure profile reduction
}


void Sequencer::submitCollections_SOA(int step, int zeroVel /* = 0 */)
{
  //
  // Copy updates of SOA back into AOS for collections.
  //
  // XXX Could update positions and velocities separately.
  //
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::SUBMIT_COLLECTIONS_SOA);
  //
  // XXX Poor implementation here!
  // The selector functions called below in Output.C are
  // doing several tests and in an average use case calculating
  // at least two mod functions.
  //
  // However, most steps are NOT output steps!
  //
  int is_pos_needed = Output::coordinateNeeded(step);
  int is_vel_needed = Output::velocityNeeded(step);
  int is_f_needed = Output::forceNeeded(step);
  if (!simParams->useDeviceMigration) {  // This is already done for GPU migration
    if ( is_pos_needed || is_vel_needed ) {
      patch->copy_updates_to_AOS();
    }
  }
  if ( is_pos_needed ) {
    collection->submitPositions(step,patch->atom,patch->lattice,is_pos_needed);
  }
  if ( is_vel_needed ) {
    collection->submitVelocities(step,zeroVel,patch->atom,is_vel_needed);
  }
  if ( is_f_needed ) {
    int maxForceUsed = patch->flags.maxForceUsed;
    if ( maxForceUsed > Results::slow ) maxForceUsed = Results::slow;
    collection->submitForces(step,patch->atom,maxForceUsed,patch->f,is_f_needed);
  }
}


void Sequencer::maximumMove_SOA(
    const double dt,  ///< scaled timestep
    const double maxvel2  ///< square of bound on velocity
#ifndef SOA_SIMPLIFY_PARAMS
    ,
    const double * __restrict vel_x,
    const double * __restrict vel_y,
    const double * __restrict vel_z,
    int numAtoms      ///< number of atoms
#endif
    ) {
  NAMD_EVENT_RANGE_2(patch->flags.event_on, NamdProfileEvent::MAXIMUM_MOVE_SOA);
#ifdef SOA_SIMPLIFY_PARAMS
  const double * __restrict vel_x = patch->patchDataSOA.vel_x;
  const double * __restrict vel_y = patch->patchDataSOA.vel_y;
  const double * __restrict vel_z = patch->patchDataSOA.vel_z;
  int numAtoms = patch->patchDataSOA.numAtoms;
#endif

  // XXX missing maximum move

  // Loop vectorizes when replacing logical OR with summing.
  int killme = 0;
  for (int i=0;  i < numAtoms;  i++) {
    BigReal vel2 =
      vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] + vel_z[i] * vel_z[i];
    killme = killme + ( vel2 > maxvel2 );
  }
  if (killme) {
    // Found at least one atom that is moving too fast.
    // Terminating, so loop performance below doesn't matter.
    // Loop does not vectorize.
    killme = 0;
    for (int i=0;  i < numAtoms;  i++) {
      BigReal vel2 =
        vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] + vel_z[i] * vel_z[i];
      if (vel2 > maxvel2) {
        const FullAtom *a = patch->atom.begin();
        const Vector vel(vel_x[i], vel_y[i], vel_z[i]);
        const BigReal maxvel = sqrt(maxvel2);
        ++killme;
        iout << iERROR << "Atom " << (a[i].id + 1) << " velocity is "
          << ( PDBVELFACTOR * vel ) << " (limit is "
          << ( PDBVELFACTOR * maxvel ) << ", atom "
          << i << " of " << numAtoms << " on patch "
          << patch->patchID << " pe " << CkMyPe() << ")\n" << endi;
      }
    }
    iout << iERROR <<
      "Atoms moving too fast; simulation has become unstable ("
      << killme << " atoms on patch " << patch->patchID
      << " pe " << CkMyPe() << ").\n" << endi;
    Node::Object()->enableEarlyExit();
    terminate();
  }
}


void Sequencer::langevinVelocitiesBBK1_SOA(
    BigReal timestep
#ifndef SOA_SIMPLIFY_PARAMS
    ,
    const float * __restrict langevinParam,
    double      * __restrict vel_x,
    double      * __restrict vel_y,
    double      * __restrict vel_z,
    int numAtoms
#endif
    ) {
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::LANGEVIN_VELOCITIES_BBK1_SOA);
#ifdef SOA_SIMPLIFY_PARAMS
  const float * __restrict langevinParam = patch->patchDataSOA.langevinParam;
  double      * __restrict vel_x = patch->patchDataSOA.vel_x;
  double      * __restrict vel_y = patch->patchDataSOA.vel_y;
  double      * __restrict vel_z = patch->patchDataSOA.vel_z;
  int numAtoms = patch->patchDataSOA.numAtoms;
#endif
  if ( simParams->langevinOn /* && !simParams->langevin_useBAOAB */ )
  {
    // scale by TIMEFACTOR to convert to fs and then by 0.001 to ps
    // multiply by the Langevin damping coefficient, units 1/ps
    // XXX we could instead store time-scaled Langevin parameters
    BigReal dt = timestep * (0.001 * TIMEFACTOR);

    // XXX missing Drude

    //
    // The conditional inside loop prevents vectorization and doesn't
    // avoid much work since addition and multiplication are cheap.
    //
    for (int i=0;  i < numAtoms;  i++) {
      BigReal dt_gamma = dt * langevinParam[i];
      //if ( ! dt_gamma ) continue;

      BigReal scaling = 1. - 0.5 * dt_gamma;
      vel_x[i] *= scaling;
      vel_y[i] *= scaling;
      vel_z[i] *= scaling;
    }
  } // end if langevinOn
}


void Sequencer::langevinVelocitiesBBK2_SOA(
    BigReal timestep
#ifndef SOA_SIMPLIFY_PARAMS
    ,
    const float * __restrict langevinParam,
    const float * __restrict langScalVelBBK2,
    const float * __restrict langScalRandBBK2,
    float       * __restrict gaussrand_x,
    float       * __restrict gaussrand_y,
    float       * __restrict gaussrand_z,
    double      * __restrict vel_x,
    double      * __restrict vel_y,
    double      * __restrict vel_z,
    int numAtoms
#endif
    )
{
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::LANGEVIN_VELOCITIES_BBK2_SOA);
#ifdef SOA_SIMPLIFY_PARAMS
  const float * __restrict langevinParam = patch->patchDataSOA.langevinParam;
  const float * __restrict langScalVelBBK2 = patch->patchDataSOA.langScalVelBBK2;
  const float * __restrict langScalRandBBK2 = patch->patchDataSOA.langScalRandBBK2;
  float       * __restrict gaussrand_x = patch->patchDataSOA.gaussrand_x;
  float       * __restrict gaussrand_y = patch->patchDataSOA.gaussrand_y;
  float       * __restrict gaussrand_z = patch->patchDataSOA.gaussrand_z;
  double      * __restrict vel_x = patch->patchDataSOA.vel_x;
  double      * __restrict vel_y = patch->patchDataSOA.vel_y;
  double      * __restrict vel_z = patch->patchDataSOA.vel_z;
  int numAtoms = patch->patchDataSOA.numAtoms;
#endif
  if ( simParams->langevinOn /* && !simParams->langevin_useBAOAB */ )
  {
    // XXX missing Drude

    // Scale by TIMEFACTOR to convert to fs and then by 0.001 to ps
    // multiply by the Langevin damping coefficient, units 1/ps.
    // XXX we could instead store time-scaled Langevin parameters
    BigReal dt = timestep * (0.001 * TIMEFACTOR);
    // Buffer the Gaussian random numbers
    if ( simParams->langevinGammasDiffer ) {
      // Must re-satisfy constraints if Langevin gammas differ.
      // (conserve momentum?)
      TIMER_START(patch->timerSet, RATTLE1);
      rattle1_SOA(timestep, 1);
      TIMER_STOP(patch->timerSet, RATTLE1);
      //
      // We don't need random numbers for atoms such that gamma=0.
      // If gammas differ, the likely case is that we aren't applying
      // Langevin damping to hydrogen, making those langevinParam=0,
      // in which case we need only numAtoms/3 random vectors.
      //
      // XXX can refine code below, count in advance how many
      // random numbers we need to use Random array filling routine
      //
      // XXX Loop does not vectorize!
      for (int i=0;  i < numAtoms;  i++) {
        Vector rg;  // = 0
        if (langevinParam[i] != 0)  rg = random->gaussian_vector();
        gaussrand_x[i] = float(rg.x);
        gaussrand_y[i] = float(rg.y);
        gaussrand_z[i] = float(rg.z);
      }
    }
    else {
      // Need to completely fill random number arrays.
      random->gaussian_array_f(gaussrand_x, numAtoms);
      random->gaussian_array_f(gaussrand_y, numAtoms);
      random->gaussian_array_f(gaussrand_z, numAtoms);
    }

    // do the velocity updates
    for (int i=0;  i < numAtoms;  i++) {
      vel_x[i] += gaussrand_x[i] * langScalRandBBK2[i];
      vel_y[i] += gaussrand_y[i] * langScalRandBBK2[i];
      vel_z[i] += gaussrand_z[i] * langScalRandBBK2[i];
      vel_x[i] *= langScalVelBBK2[i];
      vel_y[i] *= langScalVelBBK2[i];
      vel_z[i] *= langScalVelBBK2[i];
    }
  } // end if langevinOn
}

void Sequencer::berendsenPressure_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
    const int    * __restrict hydrogenGroupSize,
    const float  * __restrict mass,
    double       * __restrict pos_x,
    double       * __restrict pos_y,
    double       * __restrict pos_z,
    int numAtoms,
#endif
    int step)
{
#ifdef SOA_SIMPLIFY_PARAMS
  const int    * __restrict hydrogenGroupSize = patch->patchDataSOA.hydrogenGroupSize;
  const float  * __restrict mass = patch->patchDataSOA.mass;
  double       * __restrict pos_x = patch->patchDataSOA.pos_x;
  double       * __restrict pos_y = patch->patchDataSOA.pos_y;
  double       * __restrict pos_z = patch->patchDataSOA.pos_z;
  int numAtoms = patch->patchDataSOA.numAtoms;
#endif

  //
  // Loops below simplify if we lift out special cases of fixed atoms
  // and pressure excluded atoms and make them their own branch.
  //

  ++berendsenPressure_count;
  if ( berendsenPressure_count == simParams->berendsenPressureFreq ) {
    berendsenPressure_count = 0;
    // Blocking receive for the updated lattice scaling factor.
    Tensor factor = broadcast->positionRescaleFactor.get(step);
    patch->lattice.rescale(factor);
    Vector origin = patch->lattice.origin();

    if ( simParams->useGroupPressure ) {
      int hgs;
      for (int i = 0;  i < numAtoms;  i += hgs) {
        int j;
        hgs = hydrogenGroupSize[i];
        // missing fixed atoms implementation
        BigReal m_cm = 0;
        BigReal r_cm_x = 0;
        BigReal r_cm_y = 0;
        BigReal r_cm_z = 0;
        // calculate the center of mass
        for ( j = i; j < (i+hgs); ++j ) {
          m_cm += mass[j];
          r_cm_x += mass[j] * pos_x[j];
          r_cm_y += mass[j] * pos_y[j];
          r_cm_z += mass[j] * pos_z[j];
        }
        BigReal inv_m_cm = namd_reciprocal(m_cm);
        r_cm_x *= inv_m_cm;
        r_cm_y *= inv_m_cm;
        r_cm_z *= inv_m_cm;
        // scale the center of mass with factor
        // shift to origin
        double tx = r_cm_x - origin.x;
        double ty = r_cm_y - origin.y;
        double tz = r_cm_z - origin.z;
        // apply transformation 
        double new_r_cm_x = factor.xx*tx + factor.xy*ty + factor.xz*tz;
        double new_r_cm_y = factor.yx*tx + factor.yy*ty + factor.yz*tz;
        double new_r_cm_z = factor.zx*tx + factor.zy*ty + factor.zz*tz;
        // shift back
        new_r_cm_x += origin.x;
        new_r_cm_y += origin.y;
        new_r_cm_z += origin.z;
        // translation vector from old COM and new COM
        double delta_r_cm_x = new_r_cm_x - r_cm_x;
        double delta_r_cm_y = new_r_cm_y - r_cm_y;
        double delta_r_cm_z = new_r_cm_z - r_cm_z;
        // shift the hydrogen group with translation vector
        for (j = i;  j < (i+hgs);  ++j) {
          pos_x[j] += delta_r_cm_x;
          pos_y[j] += delta_r_cm_y;
          pos_z[j] += delta_r_cm_z;
        }
      }
    } else {
      for (int i = 0;  i < numAtoms;  ++i) {
        // missing fixed atoms implementation
        // scale the coordinates with factor
        // shift to origin
        double tx = pos_x[i] - origin.x;
        double ty = pos_y[i] - origin.y;
        double tz = pos_z[i] - origin.z;
        // apply transformation
        double ftx = factor.xx*tx + factor.xy*ty + factor.xz*tz;
        double fty = factor.yx*tx + factor.yy*ty + factor.yz*tz;
        double ftz = factor.zx*tx + factor.zy*ty + factor.zz*tz;
        // shift back
        pos_x[i] = ftx + origin.x;
        pos_y[i] = fty + origin.y;
        pos_z[i] = ftz + origin.z;
      }
    }
  }
}

void Sequencer::langevinPiston_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
    const int    * __restrict hydrogenGroupSize,
    const float  * __restrict mass,
    double       * __restrict pos_x,
    double       * __restrict pos_y,
    double       * __restrict pos_z,
    double       * __restrict vel_x,
    double       * __restrict vel_y,
    double       * __restrict vel_z,
    int numAtoms,
#endif
    int step
    )
{
#ifdef SOA_SIMPLIFY_PARAMS
  const int    * __restrict hydrogenGroupSize = patch->patchDataSOA.hydrogenGroupSize;
  const float  * __restrict mass = patch->patchDataSOA.mass;
  double       * __restrict pos_x = patch->patchDataSOA.pos_x;
  double       * __restrict pos_y = patch->patchDataSOA.pos_y;
  double       * __restrict pos_z = patch->patchDataSOA.pos_z;
  double       * __restrict vel_x = patch->patchDataSOA.vel_x;
  double       * __restrict vel_y = patch->patchDataSOA.vel_y;
  double       * __restrict vel_z = patch->patchDataSOA.vel_z;
  int numAtoms = patch->patchDataSOA.numAtoms;
#endif

  //
  // Loops below simplify if we lift out special cases of fixed atoms
  // and pressure excluded atoms and make them their own branch.
  //

  // Blocking receive for the updated lattice scaling factor.

  Tensor factor = broadcast->positionRescaleFactor.get(step);

  TIMER_START(patch->timerSet, PISTON);
  // JCP FIX THIS!!!
  double velFactor_x = namd_reciprocal(factor.xx);
  double velFactor_y = namd_reciprocal(factor.yy);
  double velFactor_z = namd_reciprocal(factor.zz);
  patch->lattice.rescale(factor);
  Vector origin = patch->lattice.origin();
  if ( simParams->useGroupPressure ) {
    int hgs;
    for (int i=0;  i < numAtoms;  i += hgs) {
      int j;
      hgs = hydrogenGroupSize[i];
      // missing fixed atoms
      BigReal m_cm = 0;
      BigReal r_cm_x = 0;
      BigReal r_cm_y = 0;
      BigReal r_cm_z = 0;
      BigReal v_cm_x = 0;
      BigReal v_cm_y = 0;
      BigReal v_cm_z = 0;
      for ( j = i; j < (i+hgs); ++j ) {
        m_cm += mass[j];
        r_cm_x += mass[j] * pos_x[j];
        r_cm_y += mass[j] * pos_y[j];
        r_cm_z += mass[j] * pos_z[j];
        v_cm_x += mass[j] * vel_x[j];
        v_cm_y += mass[j] * vel_y[j];
        v_cm_z += mass[j] * vel_z[j];
      }
      BigReal inv_m_cm = namd_reciprocal(m_cm);
      r_cm_x *= inv_m_cm;
      r_cm_y *= inv_m_cm;
      r_cm_z *= inv_m_cm;

      double tx = r_cm_x - origin.x;
      double ty = r_cm_y - origin.y;
      double tz = r_cm_z - origin.z;
      double new_r_cm_x = factor.xx*tx + factor.xy*ty + factor.xz*tz;
      double new_r_cm_y = factor.yx*tx + factor.yy*ty + factor.yz*tz;
      double new_r_cm_z = factor.zx*tx + factor.zy*ty + factor.zz*tz;
      new_r_cm_x += origin.x;
      new_r_cm_y += origin.y;
      new_r_cm_z += origin.z;

      double delta_r_cm_x = new_r_cm_x - r_cm_x;
      double delta_r_cm_y = new_r_cm_y - r_cm_y;
      double delta_r_cm_z = new_r_cm_z - r_cm_z;
      v_cm_x *= inv_m_cm;
      v_cm_y *= inv_m_cm;
      v_cm_z *= inv_m_cm;
      double delta_v_cm_x = ( velFactor_x - 1 ) * v_cm_x;
      double delta_v_cm_y = ( velFactor_y - 1 ) * v_cm_y;
      double delta_v_cm_z = ( velFactor_z - 1 ) * v_cm_z;
      for (j = i;  j < (i+hgs);  j++) {
        pos_x[j] += delta_r_cm_x;
        pos_y[j] += delta_r_cm_y;
        pos_z[j] += delta_r_cm_z;
        vel_x[j] += delta_v_cm_x;
        vel_y[j] += delta_v_cm_y;
        vel_z[j] += delta_v_cm_z;
      }
      // if (i < 10)
      //   printf("cpu: %d, %f, %f, %f, %f, %f, %f\n", i,
      //          pos_x[i], pos_y[i], pos_z[i],
      //          vel_x[i], vel_y[i], vel_z[i]);
    }
  }
  else {
    for (int i=0;  i < numAtoms;  i++) {
      double tx = pos_x[i] - origin.x;
      double ty = pos_y[i] - origin.y;
      double tz = pos_z[i] - origin.z;
      double ftx = factor.xx*tx + factor.xy*ty + factor.xz*tz;
      double fty = factor.yx*tx + factor.yy*ty + factor.yz*tz;
      double ftz = factor.zx*tx + factor.zy*ty + factor.zz*tz;
      pos_x[i] = ftx + origin.x;
      pos_y[i] = fty + origin.y;
      pos_z[i] = ftz + origin.z;
      vel_x[i] *= velFactor_x;
      vel_y[i] *= velFactor_y;
      vel_z[i] *= velFactor_z;
      // if (i < 10)
      //   printf("cpu: %d, %f, %f, %f, %f, %f, %f\n", i,
      //          pos_x[i], pos_y[i], pos_z[i],
      //          vel_x[i], vel_y[i], vel_z[i]);
    }
  }
  TIMER_STOP(patch->timerSet, PISTON);
  // exit(0);
}


// timestep scaled by 1/TIMEFACTOR
void Sequencer::rattle1_SOA(BigReal timestep, int pressure)
{
  NAMD_EVENT_RANGE_2(patch->flags.event_on, NamdProfileEvent::RATTLE1_SOA);
  if ( simParams->rigidBonds != RIGID_NONE ) {
    Tensor virial;
    Tensor *vp = ( pressure ? &virial : 0 );
    // XXX pressureProfileReduction == NULL?
    if ( patch->rattle1_SOA(timestep, vp, pressureProfileReduction) ) {
      iout << iERROR <<
        "Constraint failure; simulation has become unstable.\n" << endi;
      Node::Object()->enableEarlyExit();
      terminate();
    }
#ifdef NODEGROUP_FORCE_REGISTER
    if(simParams->CUDASOAintegrate){
      ADD_TENSOR_OBJECT(CUDASequencer->patchData->reduction,REDUCTION_VIRIAL_NORMAL,virial);
    }
    else{
#endif
      ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,virial);
#ifdef NODEGROUP_FORCE_REGISTER
    }
#endif
  }
}

void Sequencer::runComputeObjects_SOA(int migration, int pairlists, int nstep)
{
  if ( migration ) pairlistsAreValid = 0;
#if (defined(NAMD_CUDA) || defined(NAMD_HIP)) || defined(NAMD_MIC)
  if ( pairlistsAreValid &&
       ( patch->flags.doFullElectrostatics || ! simParams->fullElectFrequency )
                         && ( pairlistsAge > pairlistsAgeLimit ) ) {
    pairlistsAreValid = 0;
  }
#else
  if ( pairlistsAreValid && ( pairlistsAge > pairlistsAgeLimit ) ) {
    pairlistsAreValid = 0;
  }
#endif
  if ( ! simParams->usePairlists ) pairlists = 0;
  patch->flags.usePairlists = pairlists || pairlistsAreValid;
  patch->flags.savePairlists = pairlists && ! pairlistsAreValid;

#if defined(NTESTPID)
  if (1 && patch->patchID == NTESTPID) {
    int step = patch->flags.step;
    int numAtoms = patch->numAtoms;
    double *xyzq = new double[4*numAtoms];
    double *x = patch->patchDataSOA.pos_x;
    double *y = patch->patchDataSOA.pos_y;
    double *z = patch->patchDataSOA.pos_z;
    float *q = patch->patchDataSOA.charge;
    for (int i=0;  i < numAtoms;  i++) {
      xyzq[4*i  ] = x[i];
      xyzq[4*i+1] = y[i];
      xyzq[4*i+2] = z[i];
      xyzq[4*i+3] = q[i];
    }
    char fname[128], remark[128];
    sprintf(fname, "xyzq_soa_pid%d_step%d.bin", NTESTPID, step);
    sprintf(remark, "SOA xyzq, patch %d, step %d", NTESTPID, step);
    TestArray_write<double>(fname, remark, xyzq, 4*numAtoms);
    delete[] xyzq;
  }
#endif
  // Zero all SOA global forces before computing force 
  patch->zero_global_forces_SOA();
  patch->positionsReady_SOA(migration);  // updates flags.sequence

  int seq = patch->flags.sequence;
  int basePriority = ( (seq & 0xffff) << 15 )
                     + PATCH_PRIORITY(patch->getPatchID());

  // XXX missing GBIS
  priority = basePriority + COMPUTE_HOME_PRIORITY;
  //char prbuf[32];
  //sprintf(prbuf, "%s: %d", NamdProfileEventStr[NamdProfileEvent::SEQ_SUSPEND], patch->getPatchID());
  //NAMD_EVENT_START_EX(1, NamdProfileEvent::SEQ_SUSPEND, prbuf);
  suspend(); // until all deposit boxes close
  //NAMD_EVENT_STOP(1, NamdProfileEvent::SEQ_SUSPEND);

#ifdef NODEGROUP_FORCE_REGISTER
  if(!simParams->CUDASOAintegrate || migration){
     patch->copy_forces_to_SOA();
  }
#else
  patch->copy_forces_to_SOA();
#endif

#if defined(NTESTPID)
  if (1 && patch->patchID == NTESTPID) {
    int step = patch->flags.step;
    int numAtoms = patch->numAtoms;
    char fname[128];
    char remark[128];
    double *fxyz = new double[3*numAtoms];
    double *fx = patch->patchDataSOA.f_normal_x;
    double *fy = patch->patchDataSOA.f_normal_y;
    double *fz = patch->patchDataSOA.f_normal_z;
    for (int i=0;  i < numAtoms;  i++) {
      fxyz[3*i  ] = fx[i];
      fxyz[3*i+1] = fy[i];
      fxyz[3*i+2] = fz[i];
    }
    sprintf(fname, "fxyz_normal_soa_pid%d_step%d.bin", NTESTPID, step);
    sprintf(remark, "SOA fxyz normal, patch %d, step %d", NTESTPID, step);
    TestArray_write<double>(fname, remark, fxyz, 3*numAtoms);
    fx = patch->patchDataSOA.f_nbond_x;
    fy = patch->patchDataSOA.f_nbond_y;
    fz = patch->patchDataSOA.f_nbond_z;
    for (int i=0;  i < numAtoms;  i++) {
      fxyz[3*i  ] = fx[i];
      fxyz[3*i+1] = fy[i];
      fxyz[3*i+2] = fz[i];
    }
    sprintf(fname, "fxyz_nbond_soa_pid%d_step%d.bin", NTESTPID, step);
    sprintf(remark, "SOA fxyz nonbonded, patch %d, step %d", NTESTPID, step);
    TestArray_write<double>(fname, remark, fxyz, 3*numAtoms);
    fx = patch->patchDataSOA.f_slow_x;
    fy = patch->patchDataSOA.f_slow_y;
    fz = patch->patchDataSOA.f_slow_z;
    for (int i=0;  i < numAtoms;  i++) {
      fxyz[3*i  ] = fx[i];
      fxyz[3*i+1] = fy[i];
      fxyz[3*i+2] = fz[i];
    }
    sprintf(fname, "fxyz_slow_soa_pid%d_step%d.bin", NTESTPID, step);
    sprintf(remark, "SOA fxyz slow, patch %d, step %d", NTESTPID, step);
    TestArray_write<double>(fname, remark, fxyz, 3*numAtoms);
    delete[] fxyz;
  }
#endif

#if 0
  if (1 && patch->patchID == 0) {
    int numAtoms = patch->numAtoms;
    double *fxyz = new double[3*numAtoms];
    double *fx, *fy, *fz;
    char fname[64], remark[128];
    int step = patch->flags.step;

    fx = patch->patchDataSOA.f_slow_x;
    fy = patch->patchDataSOA.f_slow_y;
    fz = patch->patchDataSOA.f_slow_z;
    for (int i=0;  i < numAtoms;  i++) {
      fxyz[3*i  ] = fx[i];
      fxyz[3*i+1] = fy[i];
      fxyz[3*i+2] = fz[i];
    }
    sprintf(fname, "fslow_soa_%d.bin", step);
    sprintf(remark, "SOA slow forces, step %d\n", step);
    TestArray_write<double>(fname, remark, fxyz, 3*numAtoms);

    fx = patch->patchDataSOA.f_nbond_x;
    fy = patch->patchDataSOA.f_nbond_y;
    fz = patch->patchDataSOA.f_nbond_z;
    for (int i=0;  i < numAtoms;  i++) {
      fxyz[3*i  ] = fx[i];
      fxyz[3*i+1] = fy[i];
      fxyz[3*i+2] = fz[i];
    }
    sprintf(fname, "fnbond_soa_%d.bin", step);
    sprintf(remark, "SOA nonbonded forces, step %d\n", step);
    TestArray_write<double>(fname, remark, fxyz, 3*numAtoms);

    fx = patch->patchDataSOA.f_normal_x;
    fy = patch->patchDataSOA.f_normal_y;
    fz = patch->patchDataSOA.f_normal_z;
    for (int i=0;  i < numAtoms;  i++) {
      fxyz[3*i  ] = fx[i];
      fxyz[3*i+1] = fy[i];
      fxyz[3*i+2] = fz[i];
    }
    sprintf(fname, "fnormal_soa_%d.bin", step);
    sprintf(remark, "SOA normal forces, step %d\n", step);
    TestArray_write<double>(fname, remark, fxyz, 3*numAtoms);

    delete[] fxyz;
  }
#endif

#if 0
     //Will print forces here after runComputeObjects
     if(nstep == 1){
        fprintf(stderr, "CPU force arrays for alanin\n" );
        for(int i = 0; i < patch->patchDataSOA.numAtoms; i++){
          fprintf(stderr, "f[%i] = %lf %lf %lf | %lf %lf %lf | %lf %lf %lf\n", i,
             patch->patchDataSOA.f_normal_x[i], patch->patchDataSOA.f_normal_y[i], patch->patchDataSOA.f_normal_z[i],
             patch->patchDataSOA.f_nbond_x[i],  patch->patchDataSOA.f_nbond_y[i],  patch->patchDataSOA.f_nbond_z[i],
             patch->patchDataSOA.f_slow_x[i],   patch->patchDataSOA.f_slow_y[i],   patch->patchDataSOA.f_slow_z[i]);
        }
     }
#endif

  if ( patch->flags.savePairlists && patch->flags.doNonbonded ) {
    pairlistsAreValid = 1;
    pairlistsAge = 0;
  }
  // For multigrator, do not age pairlist during pressure step
  // NOTE: for non-multigrator pressureStep = 0 always
  if ( pairlistsAreValid /* && !pressureStep */ ) ++pairlistsAge;

  // XXX missing lonepairs
  // XXX missing Molly
  // XXX missing Lowe-Andersen
}

/** Rescale velocities with the scale factor sent from the Controller.
 *
 *  \param step The current timestep
 */
void Sequencer::stochRescaleVelocities_SOA(int step)
{
  ++stochRescale_count;
  if ( stochRescale_count == simParams->stochRescaleFreq ) {
    double * __restrict vel_x = patch->patchDataSOA.vel_x;
    double * __restrict vel_y = patch->patchDataSOA.vel_y;
    double * __restrict vel_z = patch->patchDataSOA.vel_z;
    int numAtoms = patch->patchDataSOA.numAtoms;
    // Blocking receive for the temperature coupling coefficient.
    BigReal velrescaling = broadcast->stochRescaleCoefficient.get(step);
    DebugM(4, "stochastically rescaling velocities at step " << step << " by " << velrescaling << "\n");
    for ( int i = 0; i < numAtoms; ++i ) {
      vel_x[i] *= velrescaling;
      vel_y[i] *= velrescaling;
      vel_z[i] *= velrescaling;
    }
    stochRescale_count = 0;
  }
}

//
// end SOA code
//
//////////////////////////////////////////////////////////////////////////

#endif // SEQUENCER_SOA


extern int eventEndOfTimeStep;

void Sequencer::integrate(int scriptTask) {
    char traceNote[24];
    char tracePrefix[20];
    sprintf(tracePrefix, "p:%d,s:",patch->patchID);
//    patch->write_tip4_props();

    //
    // DJH: Copy all data into SOA (structure of arrays)
    // from AOS (array of structures) data structure.
    //
    //patch->copy_all_to_SOA();

#ifdef TIMER_COLLECTION
    TimerSet& t = patch->timerSet;
#endif
    TIMER_INIT_WIDTH(t, KICK, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, MAXMOVE, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, DRIFT, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, PISTON, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, SUBMITHALF, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, VELBBK1, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, VELBBK2, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, RATTLE1, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, SUBMITFULL, simParams->timerBinWidth);
    TIMER_INIT_WIDTH(t, SUBMITCOLLECT, simParams->timerBinWidth);

    int &step = patch->flags.step;
    step = simParams->firstTimestep;

    // drag switches
    const Bool rotDragOn = simParams->rotDragOn;
    const Bool movDragOn = simParams->movDragOn;

    const int commOnly = simParams->commOnly;

    int &maxForceUsed = patch->flags.maxForceUsed;
    int &maxForceMerged = patch->flags.maxForceMerged;
    maxForceUsed = Results::normal;
    maxForceMerged = Results::normal;

    const int numberOfSteps = simParams->N;
    const int stepsPerCycle = simParams->stepsPerCycle;
    const BigReal timestep = simParams->dt;

    // what MTS method?
    const int staleForces = ( simParams->MTSAlgorithm == NAIVE );

    const int nonbondedFrequency = simParams->nonbondedFrequency;
    slowFreq = nonbondedFrequency;
    const BigReal nbondstep = timestep * (staleForces?1:nonbondedFrequency);
    int &doNonbonded = patch->flags.doNonbonded;
    doNonbonded = (step >= numberOfSteps) || !(step%nonbondedFrequency);
    if ( nonbondedFrequency == 1 ) maxForceMerged = Results::nbond;
    if ( doNonbonded ) maxForceUsed = Results::nbond;

    // Do we do full electrostatics?
    const int dofull = ( simParams->fullElectFrequency ? 1 : 0 );
    const int fullElectFrequency = simParams->fullElectFrequency;
    if ( dofull ) slowFreq = fullElectFrequency;
    const BigReal slowstep = timestep * (staleForces?1:fullElectFrequency);
    int &doFullElectrostatics = patch->flags.doFullElectrostatics;
    doFullElectrostatics = (dofull && ((step >= numberOfSteps) || !(step%fullElectFrequency)));
    if ( dofull && (fullElectFrequency == 1) && !(simParams->mollyOn) )
					maxForceMerged = Results::slow;
    if ( doFullElectrostatics ) maxForceUsed = Results::slow;

//#ifndef UPPER_BOUND
    const Bool accelMDOn = simParams->accelMDOn;
    const Bool accelMDdihe = simParams->accelMDdihe;
    const Bool accelMDdual = simParams->accelMDdual;
    if ( accelMDOn && (accelMDdihe || accelMDdual)) maxForceUsed = Results::amdf;

    // Is adaptive tempering on?
    const Bool adaptTempOn = simParams->adaptTempOn;
    adaptTempT = simParams->initialTemp;
    if (simParams->langevinOn)
        adaptTempT = simParams->langevinTemp;
    else if (simParams->rescaleFreq > 0)
        adaptTempT = simParams->rescaleTemp;


    int &doMolly = patch->flags.doMolly;
    doMolly = simParams->mollyOn && doFullElectrostatics;
    // BEGIN LA
    int &doLoweAndersen = patch->flags.doLoweAndersen;
    doLoweAndersen = simParams->loweAndersenOn && doNonbonded;
    // END LA

    int &doGBIS = patch->flags.doGBIS;
    doGBIS = simParams->GBISOn;

    int &doLCPO = patch->flags.doLCPO;
    doLCPO = simParams->LCPOOn;

    int zeroMomentum = simParams->zeroMomentum;

    // Do we need to return forces to TCL script or Colvar module?
    int doTcl = simParams->tclForcesOn;
	int doColvars = simParams->colvarsOn;
//#endif
    ComputeGlobal *computeGlobal = Node::Object()->computeMgr->computeGlobalObject;

    // Bother to calculate energies?
    int &doEnergy = patch->flags.doEnergy;
    int energyFrequency = simParams->computeEnergies;
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
    if(simParams->alchOn) energyFrequency = NAMD_gcd(energyFrequency, simParams->alchOutFreq);
#endif
#ifndef UPPER_BOUND
    const int reassignFreq = simParams->reassignFreq;
#endif

    int &doVirial = patch->flags.doVirial;
    doVirial = 1;

  if ( scriptTask == SCRIPT_RUN ) {

//    print_vel_AOS(patch->atom.begin(), 0, patch->numAtoms);

#ifndef UPPER_BOUND
//    printf("Doing initial rattle\n");
#ifndef UPPER_BOUND
D_MSG("rattle1()");
    TIMER_START(t, RATTLE1);
    rattle1(0.,0);  // enforce rigid bond constraints on initial positions
    TIMER_STOP(t, RATTLE1);
#endif

    if (simParams->lonepairs || simParams->singleTopology) {
      patch->atomMapper->registerIDsFullAtom(
		patch->atom.begin(),patch->atom.end());
    }

    if ( !commOnly && ( reassignFreq>0 ) && ! (step%reassignFreq) ) {
       reassignVelocities(timestep,step);
    }
#endif

    doEnergy = ! ( step % energyFrequency );
#ifndef UPPER_BOUND
    if ( accelMDOn && !accelMDdihe ) doEnergy=1;
    //Update energy every timestep for adaptive tempering
    if ( adaptTempOn ) doEnergy=1;
#endif
//    print_vel_AOS(patch->atom.begin(), 0, patch->numAtoms);
D_MSG("runComputeObjects()");
    runComputeObjects(1,step<numberOfSteps); // must migrate here!
#ifndef UPPER_BOUND
    rescaleaccelMD(step, doNonbonded, doFullElectrostatics); // for accelMD
    adaptTempUpdate(step); // update adaptive tempering temperature
#endif

#ifndef UPPER_BOUND
    if ( staleForces || doTcl || doColvars ) {
      if ( doNonbonded ) saveForce(Results::nbond);
      if ( doFullElectrostatics ) saveForce(Results::slow);
    }
//    print_vel_AOS(patch->atom.begin(), 0, patch->numAtoms);
    if ( ! commOnly ) {
D_MSG("newtonianVelocities()");
      TIMER_START(t, KICK);
      newtonianVelocities(-0.5,timestep,nbondstep,slowstep,0,1,1);
      TIMER_STOP(t, KICK);
    }
    minimizationQuenchVelocity();
//    print_vel_AOS(patch->atom.begin(), 0, patch->numAtoms);
#ifndef UPPER_BOUND
D_MSG("rattle1()");
    TIMER_START(t, RATTLE1);
    rattle1(-timestep,0);
    TIMER_STOP(t, RATTLE1);
#endif
//    print_vel_AOS(patch->atom.begin(), 0, patch->numAtoms);
D_MSG("submitHalfstep()");
    TIMER_START(t, SUBMITHALF);
    submitHalfstep(step);
    TIMER_STOP(t, SUBMITHALF);
//    print_vel_AOS(patch->atom.begin(), 0, patch->numAtoms);
    if ( ! commOnly ) {
D_MSG("newtonianVelocities()");
      TIMER_START(t, KICK);
      newtonianVelocities(1.0,timestep,nbondstep,slowstep,0,1,1);
      TIMER_STOP(t, KICK);
    }
//    print_vel_AOS(patch->atom.begin(), 0, patch->numAtoms);
D_MSG("rattle1()");
    TIMER_START(t, RATTLE1);
    rattle1(timestep,1);
    TIMER_STOP(t, RATTLE1);
    if (doTcl || doColvars)  // include constraint forces
      computeGlobal->saveTotalForces(patch);
//    print_vel_AOS(patch->atom.begin(), 0, patch->numAtoms);
D_MSG("submitHalfstep()");
    TIMER_START(t, SUBMITHALF);
    submitHalfstep(step);
    TIMER_STOP(t, SUBMITHALF);
    if ( zeroMomentum && doFullElectrostatics ) submitMomentum(step);
    if ( ! commOnly ) {
D_MSG("newtonianVelocities()");
      TIMER_START(t, KICK);
      newtonianVelocities(-0.5,timestep,nbondstep,slowstep,0,1,1);
      TIMER_STOP(t, KICK);
    }
//    print_vel_AOS(patch->atom.begin(), 0, patch->numAtoms);
#endif
D_MSG("submitReductions()");
    TIMER_START(t, SUBMITFULL);
    submitReductions(step);
    TIMER_STOP(t, SUBMITFULL);
//    print_vel_AOS(patch->atom.begin(), 0, patch->numAtoms);
#ifndef UPPER_BOUND
    if(0){ // if(traceIsOn()){
        traceUserEvent(eventEndOfTimeStep);
        sprintf(traceNote, "%s%d",tracePrefix,step);
        traceUserSuppliedNote(traceNote);
    }
#endif
    rebalanceLoad(step);

  } // scriptTask == SCRIPT_RUN

#ifndef UPPER_BOUND
  bool doMultigratorRattle = false;
#endif

  //
  // DJH: There are a lot of mod operations below and elsewhere to
  // test step number against the frequency of something happening.
  // Mod and integer division are expensive!
  // Might be better to replace with counters and test equality.
  //
#if 0
    for(int i = 0; i < NamdProfileEvent::EventsCount; i++)
	CkPrintf("-------------- [%d] %s -------------\n", i, NamdProfileEventStr[i]);
#endif

#if defined(NAMD_NVTX_ENABLED) || defined(NAMD_CMK_TRACE_ENABLED) || defined(NAMD_ROCTX_ENABLED)
  int& eon = patch->flags.event_on;
  int epid = (simParams->beginEventPatchID <= patch->getPatchID()
      && patch->getPatchID() <= simParams->endEventPatchID);
  int beginStep = simParams->beginEventStep;
  int endStep = simParams->endEventStep;
  bool controlProfiling = patch->getPatchID() == 0;
#endif

    for ( ++step; step <= numberOfSteps; ++step )
    {
#if defined(NAMD_NVTX_ENABLED) || defined(NAMD_CMK_TRACE_ENABLED)  || defined(NAMD_ROCTX_ENABLED)
      eon = epid && (beginStep < step && step <= endStep);

      if (controlProfiling && step == beginStep) {
        NAMD_PROFILE_START();
      }
      if (controlProfiling && step == endStep) {
        NAMD_PROFILE_STOP();
      }
      char buf[32];
      sprintf(buf, "%s: %d", NamdProfileEventStr[NamdProfileEvent::INTEGRATE_1], patch->getPatchID());
      NAMD_EVENT_START_EX(eon, NamdProfileEvent::INTEGRATE_1, buf);
#endif
#ifndef UPPER_BOUND

      rescaleVelocities(step);
      tcoupleVelocities(timestep,step);
      if ( simParams->stochRescaleOn ) {
        stochRescaleVelocities(step);
      }
      berendsenPressure(step);

      if ( ! commOnly ) {
        TIMER_START(t, KICK);
        newtonianVelocities(0.5,timestep,nbondstep,slowstep,staleForces,doNonbonded,doFullElectrostatics);
        TIMER_STOP(t, KICK);
      }

      // We do RATTLE here if multigrator thermostat was applied in the previous step
      if (doMultigratorRattle) rattle1(timestep, 1);

      /* reassignment based on half-step velocities
         if ( !commOnly && ( reassignFreq>0 ) && ! (step%reassignFreq) ) {
         addVelocityToPosition(0.5*timestep);
         reassignVelocities(timestep,step);
         addVelocityToPosition(0.5*timestep);
         rattle1(0.,0);
         rattle1(-timestep,0);
         addVelocityToPosition(-1.0*timestep);
         rattle1(timestep,0);
         } */

      TIMER_START(t, MAXMOVE);
      maximumMove(timestep);
      TIMER_STOP(t, MAXMOVE);

      NAMD_EVENT_STOP(eon, NamdProfileEvent::INTEGRATE_1);  // integrate 1

      if ( simParams->langevinPistonOn || (simParams->langevinOn && simParams->langevin_useBAOAB) ) {
        if ( ! commOnly ) {
          TIMER_START(t, DRIFT);
          addVelocityToPosition(0.5*timestep);
          TIMER_STOP(t, DRIFT);
        }
        // We add an Ornstein-Uhlenbeck integration step for the case of BAOAB (Langevin)
        langevinVelocities(timestep);

        // There is a blocking receive inside of langevinPiston()
        // that might suspend the current thread of execution,
        // so split profiling around this conditional block.
        langevinPiston(step);

        if ( ! commOnly ) {
          TIMER_START(t, DRIFT);
          addVelocityToPosition(0.5*timestep);
          TIMER_STOP(t, DRIFT);
        }
      } else {
        // If Langevin is not used, take full time step directly instread of two half steps
        if ( ! commOnly ) {
          TIMER_START(t, DRIFT);
          addVelocityToPosition(timestep);
          TIMER_STOP(t, DRIFT);
        }
      }

      NAMD_EVENT_START(eon, NamdProfileEvent::INTEGRATE_2);

      // impose hard wall potential for Drude bond length
      hardWallDrude(timestep, 1);

      minimizationQuenchVelocity();
#endif // UPPER_BOUND

      doNonbonded = !(step%nonbondedFrequency);
      doFullElectrostatics = (dofull && !(step%fullElectFrequency));

#ifndef UPPER_BOUND
      if ( zeroMomentum && doFullElectrostatics ) {
        // There is a blocking receive inside of correctMomentum().
        correctMomentum(step,slowstep);
      }

      // There are NO sends in submitHalfstep() just local summation
      // into the Reduction struct.
      TIMER_START(t, SUBMITHALF);
      submitHalfstep(step);
      TIMER_STOP(t, SUBMITHALF);

      doMolly = simParams->mollyOn && doFullElectrostatics;
      // BEGIN LA
      doLoweAndersen = simParams->loweAndersenOn && doNonbonded;
      // END LA

      maxForceUsed = Results::normal;
      if ( doNonbonded ) maxForceUsed = Results::nbond;
      if ( doFullElectrostatics ) maxForceUsed = Results::slow;
      if ( accelMDOn && (accelMDdihe || accelMDdual))  maxForceUsed = Results::amdf;

      // Migrate Atoms on stepsPerCycle
      doEnergy = ! ( step % energyFrequency );
      if ( accelMDOn && !accelMDdihe ) doEnergy=1;
      if ( adaptTempOn ) doEnergy=1;

      // Multigrator
      if (simParams->multigratorOn) {
        doVirial = (!(step % energyFrequency) || ((simParams->outputPressure > 0) && !(step % simParams->outputPressure))
          || !(step % simParams->multigratorPressureFreq));
        doKineticEnergy = (!(step % energyFrequency) || !(step % simParams->multigratorTemperatureFreq));
        doMomenta = (simParams->outputMomenta > 0) && !(step % simParams->outputMomenta);
      } else {
        doVirial = 1;
        doKineticEnergy = 1;
        doMomenta = 1;
      }
#endif
      NAMD_EVENT_STOP(eon, NamdProfileEvent::INTEGRATE_2);  // integrate 2

      // The current thread of execution will suspend in runComputeObjects().
      runComputeObjects(!(step%stepsPerCycle),step<numberOfSteps);

      NAMD_EVENT_START(eon, NamdProfileEvent::INTEGRATE_3);

#ifndef UPPER_BOUND
      rescaleaccelMD(step, doNonbonded, doFullElectrostatics); // for accelMD

      if ( staleForces || doTcl || doColvars ) {
        if ( doNonbonded ) saveForce(Results::nbond);
        if ( doFullElectrostatics ) saveForce(Results::slow);
      }

      // reassignment based on full-step velocities
      if ( !commOnly && ( reassignFreq>0 ) && ! (step%reassignFreq) ) {
        reassignVelocities(timestep,step);
        newtonianVelocities(-0.5,timestep,nbondstep,slowstep,staleForces,doNonbonded,doFullElectrostatics);
        rattle1(-timestep,0);
      }

      if ( ! commOnly ) {
        TIMER_START(t, VELBBK1);
        langevinVelocitiesBBK1(timestep);
        TIMER_STOP(t, VELBBK1);
        TIMER_START(t, KICK);
        newtonianVelocities(1.0,timestep,nbondstep,slowstep,staleForces,doNonbonded,doFullElectrostatics);
        TIMER_STOP(t, KICK);
        TIMER_START(t, VELBBK2);
        langevinVelocitiesBBK2(timestep);
        TIMER_STOP(t, VELBBK2);
      }

      // add drag to each atom's positions
      if ( ! commOnly && movDragOn ) addMovDragToPosition(timestep);
      if ( ! commOnly && rotDragOn ) addRotDragToPosition(timestep);

      TIMER_START(t, RATTLE1);
      rattle1(timestep,1);
      TIMER_STOP(t, RATTLE1);
      if (doTcl || doColvars)  // include constraint forces
        computeGlobal->saveTotalForces(patch);

      TIMER_START(t, SUBMITHALF);
      submitHalfstep(step);
      TIMER_STOP(t, SUBMITHALF);
      if ( zeroMomentum && doFullElectrostatics ) submitMomentum(step);

      if ( ! commOnly ) {
        TIMER_START(t, KICK);
        newtonianVelocities(-0.5,timestep,nbondstep,slowstep,staleForces,doNonbonded,doFullElectrostatics);
        TIMER_STOP(t, KICK);
      }

	// rattle2(timestep,step);
#endif

        TIMER_START(t, SUBMITFULL);
	submitReductions(step);
        TIMER_STOP(t, SUBMITFULL);
        TIMER_START(t, SUBMITCOLLECT);
	submitCollections(step);
        TIMER_STOP(t, SUBMITCOLLECT);
#ifndef UPPER_BOUND
       //Update adaptive tempering temperature
        adaptTempUpdate(step);

      // Multigrator temperature and pressure steps
      multigratorTemperature(step, 1);
      multigratorPressure(step, 1);
      multigratorPressure(step, 2);
      multigratorTemperature(step, 2);
      doMultigratorRattle = (simParams->multigratorOn && !(step % simParams->multigratorTemperatureFreq));

      NAMD_EVENT_STOP(eon, NamdProfileEvent::INTEGRATE_3); // integrate 3
#endif

#if CYCLE_BARRIER
        cycleBarrier(!((step+1) % stepsPerCycle), step);
#elif PME_BARRIER
        cycleBarrier(doFullElectrostatics, step);
#elif  STEP_BARRIER
        cycleBarrier(1, step);
#endif

#ifndef UPPER_BOUND
	 if(Node::Object()->specialTracing || simParams->statsOn){
		 int bstep = simParams->traceStartStep;
		 int estep = bstep + simParams->numTraceSteps;
		 if(step == bstep || step == estep){
			 traceBarrier(step);
		 }
	 }

#ifdef MEASURE_NAMD_WITH_PAPI
	 if(simParams->papiMeasure) {
		 int bstep = simParams->papiMeasureStartStep;
		 int estep = bstep + simParams->numPapiMeasureSteps;
		 if(step == bstep || step==estep) {
			 papiMeasureBarrier(step);
		 }
	 }
#endif

        if(0){ // if(traceIsOn()){
            traceUserEvent(eventEndOfTimeStep);
            sprintf(traceNote, "%s%d",tracePrefix,step);
            traceUserSuppliedNote(traceNote);
        }
#endif // UPPER_BOUND
	rebalanceLoad(step);

#if PME_BARRIER
	// a step before PME
        cycleBarrier(dofull && !((step+1)%fullElectFrequency),step);
#endif

#if USE_HPM
        if(step == START_HPM_STEP)
          (CProxy_Node(CkpvAccess(BOCclass_group).node)).startHPM();

        if(step == STOP_HPM_STEP)
          (CProxy_Node(CkpvAccess(BOCclass_group).node)).stopHPM();
#endif

    }

  TIMER_DONE(t);
#ifdef TIMER_COLLECTION
  if (patch->patchID == SPECIAL_PATCH_ID) {
    printf("Timer collection reporting in microseconds for "
        "Patch %d\n", patch->patchID);
    TIMER_REPORT(t);
  }
#endif // TIMER_COLLECTION
    //
    // DJH: Copy updates of SOA back into AOS.
    //
    //patch->copy_updates_to_AOS();
}

// add moving drag to each atom's position
void Sequencer::addMovDragToPosition(BigReal timestep) {
  FullAtom *atom = patch->atom.begin();
  int numAtoms = patch->numAtoms;
  Molecule *molecule = Node::Object()->molecule;   // need its methods
  const BigReal movDragGlobVel = simParams->movDragGlobVel;
  const BigReal dt = timestep / TIMEFACTOR;   // MUST be as in the integrator!
  Vector movDragVel, dragIncrement;
  for ( int i = 0; i < numAtoms; ++i )
  {
    // skip if fixed atom or zero drag attribute
    if ( (simParams->fixedAtomsOn && atom[i].atomFixed)
	 || !(molecule->is_atom_movdragged(atom[i].id)) ) continue;
    molecule->get_movdrag_params(movDragVel, atom[i].id);
    dragIncrement = movDragGlobVel * movDragVel * dt;
    atom[i].position += dragIncrement;
  }
}

// add rotating drag to each atom's position
void Sequencer::addRotDragToPosition(BigReal timestep) {
  FullAtom *atom = patch->atom.begin();
  int numAtoms = patch->numAtoms;
  Molecule *molecule = Node::Object()->molecule;   // need its methods
  const BigReal rotDragGlobVel = simParams->rotDragGlobVel;
  const BigReal dt = timestep / TIMEFACTOR;   // MUST be as in the integrator!
  BigReal rotDragVel, dAngle;
  Vector atomRadius;
  Vector rotDragAxis, rotDragPivot, dragIncrement;
  for ( int i = 0; i < numAtoms; ++i )
  {
    // skip if fixed atom or zero drag attribute
    if ( (simParams->fixedAtomsOn && atom[i].atomFixed)
	 || !(molecule->is_atom_rotdragged(atom[i].id)) ) continue;
    molecule->get_rotdrag_params(rotDragVel, rotDragAxis, rotDragPivot, atom[i].id);
    dAngle = rotDragGlobVel * rotDragVel * dt;
    rotDragAxis /= rotDragAxis.length();
    atomRadius = atom[i].position - rotDragPivot;
    dragIncrement = cross(rotDragAxis, atomRadius) * dAngle;
    atom[i].position += dragIncrement;
  }
}

void Sequencer::minimize() {
    //
    // DJH: Copy all data into SOA (structure of arrays)
    // from AOS (array of structures) data structure.
    //
    //patch->copy_all_to_SOA();

  const int numberOfSteps = simParams->N;
  const int stepsPerCycle = simParams->stepsPerCycle;
#if 0 && defined(NODEGROUP_FORCE_REGISTER)
  // XXX DJH: This is a hack that is found to get GPU nonbonded 
  //   force calculation right for --with-single-node-cuda builds
  const int stepsPerCycle_save = stepsPerCycle;
  simParams->stepsPerCycle = 1;
#endif
  int &step = patch->flags.step;
  step = simParams->firstTimestep;

  int &maxForceUsed = patch->flags.maxForceUsed;
  int &maxForceMerged = patch->flags.maxForceMerged;
  maxForceUsed = Results::normal;
  maxForceMerged = Results::normal;
  int &doNonbonded = patch->flags.doNonbonded;
  doNonbonded = 1;
  maxForceUsed = Results::nbond;
  maxForceMerged = Results::nbond;
  const int dofull = ( simParams->fullElectFrequency ? 1 : 0 );
  int &doFullElectrostatics = patch->flags.doFullElectrostatics;
  doFullElectrostatics = dofull;
  if ( dofull ) {
    maxForceMerged = Results::slow;
    maxForceUsed = Results::slow;
  }
  int &doMolly = patch->flags.doMolly;
  doMolly = simParams->mollyOn && doFullElectrostatics;
  int &doMinimize = patch->flags.doMinimize;
  doMinimize = 1;
  // BEGIN LA
  int &doLoweAndersen = patch->flags.doLoweAndersen;
  doLoweAndersen = 0;
  // END LA

  int &doGBIS = patch->flags.doGBIS;
  doGBIS = simParams->GBISOn;

    int &doLCPO = patch->flags.doLCPO;
    doLCPO = simParams->LCPOOn;

    int doTcl = simParams->tclForcesOn;
	int doColvars = simParams->colvarsOn;
    ComputeGlobal *computeGlobal = Node::Object()->computeMgr->computeGlobalObject;

  int &doEnergy = patch->flags.doEnergy;
  doEnergy = 1;

  // Do this to stabilize the minimizer, whether or not the user
  // wants rigid bond constraints enabled for dynamics.
  // In order to enforce, we have to call HomePatch::rattle1() directly.
  patch->rattle1(0.,0,0);  // enforce rigid bond constraints on initial positions

  if (simParams->lonepairs || simParams->singleTopology) {
    patch->atomMapper->registerIDsFullAtom(
		patch->atom.begin(),patch->atom.end());
  }

  runComputeObjects(1,step<numberOfSteps); // must migrate here!

  if ( doTcl || doColvars ) {
#ifdef DEBUG_MINIMIZE
    printf("doTcl = %d   doColvars = %d\n", doTcl, doColvars);
#endif
    if ( doNonbonded ) saveForce(Results::nbond);
    if ( doFullElectrostatics ) saveForce(Results::slow);
    computeGlobal->saveTotalForces(patch);
  }
#ifdef DEBUG_MINIMIZE
  else { printf("No computeGlobal\n"); }
#endif

  BigReal fmax2 = TIMEFACTOR * TIMEFACTOR * TIMEFACTOR * TIMEFACTOR;

  submitMinimizeReductions(step,fmax2);
  rebalanceLoad(step);

  int downhill = 1;  // start out just fixing bad contacts
  int minSeq = 0;
  for ( ++step; step <= numberOfSteps; ++step ) {

   // Blocking receive for the minimization coefficient.
   BigReal c = broadcast->minimizeCoefficient.get(minSeq++);

   if ( downhill ) {
    if ( c ) minimizeMoveDownhill(fmax2);
    else {
      downhill = 0;
      fmax2 *= 10000.;
    }
   }
   if ( ! downhill ) {
    if ( ! c ) {  // new direction

      // Blocking receive for the minimization coefficient.
      c = broadcast->minimizeCoefficient.get(minSeq++);

      newMinimizeDirection(c);  // v = c * v + f

      // Blocking receive for the minimization coefficient.
      c = broadcast->minimizeCoefficient.get(minSeq++);

    }  // same direction
    newMinimizePosition(c);  // x = x + c * v
   }

    runComputeObjects(!(step%stepsPerCycle),step<numberOfSteps);
    if ( doTcl || doColvars ) {
      if ( doNonbonded ) saveForce(Results::nbond);
      if ( doFullElectrostatics ) saveForce(Results::slow);
      computeGlobal->saveTotalForces(patch);
    }
    submitMinimizeReductions(step,fmax2);
    submitCollections(step, 1);  // write out zeros for velocities
    rebalanceLoad(step);
  }
  quenchVelocities();  // zero out bogus velocity

  doMinimize = 0;

#if 0
  // when using CUDASOAintegrate, need to update SOA data structures
  if (simParams->CUDASOAintegrateMode && !simParams->useDeviceMigration) {
    patch->copy_atoms_to_SOA();
  }
#endif

#if 0 && defined(NODEGROUP_FORCE_REGISTER)
  // XXX DJH: all patches in a PE are writing into simParams
  //   so this hack needs a guard
  simParams->stepsPerCycle = stepsPerCycle_save;
#endif
    //
    // DJH: Copy updates of SOA back into AOS.
    //
    //patch->copy_updates_to_AOS();
}

// x = x + 0.1 * unit(f) for large f
void Sequencer::minimizeMoveDownhill(BigReal fmax2) {

  FullAtom *a = patch->atom.begin();
  Force *f1 = patch->f[Results::normal].begin();  // includes nbond and slow
  int numAtoms = patch->numAtoms;

  for ( int i = 0; i < numAtoms; ++i ) {
    if ( simParams->fixedAtomsOn && a[i].atomFixed ) continue;
    Force f = f1[i];
    if ( f.length2() > fmax2 ) {
      a[i].position += ( 0.1 * f.unit() );
      int hgs = a[i].hydrogenGroupSize;  // 0 if not parent
      for ( int j=1; j<hgs; ++j ) {
        a[++i].position += ( 0.1 * f.unit() );
      }
    }
  }

  patch->rattle1(0.,0,0);
}

// v = c * v + f
void Sequencer::newMinimizeDirection(BigReal c) {
  FullAtom *a = patch->atom.begin();
  Force *f1 = patch->f[Results::normal].begin(); // includes nbond and slow
  const bool fixedAtomsOn = simParams->fixedAtomsOn;
  const bool drudeHardWallOn = simParams->drudeHardWallOn;
  int numAtoms = patch->numAtoms;
  BigReal maxv2 = 0.;

  for ( int i = 0; i < numAtoms; ++i ) {
    a[i].velocity *= c;
    a[i].velocity += f1[i];
    if ( drudeHardWallOn && i && (0.05 < a[i].mass) && ((a[i].mass < 1.0)) ) { // drude particle
      a[i].velocity = a[i-1].velocity;
    }
    if ( fixedAtomsOn && a[i].atomFixed ) a[i].velocity = 0;
    BigReal v2 = a[i].velocity.length2();
    if ( v2 > maxv2 ) maxv2 = v2;
  }

  { Tensor virial; patch->minimize_rattle2( 0.1 * TIMEFACTOR / sqrt(maxv2), &virial); }

  maxv2 = 0.;
  for ( int i = 0; i < numAtoms; ++i ) {
    if ( drudeHardWallOn && i && (0.05 < a[i].mass) && ((a[i].mass < 1.0)) ) { // drude particle
      a[i].velocity = a[i-1].velocity;
    }
    if ( fixedAtomsOn && a[i].atomFixed ) a[i].velocity = 0;
    BigReal v2 = a[i].velocity.length2();
    if ( v2 > maxv2 ) maxv2 = v2;
  }

  min_reduction->max(0,maxv2);
  min_reduction->submit();

  // prevent hydrogens from being left behind
  BigReal fmax2 = 0.01 * TIMEFACTOR * TIMEFACTOR * TIMEFACTOR * TIMEFACTOR;
  // int adjustCount = 0;
  int hgs;
  for ( int i = 0; i < numAtoms; i += hgs ) {
    hgs = a[i].hydrogenGroupSize;
    BigReal minChildVel = a[i].velocity.length2();
    if ( minChildVel < fmax2 ) continue;
    int adjustChildren = 1;
    for ( int j = i+1; j < (i+hgs); ++j ) {
      if ( a[j].velocity.length2() > minChildVel ) adjustChildren = 0;
    }
    if ( adjustChildren ) {
      // if ( hgs > 1 ) ++adjustCount;
      for ( int j = i+1; j < (i+hgs); ++j ) {
        if (a[i].mass < 0.01) continue;  // lone pair
        a[j].velocity = a[i].velocity;
      }
    }
  }
  // if (adjustCount) CkPrintf("Adjusting %d hydrogen groups\n", adjustCount);

}

// x = x + c * v
void Sequencer::newMinimizePosition(BigReal c) {
  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;

  for ( int i = 0; i < numAtoms; ++i ) {
    a[i].position += c * a[i].velocity;
  }

  if ( simParams->drudeHardWallOn ) {
    for ( int i = 1; i < numAtoms; ++i ) {
      if ( (0.05 < a[i].mass) && ((a[i].mass < 1.0)) ) { // drude particle
        a[i].position -= a[i-1].position;
      }
    }
  }

  patch->rattle1(0.,0,0);

  if ( simParams->drudeHardWallOn ) {
    for ( int i = 1; i < numAtoms; ++i ) {
      if ( (0.05 < a[i].mass) && ((a[i].mass < 1.0)) ) { // drude particle
        a[i].position += a[i-1].position;
      }
    }
  }
}

// v = 0
void Sequencer::quenchVelocities() {
  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;

  for ( int i = 0; i < numAtoms; ++i ) {
    a[i].velocity = 0;
  }
}

void Sequencer::submitMomentum(int step) {

  FullAtom *a = patch->atom.begin();
  const int numAtoms = patch->numAtoms;

  Vector momentum = 0;
  BigReal mass = 0;
if ( simParams->zeroMomentumAlt ) {
  for ( int i = 0; i < numAtoms; ++i ) {
    momentum += a[i].mass * a[i].velocity;
    mass += 1.;
  }
} else {
  for ( int i = 0; i < numAtoms; ++i ) {
    momentum += a[i].mass * a[i].velocity;
    mass += a[i].mass;
  }
}

  ADD_VECTOR_OBJECT(reduction,REDUCTION_HALFSTEP_MOMENTUM,momentum);
  reduction->item(REDUCTION_MOMENTUM_MASS) += mass;
}

void Sequencer::correctMomentum(int step, BigReal drifttime) {

  //
  // DJH: This test should be done in SimParameters.
  //
  if ( simParams->fixedAtomsOn )
    NAMD_die("Cannot zero momentum when fixed atoms are present.");

  // Blocking receive for the momentum correction vector.
  const Vector dv = broadcast->momentumCorrection.get(step);

  const Vector dx = dv * ( drifttime / TIMEFACTOR );

  FullAtom *a = patch->atom.begin();
  const int numAtoms = patch->numAtoms;

if ( simParams->zeroMomentumAlt ) {
  for ( int i = 0; i < numAtoms; ++i ) {
    a[i].velocity += dv * a[i].recipMass;
    a[i].position += dx * a[i].recipMass;
  }
} else {
  for ( int i = 0; i < numAtoms; ++i ) {
    a[i].velocity += dv;
    a[i].position += dx;
  }
}

}

// --------- For Multigrator ---------
void Sequencer::scalePositionsVelocities(const Tensor& posScale, const Tensor& velScale) {
  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;
  Position origin = patch->lattice.origin();
  if ( simParams->fixedAtomsOn ) {
    NAMD_bug("Sequencer::scalePositionsVelocities, fixed atoms not implemented");
  }
  if ( simParams->useGroupPressure ) {
    int hgs;
    for ( int i = 0; i < numAtoms; i += hgs ) {
      hgs = a[i].hydrogenGroupSize;
      Position pos_cm(0.0, 0.0, 0.0);
      Velocity vel_cm(0.0, 0.0, 0.0);
      BigReal m_cm = 0.0;
      for (int j=0;j < hgs;++j) {
        m_cm += a[i+j].mass;
        pos_cm += a[i+j].mass*a[i+j].position;
        vel_cm += a[i+j].mass*a[i+j].velocity;
      }
      pos_cm /= m_cm;
      vel_cm /= m_cm;
      pos_cm -= origin;
      Position dpos = posScale*pos_cm;
      Velocity dvel = velScale*vel_cm;
      for (int j=0;j < hgs;++j) {
        a[i+j].position += dpos;
        a[i+j].velocity += dvel;
      }
    }
  } else {
    for ( int i = 0; i < numAtoms; i++) {
      a[i].position += posScale*(a[i].position-origin);
      a[i].velocity = velScale*a[i].velocity;
    }
  }
}

void Sequencer::multigratorPressure(int step, int callNumber) {
// Calculate new positions, momenta, and volume using positionRescaleFactor and
// velocityRescaleTensor values returned from Controller::multigratorPressureCalcScale()
  if (simParams->multigratorOn && !(step % simParams->multigratorPressureFreq)) {
    FullAtom *a = patch->atom.begin();
    int numAtoms = patch->numAtoms;

    // Blocking receive (get) scaling factors from Controller
    Tensor scaleTensor    = (callNumber == 1) ? broadcast->positionRescaleFactor.get(step) : broadcast->positionRescaleFactor2.get(step);
    Tensor velScaleTensor = (callNumber == 1) ? broadcast->velocityRescaleTensor.get(step) : broadcast->velocityRescaleTensor2.get(step);
    Tensor posScaleTensor = scaleTensor;
    posScaleTensor -= Tensor::identity();
    if (simParams->useGroupPressure) {
      velScaleTensor -= Tensor::identity();
    }

    // Scale volume
    patch->lattice.rescale(scaleTensor);
    // Scale positions and velocities
    scalePositionsVelocities(posScaleTensor, velScaleTensor);

    if (!patch->flags.doFullElectrostatics) NAMD_bug("Sequencer::multigratorPressure, doFullElectrostatics must be true");

    // Calculate new forces
    // NOTE: We should not need to migrate here since any migration should have happened in the
    // previous call to runComputeObjects inside the MD loop in Sequencer::integrate()
    const int numberOfSteps = simParams->N;
    const int stepsPerCycle = simParams->stepsPerCycle;
    runComputeObjects(0 /*!(step%stepsPerCycle)*/, step<numberOfSteps, 1);

    reduction->item(REDUCTION_ATOM_CHECKSUM) += numAtoms;
    reduction->item(REDUCTION_MARGIN_VIOLATIONS) += patch->marginViolations;

    // Virials etc.
    Tensor virialNormal;
    Tensor momentumSqrSum;
    BigReal kineticEnergy = 0;
    if ( simParams->pairInteractionOn ) {
      if ( simParams->pairInteractionSelf ) {
        for ( int i = 0; i < numAtoms; ++i ) {
          if ( a[i].partition != 1 ) continue;
          kineticEnergy += a[i].mass * a[i].velocity.length2();
          virialNormal.outerAdd(a[i].mass, a[i].velocity, a[i].velocity);
        }
      }
    } else {
      for ( int i = 0; i < numAtoms; ++i ) {
        if (a[i].mass < 0.01) continue;
        kineticEnergy += a[i].mass * a[i].velocity.length2();
        virialNormal.outerAdd(a[i].mass, a[i].velocity, a[i].velocity);
      }
    }
    if (!simParams->useGroupPressure) momentumSqrSum = virialNormal;
    kineticEnergy *= 0.5;
    reduction->item(REDUCTION_CENTERED_KINETIC_ENERGY) += kineticEnergy;
    ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NORMAL, virialNormal);

    if ( simParams->fixedAtomsOn ) {
      Tensor fixVirialNormal;
      Tensor fixVirialNbond;
      Tensor fixVirialSlow;
      Vector fixForceNormal = 0;
      Vector fixForceNbond = 0;
      Vector fixForceSlow = 0;

      calcFixVirial(fixVirialNormal, fixVirialNbond, fixVirialSlow, fixForceNormal, fixForceNbond, fixForceSlow);

      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NORMAL, fixVirialNormal);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NBOND, fixVirialNbond);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_SLOW, fixVirialSlow);
      ADD_VECTOR_OBJECT(reduction, REDUCTION_EXT_FORCE_NORMAL, fixForceNormal);
      ADD_VECTOR_OBJECT(reduction, REDUCTION_EXT_FORCE_NBOND, fixForceNbond);
      ADD_VECTOR_OBJECT(reduction, REDUCTION_EXT_FORCE_SLOW, fixForceSlow);
    }

    // Internal virial and group momentum
    Tensor intVirialNormal;
    Tensor intVirialNormal2;
    Tensor intVirialNbond;
    Tensor intVirialSlow;
    int hgs;
    for ( int i = 0; i < numAtoms; i += hgs ) {
      hgs = a[i].hydrogenGroupSize;
      int j;
      BigReal m_cm = 0;
      Position x_cm(0,0,0);
      Velocity v_cm(0,0,0);
      for ( j = i; j < (i+hgs); ++j ) {
        m_cm += a[j].mass;
        x_cm += a[j].mass * a[j].position;
        v_cm += a[j].mass * a[j].velocity;
      }
      if (simParams->useGroupPressure) momentumSqrSum.outerAdd(1.0/m_cm, v_cm, v_cm);
      x_cm /= m_cm;
      v_cm /= m_cm;
      if (simParams->fixedAtomsOn) NAMD_bug("Sequencer::multigratorPressure, simParams->fixedAtomsOn not implemented yet");
      if ( simParams->pairInteractionOn ) {
        if ( simParams->pairInteractionSelf ) {
          NAMD_bug("Sequencer::multigratorPressure, this part needs to be implemented correctly");
          for ( j = i; j < (i+hgs); ++j ) {
            if ( a[j].partition != 1 ) continue;
            BigReal mass = a[j].mass;
            Vector v = a[j].velocity;
            Vector dv = v - v_cm;
            intVirialNormal2.outerAdd (mass, v, dv);
            Vector dx = a[j].position - x_cm;
            intVirialNormal.outerAdd(1.0, patch->f[Results::normal][j], dx);
            intVirialNbond.outerAdd(1.0, patch->f[Results::nbond][j], dx);
            intVirialSlow.outerAdd(1.0, patch->f[Results::slow][j], dx);
          }
        }
      } else {
        for ( j = i; j < (i+hgs); ++j ) {
          BigReal mass = a[j].mass;
          Vector v = a[j].velocity;
          Vector dv = v - v_cm;
          intVirialNormal2.outerAdd(mass, v, dv);
          Vector dx = a[j].position - x_cm;
          intVirialNormal.outerAdd(1.0, patch->f[Results::normal][j], dx);
          intVirialNbond.outerAdd(1.0, patch->f[Results::nbond][j], dx);
          intVirialSlow.outerAdd(1.0, patch->f[Results::slow][j], dx);
        }
      }
    }

    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NORMAL, intVirialNormal);
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NORMAL, intVirialNormal2);
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NBOND, intVirialNbond);
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_SLOW, intVirialSlow);
    ADD_TENSOR_OBJECT(reduction, REDUCTION_MOMENTUM_SQUARED, momentumSqrSum);

    reduction->submit();
  }
}

void Sequencer::scaleVelocities(const BigReal velScale) {
  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;
  for ( int i = 0; i < numAtoms; i++) {
    a[i].velocity *= velScale;
  }
}

BigReal Sequencer::calcKineticEnergy() {
  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;
  BigReal kineticEnergy = 0.0;
  if ( simParams->pairInteractionOn ) {
    if ( simParams->pairInteractionSelf ) {
      for (int i = 0; i < numAtoms; ++i ) {
        if ( a[i].partition != 1 ) continue;
        kineticEnergy += a[i].mass * a[i].velocity.length2();
      }
    }
  } else {
    for (int i = 0; i < numAtoms; ++i ) {
      kineticEnergy += a[i].mass * a[i].velocity.length2();
    }
  }
  kineticEnergy *= 0.5;
  return kineticEnergy;
}

void Sequencer::multigratorTemperature(int step, int callNumber) {
  if (simParams->multigratorOn && !(step % simParams->multigratorTemperatureFreq)) {
    // Blocking receive (get) velocity scaling factor.
    BigReal velScale = (callNumber == 1) ? broadcast->velocityRescaleFactor.get(step) : broadcast->velocityRescaleFactor2.get(step);
    scaleVelocities(velScale);
    // Calculate new kineticEnergy
    BigReal kineticEnergy = calcKineticEnergy();
    multigratorReduction->item(MULTIGRATOR_REDUCTION_KINETIC_ENERGY) += kineticEnergy;
    if (callNumber == 1 && !(step % simParams->multigratorPressureFreq)) {
      // If this is a pressure cycle, calculate new momentum squared sum
      FullAtom *a = patch->atom.begin();
      int numAtoms = patch->numAtoms;
      Tensor momentumSqrSum;
      if (simParams->useGroupPressure) {
        int hgs;
        for ( int i = 0; i < numAtoms; i += hgs ) {
          hgs = a[i].hydrogenGroupSize;
          int j;
          BigReal m_cm = 0;
          Position x_cm(0,0,0);
          Velocity v_cm(0,0,0);
          for ( j = i; j < (i+hgs); ++j ) {
            m_cm += a[j].mass;
            x_cm += a[j].mass * a[j].position;
            v_cm += a[j].mass * a[j].velocity;
          }
          momentumSqrSum.outerAdd(1.0/m_cm, v_cm, v_cm);
        }
      } else {
        for ( int i = 0; i < numAtoms; i++) {
          momentumSqrSum.outerAdd(a[i].mass, a[i].velocity, a[i].velocity);
        }
      }
      ADD_TENSOR_OBJECT(multigratorReduction, MULTIGRATOR_REDUCTION_MOMENTUM_SQUARED, momentumSqrSum);
    }
    // Submit reductions (kineticEnergy and, if applicable, momentumSqrSum)
    multigratorReduction->submit();

  }
}
// --------- End Multigrator ---------

//
// DJH: Calls one or more addForceToMomentum which in turn calls HomePatch
// versions. We should inline to reduce the number of function calls.
//
void Sequencer::newtonianVelocities(BigReal stepscale, const BigReal timestep,
                                    const BigReal nbondstep,
                                    const BigReal slowstep,
                                    const int staleForces,
                                    const int doNonbonded,
                                    const int doFullElectrostatics)
{
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::NEWTONIAN_VELOCITIES);

  // Deterministic velocity update, account for multigrator
  if (staleForces || (doNonbonded && doFullElectrostatics)) {
    addForceToMomentum3(stepscale*timestep, Results::normal, 0,
                        stepscale*nbondstep, Results::nbond, staleForces,
                        stepscale*slowstep, Results::slow, staleForces);
  } else {
    addForceToMomentum(stepscale*timestep);
    if (staleForces || doNonbonded)
      addForceToMomentum(stepscale*nbondstep, Results::nbond, staleForces);
    if (staleForces || doFullElectrostatics)
      addForceToMomentum(stepscale*slowstep, Results::slow, staleForces);
  }
}

void Sequencer::langevinVelocities(BigReal dt_fs)
{
// This routine is used for the BAOAB integrator,
// Ornstein-Uhlenbeck exact solve for the O-part.
// See B. Leimkuhler and C. Matthews, AMRX (2012)
// Routine originally written by JPhillips, with fresh errors by CMatthews June2012

  if ( simParams->langevinOn && simParams->langevin_useBAOAB )
  {
    FullAtom *a = patch->atom.begin();
    int numAtoms = patch->numAtoms;
    Molecule *molecule = Node::Object()->molecule;
    BigReal dt = dt_fs * 0.001;  // convert to ps
    BigReal kbT = BOLTZMANN*(simParams->langevinTemp);
    if (simParams->adaptTempOn && simParams->adaptTempLangevin)
    {
        kbT = BOLTZMANN*adaptTempT;
    }

    int lesReduceTemp = simParams->lesOn && simParams->lesReduceTemp;
    BigReal tempFactor = lesReduceTemp ? 1.0 / simParams->lesFactor : 1.0;

    for ( int i = 0; i < numAtoms; ++i )
    {
      BigReal dt_gamma = dt * a[i].langevinParam;
      if ( ! dt_gamma ) continue;

      BigReal f1 = exp( -dt_gamma );
      BigReal f2 = sqrt( ( 1. - f1*f1 ) * kbT *
                         ( a[i].partition ? tempFactor : 1.0 ) *
                         a[i].recipMass );
      a[i].velocity *= f1;
      a[i].velocity += f2 * random->gaussian_vector();
    }
  }
}

void Sequencer::langevinVelocitiesBBK1(BigReal dt_fs)
{
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::LANGEVIN_VELOCITIES_BBK1);
  if ( simParams->langevinOn && !simParams->langevin_useBAOAB )
  {
    FullAtom *a = patch->atom.begin();
    int numAtoms = patch->numAtoms;
    Molecule *molecule = Node::Object()->molecule;
    BigReal dt = dt_fs * 0.001;  // convert to ps
    int i;

    if (simParams->drudeOn) {
      for (i = 0;  i < numAtoms;  i++) {

        if (i < numAtoms-1 &&
            a[i+1].mass < 1.0 && a[i+1].mass > 0.05) {
          //printf("*** Found Drude particle %d\n", a[i+1].id);
          // i+1 is a Drude particle with parent i

          // convert from Cartesian coordinates to (COM,bond) coordinates
          BigReal m = a[i+1].mass / (a[i].mass + a[i+1].mass);  // mass ratio
          Vector v_bnd = a[i+1].velocity - a[i].velocity;  // vel of bond
          Vector v_com = a[i].velocity + m * v_bnd;  // vel of COM
          BigReal dt_gamma;

          // use Langevin damping factor i for v_com
          dt_gamma = dt * a[i].langevinParam;
          if (dt_gamma != 0.0) {
            v_com *= ( 1. - 0.5 * dt_gamma );
          }

          // use Langevin damping factor i+1 for v_bnd
          dt_gamma = dt * a[i+1].langevinParam;
          if (dt_gamma != 0.0) {
            v_bnd *= ( 1. - 0.5 * dt_gamma );
          }

          // convert back
          a[i].velocity = v_com - m * v_bnd;
          a[i+1].velocity = v_bnd + a[i].velocity;

          i++;  // +1 from loop, we've updated both particles
        }
        else {
          BigReal dt_gamma = dt * a[i].langevinParam;
          if ( ! dt_gamma ) continue;

          a[i].velocity *= ( 1. - 0.5 * dt_gamma );
        }

      } // end for
    } // end if drudeOn
    else {

      //
      // DJH: The conditional inside loop prevents vectorization and doesn't
      // avoid much work since addition and multiplication are cheap.
      //
      for ( i = 0; i < numAtoms; ++i )
      {
        BigReal dt_gamma = dt * a[i].langevinParam;
        if ( ! dt_gamma ) continue;

        a[i].velocity *= ( 1. - 0.5 * dt_gamma );
      }

    } // end else

  } // end if langevinOn
}


void Sequencer::langevinVelocitiesBBK2(BigReal dt_fs)
{
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::LANGEVIN_VELOCITIES_BBK2);
  if ( simParams->langevinOn && !simParams->langevin_useBAOAB )
  {
    //
    // DJH: This call is expensive. Avoid calling when gammas don't differ.
    // Set flag in SimParameters and make this call conditional.
    //
    TIMER_START(patch->timerSet, RATTLE1);
    rattle1(dt_fs,1);  // conserve momentum if gammas differ
    TIMER_STOP(patch->timerSet, RATTLE1);

    FullAtom *a = patch->atom.begin();
    int numAtoms = patch->numAtoms;
    Molecule *molecule = Node::Object()->molecule;
    BigReal dt = dt_fs * 0.001;  // convert to ps
    BigReal kbT = BOLTZMANN*(simParams->langevinTemp);
    if (simParams->adaptTempOn && simParams->adaptTempLangevin)
    {
        kbT = BOLTZMANN*adaptTempT;
    }
    int lesReduceTemp = simParams->lesOn && simParams->lesReduceTemp;
    BigReal tempFactor = lesReduceTemp ? 1.0 / simParams->lesFactor : 1.0;
    int i;

    if (simParams->drudeOn) {
      BigReal kbT_bnd = BOLTZMANN*(simParams->drudeTemp);  // drude bond Temp

      for (i = 0;  i < numAtoms;  i++) {

        if (i < numAtoms-1 &&
            a[i+1].mass < 1.0 && a[i+1].mass > 0.05) {
          //printf("*** Found Drude particle %d\n", a[i+1].id);
          // i+1 is a Drude particle with parent i

          // convert from Cartesian coordinates to (COM,bond) coordinates
          BigReal m = a[i+1].mass / (a[i].mass + a[i+1].mass);  // mass ratio
          Vector v_bnd = a[i+1].velocity - a[i].velocity;  // vel of bond
          Vector v_com = a[i].velocity + m * v_bnd;  // vel of COM
          BigReal dt_gamma;

          // use Langevin damping factor i for v_com
          dt_gamma = dt * a[i].langevinParam;
          if (dt_gamma != 0.0) {
            BigReal mass = a[i].mass + a[i+1].mass;
            v_com += random->gaussian_vector() *
              sqrt( 2 * dt_gamma * kbT *
                  ( a[i].partition ? tempFactor : 1.0 ) / mass );
            v_com /= ( 1. + 0.5 * dt_gamma );
          }

          // use Langevin damping factor i+1 for v_bnd
          dt_gamma = dt * a[i+1].langevinParam;
          if (dt_gamma != 0.0) {
            BigReal mass = a[i+1].mass * (1. - m);
            v_bnd += random->gaussian_vector() *
              sqrt( 2 * dt_gamma * kbT_bnd *
                  ( a[i+1].partition ? tempFactor : 1.0 ) / mass );
            v_bnd /= ( 1. + 0.5 * dt_gamma );
          }

          // convert back
          a[i].velocity = v_com - m * v_bnd;
          a[i+1].velocity = v_bnd + a[i].velocity;

          i++;  // +1 from loop, we've updated both particles
        }
        else {
          BigReal dt_gamma = dt * a[i].langevinParam;
          if ( ! dt_gamma ) continue;

          a[i].velocity += random->gaussian_vector() *
            sqrt( 2 * dt_gamma * kbT *
                ( a[i].partition ? tempFactor : 1.0 ) * a[i].recipMass );
          a[i].velocity /= ( 1. + 0.5 * dt_gamma );
        }

      } // end for
    } // end if drudeOn
    else {

      //
      // DJH: For case using same gamma (the Langevin parameter),
      // no partitions (e.g. FEP), and no adaptive tempering (adaptTempMD),
      // we can precompute constants. Then by lifting the RNG from the
      // loop (filling up an array of random numbers), we can vectorize
      // loop and simplify arithmetic to just addition and multiplication.
      //
      for ( i = 0; i < numAtoms; ++i )
      {
        BigReal dt_gamma = dt * a[i].langevinParam;
        if ( ! dt_gamma ) continue;

        a[i].velocity += random->gaussian_vector() *
          sqrt( 2 * dt_gamma * kbT *
              ( a[i].partition ? tempFactor : 1.0 ) * a[i].recipMass );
        a[i].velocity /= ( 1. + 0.5 * dt_gamma );
      }

    } // end else

  } // end if langevinOn
}


void Sequencer::berendsenPressure(int step)
{
  if ( simParams->berendsenPressureOn ) {
  berendsenPressure_count += 1;
  const int freq = simParams->berendsenPressureFreq;
  if ( ! (berendsenPressure_count % freq ) ) {
   berendsenPressure_count = 0;
   FullAtom *a = patch->atom.begin();
   int numAtoms = patch->numAtoms;
   // Blocking receive for the updated lattice scaling factor.
   Tensor factor = broadcast->positionRescaleFactor.get(step);
   patch->lattice.rescale(factor);
   if ( simParams->useGroupPressure )
   {
    int hgs;
    for ( int i = 0; i < numAtoms; i += hgs ) {
      int j;
      hgs = a[i].hydrogenGroupSize;
      if ( simParams->fixedAtomsOn && a[i].groupFixed ) {
        for ( j = i; j < (i+hgs); ++j ) {
          a[j].position = patch->lattice.apply_transform(
				a[j].fixedPosition,a[j].transform);
        }
        continue;
      }
      BigReal m_cm = 0;
      Position x_cm(0,0,0);
      for ( j = i; j < (i+hgs); ++j ) {
        if ( simParams->fixedAtomsOn && a[j].atomFixed ) continue;
        m_cm += a[j].mass;
        x_cm += a[j].mass * a[j].position;
      }
      x_cm /= m_cm;
      Position new_x_cm = x_cm;
      patch->lattice.rescale(new_x_cm,factor);
      Position delta_x_cm = new_x_cm - x_cm;
      for ( j = i; j < (i+hgs); ++j ) {
        if ( simParams->fixedAtomsOn && a[j].atomFixed ) {
          a[j].position = patch->lattice.apply_transform(
				a[j].fixedPosition,a[j].transform);
          continue;
        }
        a[j].position += delta_x_cm;
      }
    }
   }
   else
   {
    for ( int i = 0; i < numAtoms; ++i )
    {
      if ( simParams->fixedAtomsOn && a[i].atomFixed ) {
        a[i].position = patch->lattice.apply_transform(
				a[i].fixedPosition,a[i].transform);
        continue;
      }
      patch->lattice.rescale(a[i].position,factor);
    }
   }
  }
  } else {
    berendsenPressure_count = 0;
  }
}

void Sequencer::langevinPiston(int step)
{
  if ( simParams->langevinPistonOn && ! ( (step-1-slowFreq/2) % slowFreq ) )
  {
    //
    // DJH: Loops below simplify if we lift out special cases of fixed atoms
    // and pressure excluded atoms and make them their own branch.
    //
   FullAtom *a = patch->atom.begin();
   int numAtoms = patch->numAtoms;
   // Blocking receive for the updated lattice scaling factor.
   Tensor factor = broadcast->positionRescaleFactor.get(step);
   TIMER_START(patch->timerSet, PISTON);
   // JCP FIX THIS!!!
   Vector velFactor(1/factor.xx,1/factor.yy,1/factor.zz);
   patch->lattice.rescale(factor);
   Molecule *mol = Node::Object()->molecule;
   if ( simParams->useGroupPressure )
   {
    int hgs;
    for ( int i = 0; i < numAtoms; i += hgs ) {
      int j;
      hgs = a[i].hydrogenGroupSize;
      if ( simParams->fixedAtomsOn && a[i].groupFixed ) {
        for ( j = i; j < (i+hgs); ++j ) {
          a[j].position = patch->lattice.apply_transform(
				a[j].fixedPosition,a[j].transform);
        }
        continue;
      }
      BigReal m_cm = 0;
      Position x_cm(0,0,0);
      Velocity v_cm(0,0,0);
      for ( j = i; j < (i+hgs); ++j ) {
        if ( simParams->fixedAtomsOn && a[j].atomFixed ) continue;
        m_cm += a[j].mass;
        x_cm += a[j].mass * a[j].position;
        v_cm += a[j].mass * a[j].velocity;
      }
      x_cm /= m_cm;
      Position new_x_cm = x_cm;
      patch->lattice.rescale(new_x_cm,factor);
      Position delta_x_cm = new_x_cm - x_cm;
      v_cm /= m_cm;
      Velocity delta_v_cm;
      delta_v_cm.x = ( velFactor.x - 1 ) * v_cm.x;
      delta_v_cm.y = ( velFactor.y - 1 ) * v_cm.y;
      delta_v_cm.z = ( velFactor.z - 1 ) * v_cm.z;
      for ( j = i; j < (i+hgs); ++j ) {
        if ( simParams->fixedAtomsOn && a[j].atomFixed ) {
          a[j].position = patch->lattice.apply_transform(
				a[j].fixedPosition,a[j].transform);
          continue;
        }
        if ( mol->is_atom_exPressure(a[j].id) ) continue;
        a[j].position += delta_x_cm;
        a[j].velocity += delta_v_cm;
      }
    }
   }
   else
   {
    for ( int i = 0; i < numAtoms; ++i )
    {
      if ( simParams->fixedAtomsOn && a[i].atomFixed ) {
        a[i].position = patch->lattice.apply_transform(
				a[i].fixedPosition,a[i].transform);
        continue;
      }
      if ( mol->is_atom_exPressure(a[i].id) ) continue;
      patch->lattice.rescale(a[i].position,factor);
      a[i].velocity.x *= velFactor.x;
      a[i].velocity.y *= velFactor.y;
      a[i].velocity.z *= velFactor.z;
    }
   }
   TIMER_STOP(patch->timerSet, PISTON);
  }
}

void Sequencer::rescaleVelocities(int step)
{
  const int rescaleFreq = simParams->rescaleFreq;
  if ( rescaleFreq > 0 ) {
    FullAtom *a = patch->atom.begin();
    int numAtoms = patch->numAtoms;
    ++rescaleVelocities_numTemps;
    if ( rescaleVelocities_numTemps == rescaleFreq ) {
      // Blocking receive for the velcity scaling factor.
      BigReal factor = broadcast->velocityRescaleFactor.get(step);
      for ( int i = 0; i < numAtoms; ++i )
      {
        a[i].velocity *= factor;
      }
      rescaleVelocities_numTemps = 0;
    }
  }
}

void Sequencer::rescaleaccelMD (int step, int doNonbonded, int doFullElectrostatics)
{
   if (!simParams->accelMDOn) return;
   if ((step < simParams->accelMDFirstStep) || ( simParams->accelMDLastStep >0 && step > simParams->accelMDLastStep)) return;

   // Blocking receive for the Accelerated MD scaling factors.
   Vector accelMDfactor = broadcast->accelMDRescaleFactor.get(step);
   const BigReal factor_dihe = accelMDfactor[0];
   const BigReal factor_tot  = accelMDfactor[1];
   const int numAtoms = patch->numAtoms;

   if (simParams->accelMDdihe && factor_tot <1 )
       NAMD_die("accelMD broadcasting error!\n");
   if (!simParams->accelMDdihe && !simParams->accelMDdual && factor_dihe <1 )
       NAMD_die("accelMD broadcasting error!\n");

   if (simParams->accelMDdihe && factor_dihe < 1) {
        for (int i = 0; i < numAtoms; ++i)
          if (patch->f[Results::amdf][i][0] || patch->f[Results::amdf][i][1] || patch->f[Results::amdf][i][2])
              patch->f[Results::normal][i] += patch->f[Results::amdf][i]*(factor_dihe - 1);
   }

   if ( !simParams->accelMDdihe && factor_tot < 1) {
        for (int i = 0; i < numAtoms; ++i)
            patch->f[Results::normal][i] *= factor_tot;
        if (doNonbonded) {
            for (int i = 0; i < numAtoms; ++i)
                 patch->f[Results::nbond][i] *= factor_tot;
        }
        if (doFullElectrostatics) {
            for (int i = 0; i < numAtoms; ++i)
                 patch->f[Results::slow][i] *= factor_tot;
        }
   }

   if (simParams->accelMDdual && factor_dihe < 1) {
        for (int i = 0; i < numAtoms; ++i)
           if (patch->f[Results::amdf][i][0] || patch->f[Results::amdf][i][1] || patch->f[Results::amdf][i][2])
               patch->f[Results::normal][i] += patch->f[Results::amdf][i]*(factor_dihe - factor_tot);
   }

}

void Sequencer::adaptTempUpdate(int step)
{
   //check if adaptive tempering is enabled and in the right timestep range
   if (!simParams->adaptTempOn) return;
   if ( (step < simParams->adaptTempFirstStep ) ||
     ( simParams->adaptTempLastStep > 0 && step > simParams->adaptTempLastStep )) {
        if (simParams->langevinOn) // restore langevin temperature
            adaptTempT = simParams->langevinTemp;
        return;
   }
   // Get Updated Temperature
   if ( !(step % simParams->adaptTempFreq ) && (step > simParams->firstTimestep ))
     // Blocking receive for the updated adaptive tempering temperature.
     adaptTempT = broadcast->adaptTemperature.get(step);
}

void Sequencer::reassignVelocities(BigReal timestep, int step)
{
  const int reassignFreq = simParams->reassignFreq;
  if ( ( reassignFreq > 0 ) && ! ( step % reassignFreq ) ) {
    FullAtom *a = patch->atom.begin();
    int numAtoms = patch->numAtoms;
    BigReal newTemp = simParams->reassignTemp;
    newTemp += ( step / reassignFreq ) * simParams->reassignIncr;
    if ( simParams->reassignIncr > 0.0 ) {
      if ( newTemp > simParams->reassignHold && simParams->reassignHold > 0.0 )
        newTemp = simParams->reassignHold;
    } else {
      if ( newTemp < simParams->reassignHold )
        newTemp = simParams->reassignHold;
    }
    BigReal kbT = BOLTZMANN * newTemp;

    int lesReduceTemp = simParams->lesOn && simParams->lesReduceTemp;
    BigReal tempFactor = lesReduceTemp ? 1.0 / simParams->lesFactor : 1.0;

    for ( int i = 0; i < numAtoms; ++i )
    {
      a[i].velocity = ( ( simParams->fixedAtomsOn &&
            a[i].atomFixed && a[i].mass > 0.) ? Vector(0,0,0) :
          sqrt(kbT * (a[i].partition ? tempFactor : 1.0) * a[i].recipMass) *
          random->gaussian_vector() );
    }
  } else {
    NAMD_bug("Sequencer::reassignVelocities called improperly!");
  }
}

void Sequencer::reinitVelocities(void)
{
  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;
  BigReal newTemp = simParams->initialTemp;
  BigReal kbT = BOLTZMANN * newTemp;

  int lesReduceTemp = simParams->lesOn && simParams->lesReduceTemp;
  BigReal tempFactor = lesReduceTemp ? 1.0 / simParams->lesFactor : 1.0;

  for ( int i = 0; i < numAtoms; ++i )
  {
    a[i].velocity = ( ( (simParams->fixedAtomsOn && a[i].atomFixed) ||
          a[i].mass <= 0.) ? Vector(0,0,0) :
        sqrt(kbT * (a[i].partition ? tempFactor : 1.0) * a[i].recipMass) *
        random->gaussian_vector() );
    if ( simParams->drudeOn && i+1 < numAtoms && a[i+1].mass < 1.0 && a[i+1].mass > 0.05 ) {
      a[i+1].velocity = a[i].velocity;  // zero is good enough
      ++i;
    }
  }
}

void Sequencer::rescaleVelocitiesByFactor(BigReal factor)
{
  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;
  for ( int i = 0; i < numAtoms; ++i )
  {
    a[i].velocity *= factor;
  }
}

void Sequencer::reloadCharges()
{
  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;
  Molecule *molecule = Node::Object()->molecule;
  for ( int i = 0; i < numAtoms; ++i )
  {
    a[i].charge = molecule->atomcharge(a[i].id);
  }
}

// REST2 solute charge scaling
void Sequencer::rescaleSoluteCharges(BigReal factor)
{
  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;
  Molecule *molecule = Node::Object()->molecule;
  BigReal sqrt_factor = sqrt(factor);
  // apply scaling to the original charge (stored in molecule)
  // of just the marked solute atoms
  for ( int i = 0; i < numAtoms; ++i ) {
    if (molecule->get_ss_type(a[i].id)) {
      a[i].charge = sqrt_factor * molecule->atomcharge(a[i].id);
      if (simParams->SOAintegrateOn) patch->patchDataSOA.charge[i] = a[i].charge;
    }
  }
}

void Sequencer::tcoupleVelocities(BigReal dt_fs, int step)
{
  if ( simParams->tCoupleOn )
  {
    FullAtom *a = patch->atom.begin();
    int numAtoms = patch->numAtoms;
    // Blocking receive for the temperature coupling coefficient.
    BigReal coefficient = broadcast->tcoupleCoefficient.get(step);
    Molecule *molecule = Node::Object()->molecule;
    BigReal dt = dt_fs * 0.001;  // convert to ps
    coefficient *= dt;
    for ( int i = 0; i < numAtoms; ++i )
    {
      BigReal f1 = exp( coefficient * a[i].langevinParam );
      a[i].velocity *= f1;
    }
  }
}

/** Rescale velocities with the scale factor sent from the Controller.
 *
 *  \param step The current timestep
 */
void Sequencer::stochRescaleVelocities(int step)
{
  ++stochRescale_count;
  if ( stochRescale_count == simParams->stochRescaleFreq ) {
    FullAtom *a = patch->atom.begin();
    int numAtoms = patch->numAtoms;
    // Blocking receive for the temperature coupling coefficient.
    BigReal velrescaling = broadcast->stochRescaleCoefficient.get(step);
    DebugM(4, "stochastically rescaling velocities at step " << step << " by " << velrescaling << "\n");
    for ( int i = 0; i < numAtoms; ++i ) {
      a[i].velocity *= velrescaling;
    }
    stochRescale_count = 0;
  }
}

void Sequencer::saveForce(const int ftag)
{
  patch->saveForce(ftag);
}

//
// DJH: Need to change division by TIMEFACTOR into multiplication by
// reciprocal of TIMEFACTOR. Done several times for each iteration of
// the integrate() loop.
//

void Sequencer::addForceToMomentum(
    BigReal timestep, const int ftag, const int useSaved
    ) {
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::ADD_FORCE_TO_MOMENTUM);
#if CMK_BLUEGENEL
  CmiNetworkProgressAfter (0);
#endif
  const BigReal dt = timestep / TIMEFACTOR;
  FullAtom *atom_arr  = patch->atom.begin();
  ForceList *f_use = (useSaved ? patch->f_saved : patch->f);
  const Force *force_arr = f_use[ftag].const_begin();
  patch->addForceToMomentum(atom_arr, force_arr, dt, patch->numAtoms);
}

void Sequencer::addForceToMomentum3(
    const BigReal timestep1, const int ftag1, const int useSaved1,
    const BigReal timestep2, const int ftag2, const int useSaved2,
    const BigReal timestep3, const int ftag3, const int useSaved3
    ) {
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::ADD_FORCE_TO_MOMENTUM);
#if CMK_BLUEGENEL
  CmiNetworkProgressAfter (0);
#endif
  const BigReal dt1 = timestep1 / TIMEFACTOR;
  const BigReal dt2 = timestep2 / TIMEFACTOR;
  const BigReal dt3 = timestep3 / TIMEFACTOR;
  ForceList *f_use1 = (useSaved1 ? patch->f_saved : patch->f);
  ForceList *f_use2 = (useSaved2 ? patch->f_saved : patch->f);
  ForceList *f_use3 = (useSaved3 ? patch->f_saved : patch->f);
  FullAtom *atom_arr  = patch->atom.begin();
  const Force *force_arr1 = f_use1[ftag1].const_begin();
  const Force *force_arr2 = f_use2[ftag2].const_begin();
  const Force *force_arr3 = f_use3[ftag3].const_begin();
  patch->addForceToMomentum3 (atom_arr, force_arr1, force_arr2, force_arr3,
      dt1, dt2, dt3, patch->numAtoms);
}

void Sequencer::addVelocityToPosition(BigReal timestep)
{
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::ADD_VELOCITY_TO_POSITION);
#if CMK_BLUEGENEL
  CmiNetworkProgressAfter (0);
#endif
  const BigReal dt = timestep / TIMEFACTOR;
  FullAtom *atom_arr  = patch->atom.begin();
  patch->addVelocityToPosition(atom_arr, dt, patch->numAtoms);
}

void Sequencer::hardWallDrude(BigReal dt, int pressure)
{
  if ( simParams->drudeHardWallOn ) {
    Tensor virial;
    Tensor *vp = ( pressure ? &virial : 0 );
    if ( patch->hardWallDrude(dt, vp, pressureProfileReduction) ) {
      iout << iERROR << "Constraint failure in HardWallDrude(); "
        << "simulation may become unstable.\n" << endi;
      Node::Object()->enableEarlyExit();
      terminate();
    }
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,virial);
  }
}

void Sequencer::rattle1(BigReal dt, int pressure)
{
  NAMD_EVENT_RANGE_2(patch->flags.event_on, NamdProfileEvent::RATTLE1);
  if ( simParams->rigidBonds != RIGID_NONE ) {
    Tensor virial;
    Tensor *vp = ( pressure ? &virial : 0 );
    if ( patch->rattle1(dt, vp, pressureProfileReduction) ) {
      iout << iERROR <<
        "Constraint failure; simulation has become unstable.\n" << endi;
      Node::Object()->enableEarlyExit();
      terminate();
    }
#if 0
    printf("virial = %g %g %g  %g %g %g  %g %g %g\n",
        virial.xx, virial.xy, virial.xz,
        virial.yx, virial.yy, virial.yz,
        virial.zx, virial.zy, virial.zz);
#endif
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,virial);
#if 0
    {
      const FullAtom *a = patch->atom.const_begin();
      for (int n=0;  n < patch->numAtoms;  n++) {
        printf("pos[%d] =  %g %g %g\n", n,
            a[n].position.x, a[n].position.y, a[n].position.z);
      }
      for (int n=0;  n < patch->numAtoms;  n++) {
        printf("vel[%d] =  %g %g %g\n", n,
            a[n].velocity.x, a[n].velocity.y, a[n].velocity.z);
      }
      if (pressure) {
        for (int n=0;  n < patch->numAtoms;  n++) {
          printf("force[%d] =  %g %g %g\n", n,
              patch->f[Results::normal][n].x,
              patch->f[Results::normal][n].y,
              patch->f[Results::normal][n].z);
        }
      }
    }
#endif
  }
}

// void Sequencer::rattle2(BigReal dt, int step)
// {
//   if ( simParams->rigidBonds != RIGID_NONE ) {
//     Tensor virial;
//     patch->rattle2(dt, &virial);
//     ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,virial);
//     // we need to add to alt and int virial because not included in forces
// #ifdef ALTVIRIAL
//     ADD_TENSOR_OBJECT(reduction,REDUCTION_ALT_VIRIAL_NORMAL,virial);
// #endif
//     ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_NORMAL,virial);
//   }
// }

void Sequencer::maximumMove(BigReal timestep)
{
  NAMD_EVENT_RANGE_2(patch->flags.event_on, NamdProfileEvent::MAXIMUM_MOVE);

  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;
  if ( simParams->maximumMove ) {
    const BigReal dt = timestep / TIMEFACTOR;
    const BigReal maxvel = simParams->maximumMove / dt;
    const BigReal maxvel2 = maxvel * maxvel;
    for ( int i=0; i<numAtoms; ++i ) {
      if ( a[i].velocity.length2() > maxvel2 ) {
	a[i].velocity *= ( maxvel / a[i].velocity.length() );
      }
    }
  } else {
    const BigReal dt = timestep / TIMEFACTOR;
    const BigReal maxvel = simParams->cutoff / dt;
    const BigReal maxvel2 = maxvel * maxvel;
    int killme = 0;
    for ( int i=0; i<numAtoms; ++i ) {
      killme = killme || ( a[i].velocity.length2() > maxvel2 );
    }
    if ( killme ) {
      killme = 0;
      for ( int i=0; i<numAtoms; ++i ) {
        if ( a[i].velocity.length2() > maxvel2 ) {
          ++killme;
          iout << iERROR << "Atom " << (a[i].id + 1) << " velocity is "
            << ( PDBVELFACTOR * a[i].velocity ) << " (limit is "
            << ( PDBVELFACTOR * maxvel ) << ", atom "
            << i << " of " << numAtoms << " on patch "
            << patch->patchID << " pe " << CkMyPe() << ")\n" << endi;
        }
      }
      iout << iERROR <<
        "Atoms moving too fast; simulation has become unstable ("
        << killme << " atoms on patch " << patch->patchID
        << " pe " << CkMyPe() << ").\n" << endi;
      Node::Object()->enableEarlyExit();
      terminate();
    }
  }
}

void Sequencer::minimizationQuenchVelocity(void)
{
  if ( simParams->minimizeOn ) {
    FullAtom *a = patch->atom.begin();
    int numAtoms = patch->numAtoms;
    for ( int i=0; i<numAtoms; ++i ) {
      a[i].velocity = 0.;
    }
  }
}

void Sequencer::submitHalfstep(int step)
{
  NAMD_EVENT_RANGE_2(patch->flags.event_on, NamdProfileEvent::SUBMIT_HALFSTEP);

  // velocity-dependent quantities *** ONLY ***
  // positions are not at half-step when called
  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;

#if CMK_BLUEGENEL
  CmiNetworkProgressAfter (0);
#endif

  // For non-Multigrator doKineticEnergy = 1 always
  Tensor momentumSqrSum;
  if (doKineticEnergy || patch->flags.doVirial)
  {
    BigReal kineticEnergy = 0;
    Tensor virial;
    if ( simParams->pairInteractionOn ) {
      if ( simParams->pairInteractionSelf ) {
        for ( int i = 0; i < numAtoms; ++i ) {
          if ( a[i].partition != 1 ) continue;
          kineticEnergy += a[i].mass * a[i].velocity.length2();
          virial.outerAdd(a[i].mass, a[i].velocity, a[i].velocity);
        }
      }
    } else {
      for ( int i = 0; i < numAtoms; ++i ) {
        if (a[i].mass < 0.01) continue;
        kineticEnergy += a[i].mass * a[i].velocity.length2();
        virial.outerAdd(a[i].mass, a[i].velocity, a[i].velocity);
      }
    }

    if (simParams->multigratorOn && !simParams->useGroupPressure) {
      momentumSqrSum = virial;
    }
    kineticEnergy *= 0.5 * 0.5;
    reduction->item(REDUCTION_HALFSTEP_KINETIC_ENERGY) += kineticEnergy;
    virial *= 0.5;
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,virial);
#ifdef ALTVIRIAL
    ADD_TENSOR_OBJECT(reduction,REDUCTION_ALT_VIRIAL_NORMAL,virial);
#endif
  }

  if (pressureProfileReduction) {
    int nslabs = simParams->pressureProfileSlabs;
    const Lattice &lattice = patch->lattice;
    BigReal idz = nslabs/lattice.c().z;
    BigReal zmin = lattice.origin().z - 0.5*lattice.c().z;
    int useGroupPressure = simParams->useGroupPressure;

    // Compute kinetic energy partition, possibly subtracting off
    // internal kinetic energy if group pressure is enabled.
    // Since the regular pressure is 1/2 mvv and the internal kinetic
    // term that is subtracted off for the group pressure is
    // 1/2 mv (v-v_cm), the group pressure kinetic contribution is
    // 1/2 m * v * v_cm.  The factor of 1/2 is because submitHalfstep
    // gets called twice per timestep.
    int hgs;
    for (int i=0; i<numAtoms; i += hgs) {
      int j, ppoffset;
      hgs = a[i].hydrogenGroupSize;
      int partition = a[i].partition;

      BigReal m_cm = 0;
      Velocity v_cm(0,0,0);
      for (j=i; j< i+hgs; ++j) {
        m_cm += a[j].mass;
        v_cm += a[j].mass * a[j].velocity;
      }
      v_cm /= m_cm;
      for (j=i; j < i+hgs; ++j) {
        BigReal mass = a[j].mass;
        if (! (useGroupPressure && j != i)) {
          BigReal z = a[j].position.z;
          int slab = (int)floor((z-zmin)*idz);
          if (slab < 0) slab += nslabs;
          else if (slab >= nslabs) slab -= nslabs;
          ppoffset = 3*(slab + partition*nslabs);
        }
        BigReal wxx, wyy, wzz;
        if (useGroupPressure) {
          wxx = 0.5*mass * a[j].velocity.x * v_cm.x;
          wyy = 0.5*mass * a[j].velocity.y * v_cm.y;
          wzz = 0.5*mass * a[j].velocity.z * v_cm.z;
        } else {
          wxx = 0.5*mass * a[j].velocity.x * a[j].velocity.x;
          wyy = 0.5*mass * a[j].velocity.y * a[j].velocity.y;
          wzz = 0.5*mass * a[j].velocity.z * a[j].velocity.z;
        }
        pressureProfileReduction->item(ppoffset  ) += wxx;
        pressureProfileReduction->item(ppoffset+1) += wyy;
        pressureProfileReduction->item(ppoffset+2) += wzz;
      }
    }
  }

  // For non-Multigrator doKineticEnergy = 1 always
  if (doKineticEnergy || patch->flags.doVirial)
  {
    BigReal intKineticEnergy = 0;
    Tensor intVirialNormal;

    int hgs;
    for ( int i = 0; i < numAtoms; i += hgs ) {

#if CMK_BLUEGENEL
      CmiNetworkProgress ();
#endif

      hgs = a[i].hydrogenGroupSize;
      int j;
      BigReal m_cm = 0;
      Velocity v_cm(0,0,0);
      for ( j = i; j < (i+hgs); ++j ) {
        m_cm += a[j].mass;
        v_cm += a[j].mass * a[j].velocity;
      }
      if (simParams->multigratorOn && simParams->useGroupPressure) {
        momentumSqrSum.outerAdd(1.0/m_cm, v_cm, v_cm);
      }
      v_cm /= m_cm;
      if ( simParams->pairInteractionOn ) {
        if ( simParams->pairInteractionSelf ) {
          for ( j = i; j < (i+hgs); ++j ) {
            if ( a[j].partition != 1 ) continue;
            BigReal mass = a[j].mass;
            Vector v = a[j].velocity;
            Vector dv = v - v_cm;
            intKineticEnergy += mass * (v * dv);
            intVirialNormal.outerAdd (mass, v, dv);
          }
        }
      } else {
        for ( j = i; j < (i+hgs); ++j ) {
          BigReal mass = a[j].mass;
          Vector v = a[j].velocity;
          Vector dv = v - v_cm;
          intKineticEnergy += mass * (v * dv);
          intVirialNormal.outerAdd(mass, v, dv);
        }
      }
    }
    intKineticEnergy *= 0.5 * 0.5;
    reduction->item(REDUCTION_INT_HALFSTEP_KINETIC_ENERGY) += intKineticEnergy;
    intVirialNormal *= 0.5;
    ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_NORMAL,intVirialNormal);
    if ( simParams->multigratorOn) {
      momentumSqrSum *= 0.5;
      ADD_TENSOR_OBJECT(reduction,REDUCTION_MOMENTUM_SQUARED,momentumSqrSum);
    }
  }

}

void Sequencer::calcFixVirial(Tensor& fixVirialNormal, Tensor& fixVirialNbond, Tensor& fixVirialSlow,
  Vector& fixForceNormal, Vector& fixForceNbond, Vector& fixForceSlow) {

  FullAtom *a = patch->atom.begin();
  int numAtoms = patch->numAtoms;

  for ( int j = 0; j < numAtoms; j++ ) {
    if ( simParams->fixedAtomsOn && a[j].atomFixed ) {
      Vector dx = a[j].fixedPosition;
      // all negative because fixed atoms cancels these forces
      fixVirialNormal.outerAdd(-1.0, patch->f[Results::normal][j], dx);
      fixVirialNbond.outerAdd(-1.0, patch->f[Results::nbond][j], dx);
      fixVirialSlow.outerAdd(-1.0, patch->f[Results::slow][j], dx);
      fixForceNormal -= patch->f[Results::normal][j];
      fixForceNbond -= patch->f[Results::nbond][j];
      fixForceSlow -= patch->f[Results::slow][j];
    }
  }
}

void Sequencer::submitReductions(int step)
{
#ifndef UPPER_BOUND
  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::SUBMIT_REDUCTIONS);
  FullAtom *a = patch->atom.begin();
#endif
  int numAtoms = patch->numAtoms;

#if CMK_BLUEGENEL
  CmiNetworkProgressAfter(0);
#endif

  reduction->item(REDUCTION_ATOM_CHECKSUM) += numAtoms;
  reduction->item(REDUCTION_MARGIN_VIOLATIONS) += patch->marginViolations;

#ifndef UPPER_BOUND
  // For non-Multigrator doKineticEnergy = 1 always
  if (doKineticEnergy || doMomenta || patch->flags.doVirial)
  {
    BigReal kineticEnergy = 0;
    Vector momentum = 0;
    Vector angularMomentum = 0;
    Vector o = patch->lattice.origin();
    int i;
    if ( simParams->pairInteractionOn ) {
      if ( simParams->pairInteractionSelf ) {
        for (i = 0; i < numAtoms; ++i ) {
          if ( a[i].partition != 1 ) continue;
          kineticEnergy += a[i].mass * a[i].velocity.length2();
          momentum += a[i].mass * a[i].velocity;
          angularMomentum += cross(a[i].mass,a[i].position-o,a[i].velocity);
        }
      }
    } else {
      for (i = 0; i < numAtoms; ++i ) {
        kineticEnergy += a[i].mass * a[i].velocity.length2();
        momentum += a[i].mass * a[i].velocity;
        angularMomentum += cross(a[i].mass,a[i].position-o,a[i].velocity);
      }
      if (simParams->drudeOn) {
        BigReal drudeComKE = 0.;
        BigReal drudeBondKE = 0.;

        for (i = 0;  i < numAtoms;  i++) {
          if (i < numAtoms-1 &&
              a[i+1].mass < 1.0 && a[i+1].mass > 0.05) {
            // i+1 is a Drude particle with parent i

            // convert from Cartesian coordinates to (COM,bond) coordinates
            BigReal m_com = (a[i].mass + a[i+1].mass);  // mass of COM
            BigReal m = a[i+1].mass / m_com;  // mass ratio
            BigReal m_bond = a[i+1].mass * (1. - m);  // mass of bond
            Vector v_bond = a[i+1].velocity - a[i].velocity;  // vel of bond
            Vector v_com = a[i].velocity + m * v_bond;  // vel of COM

            drudeComKE += m_com * v_com.length2();
            drudeBondKE += m_bond * v_bond.length2();

            i++;  // +1 from loop, we've updated both particles
          }
          else {
            drudeComKE += a[i].mass * a[i].velocity.length2();
          }
        } // end for

        drudeComKE *= 0.5;
        drudeBondKE *= 0.5;
        reduction->item(REDUCTION_DRUDECOM_CENTERED_KINETIC_ENERGY)
          += drudeComKE;
        reduction->item(REDUCTION_DRUDEBOND_CENTERED_KINETIC_ENERGY)
          += drudeBondKE;
      } // end drudeOn

    } // end else

    kineticEnergy *= 0.5;
    reduction->item(REDUCTION_CENTERED_KINETIC_ENERGY) += kineticEnergy;
    ADD_VECTOR_OBJECT(reduction,REDUCTION_MOMENTUM,momentum);
    ADD_VECTOR_OBJECT(reduction,REDUCTION_ANGULAR_MOMENTUM,angularMomentum);
  }

#ifdef ALTVIRIAL
  // THIS IS NOT CORRECTED FOR PAIR INTERACTIONS
  {
    Tensor altVirial;
    for ( int i = 0; i < numAtoms; ++i ) {
      altVirial.outerAdd(1.0, patch->f[Results::normal][i], a[i].position);
    }
    ADD_TENSOR_OBJECT(reduction,REDUCTION_ALT_VIRIAL_NORMAL,altVirial);
  }
  {
    Tensor altVirial;
    for ( int i = 0; i < numAtoms; ++i ) {
      altVirial.outerAdd(1.0, patch->f[Results::nbond][i], a[i].position);
    }
    ADD_TENSOR_OBJECT(reduction,REDUCTION_ALT_VIRIAL_NBOND,altVirial);
  }
  {
    Tensor altVirial;
    for ( int i = 0; i < numAtoms; ++i ) {
      altVirial.outerAdd(1.0, patch->f[Results::slow][i], a[i].position);
    }
    ADD_TENSOR_OBJECT(reduction,REDUCTION_ALT_VIRIAL_SLOW,altVirial);
  }
#endif

  // For non-Multigrator doKineticEnergy = 1 always
  if (doKineticEnergy || patch->flags.doVirial)
  {
    BigReal intKineticEnergy = 0;
    Tensor intVirialNormal;
    Tensor intVirialNbond;
    Tensor intVirialSlow;

    int hgs;
    for ( int i = 0; i < numAtoms; i += hgs ) {
#if CMK_BLUEGENEL
      CmiNetworkProgress();
#endif
      hgs = a[i].hydrogenGroupSize;
      int j;
      BigReal m_cm = 0;
      Position x_cm(0,0,0);
      Velocity v_cm(0,0,0);
      for ( j = i; j < (i+hgs); ++j ) {
        m_cm += a[j].mass;
        x_cm += a[j].mass * a[j].position;
        v_cm += a[j].mass * a[j].velocity;
      }
      x_cm /= m_cm;
      v_cm /= m_cm;
      int fixedAtomsOn = simParams->fixedAtomsOn;
      if ( simParams->pairInteractionOn ) {
        int pairInteractionSelf = simParams->pairInteractionSelf;
        for ( j = i; j < (i+hgs); ++j ) {
          if ( a[j].partition != 1 &&
               ( pairInteractionSelf || a[j].partition != 2 ) ) continue;
          // net force treated as zero for fixed atoms
          if ( fixedAtomsOn && a[j].atomFixed ) continue;
          BigReal mass = a[j].mass;
          Vector v = a[j].velocity;
          Vector dv = v - v_cm;
          intKineticEnergy += mass * (v * dv);
          Vector dx = a[j].position - x_cm;
          intVirialNormal.outerAdd(1.0, patch->f[Results::normal][j], dx);
          intVirialNbond.outerAdd(1.0, patch->f[Results::nbond][j], dx);
          intVirialSlow.outerAdd(1.0, patch->f[Results::slow][j], dx);
        }
      } else {
        for ( j = i; j < (i+hgs); ++j ) {
          // net force treated as zero for fixed atoms
          if ( fixedAtomsOn && a[j].atomFixed ) continue;
          BigReal mass = a[j].mass;
          Vector v = a[j].velocity;
          Vector dv = v - v_cm;
          intKineticEnergy += mass * (v * dv);
          Vector dx = a[j].position - x_cm;
          intVirialNormal.outerAdd(1.0, patch->f[Results::normal][j], dx);
          intVirialNbond.outerAdd(1.0, patch->f[Results::nbond][j], dx);
          intVirialSlow.outerAdd(1.0, patch->f[Results::slow][j], dx);
        }
      }
    }

    intKineticEnergy *= 0.5;
    reduction->item(REDUCTION_INT_CENTERED_KINETIC_ENERGY) += intKineticEnergy;
    ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_NORMAL,intVirialNormal);
    ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_NBOND,intVirialNbond);
    ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_SLOW,intVirialSlow);
  }

  if (pressureProfileReduction && simParams->useGroupPressure) {
    // subtract off internal virial term, calculated as for intVirial.
    int nslabs = simParams->pressureProfileSlabs;
    const Lattice &lattice = patch->lattice;
    BigReal idz = nslabs/lattice.c().z;
    BigReal zmin = lattice.origin().z - 0.5*lattice.c().z;
    int useGroupPressure = simParams->useGroupPressure;

    int hgs;
    for (int i=0; i<numAtoms; i += hgs) {
      int j;
      hgs = a[i].hydrogenGroupSize;
      BigReal m_cm = 0;
      Position x_cm(0,0,0);
      for (j=i; j< i+hgs; ++j) {
        m_cm += a[j].mass;
        x_cm += a[j].mass * a[j].position;
      }
      x_cm /= m_cm;

      BigReal z = a[i].position.z;
      int slab = (int)floor((z-zmin)*idz);
      if (slab < 0) slab += nslabs;
      else if (slab >= nslabs) slab -= nslabs;
      int partition = a[i].partition;
      int ppoffset = 3*(slab + nslabs*partition);
      for (j=i; j < i+hgs; ++j) {
        BigReal mass = a[j].mass;
        Vector dx = a[j].position - x_cm;
        const Vector &fnormal = patch->f[Results::normal][j];
        const Vector &fnbond  = patch->f[Results::nbond][j];
        const Vector &fslow   = patch->f[Results::slow][j];
        BigReal wxx = (fnormal.x + fnbond.x + fslow.x) * dx.x;
        BigReal wyy = (fnormal.y + fnbond.y + fslow.y) * dx.y;
        BigReal wzz = (fnormal.z + fnbond.z + fslow.z) * dx.z;
        pressureProfileReduction->item(ppoffset  ) -= wxx;
        pressureProfileReduction->item(ppoffset+1) -= wyy;
        pressureProfileReduction->item(ppoffset+2) -= wzz;
      }
    }
  }

  // For non-Multigrator doVirial = 1 always
  if (patch->flags.doVirial)
  {
    if ( simParams->fixedAtomsOn ) {
      Tensor fixVirialNormal;
      Tensor fixVirialNbond;
      Tensor fixVirialSlow;
      Vector fixForceNormal = 0;
      Vector fixForceNbond = 0;
      Vector fixForceSlow = 0;

      calcFixVirial(fixVirialNormal, fixVirialNbond, fixVirialSlow, fixForceNormal, fixForceNbond, fixForceSlow);

      ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,fixVirialNormal);
      ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NBOND,fixVirialNbond);
      ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_SLOW,fixVirialSlow);
      ADD_VECTOR_OBJECT(reduction,REDUCTION_EXT_FORCE_NORMAL,fixForceNormal);
      ADD_VECTOR_OBJECT(reduction,REDUCTION_EXT_FORCE_NBOND,fixForceNbond);
      ADD_VECTOR_OBJECT(reduction,REDUCTION_EXT_FORCE_SLOW,fixForceSlow);
    }
  }
#endif // UPPER_BOUND

  reduction->submit();
#ifndef UPPER_BOUND
  if (pressureProfileReduction) pressureProfileReduction->submit();
#endif
}

void Sequencer::submitMinimizeReductions(int step, BigReal fmax2)
{
  FullAtom *a = patch->atom.begin();
  Force *f1 = patch->f[Results::normal].begin();
  Force *f2 = patch->f[Results::nbond].begin();
  Force *f3 = patch->f[Results::slow].begin();
  const bool fixedAtomsOn = simParams->fixedAtomsOn;
  const bool drudeHardWallOn = simParams->drudeHardWallOn;
  const double drudeBondLen = simParams->drudeBondLen;
  const double drudeBondLen2 = drudeBondLen * drudeBondLen;
  const double drudeStep = 0.1/(TIMEFACTOR*TIMEFACTOR);
  const double drudeMove = 0.01;
  const double drudeStep2 = drudeStep * drudeStep;
  const double drudeMove2 = drudeMove * drudeMove;
  int numAtoms = patch->numAtoms;

  reduction->item(REDUCTION_ATOM_CHECKSUM) += numAtoms;

  for ( int i = 0; i < numAtoms; ++i ) {
#if 0
    printf("ap[%2d]=  %f  %f  %f\n", i, a[i].position.x, a[i].position.y, a[i].position.z);
    printf("f1[%2d]=  %f  %f  %f\n", i, f1[i].x, f1[i].y, f1[i].z);
    printf("f2[%2d]=  %f  %f  %f\n", i, f2[i].x, f2[i].y, f2[i].z);
    //printf("f3[%2d]=  %f  %f  %f\n", i, f3[i].x, f3[i].y, f3[i].z);
#endif
    f1[i] += f2[i] + f3[i];  // add all forces
    if ( drudeHardWallOn && i && (a[i].mass > 0.05) && ((a[i].mass < 1.0)) ) { // drude particle
      if ( ! fixedAtomsOn || ! a[i].atomFixed ) {
        if ( drudeStep2 * f1[i].length2() > drudeMove2 ) {
          a[i].position += drudeMove * f1[i].unit();
        } else {
          a[i].position += drudeStep * f1[i];
        }
        if ( (a[i].position - a[i-1].position).length2() > drudeBondLen2 ) {
          a[i].position = a[i-1].position + drudeBondLen * (a[i].position - a[i-1].position).unit();
        }
      }
      Vector netf = f1[i-1] + f1[i];
      if ( fixedAtomsOn && a[i-1].atomFixed ) netf = 0;
      f1[i-1] = netf;
      f1[i] = 0.;
    }
    if ( fixedAtomsOn && a[i].atomFixed ) f1[i] = 0;
  }

  f2 = f3 = 0;  // included in f1

  BigReal maxv2 = 0.;

  for ( int i = 0; i < numAtoms; ++i ) {
    BigReal v2 = a[i].velocity.length2();
    if ( v2 > 0. ) {
      if ( v2 > maxv2 ) maxv2 = v2;
    } else {
      v2 = f1[i].length2();
      if ( v2 > maxv2 ) maxv2 = v2;
    }
  }

  if ( fmax2 > 10. * TIMEFACTOR * TIMEFACTOR * TIMEFACTOR * TIMEFACTOR )
  { Tensor virial; patch->minimize_rattle2( 0.1 * TIMEFACTOR / sqrt(maxv2), &virial, true /* forces */); }

  BigReal fdotf = 0;
  BigReal fdotv = 0;
  BigReal vdotv = 0;
  int numHuge = 0;
  for ( int i = 0; i < numAtoms; ++i ) {
    if ( simParams->fixedAtomsOn && a[i].atomFixed ) continue;
    if ( drudeHardWallOn && (a[i].mass > 0.05) && ((a[i].mass < 1.0)) ) continue; // drude particle
    Force f = f1[i];
    BigReal ff = f * f;
    if ( ff > fmax2 ) {
      if (simParams->printBadContacts) {
        CkPrintf("STEP(%i) MIN_HUGE[%i] f=%e kcal/mol/A\n",patch->flags.sequence,patch->pExt[i].id,ff);
      }
      ++numHuge;
      // pad scaling so minimizeMoveDownhill() doesn't miss them
      BigReal fmult = 1.01 * sqrt(fmax2/ff);
      f *= fmult;  ff = f * f;
      f1[i] *= fmult;
    }
    fdotf += ff;
    fdotv += f * a[i].velocity;
    vdotv += a[i].velocity * a[i].velocity;
  }

#if 0
  printf("fdotf = %f\n", fdotf);
  printf("fdotv = %f\n", fdotv);
  printf("vdotv = %f\n", vdotv);
#endif
  reduction->item(REDUCTION_MIN_F_DOT_F) += fdotf;
  reduction->item(REDUCTION_MIN_F_DOT_V) += fdotv;
  reduction->item(REDUCTION_MIN_V_DOT_V) += vdotv;
  reduction->item(REDUCTION_MIN_HUGE_COUNT) += numHuge;

  {
    Tensor intVirialNormal;
    Tensor intVirialNbond;
    Tensor intVirialSlow;

    int hgs;
    for ( int i = 0; i < numAtoms; i += hgs ) {
      hgs = a[i].hydrogenGroupSize;
      int j;
      BigReal m_cm = 0;
      Position x_cm(0,0,0);
      for ( j = i; j < (i+hgs); ++j ) {
        m_cm += a[j].mass;
        x_cm += a[j].mass * a[j].position;
      }
      x_cm /= m_cm;
      for ( j = i; j < (i+hgs); ++j ) {
        BigReal mass = a[j].mass;
	// net force treated as zero for fixed atoms
        if ( simParams->fixedAtomsOn && a[j].atomFixed ) continue;
        Vector dx = a[j].position - x_cm;
        intVirialNormal.outerAdd(1.0, patch->f[Results::normal][j], dx);
        intVirialNbond.outerAdd(1.0, patch->f[Results::nbond][j], dx);
        intVirialSlow.outerAdd(1.0, patch->f[Results::slow][j], dx);
      }
    }

    ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_NORMAL,intVirialNormal);
    ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_NBOND,intVirialNbond);
    ADD_TENSOR_OBJECT(reduction,REDUCTION_INT_VIRIAL_SLOW,intVirialSlow);
  }

  if ( simParams->fixedAtomsOn ) {
    Tensor fixVirialNormal;
    Tensor fixVirialNbond;
    Tensor fixVirialSlow;
    Vector fixForceNormal = 0;
    Vector fixForceNbond = 0;
    Vector fixForceSlow = 0;

    calcFixVirial(fixVirialNormal, fixVirialNbond, fixVirialSlow, fixForceNormal, fixForceNbond, fixForceSlow);

    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,fixVirialNormal);
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NBOND,fixVirialNbond);
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_SLOW,fixVirialSlow);
    ADD_VECTOR_OBJECT(reduction,REDUCTION_EXT_FORCE_NORMAL,fixForceNormal);
    ADD_VECTOR_OBJECT(reduction,REDUCTION_EXT_FORCE_NBOND,fixForceNbond);
    ADD_VECTOR_OBJECT(reduction,REDUCTION_EXT_FORCE_SLOW,fixForceSlow);
  }

  reduction->submit();
}

void Sequencer::submitCollections(int step, int zeroVel)
{
  //
  // DJH: Copy updates of SOA back into AOS.
  // Do we need to update everything or is it safe to just update
  // positions and velocities separately, as needed?
  //
  //patch->copy_updates_to_AOS();

  NAMD_EVENT_RANGE_2(patch->flags.event_on,
      NamdProfileEvent::SUBMIT_COLLECTIONS);
  int prec = Output::coordinateNeeded(step);
  if ( prec ) {
    collection->submitPositions(step,patch->atom,patch->lattice,prec);
  }
  if ( Output::velocityNeeded(step) ) {
    collection->submitVelocities(step,zeroVel,patch->atom,prec);
  }
  if ( Output::forceNeeded(step) ) {
    int maxForceUsed = patch->flags.maxForceUsed;
    if ( maxForceUsed > Results::slow ) maxForceUsed = Results::slow;
    collection->submitForces(step,patch->atom,maxForceUsed,patch->f,prec);
  }
}

void Sequencer::runComputeObjects(int migration, int pairlists, int pressureStep)
{
  if ( migration ) pairlistsAreValid = 0;
#if defined(NAMD_CUDA) || defined(NAMD_HIP) || defined(NAMD_MIC)
  if ( pairlistsAreValid &&
       ( patch->flags.doFullElectrostatics || ! simParams->fullElectFrequency )
                         && ( pairlistsAge > pairlistsAgeLimit ) ) {
    pairlistsAreValid = 0;
  }
#else
  if ( pairlistsAreValid && ( pairlistsAge > pairlistsAgeLimit ) ) {
    pairlistsAreValid = 0;
  }
#endif
  if ( ! simParams->usePairlists ) pairlists = 0;
  patch->flags.usePairlists = pairlists || pairlistsAreValid;
  patch->flags.savePairlists =
	pairlists && ! pairlistsAreValid;

  if ( simParams->singleTopology ) patch->reposition_all_alchpairs();
  if ( simParams->lonepairs ) patch->reposition_all_lonepairs();

  //
  // DJH: Copy updates of SOA back into AOS.
  // The positionsReady() routine starts force computation and atom migration.
  //
  // We could reduce amount of copying here by checking migration status
  // and copying velocities only when migrating. Some types of simulation
  // always require velocities, such as Lowe-Anderson.
  //
  //patch->copy_updates_to_AOS();

  patch->positionsReady(migration);  // updates flags.sequence

  int seq = patch->flags.sequence;
  int basePriority = ( (seq & 0xffff) << 15 )
                     + PATCH_PRIORITY(patch->getPatchID());
  if ( patch->flags.doGBIS && patch->flags.doNonbonded) {
    priority = basePriority + GB1_COMPUTE_HOME_PRIORITY;
    suspend(); // until all deposit boxes close
    patch->gbisComputeAfterP1();
    priority = basePriority + GB2_COMPUTE_HOME_PRIORITY;
    suspend();
    patch->gbisComputeAfterP2();
    priority = basePriority + COMPUTE_HOME_PRIORITY;
    suspend();
  } else {
    priority = basePriority + COMPUTE_HOME_PRIORITY;
    suspend(); // until all deposit boxes close
  }

  //
  // DJH: Copy all data into SOA from AOS.
  //
  // We need everything copied after atom migration.
  // When doing force computation without atom migration,
  // all data except forces will already be up-to-date in SOA
  // (except maybe for some special types of simulation).
  //
  //patch->copy_all_to_SOA();

  //
  // DJH: Copy forces to SOA.
  // Force available after suspend() has returned.
  //
  //patch->copy_forces_to_SOA();

  if ( patch->flags.savePairlists && patch->flags.doNonbonded ) {
    pairlistsAreValid = 1;
    pairlistsAge = 0;
  }
  // For multigrator, do not age pairlist during pressure step
  // NOTE: for non-multigrator pressureStep = 0 always
  if ( pairlistsAreValid && !pressureStep ) ++pairlistsAge;

  if (simParams->lonepairs) {
    {
      Tensor virial;
      patch->redistrib_lonepair_forces(Results::normal, &virial);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NORMAL, virial);
    }
    if (patch->flags.doNonbonded) {
      Tensor virial;
      patch->redistrib_lonepair_forces(Results::nbond, &virial);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NBOND, virial);
    }
    if (patch->flags.doFullElectrostatics) {
      Tensor virial;
      patch->redistrib_lonepair_forces(Results::slow, &virial);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_SLOW, virial);
    }
  } else if (simParams->watmodel == WaterModel::TIP4) {
    {
      Tensor virial;
      patch->redistrib_tip4p_forces(Results::normal, &virial);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NORMAL, virial);
    }
    if (patch->flags.doNonbonded) {
      Tensor virial;
      patch->redistrib_tip4p_forces(Results::nbond, &virial);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NBOND, virial);
    }
    if (patch->flags.doFullElectrostatics) {
      Tensor virial;
      patch->redistrib_tip4p_forces(Results::slow, &virial);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_SLOW, virial);
    }
  } else if (simParams->watmodel == WaterModel::SWM4) {
    {
      Tensor virial;
      patch->redistrib_swm4_forces(Results::normal, &virial);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NORMAL, virial);
    }
    if (patch->flags.doNonbonded) {
      Tensor virial;
      patch->redistrib_swm4_forces(Results::nbond, &virial);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NBOND, virial);
    }
    if (patch->flags.doFullElectrostatics) {
      Tensor virial;
      patch->redistrib_swm4_forces(Results::slow, &virial);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_SLOW, virial);
    }
  }

  if (simParams->singleTopology) {
    patch->redistrib_alchpair_forces(Results::normal);
    if (patch->flags.doNonbonded) {
      patch->redistrib_alchpair_forces(Results::nbond);
    }
    if (patch->flags.doFullElectrostatics) {
      patch->redistrib_alchpair_forces(Results::slow);
    }
  }

  if ( patch->flags.doMolly ) {
    Tensor virial;
    patch->mollyMollify(&virial);
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_SLOW,virial);
  }


  // BEGIN LA
  if (patch->flags.doLoweAndersen) {
      patch->loweAndersenFinish();
  }
  // END LA
//TODO:HIP check if this applies to HIP
#ifdef NAMD_CUDA_XXX
  int numAtoms = patch->numAtoms;
  FullAtom *a = patch->atom.begin();
  for ( int i=0; i<numAtoms; ++i ) {
    CkPrintf("%d %g %g %g\n", a[i].id,
        patch->f[Results::normal][i].x +
        patch->f[Results::nbond][i].x +
        patch->f[Results::slow][i].x,
        patch->f[Results::normal][i].y +
        patch->f[Results::nbond][i].y +
        patch->f[Results::slow][i].y,
        patch->f[Results::normal][i].z +
        patch->f[Results::nbond][i].z +
        patch->f[Results::slow][i].z);
    CkPrintf("%d %g %g %g\n", a[i].id,
        patch->f[Results::normal][i].x,
        patch->f[Results::nbond][i].x,
        patch->f[Results::slow][i].x);
    CkPrintf("%d %g %g %g\n", a[i].id,
        patch->f[Results::normal][i].y,
        patch->f[Results::nbond][i].y,
        patch->f[Results::slow][i].y);
    CkPrintf("%d %g %g %g\n", a[i].id,
        patch->f[Results::normal][i].z,
        patch->f[Results::nbond][i].z,
        patch->f[Results::slow][i].z);
  }
#endif

//#undef PRINT_FORCES
//#define PRINT_FORCES 1
#if PRINT_FORCES
  int numAtoms = patch->numAtoms;
  FullAtom *a = patch->atom.begin();
  for ( int i=0; i<numAtoms; ++i ) {
    float fxNo = patch->f[Results::normal][i].x;
    float fxNb = patch->f[Results::nbond][i].x;
    float fxSl = patch->f[Results::slow][i].x;
    float fyNo = patch->f[Results::normal][i].y;
    float fyNb = patch->f[Results::nbond][i].y;
    float fySl = patch->f[Results::slow][i].y;
    float fzNo = patch->f[Results::normal][i].z;
    float fzNb = patch->f[Results::nbond][i].z;
    float fzSl = patch->f[Results::slow][i].z;
    float fx = fxNo+fxNb+fxSl;
    float fy = fyNo+fyNb+fySl;
    float fz = fzNo+fzNb+fzSl;

		float f = sqrt(fx*fx+fy*fy+fz*fz);
    int id = patch->pExt[i].id;
    int seq = patch->flags.sequence;
    float x = patch->p[i].position.x;
    float y = patch->p[i].position.y;
    float z = patch->p[i].position.z;
    //CkPrintf("FORCE(%04i)[%04i] = <% .4e, % .4e, % .4e> <% .4e, % .4e, % .4e> <% .4e, % .4e, % .4e> <<% .4e, % .4e, % .4e>>\n", seq,id,
    CkPrintf("FORCE(%04i)[%04i] = % .9e % .9e % .9e\n", seq,id,
    //CkPrintf("FORCE(%04i)[%04i] = <% .4e, % .4e, % .4e> <% .4e, % .4e, % .4e> <% .4e, % .4e, % .4e>\n", seq,id,
//fxNo,fyNo,fzNo,
fxNb,fyNb,fzNb
//fxSl,fySl,fzSl,
//fx,fy,fz
);
	}
#endif
}

void Sequencer::rebalanceLoad(int timestep) {
  if ( ! ldbSteps ) {
    ldbSteps = LdbCoordinator::Object()->getNumStepsToRun();
  }
  if ( ! --ldbSteps ) {
    patch->submitLoadStats(timestep);
    ldbCoordinator->rebalance(this,patch->getPatchID());
    pairlistsAreValid = 0;
  }
}

void Sequencer::cycleBarrier(int doBarrier, int step) {
#if USE_BARRIER
	if (doBarrier)
          // Blocking receive for the cycle barrier.
	  broadcast->cycleBarrier.get(step);
#endif
}

void Sequencer::traceBarrier(int step){
        // Blocking receive for the trace barrier.
	broadcast->traceBarrier.get(step);
}

#ifdef MEASURE_NAMD_WITH_PAPI
void Sequencer::papiMeasureBarrier(int step){
        // Blocking receive for the PAPI measure barrier.
	broadcast->papiMeasureBarrier.get(step);
}
#endif

void Sequencer::terminate() {
  LdbCoordinator::Object()->pauseWork(patch->ldObjHandle);
  CthFree(thread);
  CthSuspend();
}
