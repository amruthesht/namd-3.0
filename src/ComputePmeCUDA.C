#include <numeric>
#include <algorithm>
#ifdef NAMD_CUDA
#include <cuda_runtime.h>
#endif
#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif
#include "Node.h"
#include "SimParameters.h"
#include "Priorities.h"
#include "ComputeNonbondedUtil.h"
#include "ComputePmeCUDA.h"
#include "ComputePmeCUDAMgr.h"
#include "PmeSolver.h"
#include "HomePatch.h"
#include "PatchData.h"
#include "NamdEventsProfiling.h"
#include "TestArray.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
//
// Class creator, multiple patches
//
ComputePmeCUDA::ComputePmeCUDA(ComputeID c, PatchIDList& pids) : Compute(c) {
  setNumPatches(pids.size());
  patches.resize(getNumPatches());
  for (int i=0;i < getNumPatches();i++) {
    patches[i].patchID = pids[i];
  }
  selfEnergy = 0;
  selfEnergyFEP = 0;
  selfEnergyTI1 = 0;
  selfEnergyTI2 = 0;
}

//
// Class creator, single patch
//
ComputePmeCUDA::ComputePmeCUDA(ComputeID c, PatchID pid) : Compute(c) {
	setNumPatches(1);
  patches.resize(getNumPatches());
  patches[0].patchID = pid;
  selfEnergy = 0;
  selfEnergyFEP = 0;
  selfEnergyTI1 = 0;
  selfEnergyTI2 = 0;
}

//
// Class destructor
//
ComputePmeCUDA::~ComputePmeCUDA() {
  for (int i=0;i < getNumPatches();i++) {
  	if (patches[i].positionBox != NULL) {
  		PatchMap::Object()->patch(patches[i].patchID)->unregisterPositionPickup(this, &patches[i].positionBox);
    }
    if (patches[i].avgPositionBox != NULL) {
    	PatchMap::Object()->patch(patches[i].patchID)->unregisterAvgPositionPickup(this, &patches[i].avgPositionBox);
    }
    if (patches[i].forceBox != NULL) {
    	PatchMap::Object()->patch(patches[i].patchID)->unregisterForceDeposit(this, &patches[i].forceBox);
    }
  }
  delete reduction;
  CmiDestroyLock(lock);
}

//
// Initialize
//
void ComputePmeCUDA::initialize() {
  lock = CmiCreateLock();

  // Sanity Check
  SimParameters *simParams = Node::Object()->simParameters;
  if (simParams->lesOn) NAMD_bug("ComputePmeCUDA::ComputePmeCUDA, lesOn not yet implemented");
  if (simParams->pairInteractionOn) NAMD_bug("ComputePmeCUDA::ComputePmeCUDA, pairInteractionOn not yet implemented");

  sendAtomsDone = false;
  reduction = ReductionMgr::Object()->willSubmit(REDUCTIONS_BASIC);
#ifdef NODEGROUP_FORCE_REGISTER
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();
  nodeReduction = patchData->reduction;
#endif
  // basePriority = PME_PRIORITY;
  patchCounter = getNumPatches();

  // Get proxy to ComputePmeCUDAMgr
  computePmeCUDAMgrProxy = CkpvAccess(BOCclass_group).computePmeCUDAMgr;
  mgr = computePmeCUDAMgrProxy.ckLocalBranch();
  if (mgr == NULL)
    NAMD_bug("ComputePmeCUDA::ComputePmeCUDA, unable to locate local branch of BOC entry computePmeCUDAMgr");
  pmeGrid = mgr->getPmeGrid();

  for (int i=0;i < getNumPatches();i++) {
    if (patches[i].positionBox != NULL || patches[i].avgPositionBox != NULL
      || patches[i].forceBox != NULL || patches[i].patch != NULL)
      NAMD_bug("ComputePmeCUDA::initialize() called twice or boxes not set to NULL");
    if (!(patches[i].patch = PatchMap::Object()->patch(patches[i].patchID))) {
      NAMD_bug("ComputePmeCUDA::initialize() patch not found");
    }
    patches[i].positionBox = patches[i].patch->registerPositionPickup(this);
    patches[i].forceBox = patches[i].patch->registerForceDeposit(this);
  	patches[i].avgPositionBox = patches[i].patch->registerAvgPositionPickup(this);
  }

  setupActivePencils();
}

void ComputePmeCUDA::atomUpdate() {
  atomsChanged = true;
}

//
// Setup, see which pencils overlap with the patches held by this compute
//
void ComputePmeCUDA::setupActivePencils() {
  PatchMap *patchMap = PatchMap::Object();

  for (int i=0;i < getNumPatches();i++) {
    int homey = -1;
    int homez = -1;
    mgr->getHomePencil(patches[i].patchID, homey, homez);

    patches[i].homePencilY = homey;
    patches[i].homePencilZ = homez;
    patches[i].homePencilNode = mgr->getNode(homey,homez);
    RegisterPatchMsg *msg = new RegisterPatchMsg();
    msg->i = homey;
    msg->j = homez;
    computePmeCUDAMgrProxy[patches[i].homePencilNode].registerPatch(msg);
  }

  atomsChanged = true;

}

int ComputePmeCUDA::noWork() {

  if (patches[0].patch->flags.doFullElectrostatics) return 0;

  reduction->submit();

  for (int i=0;i < getNumPatches();i++) {
    patches[i].positionBox->skip();
    patches[i].forceBox->skip();
    // We only need to call skip() once
    if (patches[i].patchID == 0) computePmeCUDAMgrProxy[patches[i].homePencilNode].skip();
  }

  return 1;
}

void ComputePmeCUDA::doWork() {
  NAMD_EVENT_START(1, NamdProfileEvent::COMPUTE_PME_CUDA);

  if (sendAtomsDone) {
    // Second part of computation: receive forces from ComputePmeCUDAMgr
    // basePriority = PME_OFFLOAD_PRIORITY;
    sendAtomsDone = false;
    recvForces();
  } else {
    // First part of computation: send atoms to ComputePmeCUDAMgr
    sendAtomsDone = true;
    // basePriority = COMPUTE_HOME_PRIORITY + PATCH_PRIORITY(patchID);
    sendAtoms();
  }
  NAMD_EVENT_STOP(1, NamdProfileEvent::COMPUTE_PME_CUDA);
}

void ComputePmeCUDA::sendAtoms() {

  Lattice& lattice = patches[0].patch->lattice;
  Vector origin = lattice.origin();
  Vector recip1 = lattice.a_r();
  Vector recip2 = lattice.b_r();
  Vector recip3 = lattice.c_r();
  double ox = origin.x;
  double oy = origin.y;
  double oz = origin.z;
  double r1x = recip1.x;
  double r1y = recip1.y;
  double r1z = recip1.z;
  double r2x = recip2.x;
  double r2y = recip2.y;
  double r2z = recip2.z;
  double r3x = recip3.x;
  double r3y = recip3.y;
  double r3z = recip3.z;

  SimParameters *simParams = Node::Object()->simParameters;
  for (int i=0;i < getNumPatches();i++) {
    if (patches[i].pmeForceMsg != NULL)
      NAMD_bug("ComputePmeCUDA::sendAtoms, pmeForceMsg is not empty");
   
  	const BigReal coulomb_sqrt = sqrt( COULOMB * ComputeNonbondedUtil::scaling
  				     * ComputeNonbondedUtil::dielectric_1 );
    
  	bool doMolly = patches[i].patch->flags.doMolly;
    bool doEnergy = patches[i].patch->flags.doEnergy;
    bool doVirial = patches[i].patch->flags.doVirial;
    PatchMap *patchMap = PatchMap::Object();
    
    // Send atom patch to pencil(s)
    // #ifdef NETWORK_PROGRESS
    //   CmiNetworkProgress();
    // #endif

    CompAtom *x = patches[i].positionBox->open();
    if ( doMolly ) {
      patches[i].positionBox->close(&x);
      x = patches[i].avgPositionBox->open();
    }

    int numAtoms = patches[i].patch->getNumAtoms();

    // FIXME:
    // the computation of self energies makes a bold assumption that charges are never changed
    // this is not true when the system has some kind of chemical interactions, like reduction-oxidation reaction
    // we may need an additional flag to determine whether NAMD is simulating a system with varying atomic charges
    const bool isFirstStep = (patches[0].patch->flags.step == simParams->firstTimestep) || (selfEnergy == 0);
    if (doEnergy) {
      if (simParams->alchFepOn) {
        calcSelfEnergyFEP(numAtoms, x, isFirstStep);
//         fprintf(stdout, "self lambda1 = %lf ; self lambda2 = %lf\n", selfEnergy, selfEnergy_F);
      } else if (simParams->alchThermIntOn) {
        calcSelfEnergyTI(numAtoms, x, isFirstStep);
      } else {
        calcSelfEnergy(numAtoms, x, isFirstStep);
      }
    }

    // const Vector ucenter = patches[i].patch->lattice.unscale(patchMap->center(patches[i].patchID));
    // const BigReal recip11 = patches[i].patch->lattice.a_r().x;
    // const BigReal recip22 = patches[i].patch->lattice.b_r().y;
    // const BigReal recip33 = patches[i].patch->lattice.c_r().z;

//     PmeAtomMsg *msg = new (numAtoms, numAtoms, numAtoms, PRIORITY_SIZE) PmeAtomMsg;
    PmeAtomMsg *msg;
    const int alchGrid = simParams->alchOn ? 1 : 0;
    const int alchDecoupleGrid = simParams->alchDecouple ? 1: 0;
    const int alchSoftCoreOrTI = (simParams->alchElecLambdaStart > 0 || simParams->alchThermIntOn) ? 1 : 0;
    msg = new (numAtoms, alchGrid * numAtoms, alchGrid * numAtoms,
               alchDecoupleGrid * numAtoms,  alchDecoupleGrid * numAtoms,
               alchSoftCoreOrTI * numAtoms, PRIORITY_SIZE) PmeAtomMsg;

    SET_PRIORITY(msg, sequence(), PME_PRIORITY)
    // NOTE:
    // patch already contains the centered coordinates and scaled charges
  	//    memcpy(msg->atoms, patch->getCudaAtomList(), sizeof(CudaAtom)*numAtoms);
    

    msg->numAtoms = numAtoms;
    // msg->patchIndex = i;
    msg->i = patches[i].homePencilY;
    msg->j = patches[i].homePencilZ;
    msg->compute = this;
    msg->pe = CkMyPe();
    msg->doEnergy = doEnergy;
    msg->doVirial = doVirial;
    msg->lattice = lattice;
    msg->simulationStep = patches[0].patch->flags.step;
    CudaAtom *atoms = msg->atoms;
    float *atomChargeFactors1 = msg->chargeFactors1; // normal + appearing atoms
    float *atomChargeFactors2 = msg->chargeFactors2; // normal + disappearing atoms
    float *atomChargeFactors3 = msg->chargeFactors3; // only appearing atoms
    float *atomChargeFactors4 = msg->chargeFactors4; // only disappearing atoms
    float *atomChargeFactors5 = msg->chargeFactors5; // only normal atoms
    // BigReal miny = 1.0e20;
    // BigReal minz = 1.0e20;
    for (int j=0;j < numAtoms;j++) {
    	CudaAtom atom;
        float factor1 = 1.0f;
        float factor2 = 1.0f;
        float factor3 = 1.0f;
        float factor4 = 1.0f;
        float factor5 = 1.0f;
#if 1
      BigReal q = x[j].charge;
      // Convert atom positions to range [0,1)
      double px = x[j].position.x - ox;
      double py = x[j].position.y - oy;
      double pz = x[j].position.z - oz;
      double wx = px*r1x + py*r1y + pz*r1z;
      double wy = px*r2x + py*r2y + pz*r2z;
      double wz = px*r3x + py*r3y + pz*r3z;
      // double wx = x[j].position.x*recip11;
      // double wy = x[j].position.y*recip22;
      // double wz = x[j].position.z*recip33;
      wx = (wx - (floor(wx + 0.5) - 0.5));
      wy = (wy - (floor(wy + 0.5) - 0.5));
      wz = (wz - (floor(wz + 0.5) - 0.5));
      // wx = (wx - floor(wx));
      // wy = (wy - floor(wy));
      // wz = (wz - floor(wz));
      // if (wx >= 1.0) wx -= 1.0;
      // if (wy >= 1.0) wy -= 1.0;
      // if (wz >= 1.0) wz -= 1.0;
      atom.x = (float)wx;
      atom.y = (float)wy;
      atom.z = (float)wz;
      if (atom.x >= 1.0f) atom.x -= 1.0f;
      if (atom.y >= 1.0f) atom.y -= 1.0f;
      if (atom.z >= 1.0f) atom.z -= 1.0f;
    	atom.q = (float)(q*coulomb_sqrt);
      // CHC: for multiple grids, charges shouldn't be scaled here!
      int part = x[j].partition;
      if(simParams->alchOn){
        switch(part){
          case 0: {
              factor1 = 1.0f;
              factor2 = 1.0f;
              if (simParams->alchDecouple) {
                factor3 = 0.0f;
                factor4 = 0.0f;
              }
              if (simParams->alchElecLambdaStart || simParams->alchThermIntOn) {
                factor5 = 1.0f;
              }
              break;
          }
          case 1: {
              factor1 = 1.0f;
              factor2 = 0.0f;
              if (simParams->alchDecouple) {
                factor3 = 1.0f;
                factor4 = 0.0f;
              }
              if (simParams->alchElecLambdaStart || simParams->alchThermIntOn) {
                factor5 = 0.0f;
              }
              break;
          }
          case 2: {
              factor1 = 0.0f;
              factor2 = 1.0f;
              if (simParams->alchDecouple) {
                factor3 = 0.0f;
                factor4 = 1.0f;
              }
              if (simParams->alchElecLambdaStart || simParams->alchThermIntOn) {
                factor5 = 0.0f;
              }
              break;
          }
          default: NAMD_bug("Invalid partition number"); break;
        }
//         atom.q *= elecLambda;
        atomChargeFactors1[j] = factor1;
        atomChargeFactors2[j] = factor2;
        if (simParams->alchDecouple) {
          atomChargeFactors3[j] = factor3;
          atomChargeFactors4[j] = factor4;
        }
        if (simParams->alchElecLambdaStart || simParams->alchThermIntOn) {
          atomChargeFactors5[j] = factor5;
        }
      }
      atoms[j] = atom;
      // miny = std::min(x[j].position.y, miny);
      // minz = std::min(x[j].position.z, minz);
#else
      atom.x = (float) x[j].position.x;
      atom.y = (float) x[j].position.y;
      atom.z = (float) x[j].position.z;
      atom.q = x[j].charge;
#endif
    }
#if defined(NTESTPID)
    if (NTESTPID == patches[i].patch->getPatchID()) {
      char fname[128];
      char remark[128];
      sprintf(fname, "pme_xyzq_soa_pid%d_step%d.bin", NTESTPID,
          patches[i].patch->flags.step);
      sprintf(remark, "SOA PME xyzq, patch %d, step %d", NTESTPID,
          patches[i].patch->flags.step);
      TestArray_write<float>(fname, remark, (float *) atoms, 4*numAtoms);
    }
#endif // NTESTPID
    // Calculate corner with minimum y and z for this patch
    // double wy = miny*recip22;
    // double wz = minz*recip33;
    // msg->miny = (int)((double)pmeGrid.K2*(wy - (floor(wy + 0.5) - 0.5)));
    // msg->minz = (int)((double)pmeGrid.K3*(wz - (floor(wz + 0.5) - 0.5)));

    // For local (within shared memory node), get pointer to memory location and do direct memcpy
    // For global (on different shread memory nodes), 
    if (patches[i].homePencilNode == CkMyNode()) {
      mgr->recvAtoms(msg);
    } else {
      computePmeCUDAMgrProxy[patches[i].homePencilNode].recvAtoms(msg);
    }
    
    if ( doMolly )
      patches[i].avgPositionBox->close(&x);
    else 
      patches[i].positionBox->close(&x);
  }

#ifdef NODEGROUP_FORCE_REGISTER
  // XXX Expect a race condition, need atomic access to nodeReduction
  nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW) += selfEnergy;
  if (simParams->alchFepOn) {
    nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_F) += selfEnergyFEP;
  }
  if (simParams->alchThermIntOn) {
    nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1) += selfEnergyTI1;
    nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2) += selfEnergyTI2;
  }
#endif
  reduction->item(REDUCTION_ELECT_ENERGY_SLOW) += selfEnergy;
  if (simParams->alchFepOn) {
    reduction->item(REDUCTION_ELECT_ENERGY_SLOW_F) += selfEnergyFEP;
  }
  if (simParams->alchThermIntOn) {
    reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1) += selfEnergyTI1;
    reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2) += selfEnergyTI2;
  }
  reduction->submit();
}

//
// Calculate self-energy and send to PmeSolver
//
void ComputePmeCUDA::calcSelfEnergy(int numAtoms, CompAtom *x, bool isFirstStep) {
  // FIXME: check if the atomic charges are constant during the simulation.
  bool constantAtomicCharges = true;
  if (isFirstStep || !constantAtomicCharges) {
    selfEnergy = 0;
    for (int i=0;i < numAtoms;i++) {
      selfEnergy += x[i].charge*x[i].charge;
    }
    //const double SQRT_PI = 1.7724538509055160273; /* mathematica 15 digits*/
    selfEnergy *= -ComputeNonbondedUtil::ewaldcof*COULOMB * ComputeNonbondedUtil::scaling 
              * ComputeNonbondedUtil::dielectric_1 / SQRT_PI;
  }
}

void ComputePmeCUDA::calcSelfEnergyFEP(int numAtoms, CompAtom *atoms, bool isFirstStep) {
  SimParameters *simParams = Node::Object()->simParameters;
  if(simParams->alchFepOn == false){
    NAMD_bug("Called calcSelfEnergyFEP() in non-FEP code!");
  }
  // FIXME: check if the atomic charges are constant during the simulation.
  bool constantAtomicCharges = true;
  BigReal scaleLambda1, scaleLambda2; // scale factors for λ_1 and λ_2
  const  BigReal alchLambda1 = simParams->getCurrentLambda(patches[0].patch->flags.step);
  const  BigReal alchLambda2 = simParams->getCurrentLambda2(patches[0].patch->flags.step);
  static thread_local BigReal lambda1Up   = simParams->getElecLambda(alchLambda1);
  static thread_local BigReal lambda2Up   = simParams->getElecLambda(alchLambda2);
  static thread_local BigReal lambda1Down = simParams->getElecLambda(1.0 - alchLambda1);
  static thread_local BigReal lambda2Down = simParams->getElecLambda(1.0 - alchLambda2);
  if (isFirstStep || !constantAtomicCharges ||
      lambda1Up   != simParams->getElecLambda(alchLambda1) ||
      lambda2Up   != simParams->getElecLambda(alchLambda2) ||
      lambda1Down != simParams->getElecLambda(1.0 - alchLambda1) ||
      lambda2Down != simParams->getElecLambda(1.0 - alchLambda2))
  {
    lambda1Up   = simParams->getElecLambda(alchLambda1);
    lambda2Up   = simParams->getElecLambda(alchLambda2);
    lambda1Down = simParams->getElecLambda(1.0 - alchLambda1);
    lambda2Down = simParams->getElecLambda(1.0 - alchLambda2);
    selfEnergy = 0.0; // self energy for λ_1
    selfEnergyFEP = 0.0; // self energy for λ_2
    for (int i = 0; i < numAtoms; ++i) {
      switch (atoms[i].partition) {
        case 0: {
          scaleLambda1 = 1.0f;
          scaleLambda2 = 1.0f;
          break;
        }
        case 1: {
          scaleLambda1 = lambda1Up; // lambda1 up
          scaleLambda2 = lambda2Up; // lambda2 up
          break;
        }
        case 2: {
          scaleLambda1 = lambda1Down; // lambda1 down
          scaleLambda2 = lambda2Down; // lambda2 down
          break;
        }
        default: NAMD_bug("Invalid partition number");
      }
      selfEnergy    += scaleLambda1 * (atoms[i].charge * atoms[i].charge);
      selfEnergyFEP += scaleLambda2 * (atoms[i].charge * atoms[i].charge);
      if (simParams->alchDecouple) {
        selfEnergy    += (1.0 - scaleLambda1) * (atoms[i].charge * atoms[i].charge);
        selfEnergyFEP += (1.0 - scaleLambda2) * (atoms[i].charge * atoms[i].charge);
      }
      // scale factor for partition 0 is always 1
      // so it's not necessary to compensate the self energy with (lambda1Up + lambda1Down - 1.0) * -1.0
    }
    selfEnergy    *= -ComputeNonbondedUtil::ewaldcof*COULOMB * ComputeNonbondedUtil::scaling * ComputeNonbondedUtil::dielectric_1 / SQRT_PI;
    selfEnergyFEP *= -ComputeNonbondedUtil::ewaldcof*COULOMB * ComputeNonbondedUtil::scaling * ComputeNonbondedUtil::dielectric_1 / SQRT_PI;
  }
}

void ComputePmeCUDA::calcSelfEnergyTI(int numAtoms, CompAtom *atoms, bool isFirstStep) {
  SimParameters *simParams = Node::Object()->simParameters;
  if(simParams->alchThermIntOn == false){
    NAMD_bug("Called calcSelfEnergyTI() in non-FEP code!");
  }
  // FIXME: check if the atomic charges are constant during the simulation.
  bool constantAtomicCharges = true;
  const  BigReal alchLambda1   = simParams->getCurrentLambda(patches[0].patch->flags.step);
  static thread_local BigReal lambda1Up     = simParams->getElecLambda(alchLambda1);
  static thread_local BigReal lambda1Down   = simParams->getElecLambda(1.0 - alchLambda1);
  if (isFirstStep || !constantAtomicCharges ||
      lambda1Up   != simParams->getElecLambda(alchLambda1) ||
      lambda1Down != simParams->getElecLambda(1.0 - alchLambda1))
  {
    lambda1Up   = simParams->getElecLambda(alchLambda1);
    lambda1Down = simParams->getElecLambda(1.0 - alchLambda1);
    selfEnergy    = 0.0;
    selfEnergyTI1 = 0.0;
    selfEnergyTI2 = 0.0;
    BigReal elecLambda1 = 0.0;
    double factor_ti_1  = 0.0;
    double factor_ti_2  = 0.0;
    for (int i = 0; i < numAtoms; ++i) {
      switch (atoms[i].partition) {
        case 0: {
          elecLambda1 = 1.0;
          factor_ti_1 = 0.0;
          factor_ti_2 = 0.0;
          break;
        }
        case 1: {
          elecLambda1 = lambda1Up;
          factor_ti_1 = 1.0;
          factor_ti_2 = 0.0;
          break;
        }
        case 2: {
          elecLambda1 = lambda1Down;
          factor_ti_1 = 0.0;
          factor_ti_2 = 1.0;
          break;
        }
        default: NAMD_bug("Invalid partition number");
      }
      selfEnergy    += elecLambda1 * (atoms[i].charge * atoms[i].charge);
      selfEnergyTI1 += factor_ti_1 * (atoms[i].charge * atoms[i].charge);
      selfEnergyTI2 += factor_ti_2 * (atoms[i].charge * atoms[i].charge);
      if (simParams->alchDecouple) {
        selfEnergy    += (1.0 - elecLambda1) * (atoms[i].charge * atoms[i].charge);
        selfEnergyTI1 -= factor_ti_1 * (atoms[i].charge * atoms[i].charge);
        selfEnergyTI2 -= factor_ti_2 * (atoms[i].charge * atoms[i].charge);
      }
    }
    selfEnergy    *= -ComputeNonbondedUtil::ewaldcof*COULOMB * ComputeNonbondedUtil::scaling * ComputeNonbondedUtil::dielectric_1 / SQRT_PI;
    selfEnergyTI1 *= -ComputeNonbondedUtil::ewaldcof*COULOMB * ComputeNonbondedUtil::scaling * ComputeNonbondedUtil::dielectric_1 / SQRT_PI;
    selfEnergyTI2 *= -ComputeNonbondedUtil::ewaldcof*COULOMB * ComputeNonbondedUtil::scaling * ComputeNonbondedUtil::dielectric_1 / SQRT_PI;
  }
}

void ComputePmeCUDA::recvForces() {

  Lattice& lattice = patches[0].patch->lattice;
  Vector origin = lattice.origin();
  Vector recip1 = lattice.a_r();
  Vector recip2 = lattice.b_r();
  Vector recip3 = lattice.c_r();
  double r1x = recip1.x;
  double r1y = recip1.y;
  double r1z = recip1.z;
  double r2x = recip2.x;
  double r2y = recip2.y;
  double r2z = recip2.z;
  double r3x = recip3.x;
  double r3y = recip3.y;
  double r3z = recip3.z;

  SimParameters *simParams = Node::Object()->simParameters;

  double alchLambda1, lambda1Up, lambda1Down;
  double lambda3Up, lambda3Down;
  if(simParams->alchOn){
    alchLambda1 = simParams->getCurrentLambda(patches[0].patch->flags.step);
    lambda1Up   = simParams->getElecLambda(alchLambda1);
    lambda1Down = simParams->getElecLambda(1 - alchLambda1);
    if (simParams->alchDecouple) {
      lambda3Up   = 1 - lambda1Up;
      lambda3Down = 1 - lambda1Down;
    }
  }

  //fprintf(stderr, "AP recvForces\n");
  for (int i=0;i < getNumPatches();i++) {
    if (patches[i].pmeForceMsg == NULL)
      NAMD_bug("ComputePmeCUDA::recvForces, no message in pmeForceMsg");

    const CudaForce* force = patches[i].pmeForceMsg->force;
    const CudaForce* force2 = patches[i].pmeForceMsg->force2;
    const CudaForce* force3 = patches[i].pmeForceMsg->force3;
    const CudaForce* force4 = patches[i].pmeForceMsg->force4;
    const CudaForce* force5 = patches[i].pmeForceMsg->force5;
    Results *r = patches[i].forceBox->open();
    int numAtoms =  patches[i].pmeForceMsg->numAtoms;
    Force *f = r->f[Results::slow];
#ifdef TESTPID
    if (patches[i].patchID == TESTPID) {
      fprintf(stderr, "Storing scaled PME CudaForce array for patch %d\n",
          patches[i].patchID);
      TestArray_write<float>("scaled_pme_force_good.bin",
          "scaled PME CudaForce good",
          (float *) force, 3*numAtoms);
      TestArray_write<double>("lattice_good.bin", "Lattice good",
          (double *) &lattice, 3*7);
    }
#endif
#if defined(NTESTPID)
    if (NTESTPID == patches[i].patch->getPatchID()) {
      char fname[128];
      char remark[128];
      sprintf(fname, "pme_fxyz_soa_pid%d_step%d.bin", NTESTPID,
          patches[i].patch->flags.step);
      sprintf(remark, "SOA PME fxyz, patch %d, step %d", NTESTPID,
          patches[i].patch->flags.step);
      TestArray_write<float>(fname, remark, (float *) force, 3*numAtoms);
    }
#endif // NTESTPID
    if (!patches[i].pmeForceMsg->numStrayAtoms && !simParams->commOnly) {
      for(int j=0;j < numAtoms;j++) {
        double f1 = 0.0;
        double f2 = 0.0;
        double f3 = 0.0;
        if (simParams->alchOn) {
          f1 += force2[j].x * lambda1Down + force[j].x * lambda1Up;
          f2 += force2[j].y * lambda1Down + force[j].y * lambda1Up;
          f3 += force2[j].z * lambda1Down + force[j].z * lambda1Up;
          if (simParams->alchDecouple) {
            f1 += force3[j].x * lambda3Up + force4[j].x * lambda3Down;
            f2 += force3[j].y * lambda3Up + force4[j].y * lambda3Down;
            f3 += force3[j].z * lambda3Up + force4[j].z * lambda3Down;
          }
          if (bool(simParams->alchElecLambdaStart) || simParams->alchThermIntOn) {
            f1 += force5[j].x * (lambda1Up + lambda1Down - 1.0) * (-1.0);
            f2 += force5[j].y * (lambda1Up + lambda1Down - 1.0) * (-1.0);
            f3 += force5[j].z * (lambda1Up + lambda1Down - 1.0) * (-1.0);
          }
        } else {
          f1 += force[j].x;
          f2 += force[j].y;
          f3 += force[j].z;
        }
        f[j].x += f1*r1x + f2*r2x + f3*r3x;
        f[j].y += f1*r1y + f2*r2y + f3*r3y;
        f[j].z += f1*r1z + f2*r2z + f3*r3z;
      }
    }

#ifdef TESTPID
    if (patches[i].patchID == TESTPID) {
      fprintf(stderr, "Storing slow force array for patch %d\n",
          patches[i].patchID);
      TestArray_write<double>("pme_force_good.bin", "PME force good",
          (double *) f, 3*numAtoms);
    }
#endif

    patches[i].forceBox->close(&r);
    delete patches[i].pmeForceMsg;
    patches[i].pmeForceMsg = NULL;
  }

}

bool ComputePmeCUDA::storePmeForceMsg(PmeForceMsg *msg) {
  bool done = false;
  int i;
  CmiLock(lock);
  patchCounter--;
  i = patchCounter;
  if (patchCounter == 0) {
    patchCounter = getNumPatches();
    done = true;
  }
  CmiUnlock(lock);
  if (patches[i].pmeForceMsg != NULL)
    NAMD_bug("ComputePmeCUDA::storePmeForceMsg, already contains message");
  patches[i].pmeForceMsg = msg;
  return done;
}
#endif // NAMD_CUDA
