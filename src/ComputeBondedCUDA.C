#include <algorithm>   // std::find

#include "charm++.h"
#include "NamdTypes.h"
#include "ComputeMgr.h"
#include "WorkDistrib.h"
#include "ProxyMgr.h"
#include "CudaUtils.h"
#include "DeviceCUDA.h"
#include "ComputeBonds.h"
#include "ComputeAngles.h"
#include "ComputeDihedrals.h"
#include "ComputeImpropers.h"
#include "ComputeCrossterms.h"
#include "ComputeNonbondedCUDAExcl.h"
#include "ComputeBondedCUDA.h"
#include "PatchData.h"
#include "HipDefines.h"
#include "NamdEventsProfiling.h"
#include "CudaRecord.h"

#include "TestArray.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#ifdef WIN32
#define __thread __declspec(thread)
#endif
extern __thread DeviceCUDA *deviceCUDA;

#ifdef BONDED_CUDA


const int ComputeBondedCUDA::CudaTupleTypeSize[Tuples::NUM_TUPLE_TYPES] = {
  sizeof(CudaBond),          // Bonds
  sizeof(CudaAngle),         // Angles
  sizeof(CudaDihedral),      // Dihedrals
  sizeof(CudaDihedral),      // Impropers
  sizeof(CudaExclusion),     // Exclusions
  sizeof(CudaCrossterm)      // Crossterms
};

const int ComputeBondedCUDA::CudaTupleTypeSizeStage[Tuples::NUM_TUPLE_TYPES] = {
  sizeof(CudaBondStage),          // Bonds
  sizeof(CudaAngleStage),         // Angles
  sizeof(CudaDihedralStage),      // Dihedrals
  sizeof(CudaDihedralStage),      // Impropers
  sizeof(CudaExclusionStage),     // Exclusions
  sizeof(CudaCrossterm)      // Crossterms
};

extern "C" void CcdCallBacksReset(void *ignored, double curWallTime);  // fix Charm++

//
// Class constructor
//
ComputeBondedCUDA::ComputeBondedCUDA(ComputeID c, ComputeMgr* computeMgr, int deviceID,
  CudaNonbondedTables& cudaNonbondedTables) :
Compute(c), computeMgr(computeMgr), deviceID(deviceID), masterPe(CkMyPe()),
bondedKernel(deviceID, cudaNonbondedTables)
{

  computes.resize(CkMyNodeSize());
  patchIDsPerRank.resize(CkMyNodeSize());
  numExclPerRank.resize(CkMyNodeSize());
  for (int i=0;i < numExclPerRank.size();i++) {
    numExclPerRank[i].numModifiedExclusions = 0;
    numExclPerRank[i].numExclusions = 0;
  }

  atomMap.allocateMap(Node::Object()->molecule->numAtoms);

  flags = NULL;
  step = 0;

  tupleData = NULL;
  tupleDataSize = 0;

  atoms = NULL;
  atomsSize = 0;

  forces       = NULL;
  forcesSize   = 0;

  energies_virials = NULL;

  initializeCalled = false;

  params = Node::Object()->simParameters;
  accelMDdoDihe = false;
  if (params->accelMDOn) {
     if (params->accelMDdihe || params->accelMDdual) accelMDdoDihe=true;
  }
  /*pswitchTable = {0, 1, 2,
                    1, 1, 99,
                    2, 99, 2};
   */
  pswitchTable[0] = 0; pswitchTable[1] = 1;  pswitchTable[2] = 2;
  pswitchTable[3] = 1; pswitchTable[4] = 1;  pswitchTable[5] = 99;
  pswitchTable[6] = 2; pswitchTable[7] = 99; pswitchTable[8] = 2;

  h_patchRecord = NULL;
  d_patchRecord = NULL;

  h_patchMapCenter = NULL;
  d_patchMapCenter = NULL;
}

//
// Class destructor
//
ComputeBondedCUDA::~ComputeBondedCUDA() {
  cudaCheck(cudaSetDevice(deviceID));

  if (atoms != NULL) deallocate_host<CudaAtom>(&atoms);
  if (forces != NULL) deallocate_host<FORCE_TYPE>(&forces);
  if (energies_virials != NULL) deallocate_host<double>(&energies_virials);
  if (tupleData != NULL) deallocate_host<char>(&tupleData);

  if (initializeCalled) {
    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaEventDestroy(forceDoneEvent));
    CmiDestroyLock(lock);
    CmiDestroyLock(printLock);
    delete reduction;
  }

  if (h_patchMapCenter != NULL) deallocate_host<double3>(&h_patchMapCenter);
  if (d_patchMapCenter != NULL) deallocate_device<double3>(&d_patchMapCenter);

  if (h_patchRecord != NULL) deallocate_host<PatchRecord>(&h_patchRecord);
  if (d_patchRecord != NULL) deallocate_device<PatchRecord>(&d_patchRecord);

  // NOTE: unregistering happens in [sync] -entry method
  computeMgr->sendUnregisterBoxesOnPe(pes, this);
}

void ComputeBondedCUDA::unregisterBoxesOnPe() {
  for (int i=0;i < patchIDsPerRank[CkMyRank()].size();i++) {
    PatchID patchID = patchIDsPerRank[CkMyRank()][i];
    TuplePatchElem* tpe = tuplePatchList.find(TuplePatchElem(patchID));
    if (tpe == NULL || tpe->p == NULL) {
      NAMD_bug("ComputeBondedCUDA::unregisterBoxesOnPe, TuplePatchElem not found or setup incorrectly");
    }
    Patch* patch = tpe->p;
    if (tpe->positionBox != NULL) patch->unregisterPositionPickup(this, &tpe->positionBox);
    if (tpe->avgPositionBox != NULL) patch->unregisterAvgPositionPickup(this, &tpe->avgPositionBox);
    if (tpe->forceBox != NULL) patch->unregisterForceDeposit(this, &tpe->forceBox);
  }
}

//
// Register compute for a given PE. pids is a list of patches the PE has
// Called by master PE
//
void ComputeBondedCUDA::registerCompute(int pe, int type, PatchIDList& pids) {

  if (CkMyPe() != masterPe)
    NAMD_bug("ComputeBondedCUDA::registerCompute() called on non master PE");

  int rank = CkRankOf(pe);

  HomeCompute& homeCompute = computes[rank].homeCompute;
  if (homeCompute.patchIDs.size() == 0) {
    homeCompute.isBasePatch.resize(PatchMap::Object()->numPatches(), 0);
    homeCompute.patchIDs.resize(pids.size());
    for (int i=0;i < pids.size();i++) {
      homeCompute.patchIDs[i] = pids[i];
      homeCompute.isBasePatch[pids[i]] = 1;
    }
  } else {
    if (homeCompute.patchIDs.size() != pids.size()) {
      NAMD_bug("ComputeBondedCUDA::registerCompute(), homeComputes, patch IDs do not match (1)");
    }
    for (int i=0;i < pids.size();i++) {
      if (homeCompute.patchIDs[i] != pids[i]) {
        NAMD_bug("ComputeBondedCUDA::registerCompute(), homeComputes, patch IDs do not match (2)");
      }
    }
  }

  switch(type) {
    case computeBondsType:
    homeCompute.tuples.push_back(new HomeTuples<BondElem, Bond, BondValue>(Tuples::BOND));
    break;

    case computeAnglesType:
    homeCompute.tuples.push_back(new HomeTuples<AngleElem, Angle, AngleValue>(Tuples::ANGLE));
    break;

    case computeDihedralsType:
    homeCompute.tuples.push_back(new HomeTuples<DihedralElem, Dihedral, DihedralValue>(Tuples::DIHEDRAL));
    break;

    case computeImpropersType:
    homeCompute.tuples.push_back(new HomeTuples<ImproperElem, Improper, ImproperValue>(Tuples::IMPROPER));
    break;

    case computeExclsType:
    homeCompute.tuples.push_back(new HomeTuples<ExclElem, Exclusion, int>(Tuples::EXCLUSION));
    break;

    case computeCrosstermsType:
    homeCompute.tuples.push_back(new HomeTuples<CrosstermElem, Crossterm, CrosstermValue>(Tuples::CROSSTERM));
    break;

    default:
    NAMD_bug("ComputeBondedCUDA::registerCompute(), Unsupported compute type");
    break;
  }

}

//
// Register self compute for a given PE
// Called by master PE
//
void ComputeBondedCUDA::registerSelfCompute(int pe, int type, int pid) {

  if (CkMyPe() != masterPe)
    NAMD_bug("ComputeBondedCUDA::registerSelfCompute() called on non master PE");

  int rank = CkRankOf(pe);

  std::vector< SelfCompute >& selfComputes = computes[rank].selfComputes;
  auto it = find(selfComputes.begin(), selfComputes.end(), SelfCompute(type));
  if (it == selfComputes.end()) {
    // Type not found, add new one
    selfComputes.push_back(SelfCompute(type));
    it = selfComputes.begin() + (selfComputes.size() - 1);

    switch(type) {
      case computeSelfBondsType:
      it->tuples = new SelfTuples<BondElem, Bond, BondValue>(Tuples::BOND);
      break;

      case computeSelfAnglesType:
      it->tuples = new SelfTuples<AngleElem, Angle, AngleValue>(Tuples::ANGLE);
      break;

      case computeSelfDihedralsType:
      it->tuples = new SelfTuples<DihedralElem, Dihedral, DihedralValue>(Tuples::DIHEDRAL);
      break;

      case computeSelfImpropersType:
      it->tuples = new SelfTuples<ImproperElem, Improper, ImproperValue>(Tuples::IMPROPER);
      break;

      case computeSelfExclsType:
      it->tuples = new SelfTuples<ExclElem, Exclusion, int>(Tuples::EXCLUSION);
      break;

      case computeSelfCrosstermsType:
      it->tuples = new SelfTuples<CrosstermElem, Crossterm, CrosstermValue>(Tuples::CROSSTERM);
      break;

      default:
      NAMD_bug("ComputeBondedCUDA::registerSelfCompute(), Unsupported compute type");
      break;
    }

  }

  // Add patch ID for this type
  it->patchIDs.push_back(pid);
}

void ComputeBondedCUDA::assignPatchesOnPe() {

  PatchMap* patchMap = PatchMap::Object();
  for (int i=0;i < patchIDsPerRank[CkMyRank()].size();i++) {
    PatchID patchID = patchIDsPerRank[CkMyRank()][i];
    ProxyMgr::Object()->createProxy(patchID);
    Patch* patch = patchMap->patch(patchID);
    if (patch == NULL)
      NAMD_bug("ComputeBondedCUDA::assignPatchesOnPe, patch not found");
    if (flags == NULL) flags = &patchMap->patch(patchID)->flags;
    TuplePatchElem* tpe = tuplePatchList.find(TuplePatchElem(patchID));
    if (tpe == NULL) {
      NAMD_bug("ComputeBondedCUDA::assignPatchesOnPe, TuplePatchElem not found");
    }
    if (tpe->p != NULL) {
      NAMD_bug("ComputeBondedCUDA::assignPatchesOnPe, TuplePatchElem already registered");
    }
    // Assign patch and register coordinates and forces manually
    tpe->p = patch;
    tpe->positionBox = patch->registerPositionPickup(this);
    tpe->avgPositionBox = patch->registerAvgPositionPickup(this);
    tpe->forceBox = patch->registerForceDeposit(this);
  }
}

//
// atomUpdate() can be called by any Pe
//
void ComputeBondedCUDA::atomUpdate() {
  atomsChangedIn = true;
}

//
// Enqueu doWork on masterPe and return "no work"
// Can be called by any Pe
//
int ComputeBondedCUDA::noWork() {
  computeMgr->sendMessageEnqueueWork(masterPe, this);
  return 1;
}

void ComputeBondedCUDA::messageEnqueueWork() {
  if (masterPe != CkMyPe())
    NAMD_bug("ComputeBondedCUDA::messageEnqueueWork() must be called from master PE");
  WorkDistrib::messageEnqueueWork(this);
}

//
// Sends open-box commands to PEs
// Called on master PE
//
void ComputeBondedCUDA::doWork() {
  if (CkMyPe() != masterPe)
    NAMD_bug("ComputeBondedCUDA::doWork() called on non master PE");

  // Read value of atomsChangedIn, which is set in atomUpdate(), and reset it.
  // atomsChangedIn can be set to true by any Pe
  // atomsChanged can only be set by masterPe
  // This use of double varibles makes sure we don't have race condition
  atomsChanged = atomsChangedIn;
  atomsChangedIn = false;

  if (getNumPatches() == 0) {
    return;  // No work do to
  }

  if (flags == NULL)
    NAMD_bug("ComputeBondedCUDA::doWork(), no flags set");

  // Read flags
  // what is flags...
  lattice  = flags->lattice;
  doEnergy = flags->doEnergy;
  doVirial = flags->doVirial;
  doSlow   = flags->doFullElectrostatics;
  doMolly  = flags->doMolly;
  step     = flags->step;

  if (hostAlchFlags.alchOn) {
    updateHostCudaAlchLambdas();
    updateKernelCudaAlchLambdas();
    const int& alchOutFreq = Node::Object()->simParameters->alchOutFreq;
    if (alchOutFreq > 0 && (step % alchOutFreq == 0)) {
      doEnergy = true;
    }
  }

  if (atomsChanged) {
    // Re-calculate patch atom numbers and storage
    updatePatches();
  }
  // Open boxes on Pes and launch work to masterPe
  if(params->CUDASOAintegrate) {
    if (!atomsChanged) this->openBoxesOnPe();
  }
  else computeMgr->sendOpenBoxesOnPe(pes, this);
}

//
// This gets called when patch finishes on a PE
//
void ComputeBondedCUDA::patchReady(PatchID pid, int doneMigration, int seq) {
  if (doneMigration) {
    // auto it = patchIndex.find(pid);
    // if (it == patchIndex.end())
    //   NAMD_bug("ComputeBondedCUDA::patchReady, Patch ID not found");
    patches[patchIndex[pid]].numAtoms = PatchMap::Object()->patch(pid)->getNumAtoms();
#ifdef NODEGROUP_FORCE_REGISTER
    patches[patchIndex[pid]].patchID = pid;
#endif
  }
  // XXX first sum locally, then sum globally
  // DMC: This isn't need into CUDASOAintegrate scheme. All it does is call atomUpdate()
  // however that is already called in Sequencer::runComputeObjects_CUDA
  if (!params->CUDASOAintegrate || !params->useDeviceMigration) {
    CmiLock(lock);
    Compute::patchReady(pid, doneMigration, seq);
    CmiUnlock(lock);
  }
}

//
//
//
void ComputeBondedCUDA::updatePatches() {
  if (!Node::Object()->simParameters->CUDASOAintegrate) {
    int atomStart = 0;
    for (int i=0;i < patches.size();i++) {
      patches[i].atomStart = atomStart;
      atomStart += patches[i].numAtoms;
    }
    atomStorageSize = atomStart;

    // Re-allocate atoms
    reallocate_host<CudaAtom>(&atoms, &atomsSize, atomStorageSize, 1.4f);

  } else {
#ifdef NODEGROUP_FORCE_REGISTER
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    PatchData *patchData = cpdata.ckLocalBranch();

    std::vector<CudaLocalRecord>& localPatches = 
      patchData->devData[deviceIndex].h_localPatches;
    const int numPatchesHomeAndProxy = 
      patchData->devData[deviceIndex].numPatchesHomeAndProxy;

    int atomStart = 0;
    for (int i=0;i < numPatchesHomeAndProxy; i++) {
      patches[i].numAtoms = localPatches[i].numAtoms;
      patches[i].atomStart = localPatches[i].bufferOffset;
      atomStart += patches[i].numAtoms;
    }

    atomStorageSize = atomStart;
    reallocate_host<CudaAtom>(&atoms, &atomsSize, atomStorageSize, 1.4f);

    if (params->CUDASOAintegrate && params->useDeviceMigration) {
      bondedKernel.updateAtomBuffer(atomStorageSize, stream);
      updatePatchRecords();
    }
#endif  // NODEGROUP_FORCE_REGISTER
  }
}

//
// Map atoms GPU-wide
//
void ComputeBondedCUDA::mapAtoms() {
  for (int i=0;i < getNumPatches();i++) {
    TuplePatchElem* tpe = tuplePatchList.find(TuplePatchElem(allPatchIDs[i]));
    atomMappers[i]->registerIDsCompAtomExt(tpe->xExt, tpe->xExt + tpe->p->getNumAtoms());
  }

}

//
// Unmap atoms GPU-wide
//
void ComputeBondedCUDA::unmapAtoms() {
  for (int i=0;i < getNumPatches();i++) {
    TuplePatchElem* tpe = tuplePatchList.find(TuplePatchElem(allPatchIDs[i]));
    atomMappers[i]->unregisterIDsCompAtomExt(tpe->xExt, tpe->xExt + tpe->p->getNumAtoms());
  }
}

//
// Open all patches that have been assigned to this Pe
//
void ComputeBondedCUDA::openBoxesOnPe(int startup) {
  NAMD_EVENT_START(1, NamdProfileEvent::COMPUTE_BONDED_CUDA_OPEN_BOXES);
  std::vector<int>& patchIDs = patchIDsPerRank[CkMyRank()];
#if 0
  fprintf(stderr, "PE[%d] calling ComputeBondedCUDA::openBoxesOnePE(%p)\n", CkMyPe(), this);
#endif
#ifdef NODEGROUP_FORCE_REGISTER
  if( Node::Object()->simParameters->CUDASOAintegrate && !atomsChanged) {
      for (auto it=patchIDs.begin();it != patchIDs.end();it++) {
         PatchID patchID = *it;
         TuplePatchElem* tpe = tuplePatchList.find(TuplePatchElem(patchID));
         tpe->x = tpe->positionBox->open();
         tpe->r = tpe->forceBox->open();
         //tpe->f = tpe->r->f[Results::normal];
      }
      this->launchWork();
  }
  else{
#endif
  for (auto it=patchIDs.begin();it != patchIDs.end();it++) {
    PatchID patchID = *it;
    TuplePatchElem* tpe = tuplePatchList.find(TuplePatchElem(patchID));
    tpe->x = tpe->positionBox->open();
    tpe->xExt = tpe->p->getCompAtomExtInfo();
    //fprintf(stderr, "::openBoxesOnPe(%p) PE[%d] step %d PID[%d] = atomExt = %p\n", this, CkMyPe(), tpe->p->flags.step, tpe->p->getPatchID(), tpe->xExt);
    if ( doMolly ) tpe->x_avg = tpe->avgPositionBox->open();
    tpe->r = tpe->forceBox->open();
    tpe->f = tpe->r->f[Results::normal];
    if (accelMDdoDihe) tpe->af = tpe->r->f[Results::amdf]; // for dihedral-only or dual-boost accelMD
    // Copy atoms
    if (!params->CUDASOAintegrate || !params->useDeviceMigration) {
      int pi = patchIndex[patchID];
      int atomStart = patches[pi].atomStart;
      int numAtoms = patches[pi].numAtoms;
      CompAtom* compAtom = tpe->x;
      const CompAtomExt *aExt = tpe->p->getCompAtomExtInfo();
      const CudaAtom *src = tpe->p->getCudaAtomList();
      for (int i=0;i < numAtoms;i++) {
        int j = aExt[i].sortOrder;
        // We have an atomStart here, and J is the sortOrder: unsorting atoms
        atoms[atomStart + j] = src[i];
      }
    }
  }
  bool done = false;
  // NOTE: this whole scheme of counting down the patches will help
  //        when I have multiple masterPes
  CmiLock(lock);
  patchesCounter -= patchIDs.size();
  if (patchesCounter == 0) {
    patchesCounter = getNumPatches();
    done = true;
  }
  CmiUnlock(lock);
  if (done) {
    if (atomsChanged) {
      if (!params->CUDASOAintegrate || !params->useDeviceMigration || startup) {
        mapAtoms();
      }
      //computeMgr->sendLoadTuplesOnPe(pes, this);
      if(!params->CUDASOAintegrate) computeMgr->sendLoadTuplesOnPe(pes, this);
    } else {
      //computeMgr->sendLaunchWork(masterPe, this);
      if(params->CUDASOAintegrate){
         if(!atomsChanged) this->launchWork();
      }
      else computeMgr->sendLaunchWork(masterPe, this);
    }
  }
#ifdef NODEGROUP_FORCE_REGISTER
  }
#endif
  NAMD_EVENT_STOP(1, NamdProfileEvent::COMPUTE_BONDED_CUDA_OPEN_BOXES);

#if 0
  // CmiNodeBarrier();
  // Patches are using different
  CmiLock(printLock);
  fprintf(stderr, "PE[%d] (%p) tuplePatchList printout\n", CkMyPe(), this);
  for(int i = 0 ; i < tuplePatchList.size(); i++){
    // how the fuck do I print this structure. argh
    TuplePatchElem *tpe = tuplePatchList.find(TuplePatchElem(i));
    if(tpe == NULL) break;
    fprintf(stderr, "PE[%d] (%p) %d PID[%d] atomExt = %p\n",CkMyPe(), this, i, tpe->p->getPatchID(), tpe->xExt);
  }
  CmiUnlock(printLock);
  // CmiNodeBarrier(); // Let's see
#endif
}

void countNumExclusions(Tuples* tuples, int& numModifiedExclusions, int& numExclusions) {
  numModifiedExclusions = 0;
  int ntuples = tuples->getNumTuples();
  ExclElem* src = (ExclElem *)(tuples->getTupleList());
  for (int ituple=0;ituple < ntuples;ituple++) {
    if (src[ituple].modified) numModifiedExclusions++;
  }
  numExclusions = ntuples - numModifiedExclusions;
}

//
// Load tuples on PE. Note: this can only after boxes on all PEs have been opened
//
void ComputeBondedCUDA::loadTuplesOnPe(const int startup) {
  NAMD_EVENT_START(1, NamdProfileEvent::COMPUTE_BONDED_LOAD_TUPLES);

  int numModifiedExclusions = 0;
  int numExclusions = 0;
  if (startup || (!params->CUDASOAintegrate || !params->useDeviceMigration)) {
    std::vector< SelfCompute >& selfComputes = computes[CkMyRank()].selfComputes;
    // Loop over self compute types
    for (auto it=selfComputes.begin();it != selfComputes.end();it++) {
      // clears tupleList and reloads the tuples
      it->tuples->loadTuples(tuplePatchList, NULL, &atomMap, it->patchIDs);
      // For exclusions, we must count the number of modified and non-modified exclusions
      if (it->tuples->getType() == Tuples::EXCLUSION) {
        int tmp1, tmp2;
        countNumExclusions(it->tuples, tmp1, tmp2);
        numModifiedExclusions += tmp1;
        numExclusions += tmp2;
      }
    }

    HomeCompute& homeCompute = computes[CkMyRank()].homeCompute;
    for (int i=0;i < homeCompute.tuples.size();i++) {
      homeCompute.tuples[i]->loadTuples(tuplePatchList,
        homeCompute.isBasePatch.data(), &atomMap,
        homeCompute.patchIDs);
      // For exclusions, we must count the number of modified and non-modified exclusions
      if (homeCompute.tuples[i]->getType() == Tuples::EXCLUSION) {
        int tmp1, tmp2;
        countNumExclusions(homeCompute.tuples[i], tmp1, tmp2);
        numModifiedExclusions += tmp1;
        numExclusions += tmp2;
      }
    }
  } else {
    numModifiedExclusions = modifiedExclusionTupleData.size();
    numExclusions = exclusionTupleData.size();
  }

  // Store number of exclusions
  numExclPerRank[CkMyRank()].numModifiedExclusions = numModifiedExclusions;
  numExclPerRank[CkMyRank()].numExclusions         = numExclusions;


  if (!params->CUDASOAintegrate || !params->useDeviceMigration) {
    bool done = false;
    
    // TODO: Swap that for a more efficient atomic operation
    // Think of a lock-free design
    CmiLock(lock);
    patchesCounter -= patchIDsPerRank[CkMyRank()].size();
    if (patchesCounter == 0) {
      patchesCounter = getNumPatches();
      done = true;
    }
    CmiUnlock(lock);
    if (done) {
      //computeMgr->sendLaunchWork(masterPe, this);
      //if(params->CUDASOAintegrate && !first) this->launchWork();
      //else computeMgr->sendLaunchWork(masterPe, this);
      if(!params->CUDASOAintegrate)computeMgr->sendLaunchWork(masterPe, this);
    }
  }

  NAMD_EVENT_STOP(1, NamdProfileEvent::COMPUTE_BONDED_LOAD_TUPLES);
}

void ComputeBondedCUDA::copyBondData(const int ntuples, const BondElem* __restrict__ src,
  const BondValue* __restrict__ bond_array, CudaBond* __restrict__ dst) {

  PatchMap* patchMap = PatchMap::Object();
  for (int ituple=0;ituple < ntuples;ituple++) {
    CudaBond dstval;
    auto p0 = src[ituple].p[0];
    auto p1 = src[ituple].p[1];
    int pi0 = patchIndex[p0->patchID];
    int pi1 = patchIndex[p1->patchID];
    int l0 = src[ituple].localIndex[0];
    int l1 = src[ituple].localIndex[1];
    dstval.i = l0 + patches[pi0].atomStart;
    dstval.j = l1 + patches[pi1].atomStart;
    dstval.itype = (src[ituple].value - bond_array);
    Position position1 = p0->x[l0].position;
    Position position2 = p1->x[l1].position;
    Vector shiftVec = lattice.wrap_delta_scaled(position1, position2);
    if(pi0 != pi1) shiftVec += patchMap->center(p0->patchID) - patchMap->center(p1->patchID);
    dstval.ioffsetXYZ = make_float3((float)shiftVec.x, (float)shiftVec.y, (float)shiftVec.z);
    dstval.scale = src[ituple].scale;
    if (hostAlchFlags.alchOn) {
      const AtomID (&atomID)[sizeof(src[ituple].atomID)/sizeof(AtomID)](src[ituple].atomID);
      const Molecule& mol = *(Node::Object()->molecule);
      dstval.fepBondedType = mol.get_fep_bonded_type(atomID, 2);
    } else {
      dstval.fepBondedType = 0;
    }
    dst[ituple] = dstval;
  }
}

#ifdef NODEGROUP_FORCE_REGISTER
template<>
void ComputeBondedCUDA::copyTupleToStage(const BondElem& src,
  const BondValue* __restrict__ p_array, CudaBondStage& dstval) {

  auto p0 = src.p[0];
  auto p1 = src.p[1];
  int pi0 = patchIndex[p0->patchID];
  int pi1 = patchIndex[p1->patchID];
  int l0 = src.localIndex[0];
  int l1 = src.localIndex[1];
  dstval.itype = (src.value - p_array);
  dstval.scale = src.scale;
  dstval.patchIDs[0] = p0->patchID;
  dstval.patchIDs[1] = p1->patchID;

  dstval.index[0] = l0;
  dstval.index[1] = l1;
  
  if (hostAlchFlags.alchOn) {
    const AtomID (&atomID)[sizeof(src.atomID)/sizeof(AtomID)](src.atomID);
    const Molecule& mol = *(Node::Object()->molecule);
    dstval.fepBondedType = mol.get_fep_bonded_type(atomID, 2);
  } else {
    dstval.fepBondedType = 0;
  }
}
#endif  // NODEGROUP_FORCE_REGISTER

// XXX NOTE: Modified FP32 version
void ComputeBondedCUDA::copyBondDatafp32(const int ntuples, const BondElem* __restrict__ src,
  const BondValue* __restrict__ bond_array, CudaBond* __restrict__ dst) {

  PatchMap* patchMap = PatchMap::Object();
  float3 b1f, b2f, b3f;
  b1f = make_float3(lattice.a_r().x, lattice.a_r().y, lattice.a_r().z);
  b2f = make_float3(lattice.b_r().x, lattice.b_r().y, lattice.b_r().z);
  b3f = make_float3(lattice.c_r().x, lattice.c_r().y, lattice.c_r().z);

  for (int ituple=0;ituple < ntuples;ituple++) {
    CudaBond dstval;
    auto p0 = src[ituple].p[0];
    auto p1 = src[ituple].p[1];
    int pi0 = patchIndex[p0->patchID];
    int pi1 = patchIndex[p1->patchID];
    int l0 = src[ituple].localIndex[0];
    int l1 = src[ituple].localIndex[1];
    dstval.i = l0 + patches[pi0].atomStart;
    dstval.j = l1 + patches[pi1].atomStart;
    dstval.itype = (src[ituple].value - bond_array);
#if 0
    Position position1 = p0->x[l0].position;
    Position position2 = p1->x[l1].position;
    Vector shiftVec = lattice.wrap_delta_scaled_fast(position1, position2);
    if(pi0 != pi1) shiftVec += patchMap->center(p0->patchID) - patchMap->center(p1->patchID);
#endif
    float3 position1 = make_float3(p0->x[l0].position.x, p0->x[l0].position.y, p0->x[l0].position.z);
    float3 position2 = make_float3(p1->x[l1].position.x, p1->x[l1].position.y, p1->x[l1].position.z);
    float3 diff = position1 - position2;
    float d1 = -floorf(b1f.x * diff.x + b1f.y * diff.y + b1f.z * diff.z + 0.5f);
    float d2 = -floorf(b2f.x * diff.x + b2f.y * diff.y + b2f.z * diff.z + 0.5f);
    float d3 = -floorf(b3f.x * diff.x + b3f.y * diff.y + b3f.z * diff.z + 0.5f);
    if(pi0 != pi1){
      Vector c = patchMap->center(p0->patchID) - patchMap->center(p1->patchID);
      d1 += c.x;
      d2 += c.y;
      d3 += c.z;
    }
    dstval.ioffsetXYZ = make_float3(d1, d2, d3);
    dstval.scale = src[ituple].scale;
    if (hostAlchFlags.alchOn) {
      const AtomID (&atomID)[sizeof(src[ituple].atomID)/sizeof(AtomID)](src[ituple].atomID);
      const Molecule& mol = *(Node::Object()->molecule);
      dstval.fepBondedType = mol.get_fep_bonded_type(atomID, 2);
    } else {
      dstval.fepBondedType = 0;
    }
    dst[ituple] = dstval;
  }
}

void ComputeBondedCUDA::copyAngleData(const int ntuples, const AngleElem* __restrict__ src,
  const AngleValue* __restrict__ angle_array, CudaAngle* __restrict__ dst) {
  PatchMap* patchMap = PatchMap::Object();
  for (int ituple=0;ituple < ntuples;ituple++) {
    CudaAngle dstval;
    auto p0 = src[ituple].p[0];
    auto p1 = src[ituple].p[1];
    auto p2 = src[ituple].p[2];
    int pi0 = patchIndex[p0->patchID];
    int pi1 = patchIndex[p1->patchID];
    int pi2 = patchIndex[p2->patchID];
    int l0 = src[ituple].localIndex[0];
    int l1 = src[ituple].localIndex[1];
    int l2 = src[ituple].localIndex[2];
    dstval.i = l0 + patches[pi0].atomStart;
    dstval.j = l1 + patches[pi1].atomStart;
    dstval.k = l2 + patches[pi2].atomStart;
    dstval.itype = (src[ituple].value - angle_array);
    Position position1 = p0->x[l0].position;
    Position position2 = p1->x[l1].position;
    Position position3 = p2->x[l2].position;
    Vector shiftVec12 = lattice.wrap_delta_scaled(position1, position2);
    Vector shiftVec32 = lattice.wrap_delta_scaled(position3, position2);
    if(pi0 != pi1) shiftVec12 += patchMap->center(p0->patchID) - patchMap->center(p1->patchID);
    if(pi2 != pi1) shiftVec32 += patchMap->center(p2->patchID) - patchMap->center(p1->patchID);

    dstval.ioffsetXYZ = make_float3((float)shiftVec12.x, (float)shiftVec12.y, (float)shiftVec12.z);
    dstval.koffsetXYZ = make_float3((float)shiftVec32.x, (float)shiftVec32.y, (float)shiftVec32.z);
    dstval.scale = src[ituple].scale;
    if (hostAlchFlags.alchOn) {
      const AtomID (&atomID)[sizeof(src[ituple].atomID)/sizeof(AtomID)](src[ituple].atomID);
      const Molecule& mol = *(Node::Object()->molecule);
      dstval.fepBondedType = mol.get_fep_bonded_type(atomID, 3);
    } else {
      dstval.fepBondedType = 0;
    }
    dst[ituple] = dstval;
  }
}

#ifdef NODEGROUP_FORCE_REGISTER
template<>
void ComputeBondedCUDA::copyTupleToStage(const AngleElem& src,
  const AngleValue* __restrict__ p_array, CudaAngleStage& dstval) {

  auto p0 = src.p[0];
  auto p1 = src.p[1];
  auto p2 = src.p[2];
  int pi0 = patchIndex[p0->patchID];
  int pi1 = patchIndex[p1->patchID];
  int pi2 = patchIndex[p2->patchID];
  int l0 = src.localIndex[0];
  int l1 = src.localIndex[1];
  int l2 = src.localIndex[2];
  dstval.itype = (src.value - p_array);

  dstval.patchIDs[0] = p0->patchID;
  dstval.patchIDs[1] = p1->patchID;
  dstval.patchIDs[2] = p2->patchID;

  dstval.index[0] = l0;
  dstval.index[1] = l1;
  dstval.index[2] = l2;

  dstval.scale = src.scale;
  if (hostAlchFlags.alchOn) {
    const AtomID (&atomID)[sizeof(src.atomID)/sizeof(AtomID)](src.atomID);
    const Molecule& mol = *(Node::Object()->molecule);
    dstval.fepBondedType = mol.get_fep_bonded_type(atomID, 3);
  } else {
    dstval.fepBondedType = 0;
  }
}
#endif  // NODEGROUP_FORCE_REGISTER

//
// Used for both dihedrals and impropers
//
template <bool doDihedral, typename T, typename P>
void ComputeBondedCUDA::copyDihedralData(const int ntuples, const T* __restrict__ src,
  const P* __restrict__ p_array, CudaDihedral* __restrict__ dst) {

  PatchMap* patchMap = PatchMap::Object();

  for (int ituple=0;ituple < ntuples;ituple++) {
    CudaDihedral dstval;
    auto p0 = src[ituple].p[0];
    auto p1 = src[ituple].p[1];
    auto p2 = src[ituple].p[2];
    auto p3 = src[ituple].p[3];
    int pi0 = patchIndex[p0->patchID];
    int pi1 = patchIndex[p1->patchID];
    int pi2 = patchIndex[p2->patchID];
    int pi3 = patchIndex[p3->patchID];
    int l0 = src[ituple].localIndex[0];
    int l1 = src[ituple].localIndex[1];
    int l2 = src[ituple].localIndex[2];
    int l3 = src[ituple].localIndex[3];
    dstval.i = l0 + patches[pi0].atomStart;
    dstval.j = l1 + patches[pi1].atomStart;
    dstval.k = l2 + patches[pi2].atomStart;
    dstval.l = l3 + patches[pi3].atomStart;
    if (doDihedral) {
      dstval.itype = dihedralMultMap[(src[ituple].value - p_array)];
    } else {
      dstval.itype = improperMultMap[(src[ituple].value - p_array)];
    }
    Position position1 = p0->x[l0].position;
    Position position2 = p1->x[l1].position;
    Position position3 = p2->x[l2].position;
    Position position4 = p3->x[l3].position;
    Vector shiftVec12 = lattice.wrap_delta_scaled(position1, position2);
    Vector shiftVec23 = lattice.wrap_delta_scaled(position2, position3);
    Vector shiftVec43 = lattice.wrap_delta_scaled(position4, position3);
    if(pi0 != pi1) shiftVec12 += patchMap->center(p0->patchID) - patchMap->center(p1->patchID);
    if(pi1 != pi2) shiftVec23 += patchMap->center(p1->patchID) - patchMap->center(p2->patchID);
    if(pi3 != pi2) shiftVec43 += patchMap->center(p3->patchID) - patchMap->center(p2->patchID);

    dstval.ioffsetXYZ = make_float3((float)shiftVec12.x, (float)shiftVec12.y, (float)shiftVec12.z);
    dstval.joffsetXYZ = make_float3((float)shiftVec23.x, (float)shiftVec23.y, (float)shiftVec23.z);
    dstval.loffsetXYZ = make_float3((float)shiftVec43.x, (float)shiftVec43.y, (float)shiftVec43.z);
    dstval.scale = src[ituple].scale;
    if (hostAlchFlags.alchOn) {
      const AtomID (&atomID)[sizeof(src[ituple].atomID)/sizeof(AtomID)](src[ituple].atomID);
      const Molecule& mol = *(Node::Object()->molecule);
      dstval.fepBondedType = mol.get_fep_bonded_type(atomID, 4);
    } else {
      dstval.fepBondedType = 0;
    }
    dst[ituple] = dstval;
  }
}

#ifdef NODEGROUP_FORCE_REGISTER
template <>
void ComputeBondedCUDA::copyTupleToStage(const DihedralElem& src,
  const DihedralValue* __restrict__ p_array, CudaDihedralStage& dstval) {

  auto p0 = src.p[0];
  auto p1 = src.p[1];
  auto p2 = src.p[2];
  auto p3 = src.p[3];
  int pi0 = patchIndex[p0->patchID];
  int pi1 = patchIndex[p1->patchID];
  int pi2 = patchIndex[p2->patchID];
  int pi3 = patchIndex[p3->patchID];
  int l0 = src.localIndex[0];
  int l1 = src.localIndex[1];
  int l2 = src.localIndex[2];
  int l3 = src.localIndex[3];
  dstval.itype = dihedralMultMap[(src.value - p_array)];

  dstval.patchIDs[0] = p0->patchID;
  dstval.patchIDs[1] = p1->patchID;
  dstval.patchIDs[2] = p2->patchID;
  dstval.patchIDs[3] = p3->patchID;

  dstval.index[0] = l0;
  dstval.index[1] = l1;
  dstval.index[2] = l2;
  dstval.index[3] = l3;

  dstval.scale = src.scale;
  if (hostAlchFlags.alchOn) {
    const AtomID (&atomID)[sizeof(src.atomID)/sizeof(AtomID)](src.atomID);
    const Molecule& mol = *(Node::Object()->molecule);
    dstval.fepBondedType = mol.get_fep_bonded_type(atomID, 4);
  } else {
    dstval.fepBondedType = 0;
  }
}

template <>
void ComputeBondedCUDA::copyTupleToStage(const ImproperElem& src,
  const ImproperValue* __restrict__ p_array, CudaDihedralStage& dstval) {

  auto p0 = src.p[0];
  auto p1 = src.p[1];
  auto p2 = src.p[2];
  auto p3 = src.p[3];
  int pi0 = patchIndex[p0->patchID];
  int pi1 = patchIndex[p1->patchID];
  int pi2 = patchIndex[p2->patchID];
  int pi3 = patchIndex[p3->patchID];
  int l0 = src.localIndex[0];
  int l1 = src.localIndex[1];
  int l2 = src.localIndex[2];
  int l3 = src.localIndex[3];
  dstval.itype = improperMultMap[(src.value - p_array)];

  dstval.patchIDs[0] = p0->patchID;
  dstval.patchIDs[1] = p1->patchID;
  dstval.patchIDs[2] = p2->patchID;
  dstval.patchIDs[3] = p3->patchID;

  dstval.index[0] = l0;
  dstval.index[1] = l1;
  dstval.index[2] = l2;
  dstval.index[3] = l3;

  dstval.scale = src.scale;
  if (hostAlchFlags.alchOn) {
    const AtomID (&atomID)[sizeof(src.atomID)/sizeof(AtomID)](src.atomID);
    const Molecule& mol = *(Node::Object()->molecule);
    dstval.fepBondedType = mol.get_fep_bonded_type(atomID, 4);
  } else {
    dstval.fepBondedType = 0;
  }
}

template <typename T, typename P, typename D>
  void ComputeBondedCUDA::copyToStage(const int ntuples, const T* __restrict__ src,
  const P* __restrict__ p_array, std::vector<D>& dst) {

  for (int ituple=0;ituple < ntuples;ituple++) {
    D dstval;
    copyTupleToStage<T, P, D>(src[ituple], p_array, dstval);
    dst.push_back(dstval);
  }
}
#endif  // NODEGROUP_FORCE_REGISTER


//
// Used for both dihedrals and impropers
//
template <bool doDihedral, typename T, typename P>
void ComputeBondedCUDA::copyDihedralDatafp32(const int ntuples, const T* __restrict__ src,
  const P* __restrict__ p_array, CudaDihedral* __restrict__ dst) {

  PatchMap* patchMap = PatchMap::Object();
  float3 b1f = make_float3(lattice.a_r().x, lattice.a_r().y, lattice.a_r().z);
  float3 b2f = make_float3(lattice.b_r().x, lattice.b_r().y, lattice.b_r().z);
  float3 b3f = make_float3(lattice.c_r().x, lattice.c_r().y, lattice.c_r().z);

  for (int ituple=0;ituple < ntuples;ituple++) {
    CudaDihedral dstval;
    auto p0 = src[ituple].p[0];
    auto p1 = src[ituple].p[1];
    auto p2 = src[ituple].p[2];
    auto p3 = src[ituple].p[3];
    int pi0 = patchIndex[p0->patchID];
    int pi1 = patchIndex[p1->patchID];
    int pi2 = patchIndex[p2->patchID];
    int pi3 = patchIndex[p3->patchID];
    int l0 = src[ituple].localIndex[0];
    int l1 = src[ituple].localIndex[1];
    int l2 = src[ituple].localIndex[2];
    int l3 = src[ituple].localIndex[3];
    dstval.i = l0 + patches[pi0].atomStart;
    dstval.j = l1 + patches[pi1].atomStart;
    dstval.k = l2 + patches[pi2].atomStart;
    dstval.l = l3 + patches[pi3].atomStart;
    if (doDihedral) {
      dstval.itype = dihedralMultMap[(src[ituple].value - p_array)];
    } else {
      dstval.itype = improperMultMap[(src[ituple].value - p_array)];
    }
#if 0
    Position position1 = p0->x[l0].position;
    Position position2 = p1->x[l1].position;
    Position position3 = p2->x[l2].position;
    Position position4 = p3->x[l3].position;
    Vector shiftVec12 = lattice.wrap_delta_scaled_fast(position1, position2);
    Vector shiftVec23 = lattice.wrap_delta_scaled_fast(position2, position3);
    Vector shiftVec43 = lattice.wrap_delta_scaled_fast(position4, position3);
    if(pi0 != pi1) shiftVec12 += patchMap->center(p0->patchID) - patchMap->center(p1->patchID);
    if(pi1 != pi2) shiftVec23 += patchMap->center(p1->patchID) - patchMap->center(p2->patchID);
    if(pi3 != pi2) shiftVec43 += patchMap->center(p3->patchID) - patchMap->center(p2->patchID);

    dstval.ioffsetXYZ = make_float3((float)shiftVec12.x, (float)shiftVec12.y, (float)shiftVec12.z);
    dstval.joffsetXYZ = make_float3((float)shiftVec23.x, (float)shiftVec23.y, (float)shiftVec23.z);
    dstval.loffsetXYZ = make_float3((float)shiftVec43.x, (float)shiftVec43.y, (float)shiftVec43.z);
#endif

    float3 position1 = make_float3(p0->x[l0].position.x, p0->x[l0].position.y, p0->x[l0].position.z);
    float3 position2 = make_float3(p1->x[l1].position.x, p1->x[l1].position.y, p1->x[l1].position.z);
    float3 position3 = make_float3(p2->x[l2].position.x, p2->x[l2].position.y, p2->x[l2].position.z);
    float3 position4 = make_float3(p3->x[l3].position.x, p3->x[l3].position.y, p3->x[l3].position.z);

    float3 diff12, diff23, diff43;
    diff12 = position1 - position2;
    diff23 = position2 - position3;
    diff43 = position4 - position3;

    float d12_x = -floorf(b1f.x * diff12.x + b1f.y * diff12.y + b1f.z * diff12.z + 0.5f);
    float d12_y = -floorf(b2f.x * diff12.x + b2f.y * diff12.y + b2f.z * diff12.z + 0.5f);
    float d12_z = -floorf(b3f.x * diff12.x + b3f.y * diff12.y + b3f.z * diff12.z + 0.5f);

    float d23_x = -floorf(b1f.x * diff23.x + b1f.y * diff23.y + b1f.z * diff23.z + 0.5f);
    float d23_y = -floorf(b2f.x * diff23.x + b2f.y * diff23.y + b2f.z * diff23.z + 0.5f);
    float d23_z = -floorf(b3f.x * diff23.x + b3f.y * diff23.y + b3f.z * diff23.z + 0.5f);

    float d43_x = -floorf(b1f.x * diff43.x + b1f.y * diff43.y + b1f.z * diff43.z + 0.5f);
    float d43_y = -floorf(b2f.x * diff43.x + b2f.y * diff43.y + b2f.z * diff43.z + 0.5f);
    float d43_z = -floorf(b3f.x * diff43.x + b3f.y * diff43.y + b3f.z * diff43.z + 0.5f);

    if(pi0 != pi1){
      Vector c = patchMap->center(p0->patchID) - patchMap->center(p1->patchID);
      d12_x += c.x;
      d12_y += c.y;
      d12_z += c.z;
    }

    if(pi1 != pi2){
      Vector c = patchMap->center(p1->patchID) - patchMap->center(p2->patchID);
      d23_x += c.x;
      d23_y += c.y;
      d23_z += c.z;
    }

    if(pi3 != pi2){
      Vector c = patchMap->center(p3->patchID) - patchMap->center(p2->patchID);
      d43_x += c.x;
      d43_y += c.y;
      d43_z += c.z;
    }

    dstval.ioffsetXYZ = make_float3(d12_x, d12_y, d12_z);
    dstval.joffsetXYZ = make_float3(d23_x, d23_y, d23_z);
    dstval.loffsetXYZ = make_float3(d43_x, d43_y, d43_z);

    dstval.scale = src[ituple].scale;
    if (hostAlchFlags.alchOn) {
      const AtomID (&atomID)[sizeof(src[ituple].atomID)/sizeof(AtomID)](src[ituple].atomID);
      const Molecule& mol = *(Node::Object()->molecule);
      dstval.fepBondedType = mol.get_fep_bonded_type(atomID, 4);
    } else {
      dstval.fepBondedType = 0;
    }
    dst[ituple] = dstval;
  }
}


void ComputeBondedCUDA::copyExclusionData(const int ntuples, const ExclElem* __restrict__ src, const int typeSize,
  CudaExclusion* __restrict__ dst1, CudaExclusion* __restrict__ dst2, int64_t& pos, int64_t& pos2) {

  PatchMap* patchMap = PatchMap::Object();
  for (int ituple=0;ituple < ntuples;ituple++) {
    auto p0 = src[ituple].p[0];
    auto p1 = src[ituple].p[1];
    int pi0 = patchIndex[p0->patchID];
    int pi1 = patchIndex[p1->patchID];
    int l0 = src[ituple].localIndex[0];
    int l1 = src[ituple].localIndex[1];
    CompAtom& ca1 = p0->x[l0];
    CompAtom& ca2 = p1->x[l1];
    Position position1 = ca1.position;
    Position position2 = ca2.position;
    Vector shiftVec = lattice.wrap_delta_scaled(position1, position2);
    if(pi0 != pi1) shiftVec += patchMap->center(p0->patchID) - patchMap->center(p1->patchID);
    CudaExclusion ce;
    ce.i            = l0 + patches[pi0].atomStart;
    ce.j            = l1 + patches[pi1].atomStart;
    ce.vdwtypei    = ca1.vdwType;
    ce.vdwtypej    = ca2.vdwType;
    ce.ioffsetXYZ = make_float3((float)shiftVec.x, (float)shiftVec.y, (float)shiftVec.z);
    if (hostAlchFlags.alchOn) {
      const unsigned char p1 = ca1.partition;
      const unsigned char p2 = ca2.partition;
      ce.pswitch = pswitchTable[size_t(p1 + 3*p2)];
    } else {
      ce.pswitch = 0;
    }
    if (src[ituple].modified) {
      *dst1 = ce;
      dst1++;
      pos += typeSize;
    } else {
      *dst2 = ce;
      dst2++;
      pos2 += typeSize;
    }
  }
}

#ifdef NODEGROUP_FORCE_REGISTER
template<>
void ComputeBondedCUDA::copyTupleToStage(const ExclElem& __restrict__ src,
  const int* __restrict__ p_array, CudaExclusionStage& dstval) {

  auto p0 = src.p[0];
  auto p1 = src.p[1];
  int pi0 = patchIndex[p0->patchID];
  int pi1 = patchIndex[p1->patchID];
  int l0 = src.localIndex[0];
  int l1 = src.localIndex[1];
  CompAtom& ca1 = p0->x[l0];
  CompAtom& ca2 = p1->x[l1];

  dstval.vdwtypei    = ca1.vdwType;
  dstval.vdwtypej    = ca2.vdwType;

  dstval.patchIDs[0] = p0->patchID;
  dstval.patchIDs[1] = p1->patchID;

  dstval.index[0] = l0;
  dstval.index[1] = l1;

  if (hostAlchFlags.alchOn) {
    const unsigned char p1 = ca1.partition;
    const unsigned char p2 = ca2.partition;
    dstval.pswitch = pswitchTable[size_t(p1 + 3*p2)];
  } else {
    dstval.pswitch = 0;
  }
}



void ComputeBondedCUDA::copyExclusionDataStage(const int ntuples, const ExclElem* __restrict__ src, const int typeSize,
  std::vector<CudaExclusionStage>& dst1, std::vector<CudaExclusionStage>& dst2, int64_t& pos, int64_t& pos2) {

  for (int ituple=0;ituple < ntuples;ituple++) {
    CudaExclusionStage ce;
    copyTupleToStage<ExclElem, int, CudaExclusionStage>(src[ituple], nullptr, ce);
    if (src[ituple].modified) {
      dst1.push_back(ce);
      pos += typeSize;
    } else {
      dst2.push_back(ce);
      pos2 += typeSize;
    }
  }
}
#endif  // NODEGROUP_FORCE_REGISTER

void ComputeBondedCUDA::copyCrosstermData(const int ntuples, const CrosstermElem* __restrict__ src,
  const CrosstermValue* __restrict__ crossterm_array, CudaCrossterm* __restrict__ dst) {

  PatchMap* patchMap = PatchMap::Object();
  for (int ituple=0;ituple < ntuples;ituple++) {
    auto p0 = src[ituple].p[0];
    auto p1 = src[ituple].p[1];
    auto p2 = src[ituple].p[2];
    auto p3 = src[ituple].p[3];
    auto p4 = src[ituple].p[4];
    auto p5 = src[ituple].p[5];
    auto p6 = src[ituple].p[6];
    auto p7 = src[ituple].p[7];
    int pi0 = patchIndex[p0->patchID];
    int pi1 = patchIndex[p1->patchID];
    int pi2 = patchIndex[p2->patchID];
    int pi3 = patchIndex[p3->patchID];
    int pi4 = patchIndex[p4->patchID];
    int pi5 = patchIndex[p5->patchID];
    int pi6 = patchIndex[p6->patchID];
    int pi7 = patchIndex[p7->patchID];
    int l0 = src[ituple].localIndex[0];
    int l1 = src[ituple].localIndex[1];
    int l2 = src[ituple].localIndex[2];
    int l3 = src[ituple].localIndex[3];
    int l4 = src[ituple].localIndex[4];
    int l5 = src[ituple].localIndex[5];
    int l6 = src[ituple].localIndex[6];
    int l7 = src[ituple].localIndex[7];
    dst[ituple].i1 = l0 + patches[pi0].atomStart;
    dst[ituple].i2 = l1 + patches[pi1].atomStart;
    dst[ituple].i3 = l2 + patches[pi2].atomStart;
    dst[ituple].i4 = l3 + patches[pi3].atomStart;
    dst[ituple].i5 = l4 + patches[pi4].atomStart;
    dst[ituple].i6 = l5 + patches[pi5].atomStart;
    dst[ituple].i7 = l6 + patches[pi6].atomStart;
    dst[ituple].i8 = l7 + patches[pi7].atomStart;
    dst[ituple].itype = (src[ituple].value - crossterm_array);
    Position position1 = p0->x[l0].position;
    Position position2 = p1->x[l1].position;
    Position position3 = p2->x[l2].position;
    Position position4 = p3->x[l3].position;
    Position position5 = p4->x[l4].position;
    Position position6 = p5->x[l5].position;
    Position position7 = p6->x[l6].position;
    Position position8 = p7->x[l7].position;
    Vector shiftVec12 = lattice.wrap_delta_scaled(position1, position2);
    Vector shiftVec23 = lattice.wrap_delta_scaled(position2, position3);
    Vector shiftVec34 = lattice.wrap_delta_scaled(position3, position4);
    Vector shiftVec56 = lattice.wrap_delta_scaled(position5, position6);
    Vector shiftVec67 = lattice.wrap_delta_scaled(position6, position7);
    Vector shiftVec78 = lattice.wrap_delta_scaled(position7, position8);
    if(pi0 != pi1) shiftVec12 += patchMap->center(p0->patchID) - patchMap->center(p1->patchID);
    if(pi1 != pi2) shiftVec23 += patchMap->center(p1->patchID) - patchMap->center(p2->patchID);
    if(pi2 != pi3) shiftVec34 += patchMap->center(p2->patchID) - patchMap->center(p3->patchID);
    if(pi4 != pi5) shiftVec56 += patchMap->center(p4->patchID) - patchMap->center(p5->patchID);
    if(pi5 != pi6) shiftVec67 += patchMap->center(p5->patchID) - patchMap->center(p6->patchID);
    if(pi6 != pi7) shiftVec78 += patchMap->center(p6->patchID) - patchMap->center(p7->patchID);
    dst[ituple].offset12XYZ = make_float3( (float)shiftVec12.x, (float)shiftVec12.y, (float)shiftVec12.z);
    dst[ituple].offset23XYZ = make_float3( (float)shiftVec23.x, (float)shiftVec23.y, (float)shiftVec23.z);
    dst[ituple].offset34XYZ = make_float3( (float)shiftVec34.x, (float)shiftVec34.y, (float)shiftVec34.z);
    dst[ituple].offset56XYZ = make_float3( (float)shiftVec56.x, (float)shiftVec56.y, (float)shiftVec56.z);
    dst[ituple].offset67XYZ = make_float3( (float)shiftVec67.x, (float)shiftVec67.y, (float)shiftVec67.z);
    dst[ituple].offset78XYZ = make_float3( (float)shiftVec78.x, (float)shiftVec78.y, (float)shiftVec78.z);
    if (hostAlchFlags.alchOn) {
      const AtomID (&atomID)[sizeof(src[ituple].atomID)/sizeof(AtomID)](src[ituple].atomID);
      const Molecule& mol = *(Node::Object()->molecule);
      int typeSum1 = 0, typeSum2 = 0;
      for (size_t i = 0; i < 4; ++i) {
        typeSum1 += (mol.get_fep_type(atomID[i]) == 2 ? -1 : mol.get_fep_type(atomID[i]));
        typeSum2 += (mol.get_fep_type(atomID[i+4]) == 2 ? -1 : mol.get_fep_type(atomID[i+4]));
      }
      int order = (hostAlchFlags.alchBondDecouple ? 5 : 4);
      if ((0 < typeSum1 && typeSum1 < order) || (0 < typeSum2 && typeSum2 < order)) {
        dst[ituple].fepBondedType = 1;
      } else if ((0 > typeSum1 && typeSum1 > -order) || (0 > typeSum2 && typeSum2 > -order)) {
        dst[ituple].fepBondedType = 2;
      }
    } else {
      dst[ituple].fepBondedType = 0;
    }
    dst[ituple].scale = src[ituple].scale;
  }
}

#ifdef NODEGROUP_FORCE_REGISTER
template <>
void ComputeBondedCUDA::copyTupleToStage(const CrosstermElem& src,
  const CrosstermValue* __restrict__ crossterm_array, CudaCrosstermStage& dstval) {

  auto p0 = src.p[0];
  auto p1 = src.p[1];
  auto p2 = src.p[2];
  auto p3 = src.p[3];
  auto p4 = src.p[4];
  auto p5 = src.p[5];
  auto p6 = src.p[6];
  auto p7 = src.p[7];
  int pi0 = patchIndex[p0->patchID];
  int pi1 = patchIndex[p1->patchID];
  int pi2 = patchIndex[p2->patchID];
  int pi3 = patchIndex[p3->patchID];
  int pi4 = patchIndex[p4->patchID];
  int pi5 = patchIndex[p5->patchID];
  int pi6 = patchIndex[p6->patchID];
  int pi7 = patchIndex[p7->patchID];
  int l0 = src.localIndex[0];
  int l1 = src.localIndex[1];
  int l2 = src.localIndex[2];
  int l3 = src.localIndex[3];
  int l4 = src.localIndex[4];
  int l5 = src.localIndex[5];
  int l6 = src.localIndex[6];
  int l7 = src.localIndex[7];
  dstval.itype = (src.value - crossterm_array);

  dstval.patchIDs[0] = p0->patchID;
  dstval.patchIDs[1] = p1->patchID;
  dstval.patchIDs[2] = p2->patchID;
  dstval.patchIDs[3] = p3->patchID;
  dstval.patchIDs[4] = p4->patchID;
  dstval.patchIDs[5] = p5->patchID;
  dstval.patchIDs[6] = p6->patchID;
  dstval.patchIDs[7] = p7->patchID;

  dstval.index[0] = l0;
  dstval.index[1] = l1;
  dstval.index[2] = l2;
  dstval.index[3] = l3;
  dstval.index[4] = l4;
  dstval.index[5] = l5;
  dstval.index[6] = l6;
  dstval.index[7] = l7;

  dstval.scale = src.scale;
  if (hostAlchFlags.alchOn) {
    const AtomID (&atomID)[sizeof(src.atomID)/sizeof(AtomID)](src.atomID);
    const Molecule& mol = *(Node::Object()->molecule);
    int typeSum1 = 0, typeSum2 = 0;
    for (size_t i = 0; i < 4; ++i) {
      typeSum1 += (mol.get_fep_type(atomID[i]) == 2 ? -1 : mol.get_fep_type(atomID[i]));
      typeSum2 += (mol.get_fep_type(atomID[i+4]) == 2 ? -1 : mol.get_fep_type(atomID[i+4]));
    }
    int order = (hostAlchFlags.alchBondDecouple ? 5 : 4);
    if ((0 < typeSum1 && typeSum1 < order) || (0 < typeSum2 && typeSum2 < order)) {
      dstval.fepBondedType = 1;
    } else if ((0 > typeSum1 && typeSum1 > -order) || (0 > typeSum2 && typeSum2 > -order)) {
      dstval.fepBondedType = 2;
    }
  } else {
    dstval.fepBondedType = 0;
  }
}
#endif  // NODEGROUP_FORCE_REGISTER


void ComputeBondedCUDA::tupleCopyWorker(int first, int last, void *result, int paraNum, void *param) {
  ComputeBondedCUDA* c = (ComputeBondedCUDA *)param;
  c->tupleCopyWorker(first, last);
}

void ComputeBondedCUDA::tupleCopyWorker(int first, int last) {
  NAMD_EVENT_START(1, NamdProfileEvent::COMPUTE_BONDED_TUPLE_COPY);
  if (first == -1) {
    // Separate exclusions into modified, and non-modified
    int64_t pos = exclusionStartPos;
    int64_t pos2 = exclusionStartPos2;
    for (auto it = tupleList[Tuples::EXCLUSION].begin();it != tupleList[Tuples::EXCLUSION].end();it++) {
      int ntuples = (*it)->getNumTuples();
      copyExclusionData(ntuples, (ExclElem *)(*it)->getTupleList(), CudaTupleTypeSize[Tuples::EXCLUSION],
        (CudaExclusion *)&tupleData[pos], (CudaExclusion *)&tupleData[pos2], pos, pos2);
    }
    first = 0;
  }

  // JM: Move the switch statement outside and do the for loop inside the cases
  for (int i=first;i <= last;i++) {
    switch (tupleCopyWorkList[i].tupletype) {

      case Tuples::BOND:
      {
#if 1
        copyBondData(tupleCopyWorkList[i].ntuples, (BondElem *)tupleCopyWorkList[i].tupleElemList,
          Node::Object()->parameters->bond_array, (CudaBond *)&tupleData[tupleCopyWorkList[i].tupleDataPos]);
#else
         copyBondDatafp32(tupleCopyWorkList[i].ntuples, (BondElem *)tupleCopyWorkList[i].tupleElemList,
           Node::Object()->parameters->bond_array, (CudaBond *)&tupleData[tupleCopyWorkList[i].tupleDataPos]);
#endif
      }
      break;

      case Tuples::ANGLE:
      {
        copyAngleData(tupleCopyWorkList[i].ntuples, (AngleElem *)tupleCopyWorkList[i].tupleElemList,
          Node::Object()->parameters->angle_array, (CudaAngle *)&tupleData[tupleCopyWorkList[i].tupleDataPos]);
      }
      break;

      case Tuples::DIHEDRAL:
      {

#if 1
        copyDihedralData<true, DihedralElem, DihedralValue>(tupleCopyWorkList[i].ntuples,
          (DihedralElem *)tupleCopyWorkList[i].tupleElemList, Node::Object()->parameters->dihedral_array,
          (CudaDihedral *)&tupleData[tupleCopyWorkList[i].tupleDataPos]);
#else
        copyDihedralDatafp32<true, DihedralElem, DihedralValue>(tupleCopyWorkList[i].ntuples,
          (DihedralElem *)tupleCopyWorkList[i].tupleElemList, Node::Object()->parameters->dihedral_array,
          (CudaDihedral *)&tupleData[tupleCopyWorkList[i].tupleDataPos]);
#endif
      }
      break;

      case Tuples::IMPROPER:
      {
        copyDihedralData<false, ImproperElem, ImproperValue>(tupleCopyWorkList[i].ntuples,
          (ImproperElem *)tupleCopyWorkList[i].tupleElemList, Node::Object()->parameters->improper_array,
          (CudaDihedral *)&tupleData[tupleCopyWorkList[i].tupleDataPos]);
      }
      break;

      case Tuples::CROSSTERM:
      {
        copyCrosstermData(tupleCopyWorkList[i].ntuples, (CrosstermElem *)tupleCopyWorkList[i].tupleElemList,
          Node::Object()->parameters->crossterm_array, (CudaCrossterm *)&tupleData[tupleCopyWorkList[i].tupleDataPos]);
      }
      break;

      default:
      NAMD_bug("ComputeBondedCUDA::tupleCopyWorker, Unsupported tuple type");
      break;
    }
  }


  NAMD_EVENT_STOP(1, NamdProfileEvent::COMPUTE_BONDED_TUPLE_COPY);

}


#ifdef NODEGROUP_FORCE_REGISTER

void ComputeBondedCUDA::updateMaxTupleCounts(TupleCounts counts) {
  // Atomically compute the max number of tuples across all GPUs
  // Each GPU will use this as their buffer size * OVERALLOC to keep things easier

  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();

  // Update bonds
  int numBondsTest = patchData->maxNumBonds.load();
  while (numBondsTest < counts.bond &&
         !patchData->maxNumBonds.compare_exchange_strong(numBondsTest, counts.bond));

  int numAnglesTest = patchData->maxNumAngles.load();
  while (numAnglesTest < counts.angle &&
         !patchData->maxNumAngles.compare_exchange_strong(numAnglesTest, counts.angle));

  int numDihedralsTest = patchData->maxNumDihedrals.load();
  while (numDihedralsTest < counts.dihedral &&
         !patchData->maxNumDihedrals.compare_exchange_strong(numDihedralsTest, counts.dihedral));

  int numImpropersTest = patchData->maxNumImpropers.load();
  while (numImpropersTest < counts.improper &&
         !patchData->maxNumImpropers.compare_exchange_strong(numImpropersTest, counts.improper));

  int numModifiedExclusionsTest = patchData->maxNumModifiedExclusions.load();
  while (numModifiedExclusionsTest < counts.modifiedExclusion &&
         !patchData->maxNumModifiedExclusions.compare_exchange_strong(numModifiedExclusionsTest, counts.modifiedExclusion));

  int numExclusionsTest = patchData->maxNumExclusions.load();
  while (numExclusionsTest < counts.exclusion &&
         !patchData->maxNumExclusions.compare_exchange_strong(numExclusionsTest, counts.exclusion));

  int numCrosstermsTest = patchData->maxNumCrossterms.load();
  while (numCrosstermsTest < counts.crossterm &&
         !patchData->maxNumCrossterms.compare_exchange_strong(numCrosstermsTest, counts.crossterm));


}

TupleCounts ComputeBondedCUDA::getMaxTupleCounts() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();

  TupleCounts counts;

  counts.bond = patchData->maxNumBonds.load();
  counts.angle = patchData->maxNumAngles.load();
  counts.dihedral = patchData->maxNumDihedrals.load();
  counts.improper = patchData->maxNumImpropers.load();
  counts.modifiedExclusion = patchData->maxNumModifiedExclusions.load();
  counts.exclusion = patchData->maxNumExclusions.load();
  counts.crossterm = patchData->maxNumCrossterms.load();

#if 0
  CkPrintf("[%d] Max: Bonds %d, angles %d, dihedral %d, improper %d, modexl %d, exl %d, cross %d\n",
         CkMyPe(), counts.bond, counts.angle, counts.dihedral, counts.improper, counts.modifiedExclusion,
         counts.exclusion, counts.crossterm);
#endif

  return counts;
}


/**
 * \brief Migrates and updates the tuples on the GPU
 * 
 * This is a multi-step process to update the tuples after a migration. It will use the atomic
 * migration destinations to determine the new location of each tuple, communicate the tuples,
 * and update their information. To do this, the tuples have two formats; for bonds they are 
 * CudaBond and CudaBondStage. The stage data structures contain static information about each
 * tuple, like the type, scale as well as the current patchID and localID of each atom in the
 * tuple. We do not include the global atomID here because we are the migration destination 
 * structures lets us go from pre-migration indexInDevice -> post-migration patchID, localID.
 * These stage structures are what actually get moved during tuple migration, and the last
 * step of tuple migration is to convert them into the structures to be used by the bonded
 * kernel.
 * 
 * Tuple migration has the following steps:
 * - computeTupleDestination: this will compute the new home patch for each tuple, where the 
 *   home patch is determined by the downstream of all the patches of the atoms in the tuple.
 *   This will also use a scan operation to compute the new index of tuples that are remaining
 *   within the patch. Before this kernel, the `index` field of the stage data refers to the 
 *   indexInDevice. However, afterwards it is the atom's index in the patch
 *   TODO DMC should confirm that this is true....
 * - reserveTupleDestination: this will compute the new indices of the migration tuples using
 *   atomics. This can probably be optimized, but it is easy to implement
 * - computePatchOffsets: based on the previous kernels, we now know the per-patch tuple count
 *   This kernel will compute the offset into the single device-wide structure of tupels. It
 *   also computes the total number of tuples on the device
 * - The next few steps compute the maximum number of tuples across all the devices. The devices
 *   maintaine symmetrically allocated tuple buffers, so this is essentially a few atomic maxes
 *   Based on these maximum tuple counts, we will reallocate the tuple buffers if necessary.
 *   The number of tuples per patch various significantly, so the number of tuples per device
 *   can also vary significantly. If reallocation occurs we need to re-register the P2P pointers.
 *   Additionally, it needs to happen in two steps. First we reallocate the scratch buffers that
 *   will be written to during migration. Then, after migration occurs, we will reallocate the 
 *   main stage and non-stage buffers before repopulating them based on the scratch buffers.
 * - Finally, we update the tuples. This converts the stage tuples into actual tuple structures,
 *   updates the stage tuples, and copies them from the scratch buffers into the main ones. This 
 *   kernel will update the index field of the stage tuples, such that it refers to the indexInDevice,
 *   instead of index in patch
 *
 */
void ComputeBondedCUDA::migrateTuples(bool startup) {

  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();
  const int devInd = deviceCUDA->getDeviceIndex();

  bool MGPU = deviceCUDA->getNumDevice() > 1;
  bool amMaster = (masterPe == CkMyPe());
  CmiNodeBarrier();


  if (amMaster) {
    cudaStreamSynchronize(stream);
    copyPatchData();
    cudaStreamSynchronize(stream);
  }
  CmiNodeBarrier();

  if (amMaster) {
    if (!startup) {
      PatchMap* patchMap = PatchMap::Object();
      const int4* migrationDestination = patchData->h_soa_migrationDestination[devInd];
      // Update the device of the tuples
      // DMC this also updates the patches of the atoms, so it needs to be done in the single GPU case too

      TupleCounts counts = bondedKernel.getTupleCounts();
      migrationKernel.computeTupleDestination(
        devInd,
        counts,
        patchData->devData[devInd].numPatchesHome,
        migrationDestination,
        patchData->devData[devInd].d_patchToDeviceMap,
        patchData->devData[devInd].d_globalToLocalID,
        patchMap->getADim(), patchMap->getBDim(),
        patchMap->getCMaxIndex(), patchMap->getBMaxIndex(), patchMap->getAMaxIndex(),
        stream
      );
      cudaCheck(cudaStreamSynchronize(stream));
    }
  }   
  CmiNodeBarrier();

  if (amMaster) {
    if (!startup) {
      migrationKernel.reserveTupleDestination(devInd, patchData->devData[devInd].numPatchesHome, stream);
      cudaCheck(cudaStreamSynchronize(stream));
    }
  }
  CmiNodeBarrier();

  TupleCounts local, newMax;
  bool realloc = false;

  if (amMaster && !startup) {
    migrationKernel.computePatchOffsets(patchData->devData[devInd].numPatchesHome, stream);

    local = migrationKernel.fetchTupleCounts(patchData->devData[devInd].numPatchesHome, stream);
    updateMaxTupleCounts(local);
#if 0
    CkPrintf("[%d] Actual: Bonds %d, angles %d, dihedral %d, improper %d, modexl %d, exl %d cross %d\n",
           CkMyPe(), local.bond, local.angle, local.dihedral, local.improper, local.modifiedExclusion,
           local.exclusion, local.crossterm);
#endif
  }
  CmiNodeBarrier();
  if (amMaster && !startup) {
    // Only multi-GPU needs to check for reallocation
    if (MGPU) { 
      newMax = getMaxTupleCounts();
      realloc = migrationKernel.reallocateBufferDst(newMax);
      patchData->tupleReallocationFlagPerDevice[devInd] = realloc;
    }
  }
  CmiNodeBarrier();

  //
  // Non-master PEs need to see if reallocation is happening
  // They can all use the 0th GPU flag since the GPUs reallocate
  // based on the max
  if (!amMaster) {
    realloc = patchData->tupleReallocationFlagPerDevice[0];
  }

  // If reallocated, then we need to register the P2P pointer
  // Since the GPUs use the max tuple counts for the buffer sizes
  // If one GPU reallocated, then they all did.

  if (realloc) {
    if (amMaster) {
      registerPointersToHost(); 
    }
    CmiNodeBarrier();

    if (amMaster) {
      copyHostRegisterToDevice();
      cudaCheck(cudaStreamSynchronize(stream));
    }
    CmiNodeBarrier();
  }

  if (amMaster && !startup) {
    // Moves the tuples to the dst buffer on all GPUS
    migrationKernel.performTupleMigration(
      bondedKernel.getTupleCounts(),
      stream
    );
    cudaCheck(cudaStreamSynchronize(stream));
  }
  CmiNodeBarrier(); // We need a barrier because of the reallocation DMC believes
   
  if (amMaster && !startup) {
    // If we previously reallocated we need to reallocate the other buffers too
    // The regular tuples must be "Reallocated" everytime, so the offsets into
    // the unified buffer can be updated. The unified buffer made since when we
    // were copying the tuples from the host each migration, but I don't think
    // it still makes sense to have since it complicates reallocation 
    if (MGPU) { 
      bondedKernel.reallocateTupleBuffer(newMax, stream);
      migrationKernel.reallocateBufferSrc(newMax);
      cudaCheck(cudaStreamSynchronize(stream));
    }
  }
  CmiNodeBarrier();

  // Update the local tuple count and copies the migrated tuples back to the
  // src buffer
  if (amMaster) {
    if (!startup) {
      bondedKernel.setTupleCounts(local);
    }

    const int* ids = patchData->h_soa_id[devInd];
    migrationKernel.updateTuples(
      bondedKernel.getTupleCounts(),
      bondedKernel.getData(),
      ids,
      d_patchRecord, 
      d_patchMapCenter, 
      bondedKernel.getAtomBuffer(),
      lattice, 
      stream
    );
    cudaCheck(cudaDeviceSynchronize());
  }
}

/**
 * The tuples need to be sorted based on their home patch, which is the downstream patch of all the atoms
 * They are not generated in this sort by default (DMC thinks), so we are explicitly sorting them now
 * We also compute the number and offsets of tuples in each patch
 * 
 */
template<typename T>
void ComputeBondedCUDA::sortTupleList(std::vector<T>& tuples, std::vector<int>& tupleCounts, std::vector<int>& tupleOffsets) {
  PatchMap* patchMap = PatchMap::Object();

  std::vector<std::pair<int,int>> downstreamPatches;
  for (int i = 0; i < tuples.size(); i++) {
    int downstream = tuples[i].patchIDs[0];
    for (int j = 1; j < T::size; j++) {
      downstream = patchMap->downstream(downstream, tuples[i].patchIDs[j]);
    }
    downstreamPatches.push_back(std::make_pair(i, downstream));
    tupleCounts[patchIndex[downstream]]++;
  }
  
  //
  // Compute the offset into the tuples based on patch
  // TODO replace with STD prefix sum
  tupleOffsets[0] = 0;
  for (int i = 0; i < tupleCounts.size(); i++) {
    tupleOffsets[i+1] = tupleCounts[i] + tupleOffsets[i];
  }

  //
  // Sort tuples based on home patch
  //
  std::stable_sort(downstreamPatches.begin(), downstreamPatches.end(),
    [](std::pair<int, int> a, std::pair<int, int> b) {
    return a.second < b.second;
  });
  
  //
  //  Propogate order to tuples
  //
  std::vector<T> copy = tuples;
  for (int i = 0; i < tuples.size(); i++) {
    tuples[i] = copy[downstreamPatches[i].first];
  }
}

void ComputeBondedCUDA::sortAndCopyToDevice() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();
  const int devInd = deviceCUDA->getDeviceIndex();
  const int numPatchesHome = patchData->devData[devInd].numPatchesHome;

  TupleDataStage h_dataStage;
  TupleIntArrays h_counts;
  TupleIntArrays h_offsets;

  // 
  // These cannot be declared within the macro because then they
  // would be out of scope
  // 
  std::vector<int> bondPatchCounts(numPatchesHome, 0);
  std::vector<int> bondPatchOffsets(numPatchesHome+1, 0);

  std::vector<int> anglePatchCounts(numPatchesHome, 0);
  std::vector<int> anglePatchOffsets(numPatchesHome+1, 0);

  std::vector<int> dihedralPatchCounts(numPatchesHome, 0);
  std::vector<int> dihedralPatchOffsets(numPatchesHome+1, 0);

  std::vector<int> improperPatchCounts(numPatchesHome, 0);
  std::vector<int> improperPatchOffsets(numPatchesHome+1, 0);

  std::vector<int> modifiedExclusionPatchCounts(numPatchesHome, 0);
  std::vector<int> modifiedExclusionPatchOffsets(numPatchesHome+1, 0);

  std::vector<int> exclusionPatchCounts(numPatchesHome, 0);
  std::vector<int> exclusionPatchOffsets(numPatchesHome+1, 0);

  std::vector<int> crosstermPatchCounts(numPatchesHome, 0);
  std::vector<int> crosstermPatchOffsets(numPatchesHome+1, 0);

  #define CALL(fieldName) do { \
    sortTupleList(fieldName##TupleData, fieldName##PatchCounts, fieldName##PatchOffsets); \
    h_dataStage.fieldName = fieldName##TupleData.data(); \
    h_counts.fieldName = fieldName##PatchCounts.data(); \
    h_offsets.fieldName = fieldName##PatchOffsets.data(); \
  } while(0);

  CALL(bond);
  CALL(angle);
  CALL(dihedral);
  CALL(improper);
  CALL(modifiedExclusion);
  CALL(exclusion);
  CALL(crossterm);

  #undef CALL

  migrationKernel.copyTupleToDevice(
    bondedKernel.getTupleCounts(),
    numPatchesHome,
    h_dataStage,
    h_counts,
    h_offsets,
    stream
  );

  cudaCheck(cudaStreamSynchronize(stream));
}

/**
 * \brief This will generate the stage tuples for a type type
 *
 * This should process both the home and self tuples from a given type. Unlike the other
 * versions of this function, it processes both home and self in one call
 *
 */
void ComputeBondedCUDA::tupleCopyWorkerType(int tupletype) {
  NAMD_EVENT_START(1, NamdProfileEvent::COMPUTE_BONDED_TUPLE_COPY);

  switch (tupletype) {
    case Tuples::EXCLUSION:
    {
      // Separate exclusions into modified, and non-modified
      modifiedExclusionTupleData.clear();
      exclusionTupleData.clear();
      int64_t pos = exclusionStartPos;
      int64_t pos2 = exclusionStartPos2;
      for (auto it = tupleList[Tuples::EXCLUSION].begin();it != tupleList[Tuples::EXCLUSION].end();it++) {
        int ntuples = (*it)->getNumTuples();
        copyExclusionDataStage(ntuples, (ExclElem *)(*it)->getTupleList(), CudaTupleTypeSizeStage[Tuples::EXCLUSION],
          modifiedExclusionTupleData, exclusionTupleData, pos, pos2);
      }
      // Reserve patch warp aligned data for copy.
      // TODO DMC is this necessary? Why are we copying the padded part too?
      modifiedExclusionTupleData.reserve(ComputeBondedCUDAKernel::warpAlign(modifiedExclusionTupleData.size()));
      exclusionTupleData.reserve(ComputeBondedCUDAKernel::warpAlign(exclusionTupleData.size()));
    }
    break;

    case Tuples::BOND:
    {
      bondTupleData.clear();
      for (auto it = tupleList[Tuples::BOND].begin();it != tupleList[Tuples::BOND].end();it++) {
        int ntuples = (*it)->getNumTuples();
        BondElem* elemList = (BondElem *)(*it)->getTupleList();
        copyToStage<BondElem, BondValue, CudaBondStage>(
          ntuples, elemList, Node::Object()->parameters->bond_array, bondTupleData
        );
      }

      bondTupleData.reserve(ComputeBondedCUDAKernel::warpAlign(bondTupleData.size()));
    }
    break;

    case Tuples::ANGLE:
    {
      angleTupleData.clear();
      for (auto it = tupleList[tupletype].begin();it != tupleList[tupletype].end();it++) {
        int ntuples = (*it)->getNumTuples();
        AngleElem* elemList = (AngleElem *)(*it)->getTupleList();
        copyToStage<AngleElem, AngleValue, CudaAngleStage>(ntuples, elemList, Node::Object()->parameters->angle_array, angleTupleData);
      }

      angleTupleData.reserve(ComputeBondedCUDAKernel::warpAlign(angleTupleData.size()));
    }
    break;

    case Tuples::DIHEDRAL:
    {
      dihedralTupleData.clear();
      for (auto it = tupleList[tupletype].begin();it != tupleList[tupletype].end();it++) {
        int ntuples = (*it)->getNumTuples();
        DihedralElem* elemList = (DihedralElem *)(*it)->getTupleList();
        copyToStage<DihedralElem, DihedralValue, CudaDihedralStage>(ntuples, elemList,
          Node::Object()->parameters->dihedral_array, dihedralTupleData);
      }

      dihedralTupleData.reserve(ComputeBondedCUDAKernel::warpAlign(dihedralTupleData.size()));
    }
    break;

    case Tuples::IMPROPER:
    {
      improperTupleData.clear();
      for (auto it = tupleList[tupletype].begin();it != tupleList[tupletype].end();it++) {
        int ntuples = (*it)->getNumTuples();
        ImproperElem* elemList = (ImproperElem *)(*it)->getTupleList();
        copyToStage<ImproperElem, ImproperValue, CudaDihedralStage>(ntuples, elemList, 
          Node::Object()->parameters->improper_array, improperTupleData);
      }

      improperTupleData.reserve(ComputeBondedCUDAKernel::warpAlign(improperTupleData.size()));
    }
    break;

    case Tuples::CROSSTERM:
    {
      crosstermTupleData.clear();
      for (auto it = tupleList[tupletype].begin();it != tupleList[tupletype].end();it++) {
        int ntuples = (*it)->getNumTuples();
        CrosstermElem* elemList = (CrosstermElem *)(*it)->getTupleList();
        copyToStage<CrosstermElem, CrosstermValue, CudaCrosstermStage>(ntuples, elemList, 
          Node::Object()->parameters->crossterm_array, crosstermTupleData);
      }

      crosstermTupleData.reserve(ComputeBondedCUDAKernel::warpAlign(crosstermTupleData.size()));
    }
    break;

    default:
    NAMD_bug("ComputeBondedCUDA::tupleCopyWorker, Unsupported tuple type");
    break;
  }

  NAMD_EVENT_STOP(1, NamdProfileEvent::COMPUTE_BONDED_TUPLE_COPY);
}
#endif  // NODEGROUP_FORCE_REGISTER

//
// Copies tuple data form individual buffers to a single contigious buffer
// NOTE: This is done on the master PE
//

void ComputeBondedCUDA::copyTupleData() {

  PatchMap* patchMap = PatchMap::Object();

  // Count the number of exclusions
  int numModifiedExclusions = 0;
  int numExclusions = 0;
  for (int i=0;i < numExclPerRank.size();i++) {
    numModifiedExclusions += numExclPerRank[i].numModifiedExclusions;
    numExclusions         += numExclPerRank[i].numExclusions;
  }
  size_t numModifiedExclusionsWA = ComputeBondedCUDAKernel::warpAlign(numModifiedExclusions);
  size_t numExclusionsWA         = ComputeBondedCUDAKernel::warpAlign(numExclusions);

  // Count the number of tuples for each type
  int64_t posWA = 0;
  exclusionStartPos = 0;
  exclusionStartPos2 = 0;
  tupleCopyWorkList.clear();
  for (int tupletype=0;tupletype < Tuples::NUM_TUPLE_TYPES;tupletype++) {
    // Take temporary position
    int64_t pos = posWA;
    if (tupletype == Tuples::EXCLUSION) {
      exclusionStartPos = pos;
      exclusionStartPos2 = pos + numModifiedExclusionsWA*CudaTupleTypeSize[Tuples::EXCLUSION];
    }
    // Count for total number of tuples for this tupletype
    int num = 0;
    for (auto it = tupleList[tupletype].begin();it != tupleList[tupletype].end();it++) {
      int ntuples = (*it)->getNumTuples();
      num += ntuples;
      if (tupletype != Tuples::EXCLUSION) {
        TupleCopyWork tupleCopyWork;
        tupleCopyWork.tupletype     = tupletype;
        tupleCopyWork.ntuples       = ntuples;
        tupleCopyWork.tupleElemList = (*it)->getTupleList();
        tupleCopyWork.tupleDataPos  = pos;
        // XXX NOTE: redundant copy happening here
        tupleCopyWorkList.push_back(tupleCopyWork);
        pos += ntuples*CudaTupleTypeSize[tupletype];
      }
    }
    numTuplesPerType[tupletype] = num;
    //
    if (tupletype == Tuples::EXCLUSION) {
      // Warp-align exclusions separately
      posWA += (numModifiedExclusionsWA + numExclusionsWA)*CudaTupleTypeSize[tupletype];
    } else {
      posWA += ((int64_t) ComputeBondedCUDAKernel::warpAlign(num))*CudaTupleTypeSize[tupletype];
    }
  }
  if (numModifiedExclusions + numExclusions != numTuplesPerType[Tuples::EXCLUSION]) {
    NAMD_bug("ComputeBondedCUDA::copyTupleData, invalid number of exclusions");
  }

  // Set flags for finishPatchesOnPe
  hasExclusions = (numExclusions > 0);
  hasModifiedExclusions = (numModifiedExclusions > 0);

  // Re-allocate storage as needed
  // reallocate_host<char>(&tupleData, &tupleDataSize, size, 1.2f);
  reallocate_host<char>(&tupleData, &tupleDataSize, posWA, 1.2f);

#if CMK_SMP && USE_CKLOOP
  int useCkLoop = Node::Object()->simParameters->useCkLoop;
  if (useCkLoop >= 1) {
    // JM NOTE: What this does is divice the work into chunks of CkMyNodeSize to be scheduled
    // between PE. To circumvent this, I have to call it by hand on all pes
    CkLoop_Parallelize(tupleCopyWorker, 1, (void *)this, CkMyNodeSize(), -1, tupleCopyWorkList.size() - 1);
  } else
#endif
  {

    tupleCopyWorker(-1, tupleCopyWorkList.size() - 1);
  }

  bondedKernel.update(numTuplesPerType[Tuples::BOND], numTuplesPerType[Tuples::ANGLE],
    numTuplesPerType[Tuples::DIHEDRAL], numTuplesPerType[Tuples::IMPROPER],
    numModifiedExclusions, numExclusions, numTuplesPerType[Tuples::CROSSTERM],
    tupleData, stream);

  // Re-allocate forces
  int forceStorageSize = bondedKernel.getAllForceSize(atomStorageSize, true);
  reallocate_host<FORCE_TYPE>(&forces, &forcesSize, forceStorageSize, 1.4f);
}

// JM: Single node version to bypass CkLoop
#ifdef NODEGROUP_FORCE_REGISTER

void ComputeBondedCUDA::copyTupleDataSN() {

  PatchMap* patchMap = PatchMap::Object();
  size_t numExclusions, numModifiedExclusions, copyIndex;
    
  
  // WORK TO BE DONE BY THE MASTERPE
  if(masterPe == CkMyPe()){
    numModifiedExclusions = 0;
    numExclusions = 0;
    for (int i=0;i < numExclPerRank.size();i++) {
      numModifiedExclusions += numExclPerRank[i].numModifiedExclusions;
      numExclusions         += numExclPerRank[i].numExclusions;
    }
    size_t numModifiedExclusionsWA = ComputeBondedCUDAKernel::warpAlign(numModifiedExclusions);
    size_t numExclusionsWA         = ComputeBondedCUDAKernel::warpAlign(numExclusions);

    // Count the number of tuples for each type
    int64_t posWA = 0;
    exclusionStartPos = 0;
    exclusionStartPos2 = 0;
    tupleCopyWorkList.clear();
    for (int tupletype=0;tupletype < Tuples::NUM_TUPLE_TYPES;tupletype++) {
      // Take temporary position
      int64_t pos = posWA;
      if (tupletype == Tuples::EXCLUSION) {
        exclusionStartPos = pos;
        exclusionStartPos2 = pos + numModifiedExclusionsWA*CudaTupleTypeSize[Tuples::EXCLUSION];
      }
      // Count for total number of tuples for this tupletype
      int num = 0;
      for (auto it = tupleList[tupletype].begin();it != tupleList[tupletype].end();it++) {
        int ntuples = (*it)->getNumTuples();
        num += ntuples;
        if (tupletype != Tuples::EXCLUSION) {
          TupleCopyWork tupleCopyWork;
          tupleCopyWork.tupletype     = tupletype;
          tupleCopyWork.ntuples       = ntuples;
          tupleCopyWork.tupleElemList = (*it)->getTupleList();
          tupleCopyWork.tupleDataPos  = pos;
          tupleCopyWorkList.push_back(tupleCopyWork);
          pos += ntuples*CudaTupleTypeSize[tupletype];
        }
      }
      numTuplesPerType[tupletype] = num;
      //
      if (tupletype == Tuples::EXCLUSION) {
        // Warp-align exclusions separately
        posWA += (numModifiedExclusionsWA + numExclusionsWA)*CudaTupleTypeSize[tupletype];
      } else {
        posWA += ((int64_t) ComputeBondedCUDAKernel::warpAlign(num))*CudaTupleTypeSize[tupletype];
      }
    }
    if (numModifiedExclusions + numExclusions != numTuplesPerType[Tuples::EXCLUSION]) {
      NAMD_bug("ComputeBondedCUDA::copyTupleData, invalid number of exclusions");
    }

    // Set flags for finishPatchesOnPe
    hasExclusions = (numExclusions > 0);
    hasModifiedExclusions = (numModifiedExclusions > 0);

    // Re-allocate storage as needed
    // reallocate_host<char>(&tupleData, &tupleDataSize, size, 1.2f);
    reallocate_host<char>(&tupleData, &tupleDataSize, posWA, 1.2f);
  }

  CmiNodeBarrier();

#if 0
  //int chunkSize = (tupleCopyWorkList.size()) /CkMyNodeSize() +
  //  (CkMyPe() == CkMyNodeSize()-1) * tupleCopyWorkList.size() % CkMyNodeSize();
  //int first = ((CkMyNodeSize() -1) - CkMyPe())*chunkSize -1*(CkMyPe() == CkMyNodeSize()-1);
  //int last  = (first + chunkSize) -1*(!CkMyPe());; // chunksize is padded for the last pe
  this->tupleCopyWorker(first, last);
#else
  // JM: This probably leads to better load balance than the option above.
  // XXX TODO: Improve this further afterwards. I don't think this scales too well
  //           due to lots of atomics, but you're not supposed to run this with
  //           lots of PEs anyways
  while( (copyIndex = tupleWorkIndex.fetch_add(1) ) <=  tupleCopyWorkList.size()){
     this->tupleCopyWorker(copyIndex -1, copyIndex -1);
  }
#endif
  CmiNodeBarrier();

  if(masterPe == CkMyPe()){
    tupleWorkIndex.store(0);
    bondedKernel.update(numTuplesPerType[Tuples::BOND], numTuplesPerType[Tuples::ANGLE],
      numTuplesPerType[Tuples::DIHEDRAL], numTuplesPerType[Tuples::IMPROPER],
      numModifiedExclusions, numExclusions, numTuplesPerType[Tuples::CROSSTERM],
      tupleData, stream);

    // Re-allocate forces
    int forceStorageSize = bondedKernel.getAllForceSize(atomStorageSize, true);
    reallocate_host<FORCE_TYPE>(&forces, &forcesSize, forceStorageSize, 1.4f);
  }
}


/**
 *  \brief Copies tuples into stage format for GPU migration
 * 
 * This is very similar to the single-node version copyTupleDataSN, but it 
 * does the tuple migration on the GPU
 * 
 */
void ComputeBondedCUDA::copyTupleDataGPU(const int startup) {

  PatchMap* patchMap = PatchMap::Object();
  size_t numExclusions, numModifiedExclusions, copyIndex;

  // WORK TO BE DONE BY THE MASTERPE
  if (startup) {
    // The atom map is no longer needed, so unmap the atoms. We need to do this in-case
    // multiple run commands are issued. It is safe to do this because there is a node
    // barrier between loadTuplesOnPe and this function
    unmapAtoms();
    
    if(masterPe == CkMyPe()){
      numModifiedExclusions = 0;
      numExclusions = 0;
      for (int i=0;i < numExclPerRank.size();i++) {
        numModifiedExclusions += numExclPerRank[i].numModifiedExclusions;
        numExclusions         += numExclPerRank[i].numExclusions;
      }
      size_t numModifiedExclusionsWA = ComputeBondedCUDAKernel::warpAlign(numModifiedExclusions);
      size_t numExclusionsWA         = ComputeBondedCUDAKernel::warpAlign(numExclusions);

      // Count the number of tuples for each type
      size_t posWA = 0;
      exclusionStartPos = 0;
      exclusionStartPos2 = 0;
      tupleCopyWorkList.clear();
      for (int tupletype=0;tupletype < Tuples::NUM_TUPLE_TYPES;tupletype++) {
        // Take temporary position
        size_t pos = posWA;
        if (tupletype == Tuples::EXCLUSION) {
          exclusionStartPos = pos;
          exclusionStartPos2 = pos + numModifiedExclusionsWA*CudaTupleTypeSizeStage[Tuples::EXCLUSION];
        }
        // Count for total number of tuples for this tupletype
        int num = 0;
        for (auto it = tupleList[tupletype].begin();it != tupleList[tupletype].end();it++) {
          int ntuples = (*it)->getNumTuples();
          num += ntuples;
          if (tupletype != Tuples::EXCLUSION) {
            TupleCopyWork tupleCopyWork;
            tupleCopyWork.tupletype     = tupletype;
            tupleCopyWork.ntuples       = ntuples;
            tupleCopyWork.tupleElemList = (*it)->getTupleList();
            tupleCopyWork.tupleDataPos  = pos;
            tupleCopyWorkList.push_back(tupleCopyWork);
            pos += ntuples*CudaTupleTypeSizeStage[tupletype];
          }
        }
        numTuplesPerType[tupletype] = num;
        //
        if (tupletype == Tuples::EXCLUSION) {
          // Warp-align exclusions separately
          posWA += (numModifiedExclusionsWA + numExclusionsWA)*CudaTupleTypeSizeStage[tupletype];
        } else {
          posWA += ((int64_t) ComputeBondedCUDAKernel::warpAlign(num))*CudaTupleTypeSize[tupletype];
        }
      }
      if (numModifiedExclusions + numExclusions != numTuplesPerType[Tuples::EXCLUSION]) {
        NAMD_bug("ComputeBondedCUDA::copyTupleData, invalid number of exclusions");
      }

      // Set flags for finishPatchesOnPe
      hasExclusions = (numExclusions > 0);
      hasModifiedExclusions = (numModifiedExclusions > 0);

      // Re-allocate storage as needed
      // reallocate_host<char>(&tupleData, &tupleDataSize, size, 1.2f);
      reallocate_host<char>(&tupleData, &tupleDataSize, posWA, 1.2f);


      TupleCounts local;
      local.bond = numTuplesPerType[Tuples::BOND];
      local.angle = numTuplesPerType[Tuples::ANGLE];
      local.dihedral = numTuplesPerType[Tuples::DIHEDRAL];
      local.improper = numTuplesPerType[Tuples::IMPROPER];
      local.modifiedExclusion = numModifiedExclusions;
      local.exclusion = numExclusions;
      local.crossterm = numTuplesPerType[Tuples::CROSSTERM];

      updateMaxTupleCounts(local);
      bondedKernel.setTupleCounts(local);
    }   
  
    // Wait for tuple counts to be updated...
    CmiNodeBarrier();

    if(masterPe == CkMyPe()){
      TupleCounts newMax = getMaxTupleCounts();
      migrationKernel.reallocateBufferSrc(newMax);
      migrationKernel.reallocateBufferDst(newMax);
      bondedKernel.reallocateTupleBuffer(newMax, stream);
      registerPointersToHost(); 
    }

    // Node barrier is because of registering
    CmiNodeBarrier();

    if(masterPe == CkMyPe()){
      copyHostRegisterToDevice();
      for (int tupletype=0;tupletype < Tuples::NUM_TUPLE_TYPES;tupletype++) {
        this->tupleCopyWorkerType(tupletype);
      }
      sortAndCopyToDevice();
    }
  }

  migrateTuples(startup);

  if(masterPe == CkMyPe()){
    TupleCounts count = bondedKernel.getTupleCounts();
    numTuplesPerType[Tuples::BOND] = count.bond;
    numTuplesPerType[Tuples::ANGLE] = count.angle;
    numTuplesPerType[Tuples::DIHEDRAL] = count.dihedral;
    numTuplesPerType[Tuples::IMPROPER] = count.improper;
    numModifiedExclusions = count.modifiedExclusion;
    numExclusions = count.exclusion;
    numTuplesPerType[Tuples::CROSSTERM] = count.crossterm;

    // Set flags for finishPatchesOnPe
    hasExclusions = (numExclusions > 0);
    hasModifiedExclusions = (numModifiedExclusions > 0);

    // Re-allocate forces
    int forceStorageSize = bondedKernel.getAllForceSize(atomStorageSize, true);
    reallocate_host<FORCE_TYPE>(&forces, &forcesSize, forceStorageSize, 1.4f);
  }
}

#endif


/**
 * \brief Updates the patch records
 * 
 * This code was originally moved from launchWork
 * I don't think a lot of these structures are used anymore. At least the pointers in PatchData
 * so TODO clean this up...
 *
 */
#ifdef NODEGROUP_FORCE_REGISTER
void ComputeBondedCUDA::updatePatchRecords() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();
  int devInd = deviceCUDA->getDeviceIndex();
  PatchRecord **d_pr   = &(patchData->devData[devInd].bond_pr);
  int **d_pid          = &(patchData->devData[devInd].bond_pi);
  size_t st_d_pid_size = (size_t)(patchData->devData[devInd].bond_pi_size);
  size_t st_d_pr_size  = (size_t)(patchData->devData[devInd].bond_pr_size);
  patchData->devData[devInd].forceStride = bondedKernel.getForceStride(atomStorageSize);
  reallocate_device<PatchRecord>(d_pr, &st_d_pr_size, patches.size()); // st_d_pr_size changed to patches.size() here
  patchData->devData[devInd].bond_pr_size = (int)(st_d_pr_size); // copy updated value back into devData
  reallocate_device<int>(d_pid, &st_d_pid_size, patches.size());
  patchData->devData[devInd].bond_pi_size = (int)(st_d_pid_size); // copy updated value back into devData
  copy_HtoD<PatchRecord>(&(patches[0]), *d_pr, patches.size(), stream);
  copy_HtoD<int>(&(patchIndex[0]), *d_pid, patches.size(), stream);
  if (params->CUDASOAintegrate && params->useDeviceMigration) {
    patchData->devData[devInd].b_datoms = bondedKernel.getAtomBuffer();
  }
}
#endif

//
// Launch work on GPU
//
void ComputeBondedCUDA::launchWork() {
  SimParameters *params = Node::Object()->simParameters;
  if (CkMyPe() != masterPe)
    NAMD_bug("ComputeBondedCUDA::launchWork() called on non master PE");

  cudaCheck(cudaSetDevice(deviceID));
  if (atomsChanged) {
    if(!params->CUDASOAintegrate) copyTupleData();
    // copyTupledata nees to be called before launchWork then

// Move this once GPU migration has been merged
#if 1
#ifdef NODEGROUP_FORCE_REGISTER
    if(params->CUDASOAintegrate && !params->useDeviceMigration){
      updatePatchRecords();
    }
#endif
#endif
  }

  float3 lata = make_float3(lattice.a().x, lattice.a().y, lattice.a().z);
  float3 latb = make_float3(lattice.b().x, lattice.b().y, lattice.b().z);
  float3 latc = make_float3(lattice.c().x, lattice.c().y, lattice.c().z);

  int r2_delta_expc = 64 * (ComputeNonbondedUtil::r2_delta_exp - 127);

#if 0
  if(!atomsChanged){
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    PatchData *patchData = cpdata.ckLocalBranch();

    float4 *d_atoms = bondedKernel.getAtomBuffer();
    copy_DtoH_sync<float4>(d_atoms, (float4*) atoms, atomStorageSize);
    if(!params->CUDASOAintegrate) CmiLock(printLock);
    else CmiLock(patchData->printlock);

    fprintf(stderr, "DEV[%d] BOND POS PRINTOUT\n", deviceID);

    for(int i = 0 ; i < atomStorageSize; i++){
      fprintf(stderr, " ATOMS[%d] = %lf %lf %lf %lf\n",
        i, atoms[i].x, atoms[i].y, atoms[i].z, atoms[i].q);
    }

    if(!params->CUDASOAintegrate) CmiUnlock(printLock);
    else CmiUnlock(patchData->printlock);
  }
#endif

  const CudaNBConstants nbConstants = CudaComputeNonbonded::getNonbondedCoef(params);
  const bool doTable = CudaComputeNonbonded::getDoTable(params, doSlow, doVirial);

  bondedKernel.bondedForce(
    ComputeNonbondedUtil::scale14,
    atomStorageSize,
    doEnergy, doVirial, doSlow, doTable,
    lata, latb, latc,
    (float)ComputeNonbondedUtil::cutoff2,
    (float)ComputeNonbondedUtil::r2_delta, r2_delta_expc,
    nbConstants,
    (const float4*)atoms, forces, 
    energies_virials, atomsChanged, params->CUDASOAintegrate, 
    params->useDeviceMigration, stream);


#ifdef NODEGROUP_FORCE_REGISTER
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    PatchData *patchData = cpdata.ckLocalBranch();
    int devInd = deviceCUDA->getDeviceIndex();
    if (!params->CUDASOAintegrate || !params->useDeviceMigration) {
      patchData->devData[devInd].b_datoms = bondedKernel.getAtomBuffer();
    }
    patchData->devData[devInd].f_bond = bondedKernel.getForces();
    patchData->devData[devInd].f_bond_nbond = patchData->devData[devInd].f_bond + bondedKernel.getForceSize(atomStorageSize);
    patchData->devData[devInd].f_bond_slow  = patchData->devData[devInd].f_bond + 2*bondedKernel.getForceSize(atomStorageSize);
    patchData->devData[devInd].f_bond_size = bondedKernel.getForceSize(atomStorageSize);
#endif

#if 0
  if(!params->CUDASOAintegrate) CmiLock(printLock);
  else CmiLock(patchData->printlock);
  int forceStorageSize = bondedKernel.getAllForceSize(atomStorageSize, true);
  // synchronously copy array back to the host
  copy_DtoH_sync(bondedKernel.getForces(), forces,  forceStorageSize);
  fprintf(stderr, "DEV[%d] BOND FORCE PRINTOUT\n", deviceID);
  for(int i = 0; i < forceStorageSize; i++){
    // prints entire forces datastructure
    fprintf(stderr, "BOND[%d] = %lf\n", i, forces[i]);
  }
  if(!params->CUDASOAintegrate) CmiUnlock(printLock);
  else CmiUnlock(patchData->printlock);
#endif
  forceDoneSetCallback();
}

void ComputeBondedCUDA::forceDoneCheck(void *arg, double walltime) {
  ComputeBondedCUDA* c = (ComputeBondedCUDA *)arg;

  if (CkMyPe() != c->masterPe)
    NAMD_bug("ComputeBondedCUDA::forceDoneCheck called on non masterPe");

  cudaCheck(cudaSetDevice(c->deviceID));

  cudaError_t err = cudaEventQuery(c->forceDoneEvent);
  if (err == cudaSuccess) {
    // Event has occurred
    c->checkCount = 0;
    traceUserBracketEvent(CUDA_BONDED_KERNEL_EVENT, c->beforeForceCompute, walltime);
    c->finishPatches();
    return;
  } else if (err != cudaErrorNotReady) {
    // Anything else is an error
    char errmsg[256];
    sprintf(errmsg,"in ComputeBondedCUDA::forceDoneCheck after polling %d times over %f s",
            c->checkCount, walltime - c->beforeForceCompute);
    cudaDie(errmsg,err);
  }

  // Event has not occurred
  c->checkCount++;
  if (c->checkCount >= 1000000) {
    char errmsg[256];
    sprintf(errmsg,"ComputeBondedCUDA::forceDoneCheck polled %d times over %f s",
            c->checkCount, walltime - c->beforeForceCompute);
    cudaDie(errmsg,err);
  }

  // Call again
  CcdCallBacksReset(0, walltime);
  CcdCallFnAfter(forceDoneCheck, arg, 0.1);
}

//
// Set call back for all the work in the stream at this point
//
void ComputeBondedCUDA::forceDoneSetCallback() {
  if (CkMyPe() != masterPe)
    NAMD_bug("ComputeBondedCUDA::forceDoneSetCallback called on non masterPe");
  cudaCheck(cudaSetDevice(deviceID));
  cudaCheck(cudaEventRecord(forceDoneEvent, stream));
  checkCount = 0;
  CcdCallBacksReset(0, CmiWallTimer());
  // Start timer for CUDA kernel
  beforeForceCompute = CkWallTimer();
  // Set the call back at 0.1ms
  if(!params->CUDASOAintegrate) CcdCallFnAfter(forceDoneCheck, this, 0.1);
}

template <bool sumNbond, bool sumSlow>
void finishForceLoop(const int numAtoms, const int forceStride,
  const double* __restrict__ af,
  const double* __restrict__ af_nbond,
  const double* __restrict__ af_slow,
  Force* __restrict__ f,
  Force* __restrict__ f_nbond,
  Force* __restrict__ f_slow) {

#if 0
  for (int j=0;j < numAtoms;j++) {
    {
      //double afx, afy, afz;
      //convertForceToDouble(af + j, forceStride, afx, afy, afz);
      //f[j].x += af[j];
      //f[j].y += af[j + forceStride];
      //f[j].z += af[j + forceStride*2];
      f[j].x += af[j];
      f[j].y += af[j + forceStride];
      f[j].z += af[j + forceStride*2];
    }
    if (sumNbond)
    {
      //double afx, afy, afz;
      //convertForceToDouble(af_nbond + j, forceStride, afx, afy, afz);
      f_nbond[j].x += af_nbond[j];
      f_nbond[j].y += af_nbond[j + forceStride];
      f_nbond[j].z += af_nbond[j + forceStride*2];

    }
    if (sumSlow)
    {
      //double afx, afy, afz;
      //convertForceToDouble(af_slow + j, forceStride, afx, afy, afz);
      f_slow[j].x += af_slow[j];
      f_slow[j].y += af_slow[j + forceStride];
      f_slow[j].z += af_slow[j + forceStride*2];

    }
  }
#endif

#if 0
  for(int j=0; j < numAtoms; j++){
    f[j].x += af[j];
    f[j].y += af[j+ forceStride];
    f[j].z += af[j+ 2*forceStride];
  }
  if(sumNbond){
    for(int j=0; j < numAtoms; j++){
      f_nbond[j].x += af_nbond[j];
      f_nbond[j].y += af_nbond[j + forceStride];
      f_nbond[j].z += af_nbond[j + 2*forceStride];
    }
  }
  if(sumSlow){
    for(int j=0; j < numAtoms; j++){
      f_slow[j].x += af_slow[j];
      f_slow[j].y += af_slow[j + forceStride];
      f_slow[j].z += af_slow[j + 2*forceStride];
    }
  }
#endif
  // XXX Summing into AOS buffer from SOA data layout
  // Try to reduce cache misses by separating loops.
  for(int j=0; j < numAtoms; j++) f[j].x += af[j];
  for(int j=0; j < numAtoms; j++) f[j].y += af[j+ forceStride];
  for(int j=0; j < numAtoms; j++) f[j].z += af[j+ 2*forceStride];

  if(sumNbond){
#ifdef DEBUG_MINIMIZE
    int k=0;
    printf("%s, line %d\n", __FILE__, __LINE__);
    printf("  before:  f_nbond[%d] = %f %f %f\n",
        k, f_nbond[k].x, f_nbond[k].y, f_nbond[k].z);
#endif
    for(int j=0; j < numAtoms; j++) f_nbond[j].x += af_nbond[j];
    for(int j=0; j < numAtoms; j++) f_nbond[j].y += af_nbond[j + forceStride];
    for(int j=0; j < numAtoms; j++) f_nbond[j].z += af_nbond[j + 2*forceStride];
#ifdef DEBUG_MINIMIZE
    printf("  after:   f_nbond[%d] = %f %f %f\n",
        k, f_nbond[k].x, f_nbond[k].y, f_nbond[k].z);
#endif
  }
  if(sumSlow){
    for(int j=0; j < numAtoms; j++) f_slow[j].x += af_slow[j];
    for(int j=0; j < numAtoms; j++) f_slow[j].y += af_slow[j + forceStride];
    for(int j=0; j < numAtoms; j++) f_slow[j].z += af_slow[j + 2*forceStride];
  }
#if 0
  for(int j = 0; j < numAtoms; j++){
    fprintf(stderr, "f[%d] = %lf %lf %lf | %lf %lf %lf | %lf %lf %lf\n", j,
      af[j], af[j + forceStride], af[j + 2*forceStride],
      af_nbond[j], af_nbond[j + forceStride], af_nbond[j + 2*forceStride],
      af_slow[j],  af_slow[j + forceStride],  af_slow[j + 2*forceStride]);
  }
#endif
}

//
// Finish all patches that are on this pe
//
void ComputeBondedCUDA::finishPatchesOnPe() {
  NAMD_EVENT_START(1, NamdProfileEvent::COMPUTE_BONDED_CUDA_FINISH_PATCHES);

  PatchMap* patchMap = PatchMap::Object();
  int myRank = CkMyRank();

  const int forceStride = bondedKernel.getForceStride(atomStorageSize);
  const int forceSize = bondedKernel.getForceSize(atomStorageSize);
  const bool sumNbond = hasModifiedExclusions;
  const bool sumSlow = (hasModifiedExclusions || hasExclusions) && doSlow;

  // I need to print patchIDsPerRank
  for (int i=0;i < patchIDsPerRank[myRank].size();i++) {
    PatchID patchID = patchIDsPerRank[myRank][i];
    Patch* patch = patchMap->patch(patchID);
    TuplePatchElem* tpe = tuplePatchList.find(TuplePatchElem(patchID));
    if (tpe == NULL) {
      NAMD_bug("ComputeBondedCUDA::finishPatchesOnPe, TuplePatchElem not found");
    }

    int pi = patchIndex[patchID];
    int numAtoms = patches[pi].numAtoms;
    int atomStart = patches[pi].atomStart;

    // XXX TODO: this is a large CPU bottleneck
    // Ok, so this is
#ifdef NODEGROUP_FORCE_REGISTER
    double *af       = forces + atomStart;
    double *af_nbond = forces + forceSize + atomStart;
    double *af_slow  = forces + 2*forceSize + atomStart;
    // XXX TODO: We still need forces inside patches for migration steps it seems
    if(!params->CUDASOAintegrate || (atomsChanged && !params->useDeviceMigration) ){
      Force *f = tpe->f;
      Force *f_nbond = tpe->r->f[Results::nbond];
      Force *f_slow = tpe->r->f[Results::slow];
      if (!sumNbond && !sumSlow) {
        finishForceLoop<false, false>(numAtoms, forceStride, af, af_nbond, af_slow, f, f_nbond, f_slow);
      } else if (sumNbond && !sumSlow) {
        finishForceLoop<true, false>(numAtoms, forceStride, af, af_nbond, af_slow, f, f_nbond, f_slow);
      } else if (!sumNbond && sumSlow) {
        finishForceLoop<false, true>(numAtoms, forceStride, af, af_nbond, af_slow, f, f_nbond, f_slow);
      } else if (sumNbond && sumSlow) {
        finishForceLoop<true, true>(numAtoms, forceStride, af, af_nbond, af_slow, f, f_nbond, f_slow);
      } else {
        NAMD_bug("ComputeBondedCUDA::finishPatchesOnPe, logically impossible choice");
      }
      tpe->forceBox->close(&tpe->r);
      tpe->positionBox->close(&tpe->x);
      if ( doMolly ) tpe->avgPositionBox->close(&tpe->x_avg);
    }
#else
    Force *f = tpe->f;
    Force *f_nbond = tpe->r->f[Results::nbond];
    Force *f_slow = tpe->r->f[Results::slow];

    double *af       = forces + atomStart;
    double *af_nbond = forces + forceSize + atomStart;
    double *af_slow  = forces + 2*forceSize + atomStart;

    if (!sumNbond && !sumSlow) {
      finishForceLoop<false, false>(numAtoms, forceStride, af, af_nbond, af_slow, f, f_nbond, f_slow);
    } else if (sumNbond && !sumSlow) {
      finishForceLoop<true, false>(numAtoms, forceStride, af, af_nbond, af_slow, f, f_nbond, f_slow);
    } else if (!sumNbond && sumSlow) {
      finishForceLoop<false, true>(numAtoms, forceStride, af, af_nbond, af_slow, f, f_nbond, f_slow);
    } else if (sumNbond && sumSlow) {
      finishForceLoop<true, true>(numAtoms, forceStride, af, af_nbond, af_slow, f, f_nbond, f_slow);
    } else {
      NAMD_bug("ComputeBondedCUDA::finishPatchesOnPe, logically impossible choice");
    }

    tpe->forceBox->close(&tpe->r);
    tpe->positionBox->close(&tpe->x);
    if ( doMolly ) tpe->avgPositionBox->close(&tpe->x_avg);

#endif
// #endif

  }


  NAMD_EVENT_STOP(1, NamdProfileEvent::COMPUTE_BONDED_CUDA_FINISH_PATCHES);

  NAMD_EVENT_START(1, NamdProfileEvent::COMPUTE_BONDED_CUDA_FINISH_PATCHES_LOCK);

  bool done = false;
  CmiLock(lock);
  patchesCounter -= patchIDsPerRank[CkMyRank()].size();
  if(params->CUDASOAintegrate && !atomsChanged){
    // masterPe is executing this, so we can go ahead and do reductions 
    // by themselves. However, for migrations, I still need to follow the usual 
    // codepath, because of the box setup
    patchesCounter = 0;
  }
  if (patchesCounter == 0) {
    patchesCounter = getNumPatches();
    done = true;
  }
  CmiUnlock(lock);
  if (done) {
    //computeMgr->sendFinishReductions(masterPe, this);
    if(params->CUDASOAintegrate ) {
      if(!atomsChanged) this->finishReductions();
    }
    else computeMgr->sendFinishReductions(masterPe, this);
  }

  NAMD_EVENT_STOP(1, NamdProfileEvent::COMPUTE_BONDED_CUDA_FINISH_PATCHES_LOCK);

}

void ComputeBondedCUDA::finishPatches() {

  if (atomsChanged && (!params->CUDASOAintegrate || !params->useDeviceMigration)) {
    unmapAtoms();
  }
  // do I have to synchronize the stream here??????
  //computeMgr->sendFinishPatchesOnPe(pes, this);
  if(params->CUDASOAintegrate) this->finishPatchesOnPe();
  else{
    computeMgr->sendFinishPatchesOnPe(pes, this);
  }
}

//
// Finish & submit reductions
//
void ComputeBondedCUDA::finishReductions() {

  if (CkMyPe() != masterPe)
    NAMD_bug("ComputeBondedCUDA::finishReductions() called on non masterPe");

  //fprintf(stderr, "finishReductions\n");

  // static int ncall = 0;
  // ncall++;

  int pos = 0;
  // do I need this streamSynchronize here?
  if(params->CUDASOAintegrate) {
    cudaCheck(cudaStreamSynchronize(stream));
  }
  for (int tupletype=0;tupletype < Tuples::NUM_TUPLE_TYPES;tupletype++) {
    if (numTuplesPerType[tupletype] > 0) {

      if (doEnergy) {
        switch (tupletype) {
          case Tuples::BOND:
#ifdef NODEGROUP_FORCE_REGISTER
          if (params->CUDASOAintegrate) {
            nodeReduction->item(REDUCTION_BOND_ENERGY) += energies_virials[ComputeBondedCUDAKernel::energyIndex_BOND];
          } else
#endif
          {
            reduction->item(REDUCTION_BOND_ENERGY) += energies_virials[ComputeBondedCUDAKernel::energyIndex_BOND];
          }
          if (hostAlchFlags.alchFepOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_BONDED_ENERGY_F) += energies_virials[ComputeBondedCUDAKernel::energyIndex_BOND_F];
            } else 
#endif
            {
              reduction->item(REDUCTION_BONDED_ENERGY_F) += energies_virials[ComputeBondedCUDAKernel::energyIndex_BOND_F];
            }
          }
          if (hostAlchFlags.alchThermIntOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_BONDED_ENERGY_TI_1) += energies_virials[ComputeBondedCUDAKernel::energyIndex_BOND_TI_1];
              nodeReduction->item(REDUCTION_BONDED_ENERGY_TI_2) += energies_virials[ComputeBondedCUDAKernel::energyIndex_BOND_TI_2];
            } else
#endif
            {
              reduction->item(REDUCTION_BONDED_ENERGY_TI_1) += energies_virials[ComputeBondedCUDAKernel::energyIndex_BOND_TI_1];
              reduction->item(REDUCTION_BONDED_ENERGY_TI_2) += energies_virials[ComputeBondedCUDAKernel::energyIndex_BOND_TI_2];
            }
          }
          break;

        case Tuples::ANGLE:
#ifdef NODEGROUP_FORCE_REGISTER
          if (params->CUDASOAintegrate) {
            nodeReduction->item(REDUCTION_ANGLE_ENERGY) += energies_virials[ComputeBondedCUDAKernel::energyIndex_ANGLE];
          } else
#endif
          {
            reduction->item(REDUCTION_ANGLE_ENERGY) += energies_virials[ComputeBondedCUDAKernel::energyIndex_ANGLE];
          }
          if (hostAlchFlags.alchFepOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_BONDED_ENERGY_F) += energies_virials[ComputeBondedCUDAKernel::energyIndex_ANGLE_F];
            } else 
#endif
            {
              reduction->item(REDUCTION_BONDED_ENERGY_F) += energies_virials[ComputeBondedCUDAKernel::energyIndex_ANGLE_F];
            }
          }
          if (hostAlchFlags.alchThermIntOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_BONDED_ENERGY_TI_1) += energies_virials[ComputeBondedCUDAKernel::energyIndex_ANGLE_TI_1];
              nodeReduction->item(REDUCTION_BONDED_ENERGY_TI_2) += energies_virials[ComputeBondedCUDAKernel::energyIndex_ANGLE_TI_2];
            } else 
#endif
            {
              reduction->item(REDUCTION_BONDED_ENERGY_TI_1) += energies_virials[ComputeBondedCUDAKernel::energyIndex_ANGLE_TI_1];
              reduction->item(REDUCTION_BONDED_ENERGY_TI_2) += energies_virials[ComputeBondedCUDAKernel::energyIndex_ANGLE_TI_2];
            }
          }
          break;

        case Tuples::DIHEDRAL:
#ifdef NODEGROUP_FORCE_REGISTER
          if (params->CUDASOAintegrate) {
            nodeReduction->item(REDUCTION_DIHEDRAL_ENERGY) += energies_virials[ComputeBondedCUDAKernel::energyIndex_DIHEDRAL];
          } else
#endif
          {
            reduction->item(REDUCTION_DIHEDRAL_ENERGY) += energies_virials[ComputeBondedCUDAKernel::energyIndex_DIHEDRAL];
          }
          if (hostAlchFlags.alchFepOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_BONDED_ENERGY_F) += energies_virials[ComputeBondedCUDAKernel::energyIndex_DIHEDRAL_F];
            } else
#endif
            {
              reduction->item(REDUCTION_BONDED_ENERGY_F) += energies_virials[ComputeBondedCUDAKernel::energyIndex_DIHEDRAL_F];
            }
          }
          if (hostAlchFlags.alchThermIntOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_BONDED_ENERGY_TI_1) += energies_virials[ComputeBondedCUDAKernel::energyIndex_DIHEDRAL_TI_1];
              nodeReduction->item(REDUCTION_BONDED_ENERGY_TI_2) += energies_virials[ComputeBondedCUDAKernel::energyIndex_DIHEDRAL_TI_2];
            } else 
#endif
            {
              reduction->item(REDUCTION_BONDED_ENERGY_TI_1) += energies_virials[ComputeBondedCUDAKernel::energyIndex_DIHEDRAL_TI_1];
              reduction->item(REDUCTION_BONDED_ENERGY_TI_2) += energies_virials[ComputeBondedCUDAKernel::energyIndex_DIHEDRAL_TI_2];
            }
          }
          break;

        case Tuples::IMPROPER:
#ifdef NODEGROUP_FORCE_REGISTER
          if (params->CUDASOAintegrate) {
            nodeReduction->item(REDUCTION_IMPROPER_ENERGY) += energies_virials[ComputeBondedCUDAKernel::energyIndex_IMPROPER];
          } else
#endif
          {
            reduction->item(REDUCTION_IMPROPER_ENERGY) += energies_virials[ComputeBondedCUDAKernel::energyIndex_IMPROPER];
          }
          if (hostAlchFlags.alchFepOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_BONDED_ENERGY_F) += energies_virials[ComputeBondedCUDAKernel::energyIndex_IMPROPER_F];
            } else
#endif
            {
              reduction->item(REDUCTION_BONDED_ENERGY_F) += energies_virials[ComputeBondedCUDAKernel::energyIndex_IMPROPER_F];
            }
          }
          if (hostAlchFlags.alchThermIntOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_BONDED_ENERGY_TI_1) += energies_virials[ComputeBondedCUDAKernel::energyIndex_IMPROPER_TI_1];
              nodeReduction->item(REDUCTION_BONDED_ENERGY_TI_2) += energies_virials[ComputeBondedCUDAKernel::energyIndex_IMPROPER_TI_2];
            } else
#endif
            {
              reduction->item(REDUCTION_BONDED_ENERGY_TI_1) += energies_virials[ComputeBondedCUDAKernel::energyIndex_IMPROPER_TI_1];
              reduction->item(REDUCTION_BONDED_ENERGY_TI_2) += energies_virials[ComputeBondedCUDAKernel::energyIndex_IMPROPER_TI_2];
            }
          }
          break;

        case Tuples::EXCLUSION:
#ifdef NODEGROUP_FORCE_REGISTER
          if (params->CUDASOAintegrate) {
            nodeReduction->item(REDUCTION_ELECT_ENERGY)      += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT];
            nodeReduction->item(REDUCTION_LJ_ENERGY)         += energies_virials[ComputeBondedCUDAKernel::energyIndex_LJ];
            nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW) += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW];
          } else
#endif
          {
            reduction->item(REDUCTION_ELECT_ENERGY)      += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT];
            reduction->item(REDUCTION_LJ_ENERGY)         += energies_virials[ComputeBondedCUDAKernel::energyIndex_LJ];
            reduction->item(REDUCTION_ELECT_ENERGY_SLOW) += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW];
          }
          if (hostAlchFlags.alchFepOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_ELECT_ENERGY_F)    += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_F];
              nodeReduction->item(REDUCTION_LJ_ENERGY_F)           += energies_virials[ComputeBondedCUDAKernel::energyIndex_LJ_F];
              nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_F)   += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_F];
            } else
#endif
            {
              reduction->item(REDUCTION_ELECT_ENERGY_F)    += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_F];
              reduction->item(REDUCTION_LJ_ENERGY_F)           += energies_virials[ComputeBondedCUDAKernel::energyIndex_LJ_F];
              reduction->item(REDUCTION_ELECT_ENERGY_SLOW_F)   += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_F];
            }
          }
          if (hostAlchFlags.alchThermIntOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_ELECT_ENERGY_TI_1)     += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_TI_1];
              nodeReduction->item(REDUCTION_ELECT_ENERGY_TI_2)     += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_TI_2];
              nodeReduction->item(REDUCTION_LJ_ENERGY_TI_1)        += energies_virials[ComputeBondedCUDAKernel::energyIndex_LJ_TI_1];
              nodeReduction->item(REDUCTION_LJ_ENERGY_TI_2)        += energies_virials[ComputeBondedCUDAKernel::energyIndex_LJ_TI_2];
              nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1)+= energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_TI_1];
              nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2)+= energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_TI_2];
            } else
#endif
            {
              reduction->item(REDUCTION_ELECT_ENERGY_TI_1)     += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_TI_1];
              reduction->item(REDUCTION_ELECT_ENERGY_TI_2)     += energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_TI_2];
              reduction->item(REDUCTION_LJ_ENERGY_TI_1)        += energies_virials[ComputeBondedCUDAKernel::energyIndex_LJ_TI_1];
              reduction->item(REDUCTION_LJ_ENERGY_TI_2)        += energies_virials[ComputeBondedCUDAKernel::energyIndex_LJ_TI_2];
              reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1)+= energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_TI_1];
              reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2)+= energies_virials[ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_TI_2];
            }
          }
          break;

          case Tuples::CROSSTERM:
#ifdef NODEGROUP_FORCE_REGISTER
          if (params->CUDASOAintegrate) {
            nodeReduction->item(REDUCTION_CROSSTERM_ENERGY) += energies_virials[ComputeBondedCUDAKernel::energyIndex_CROSSTERM];
          } else
#endif
          {
            reduction->item(REDUCTION_CROSSTERM_ENERGY) += energies_virials[ComputeBondedCUDAKernel::energyIndex_CROSSTERM];
          }
          if (hostAlchFlags.alchFepOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_BONDED_ENERGY_F) += energies_virials[ComputeBondedCUDAKernel::energyIndex_CROSSTERM_F];
            } else
#endif
            {
              reduction->item(REDUCTION_BONDED_ENERGY_F) += energies_virials[ComputeBondedCUDAKernel::energyIndex_CROSSTERM_F];
            }
          }
          if (hostAlchFlags.alchThermIntOn) {
#ifdef NODEGROUP_FORCE_REGISTER
            if (params->CUDASOAintegrate) {
              nodeReduction->item(REDUCTION_BONDED_ENERGY_TI_1) += energies_virials[ComputeBondedCUDAKernel::energyIndex_CROSSTERM_TI_1];
              nodeReduction->item(REDUCTION_BONDED_ENERGY_TI_2) += energies_virials[ComputeBondedCUDAKernel::energyIndex_CROSSTERM_TI_2];
            } else
#endif
            {
              reduction->item(REDUCTION_BONDED_ENERGY_TI_1) += energies_virials[ComputeBondedCUDAKernel::energyIndex_CROSSTERM_TI_1];
              reduction->item(REDUCTION_BONDED_ENERGY_TI_2) += energies_virials[ComputeBondedCUDAKernel::energyIndex_CROSSTERM_TI_2];
            }
          }
          break;
          default:
          NAMD_bug("ComputeBondedCUDA::finishReductions, Unsupported tuple type");
          break;
        }
      }

      auto it = tupleList[tupletype].begin();
      if (!params->CUDASOAintegrate) {
        (*it)->submitTupleCount(reduction, numTuplesPerType[tupletype]);
      }
    }
  }

  if (doVirial) {
#ifndef WRITE_FULL_VIRIALS
#error "non-WRITE_FULL_VIRIALS not implemented"
#endif

#ifdef NODEGROUP_FORCE_REGISTER
    if (params->CUDASOAintegrate) {
      ADD_TENSOR(nodeReduction, REDUCTION_VIRIAL_NORMAL,energies_virials, ComputeBondedCUDAKernel::normalVirialIndex);
      ADD_TENSOR(nodeReduction, REDUCTION_VIRIAL_NBOND, energies_virials, ComputeBondedCUDAKernel::nbondVirialIndex);
      ADD_TENSOR(nodeReduction, REDUCTION_VIRIAL_SLOW,  energies_virials, ComputeBondedCUDAKernel::slowVirialIndex);
      ADD_TENSOR(nodeReduction, REDUCTION_VIRIAL_AMD_DIHE, energies_virials, ComputeBondedCUDAKernel::amdDiheVirialIndex);
      // NOTE: AMD_DIHE virial is also added to NORMAL virial.
      // This is what happens in ComputeDihedrals.C and ComputeCrossterms.C
      ADD_TENSOR(nodeReduction, REDUCTION_VIRIAL_NORMAL,   energies_virials, ComputeBondedCUDAKernel::amdDiheVirialIndex);
    } else
#endif
    {
      ADD_TENSOR(reduction, REDUCTION_VIRIAL_NORMAL,energies_virials, ComputeBondedCUDAKernel::normalVirialIndex);
      ADD_TENSOR(reduction, REDUCTION_VIRIAL_NBOND, energies_virials, ComputeBondedCUDAKernel::nbondVirialIndex);
      ADD_TENSOR(reduction, REDUCTION_VIRIAL_SLOW,  energies_virials, ComputeBondedCUDAKernel::slowVirialIndex);
      ADD_TENSOR(reduction, REDUCTION_VIRIAL_AMD_DIHE, energies_virials, ComputeBondedCUDAKernel::amdDiheVirialIndex);
      // NOTE: AMD_DIHE virial is also added to NORMAL virial.
      // This is what happens in ComputeDihedrals.C and ComputeCrossterms.C
      ADD_TENSOR(reduction, REDUCTION_VIRIAL_NORMAL,   energies_virials, ComputeBondedCUDAKernel::amdDiheVirialIndex);
    }
  }

#ifdef NODEGROUP_FORCE_REGISTER
  if (params->CUDASOAintegrate) {
    nodeReduction->item(REDUCTION_COMPUTE_CHECKSUM) += 1.;
  } else
#endif
  {
  reduction->item(REDUCTION_COMPUTE_CHECKSUM) += 1.;
  }

  if(!params->CUDASOAintegrate) reduction->submit();
}

//
// Can only be called by master PE
//
void ComputeBondedCUDA::initialize() {
#ifdef NODEGROUP_FORCE_REGISTER
  tupleWorkIndex.store(0);
#endif

  if (CkMyPe() != masterPe)
    NAMD_bug("ComputeBondedCUDA::initialize() called on non master PE");

  // Build list of PEs
  for (int rank=0;rank < computes.size();rank++) {
    if (computes[rank].selfComputes.size() > 0 || computes[rank].homeCompute.patchIDs.size() > 0) {
      pes.push_back(CkNodeFirst(CkMyNode()) + rank);
    }
  }

  // Return if no work to do
  if (pes.size() == 0) return;

  initializeCalled = false;
  cudaCheck(cudaSetDevice(deviceID));

#if CUDA_VERSION >= 5050 || defined(NAMD_HIP)
  int leastPriority, greatestPriority;
  cudaCheck(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  cudaCheck(cudaStreamCreateWithPriority(&stream, cudaStreamDefault, greatestPriority));
#else
  cudaCheck(cudaStreamCreate(&stream));
#endif
  cudaCheck(cudaEventCreate(&forceDoneEvent));
  lock = CmiCreateLock();
  printLock = CmiCreateLock();

  reduction = ReductionMgr::Object()->willSubmit(REDUCTIONS_BASIC);

  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();
  nodeReduction = patchData->reduction;

  PatchMap* patchMap = PatchMap::Object();
  // First, assign all patches in self computes.
  // NOTE: These never overlap between PEs. No proxies added.
  for (int rank=0;rank < computes.size();rank++) {
    std::vector< SelfCompute >& selfComputes = computes[rank].selfComputes;
    for (auto it=selfComputes.begin();it != selfComputes.end();it++) {
      for (auto jt=it->patchIDs.begin();jt != it->patchIDs.end();jt++) {
        if (!tuplePatchList.find( TuplePatchElem(*jt) ) ) {
          tuplePatchList.add( TuplePatchElem(*jt) );
          patchIDsPerRank[rank].push_back(*jt);
          allPatchIDs.push_back(*jt);
        }
      }
    }
  }

  // Second, assign all patches in home computes.
  // NOTE: The ranks always have these patches. No proxies added.
  for (int rank=0;rank < computes.size();rank++) {
    HomeCompute& homeCompute = computes[rank].homeCompute;
    std::vector<int>& patchIDs = homeCompute.patchIDs;
    for (int i=0;i < patchIDs.size();i++) {
      int patchID = patchIDs[i];
      if (!tuplePatchList.find( TuplePatchElem(patchID) ) ) {
        tuplePatchList.add( TuplePatchElem(patchID) );
        patchIDsPerRank[rank].push_back(patchID);
        allPatchIDs.push_back(patchID);
      }
    }
  }

  std::vector< std::vector<int> > patchIDsToAppend(CkMyNodeSize());
  // Find neighbors that are not added yet
  std::vector<int> neighborPids;
  for (int rank=0;rank < computes.size();rank++) {
    PatchID neighbors[PatchMap::MaxOneOrTwoAway];
    HomeCompute& homeCompute = computes[rank].homeCompute;
    std::vector<int>& patchIDs = homeCompute.patchIDs;
    for (int i=0;i < patchIDs.size();i++) {
      int patchID = patchIDs[i];
      int numNeighbors = patchMap->upstreamNeighbors(patchID, neighbors);
      for (int j=0;j < numNeighbors;j++) {
        if (!tuplePatchList.find( TuplePatchElem(neighbors[j]) ) ) {
          neighborPids.push_back(neighbors[j]);
        }
      }
    }
  }
  // Remove duplicates from neighborPids
  {
    std::sort(neighborPids.begin(), neighborPids.end());
    auto it_end = std::unique(neighborPids.begin(), neighborPids.end());
    neighborPids.resize(std::distance(neighborPids.begin(), it_end));
  }
  // Assign neighbors to the PEs on this node that have them
  for (int i=0;i < neighborPids.size();i++) {
    for (int rank=0;rank < computes.size();rank++) {
      int pid = neighborPids[i];
      int pe = rank + CkNodeFirst(CkMyNode());
      if (patchMap->node(pid) == pe) {
        // Patch pid found on PE "pe" on this node
        tuplePatchList.add( TuplePatchElem(pid) );
        patchIDsPerRank[rank].push_back(pid);
        allPatchIDs.push_back(pid);
        // Add to this rank's patches
        patchIDsToAppend[rank].push_back(pid);
        // Add to the list of PEs
        pes.push_back(CkNodeFirst(CkMyNode()) + rank);
        break;
      }
    }
  }
  // Remove duplicates from pes
  {
    std::sort(pes.begin(), pes.end());
    auto it_end = std::unique(pes.begin(), pes.end());
    pes.resize(std::distance(pes.begin(), it_end));
  }

  // Last, assign all patches in neighbors of home computes
  // NOTE: Will create proxies on multiple nodes
  for (int rank=0;rank < computes.size();rank++) {
    PatchID neighbors[PatchMap::MaxOneOrTwoAway];
    HomeCompute& homeCompute = computes[rank].homeCompute;
    std::vector<int>& patchIDs = homeCompute.patchIDs;
    std::vector<int> neighborPatchIDs;
    for (int i=0;i < patchIDs.size();i++) {
      int patchID = patchIDs[i];
      int numNeighbors = patchMap->upstreamNeighbors(patchID, neighbors);
      for (int j=0;j < numNeighbors;j++) {
        if (!tuplePatchList.find( TuplePatchElem(neighbors[j]) ) ) {
          // Patch not found => Add Proxy
          tuplePatchList.add( TuplePatchElem(neighbors[j]) );
          patchIDsPerRank[rank].push_back(neighbors[j]);
          allPatchIDs.push_back(neighbors[j]);
        }
        if ( std::count(patchIDs.begin(), patchIDs.end(), neighbors[j]) == 0
          && std::count(neighborPatchIDs.begin(), neighborPatchIDs.end(), neighbors[j]) == 0 ) {
          neighborPatchIDs.push_back(neighbors[j]);
        }
      }
    }
    // Append neighboring patchIDs to homeCompute.patchIDs
    // int start = patchIDs.size();
    // patchIDs.resize(patchIDs.size() + neighborPatchIDs.size());
    // for (int i=0;i < neighborPatchIDs.size();i++) {
    //   patchIDs[start + i] = neighborPatchIDs[i];
    // }
    for (int i=0;i < neighborPatchIDs.size();i++) {
      patchIDsToAppend[rank].push_back(neighborPatchIDs[i]);
    }
  }

  for (int rank=0;rank < patchIDsToAppend.size();rank++) {
    for (int i=0;i < patchIDsToAppend[rank].size();i++) {
      computes[rank].homeCompute.patchIDs.push_back(patchIDsToAppend[rank][i]);
    }
  }

  // Remove duplicate patch IDs
  {
    std::sort(allPatchIDs.begin(), allPatchIDs.end());
    auto it_end = std::unique(allPatchIDs.begin(), allPatchIDs.end());
    allPatchIDs.resize(std::distance(allPatchIDs.begin(), it_end));
  }

  // Set number of (unique) patches
  setNumPatches(allPatchIDs.size());

  // Reset patchesCounter
  patchesCounter = getNumPatches();

  patches.resize(getNumPatches());
  // Setup tupleList
  // tuple list never gets updated here, where the hell is it updated?
  for (int rank=0;rank < computes.size();rank++) {
    std::vector< SelfCompute >& selfComputes = computes[rank].selfComputes;
    for (auto it=selfComputes.begin();it != selfComputes.end();it++) {
      tupleList[it->tuples->getType()].push_back(it->tuples);
    }
    HomeCompute& homeCompute = computes[rank].homeCompute;
    for (int i=0;i < homeCompute.tuples.size();i++) {
      tupleList[homeCompute.tuples[i]->getType()].push_back(homeCompute.tuples[i]);
    }
  }

  // Allocate host memory for energies and virials
  allocate_host<double>(&energies_virials, ComputeBondedCUDAKernel::energies_virials_SIZE);

  // Finally, do sanity checks
  std::vector<char> patchIDset(patchMap->numPatches(), 0);
  int numPatchIDset = 0;
  int numPatchIDs = 0;
  for (int rank=0;rank < computes.size();rank++) {
    numPatchIDs += patchIDsPerRank[rank].size();
    for (int i=0;i < patchIDsPerRank[rank].size();i++) {
      PatchID patchID = patchIDsPerRank[rank][i];
      if (patchIDset[patchID] == 0) numPatchIDset++;
      patchIDset[patchID] = 1;
      if ( !std::count(allPatchIDs.begin(), allPatchIDs.end(), patchID) ) {
        NAMD_bug("ComputeBondedCUDA::initialize(), inconsistent patch mapping");
      }
    }
  }
  if (numPatchIDs != getNumPatches() || numPatchIDset != getNumPatches()) {
    NAMD_bug("ComputeBondedCUDA::initialize(), inconsistent patch mapping");
  }

  // Warning: Direct indexing used, patchIndex could use up a lot of memory for large systems
  patchIndex.resize(patchMap->numPatches());
  atomMappers.resize(getNumPatches());
  for (int i=0;i < getNumPatches();i++) {
    atomMappers[i] = new AtomMapper(allPatchIDs[i], &atomMap);
    patchIndex[allPatchIDs[i]] = i;
  }

    // Initialize the alchemical flags, parameters and lambda values
  updateHostCudaAlchFlags();
  updateKernelCudaAlchFlags();
  if (hostAlchFlags.alchOn) {
    updateHostCudaAlchParameters();
    bondedKernel.updateCudaAlchParameters(&hostAlchParameters, stream);
    updateHostCudaAlchLambdas();
    bondedKernel.updateCudaAlchLambdas(&hostAlchLambdas, stream);
    if (hostAlchFlags.alchDecouple) {
      pswitchTable[1+3*1] = 0;
      pswitchTable[2+3*2] = 0;
    }
  }

  // Copy coefficients to GPU
  Parameters* parameters = Node::Object()->parameters;
  for (int tupletype=0;tupletype < Tuples::NUM_TUPLE_TYPES;tupletype++) {
    if (tupleList[tupletype].size() > 0) {
      switch(tupletype) {

        case Tuples::BOND:
        {
          int NumBondParams = parameters->NumBondParams;
          BondValue* bond_array = parameters->bond_array;
          std::vector<CudaBondValue> bondValues(NumBondParams);
          for (int i=0;i < NumBondParams;i++) {
            bondValues[i].k  = bond_array[i].k;
            bondValues[i].x0 = bond_array[i].x0;
            bondValues[i].x1 = bond_array[i].x1;
          }
          bondedKernel.setupBondValues(NumBondParams, bondValues.data());
        }
        break;

        case Tuples::ANGLE:
        {
          int NumAngleParams = parameters->NumAngleParams;
          AngleValue* angle_array = parameters->angle_array;
          std::vector<CudaAngleValue> angleValues(NumAngleParams);
          bool normal_ub_error = false;
          for (int i=0;i < NumAngleParams;i++) {
            angleValues[i].k      = angle_array[i].k;
            if (angle_array[i].normal == 1) {
              angleValues[i].theta0 = angle_array[i].theta0;
            } else {
              angleValues[i].theta0 = cos(angle_array[i].theta0);
            }
            normal_ub_error |= (angle_array[i].normal == 0 && angle_array[i].k_ub);
            angleValues[i].k_ub   = angle_array[i].k_ub;
            angleValues[i].r_ub   = angle_array[i].r_ub;
            angleValues[i].normal = angle_array[i].normal;
          }
          if (normal_ub_error) NAMD_die("ERROR: Can't use cosAngles with Urey-Bradley angles");
          bondedKernel.setupAngleValues(NumAngleParams, angleValues.data());
        }
        break;

        case Tuples::DIHEDRAL:
        {
          int NumDihedralParams = parameters->NumDihedralParams;
          DihedralValue* dihedral_array = parameters->dihedral_array;
          int NumDihedralParamsMult = 0;
          for (int i=0;i < NumDihedralParams;i++) {
            NumDihedralParamsMult += std::max(0, dihedral_array[i].multiplicity);
          }
          std::vector<CudaDihedralValue> dihedralValues(NumDihedralParamsMult);
          dihedralMultMap.resize(NumDihedralParams);
          int k = 0;
          for (int i=0;i < NumDihedralParams;i++) {
            int multiplicity = dihedral_array[i].multiplicity;
            dihedralMultMap[i] = k;
            for (int j=0;j < multiplicity;j++) {
              dihedralValues[k].k     = dihedral_array[i].values[j].k;
              dihedralValues[k].n     = (dihedral_array[i].values[j].n << 1) | (j < (multiplicity - 1));
              dihedralValues[k].delta = dihedral_array[i].values[j].delta;
              k++;
            }
          }
          bondedKernel.setupDihedralValues(NumDihedralParamsMult, dihedralValues.data());
        }
        break;

        case Tuples::IMPROPER:
        {
          int NumImproperParams = parameters->NumImproperParams;
          ImproperValue* improper_array = parameters->improper_array;
          int NumImproperParamsMult = 0;
          for (int i=0;i < NumImproperParams;i++) {
            NumImproperParamsMult += std::max(0, improper_array[i].multiplicity);
          }
          std::vector<CudaDihedralValue> improperValues(NumImproperParamsMult);
          improperMultMap.resize(NumImproperParams);
          int k = 0;
          for (int i=0;i < NumImproperParams;i++) {
            int multiplicity = improper_array[i].multiplicity;
            improperMultMap[i] = k;
            for (int j=0;j < multiplicity;j++) {
              improperValues[k].k     = improper_array[i].values[j].k;
              improperValues[k].n     = (improper_array[i].values[j].n << 1) | (j < (multiplicity - 1));
              improperValues[k].delta = improper_array[i].values[j].delta;
              k++;
            }
          }
          bondedKernel.setupImproperValues(NumImproperParamsMult, improperValues.data());
        }
        break;

        case Tuples::CROSSTERM:
        {
          int NumCrosstermParams = parameters->NumCrosstermParams;
          CrosstermValue* crossterm_array = parameters->crossterm_array;
          std::vector<CudaCrosstermValue> crosstermValues(NumCrosstermParams);
          const int D = CrosstermValue::dim;
          const int N = CrosstermValue::dim - 1;
          for (int ipar=0;ipar < NumCrosstermParams;ipar++) {
            for (int i=0;i < N;i++) {
              for (int j=0;j < N;j++) {

                // Setups coefficients for bi-cubic interpolation.
                // See https://en.wikipedia.org/wiki/Bicubic_interpolation

                #define INDEX(ncols,i,j)  ((i)*ncols + (j))
                CrosstermData* table = &crossterm_array[ipar].c[0][0];

                const double Ainv[16][16] = {
                  { 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
                  { 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
                  {-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
                  { 2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
                  { 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
                  { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
                  { 0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0},
                  { 0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0},
                  {-3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0},
                  { 0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0},
                  { 9, -9, -9,  9,  6,  3, -6, -3,  6, -6,  3, -3,  4,  2,  2,  1},
                  {-6,  6,  6, -6, -3, -3,  3,  3, -4,  4, -2,  2, -2, -2, -1, -1},
                  { 2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0},
                  { 0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0},
                  {-6,  6,  6, -6, -4, -2,  4,  2, -3,  3, -3,  3, -2, -1, -2, -1},
                  { 4, -4, -4,  4,  2,  2, -2, -2,  2, -2,  2, -2,  1,  1,  1,  1}
                };

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

                const double h = M_PI/12.0;

                const double x[16] = {
                  table[INDEX(D,i,j)].d00, table[INDEX(D,i+1,j)].d00, table[INDEX(D,i,j+1)].d00, table[INDEX(D,i+1,j+1)].d00,
                  table[INDEX(D,i,j)].d10*h, table[INDEX(D,i+1,j)].d10*h, table[INDEX(D,i,j+1)].d10*h, table[INDEX(D,i+1,j+1)].d10*h,
                  table[INDEX(D,i,j)].d01*h, table[INDEX(D,i+1,j)].d01*h, table[INDEX(D,i,j+1)].d01*h, table[INDEX(D,i+1,j+1)].d01*h,
                  table[INDEX(D,i,j)].d11*h*h, table[INDEX(D,i+1,j)].d11*h*h, table[INDEX(D,i,j+1)].d11*h*h, table[INDEX(D,i+1,j+1)].d11*h*h
                };

                // a = Ainv*x
                float* a = (float *)&crosstermValues[ipar].c[i][j][0];
                for (int k=0;k < 16;k++) {
                  double a_val = 0.0;
                  for (int l=0;l < 16;l++) {
                    a_val += Ainv[k][l]*x[l];
                  }
                  a[k] = (float)a_val;
                }

              }
            }
          }
          bondedKernel.setupCrosstermValues(NumCrosstermParams, crosstermValues.data());
        }
        break;

        case Tuples::EXCLUSION:
        // Nothing to do
        break;

        default:
        NAMD_bug("ComputeBondedCUDA::initialize, Undefined tuple type");
        break;
      }
    }
  }

  computeMgr->sendAssignPatchesOnPe(pes, this);

  int nDevices = deviceCUDA->getNumDevice();

#ifdef NODEGROUP_FORCE_REGISTER
  // TODO DMC this isn't the number of home patches, but that is okay for now...
  if (params->CUDASOAintegrateMode && params->useDeviceMigration) {
    migrationKernel.setup(nDevices, patchMap->numPatches());

    // Tuple Migration Structures
    // Patch record Data
    allocate_host<PatchRecord>(&h_patchRecord, patchMap->numPatches());
    allocate_device<PatchRecord>(&d_patchRecord, patchMap->numPatches());

    // Patch Centers
    allocate_host<double3>(&h_patchMapCenter, patchMap->numPatches());
    allocate_device<double3>(&d_patchMapCenter, patchMap->numPatches());
    for (int i = 0; i < patchMap->numPatches(); i++) {
      COPY_CUDAVECTOR(patchMap->center(i), h_patchMapCenter[i]); 
    }
    copy_HtoD_sync<double3>(h_patchMapCenter, d_patchMapCenter, patchMap->numPatches());
  }
#endif  // NODEGROUP_FORCE_REGISTER
}

#ifdef NODEGROUP_FORCE_REGISTER
void ComputeBondedCUDA::updatePatchOrder(std::vector<CudaLocalRecord>& data) {
  // TODO is there anywhere else that the order matters??
  // DMC This vector of CudaLocalRecords doesn't have the correct number of peer records
  std::vector<int>& patchIDs = patchIDsPerRank[CkMyRank()];
  for (int i=0;i < data.size();i++) {
    patchIndex[data[i].patchID] = i;
  }

  patches.clear();
  for (int i=0;i < data.size();i++) {
    PatchRecord p;
    p.patchID = data[i].patchID;
    p.atomStart = 0;
    p.numAtoms = 0;
    patches.push_back(p);
  }
}
#endif  // NODEGROUP_FORCE_REGISTER

void ComputeBondedCUDA::updateHostCudaAlchFlags() {
  const SimParameters& sim_params     = *(Node::Object()->simParameters);
  hostAlchFlags.alchOn                = sim_params.alchOn;
  hostAlchFlags.alchFepOn             = sim_params.alchFepOn;
  hostAlchFlags.alchThermIntOn        = sim_params.alchThermIntOn;
  hostAlchFlags.alchWCAOn             = sim_params.alchWCAOn;
  hostAlchFlags.alchLJcorrection      = sim_params.LJcorrection;
  hostAlchFlags.alchVdwForceSwitching = ComputeNonbondedUtil::vdwForceSwitching;
  hostAlchFlags.alchDecouple          = sim_params.alchDecouple;
  hostAlchFlags.alchBondDecouple      = sim_params.alchBondDecouple;
}

void ComputeBondedCUDA::updateKernelCudaAlchFlags() {
  bondedKernel.updateCudaAlchFlags(hostAlchFlags);
}

void ComputeBondedCUDA::updateHostCudaAlchParameters() {
  const SimParameters& sim_params      = *(Node::Object()->simParameters);
  hostAlchParameters.switchDist2       = ComputeNonbondedUtil::switchOn * ComputeNonbondedUtil::switchOn;
  hostAlchParameters.alchVdwShiftCoeff = sim_params.alchVdwShiftCoeff;
}

void ComputeBondedCUDA::updateKernelCudaAlchParameters() {
  bondedKernel.updateCudaAlchParameters(&hostAlchParameters, stream);
}

void ComputeBondedCUDA::updateHostCudaAlchLambdas() {
  SimParameters& sim_params     = *(Node::Object()->simParameters);
  hostAlchLambdas.currentLambda       = float(sim_params.getCurrentLambda(step));
  hostAlchLambdas.currentLambda2      = float(sim_params.getCurrentLambda2(step));
  hostAlchLambdas.bondLambda1         = float(sim_params.getBondLambda(hostAlchLambdas.currentLambda));
  hostAlchLambdas.bondLambda2         = float(sim_params.getBondLambda(1.0 - hostAlchLambdas.currentLambda));
  hostAlchLambdas.bondLambda12        = float(sim_params.getBondLambda(hostAlchLambdas.currentLambda2));
  hostAlchLambdas.bondLambda22        = float(sim_params.getBondLambda(1.0 - hostAlchLambdas.currentLambda2));
  hostAlchLambdas.elecLambdaUp        = float(sim_params.getElecLambda(hostAlchLambdas.currentLambda));
  hostAlchLambdas.elecLambda2Up       = float(sim_params.getElecLambda(hostAlchLambdas.currentLambda2));
  hostAlchLambdas.elecLambdaDown      = float(sim_params.getElecLambda(1.0 - hostAlchLambdas.currentLambda));
  hostAlchLambdas.elecLambda2Down     = float(sim_params.getElecLambda(1.0 - hostAlchLambdas.currentLambda2));
  hostAlchLambdas.vdwLambdaUp         = float(sim_params.getVdwLambda(hostAlchLambdas.currentLambda));
  hostAlchLambdas.vdwLambda2Up        = float(sim_params.getVdwLambda(hostAlchLambdas.currentLambda2));
  hostAlchLambdas.vdwLambdaDown       = float(sim_params.getVdwLambda(1.0 - hostAlchLambdas.currentLambda));
  hostAlchLambdas.vdwLambda2Down      = float(sim_params.getVdwLambda(1.0 - hostAlchLambdas.currentLambda2));
}

void ComputeBondedCUDA::updateKernelCudaAlchLambdas() {
  bondedKernel.updateCudaAlchLambdas(&hostAlchLambdas, stream);
}

#ifdef NODEGROUP_FORCE_REGISTER
void ComputeBondedCUDA::registerPointersToHost() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();
  
  const int deviceIndex = deviceCUDA->getDeviceIndex();

  TupleDataStage dst = migrationKernel.getDstBuffers();
  patchData->h_tupleDataStage.bond[deviceIndex] = dst.bond;
  patchData->h_tupleDataStage.angle[deviceIndex] = dst.angle;
  patchData->h_tupleDataStage.dihedral[deviceIndex] = dst.dihedral;
  patchData->h_tupleDataStage.improper[deviceIndex] = dst.improper;
  patchData->h_tupleDataStage.modifiedExclusion[deviceIndex] = dst.modifiedExclusion;
  patchData->h_tupleDataStage.exclusion[deviceIndex] = dst.exclusion;
  patchData->h_tupleDataStage.crossterm[deviceIndex] = dst.crossterm;

  TupleIntArraysContiguous count = migrationKernel.getDeviceTupleCounts();
  patchData->h_tupleCount.bond[deviceIndex] = count.bond();
  patchData->h_tupleCount.angle[deviceIndex] = count.angle();
  patchData->h_tupleCount.dihedral[deviceIndex] = count.dihedral();
  patchData->h_tupleCount.improper[deviceIndex] = count.improper();
  patchData->h_tupleCount.modifiedExclusion[deviceIndex] = count.modifiedExclusion();
  patchData->h_tupleCount.exclusion[deviceIndex] = count.exclusion();
  patchData->h_tupleCount.crossterm[deviceIndex] = count.crossterm();

  TupleIntArraysContiguous offset = migrationKernel.getDeviceTupleOffsets();
  patchData->h_tupleOffset.bond[deviceIndex] = offset.bond();
  patchData->h_tupleOffset.angle[deviceIndex] = offset.angle();
  patchData->h_tupleOffset.dihedral[deviceIndex] = offset.dihedral();
  patchData->h_tupleOffset.improper[deviceIndex] = offset.improper();
  patchData->h_tupleOffset.modifiedExclusion[deviceIndex] = offset.modifiedExclusion();
  patchData->h_tupleOffset.exclusion[deviceIndex] = offset.exclusion();
  patchData->h_tupleOffset.crossterm[deviceIndex] = offset.crossterm();
}

void ComputeBondedCUDA::copyHostRegisterToDevice() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();

  const int deviceIndex = deviceCUDA->getDeviceIndex();
  const int nDevices = deviceCUDA->getNumDevice();
  const int deviceID = deviceCUDA->getDeviceID();
  cudaCheck(cudaSetDevice(deviceID));

  migrationKernel.copyPeerDataToDevice(
    patchData->h_tupleDataStage, 
    patchData->h_tupleCount, 
    patchData->h_tupleOffset, 
    nDevices, 
    stream
  );
}

void ComputeBondedCUDA::copyPatchData() {
  PatchMap* patchMap = PatchMap::Object();
  int numPatches = patchMap->numPatches();
  // Constructs a patch record array that is indexed by global patchID
  for (int i = 0; i < numPatches; i++) {
    h_patchRecord[i] = patches[patchIndex[i]];
  }

  copy_HtoD<PatchRecord>(h_patchRecord, d_patchRecord, numPatches, stream);
}
#endif  // NODEGROUP_FORCE_REGISTER


#endif // BONDED_CUDA
#endif // NAMD_CUDA
