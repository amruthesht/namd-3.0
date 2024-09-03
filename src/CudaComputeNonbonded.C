#include <algorithm>
#include <map>
#include <vector>
#include "NamdTypes.h"
#include "charm++.h"
#include "Patch.h"
#include "PatchMap.h"
#include "ProxyMgr.h"
#include "LJTable.h"
#include "Node.h"
#include "ObjectArena.h"
// #include "ComputeCUDAMgr.h"
#include "ReductionMgr.h"
#include "CudaComputeNonbonded.h"
#include "WorkDistrib.h"
#include "HomePatch.h"
#include "Priorities.h"
#include "ComputePmeCUDAMgr.h"
#include "ComputeNonbondedUtil.h"
#include "PatchData.h"
//#include "CudaUtils.h"

#include "NamdEventsProfiling.h"

#include "DeviceCUDA.h"
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#ifdef WIN32
#define __thread __declspec(thread)
#endif
extern __thread DeviceCUDA *deviceCUDA;
#endif

#if defined(NAMD_CUDA) || defined(NAMD_HIP)

extern "C" void CcdCallBacksReset(void *ignored, double curWallTime);  // fix Charm++
//
// Class constructor
//
CudaComputeNonbonded::CudaComputeNonbonded(ComputeID c, int deviceID,
  CudaNonbondedTables& cudaNonbondedTables, bool doStreaming) :
Compute(c), deviceID(deviceID), doStreaming(doStreaming), nonbondedKernel(deviceID, cudaNonbondedTables, doStreaming),
tileListKernel(deviceID, doStreaming), GBISKernel(deviceID) {

  cudaCheck(cudaSetDevice(deviceID));

	exclusionsByAtom = NULL;

  vdwTypes = NULL;
  vdwTypesSize = 0;

  exclIndexMaxDiff = NULL;
  exclIndexMaxDiffSize = 0;

  atomIndex = NULL;
  atomIndexSize = 0;

  atomStorageSize = 0;

  // Atom and charge storage
  atoms = NULL;
  atomsSize = 0;
  part = NULL;
  partSize = 0;
  doAlch = false;
  lambdaWindowUpdated = false;

  // Force storage
  h_forces = NULL;
  h_forcesSize = 0;
  h_forcesSlow = NULL;
  h_forcesSlowSize = 0;

  d_forces = NULL;
  d_forcesSize = 0;
  d_forcesSlow = NULL;
  d_forcesSlowSize = 0;

  // GBIS
  intRad0H = NULL;
  intRad0HSize = 0;
  intRadSH = NULL;
  intRadSHSize = 0;
  psiSumH = NULL;
  psiSumHSize = 0;
  bornRadH = NULL;
  bornRadHSize = 0;
  dEdaSumH = NULL;
  dEdaSumHSize = 0;
  dHdrPrefixH = NULL;
  dHdrPrefixHSize = 0;
  maxShmemPerBlock = 0;
  cudaPatches = NULL;

  atomsChangedIn = true;
  atomsChanged = true;
  computesChanged = true;

  forceDoneEventRecord = false;

  SimParameters *simParams = Node::Object()->simParameters;
  if (simParams->pressureProfileOn) {
    NAMD_die("CudaComputeNonbonded, pressure profile not supported");
  }
  
  if (simParams->GBISOn) gbisPhase = 3;

  doSkip = false;
}

//
// Class destructor
//
CudaComputeNonbonded::~CudaComputeNonbonded() {
  // fprintf(stderr, "Pe %d calling destructor ", CkMyPe());
  cudaCheck(cudaSetDevice(deviceID));
	if (exclusionsByAtom != NULL) delete [] exclusionsByAtom;
  if (vdwTypes != NULL) deallocate_host<int>(&vdwTypes);
  if (exclIndexMaxDiff != NULL) deallocate_host<int2>(&exclIndexMaxDiff);
  if (atoms != NULL) deallocate_host<CudaAtom>(&atoms);
  if (part != NULL) deallocate_host<char>(&part);
  if (h_forces != NULL) deallocate_host<float4>(&h_forces);
  if (h_forcesSlow != NULL) deallocate_host<float4>(&h_forcesSlow);
  if (d_forces != NULL) deallocate_device<float4>(&d_forces);
  if (d_forcesSlow != NULL) deallocate_device<float4>(&d_forcesSlow);

  // GBIS
  if (intRad0H != NULL) deallocate_host<float>(&intRad0H);
  if (intRadSH != NULL) deallocate_host<float>(&intRadSH);
  if (psiSumH != NULL) deallocate_host<GBReal>(&psiSumH);
  if (bornRadH != NULL) deallocate_host<float>(&bornRadH);
  if (dEdaSumH != NULL) deallocate_host<GBReal>(&dEdaSumH);
  if (dHdrPrefixH != NULL) deallocate_host<float>(&dHdrPrefixH);

  if (cudaPatches != NULL) deallocate_host<CudaPatchRecord>(&cudaPatches);

  if (patches.size() > 0) {
    deallocate_host<VirialEnergy>(&h_virialEnergy);
    deallocate_device<VirialEnergy>(&d_virialEnergy);
    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaEventDestroy(forceDoneEvent));
    CmiDestroyLock(lock);
    delete reduction;
  }

  // NOTE: unregistering happens in [sync] -entry method
  // fprintf(stderr, "unregistering patches on pe %d\n", CkMyPe());
  computeMgr->sendUnregisterBoxesOnPe(pes, this);

}

void CudaComputeNonbonded::unregisterBox(int i) {
  if (patches[i].positionBox != NULL) patches[i].patch->unregisterPositionPickup(this, &patches[i].positionBox);
  if (patches[i].forceBox != NULL) patches[i].patch->unregisterForceDeposit(this, &patches[i].forceBox);
  if (patches[i].intRadBox != NULL) patches[i].patch->unregisterIntRadPickup(this, &patches[i].intRadBox);
  if (patches[i].psiSumBox != NULL) patches[i].patch->unregisterPsiSumDeposit(this, &patches[i].psiSumBox);
  if (patches[i].bornRadBox != NULL) patches[i].patch->unregisterBornRadPickup(this, &patches[i].bornRadBox);
  if (patches[i].dEdaSumBox != NULL) patches[i].patch->unregisterDEdaSumDeposit(this, &patches[i].dEdaSumBox);
  if (patches[i].dHdrPrefixBox != NULL) patches[i].patch->unregisterDHdrPrefixPickup(this, &patches[i].dHdrPrefixBox);
}

void CudaComputeNonbonded::unregisterBoxesOnPe() {
  if (rankPatches[CkMyRank()].size() == 0)
    NAMD_bug("CudaComputeNonbonded::unregisterBoxesOnPe, empty rank");
  for (int i=0;i < rankPatches[CkMyRank()].size();i++) {
    unregisterBox(rankPatches[CkMyRank()][i]);
  }
}

//
// Register inter-patch (self) compute.
// Only serialized calls allowed
//
void CudaComputeNonbonded::registerComputeSelf(ComputeID cid, PatchID pid) {
  computesChanged = true;
  addPatch(pid);
  addCompute(cid, pid, pid, 0.);
}

//
// Register pair-patch compute.
// Only serialized calls allowed
//
void CudaComputeNonbonded::registerComputePair(ComputeID cid, PatchID* pid, int* trans) {
  computesChanged = true;
  addPatch(pid[0]);
  addPatch(pid[1]);
  PatchMap* patchMap = PatchMap::Object();
  int t1 = trans[0];
  int t2 = trans[1];
  Vector offset = patchMap->center(pid[0]) - patchMap->center(pid[1]);
  offset.x += (t1%3-1) - (t2%3-1);
  offset.y += ((t1/3)%3-1) - ((t2/3)%3-1);
  offset.z += (t1/9-1) - (t2/9-1);
  addCompute(cid, pid[0], pid[1], offset);
}

//
// Add patch
//
void CudaComputeNonbonded::addPatch(PatchID pid) {
  patches.push_back(PatchRecord(pid));
}

//
// Add compute
//
void CudaComputeNonbonded::addCompute(ComputeID cid, PatchID pid1, PatchID pid2, Vector offset) {
  ComputeRecord cr;
  cr.cid = cid;
  cr.pid[0] = pid1;
  cr.pid[1] = pid2;
  cr.offset = offset;
  computes.push_back(cr);
}

//
// Update numAtoms and numFreeAtoms on a patch
//
void CudaComputeNonbonded::updatePatch(int i) {
  int numAtoms = patches[i].patch->getNumAtoms();
  int numFreeAtoms = numAtoms;
  if ( fixedAtomsOn ) {
    const CompAtomExt *aExt = patches[i].patch->getCompAtomExtInfo();
    for ( int j=0; j< numAtoms; ++j ) {
      if ( aExt[j].atomFixed ) --numFreeAtoms;
    }
  }
  patches[i].numAtoms = numAtoms;
  patches[i].numFreeAtoms = numFreeAtoms;
  cudaPatches[i].numAtoms = numAtoms;
  cudaPatches[i].numFreeAtoms = numFreeAtoms;
#ifdef NODEGROUP_FORCE_REGISTER
  cudaPatches[i].patchID = patches[i].patchID;
#endif
}

int CudaComputeNonbonded::findPid(PatchID pid) {
  for (int i=0;i < rankPatches[CkMyRank()].size();i++) {
    int j = rankPatches[CkMyRank()][i];
    if (patches[j].patchID == pid) return j;
  }
  return -1;
}

void CudaComputeNonbonded::patchReady(PatchID pid, int doneMigration, int seq) {
  // DMC: This isn't need into CUDASOAintegrate scheme. All it does is call atomUpdate()
  // however that is already called in Sequencer::runComputeObjects_CUDA
  // The functionality of updatePatch() was moved into updatePatches()
  if (!(params->CUDASOAintegrate && params->useDeviceMigration)) {
    if (doneMigration) {
      int i = findPid(pid);
      if (i == -1)
        NAMD_bug("CudaComputeNonbonded::patchReady, Patch ID not found");
      updatePatch(i);
    }
    CmiLock(lock);
    Compute::patchReady(pid, doneMigration, seq);
    CmiUnlock(lock);
  }
}

void CudaComputeNonbonded::gbisP2PatchReady(PatchID pid, int seq) {
  CmiLock(lock);
  Compute::gbisP2PatchReady(pid, seq);
  CmiUnlock(lock);
}

void CudaComputeNonbonded::gbisP3PatchReady(PatchID pid, int seq) {
  CmiLock(lock);
  Compute::gbisP3PatchReady(pid, seq);
  CmiUnlock(lock);
}

void CudaComputeNonbonded::assignPatch(int i) {
  
  PatchMap* patchMap = PatchMap::Object();
  PatchID pid = patches[i].patchID;
  Patch* patch = patchMap->patch(pid);
  if (patch == NULL) {
    // Create ProxyPatch if none exists
    ProxyMgr::Object()->createProxy(pid);
    patch = patchMap->patch(pid);
  }
  patches[i].patch = patch;
  if (patches[i].patch == NULL) {
    NAMD_bug("CudaComputeNonbonded::assignPatch, patch not found");
  }
  patches[i].positionBox = patches[i].patch->registerPositionPickup(this);
  patches[i].forceBox    = patches[i].patch->registerForceDeposit(this);
  SimParameters *simParams = Node::Object()->simParameters;
  if (simParams->GBISOn) {
    patches[i].intRadBox     = patches[i].patch->registerIntRadPickup(this);
    patches[i].psiSumBox     = patches[i].patch->registerPsiSumDeposit(this);
    patches[i].bornRadBox    = patches[i].patch->registerBornRadPickup(this);
    patches[i].dEdaSumBox    = patches[i].patch->registerDEdaSumDeposit(this);
    patches[i].dHdrPrefixBox = patches[i].patch->registerDHdrPrefixPickup(this);
  }
  // Store Pe where this patch was registered
#if 1
  if (patches[i].pe != CkMyPe()) {
    NAMD_bug("CudaComputeNonbonded::assignPatch, patch assigned to incorrect Pe");
  }
#else
  patches[i].pe = CkMyPe();
#endif
  //
  patches[i].isSamePhysicalNode = ( CmiPhysicalNodeID(patchMap->node(pid)) == CmiPhysicalNodeID(CkMyPe()) );
  patches[i].isSameNode = ( CkNodeOf(patchMap->node(pid)) == CkMyNode() );
}

struct pid_sortop_reverse_priority {
  bool operator() (int2 pidj, int2 pidi) {  // i and j reversed
    int ppi = PATCH_PRIORITY(pidi.x);
    int ppj = PATCH_PRIORITY(pidj.x);
    if ( ppi != ppj ) return ppi < ppj;
    return pidi.x < pidj.x;
  }
};

void CudaComputeNonbonded::assignPatchesOnPe() {
  if (rankPatches[CkMyRank()].size() == 0)
    NAMD_bug("CudaComputeNonbonded::assignPatchesOnPe, empty rank");

  // calculate priority rank of local home patch within pe
  {
    PatchMap* patchMap = PatchMap::Object();
    ResizeArray< ResizeArray<int2> > homePatchByRank(CkMyNodeSize());
    for ( int k=0; k < rankPatches[CkMyRank()].size(); ++k ) {
      int i = rankPatches[CkMyRank()][k];
      int pid = patches[i].patchID;
      int homePe = patchMap->node(pid);
      if ( CkNodeOf(homePe) == CkMyNode() ) {
        int2 pid_index;
        pid_index.x = pid;
        pid_index.y = i;
        homePatchByRank[CkRankOf(homePe)].add(pid_index);
      }
    }
    for ( int i=0; i<CkMyNodeSize(); ++i ) {
      pid_sortop_reverse_priority so;
      std::sort(homePatchByRank[i].begin(),homePatchByRank[i].end(),so);
      int masterBoost = ( CkMyRank() == i ? 2 : 0 );
      for ( int j=0; j<homePatchByRank[i].size(); ++j ) {
        int index = homePatchByRank[i][j].y;
        patches[index].reversePriorityRankInPe = j + masterBoost;
      }
    }
  }

  for (int i=0;i < rankPatches[CkMyRank()].size();i++) {
    assignPatch(rankPatches[CkMyRank()][i]);
  }
}

//
// Returns Pe of Patch ID "pid", -1 otherwise
//
// int findHomePatchPe(std::vector<PatchIDList>& rankPatchIDs, PatchID pid) {
int findHomePatchPe(PatchIDList* rankPatchIDs, PatchID pid) {
  // for (int i=0;i < rankPatchIDs.size();i++) {
  for (int i=0;i < CkMyNodeSize();i++) {
    if (rankPatchIDs[i].find(pid) != -1) return CkNodeFirst(CkMyNode()) + i;
  }
  return -1;
}

//
// Find all PEs that have Patch
//
void findProxyPatchPes(std::vector<int>& proxyPatchPes, PatchID pid) {
  proxyPatchPes.clear();
  for (int i=0;i < CkMyNodeSize();i++) {
    int pe = CkNodeFirst(CkMyNode()) + i;
    if (PatchMap::ObjectOnPe(pe)->patch(pid) != NULL)
      proxyPatchPes.push_back(pe);
  }
}

//
// Called after all computes have been registered
//
void CudaComputeNonbonded::assignPatches(ComputeMgr* computeMgrIn) {
  // Remove duplicate patches
  std::sort(patches.begin(), patches.end());
  std::vector<PatchRecord>::iterator last = std::unique(patches.begin(), patches.end());
  patches.erase(last, patches.end());
  // Set number of patches
  setNumPatches(patches.size());
  masterPe = CkMyPe();
  computeMgr = computeMgrIn;
  // Start patch counter
  patchesCounter = getNumPatches();
  // Patch ID map
  std::map<PatchID, int> pidMap;
#if 1
  //-------------------------------------------------------
  // Copied in from ComputeNonbondedCUDA::assignPatches()
  //-------------------------------------------------------

  std::vector<int> pesOnNodeSharingDevice(CkMyNodeSize());
  int numPesOnNodeSharingDevice = 0;
  int masterIndex = -1;
  for ( int i=0; i<deviceCUDA->getNumPesSharingDevice(); ++i ) {
    int pe = deviceCUDA->getPesSharingDevice(i);
    if ( pe == CkMyPe() ) masterIndex = numPesOnNodeSharingDevice;
    if ( CkNodeOf(pe) == CkMyNode() ) {
      pesOnNodeSharingDevice[numPesOnNodeSharingDevice++] = pe;
    }
  }

  std::vector<int> count(patches.size(), 0);
  std::vector<int> pcount(numPesOnNodeSharingDevice, 0);
  std::vector<int> rankpcount(CkMyNodeSize(), 0);
  std::vector<char> table(patches.size()*numPesOnNodeSharingDevice, 0);

  PatchMap* patchMap = PatchMap::Object();

  int unassignedpatches = patches.size();

  for (int i=0;i < patches.size(); ++i) {
    patches[i].pe = -1;
  }

  // assign if home pe and build table of natural proxies
  for (int i=0;i < patches.size(); ++i) {
    int pid = patches[i].patchID;
    // homePe = PE where the patch currently resides
    int homePe = patchMap->node(pid);
    for ( int j=0; j < numPesOnNodeSharingDevice; ++j ) {
      int pe = pesOnNodeSharingDevice[j];
      // If homePe is sharing this device, assign this patch to homePe
      if ( pe == homePe ) {
        patches[i].pe = pe;
        --unassignedpatches;
        pcount[j] += 1;
      }
      if ( PatchMap::ObjectOnPe(pe)->patch(pid) ) {
        table[i*numPesOnNodeSharingDevice + j] = 1;
      }
    }
    // Assign this patch to homePe, if it resides on the same node
    if ( patches[i].pe == -1 && CkNodeOf(homePe) == CkMyNode() ) {
      patches[i].pe = homePe;
      --unassignedpatches;
      rankpcount[CkRankOf(homePe)] += 1;
    }
  }
  // assign if only one pe has a required proxy
  for (int i=0; i < patches.size(); ++i) {
    int pid = patches[i].patchID;
    if ( patches[i].pe != -1 ) continue;
    int c = 0;
    int lastj;
    for (int j=0; j < numPesOnNodeSharingDevice; ++j) {
      if ( table[i*numPesOnNodeSharingDevice + j] ) {
        ++c;
        lastj = j;
      }
    }
    count[i] = c;
    if ( c == 1 ) {
      patches[i].pe = pesOnNodeSharingDevice[lastj];
      --unassignedpatches;
      pcount[lastj] += 1;
    }
  }
  int assignj = 0;
  while ( unassignedpatches ) {
    int i;
    for (i=0;i < patches.size(); ++i) {
      if ( ! table[i*numPesOnNodeSharingDevice + assignj] ) continue;
      int pid = patches[i].patchID;
      // patch_record &pr = patchRecords[pid];
      if ( patches[i].pe != -1 ) continue;
      patches[i].pe = pesOnNodeSharingDevice[assignj];
      --unassignedpatches;
      pcount[assignj] += 1;
      if ( ++assignj == numPesOnNodeSharingDevice ) assignj = 0;
      break;
    }
    if (i < patches.size() ) continue;  // start search again
    for ( i=0;i < patches.size(); ++i ) {
      int pid = patches[i].patchID;
      // patch_record &pr = patchRecords[pid];
      if ( patches[i].pe != -1 ) continue;
      if ( count[i] ) continue;
      patches[i].pe = pesOnNodeSharingDevice[assignj];
      --unassignedpatches;
      pcount[assignj] += 1;
      if ( ++assignj == numPesOnNodeSharingDevice ) assignj = 0;
      break;
    }
    if ( i < patches.size() ) continue;  // start search again
    if ( ++assignj == numPesOnNodeSharingDevice ) assignj = 0;
  }

  // For each rank, list of patches
  rankPatches.resize(CkMyNodeSize());
  for (int i=0; i < patches.size(); ++i) {
    rankPatches[CkRankOf(patches[i].pe)].push_back(i);
    pidMap[patches[i].patchID] = i;
  }

  // for ( int i=0; i < patches.size(); ++i ) {
  //   CkPrintf("Pe %d patch %d hostPe %d\n", CkMyPe(), patches[i].patchID, patches[i].pe);
  // }
#else
  // For each rank, list of patches
  rankPatches.resize(CkMyNodeSize());
  // For each rank, list of home patch IDs
  PatchIDList* rankHomePatchIDs = new PatchIDList[CkMyNodeSize()];
  for (int i=0;i < CkMyNodeSize();i++) {
    int pe = CkNodeFirst(CkMyNode()) + i;
    PatchMap::Object()->basePatchIDList(pe, rankHomePatchIDs[i]);
  }
  std::vector<int> proxyPatchPes;
  std::vector<int> peProxyPatchCounter(CkMyNodeSize(), 0);
  //--------------------------------------------------------
  // Build a list of PEs to avoid
  std::vector<int> pesToAvoid;
#if 0
  // Avoid other GPUs' master PEs
  for (int i=0;i < deviceCUDA->getDeviceCount();i++) {
    int pe = deviceCUDA->getMasterPeForDeviceID(i);
    if (pe != -1 && pe != masterPe) pesToAvoid.push_back(pe);
  }
  // Avoid PEs that are involved in PME
  ComputePmeCUDAMgr *computePmeCUDAMgr = ComputePmeCUDAMgr::Object();
  for (int pe=CkNodeFirst(CkMyNode());pe < CkNodeFirst(CkMyNode()) + CkMyNodeSize();pe++) {
    if (computePmeCUDAMgr->isPmePe(pe)) pesToAvoid.push_back(pe);
  }
  // Set counters of avoidable PEs to high numbers
  for (int i=0;i < pesToAvoid.size();i++) {
    int pe = pesToAvoid[i];
    peProxyPatchCounter[CkRankOf(pe)] = (1 << 20);
  }
#endif
  // Avoid master Pe somewhat
  peProxyPatchCounter[CkRankOf(masterPe)] = 2; // patches.size();
  //--------------------------------------------------------
  for (int i=0;i < patches.size();i++) {
    //if I had this datastructure "patches" on the GPU, I could use it
    PatchID pid = patches[i].patchID;
    int pe = findHomePatchPe(rankHomePatchIDs, pid);
    if (pe == -1) {
      // Patch not present on this node => try finding a ProxyPatch
      findProxyPatchPes(proxyPatchPes, pid);
      if (proxyPatchPes.size() == 0) {
        // No ProxyPatch => create one on rank that has the least ProxyPatches
        int rank = std::min_element(peProxyPatchCounter.begin(), peProxyPatchCounter.end()) - peProxyPatchCounter.begin();
        pe = CkNodeFirst(CkMyNode()) + rank;
        peProxyPatchCounter[rank]++;
      } else {
        // Choose ProxyPatch, try to avoid masterPe (current Pe) and Pes that already have a ProxyPatch,
        // this is done by finding the entry with minimum peProxyPatchCounter -value
        // Find miniumum among proxyPatchPes, i.e., find the minimum among
        // peProxyPatchCounter[CkRankOf(proxyPatchPes[j])]
        // int pppi = std::min_element(proxyPatchPes.begin(), proxyPatchPes.end(),
        //   [&](int i, int j) {return peProxyPatchCounter[CkRankOf(i)] < peProxyPatchCounter[CkRankOf(j)];})
        //   - proxyPatchPes.begin();
        // pe = proxyPatchPes[pppi];
        int minCounter = (1 << 30);
        for (int j=0;j < proxyPatchPes.size();j++) {
          if (minCounter > peProxyPatchCounter[CkRankOf(proxyPatchPes[j])]) {
            pe = proxyPatchPes[j];
            minCounter = peProxyPatchCounter[CkRankOf(pe)];
          }
        }
        if (pe == -1)
          NAMD_bug("CudaComputeNonbonded::assignPatches, Unable to choose PE with proxy patch");
        peProxyPatchCounter[CkRankOf(pe)]++;
      }
    } else if (std::find(pesToAvoid.begin(), pesToAvoid.end(), pe) != pesToAvoid.end()) {
      // Found home patch on this node, but it's on PE that should be avoided => find a new one
      int rank = std::min_element(peProxyPatchCounter.begin(), peProxyPatchCounter.end()) - peProxyPatchCounter.begin();
      pe = CkNodeFirst(CkMyNode()) + rank;
      peProxyPatchCounter[rank]++;
    }
    if (pe < CkNodeFirst(CkMyNode()) || pe >= CkNodeFirst(CkMyNode()) + CkMyNodeSize() )
      NAMD_bug("CudaComputeNonbonded::assignPatches, Invalid PE for a patch");
    rankPatches[CkRankOf(pe)].push_back(i);
    pidMap[pid] = i;
  }

  delete [] rankHomePatchIDs;
#endif
  // Setup computes using pidMap
  for (int i=0;i < computes.size();i++) {
    computes[i].patchInd[0] = pidMap[computes[i].pid[0]];
    computes[i].patchInd[1] = pidMap[computes[i].pid[1]];
  }
  for (int i=0;i < CkMyNodeSize();i++) {
    if (rankPatches[i].size() > 0) pes.push_back(CkNodeFirst(CkMyNode()) + i);
  }
  computeMgr->sendAssignPatchesOnPe(pes, this);
}

void CudaComputeNonbonded::updatePatchOrder(std::vector<CudaLocalRecord>& data) {
  // DMC This vector of CudaLocalRecords doesn't have the correct number of peer records
  std::map<int, int> pidMap;
  for (int i=0; i < data.size(); ++i) {
    pidMap[data[i].patchID] = i;
  }

  std::vector<PatchRecord> copy = patches;

  for (int i=0; i < copy.size(); i++) {
    const int new_idx = pidMap[copy[i].patchID];
    patches[new_idx] = copy[i];
  }

  for (int i=0; i < rankPatches.size(); i++) {
    rankPatches[i].clear();
  }
  for (int i=0; i < patches.size(); ++i) {
    rankPatches[CkRankOf(patches[i].pe)].push_back(i);
  } 

  // Setup computes using pidMap
  for (int i=0;i < computes.size();i++) {
    computes[i].patchInd[0] = pidMap[computes[i].pid[0]];
    computes[i].patchInd[1] = pidMap[computes[i].pid[1]];
  }
  // TODO do we need to call sendAssignPatchesOnPe with the new order?
}

void CudaComputeNonbonded::initialize() {
  if (patches.size() > 0) {
    npairlists = 0;
    // Allocate CUDA version of patches
    cudaCheck(cudaSetDevice(deviceID));
    allocate_host<CudaPatchRecord>(&cudaPatches, patches.size());

    allocate_host<VirialEnergy>(&h_virialEnergy, 1);
    allocate_device<VirialEnergy>(&d_virialEnergy, ATOMIC_BINS);

  /* JM: Queries for maximum sharedMemoryPerBlock on deviceID
   */
   cudaDeviceProp props;
   cudaCheck(cudaGetDeviceProperties(&props, deviceID)); //Gets properties of 'deviceID device'
   maxShmemPerBlock = props.sharedMemPerBlock;

#if CUDA_VERSION >= 5050 || defined(NAMD_HIP)
    int leastPriority, greatestPriority;
    cudaCheck(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    int priority = (doStreaming) ? leastPriority : greatestPriority;
    // int priority = greatestPriority;
    cudaCheck(cudaStreamCreateWithPriority(&stream,cudaStreamDefault, priority));
#else
    cudaCheck(cudaStreamCreate(&stream));
#endif
    cudaCheck(cudaEventCreate(&forceDoneEvent));

    buildExclusions();

    lock = CmiCreateLock();
    params = Node::Object()->simParameters;
    reduction = ReductionMgr::Object()->willSubmit(REDUCTIONS_BASIC);

#ifdef NODEGROUP_FORCE_REGISTER
    int devInd = deviceCUDA->getDeviceIndex();
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    PatchData *patchData = cpdata.ckLocalBranch();
    nodeReduction = patchData->reduction;
    patchData->devData[devInd].nbond_stream = stream;
    // Fill auxiliary arrays for merging forces here
    PatchMap* map = PatchMap::Object();
    int nGlobalPatches = map->numPatches();
    allocate_host<bool>( &(patchData->devData[devInd].h_hasPatches), nGlobalPatches);
    memset(patchData->devData[devInd].h_hasPatches, 0, sizeof(bool)*nGlobalPatches);

    for(int i = 0; i < patches.size(); i++){
      patchData->devData[devInd].h_hasPatches[patches[i].patchID] = true;
    }
    allocate_device<bool>( &(patchData->devData[devInd].d_hasPatches), nGlobalPatches);
    copy_HtoD_sync<bool>( patchData->devData[devInd].h_hasPatches, patchData->devData[devInd].d_hasPatches, nGlobalPatches);
#endif
  }
}

//
// atomUpdate() can be called by any Pe
//
void CudaComputeNonbonded::atomUpdate() {
  atomsChangedIn = true;
}

//
// Compute patches[].atomStart, patches[].numAtoms, patches[].numFreeAtoms, and atomStorageSize
//
void CudaComputeNonbonded::updatePatches() {
  if(params->CUDASOAintegrate && params->useDeviceMigration) {
#ifdef NODEGROUP_FORCE_REGISTER
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    PatchData *patchData = cpdata.ckLocalBranch();
    std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
    const int numPatchesHomeAndProxy = patchData->devData[deviceIndex].numPatchesHomeAndProxy;

    // Maximum number of tiles per tile list
    maxTileListLen = 0;
    int atomStart = 0;
    for (int i=0;i < numPatchesHomeAndProxy; i++) {
      patches[i].numAtoms = localPatches[i].numAtoms;
      patches[i].numFreeAtoms = localPatches[i].numAtoms;  // XXX This is incorrect if there are free atoms
      patches[i].atomStart = localPatches[i].bufferOffsetNBPad;
      cudaPatches[i].numAtoms = localPatches[i].numAtoms;
      cudaPatches[i].numFreeAtoms = localPatches[i].numAtoms;
      cudaPatches[i].atomStart = localPatches[i].bufferOffsetNBPad;
      cudaPatches[i].patchID = localPatches[i].patchID;
      int numAtoms = patches[i].numAtoms;
#if defined(NAMD_CUDA)
      int numTiles = CudaComputeNonbondedKernel::computeNumTiles(numAtoms, WARPSIZE);
      maxTileListLen = std::max(maxTileListLen, numTiles);
      // computeAtomPad will recompute the number of tiles. Recomputing for clarity
      atomStart += CudaComputeNonbondedKernel::computeAtomPad(numAtoms, WARPSIZE); 
#else
      int numTiles = CudaComputeNonbondedKernel::computeNumTiles(numAtoms, BOUNDINGBOXSIZE);
      maxTileListLen = std::max(maxTileListLen, numTiles);
      // computeAtomPad will recompute the number of tiles. Recomputing for clarity
      atomStart += CudaComputeNonbondedKernel::computeAtomPad(numAtoms, BOUNDINGBOXSIZE); 
#endif
    }
    atomStorageSize = atomStart;

    if (maxTileListLen >= 65536) {
      NAMD_bug("CudaComputeNonbonded::updatePatches, maximum number of tiles per tile lists (65536) blown");
    }
#endif
  } else {

    // Maximum number of tiles per tile list
    maxTileListLen = 0;
    int atomStart = 0;
    for (int i=0;i < patches.size();i++) {
      patches[i].atomStart = atomStart;
      cudaPatches[i].atomStart = atomStart;
#ifdef NODEGROUP_FORCE_REGISTER
      cudaPatches[i].patchID = patches[i].patchID;
#endif
    int numAtoms = patches[i].numAtoms;
#ifdef NAMD_HIP
    int numTiles = ((numAtoms-1)/BOUNDINGBOXSIZE+1);
    maxTileListLen = std::max(maxTileListLen, numTiles);
    atomStart += numTiles*BOUNDINGBOXSIZE;
#else
    int numTiles = ((numAtoms-1)/WARPSIZE+1);
    maxTileListLen = std::max(maxTileListLen, numTiles);
    atomStart += numTiles*WARPSIZE;
#endif
  }
  atomStorageSize = atomStart;

    if (maxTileListLen >= 65536) {
      NAMD_bug("CudaComputeNonbonded::updatePatches, maximum number of tiles per tile lists (65536) blown");
    }
  }
}

void CudaComputeNonbonded::skipPatch(int i) {
  if (CkMyPe() != patches[i].pe)
    NAMD_bug("CudaComputeNonbonded::skipPatch called on wrong Pe");
  Flags &flags = patches[i].patch->flags;
  patches[i].positionBox->skip();
  patches[i].forceBox->skip();
  if (flags.doGBIS) {
    patches[i].psiSumBox->skip();
    patches[i].intRadBox->skip();
    patches[i].bornRadBox->skip();
    patches[i].dEdaSumBox->skip();
    patches[i].dHdrPrefixBox->skip();
  }
}

void CudaComputeNonbonded::skipPatchesOnPe() {
  if (rankPatches[CkMyRank()].size() == 0)
    NAMD_bug("CudaComputeNonbonded::skipPatchesOnPe, empty rank");
  for (int i=0;i < rankPatches[CkMyRank()].size();i++) {
    skipPatch(rankPatches[CkMyRank()][i]);
  }
  bool done = false;
  CmiLock(lock);
  patchesCounter -= rankPatches[CkMyRank()].size();
  if (patchesCounter == 0) {
    patchesCounter = getNumPatches();
    done = true;
  }
  CmiUnlock(lock);
  if (done) {
    // Reduction must be done on masterPe
    computeMgr->sendFinishReductions(masterPe, this);
  }
}

void CudaComputeNonbonded::skip() {
  if (CkMyPe() != masterPe)
    NAMD_bug("CudaComputeNonbonded::skip() called on non masterPe");

  if (patches.size() == 0) return;

  doSkip = true;

  computeMgr->sendSkipPatchesOnPe(pes, this);
}

void CudaComputeNonbonded::getMaxMovementTolerance(float& maxAtomMovement, float& maxPatchTolerance) {
  if (CkMyPe() != masterPe)
    NAMD_bug("CudaComputeNonbonded::getMaxMovementTolerance() called on non masterPe");

  for (int i=0;i < patches.size();++i) {
    PatchRecord &pr = patches[i];

    float maxMove = pr.patch->flags.maxAtomMovement;
    if ( maxMove > maxAtomMovement ) maxAtomMovement = maxMove;

    float maxTol = pr.patch->flags.pairlistTolerance;
    //if(pr.patch->getPatchID() == 0) fprintf(stderr,
    //    "\n\nP0: Maximum mov/tol during CudaComputeNonbonded: %lf %lf\n", maxMove, maxTol);
    if ( maxTol > maxPatchTolerance ) maxPatchTolerance = maxTol;
  }
}

inline void CudaComputeNonbonded::updateVdwTypesExclLoop(int first, int last, void *result, int paraNum, void *param) {
  CudaComputeNonbonded* c = (CudaComputeNonbonded *)param;
  c->updateVdwTypesExclSubset(first, last);
}

void CudaComputeNonbonded::updateVdwTypesExclSubset(int first, int last) {
  for (int i=first;i <= last;i++) {
    PatchRecord &pr = patches[i];
    int start = pr.atomStart;
    int numAtoms = pr.numAtoms;
    const CompAtom *compAtom = pr.compAtom;
    const CompAtomExt *compAtomExt = pr.patch->getCompAtomExtInfo();
    // Atoms have changed, re-do exclusions and vdw types
    int2* exclp = exclIndexMaxDiff + start;
    int* aip = atomIndex + start;
    char* pst;
    if(doAlch) pst = part + start;
    for ( int k=0;k < numAtoms; ++k ) {
      int j = compAtomExt[k].sortOrder;
      vdwTypes[start + k] = compAtom[j].vdwType;
      aip[k] = compAtomExt[j].id;
      if(doAlch) pst[k] = compAtom[j].partition;
#ifdef MEM_OPT_VERSION
      exclp[k].x = exclusionsByAtom[compAtomExt[j].exclId].y;
      exclp[k].y = exclusionsByAtom[compAtomExt[j].exclId].x;
#else // ! MEM_OPT_VERSION
      exclp[k].x = exclusionsByAtom[compAtomExt[j].id].y;
      exclp[k].y = exclusionsByAtom[compAtomExt[j].id].x;
#endif // MEM_OPT_VERSION
    }
  }
}

//
// Called every time atoms changed
//
void CudaComputeNonbonded::updateVdwTypesExcl() {
  // Re-allocate (VdwTypes, exclIndexMaxDiff) as needed
  reallocate_host<int>(&vdwTypes, &vdwTypesSize, atomStorageSize, 1.4f);
  reallocate_host<int2>(&exclIndexMaxDiff, &exclIndexMaxDiffSize, atomStorageSize, 1.4f);
  reallocate_host<int>(&atomIndex, &atomIndexSize, atomStorageSize, 1.4f);
  if (doAlch) reallocate_host<char>(&part, &partSize, atomStorageSize, 1.4f);


  if (!(params->CUDASOAintegrate && params->useDeviceMigration)) {
#if CMK_SMP && USE_CKLOOP
    int useCkLoop = Node::Object()->simParameters->useCkLoop;
    if (useCkLoop >= 1) {
      CkLoop_Parallelize(updateVdwTypesExclLoop, 1, (void *)this, CkMyNodeSize(), 0, patches.size()-1);
    } else
#endif
    {
      updateVdwTypesExclSubset(0, patches.size()-1);
    }

    nonbondedKernel.updateVdwTypesExcl(atomStorageSize, vdwTypes, exclIndexMaxDiff, atomIndex, stream);
  } else {
#ifdef NODEGROUP_FORCE_REGISTER
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    PatchData *patchData = cpdata.ckLocalBranch();
    const int deviceIndex = deviceCUDA->getDeviceIndex();
    nonbondedKernel.updateVdwTypesExclOnGPU(tileListKernel,
      patchData->devData[deviceIndex].numPatchesHomeAndProxy,
      atomStorageSize, params->alchOn,
      patchData->devData[deviceIndex].d_localPatches,
      patchData->h_soa_vdwType[deviceIndex],
      patchData->h_soa_id[deviceIndex],
      patchData->h_soa_sortOrder[deviceIndex],
      patchData->h_soa_partition[deviceIndex],
      stream
    );
#endif  // NODEGROUP_FORCE_REGISTER
  }
}

inline void CudaComputeNonbonded::copyAtomsLoop(int first, int last, void *result, int paraNum, void *param) {
  CudaComputeNonbonded* c = (CudaComputeNonbonded *)param;
  c->copyAtomsSubset(first, last);
}

void CudaComputeNonbonded::copyAtomsSubset(int first, int last) {
  for (int i=first;i <= last;++i) {
    PatchRecord &pr = patches[i];
    int numAtoms = pr.numAtoms;
    if (numAtoms > 0) {
      int start = pr.atomStart;
      const CudaAtom *src = pr.patch->getCudaAtomList();
      CudaAtom *dst = atoms + start;
      memcpy(dst, src, sizeof(CudaAtom)*numAtoms);
      // Fill the rest with the copy of the last atom
#ifdef NAMD_HIP
      int numAtomsAlign = ((numAtoms-1)/BOUNDINGBOXSIZE+1)*BOUNDINGBOXSIZE;
#else
      int numAtomsAlign = ((numAtoms-1)/WARPSIZE+1)*WARPSIZE;
#endif
      CudaAtom lastAtom = src[numAtoms-1];
      for (int j=numAtoms;j < numAtomsAlign;j++) {
        dst[j] = lastAtom;
      }
#if 0
      fprintf(stderr, " printing patch %d\n", pr.patch->getPatchID());
      for(int k = 0; k < numAtoms; k++){
        fprintf(stderr, "%lf %lf %lf\n", dst[k].x, dst[k].y, dst[k].z);
      }
#endif
    }
  }
}

void CudaComputeNonbonded::copyGBISphase(int i) {
  if (CkMyPe() != patches[i].pe)
    NAMD_bug("CudaComputeNonbonded::copyGBISphase called on wrong Pe");
  PatchRecord &pr = patches[i];
  const CompAtomExt *aExt = pr.patch->getCompAtomExtInfo();
  if (gbisPhase == 1) {
    //Copy GBIS intRadius to Host
    if (atomsChanged) {
      float *intRad0 = intRad0H + pr.atomStart;
      float *intRadS = intRadSH + pr.atomStart;
      for (int k=0;k < pr.numAtoms;++k) {
        int j = aExt[k].sortOrder;
        intRad0[k] = pr.intRad[2*j+0];
        intRadS[k] = pr.intRad[2*j+1];
      }
    }
  } else if (gbisPhase == 2) {
    float *bornRad = bornRadH + pr.atomStart;
    for ( int k=0; k < pr.numAtoms; ++k ) {
      int j = aExt[k].sortOrder;
      bornRad[k] = pr.bornRad[j];
    }
  } else if (gbisPhase == 3) {
    float *dHdrPrefix = dHdrPrefixH + pr.atomStart;
    for ( int k=0; k < pr.numAtoms; ++k ) {
      int j = aExt[k].sortOrder;
      dHdrPrefix[k] = pr.dHdrPrefix[j];
    }
  } // end phases
}

void CudaComputeNonbonded::openBox(int i) {
  NAMD_EVENT_START(1, NamdProfileEvent::COMPUTE_NONBONDED_OPEN_BOXES);
  
  if (CkMyPe() != patches[i].pe)
    NAMD_bug("CudaComputeNonbonded::openBox called on wrong Pe");
  SimParameters *simParams = Node::Object()->simParameters;
  if (!simParams->GBISOn || gbisPhase == 1) {
    // what is positionBox????
    patches[i].compAtom = patches[i].positionBox->open();
    // the compAtom datastructure is null for PEs
    //fprintf(stderr, "opening box at patches[%d] = %p\n", i, patches[i].compAtom);
    // JM: This is not necessary if we already have the positions from integration
    // This is only necessary in the first iteration
    // XXX TODO: Find out if we really need to open the position box or if we
    //           can skip this step entirely
#ifdef NODEGROUP_FORCE_REGISTER
    if(simParams->CUDASOAintegrate){
       if(atomsChanged && !simParams->useDeviceMigration) copyAtomsSubset(i, i);
    }else copyAtomsSubset(i, i);
#else
    copyAtomsSubset(i, i);
#endif
  }
  if (simParams->GBISOn) {
    if (gbisPhase == 1) {
      patches[i].intRad     = patches[i].intRadBox->open();
      patches[i].psiSum     = patches[i].psiSumBox->open();
    } else if (gbisPhase == 2) {
      patches[i].bornRad    = patches[i].bornRadBox->open();
      patches[i].dEdaSum    = patches[i].dEdaSumBox->open();
    } else if (gbisPhase == 3) {
      patches[i].dHdrPrefix = patches[i].dHdrPrefixBox->open();
    }
    copyGBISphase(i);
  }

  NAMD_EVENT_STOP(1, NamdProfileEvent::COMPUTE_NONBONDED_OPEN_BOXES);
}

void CudaComputeNonbonded::messageEnqueueWork() {
  if (masterPe != CkMyPe())
    NAMD_bug("CudaComputeNonbonded::messageEnqueueWork() must be called from masterPe");
  WorkDistrib::messageEnqueueWork(this);
}

void CudaComputeNonbonded::openBoxesOnPe() {
  if (rankPatches[CkMyRank()].size() == 0)
    NAMD_bug("CudaComputeNonbonded::openBoxesOnPe, empty rank");

  NAMD_EVENT_START(1, NamdProfileEvent::COMPUTE_NONBONDED_OPEN_BOXES);

#ifdef NODEGROUP_FORCE_REGISTER
  if( Node::Object()->simParameters->CUDASOAintegrate && !atomsChanged) {
      // opens boxes to make sure NAMD won't complain
      for (int i=0;i < rankPatches[CkMyRank()].size();i++) {
        int j = rankPatches[CkMyRank()][i];
        patches[j].positionBox->open();
      }
      if(masterPe == CkMyPe()) {
        // we need to open boxes here...
        if(params->CUDASOAintegrate){
          if(!atomsChanged) this->launchWork();
        }
        else computeMgr->sendLaunchWork(masterPe, this); 
      }
  }
  else{
#endif
  for (int i=0;i < rankPatches[CkMyRank()].size();i++) {
    openBox(rankPatches[CkMyRank()][i]);
  }
  bool done = false;
  CmiLock(lock);
  patchesCounter -= rankPatches[CkMyRank()].size();
  if (patchesCounter == 0) {
    patchesCounter = getNumPatches();
    done = true;
  }
  CmiUnlock(lock);
  if (done) {
    if(params->CUDASOAintegrate){
       if(!atomsChanged) this->launchWork();
    }
    else computeMgr->sendLaunchWork(masterPe, this);
  }
#ifdef NODEGROUP_FORCE_REGISTER
  }
#endif
  NAMD_EVENT_STOP(1, NamdProfileEvent::COMPUTE_NONBONDED_OPEN_BOXES);
}

int CudaComputeNonbonded::noWork() {
  // Simply enqueu doWork on masterPe and return "no work"
  computeMgr->sendMessageEnqueueWork(masterPe, this);
  return 1;
}

void CudaComputeNonbonded::reallocateArrays() {
  cudaCheck(cudaSetDevice(deviceID));
  SimParameters *simParams = Node::Object()->simParameters;

  // Re-allocate atoms
  reallocate_host<CudaAtom>(&atoms, &atomsSize, atomStorageSize, 1.4f);
  
  // Re-allocate forces
  if (doStreaming) {
    reallocate_host<float4>(&h_forces, &h_forcesSize, atomStorageSize, 1.4f, cudaHostAllocMapped);
    reallocate_host<float4>(&h_forcesSlow, &h_forcesSlowSize, atomStorageSize, 1.4f, cudaHostAllocMapped);
  } else {
    reallocate_host<float4>(&h_forces, &h_forcesSize, atomStorageSize, 1.4f);
    reallocate_host<float4>(&h_forcesSlow, &h_forcesSlowSize, atomStorageSize, 1.4f);
  }
  reallocate_device<float4>(&d_forces, &d_forcesSize, atomStorageSize, 1.4f);
  reallocate_device<float4>(&d_forcesSlow, &d_forcesSlowSize, atomStorageSize, 1.4f);
  nonbondedKernel.reallocate_forceSOA(atomStorageSize);
  
  if (simParams->GBISOn) {
    reallocate_host<float>(&intRad0H, &intRad0HSize, atomStorageSize, 1.2f);
    reallocate_host<float>(&intRadSH, &intRadSHSize, atomStorageSize, 1.2f);
    reallocate_host<GBReal>(&psiSumH, &psiSumHSize, atomStorageSize, 1.2f);
    reallocate_host<GBReal>(&dEdaSumH, &dEdaSumHSize, atomStorageSize, 1.2f);
    reallocate_host<float>(&bornRadH, &bornRadHSize, atomStorageSize, 1.2f);
    reallocate_host<float>(&dHdrPrefixH, &dHdrPrefixHSize, atomStorageSize, 1.2f);
  }
}

void CudaComputeNonbonded::doWork() {
  if (CkMyPe() != masterPe)
    NAMD_bug("CudaComputeNonbonded::doWork() called on non masterPe");

  // Read value of atomsChangedIn, which is set in atomUpdate(), and reset it.
  // atomsChangedIn can be set to true by any Pe
  // atomsChanged can only be set by masterPe
  // This use of double varibles makes sure we don't have race condition
  // it seems like it's important to have the masterPe call doWork() first
  atomsChanged = atomsChangedIn;
  atomsChangedIn = false;

  SimParameters *simParams = Node::Object()->simParameters;

  if (patches.size() == 0) return;  // No work do to

  // Take the flags from the first patch on this Pe
  // Flags &flags = patches[rankPatches[CkMyRank()][0]].patch->flags;
  // these flags are probably wrong.
  Flags &flags = patches[0].patch->flags;

  doSlow = flags.doFullElectrostatics;
  doEnergy = flags.doEnergy;
  doVirial = flags.doVirial;
  doAlch = simParams->alchOn;
  doMinimize = flags.doMinimize;
  
  if (flags.doNonbonded) {

    if (simParams->GBISOn) {
      gbisPhase = 1 + (gbisPhase % 3);//1->2->3->1...
    }

    if (!simParams->GBISOn || gbisPhase == 1) {
      if ( computesChanged ) {
        updateComputes();
      }
      if (atomsChanged) {
        // Re-calculate patch atom numbers and storage
        updatePatches();
        reSortDone = false;
      }
      reallocateArrays();
#ifdef NODEGROUP_FORCE_REGISTER 
      if (simParams->CUDASOAintegrate && simParams->useDeviceMigration && atomsChanged) {
        tileListKernel.prepareBuffers(atomStorageSize, patches.size(), cudaPatches, stream);
        updatePatchRecord();
      }
#endif  // NODEGROUP_FORCE_REGISTER 
    }

    // Open boxes on Pes and launch work to masterPe
    if(params->CUDASOAintegrate){
       if(!atomsChanged) this->openBoxesOnPe();
     }
    else computeMgr->sendOpenBoxesOnPe(pes, this);

  } else {
    // No work to do, skip
    skip();
  }

}

void CudaComputeNonbonded::launchWork() {
  if (CkMyPe() != masterPe)
    NAMD_bug("CudaComputeNonbonded::launchWork() called on non masterPe");

  beforeForceCompute = CkWallTimer();
  cudaCheck(cudaSetDevice(deviceID));
  SimParameters *simParams = Node::Object()->simParameters;
  
  // So, it seems like PE's are invoking the same object, however the patches[i] is borked on the masterPe
  
  // When I get here, it seems like compAtoms are not set for all Pes? How can that be?

  //execute only during GBIS phase 1, or if not using GBIS
  if (!simParams->GBISOn || gbisPhase == 1) {

    if ( atomsChanged || computesChanged ) {
      // Invalidate pair lists
      pairlistsValid = false;
      pairlistTolerance = 0.0f;
    }

    // Get maximum atom movement and patch tolerance
    float maxAtomMovement = 0.0f;
    float maxPatchTolerance = 0.0f;
    getMaxMovementTolerance(maxAtomMovement, maxPatchTolerance);
    // Update pair-list cutoff
    Flags &flags = patches[0].patch->flags;
    savePairlists = false;
    usePairlists = false;
    if ( flags.savePairlists ) {
      savePairlists = true;
      usePairlists = true;
    } else if ( flags.usePairlists ) {
      if ( ! pairlistsValid || ( 2. * maxAtomMovement > pairlistTolerance ) ) {
        reduction->item(REDUCTION_PAIRLIST_WARNINGS) += 1;
#ifdef NODEGROUP_FORCE_REGISTER
        nodeReduction->item(REDUCTION_PAIRLIST_WARNINGS) += 1;
#endif
      } else {
        usePairlists = true;
      }
    }
    if ( ! usePairlists ) {
      pairlistsValid = false;
    }
    float plcutoff = cutoff;
    if ( savePairlists ) {
      pairlistsValid = true;
      pairlistTolerance = 2. * maxPatchTolerance;
      plcutoff += pairlistTolerance;
    }
    plcutoff2 = plcutoff * plcutoff;
    
    // fprintf(stderr, "STEP[%d] plcutoff = %f  listTolerance = %f  save = %d  maxPatchTolerance = %f maxAtomMovement = %f plvalid = %d flags.use = %d use = %d\n",
    //      flags.step, plcutoff, pairlistTolerance, savePairlists, maxPatchTolerance, maxAtomMovement, pairlistsValid, flags.usePairlists, usePairlists);
    if(savePairlists || !usePairlists){
      reSortDone = false; // Ensures pairlist resorting if doPairlist
    }

    // if (atomsChanged)
    //   CkPrintf("plcutoff = %f  listTolerance = %f  save = %d  use = %d\n",
    //     plcutoff, pairlistTolerance, savePairlists, usePairlists);

  } // if (!simParams->GBISOn || gbisPhase == 1)

  // Calculate PME & VdW forces
  if (!simParams->GBISOn || gbisPhase == 1) {
    doForce();
    if (doStreaming) {
      patchReadyQueue = nonbondedKernel.getPatchReadyQueue();
      patchReadyQueueLen = tileListKernel.getNumPatches();
      patchReadyQueueNext = 0;
      // Fill in empty patches [0 ... patchReadyQueueNext-1] at the top
      int numEmptyPatches = tileListKernel.getNumEmptyPatches();
      int* emptyPatches = tileListKernel.getEmptyPatches();
      for (int i=0;i < numEmptyPatches;i++) {
        PatchRecord &pr = patches[emptyPatches[i]];
        memset(h_forces+pr.atomStart, 0, sizeof(float4)*pr.numAtoms);
        if (doSlow) memset(h_forcesSlow+pr.atomStart, 0, sizeof(float4)*pr.numAtoms);
        patchReadyQueue[i] = emptyPatches[i];
      }
      if (patchReadyQueueLen != patches.size())
        NAMD_bug("CudaComputeNonbonded::launchWork, invalid patchReadyQueueLen");
    }
  }

  // For GBIS phase 1 at pairlist update, we must re-sort tile list
  // before calling doGBISphase1().
  if (atomsChanged && simParams->GBISOn && gbisPhase == 1) {
    // In this code path doGBISphase1() is called in forceDone()
    forceDoneSetCallback();
    return;
  }
  
  // GBIS Phases
  if (simParams->GBISOn) {
    if (gbisPhase == 1) {
      doGBISphase1();
    } else if (gbisPhase == 2) {
      doGBISphase2();
    } else if (gbisPhase == 3) {
      doGBISphase3();
    } 
  }

  // Copy forces to host
  if (!simParams->GBISOn || gbisPhase == 3) {
    if (!doStreaming) {
#ifdef NODEGROUP_FORCE_REGISTER
      if(!simParams->CUDASOAintegrate || (atomsChanged && !simParams->useDeviceMigration)){
        copy_DtoH<float4>(d_forces, h_forces, atomStorageSize, stream);
        if (doSlow) copy_DtoH<float4>(d_forcesSlow, h_forcesSlow, atomStorageSize, stream);
      }
#else
      copy_DtoH<float4>(d_forces, h_forces, atomStorageSize, stream);
      if (doSlow) copy_DtoH<float4>(d_forcesSlow, h_forcesSlow, atomStorageSize, stream);
#endif
      
    }
  }

  if ((!simParams->GBISOn || gbisPhase == 2) && (doEnergy || doVirial)) {

    NAMD_EVENT_START(1, NamdProfileEvent::REDUCE_VIRIAL_ENERGY);
    // For GBIS, energies are ready after phase 2
    nonbondedKernel.reduceVirialEnergy(tileListKernel,
      atomStorageSize, doEnergy, doVirial, doSlow, simParams->GBISOn,
      d_forces, d_forcesSlow, d_virialEnergy, stream);
    copy_DtoH<VirialEnergy>(d_virialEnergy, h_virialEnergy, 1, stream);

    NAMD_EVENT_STOP(1, NamdProfileEvent::REDUCE_VIRIAL_ENERGY);

  }

  if(simParams->CUDASOAintegrate && ((savePairlists || !usePairlists)) && !atomsChanged) reSortTileLists();

  // Setup call back
  forceDoneSetCallback();
  
#if 0
  cudaCheck(cudaStreamSynchronize(stream));
  PatchMap *map = PatchMap::Object();
  HomePatchElem *elem;
  for(elem = map->homePatchList()->begin(); elem != map->homePatchList()->end(); elem++){
    if(elem->patch->getPatchID() == 7) break;
  }
  if(elem->patch->flags.step == 11){
    // it would be good to know from which patch these atoms are...
    fprintf(stderr, "CudaNonbonded data\n");
    for(int i = 0 ; i < atomStorageSize; i++){
      fprintf(stderr, "pos[%d] = %lf, %lf, %lf, %lf | (%f %f %f) (%f %f %f) \n", 
        i, atoms[i].x, atoms[i].y, atoms[i].z, atoms[i].q,
        // for some reason, we needed to set the positions
        h_forces[i].x, h_forces[i].y, h_forces[i].z,
        h_forcesSlow[i].x, h_forcesSlow[i].y, h_forcesSlow[i].z);
    }
  }
#endif

}

//
// GBIS Phase 1
//
void CudaComputeNonbonded::doGBISphase1() {
  cudaCheck(cudaSetDevice(deviceID));

  if (atomsChanged) {
    GBISKernel.updateIntRad(atomStorageSize, intRad0H, intRadSH, stream);
  }

  SimParameters *simParams = Node::Object()->simParameters;
  Lattice lattice = patches[0].patch->flags.lattice;

  float3 lata = make_float3(lattice.a().x, lattice.a().y, lattice.a().z);
  float3 latb = make_float3(lattice.b().x, lattice.b().y, lattice.b().z);
  float3 latc = make_float3(lattice.c().x, lattice.c().y, lattice.c().z);

  GBISKernel.GBISphase1(tileListKernel, atomStorageSize,
    lata, latb, latc,
    simParams->alpha_cutoff-simParams->fsMax, psiSumH, stream);
}

//
// GBIS Phase 2
//
void CudaComputeNonbonded::doGBISphase2() {
  cudaCheck(cudaSetDevice(deviceID));

  SimParameters *simParams = Node::Object()->simParameters;
  Lattice lattice = patches[0].patch->flags.lattice;

  float3 lata = make_float3(lattice.a().x, lattice.a().y, lattice.a().z);
  float3 latb = make_float3(lattice.b().x, lattice.b().y, lattice.b().z);
  float3 latc = make_float3(lattice.c().x, lattice.c().y, lattice.c().z);

  GBISKernel.updateBornRad(atomStorageSize, bornRadH, stream);

  GBISKernel.GBISphase2(tileListKernel, atomStorageSize,
    doEnergy, doSlow,
    lata, latb, latc,
    simParams->cutoff, simParams->nonbondedScaling, simParams->kappa,
    (simParams->switchingActive ? simParams->switchingDist : -1.0),
    simParams->dielectric, simParams->solvent_dielectric,
    d_forces, dEdaSumH, stream);
}

//
// GBIS Phase 3
//
void CudaComputeNonbonded::doGBISphase3() {
  cudaCheck(cudaSetDevice(deviceID));
  SimParameters *simParams = Node::Object()->simParameters;
  Lattice lattice = patches[0].patch->flags.lattice;

  float3 lata = make_float3(lattice.a().x, lattice.a().y, lattice.a().z);
  float3 latb = make_float3(lattice.b().x, lattice.b().y, lattice.b().z);
  float3 latc = make_float3(lattice.c().x, lattice.c().y, lattice.c().z);

  if (doSlow) {
    GBISKernel.update_dHdrPrefix(atomStorageSize, dHdrPrefixH, stream);

    GBISKernel.GBISphase3(tileListKernel, atomStorageSize,
      lata, latb, latc,
      simParams->alpha_cutoff-simParams->fsMax, d_forcesSlow, stream);
  }
}

//
// Calculate electrostatic & VdW forces
//
void CudaComputeNonbonded::doForce() {
  cudaCheck(cudaSetDevice(deviceID));
  SimParameters *simParams = Node::Object()->simParameters;
  // XXX TODO: This will not work if the patch flags are not correctly set
  Lattice lattice = patches[0].patch->flags.lattice;
  bool CUDASOAintegrator = simParams->CUDASOAintegrate;
  float3 lata = make_float3(lattice.a().x, lattice.a().y, lattice.a().z);
  float3 latb = make_float3(lattice.b().x, lattice.b().y, lattice.b().z);
  float3 latc = make_float3(lattice.c().x, lattice.c().y, lattice.c().z);
  bool doPairlist = savePairlists || (!usePairlists);
  bool doFEP=false, doTI=false, doAlchVdwForceSwitching=false;
  if(doAlch){
    static thread_local bool firsttime = true;
    doTI =  simParams->alchThermIntOn;
    doFEP = simParams->alchFepOn;
    doAlchVdwForceSwitching = simParams->vdwForceSwitching;
    // Otherwise, update them only when lambda window is updated.
    // getCurrentLambda and getCurrentLambda2 are assumed to have no side effects.
    // use float here to match the type of CudaAlchLambdas 
    const decltype(alchFlags.lambdaUp) currentLambda  = simParams->getCurrentLambda(patches[0].patch->flags.step);
    const decltype(alchFlags.lambda2Up) currentLambda2 = simParams->getCurrentLambda2(patches[0].patch->flags.step);
    if (firsttime) {
      // Update the alchemical flags if this is the first time
      firsttime = false;
      lambdaWindowUpdated = true;
    } else {
      // Compare the above parameters with respect to the saved parameters.
      if (alchFlags.lambdaUp          != currentLambda  ||
          alchFlags.lambda2Up         != currentLambda2 ||
          // Could the following parameters also be changed?
          // I am not quite sure, but checking them by CPU code is not computationally expensive anyway.
          alchFlags.cutoff2           != ComputeNonbondedUtil::cutoff2 ||
          alchFlags.switchdist2       != ComputeNonbondedUtil::switchOn2 ||
          alchFlags.alchVdwShiftCoeff != ComputeNonbondedUtil::alchVdwShiftCoeff ||
          alchFlags.alchDecouple      != ComputeNonbondedUtil::alchDecouple ||
          alchFlags.scaling           != ComputeNonbondedUtil::scaling) {
        lambdaWindowUpdated = true;
      } else {
        lambdaWindowUpdated = false;
      }
    }
    if (lambdaWindowUpdated) {
      // Flags that are independent of the number of steps
      alchFlags.cutoff2           = ComputeNonbondedUtil::cutoff2;
      alchFlags.switchdist2       = ComputeNonbondedUtil::switchOn2;
      alchFlags.alchVdwShiftCoeff = ComputeNonbondedUtil::alchVdwShiftCoeff;
      alchFlags.alchDecouple      = ComputeNonbondedUtil::alchDecouple;
      alchFlags.scaling           = ComputeNonbondedUtil::scaling;
      const double factor         = alchFlags.cutoff2 - alchFlags.switchdist2;
      // When switching is off, cutoff is the same as switchdist,
      // so we need to check it to avoid passing inf for the computation of switchmul and switchmul2
      alchFlags.switchfactor      = simParams->switchingActive ? 1.0/(factor*factor*factor) : 0;
      // Step-dependent parameters (lambdas)
      // alchFlags.alchLambda is redundant because we have lambdaUp already.
      // alchFlags.alchLambda        = currentLambda;
      alchFlags.lambdaUp          = currentLambda;
      alchFlags.lambdaDown        = 1.0 - alchFlags.lambdaUp;
      alchFlags.elecLambdaUp      = simParams->getElecLambda(alchFlags.lambdaUp);
      alchFlags.elecLambdaDown    = simParams->getElecLambda(alchFlags.lambdaDown);
      alchFlags.vdwLambdaUp       = simParams->getVdwLambda(alchFlags.lambdaUp);
      alchFlags.vdwLambdaDown     = simParams->getVdwLambda(alchFlags.lambdaDown);
      alchFlags.lambda2Up         = currentLambda2;
      alchFlags.lambda2Down       = 1.0 - alchFlags.lambda2Up;
      alchFlags.elecLambda2Up     = simParams->getElecLambda(alchFlags.lambda2Up);
      alchFlags.elecLambda2Down   = simParams->getElecLambda(alchFlags.lambda2Down);
      alchFlags.vdwLambda2Up      = simParams->getVdwLambda(alchFlags.lambda2Up);
      alchFlags.vdwLambda2Down    = simParams->getVdwLambda(alchFlags.lambda2Down);
      alchFlags.vdwShiftUp        = alchFlags.alchVdwShiftCoeff*(1 - alchFlags.vdwLambdaUp);
      alchFlags.vdwShift2Up       = alchFlags.alchVdwShiftCoeff*(1 - alchFlags.vdwLambda2Up);
      alchFlags.vdwShiftDown      = alchFlags.alchVdwShiftCoeff*(1 - alchFlags.vdwLambdaDown);
      alchFlags.vdwShift2Down     = alchFlags.alchVdwShiftCoeff*(1 - alchFlags.vdwLambda2Down);
    }
  }

  if (doPairlist) {
    int numTileLists = calcNumTileLists();

    // Build initial tile lists and sort
    tileListKernel.buildTileLists(numTileLists, patches.size(), atomStorageSize,
      maxTileListLen, lata, latb, latc,
      cudaPatches, (const float4*)atoms, plcutoff2, maxShmemPerBlock, stream, atomsChanged, doAlch,
      CUDASOAintegrator, simParams->useDeviceMigration);
    // Prepare tile list for atom-based refinement
    tileListKernel.prepareTileList(stream);
    tileListKernel.clearTileListStat(stream);
  }
  
  if (atomsChanged) {
    // Update Vdw types and exclusion index & maxdiff
    updateVdwTypesExcl();
  }

  beforeForceCompute = CkWallTimer();

  // Calculate forces (and refine tile list if atomsChanged=true)
#if 0
  if(atomsChanged){
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    PatchData *patchData = cpdata.ckLocalBranch();

    CmiLock(patchData->printlock);
    fprintf(stderr, "DEV[%d] MIG POS PRINTOUT\n", deviceID);
    for (int p = 0; p < patches.size(); p++) {
      fprintf(stderr, "Patch Index %d. Patch ID %d\n", p, cudaPatches[p].patchID);
      for (int i = 0; i < patches[p].numAtoms; i++) {
        const int ai = i + patches[p].atomStart;
        fprintf(stderr, "POS[%d,%d,%d] = %lf %lf %lf %lf. Type %d\n", i, ai, atomIndex[ai],
        atoms[ai].x, atoms[ai].y, atoms[ai].z, atoms[ai].q, vdwTypes[ai]);
      }
    }
    CmiUnlock(patchData->printlock);
  }
#endif

  const bool doTable = CudaComputeNonbonded::getDoTable(params, doSlow, doVirial);
  const CudaNBConstants c = CudaComputeNonbonded::getNonbondedCoef(params);

#ifdef DEBUG_MINIMIZE
  printf("%s, line %d:\n", __FILE__, __LINE__);
  printf("  atomsChanged = %d\n", atomsChanged);
  printf("  doMinimize = %d\n", doMinimize);
  printf("  doPairlist = %d\n", doPairlist);
  printf("  doEnergy = %d\n", doEnergy);
  printf("  doVirial = %d\n", doVirial);
  printf("  doSlow = %d\n", doSlow);
  printf("\n");
#endif


  nonbondedKernel.nonbondedForce(tileListKernel, atomStorageSize,
      atomsChanged, doMinimize, doPairlist, doEnergy, doVirial, 
      doSlow, doAlch, doAlchVdwForceSwitching, doFEP, doTI,
      doTable, lata, latb, latc,
      (const float4*)atoms, cutoff2, c, 
      d_forces, d_forcesSlow, h_forces, h_forcesSlow,
      &alchFlags, lambdaWindowUpdated, part, CUDASOAintegrator, 
      params->useDeviceMigration, stream);

  if (doPairlist) {
    tileListKernel.finishTileList(stream);
  }

// TODO remove once GPU migration has been merged
#ifdef NODEGROUP_FORCE_REGISTER

  updatePatchRecord();

#endif


#if 0
  if(atomsChanged){
    
    copy_DtoH<float4>(d_forces, h_forces, atomStorageSize, stream);
    if (doSlow) copy_DtoH<float4>(d_forcesSlow, h_forcesSlow, atomStorageSize, stream);
    cudaStreamSynchronize(stream);
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    PatchData *patchData = cpdata.ckLocalBranch();

    CmiLock(patchData->printlock);
    fprintf(stderr, "DEV[%d] MIG POS PRINTOUT\n", deviceID);
    for (int p = 0; p < patches.size(); p++) {
      fprintf(stderr, "Patch Index %d. Patch ID %d\n", p, cudaPatches[p].patchID);
      for (int i = 0; i < patches[p].numAtoms; i++) {
        const int ai = i + patches[p].atomStart;
        fprintf(stderr, "POS[%d,%d,%d] = Type %d (%lf %lf %lf) (%lf %lf %lf)\n", i, ai, atomIndex[ai],
        vdwTypes[ai],
        h_forces[ai].x, h_forces[ai].y, h_forces[ai].z, 
        h_forcesSlow[ai].x, h_forcesSlow[ai].y, h_forcesSlow[ai].z);
      }
    }
    CmiUnlock(patchData->printlock);
  }
#endif




  traceUserBracketEvent(CUDA_DEBUG_EVENT, beforeForceCompute, CkWallTimer());
}

#ifdef NODEGROUP_FORCE_REGISTER
void CudaComputeNonbonded::updatePatchRecord() {
  // register device pointers inside nodegroup for later integration
  // these can be moved inside atomsChanged laters
  int devInd = deviceCUDA->getDeviceIndex();
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();
  PatchMap* patchMap = PatchMap::Object();
  patchData->devData[devInd].f_nbond = d_forces;
  patchData->devData[devInd].f_nbond_slow = d_forcesSlow;
  patchData->devData[devInd].f_nbond_size = atomStorageSize;
  // device pointer to CudaPatchRecord
  patchData->devData[devInd].nbond_precord = tileListKernel.getCudaPatches();
  patchData->devData[devInd].nb_precord_size = tileListKernel.getCudaPatchesSize();
  patchData->devData[devInd].nb_datoms = tileListKernel.get_xyzq();
  patchData->devData[devInd].nbond_tkernel = &tileListKernel;
  patchData->devData[devInd].size_nb_datoms = atomStorageSize;
}
#endif

//
// Count an upper estimate for the number of tile lists
//
int CudaComputeNonbonded::calcNumTileLists() {
  int numTileLists = 0;
  for (int i=0;i < computes.size();i++) {
    int pi1 = computes[i].patchInd[0];
    int numAtoms1 = patches[pi1].numAtoms;
#ifdef NAMD_HIP
    int numTiles1 = (numAtoms1-1)/BOUNDINGBOXSIZE+1;
#else
    int numTiles1 = (numAtoms1-1)/WARPSIZE+1;
#endif
    numTileLists += numTiles1;
  }
  return numTileLists;
}

//
// Finish & submit reductions
//
void CudaComputeNonbonded::finishReductions() {
  if (CkMyPe() != masterPe)
    NAMD_bug("CudaComputeNonbonded::finishReductions() called on non masterPe");

  // fprintf(stderr, "PE[%d]: Nbond finishReductions doSkip %d doVirial %d doEnergy %d\n", CkMyPe(), doSkip, doVirial, doEnergy);
  if (!doSkip) {

    if (doStreaming && (doVirial || doEnergy)) {
      // For streaming kernels, we must wait for virials and forces to be copied back to CPU
      if (!forceDoneEventRecord)
        NAMD_bug("CudaComputeNonbonded::finishReductions, forceDoneEvent not being recorded");
      cudaCheck(cudaEventSynchronize(forceDoneEvent));
      forceDoneEventRecord = false;
    }

    if (doVirial) {
      // if(params->CUDASOAintegrate) cudaCheck(cudaStreamSynchronize(stream));
      Tensor virialTensor;
      virialTensor.xx = h_virialEnergy->virial[0];
      virialTensor.xy = h_virialEnergy->virial[1];
      virialTensor.xz = h_virialEnergy->virial[2];
      virialTensor.yx = h_virialEnergy->virial[3];
      virialTensor.yy = h_virialEnergy->virial[4];
      virialTensor.yz = h_virialEnergy->virial[5];
      virialTensor.zx = h_virialEnergy->virial[6];
      virialTensor.zy = h_virialEnergy->virial[7];
      virialTensor.zz = h_virialEnergy->virial[8];
      // fprintf(stderr, "virialTensor %lf %lf %lf\n", virialTensor.xx, virialTensor.xy, virialTensor.xz);
      // fprintf(stderr, "virialTensor %lf %lf %lf\n", virialTensor.yx, virialTensor.yy, virialTensor.yz);
      // fprintf(stderr, "virialTensor %lf %lf %lf\n", virialTensor.zx, virialTensor.zy, virialTensor.zz);
#ifdef NODEGROUP_FORCE_REGISTER
      if (params->CUDASOAintegrate) {
        ADD_TENSOR_OBJECT(nodeReduction, REDUCTION_VIRIAL_NBOND, virialTensor);
      } else
#endif
      {
        ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NBOND, virialTensor);
      }
      if (doSlow) {
        Tensor virialTensor;
        virialTensor.xx = h_virialEnergy->virialSlow[0];
        virialTensor.xy = h_virialEnergy->virialSlow[1];
        virialTensor.xz = h_virialEnergy->virialSlow[2];
        virialTensor.yx = h_virialEnergy->virialSlow[3];
        virialTensor.yy = h_virialEnergy->virialSlow[4];
        virialTensor.yz = h_virialEnergy->virialSlow[5];
        virialTensor.zx = h_virialEnergy->virialSlow[6];
        virialTensor.zy = h_virialEnergy->virialSlow[7];
        virialTensor.zz = h_virialEnergy->virialSlow[8];
        // fprintf(stderr, "virialTensor (slow) %lf %lf %lf\n", virialTensor.xx, virialTensor.xy, virialTensor.xz);
        // fprintf(stderr, "virialTensor (slow) %lf %lf %lf\n", virialTensor.yx, virialTensor.yy, virialTensor.yz);
        // fprintf(stderr, "virialTensor (slow) %lf %lf %lf\n", virialTensor.zx, virialTensor.zy, virialTensor.zz);
#ifdef NODEGROUP_FORCE_REGISTER
        if (params->CUDASOAintegrate) {
          ADD_TENSOR_OBJECT(nodeReduction, REDUCTION_VIRIAL_SLOW, virialTensor);
        } else
#endif
        {
          ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_SLOW, virialTensor);
        }
      }
    }
    if (doEnergy) {
      // if (doSlow)
      
#ifdef NODEGROUP_FORCE_REGISTER
      if (params->CUDASOAintegrate) {
        nodeReduction->item(REDUCTION_LJ_ENERGY)    += h_virialEnergy->energyVdw;
        nodeReduction->item(REDUCTION_LJ_ENERGY_F)  += h_virialEnergy->energyVdw_s;
        nodeReduction->item(REDUCTION_ELECT_ENERGY) += h_virialEnergy->energyElec + ((params->GBISOn) ? h_virialEnergy->energyGBIS : 0.0);
        nodeReduction->item(REDUCTION_ELECT_ENERGY_F) += h_virialEnergy->energyElec_s;
        
        //Reduce values for TI
        nodeReduction->item(REDUCTION_LJ_ENERGY_TI_1) += h_virialEnergy->energyVdw_ti_1;
        nodeReduction->item(REDUCTION_LJ_ENERGY_TI_2) += h_virialEnergy->energyVdw_ti_2;
        nodeReduction->item(REDUCTION_ELECT_ENERGY_TI_1) += h_virialEnergy->energyElec_ti_1;
        nodeReduction->item(REDUCTION_ELECT_ENERGY_TI_2) += h_virialEnergy->energyElec_ti_2;
      } else 
#endif
      {
        //   printf("energyElec %lf energySlow %lf energyGBIS %lf\n", h_virialEnergy->energyElec, h_virialEnergy->energySlow, h_virialEnergy->energyGBIS);
        reduction->item(REDUCTION_LJ_ENERGY)    += h_virialEnergy->energyVdw;
        reduction->item(REDUCTION_LJ_ENERGY_F)  += h_virialEnergy->energyVdw_s;
        reduction->item(REDUCTION_ELECT_ENERGY) += h_virialEnergy->energyElec + ((params->GBISOn) ? h_virialEnergy->energyGBIS : 0.0);
        reduction->item(REDUCTION_ELECT_ENERGY_F) += h_virialEnergy->energyElec_s;
        
        //Reduce values for TI
        reduction->item(REDUCTION_LJ_ENERGY_TI_1) += h_virialEnergy->energyVdw_ti_1;
        reduction->item(REDUCTION_LJ_ENERGY_TI_2) += h_virialEnergy->energyVdw_ti_2;
        reduction->item(REDUCTION_ELECT_ENERGY_TI_1) += h_virialEnergy->energyElec_ti_1;
        reduction->item(REDUCTION_ELECT_ENERGY_TI_2) += h_virialEnergy->energyElec_ti_2;
      }

      // fprintf(stderr, "energyGBIS %lf\n", h_virialEnergy->energyGBIS);
       if (doSlow){
#ifdef NODEGROUP_FORCE_REGISTER
         if (params->CUDASOAintegrate) {
           nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW) += h_virialEnergy->energySlow;
           nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_F) += h_virialEnergy->energySlow_s;
           nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1) += h_virialEnergy->energySlow_ti_1;
           nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2) += h_virialEnergy->energySlow_ti_2;
           //fprintf(stderr, "NB h_virialEnergy->energySlow %lf\n", h_virialEnergy->energySlow);
         } else 
#endif
         {
           reduction->item(REDUCTION_ELECT_ENERGY_SLOW) += h_virialEnergy->energySlow;
           reduction->item(REDUCTION_ELECT_ENERGY_SLOW_F) += h_virialEnergy->energySlow_s;
           reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1) += h_virialEnergy->energySlow_ti_1;
           reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2) += h_virialEnergy->energySlow_ti_2;
           //fprintf(stderr, "NB h_virialEnergy->energySlow %lf\n", h_virialEnergy->energySlow);
         }
      }
    }

#ifdef NODEGROUP_FORCE_REGISTER
    if (params->CUDASOAintegrate) {
      nodeReduction->item(REDUCTION_EXCLUSION_CHECKSUM_CUDA) += tileListKernel.getNumExcluded();
    } else
#endif
    {
    reduction->item(REDUCTION_EXCLUSION_CHECKSUM_CUDA) += tileListKernel.getNumExcluded();
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


  // I need to get rid of this for every timestep. 
  if(!params->CUDASOAintegrate ) reduction->submit();
  // Reset flags
  doSkip = false;
  computesChanged = false;
}

//
// Finish a single patch
//
void CudaComputeNonbonded::finishPatch(int i) {
  if (CkMyPe() != patches[i].pe)
    NAMD_bug("CudaComputeNonbonded::finishPatch called on wrong Pe");
  
  PatchMap *map;
  PatchRecord &pr = patches[i];
  pr.results = pr.forceBox->open();
  SimParameters* simParams = Node::Object()->simParameters;
  
  const CompAtomExt *aExt = pr.patch->getCompAtomExtInfo();
  int atomStart = pr.atomStart;
  int numAtoms = pr.numAtoms;
#ifdef NODEGROUP_FORCE_REGISTER
  if (numAtoms > 0 && (!simParams->CUDASOAintegrate || (atomsChanged && !simParams->useDeviceMigration))) {
    Force *f      = pr.results->f[Results::nbond];
    Force *f_slow = pr.results->f[Results::slow];
    float4 *af      = h_forces + atomStart;
    float4 *af_slow = h_forcesSlow + atomStart;
    // float maxf = 0.0f;
    // int maxf_k;
    for ( int k=0; k<numAtoms; ++k ) {
      int j = aExt[k].sortOrder;
#ifdef DEBUG_MINIMIZE
      if (j == 0) {
        printf("%s, line %d\n", __FILE__, __LINE__);
        printf("  before:  f[%d] = %f %f %f\n", j, f[j].x, f[j].y, f[j].z);
      }
#endif
      f[j].x += af[k].x;
      f[j].y += af[k].y;
      f[j].z += af[k].z;
#ifdef DEBUG_MINIMIZE
      if (j == 0) {
        printf("  after:   f[%d] = %f %f %f\n", j, f[j].x, f[j].y, f[j].z);
      }
#endif
      // if (maxf < fabsf(af[k].x) || maxf < fabsf(af[k].y) || maxf < fabsf(af[k].z)) {
      //   maxf = std::max(maxf, fabsf(af[k].x));
      //   maxf = std::max(maxf, fabsf(af[k].y));
      //   maxf = std::max(maxf, fabsf(af[k].z));
      //   maxf_k = k;
      // }
      if ( doSlow ) {
        f_slow[j].x += af_slow[k].x;
        f_slow[j].y += af_slow[k].y;
        f_slow[j].z += af_slow[k].z;
      }
    }
    // if (maxf > 10000.0f) {
    //   fprintf(stderr, "%d %f %f %f\n", maxf_k, af[maxf_k].x, af[maxf_k].y, af[maxf_k].z);
    //   cudaCheck(cudaStreamSynchronize(stream));
    //   NAMD_die("maxf!");
    // }
  }
#else
  if (numAtoms > 0) {
    Force *f      = pr.results->f[Results::nbond];
    Force *f_slow = pr.results->f[Results::slow];
    float4 *af      = h_forces + atomStart;
    float4 *af_slow = h_forcesSlow + atomStart;
    // float maxf = 0.0f;
    // int maxf_k;
    for ( int k=0; k<numAtoms; ++k ) {
      int j = aExt[k].sortOrder;
#ifdef DEBUG_MINIMIZE
      if (j == 0) {
        printf("%s, line %d\n", __FILE__, __LINE__);
        printf("  before:  f[%d] = %f %f %f\n", j, f[j].x, f[j].y, f[j].z);
      }
#endif
      f[j].x += af[k].x;
      f[j].y += af[k].y;
      f[j].z += af[k].z;
#ifdef DEBUG_MINIMIZE
      if (j == 0) {
        printf("  after:   f[%d] = %f %f %f\n", j, f[j].x, f[j].y, f[j].z);
      }
#endif
      // if (maxf < fabsf(af[k].x) || maxf < fabsf(af[k].y) || maxf < fabsf(af[k].z)) {
      //   maxf = std::max(maxf, fabsf(af[k].x));
      //   maxf = std::max(maxf, fabsf(af[k].y));
      //   maxf = std::max(maxf, fabsf(af[k].z));
      //   maxf_k = k;
      // }
      if ( doSlow ) {
        f_slow[j].x += af_slow[k].x;
        f_slow[j].y += af_slow[k].y;
        f_slow[j].z += af_slow[k].z;
      }
    }
    // if (maxf > 10000.0f) {
    //   fprintf(stderr, "%d %f %f %f\n", maxf_k, af[maxf_k].x, af[maxf_k].y, af[maxf_k].z);
    //   cudaCheck(cudaStreamSynchronize(stream));
    //   NAMD_die("maxf!");
    // }
  }
#endif
  // should I skip the close()?
  // do I need to close it even if there's a migration?
  if(!simParams->CUDASOAintegrate || atomsChanged){
    pr.positionBox->close(&(pr.compAtom));
    pr.forceBox->close(&(pr.results));
  }
}

//
// Finish a set of patches on this pe
//
void CudaComputeNonbonded::finishSetOfPatchesOnPe(std::vector<int>& patchSet) {
  NAMD_EVENT_START(1, NamdProfileEvent::COMPUTE_NONBONDED_CUDA_FINISH_PATCHES);
  if (patchSet.size() == 0)
    NAMD_bug("CudaComputeNonbonded::finishPatchesOnPe, empty rank");
  SimParameters *simParams = Node::Object()->simParameters;
  // Save value of gbisPhase here because it can change after the last finishGBISPhase() or finishPatch() is called
  int gbisPhaseSave = gbisPhase;
  // Close Boxes depending on Phase
  if (simParams->GBISOn) {
    for (int i=0;i < patchSet.size();i++) {
      finishGBISPhase(patchSet[i]);
    }
  }
  // Finish patches
  if (!simParams->GBISOn || gbisPhaseSave == 3) {
    for (int i=0;i < patchSet.size();i++) {
      finishPatch(patchSet[i]);
    }
  }
  bool done = false;
  CmiLock(lock);
  patchesCounter -= patchSet.size();
  if(params->CUDASOAintegrate && !atomsChanged){
    //  masterPe is executing this, we can go ahead and do 
    //  reductions by themselves, However, for migrations, I still need to 
    //  follow the usual codepath, because of the box setup
    patchesCounter = 0;
  }
  if (patchesCounter == 0) {
    patchesCounter = getNumPatches();
    done = true;
  }
  CmiUnlock(lock);
  if (done) {
    // Do reductions
    if (!simParams->GBISOn || gbisPhaseSave == 3) {
      // Reduction must be done on masterPe
      if(params->CUDASOAintegrate ){
         if(!atomsChanged) this->finishReductions();
      }
      else computeMgr->sendFinishReductions(masterPe, this);
    }
  }

  NAMD_EVENT_STOP(1, NamdProfileEvent::COMPUTE_NONBONDED_CUDA_FINISH_PATCHES);

}

//
// Finish all patches that are on this pe
//
void CudaComputeNonbonded::finishPatchesOnPe() {
  finishSetOfPatchesOnPe(rankPatches[CkMyRank()]);
}

//
// Finish single patch on this pe
//
void CudaComputeNonbonded::finishPatchOnPe(int i) {
  std::vector<int> v(1, i);
  finishSetOfPatchesOnPe(v);
}

void CudaComputeNonbonded::finishPatches() {
  if(params->CUDASOAintegrate){
     if (atomsChanged || doEnergy || doVirial) cudaCheck(cudaStreamSynchronize(stream));
     this->finishPatchesOnPe();
   }
  else {
    computeMgr->sendFinishPatchesOnPe(pes, this);
  }
}

void CudaComputeNonbonded::finishGBISPhase(int i) {
  if (CkMyPe() != patches[i].pe)
    NAMD_bug("CudaComputeNonbonded::finishGBISPhase called on wrong Pe");
  PatchRecord &pr = patches[i];
  const CompAtomExt *aExt = pr.patch->getCompAtomExtInfo();
  int atomStart = pr.atomStart;
  if (gbisPhase == 1) {
    GBReal *psiSumMaster = psiSumH + atomStart;
    for ( int k=0; k<pr.numAtoms; ++k ) {
      int j = aExt[k].sortOrder;
      pr.psiSum[j] += psiSumMaster[k];
    }
    pr.psiSumBox->close(&(pr.psiSum));
  } else if (gbisPhase == 2) {
    GBReal *dEdaSumMaster = dEdaSumH + atomStart;
    for ( int k=0; k<pr.numAtoms; ++k ) {
      int j = aExt[k].sortOrder;
      pr.dEdaSum[j] += dEdaSumMaster[k];
    }
    pr.dEdaSumBox->close(&(pr.dEdaSum));
  } else if (gbisPhase == 3) {
    pr.intRadBox->close(&(pr.intRad)); //box 6
    pr.bornRadBox->close(&(pr.bornRad)); //box 7
    pr.dHdrPrefixBox->close(&(pr.dHdrPrefix)); //box 9
  } //end phases
}

void CudaComputeNonbonded::finishTimers() {
  SimParameters *simParams = Node::Object()->simParameters;

  if (simParams->GBISOn) {
    if (gbisPhase == 1)
      traceUserBracketEvent(CUDA_GBIS1_KERNEL_EVENT, beforeForceCompute, CkWallTimer());
    if (gbisPhase == 2)
      traceUserBracketEvent(CUDA_GBIS2_KERNEL_EVENT, beforeForceCompute, CkWallTimer());
    if (gbisPhase == 3)
      traceUserBracketEvent(CUDA_GBIS3_KERNEL_EVENT, beforeForceCompute, CkWallTimer());
  } else {
    traceUserBracketEvent(CUDA_NONBONDED_KERNEL_EVENT, beforeForceCompute, CkWallTimer());
  }
}

//
// Re-sort tile lists if neccessary
//
void CudaComputeNonbonded::reSortTileLists() {
  // Re-sort tile lists
  SimParameters *simParams = Node::Object()->simParameters;
  cudaCheck(cudaSetDevice(deviceID));
#ifdef NAMD_HIP
  tileListKernel.reSortTileLists(simParams->GBISOn, simParams->CUDASOAintegrateMode, stream);
#else
  tileListKernel.reSortTileLists(simParams->GBISOn, stream);
#endif
}

void CudaComputeNonbonded::forceDoneCheck(void *arg, double walltime) {
  CudaComputeNonbonded* c = (CudaComputeNonbonded *)arg;

  if (CkMyPe() != c->masterPe)
    NAMD_bug("CudaComputeNonbonded::forceDoneCheck called on non masterPe");

  SimParameters *simParams = Node::Object()->simParameters;
  cudaCheck(cudaSetDevice(c->deviceID));

  if (c->doStreaming) {
    int patchInd;
    while ( -1 != (patchInd = c->patchReadyQueue[c->patchReadyQueueNext]) ) {
      c->patchReadyQueue[c->patchReadyQueueNext] = -1;
      c->patchReadyQueueNext++;
      c->checkCount = 0;

      if ( c->patchReadyQueueNext == c->patchReadyQueueLen ) {
        c->finishTimers();
        if (c->atomsChanged && (!simParams->GBISOn || c->gbisPhase == 1) && !c->reSortDone) {
          c->reSortTileLists();
          c->reSortDone = true;
          if (simParams->GBISOn && c->gbisPhase == 1) {
            // We must do GBIS Phase 1
            c->doGBISphase1();
            c->forceDoneSetCallback();
            return;
          }
        }
      }

      // Finish patch
      int pe = c->patches[patchInd].pe;
      PatchID patchID = c->patches[patchInd].patchID;  // for priority
      //c->computeMgr->sendFinishPatchOnPe(pe, c, patchInd, patchID);
      if(c->params->CUDASOAintegrate) c->finishPatchOnPe(patchInd);
      else c->computeMgr->sendFinishPatchOnPe(pe, c, patchInd, patchID);

      // Last patch, return
      if ( c->patchReadyQueueNext == c->patchReadyQueueLen ) return;

    }
  } else {
    if (!c->forceDoneEventRecord)
      NAMD_bug("CudaComputeNonbonded::forceDoneCheck, forceDoneEvent not being recorded");
    cudaError_t err = cudaEventQuery(c->forceDoneEvent);
    if (err == cudaSuccess) {
      // Event has occurred
      c->forceDoneEventRecord = false;
      c->checkCount = 0;
      c->finishTimers();
      if (c->atomsChanged && (!simParams->GBISOn || c->gbisPhase == 1) && !c->reSortDone) {
        c->reSortTileLists();
        c->reSortDone = true;
        if (simParams->GBISOn && c->gbisPhase == 1) {
          // We must do GBIS Phase 1
          c->doGBISphase1();
          c->forceDoneSetCallback();
          return;
        }
      }
      c->finishPatches();
      return;
    } else if (err != cudaErrorNotReady) {
      // Anything else is an error
      char errmsg[256];
      sprintf(errmsg,"in CudaComputeNonbonded::forceDoneCheck after polling %d times over %f s",
              c->checkCount, walltime - c->beforeForceCompute);
      cudaDie(errmsg,err);
    }
  }

  // if (c->checkCount % 1000 == 0)
  //   fprintf(stderr, "c->patchReadyQueueNext %d\n", c->patchReadyQueueNext);

  // Event has not occurred
  c->checkCount++;
  if (c->checkCount >= 1000000) {
    char errmsg[256];
    sprintf(errmsg,"CudaComputeNonbonded::forceDoneCheck polled %d times over %f s",
            c->checkCount, walltime - c->beforeForceCompute);
    cudaDie(errmsg,cudaSuccess);
  }

  // Call again
  CcdCallBacksReset(0, walltime);
  // we need to do this only for the first timestep I guess?
  
  if(!c->params->CUDASOAintegrate) CcdCallFnAfter(forceDoneCheck, arg, 0.1);
}

//
// Set call back for all the work in the stream at this point
//
void CudaComputeNonbonded::forceDoneSetCallback() {
  if (CkMyPe() != masterPe)
    NAMD_bug("CudaComputeNonbonded::forceDoneSetCallback called on non masterPe");
  beforeForceCompute = CkWallTimer();
  cudaCheck(cudaSetDevice(deviceID));
  if (!doStreaming || doVirial || doEnergy) {
    cudaCheck(cudaEventRecord(forceDoneEvent, stream));
    forceDoneEventRecord = true;
  }
  checkCount = 0;
  CcdCallBacksReset(0, CmiWallTimer());
  // Set the call back at 0.1ms
  if(!params->CUDASOAintegrate) CcdCallFnAfter(forceDoneCheck, this, 0.1);
}

struct cr_sortop_distance {
  const Lattice &l;
  cr_sortop_distance(const Lattice &lattice) : l(lattice) { }
  bool operator() (CudaComputeNonbonded::ComputeRecord i,
      CudaComputeNonbonded::ComputeRecord j) {
    Vector a = l.a();
    Vector b = l.b();
    Vector c = l.c();
    BigReal ri = (i.offset.x * a + i.offset.y * b + i.offset.z * c).length2();
    BigReal rj = (j.offset.x * a + j.offset.y * b + j.offset.z * c).length2();
    return ( ri < rj );
  }
};

static inline bool sortop_bitreverse(int a, int b) {
  if ( a == b ) return 0;
  for ( int bit = 1; bit; bit *= 2 ) {
    if ( (a&bit) != (b&bit) ) return ((a&bit) < (b&bit));
  }
  return 0;
}

struct cr_sortop_reverse_priority {
  cr_sortop_distance &distop;
  const CudaComputeNonbonded::PatchRecord *pr;
  cr_sortop_reverse_priority(cr_sortop_distance &sod,
       const CudaComputeNonbonded::PatchRecord *patchrecs) : distop(sod), pr(patchrecs) { }
  bool pid_compare_priority(int2 pidi, int2 pidj) {
    const CudaComputeNonbonded::PatchRecord &pri = pr[pidi.y];
    const CudaComputeNonbonded::PatchRecord &prj = pr[pidj.y];
    if ( pri.isSamePhysicalNode && ! prj.isSamePhysicalNode ) return 0;
    if ( prj.isSamePhysicalNode && ! pri.isSamePhysicalNode ) return 1;
    if ( pri.isSameNode && ! prj.isSameNode ) return 0;
    if ( prj.isSameNode && ! pri.isSameNode ) return 1;
    if ( pri.isSameNode ) {  // and prj.isSameNode
      int rpri = pri.reversePriorityRankInPe;
      int rprj = prj.reversePriorityRankInPe;
      if ( rpri != rprj ) return rpri > rprj;
      return sortop_bitreverse(CkRankOf(pri.pe),CkRankOf(prj.pe));
    }
    int ppi = PATCH_PRIORITY(pidi.x);
    int ppj = PATCH_PRIORITY(pidj.x);
    if ( ppi != ppj ) return ppi < ppj;
    return pidi.x < pidj.x;
  }
  bool operator() (CudaComputeNonbonded::ComputeRecord j,
      CudaComputeNonbonded::ComputeRecord i) {  // i and j reversed
    // Choose patch i (= patch with greater priority)
    int2 pidi = pid_compare_priority(make_int2(i.pid[0], i.patchInd[0]), make_int2(i.pid[1], i.patchInd[1])) ? make_int2(i.pid[0], i.patchInd[0]) : make_int2(i.pid[1], i.patchInd[1]);
    // Choose patch j
    int2 pidj = pid_compare_priority(make_int2(j.pid[0], j.patchInd[0]), make_int2(j.pid[1], j.patchInd[1])) ? make_int2(j.pid[0], j.patchInd[0]) : make_int2(j.pid[1], j.patchInd[1]);
    if ( pidi.x != pidj.x ) return pid_compare_priority(pidi, pidj);
    return distop(i,j);
  }
};

//
// Setup computes. This is only done at the beginning and at load balancing, hence the lack of
// consideration for performance in the CPU->GPU memory copy.
//
void CudaComputeNonbonded::updateComputes() {
  cudaCheck(cudaSetDevice(deviceID));

  Lattice lattice = patches[0].patch->flags.lattice;
  cr_sortop_distance so(lattice);
  std::stable_sort(computes.begin(), computes.end(), so);

  if (doStreaming) {
    cr_sortop_reverse_priority sorp(so, patches.data());
    std::stable_sort(computes.begin(), computes.end(), sorp);
  }

  CudaComputeRecord* cudaComputes = new CudaComputeRecord[computes.size()];

  for (int i=0;i < computes.size();i++) {
    cudaComputes[i].patchInd.x = computes[i].patchInd[0];
    cudaComputes[i].patchInd.y = computes[i].patchInd[1];
    cudaComputes[i].offsetXYZ.x = computes[i].offset.x;
    cudaComputes[i].offsetXYZ.y = computes[i].offset.y;
    cudaComputes[i].offsetXYZ.z = computes[i].offset.z;
  }

  tileListKernel.updateComputes(computes.size(), cudaComputes, stream);
  cudaCheck(cudaStreamSynchronize(stream));

  delete [] cudaComputes;
}

struct exlist_sortop {
  bool operator() (int32 *li, int32 *lj) {
    return ( li[1] < lj[1] );
  }
};

//
// Builds the exclusions table. Swiped from ComputeNonbondedCUDA.C
//
void CudaComputeNonbonded::buildExclusions() {
  cudaCheck(cudaSetDevice(deviceID));

  Molecule *mol = Node::Object()->molecule;

#ifdef MEM_OPT_VERSION
  int natoms = mol->exclSigPoolSize;
#else
  int natoms = mol->numAtoms;
#endif

	if (exclusionsByAtom != NULL) delete [] exclusionsByAtom;
  exclusionsByAtom = new int2[natoms];

  // create unique sorted lists

  ObjectArena<int32> listArena;
  ResizeArray<int32*> unique_lists;
  int32 **listsByAtom = new int32*[natoms];
  SortableResizeArray<int32> curList;
  for ( int i=0; i<natoms; ++i ) {
    curList.resize(0);
    curList.add(0);  // always excluded from self
#ifdef MEM_OPT_VERSION
    const ExclusionSignature *sig = mol->exclSigPool + i;
    int n = sig->fullExclCnt;
    for ( int j=0; j<n; ++j ) { curList.add(sig->fullOffset[j]); }
    n += 1;
#else
    const int32 *mol_list = mol->get_full_exclusions_for_atom(i);
    int n = mol_list[0] + 1;
    for ( int j=1; j<n; ++j ) {
      curList.add(mol_list[j] - i);
    }
#endif
    curList.sort();

    int j;
    for ( j=0; j<unique_lists.size(); ++j ) {
      if ( n != unique_lists[j][0] ) continue;  // no match
      int k;
      for ( k=0; k<n; ++k ) {
        if ( unique_lists[j][k+3] != curList[k] ) break;
      }
      if ( k == n ) break;  // found match
    }
    if ( j == unique_lists.size() ) {  // no match
      int32 *list = listArena.getNewArray(n+3);
      list[0] = n;
      int maxdiff = 0;
      maxdiff = -1 * curList[0];
      if ( curList[n-1] > maxdiff ) maxdiff = curList[n-1];
      list[1] = maxdiff;
      for ( int k=0; k<n; ++k ) {
        list[k+3] = curList[k];
      }
      unique_lists.add(list);
    }
    listsByAtom[i] = unique_lists[j];
  }
  // sort lists by maxdiff
  std::stable_sort(unique_lists.begin(), unique_lists.end(), exlist_sortop());
  long int totalbits = 0;
  int nlists = unique_lists.size();
  for ( int j=0; j<nlists; ++j ) {
    int32 *list = unique_lists[j];
    int maxdiff = list[1];
    list[2] = totalbits + maxdiff;
    totalbits += 2*maxdiff + 1;
  }
  for ( int i=0; i<natoms; ++i ) {
    exclusionsByAtom[i].x = listsByAtom[i][1];  // maxdiff
    exclusionsByAtom[i].y = listsByAtom[i][2];  // start
  }
  delete [] listsByAtom;

  if ( totalbits & 31 ) totalbits += ( 32 - ( totalbits & 31 ) );

  {
    long int bytesneeded = totalbits / 8;
    if ( ! CmiPhysicalNodeID(CkMyPe()) ) {
    CkPrintf("Info: Found %d unique exclusion lists needing %ld bytes\n",
		unique_lists.size(), bytesneeded);
    }

    long int bytesavail = MAX_EXCLUSIONS * sizeof(unsigned int);
    if ( bytesneeded > bytesavail ) {
      char errmsg[512];
      sprintf(errmsg,"Found %d unique exclusion lists needing %ld bytes "
                     "but only %ld bytes can be addressed with 32-bit int.",
                     unique_lists.size(), bytesneeded, bytesavail);
      NAMD_die(errmsg);
    }
  }

#define SET_EXCL(EXCL,BASE,DIFF) \
         (EXCL)[((BASE)+(DIFF))>>5] |= (1<<(((BASE)+(DIFF))&31))

  unsigned int *exclusion_bits = new unsigned int[totalbits/32];
  memset(exclusion_bits, 0, totalbits/8);

  long int base = 0;
  for ( int i=0; i<unique_lists.size(); ++i ) {
    base += unique_lists[i][1];
    if ( unique_lists[i][2] != (int32)base ) {
      NAMD_bug("CudaComputeNonbonded::build_exclusions base != stored");
    }
    int n = unique_lists[i][0];
    for ( int j=0; j<n; ++j ) {
      SET_EXCL(exclusion_bits,base,unique_lists[i][j+3]);
    }
    base += unique_lists[i][1] + 1;
  }

  int numExclusions = totalbits/32;

  nonbondedKernel.bindExclusions(numExclusions, exclusion_bits);


  SimParameters *simParams = Node::Object()->simParameters;
  if(simParams->CUDASOAintegrate && simParams->useDeviceMigration){
    nonbondedKernel.setExclusionsByAtom(exclusionsByAtom, natoms);
  }

  delete [] exclusion_bits;
}

CudaNBConstants CudaComputeNonbonded::getNonbondedCoef(SimParameters *simParams) {
  const float cutoff = ComputeNonbondedUtil::cutoff;
  const float cutoff2 = ComputeNonbondedUtil::cutoff2;
  const float cutoffInv = 1.0f / cutoff;
  const float cutoff2Inv = 1.0f / cutoff2;
  const float scutoff = ComputeNonbondedUtil::switchOn;
  const float scutoff2 = ComputeNonbondedUtil::switchOn2;
  const float scutoff2Inv = 1.0f / scutoff2;
  const float scutoff_denom = ComputeNonbondedUtil::c1;
  const float ewaldcof = ComputeNonbondedUtil::ewaldcof;
  const float pi_ewaldcof = ComputeNonbondedUtil::pi_ewaldcof;
  const float slowScale = ((float) simParams->fullElectFrequency) / simParams->nonbondedFrequency;

  CudaNBConstants c;
  c.lj_0 = scutoff_denom * cutoff2 - 3.0f * scutoff2 * scutoff_denom;
  c.lj_1 = scutoff_denom * 2.0f;
  c.lj_2 = scutoff_denom * -12.0f;
  c.lj_3 = 12.0f * scutoff_denom * scutoff2;
  c.lj_4 = cutoff2;
  c.lj_5 = scutoff2;
  c.e_0 = cutoff2Inv * cutoffInv;
  c.e_0_slow = cutoff2Inv * cutoffInv * (1.0f - slowScale);
  c.e_1 = cutoff2Inv;
  c.e_2 = cutoffInv;
  c.ewald_0 = ewaldcof;
  c.ewald_1 = pi_ewaldcof;
  c.ewald_2 = ewaldcof * ewaldcof;
  c.ewald_3_slow = ewaldcof * ewaldcof * ewaldcof * slowScale;
  c.slowScale = slowScale;

  return c;
}

bool CudaComputeNonbonded::getDoTable(SimParameters *simParams, const bool doSlow, const bool doVirial) {
  // There is additional logic in SimParameters.C which guards against unsupported force fields
  // This should only be used for performance heuristics
  bool doTable = simParams->useCUDANonbondedForceTable;

  // DMC: I found the doSlow case is faster with force tables, so overriding setting
  // TODO This should be reevaluated for future architectures
  doTable = doTable || doSlow;
  // Direct math does not support virial+slow
  // Redundant but necessary for correctness so doing it explicitly
  doTable = doTable || (doSlow && doVirial);

  return doTable;
}

#endif // NAMD_CUDA


