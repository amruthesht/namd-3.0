#ifndef CUDACOMPUTENONBONDED_H
#define CUDACOMPUTENONBONDED_H

#ifdef NAMD_CUDA
#include <cuda.h>
#endif
#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif

#include <vector>
#include "Compute.h"
#include "Box.h"
#include "PatchTypes.h"
#include "CudaUtils.h"
#include "ComputeNonbondedUtil.h"
#include "CudaNonbondedTables.h"
#include "CudaTileListKernel.h"
#include "CudaTileListKernel.hip.h"
#include "CudaComputeNonbondedKernel.h"
#include "CudaComputeNonbondedKernel.hip.h"
#include "CudaComputeGBISKernel.h"
#include "ComputeMgr.h"
#include "HipDefines.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
// 2^11 ints * 2^5 bits = 2^16 bits = range of unsigned short excl_index
// 2^27 ints * 2^5 bits = 2^32 bits = range of unsigned int excl_index
#define MAX_EXCLUSIONS (1<<27)

class CudaComputeNonbonded : public Compute, public ComputeNonbondedUtil {
public:
  struct ComputeRecord {
    ComputeID cid;
    PatchID pid[2];
    // Index to patches[] -array
    int patchInd[2];
    Vector offset;
  };

  struct PatchRecord {
    PatchRecord(PatchID patchID) : patchID(patchID) {
      patch = NULL;
      compAtom = NULL;
      results = NULL;
      positionBox = NULL;
      forceBox = NULL;
      intRadBox = NULL;
      psiSumBox = NULL;
      bornRadBox = NULL;
      dEdaSumBox = NULL;
      dHdrPrefixBox = NULL;
    }
    PatchID patchID;
    Patch *patch;
    int numAtoms;
    int numFreeAtoms;
    int atomStart;
    // Pe where the patch was registered
    int pe;
    // For priority sorting
    int reversePriorityRankInPe;
    bool isSamePhysicalNode;
    bool isSameNode;
    // Storage for open positionBox
    CompAtom *compAtom;
    // Storage for open forceBox
    Results *results;
    // Boxes
    Box<Patch,CompAtom> *positionBox;
    Box<Patch,Results> *forceBox;
    Box<Patch,Real>   *intRadBox; //5 GBIS Boxes
    Box<Patch,GBReal> *psiSumBox;
    Box<Patch,Real>   *bornRadBox;
    Box<Patch,GBReal> *dEdaSumBox;
    Box<Patch,Real>   *dHdrPrefixBox;
    Real   *intRad; //5 GBIS arrays
    GBReal *psiSum;
    Real   *bornRad;
    GBReal *dEdaSum;
    Real   *dHdrPrefix;
    bool operator < (const PatchRecord& pr) const {
      return (patchID < pr.patchID);
    }
    bool operator == (const PatchRecord& pr) const {
      return (patchID == pr.patchID);
    }
  };

private:
  SimParameters *params; // convenience
  // This variable is set in atomUpdate() by any Pe
  bool atomsChangedIn;
  // This variable is set in doWork() by masterPe
  bool atomsChanged;
  int npairlists;
  
  bool computesChanged;

  const int deviceID;
  size_t maxShmemPerBlock;
  cudaStream_t stream;

  // PME and VdW CUDA kernels
  CudaComputeNonbondedKernel nonbondedKernel;

  // GBIS kernel
  CudaComputeGBISKernel GBISKernel;

  // Tile list CUDA kernels
  CudaTileListKernel tileListKernel;

  // Exclusions
  int2 *exclusionsByAtom;

  // VdW-types
  // Pinned host memory
  int* vdwTypes;
  size_t vdwTypesSize;

  // Maximum number of tiles per tile list
  int maxTileListLen;

  // Pinned host memory
  int2* exclIndexMaxDiff;
  size_t exclIndexMaxDiffSize;

  // Pinned host memory
  int* atomIndex;
  size_t atomIndexSize;

  // Required (xyzq, vdwTypes) storage
	int atomStorageSize;

  // Atom and charge storage
  // Pinned host memory
  CudaAtom* atoms;
  size_t atomsSize;

  char *part;
  size_t partSize;

  // Force storage
  float4* h_forces;
  size_t h_forcesSize;
  float4* h_forcesSlow;
  size_t h_forcesSlowSize;

  float4* d_forces;
  size_t d_forcesSize;
  float4* d_forcesSlow;
  size_t d_forcesSlowSize;

  // Virial and energy storage
  VirialEnergy* h_virialEnergy;
  VirialEnergy* d_virialEnergy;

  // GBIS storage
  //--------------
  // Pinned host memory
  float* intRad0H;
  size_t intRad0HSize;
  // Pinned host memory
  float* intRadSH;
  size_t intRadSHSize;
  // Mapped host memory
  GBReal* psiSumH;
  size_t psiSumHSize;
  // Pinned host memory
  float* bornRadH;
  size_t bornRadHSize;
  // Mapped host memory
  GBReal* dEdaSumH;
  size_t dEdaSumHSize;
  // Pinned host memory
  float* dHdrPrefixH;
  size_t dHdrPrefixHSize;

  // Event and sanity check flag for making sure event was actually recorded
  cudaEvent_t forceDoneEvent;
  bool forceDoneEventRecord;
  // Check counter for event polling
  int checkCount;

  // Node lock
  CmiNodeLock lock;
  // List of local PEs that have patches
  std::vector<int> pes;
  // List of patch indices on each rank
  std::vector< std::vector<int> > rankPatches;
  // Master Pe = Pe where this Compute and reduction lives
  int masterPe;

  // Are we in skip?
  bool doSkip;

  // Device-wide patch and compute records, and the list of patches
  std::vector<ComputeRecord> computes;
  std::vector<PatchRecord> patches;

  // CUDA versions of patches
  // Pinned host memory
  CudaPatchRecord* cudaPatches;

  SubmitReduction *reduction;
  NodeReduction   *nodeReduction;

  // Pair lists
  int pairlistsValid;
  float pairlistTolerance;
  int usePairlists;
  int savePairlists;
  float plcutoff2;

  bool reSortDone;

  // Flags
  bool doSlow;
  bool doEnergy;
  bool doVirial;
  bool doAlch;
  bool doMinimize;

  AlchData alchFlags;
  bool lambdaWindowUpdated;
  // Walltime for force compute start
  double beforeForceCompute;

  static inline void updateVdwTypesExclLoop(int first, int last, void *result, int paraNum, void *param);
  void updateVdwTypesExclSubset(int first, int last);

  static inline void copyAtomsLoop(int first, int last, void *result, int paraNum, void *param);
  void copyAtomsSubset(int first, int last);

  void addPatch(PatchID pid);
  void addCompute(ComputeID cid, PatchID pid1, PatchID pid2, Vector offset);
  void updatePatches();
  int calcNumTileLists();
  void getMaxMovementTolerance(float& maxAtomMovement, float& maxPatchTolerance);
  void updateVdwTypesExcl();
  void buildNeighborlist();
  void skip();
  void doGBISphase1();
  void doGBISphase2();
  void doGBISphase3();
  void doForce();
  void finishSetOfPatchesOnPe(std::vector<int>& patchSet);
  void finishGBISPhase(int i);
  void finishTimers();
  void forceDone();
  static void forceDoneCheck(void *arg, double walltime);
  void forceDoneSetCallback();
  void updateComputes();
  void buildExclusions();
  void skipPatch(int i);
  void openBox(int i);
  void reallocateArrays();
#ifdef NODEGROUP_FORCE_REGISTER
  void updatePatchRecord();
#endif
  void copyGBISphase(int i);
  void updatePatch(int i);
  int findPid(PatchID pid);
  void assignPatch(int i);
  ComputeMgr* computeMgr;
  int patchesCounter;

  const bool doStreaming;
  int* patchReadyQueue;
  int patchReadyQueueNext, patchReadyQueueLen;

  void finishPatch(int i);
  void unregisterBox(int i);

  // void writeId(const char* filename);
  // void writeXYZ(const char* filename);

public:
  CudaComputeNonbonded(ComputeID c, int deviceID, CudaNonbondedTables& cudaNonbondedTables, bool doStreaming);
  ~CudaComputeNonbonded();
  void registerComputeSelf(ComputeID cid, PatchID pid);
  void registerComputePair(ComputeID cid, PatchID* pid, int* trans);
  void assignPatches(ComputeMgr* computeMgrIn);
  virtual void initialize();
  virtual void atomUpdate();
  virtual int noWork();
  virtual void doWork();
  void launchWork();
  void finishReductions();
  void unregisterBoxesOnPe();
  void assignPatchesOnPe();
  void openBoxesOnPe();
  void skipPatchesOnPe();
  void finishPatchesOnPe();
  void finishPatchOnPe(int i);
  void finishPatches();
  void messageEnqueueWork();
  virtual void patchReady(PatchID, int doneMigration, int seq);
  virtual void gbisP2PatchReady(PatchID, int seq);
  virtual void gbisP3PatchReady(PatchID, int seq);
  void reSortTileLists();

  void updatePatchOrder(std::vector<CudaLocalRecord>& data);
  std::vector<PatchRecord>& getPatches() { return patches; }

  // Utility function to compute nonbonded parameters, used by ComputeBondedCUDAKernel as well
  static CudaNBConstants getNonbondedCoef(SimParameters* params);
  // Utility function to determine if force table will be used, used by ComputeBondedCUDAKernel as well
  static bool getDoTable(SimParameters* params, const bool doSlow, const bool doVirial);
};

#endif // NAMD_CUDA
#endif // CUDACOMPUTENONBONDED_H
