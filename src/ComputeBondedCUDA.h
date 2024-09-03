#ifndef COMPUTEBONDEDCUDA_H
#define COMPUTEBONDEDCUDA_H
#include "Compute.h"
#include "ComputeMap.h"
#include "CudaNonbondedTables.h"
#include "ComputeBondedCUDAKernel.h"
#include "MigrationBondedCUDAKernel.h"
#include "ComputeHomeTuples.h"
#if defined(NAMD_CUDA) || defined(NAMD_HIP)

#ifdef BONDED_CUDA

#include <vector>
#include <array>

class ComputeBondedCUDA : public Compute {

public:

  static const int CudaTupleTypeSize[Tuples::NUM_TUPLE_TYPES];
  static const int CudaTupleTypeSizeStage[Tuples::NUM_TUPLE_TYPES];

private:
  bool initializeCalled;
  SimParameters *params;
  // Device ID and stream
  const int deviceID;
  cudaStream_t stream;
#ifdef NODEGROUP_FORCE_REGISTER
  std::atomic<int> tupleWorkIndex;
#endif

  // Master PE for this compute
  const int masterPe;

  // List of all patch IDs on this object
  std::vector<int> allPatchIDs;

  // List of tuple patches for the entire compute (i.e. across all PEs)
  TuplePatchList tuplePatchList;

  // For every PE, list of patches that it has registered
  std::vector< std::vector<int> > patchIDsPerRank;

  // List of PEs involved in the computation
  std::vector<int> pes;

  // Self compute
  struct SelfCompute {
    int type;
    std::vector<int> patchIDs;
    Tuples* tuples;
    SelfCompute(int type=-1) : type(type), tuples(NULL) {}
    int operator==(const SelfCompute &elem) const {
      return (elem.type == type);
    }
  };

  // Home compute, each PE has one
  struct HomeCompute {
    std::vector<char> isBasePatch;
    std::vector<int> patchIDs;
    // Multiple tuples per PE, each of different kind
    std::vector< Tuples* > tuples;
  };

  // Computes for each PE
  struct ComputeRecord {
    HomeCompute homeCompute;
    // Self computes, organized by type
    std::vector< SelfCompute > selfComputes;
  };
  
  // Collection of all computes for each PE
  std::vector< ComputeRecord > computes;

  // For every tuple type, list of tuples
  // NOTE: These are pointers to the data recorded in "computes" and
  //       are here to make it easier to traverse across all tuples of certain kind
  std::array< std::list<Tuples*>, Tuples::NUM_TUPLE_TYPES > tupleList;

  int numTuplesPerType[Tuples::NUM_TUPLE_TYPES];

  AtomMap atomMap;
  std::vector< AtomMapper* > atomMappers;

  /*struct PatchRecord {
    int atomStart;
    int numAtoms;
  };*/
  std::vector<PatchRecord> patches;

  // Patch "patchID" is found in patches[patchIndex[patchID]]
  std::vector<int> patchIndex;

  // Maps multiplicit indices
  std::vector<int> dihedralMultMap;
  std::vector<int> improperMultMap;

  // Number of exclusions per rank, separated into modified and non-modified
  struct NumExcl {
    int numModifiedExclusions;
    int numExclusions;
  };
  std::vector<NumExcl> numExclPerRank;

  // Flags that indicate wether this GPU has exclusions and modified exclusions
  bool hasExclusions;
  bool hasModifiedExclusions;

  // All tuple data
  char* tupleData;
  size_t tupleDataSize;

  std::vector<CudaBondStage> bondTupleData;
  std::vector<CudaAngleStage> angleTupleData;
  std::vector<CudaDihedralStage> dihedralTupleData;
  std::vector<CudaDihedralStage> improperTupleData;
  std::vector<CudaExclusionStage> modifiedExclusionTupleData;
  std::vector<CudaExclusionStage> exclusionTupleData;
  std::vector<CudaCrosstermStage> crosstermTupleData;

  // Bonded CUDA kernel
  ComputeBondedCUDAKernel bondedKernel;
#ifdef NODEGROUP_FORCE_REGISTER
  MigrationBondedCUDAKernel migrationKernel;
#endif  // NODEGROUP_FORCE_REGISTER

  // Pointer to computeMgr that created this object
  ComputeMgr* computeMgr;

  // Node-wide counter for patches.
  int patchesCounter;

  // Tuple migration data structures
  double3* h_patchMapCenter;
  double3* d_patchMapCenter;

  PatchRecord* d_patchRecord;
  PatchRecord* h_patchRecord;

  // "Force done event" for event polling
  cudaEvent_t forceDoneEvent;

  // Check counter for event polling
  int checkCount;

  // Node lock
  CmiNodeLock lock;
  CmiNodeLock printLock;

  // This variable is set in atomUpdate() by any Pe
  bool atomsChangedIn;
  // This variable is set in doWork() by masterPe
  bool atomsChanged;

  // Reduction
  SubmitReduction *reduction;
  NodeReduction   *nodeReduction;

  // Required storage
  int atomStorageSize;

  // Flags pointer
  Flags* flags;

  // Lattice and energy and virial booleans
  Lattice lattice;
  bool doEnergy;
  bool doVirial;
  bool doSlow;
  bool doMolly;

  // Current step, for alchemical route
  int step;

  // Walltime for force compute start
  double beforeForceCompute;

  bool accelMDdoDihe;

  // Atom storage in pinned host memory
  CudaAtom* atoms;
  size_t atomsSize;

  // Force storage in pinned host memory
  FORCE_TYPE* forces;
  size_t forcesSize;
  int forcesSizeDP;

  double* energies_virials;

  CudaAlchFlags hostAlchFlags;
  CudaAlchParameters hostAlchParameters;
  CudaAlchLambdas hostAlchLambdas;
  int pswitchTable[3*3];

  void mapAtoms();
  void unmapAtoms();

  void updatePatches();

  static void forceDoneCheck(void *arg, double walltime);
  void forceDoneSetCallback();


  // ------------ For copyTupleData -------------------
  struct TupleCopyWork {
    int tupletype;
    int ntuples;
    void* tupleElemList;
    int64_t tupleDataPos;
  };

  std::vector<TupleCopyWork> tupleCopyWorkList;

  int64_t exclusionStartPos;
  int64_t exclusionStartPos2;
  std::vector<CudaBondStage> hostCudaBondStage;

#ifdef NODEGROUP_FORCE_REGISTER 
  template <typename T>
  void sortTupleList(std::vector<T>& tuples, std::vector<int>& tupleCounts, std::vector<int>& tupleOffsets);
  void sortAndCopyToDevice();
  void migrateTuples(bool startup);

  template <typename T, typename P, typename D>
    void copyTupleToStage(const T& src, const P* __restrict__ p_array, D& dstval);

  template <typename T, typename P, typename D>
    void copyToStage(const int ntuples, const T* __restrict__ src,
    const P* __restrict__ p_array, std::vector<D>& dst);

  void copyExclusionDataStage(const int ntuples, const ExclElem* __restrict__ src, const int typeSize,
    std::vector<CudaExclusionStage>& dst1, std::vector<CudaExclusionStage>& dst2, int64_t& pos, int64_t& pos2);
#endif

  void copyBondData(const int ntuples, const BondElem* __restrict__ src,
    const BondValue* __restrict__ bond_array, CudaBond* __restrict__ dst);

  void copyBondDatafp32(const int ntuples, const BondElem* __restrict__ src,
    const BondValue* __restrict__ bond_array, CudaBond* __restrict__ dst);

  void copyAngleData(const int ntuples, const AngleElem* __restrict__ src,
    const AngleValue* __restrict__ angle_array, CudaAngle* __restrict__ dst);

  template <bool doDihedral, typename T, typename P>
  void copyDihedralData(const int ntuples, const T* __restrict__ src,
    const P* __restrict__ p_array, CudaDihedral* __restrict__ dst);
  
  template <bool doDihedral, typename T, typename P>
  void copyDihedralDatafp32(const int ntuples, const T* __restrict__ src,
    const P* __restrict__ p_array, CudaDihedral* __restrict__ dst);

  void copyExclusionData(const int ntuples, const ExclElem* __restrict__ src, const int typeSize,
    CudaExclusion* __restrict__ dst1, CudaExclusion* __restrict__ dst2, int64_t& pos, int64_t& pos2);

  void copyCrosstermData(const int ntuples, const CrosstermElem* __restrict__ src,
    const CrosstermValue* __restrict__ crossterm_array, CudaCrossterm* __restrict__ dst);

  static void tupleCopyWorker(int first, int last, void *result, int paraNum, void *param);
  void tupleCopyWorker(int first, int last);

#ifdef NODEGROUP_FORCE_REGISTER
  void tupleCopyWorkerType(int tupletype);
#endif
  // --------------------------------------------------

public:

  ComputeBondedCUDA(ComputeID c, ComputeMgr* computeMgr, int deviceID, CudaNonbondedTables& cudaNonbondedTables);
  ~ComputeBondedCUDA();
  void registerCompute(int pe, int type, PatchIDList& pids);
  void registerSelfCompute(int pe, int type, int pid);
  void unregisterBoxesOnPe();
  void assignPatchesOnPe();
  virtual void patchReady(PatchID, int doneMigration, int seq);
  virtual void initialize();
  virtual void atomUpdate();
  virtual int noWork();
  virtual void doWork();
  void messageEnqueueWork();
  // void updatePatches();
  void openBoxesOnPe(int startup = 1);
  void loadTuplesOnPe(const int startup = 1);
  void copyTupleData();
  void copyTupleDataSN();
  void launchWork();
  void updateCudaAlchParameters();

  void updateHostCudaAlchFlags();
  void updateKernelCudaAlchFlags();
  void updateHostCudaAlchParameters();
  void updateKernelCudaAlchParameters();
  void updateHostCudaAlchLambdas();
  void updateKernelCudaAlchLambdas();

#ifdef NODEGROUP_FORCE_REGISTER
  void updatePatchRecords();
  void updateMaxTupleCounts(TupleCounts counts);
  TupleCounts getMaxTupleCounts();
  void registerPointersToHost();
  void copyHostRegisterToDevice(); 
  void copyPatchData();
  void copyTupleDataGPU(const int startup);
  void updatePatchOrder(std::vector<CudaLocalRecord>& data);
#endif  // NODEGROUP_FORCE_REGISTER

  void finishPatchesOnPe();
  void finishPatches();
  void finishReductions();

  std::vector<int>& getBondedPes(void) {return pes;}
 
  std::vector<PatchRecord>& getPatches() { return patches; }
};

#endif // BONDED_CUDA
#endif // NAMD_CUDA
#endif // COMPUTEBONDEDCUDA_H
