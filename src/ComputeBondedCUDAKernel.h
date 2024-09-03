#ifndef COMPUTEBONDEDCUDAKERNEL_H
#define COMPUTEBONDEDCUDAKERNEL_H
#include "CudaUtils.h"
#include "TupleTypesCUDA.h"
#include "CudaNonbondedTables.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)

#define FORCE_TYPE double

#define USE_STRIDED_FORCE

#ifndef USE_STRIDED_FORCE
#error "Non-USE_STRIDED_FORCE not implemented"
#endif

#define WRITE_FULL_VIRIALS

#define USE_BONDED_FORCE_ATOMIC_STORE

class ComputeBondedCUDAKernel {
public:

  // Enumeration for energies_virials[]
  enum {energyIndex_BOND=0, energyIndex_ANGLE, energyIndex_DIHEDRAL, energyIndex_IMPROPER,
    energyIndex_ELECT, energyIndex_LJ, energyIndex_ELECT_SLOW, energyIndex_CROSSTERM,
    energyIndex_BOND_F, energyIndex_BOND_TI_1, energyIndex_BOND_TI_2, // Alchemical, bond energy
    energyIndex_ANGLE_F, energyIndex_ANGLE_TI_1, energyIndex_ANGLE_TI_2, // Alchemical, angle energy
    energyIndex_DIHEDRAL_F, energyIndex_DIHEDRAL_TI_1, energyIndex_DIHEDRAL_TI_2, // Alchemical, dihedral energy
    energyIndex_IMPROPER_F, energyIndex_IMPROPER_TI_1, energyIndex_IMPROPER_TI_2, // Alchemical, improper energy
    energyIndex_ELECT_F, energyIndex_ELECT_TI_1, energyIndex_ELECT_TI_2,
    energyIndex_LJ_F, energyIndex_LJ_TI_1, energyIndex_LJ_TI_2,
    energyIndex_ELECT_SLOW_F, energyIndex_ELECT_SLOW_TI_1, energyIndex_ELECT_SLOW_TI_2,
    energyIndex_CROSSTERM_F, energyIndex_CROSSTERM_TI_1, energyIndex_CROSSTERM_TI_2, // Alchemical, cross-term energy
    normalVirialIndex_XX, normalVirialIndex_XY, normalVirialIndex_XZ,
    normalVirialIndex_YX, normalVirialIndex_YY, normalVirialIndex_YZ,
    normalVirialIndex_ZX, normalVirialIndex_ZY, normalVirialIndex_ZZ,
    nbondVirialIndex_XX, nbondVirialIndex_XY, nbondVirialIndex_XZ,
    nbondVirialIndex_YX, nbondVirialIndex_YY, nbondVirialIndex_YZ,
    nbondVirialIndex_ZX, nbondVirialIndex_ZY, nbondVirialIndex_ZZ,
    slowVirialIndex_XX, slowVirialIndex_XY, slowVirialIndex_XZ,
    slowVirialIndex_YX, slowVirialIndex_YY, slowVirialIndex_YZ,
    slowVirialIndex_ZX, slowVirialIndex_ZY, slowVirialIndex_ZZ,
    amdDiheVirialIndex_XX, amdDiheVirialIndex_XY, amdDiheVirialIndex_XZ,
    amdDiheVirialIndex_YX, amdDiheVirialIndex_YY, amdDiheVirialIndex_YZ,
    amdDiheVirialIndex_ZX, amdDiheVirialIndex_ZY, amdDiheVirialIndex_ZZ,
    energies_virials_SIZE};

  template <typename T>
  struct BondedVirial {
#ifdef WRITE_FULL_VIRIALS
    T xx;
    T xy;
    T xz;
    T yx;
    T yy;
    T yz;
    T zx;
    T zy;
    T zz;
#else
#error "non-WRITE_FULL_VIRIALS not implemented yet"
    union {
      double sforce_dp[27][3];
      long long int sforce_fp[27][3];
    };
#endif
  };
  double *getForces(){
    return forces;
  }
private:
  const int deviceID;
  CudaNonbondedTables& cudaNonbondedTables;

  // This stores all bonds, angles, dihedrals, and impropers in a single 
  // contigious memory array.
  char* tupleData;
  size_t tupleDataSize;

  // ---------------------------------------------------------------------------------
  // NOTE: bonds, angles, dihedrals, impropers, etc. - pointers below are 
  // computed pointers pointing to tupleData -array
  // DO NOT DEALLOCATE THESE!
  int numBonds;
  CudaBond* bonds;

  int numAngles;
  CudaAngle* angles;

  int numDihedrals;
  CudaDihedral* dihedrals;

  int numImpropers;
  CudaDihedral* impropers;

  int numModifiedExclusions;
  CudaExclusion* modifiedExclusions;

  int numExclusions;
  CudaExclusion* exclusions;

  int numCrossterms;
  CudaCrossterm* crossterms;
  // ---------------------------------------------------------------------------------
  
  // Device memory for coordinates
  float4* xyzq;
  size_t xyzqSize;

  FORCE_TYPE* forceList;
  int* forceListCounter;
  int* forceListStarts;
  int* forceListNexts;
  int forceListSize;
  int forceListStartsSize;
  int forceListNextsSize;

  // Device memory for forces:
  // [normal, nbond, slow]
  FORCE_TYPE* forces;
  size_t forcesSize;

  CudaBondValue* bondValues;
  CudaAngleValue* angleValues;
  CudaDihedralValue* dihedralValues;
  CudaDihedralValue* improperValues;
  CudaCrosstermValue* crosstermValues;

  // Accumulated energy values for every bonded type
  double* energies_virials;

  // Alchemical flags
  CudaAlchFlags       alchFlags;

public:

  ComputeBondedCUDAKernel(int deviceID, CudaNonbondedTables& cudaNonbondedTables);
  ~ComputeBondedCUDAKernel();

  static constexpr float kTupleOveralloc = 1.4f;
  static int warpAlign(const int n) {return ((n + WARPSIZE - 1)/WARPSIZE)*WARPSIZE;} 

  void update(
    const int numBondsIn,
    const int numAnglesIn,
    const int numDihedralsIn,
    const int numImpropersIn,
    const int numModifiedExclusionsIn,
    const int numExclusionsIn,
    const int numCrosstermsIn,
    const char* h_tupleData,
    cudaStream_t stream);

  void setTupleCounts(
    const TupleCounts count);
  size_t reallocateTupleBuffer(
    const TupleCounts countIn,
    cudaStream_t stream);
  void updateAtomBuffer(
    const int atomStorageSize,
    cudaStream_t stream);

  TupleCounts getTupleCounts();
  TupleData getData();

  void setupBondValues(int numBondValues, CudaBondValue* h_bondValues);
  void setupAngleValues(int numAngleValues, CudaAngleValue* h_angleValues);
  void setupDihedralValues(int numDihedralValues, CudaDihedralValue* h_dihedralValues);
  void setupImproperValues(int numImproperValues, CudaDihedralValue* h_improperValues);
  void setupCrosstermValues(int numCrosstermValues, CudaCrosstermValue* h_crosstermValues);

  int getForceStride(const int atomStorageSize);
  int getForceSize(const int atomStorageSize);
  int getAllForceSize(const int atomStorageSize, const bool doSlow);
  float4* getAtomBuffer(){ return xyzq;}

  void bondedForce(
    const double scale14, const int atomStorageSize,
    const bool doEnergy, const bool doVirial, const bool doSlow,
    const bool doTable,
    const float3 lata, const float3 latb, const float3 latc,
    const float cutoff2, const float r2_delta, const int r2_delta_expc,
    const CudaNBConstants nbConstants,
    const float4* h_xyzq, double* h_forces, 
    double *h_energies, bool atomsChanged, 
    bool CUDASOAintegratorOn, bool useDeviceMigration,
    cudaStream_t stream);

  void updateCudaAlchFlags(const CudaAlchFlags& h_cudaAlchFlags);
  void updateCudaAlchParameters(const CudaAlchParameters* h_cudaAlchParameters, cudaStream_t stream);
  void updateCudaAlchLambdas(const CudaAlchLambdas* h_cudaAlchLambdas, cudaStream_t stream);
};

#endif

#endif // COMPUTEBONDEDCUDAKERNEL_H
