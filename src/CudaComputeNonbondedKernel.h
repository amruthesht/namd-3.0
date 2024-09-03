#ifndef CUDACOMPUTENONBONDEDKERNEL_H
#define CUDACOMPUTENONBONDEDKERNEL_H
#include "CudaUtils.h"
#include "CudaRecord.h"
#include "CudaTileListKernel.h"
#include "CudaNonbondedTables.h"
#ifdef NAMD_CUDA


/*! \brief Alchemical datastructure that holds the lambda-relevant paramenters for FEP/TI
 * 
 * JM:  For every force evaluation, we copy the alchemical parameters in SimParameters onto the 
 *      following data structure, which then gets copied to the constant memory. This whole thing 
 *	could be done only once per lambda-window.
 *	For the elec* and vdw* parameters, we should use the getter functions on SimParams to 
 *	calculate the current lambdas. 
 * Haochuan (Updated on 2022-10-31):
 *      Some alchemical parameters could depends on the simulation step. For example, lambda2Up 
 *      could be changed every step if IDWS is on and alchIDWSFreq is set to 1 (see 
 *      SimParameters::getCurrentLambda2 in SimParameters.C for more information). To solve this issue 
 *      without sacrificing the performance, we need to compute the parameters every step on CPU, 
 *      compare them to the previous cached results, and copy them to GPU memory if they are changed.
 */
struct AlchData{
  float scaling;
  float switchdist2;
  float cutoff2;
  float switchfactor;
  float alchVdwShiftCoeff;
  // float alchLambda;
  float lambdaUp;
  float lambdaDown;
  float elecLambdaUp;
  float elecLambdaDown;
  float vdwLambdaUp;
  float vdwLambdaDown;

  float lambda2Up;
  float lambda2Down;
  float elecLambda2Up;
  float elecLambda2Down;
  float vdwLambda2Up;
  float vdwLambda2Down;

  float vdwShiftUp;
  float vdwShift2Up;
  float vdwShiftDown;
  float vdwShift2Down;
  bool  alchDecouple;
};

class CudaComputeNonbondedKernel {
private:

  const int deviceID;
  CudaNonbondedTables& cudaNonbondedTables;
  const bool doStreaming;

  // Exclusions
  int2 *d_exclusionsByAtom;
  unsigned int* overflowExclusions;
  size_t overflowExclusionsSize;

  int2* exclIndexMaxDiff;
  size_t exclIndexMaxDiffSize;

  // Atom indices
  int* atomIndex;
  size_t atomIndexSize;

  // VdW types
  int* vdwTypes;
  size_t vdwTypesSize;

  unsigned int* patchNumCount;
  size_t patchNumCountSize;

  int* patchReadyQueue;
  size_t patchReadyQueueSize;

  float *force_x, *force_y, *force_z, *force_w;
  size_t forceSize;
  float *forceSlow_x, *forceSlow_y, *forceSlow_z, *forceSlow_w;
  size_t forceSlowSize;
public:
  CudaComputeNonbondedKernel(int deviceID, CudaNonbondedTables& cudaNonbondedTables, bool doStreaming);
  ~CudaComputeNonbondedKernel();

  static __device__ __host__ __forceinline__ int 
  computeNumTiles(const int numAtoms, const int tilesize = WARPSIZE) { 
    return (numAtoms+tilesize-1)/tilesize; 
  }

  static __device__ __host__ __forceinline__ int 
  computeAtomPad(const int numAtoms, const int tilesize = WARPSIZE) { 
    return computeNumTiles(numAtoms, tilesize)*tilesize; 
  }

  void updateVdwTypesExcl(const int atomStorageSize, const int* h_vdwTypes,
    const int2* h_exclIndexMaxDiff, const int* h_atomIndex, cudaStream_t stream);

  void updateVdwTypesExclOnGPU(CudaTileListKernel& tlKernel,
    const int numPatches, const int atomStorageSize, const bool alchOn,
    CudaLocalRecord* localRecords,
    const int* d_vdwTypes, const int* d_id, const int* d_sortOrder, 
    const int* d_partition, cudaStream_t stream);

  void nonbondedForce(CudaTileListKernel& tlKernel,
    const int atomStorageSize, const bool atomsChanged, const bool doMinimize,
    const bool doPairlist, const bool doEnergy, const bool doVirial, 
    const bool doSlow, const bool doAlch, const bool doAlchVdwForceSwitching,
    const bool doFEP, const bool doTI, const bool doTable,
    const float3 lata, const float3 latb, const float3 latc,
    const float4* h_xyzq, const float cutoff2,
    const CudaNBConstants nbConstants,
    float4* d_forces, float4* d_forcesSlow,
    float4* h_forces, float4* h_forcesSlow, AlchData *fepFlags, 
    bool lambdaWindowUpdated,
    char *part, bool CUDASOAintegratorOn,  bool useDeviceMigration,
    cudaStream_t stream);

  void reduceVirialEnergy(CudaTileListKernel& tlKernel,
    const int atomStorageSize, const bool doEnergy, const bool doVirial, const bool doSlow, const bool doGBIS,
    float4* d_forces, float4* d_forcesSlow,
    VirialEnergy* d_virialEnergy, cudaStream_t stream);

  void getVirialEnergy(VirialEnergy* h_virialEnergy, cudaStream_t stream);

  void bindExclusions(int numExclusions, unsigned int* exclusion_bits);

  int* getPatchReadyQueue();

  void reallocate_forceSOA(int atomStorageSize); 

  void setExclusionsByAtom(int2* h_data, const int num_atoms);
};

#endif // NAMD_CUDA
#endif // CUDACOMPUTENONBONDEDKERNEL_H
