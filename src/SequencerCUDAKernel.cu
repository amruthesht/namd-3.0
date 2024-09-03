
#ifdef NAMD_CUDA

#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include <namd_cub/cub.cuh>
#endif

#endif //NAMD_CUDA

#ifdef NAMD_HIP //NAMD_HIP
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#define cub hipcub
#endif

#include "HipDefines.h"
#include "SequencerCUDAKernel.h"
#include "MShakeKernel.h"

#ifdef NODEGROUP_FORCE_REGISTER

#define OVERALLOC 2.0f

void NAMD_die(const char *);


__constant__ double c_pairlistTrigger;
__constant__ double c_pairlistGrow;
__constant__ double c_pairlistShrink;

__device__ void reset_atomic_counter(unsigned int *counter) {
  counter[0] = 0;
  __threadfence();
}


template <int T_MAX_FORCE_NUMBER, int T_DOGLOBAL>
__global__ void accumulateForceToSOAKernelMGPU(
    const int                          numPatches,
    CudaLocalRecord*                   localRecords,
    const double * __restrict          f_bond,
    const double * __restrict          f_bond_nbond,
    const double * __restrict          f_bond_slow,
    int                                forceStride,
    const float4 * __restrict          f_nbond,
    const float4 * __restrict          f_nbond_slow,
    const CudaForce* __restrict        f_slow,
    double * __restrict                d_f_global_x,
    double * __restrict                d_f_global_y,
    double * __restrict                d_f_global_z,
    double * __restrict                d_f_normal_x,
    double * __restrict                d_f_normal_y,
    double * __restrict                d_f_normal_z,
    double * __restrict                d_f_nbond_x,
    double * __restrict                d_f_nbond_y,
    double * __restrict                d_f_nbond_z,
    double * __restrict                d_f_slow_x,
    double * __restrict                d_f_slow_y,
    double * __restrict                d_f_slow_z,
    const int * __restrict             patchUnsortOrder,
    const Lattice                      lattice,
    unsigned int** __restrict          deviceQueues,
    unsigned int*  __restrict          queueCounters, 
    unsigned int*  __restrict          tbcatomic
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int offset = s_record.bufferOffset;
    const int offsetNB = s_record.bufferOffsetNBPad;

    double f_slow_x, f_slow_y, f_slow_z;
    double f_nbond_x, f_nbond_y, f_nbond_z;
    double f_normal_x, f_normal_y, f_normal_z;

    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
      if (T_MAX_FORCE_NUMBER >= 1) {
        int unorder = patchUnsortOrder[offset + i];
        // gather from sorted nonbonded force array
        float4 fnbond = f_nbond[offsetNB + unorder];
        // set (medium) nonbonded force accumulators
        f_nbond_x = (double)fnbond.x;
        f_nbond_y = (double)fnbond.y;
        f_nbond_z = (double)fnbond.z;
        if (T_MAX_FORCE_NUMBER == 2) {
          // gather from sorted nonbonded slow force array
          float4 fnbslow = f_nbond_slow[offsetNB + unorder];
          // accumulate slow force contributions from nonbonded calculation
          f_slow_x = (double)fnbslow.x;
          f_slow_y = (double)fnbslow.y;
          f_slow_z = (double)fnbslow.z;
        }
      }
      // gather from strided bond force array
      // set (fast) normal force accumulators
      if(T_DOGLOBAL) {
	 // Global forces is stored in d_f_global, add to bonded force
	f_normal_x = d_f_global_x[offset + i] + f_bond[offset + i];
        f_normal_y = d_f_global_y[offset + i] + f_bond[offset + i + forceStride];
        f_normal_z = d_f_global_z[offset + i] + f_bond[offset + i + 2*forceStride];
      }
      else{
	f_normal_x = f_bond[offset + i];
	f_normal_y = f_bond[offset + i +   forceStride];
	f_normal_z = f_bond[offset + i + 2*forceStride];
      }
      if (T_MAX_FORCE_NUMBER >= 1) {
        // gather from strided bond nonbonded force array
        // accumulate (medium) nonbonded force accumulators
        f_nbond_x += f_bond_nbond[offset + i];
        f_nbond_y += f_bond_nbond[offset + i +   forceStride];
        f_nbond_z += f_bond_nbond[offset + i + 2*forceStride];
        if (T_MAX_FORCE_NUMBER == 2) {
          // gather from strided bond slow force array
          // accumulate slow force accumulators
          f_slow_x += f_bond_slow[offset + i];
          f_slow_y += f_bond_slow[offset + i +   forceStride];
          f_slow_z += f_bond_slow[offset + i + 2*forceStride];
        }
      }
      // set normal, nonbonded, and slow SOA force buffers
      d_f_normal_x[offset + i] = f_normal_x;
      d_f_normal_y[offset + i] = f_normal_y;
      d_f_normal_z[offset + i] = f_normal_z;
      if (T_MAX_FORCE_NUMBER >= 1) {
        d_f_nbond_x[offset + i] = f_nbond_x;
        d_f_nbond_y[offset + i] = f_nbond_y;
        d_f_nbond_z[offset + i] = f_nbond_z;
        if (T_MAX_FORCE_NUMBER == 2){
          d_f_slow_x[offset + i] = f_slow_x;
          d_f_slow_y[offset + i] = f_slow_y;
          d_f_slow_z[offset + i] = f_slow_z;
        }
      }
    }
    __syncthreads();
  }
}

// DMC This was in the accumulateForceToSOAKernelMGPU (commented out)
#if 0
  __syncthreads(); 
  if(threadIdx.x == 0){
    // Need another value here
    unsigned int value = atomicInc(&tbcatomic[4], gridDim.x);
    isLastBlockDone = (value == (gridDim.x - 1));
  }
  __syncthreads();
  if(isLastBlockDone){
    // thread0 flags everything
    if(threadIdx.x == 0){
      for(int i = 0; i < nDev; i++ ){
        if (i == devID) continue; 
        unsigned int value = atomicInc(& (queueCounters[i]), nDev);
        printf(" Device[%d] queue[%d] pos[%d]\n", devID, i, value);
        // deviceQueues[i][value] = devID; // flags in other device's queue that we're ready for processing
        deviceQueues[i][value] = devID; // flags in other device's queue that we're ready for processing
        __threadfence_system();
      }
      // sets tcbCounter back to zero
      tbcatomic[4] = 0;
      printf(" GPU[%d] finished accumulating SOA buffers\n", devID);
    }
  }
#endif

template <int MAX_FORCE_NUMBER, int T_DOGLOBAL>
__global__ void accumulateForceToSOAKernel(
    const int                          numPatches,
    CudaLocalRecord*                   localRecords,
    const double * __restrict          f_bond,
    const double * __restrict          f_bond_nbond,
    const double * __restrict          f_bond_slow,
    int                                forceStride,
    const float4 * __restrict          f_nbond,
    const float4 * __restrict          f_nbond_slow,
    const CudaForce* __restrict        f_slow,
    double * __restrict                d_f_global_x,
    double * __restrict                d_f_global_y,
    double * __restrict                d_f_global_z,
    double * __restrict                d_f_normal_x,
    double * __restrict                d_f_normal_y,
    double * __restrict                d_f_normal_z,
    double * __restrict                d_f_nbond_x,
    double * __restrict                d_f_nbond_y,
    double * __restrict                d_f_nbond_z,
    double * __restrict                d_f_slow_x,
    double * __restrict                d_f_slow_y,
    double * __restrict                d_f_slow_z,
    const int * __restrict             patchUnsortOrder,
    const Lattice                      lattice
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int offset = s_record.bufferOffset;
    const int offsetNB = s_record.bufferOffsetNBPad;

    double f_slow_x, f_slow_y, f_slow_z;
    double f_nbond_x, f_nbond_y, f_nbond_z;
    double f_normal_x, f_normal_y, f_normal_z;

    for (int i = threadIdx.x;  i < numAtoms;  i += blockDim.x) {
      if (MAX_FORCE_NUMBER == 2) {
        CudaForce f  = f_slow[offset + i];
        double3 f_scaled = lattice.scale_force(Vector((double)f.x, (double)f.y, (double)f.z));
        // set slow force accumulators
        f_slow_x = f_scaled.x;
        f_slow_y = f_scaled.y;
        f_slow_z = f_scaled.z;
      }
      if (MAX_FORCE_NUMBER >= 1) {
        int unorder = patchUnsortOrder[offset + i];
        // gather from sorted nonbonded force array
        float4 fnbond = f_nbond[offsetNB + unorder];
        // set (medium) nonbonded force accumulators
        f_nbond_x = (double)fnbond.x;
        f_nbond_y = (double)fnbond.y;
        f_nbond_z = (double)fnbond.z;
        if (MAX_FORCE_NUMBER == 2) {
          // gather from sorted nonbonded slow force array
          float4 fnbslow = f_nbond_slow[offsetNB + unorder];
          // accumulate slow force contributions from nonbonded calculation
          f_slow_x += (double)fnbslow.x;
          f_slow_y += (double)fnbslow.y;
          f_slow_z += (double)fnbslow.z;
        }
      }
      // gather from strided bond force array
      // set (fast) normal force accumulators
      if(T_DOGLOBAL) {
	 // Global forces is stored in d_f_global, add to bonded force
	f_normal_x = d_f_global_x[offset + i] + f_bond[offset + i];
        f_normal_y = d_f_global_y[offset + i] + f_bond[offset + i + forceStride];
        f_normal_z = d_f_global_z[offset + i] + f_bond[offset + i + 2*forceStride];
      }
      else
      {
	f_normal_x = f_bond[offset + i];
	f_normal_y = f_bond[offset + i +   forceStride];
	f_normal_z = f_bond[offset + i + 2*forceStride];
      }

      if (MAX_FORCE_NUMBER >= 1) {
        // gather from strided bond nonbonded force array
        // accumulate (medium) nonbonded force accumulators
        f_nbond_x += f_bond_nbond[offset + i];
        f_nbond_y += f_bond_nbond[offset + i +   forceStride];
        f_nbond_z += f_bond_nbond[offset + i + 2*forceStride];
        if (MAX_FORCE_NUMBER == 2) {
          // gather from strided bond slow force array
          // accumulate slow force accumulators
          f_slow_x += f_bond_slow[offset + i];
          f_slow_y += f_bond_slow[offset + i +   forceStride];
          f_slow_z += f_bond_slow[offset + i + 2*forceStride];
        }
      }
      // set normal, nonbonded, and slow SOA force buffers
      d_f_normal_x[offset + i] = f_normal_x;
      d_f_normal_y[offset + i] = f_normal_y;
      d_f_normal_z[offset + i] = f_normal_z;
      if (MAX_FORCE_NUMBER >= 1) {
        d_f_nbond_x[offset + i] = f_nbond_x;
        d_f_nbond_y[offset + i] = f_nbond_y;
        d_f_nbond_z[offset + i] = f_nbond_z;
        if (MAX_FORCE_NUMBER == 2) {
          d_f_slow_x[offset + i] = f_slow_x;
          d_f_slow_y[offset + i] = f_slow_y;
          d_f_slow_z[offset + i] = f_slow_z;
        }
      }
    }
    __syncthreads();
  }
}

__global__ void accumulatePMEForces(
  const int numAtoms, 
  const CudaForce* f_slow, 
  double*   d_f_slow_x, 
  double*   d_f_slow_y, 
  double*   d_f_slow_z, 
  const int* patchOffsets, 
  const Lattice lat
){
  int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  double f_slow_x, f_slow_y, f_slow_z;

  if(tid < numAtoms){
    CudaForce f  = f_slow[tid];

    double3 f_scaled = lat.scale_force(Vector((double)f.x, (double)f.y, (double)f.z));
    f_slow_x = f_scaled.x;
    f_slow_y = f_scaled.y;
    f_slow_z = f_scaled.z;

    d_f_slow_x[tid] += f_slow_x;
    d_f_slow_y[tid] += f_slow_y;
    d_f_slow_z[tid] += f_slow_z;
  }
}

void SequencerCUDAKernel::accumulateForceToSOA(
  const int 		  doGlobal,			       
  const int               maxForceNumber,
  const int               numPatches,
  const int               nDevices,
  CudaLocalRecord*        localRecords,
  const double*           f_bond,
  const double*           f_bond_nbond,
  const double*           f_bond_slow,
  int                     forceStride,
  const float4*           f_nbond,
  const float4*           f_nbond_slow,
  const CudaForce*        f_slow,
  double*                 d_f_global_x,
  double*                 d_f_global_y,
  double*                 d_f_global_z,
  double*                 d_f_normal_x,
  double*                 d_f_normal_y,
  double*                 d_f_normal_z,
  double*                 d_f_nbond_x,
  double*                 d_f_nbond_y,
  double*                 d_f_nbond_z,
  double*                 d_f_slow_x,
  double*                 d_f_slow_y,
  double*                 d_f_slow_z,
  const int*              patchUnsortOrder,
  const Lattice           lattice,
  unsigned int**          deviceQueues,
  unsigned int*           queueCounters,
  unsigned int*           tbcatomic, 
  cudaStream_t            stream
) {
  // ASSERT( 0 <= maxForceNumber && maxForceNumber <= 2 );
  if(doGlobal) {
    if(nDevices > 1){
      switch (maxForceNumber) {
      case 0:
        accumulateForceToSOAKernelMGPU<0,1><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
          numPatches, localRecords,
          f_bond, f_bond_nbond, f_bond_slow, forceStride, 
          f_nbond, f_nbond_slow, f_slow,
       d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
          d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
          d_f_slow_x, d_f_slow_y, d_f_slow_z, 
          patchUnsortOrder, lattice, 
          deviceQueues, queueCounters, tbcatomic
        );
        break;
      case 1:
        accumulateForceToSOAKernelMGPU<1,1><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
          numPatches, localRecords,
          f_bond, f_bond_nbond, f_bond_slow, forceStride, 
          f_nbond, f_nbond_slow, f_slow,
       d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
          d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
          d_f_slow_x, d_f_slow_y, d_f_slow_z, 
          patchUnsortOrder, lattice, 
          deviceQueues, queueCounters, tbcatomic
        );
        break;
      case 2:
        accumulateForceToSOAKernelMGPU<2,1><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
          numPatches, localRecords,
          f_bond, f_bond_nbond, f_bond_slow, forceStride, 
          f_nbond, f_nbond_slow, f_slow,
       d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
          d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
          d_f_slow_x, d_f_slow_y, d_f_slow_z, 
          patchUnsortOrder, lattice, 
          deviceQueues, queueCounters, tbcatomic
        );
        break;
    }
    }else {
      switch (maxForceNumber) {
      case 0:
        accumulateForceToSOAKernel<0,1><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
            numPatches, localRecords,
            f_bond, f_bond_nbond, f_bond_slow, forceStride, 
            f_nbond, f_nbond_slow, f_slow,
         d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
            d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
            d_f_slow_x, d_f_slow_y, d_f_slow_z, 
            patchUnsortOrder, lattice
        );
        break;
      case 1:
        accumulateForceToSOAKernel<1,1><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
          numPatches, localRecords,
          f_bond, f_bond_nbond, f_bond_slow, forceStride, 
          f_nbond, f_nbond_slow, f_slow,
       d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
          d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
          d_f_slow_x, d_f_slow_y, d_f_slow_z, 
          patchUnsortOrder, lattice
        );
        break;
      case 2:
        accumulateForceToSOAKernel<2,1><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
          numPatches, localRecords,
          f_bond, f_bond_nbond, f_bond_slow, forceStride, 
          f_nbond, f_nbond_slow, f_slow,
       d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
          d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
          d_f_slow_x, d_f_slow_y, d_f_slow_z, 
          patchUnsortOrder, lattice
        );
        break;
      }
    }
  }// not doGlobal
  else{
    if(nDevices > 1){
      switch (maxForceNumber) {
      case 0:
        accumulateForceToSOAKernelMGPU<0,0><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
          numPatches, localRecords,
          f_bond, f_bond_nbond, f_bond_slow, forceStride, 
          f_nbond, f_nbond_slow, f_slow,
       d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
          d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
          d_f_slow_x, d_f_slow_y, d_f_slow_z, 
          patchUnsortOrder, lattice, 
          deviceQueues, queueCounters, tbcatomic
        );
        break;
      case 1:
        accumulateForceToSOAKernelMGPU<1,0><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
          numPatches, localRecords,
          f_bond, f_bond_nbond, f_bond_slow, forceStride, 
          f_nbond, f_nbond_slow, f_slow,
       d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
          d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
          d_f_slow_x, d_f_slow_y, d_f_slow_z, 
          patchUnsortOrder, lattice, 
          deviceQueues, queueCounters, tbcatomic
        );
        break;
      case 2:
        accumulateForceToSOAKernelMGPU<2,0><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
          numPatches, localRecords,
          f_bond, f_bond_nbond, f_bond_slow, forceStride, 
          f_nbond, f_nbond_slow, f_slow,
       d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
          d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
          d_f_slow_x, d_f_slow_y, d_f_slow_z, 
          patchUnsortOrder, lattice, 
          deviceQueues, queueCounters, tbcatomic
        );
        break;
      }
    }
    else {
      switch (maxForceNumber) {
      case 0:
        accumulateForceToSOAKernel<0,0><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
            numPatches, localRecords,
            f_bond, f_bond_nbond, f_bond_slow, forceStride, 
            f_nbond, f_nbond_slow, f_slow,
         d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
            d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
            d_f_slow_x, d_f_slow_y, d_f_slow_z, 
            patchUnsortOrder, lattice
        );
        break;
      case 1:
        accumulateForceToSOAKernel<1,0><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
          numPatches, localRecords,
          f_bond, f_bond_nbond, f_bond_slow, forceStride, 
          f_nbond, f_nbond_slow, f_slow,
       d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
          d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
          d_f_slow_x, d_f_slow_y, d_f_slow_z, 
          patchUnsortOrder, lattice
        );
        break;
      case 2:
        accumulateForceToSOAKernel<2,0><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
          numPatches, localRecords,
          f_bond, f_bond_nbond, f_bond_slow, forceStride, 
          f_nbond, f_nbond_slow, f_slow,
       d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
          d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
          d_f_slow_x, d_f_slow_y, d_f_slow_z, 
          patchUnsortOrder, lattice
        );
        break;
      }
    }
  }
}

template <int T_MAX_FORCE_NUMBER>
__global__ void mergeForcesFromPeersKernel(
  const int                   numPatches,
  const int                   devID,        // GPU ID 
  CudaLocalRecord*            localRecords,
  CudaPeerRecord*             peerRecords,
  const Lattice               lat,  
  double** __restrict         f_normal_x,      
  double** __restrict         f_normal_y, 
  double** __restrict         f_normal_z, 
  double** __restrict         f_nbond_x, 
  double** __restrict         f_nbond_y, 
  double** __restrict         f_nbond_z, 
  double** __restrict         f_slow_x, 
  double** __restrict         f_slow_y, 
  double** __restrict         f_slow_z,
  const CudaForce* __restrict pmeForces
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();
    
    const int numAtoms = s_record.numAtoms;
    const int dstOffset = s_record.bufferOffset;
    const int numPeerRecords = s_record.numPeerRecord;
    const int peerRecordStartIndexs = s_record.peerRecordStartIndex;

    const int inlinePeers = min(numPeerRecords, CudaLocalRecord::num_inline_peer);

    for (int peer = 0; peer < inlinePeers; peer++) {
      const int srcDevice = s_record.inline_peers[peer].deviceIndex;
      const int srcOffset = s_record.inline_peers[peer].bufferOffset;

      for (int i = threadIdx.x; i < numAtoms; i += blockDim.x){
        double  f_x = f_normal_x[srcDevice][srcOffset + i];
        double  f_y = f_normal_y[srcDevice][srcOffset + i];
        double  f_z = f_normal_z[srcDevice][srcOffset + i];

        f_normal_x[devID][dstOffset + i] += f_x;
        f_normal_y[devID][dstOffset + i] += f_y;
        f_normal_z[devID][dstOffset + i] += f_z;

        if(T_MAX_FORCE_NUMBER >= 1){ 
          // We don't need the nonbonded offset here since
          // this isn't the actual nonbonded buffer
          f_x = f_nbond_x[srcDevice][srcOffset + i];
          f_y = f_nbond_y[srcDevice][srcOffset + i];
          f_z = f_nbond_z[srcDevice][srcOffset + i]; 

          f_nbond_x[devID][dstOffset + i] += f_x;
          f_nbond_y[devID][dstOffset + i] += f_y;
          f_nbond_z[devID][dstOffset + i] += f_z;

          if(T_MAX_FORCE_NUMBER == 2){
            f_x = f_slow_x[srcDevice][srcOffset + i];
            f_y = f_slow_y[srcDevice][srcOffset + i];
            f_z = f_slow_z[srcDevice][srcOffset + i];

            f_slow_x[devID][dstOffset + i] += f_x;
            f_slow_y[devID][dstOffset + i] += f_y;
            f_slow_z[devID][dstOffset + i] += f_z;
          }
        }
      }
    }

    for (int peer = inlinePeers; peer < numPeerRecords; peer++) {
      const int srcDevice = peerRecords[peerRecordStartIndexs + peer].deviceIndex;
      const int srcOffset = peerRecords[peerRecordStartIndexs + peer].bufferOffset;

      for (int i = threadIdx.x; i < numAtoms; i += blockDim.x){
        double  f_x = f_normal_x[srcDevice][srcOffset + i];
        double  f_y = f_normal_y[srcDevice][srcOffset + i];
        double  f_z = f_normal_z[srcDevice][srcOffset + i];

        f_normal_x[devID][dstOffset + i] += f_x;
        f_normal_y[devID][dstOffset + i] += f_y;
        f_normal_z[devID][dstOffset + i] += f_z;

        if(T_MAX_FORCE_NUMBER >= 1){ 
          // We don't need the nonbonded offset here since
          // this isn't the actual nonbonded buffer
          f_x = f_nbond_x[srcDevice][srcOffset + i];
          f_y = f_nbond_y[srcDevice][srcOffset + i];
          f_z = f_nbond_z[srcDevice][srcOffset + i]; 

          f_nbond_x[devID][dstOffset + i] += f_x;
          f_nbond_y[devID][dstOffset + i] += f_y;
          f_nbond_z[devID][dstOffset + i] += f_z;

          if(T_MAX_FORCE_NUMBER == 2){
            f_x = f_slow_x[srcDevice][srcOffset + i];
            f_y = f_slow_y[srcDevice][srcOffset + i];
            f_z = f_slow_z[srcDevice][srcOffset + i];

            f_slow_x[devID][dstOffset + i] += f_x;
            f_slow_y[devID][dstOffset + i] += f_y;
            f_slow_z[devID][dstOffset + i] += f_z;
          }
        }
      }
    }

    // Merge PME forces here instead of in the fetch forcing kernel
    if(T_MAX_FORCE_NUMBER == 2){
      for(int i = threadIdx.x; i < numAtoms; i += blockDim.x){
        CudaForce f = pmeForces[dstOffset + i];

        double3 f_scaled = lat.scale_force(
          Vector((double)f.x, (double)f.y, (double)f.z));
        f_slow_x[devID][dstOffset + i] += f_scaled.x;
        f_slow_y[devID][dstOffset + i] += f_scaled.y;
        f_slow_z[devID][dstOffset + i] += f_scaled.z;
      }
    }
    __syncthreads();
  }
}

void SequencerCUDAKernel::mergeForcesFromPeers(
  const int              devID, 
  const int              maxForceNumber, 
  const Lattice          lat, 
  const int              numPatchesHomeAndProxy, 
  const int              numPatchesHome, 
  double**               f_normal_x,
  double**               f_normal_y,
  double**               f_normal_z,
  double**               f_nbond_x,
  double**               f_nbond_y,
  double**               f_nbond_z,
  double**               f_slow_x,
  double**               f_slow_y,
  double**               f_slow_z,
  const CudaForce*       pmeForces, 
  CudaLocalRecord*       localRecords,
  CudaPeerRecord*        peerRecords,
  std::vector<int>&      atomCounts,
  cudaStream_t           stream
){
  // atom-based kernel here
  //int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  int grid = (numPatchesHome);
  int blocks = PATCH_BLOCKS;
  // launch a single-threaded grid for debugging

  int offset = 0;
  for (int i = 0; i < devID; i++) {
    offset += atomCounts[i];
  }

  switch (maxForceNumber) {
    case 0:
      mergeForcesFromPeersKernel<0><<<grid, blocks, 0, stream>>>(
        numPatchesHome, devID, localRecords, peerRecords,
        lat,
        f_normal_x, f_normal_y, f_normal_z,
        f_nbond_x, f_nbond_y, f_nbond_z, 
        f_slow_x, f_slow_y, f_slow_z, pmeForces + offset
      );
      break;
    case 1:
      mergeForcesFromPeersKernel<1><<<grid, blocks, 0, stream>>>(
        numPatchesHome, devID, localRecords, peerRecords,
        lat,
        f_normal_x, f_normal_y, f_normal_z,
        f_nbond_x, f_nbond_y, f_nbond_z, 
        f_slow_x, f_slow_y, f_slow_z, pmeForces + offset
      );
      break;
    case 2:
      mergeForcesFromPeersKernel<2><<<grid, blocks, 0, stream>>>(
        numPatchesHome, devID, localRecords, peerRecords,
        lat,
        f_normal_x, f_normal_y, f_normal_z,
        f_nbond_x, f_nbond_y, f_nbond_z, 
        f_slow_x, f_slow_y, f_slow_z, pmeForces + offset
      );
      break;
  }
}

// JM NOTE: This is a fused version of accumulateForceToSOA + addForceToMomentum.
//          addForceToMomentum barely has any math and is memory-bandwith bound, so
//          fusing these kernels allows us to keep the forces in the registers and
//          reusing its values for velocities
template <int MAX_FORCE_NUMBER, int T_DOGLOBAL>
__global__ void accumulateForceKick(
    const int                          numPatches,
    CudaLocalRecord*                   localRecords,
    const double * __restrict          f_bond,
    const double * __restrict          f_bond_nbond,
    const double * __restrict          f_bond_slow,
    int                                forceStride,
    const float4 * __restrict          f_nbond,
    const float4 * __restrict          f_nbond_slow,
    const CudaForce* __restrict        f_slow,
    double * __restrict                d_f_global_x,
    double * __restrict                d_f_global_y,
    double * __restrict                d_f_global_z,
    double * __restrict                d_f_normal_x,
    double * __restrict                d_f_normal_y,
    double * __restrict                d_f_normal_z,
    double * __restrict                d_f_nbond_x,
    double * __restrict                d_f_nbond_y,
    double * __restrict                d_f_nbond_z,
    double * __restrict                d_f_slow_x,
    double * __restrict                d_f_slow_y,
    double * __restrict                d_f_slow_z,
    const int * __restrict             patchUnsortOrder,
    const Lattice                      lattice,
    double *d_vel_x,
    double *d_vel_y,
    double *d_vel_z, 
    const double * __restrict recipMass, 
    const double dt_normal,
    const double dt_nbond, 
    const double dt_slow,
    const double scaling
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int offset = s_record.bufferOffset;
    const int offsetNB = s_record.bufferOffsetNBPad;

    double f_slow_x, f_slow_y, f_slow_z;
    double f_nbond_x, f_nbond_y, f_nbond_z;
    double f_normal_x, f_normal_y, f_normal_z;

    for (int i = threadIdx.x;  i < numAtoms;  i += blockDim.x) {
      f_slow_x =   0.0;
      f_slow_y =   0.0;
      f_slow_z =   0.0;
      f_nbond_x =  0.0;
      f_nbond_y =  0.0;
      f_nbond_z =  0.0;
      f_normal_x = 0.0;
      f_normal_y = 0.0;
      f_normal_z = 0.0;

      if (MAX_FORCE_NUMBER == 2) {
        CudaForce f  = f_slow[offset + i];

        double3 f_scaled = lattice.scale_force(
          Vector((double)f.x, (double)f.y, (double)f.z));

        // set slow force accumulators
        f_slow_x = f_scaled.x;
        f_slow_y = f_scaled.y;
        f_slow_z = f_scaled.z;
      }
      if (MAX_FORCE_NUMBER >= 1) {
        int unorder = patchUnsortOrder[offset + i];
        // gather from sorted nonbonded force array
        float4 fnbond = f_nbond[offsetNB + unorder];
        // set (medium) nonbonded force accumulators
        f_nbond_x = (double)fnbond.x;
        f_nbond_y = (double)fnbond.y;
        f_nbond_z = (double)fnbond.z;
        if (MAX_FORCE_NUMBER == 2) {
          // gather from sorted nonbonded slow force array
          float4 fnbslow = f_nbond_slow[offsetNB + unorder];
          // accumulate slow force contributions from nonbonded calculation
          f_slow_x += (double)fnbslow.x;
          f_slow_y += (double)fnbslow.y;
          f_slow_z += (double)fnbslow.z;
        }
      }
      // gather from strided bond force array
      // set (fast) normal force accumulators
      if(T_DOGLOBAL) {
	 // Global forces is stored in d_f_global, add to bonded force
	f_normal_x = d_f_global_x[offset + i] + f_bond[offset + i];
        f_normal_y = d_f_global_y[offset + i] + f_bond[offset + i + forceStride];
        f_normal_z = d_f_global_z[offset + i] + f_bond[offset + i + 2*forceStride];
      }
      else
      {

	f_normal_x = f_bond[offset + i];
	f_normal_y = f_bond[offset + i +   forceStride];
	f_normal_z = f_bond[offset + i + 2*forceStride];
      }

      if (MAX_FORCE_NUMBER >= 1) {
        // gather from strided bond nonbonded force array
        // accumulate (medium) nonbonded force accumulators
        f_nbond_x += f_bond_nbond[offset + i];
        f_nbond_y += f_bond_nbond[offset + i +   forceStride];
        f_nbond_z += f_bond_nbond[offset + i + 2*forceStride];
        if (MAX_FORCE_NUMBER == 2) {
          // gather from strided bond slow force array
          // accumulate slow force accumulators
          f_slow_x += f_bond_slow[offset + i];
          f_slow_y += f_bond_slow[offset + i +   forceStride];
          f_slow_z += f_bond_slow[offset + i + 2*forceStride];
        }
      }
      // set normal, nonbonded, and slow SOA force buffers
      d_f_normal_x[offset + i] = f_normal_x;
      d_f_normal_y[offset + i] = f_normal_y;
      d_f_normal_z[offset + i] = f_normal_z;
      if (MAX_FORCE_NUMBER >= 1) {
        d_f_nbond_x[offset + i] = f_nbond_x;
        d_f_nbond_y[offset + i] = f_nbond_y;
        d_f_nbond_z[offset + i] = f_nbond_z;
        if (MAX_FORCE_NUMBER == 2) {
          d_f_slow_x[offset + i] = f_slow_x;
          d_f_slow_y[offset + i] = f_slow_y;
          d_f_slow_z[offset + i] = f_slow_z;
        }
      }

      /* Velocity updates */
      double rp = recipMass[offset + i];
      double vx, vy, vz;
      vx = ((f_slow_x * dt_slow) + (f_nbond_x * dt_nbond) + (f_normal_x * dt_normal)) * scaling * rp;
      vy = ((f_slow_y * dt_slow) + (f_nbond_y * dt_nbond) + (f_normal_y * dt_normal)) * scaling * rp;
      vz = ((f_slow_z * dt_slow) + (f_nbond_z * dt_nbond) + (f_normal_z * dt_normal)) * scaling * rp;

      d_vel_x[offset + i] += vx;
      d_vel_y[offset + i] += vy;
      d_vel_z[offset + i] += vz;

    }
    __syncthreads();
  }
}


#if 0
template <int MAX_FORCE_NUMBER>
__global__ void accumulateForceKick(
    const double  * __restrict f_bond,
    const double  * __restrict f_bond_nbond,
    const double  * __restrict f_bond_slow,
    int forceStride,
    const PatchRecord * __restrict bond_pr,
    const float4 * __restrict f_nbond,
    const float4 * __restrict f_nbond_slow,
    const CudaPatchRecord * __restrict nbond_pr,
    const CudaForce * __restrict f_slow,
    double *d_f_normal_x,
    double *d_f_normal_y,
    double *d_f_normal_z,
    double *d_f_nbond_x,
    double *d_f_nbond_y,
    double *d_f_nbond_z,
    double *d_f_slow_x,
    double *d_f_slow_y,
    double *d_f_slow_z,
    double *d_vel_x,
    double *d_vel_y,
    double *d_vel_z,
    const double * __restrict recipMass,
    const double dt_normal,
    const double dt_nbond,
    const double dt_slow,
    const double scaling,
    const int * __restrict patchIDs,
    const int * __restrict patchOffsets,
    const int * __restrict patchUnsortOrder,
    const CudaLattice lat)
{
  int tid = threadIdx.x;
  int stride = blockDim.x;

  __shared__ int sh_patchID;
  __shared__ int sh_patchOffset;
  // number of atoms per patch should be same no matter
  // which data structure we access it through
  __shared__ int sh_natoms;
  __shared__ int sh_bondForceOffset;
  __shared__ int sh_nbondForceOffset;

  // int patchID;
  int patchOffset;
  // number of atoms per patch should be same no matter
  // which data structure we access it through
  int natoms;
  int bondForceOffset;
  int nbondForceOffset;

  if(threadIdx.x == 0){
    sh_patchID = patchIDs[blockIdx.x];
    sh_patchOffset = patchOffsets[blockIdx.x];
    // number of atoms per patch should be same no matter
    // which data structure we access it through
    sh_natoms = bond_pr[sh_patchID].numAtoms;
    sh_bondForceOffset = bond_pr[sh_patchID].atomStart;
    if(MAX_FORCE_NUMBER >= 1){
      sh_nbondForceOffset = nbond_pr[sh_patchID].atomStart;
    }
  }

  __syncthreads();

  // pull stuff to registers
  // patchID = sh_patchID;
  patchOffset = sh_patchOffset;
  natoms = sh_natoms;
  bondForceOffset = sh_bondForceOffset;
  nbondForceOffset = (MAX_FORCE_NUMBER >= 1) ? sh_nbondForceOffset : 0;

  double r1x, r1y, r1z, r2x, r2y, r2z, r3x, r3y, r3z;

  if (MAX_FORCE_NUMBER == 2) {
    r1x = lat.b1.x;
    r1y = lat.b1.y;
    r1z = lat.b1.z;
    r2x = lat.b2.x;
    r2y = lat.b2.y;
    r2z = lat.b2.z;
    r3x = lat.b3.x;
    r3y = lat.b3.y;
    r3z = lat.b3.z;
  }

  double f_slow_x, f_slow_y, f_slow_z;
  double f_nbond_x, f_nbond_y, f_nbond_z;
  double f_normal_x, f_normal_y, f_normal_z;
  double vx, vy, vz;

  for (int i = tid;  i < natoms;  i += stride) {
    f_slow_x =   0.0;
    f_slow_y =   0.0;
    f_slow_z =   0.0;
    f_nbond_x =  0.0;
    f_nbond_y =  0.0;
    f_nbond_z =  0.0;
    f_normal_x = 0.0;
    f_normal_y = 0.0;
    f_normal_z = 0.0;

    if(T_DOGLOBAL) {
      // Global forces is stored in d_f_normal, just need to add bonded force
      f_normal_x = d_f_normal_x[patchOffset + i];
      f_normal_y = d_f_normal_y[patchOffset + i];
      f_normal_z = d_f_normal_z[patchOffset + i];
    } 

    if (MAX_FORCE_NUMBER == 2) {
      CudaForce f  = f_slow[patchOffset + i];
      double fx = (double) f.x;
      double fy = (double) f.y;
      double fz = (double) f.z;
      // set slow force accumulators
      f_slow_x = (fx*r1x + fy*r2x + fz*r3x);
      f_slow_y = (fx*r1y + fy*r2y + fz*r3y);
      f_slow_z = (fx*r1z + fy*r2z + fz*r3z);
    }
    if (MAX_FORCE_NUMBER >= 1) {
      int unorder = patchUnsortOrder[patchOffset + i];
      // gather from sorted nonbonded force array
      float4 fnbond = f_nbond[nbondForceOffset + unorder];
      // set (medium) nonbonded force accumulators
      f_nbond_x = (double)fnbond.x;
      f_nbond_y = (double)fnbond.y;
      f_nbond_z = (double)fnbond.z;
      if (MAX_FORCE_NUMBER == 2) {
        // gather from sorted nonbonded slow force array
        float4 fnbslow = f_nbond_slow[nbondForceOffset + unorder];
        // accumulate slow force contributions from nonbonded calculation
        f_slow_x += (double)fnbslow.x;
        f_slow_y += (double)fnbslow.y;
        f_slow_z += (double)fnbslow.z;
      }
    }
    // gather from strided bond force array
    // set (fast) normal force accumulators
    f_normal_x += f_bond[bondForceOffset + i];
    f_normal_y += f_bond[bondForceOffset + i +   forceStride];
    f_normal_z += f_bond[bondForceOffset + i + 2*forceStride];
    if (MAX_FORCE_NUMBER >= 1) {
      // gather from strided bond nonbonded force array
      // accumulate (medium) nonbonded force accumulators
      f_nbond_x += f_bond_nbond[bondForceOffset + i];
      f_nbond_y += f_bond_nbond[bondForceOffset + i +   forceStride];
      f_nbond_z += f_bond_nbond[bondForceOffset + i + 2*forceStride];
      if (MAX_FORCE_NUMBER == 2) {
        // gather from strided bond slow force array
        // accumulate slow force accumulators
        f_slow_x += f_bond_slow[bondForceOffset + i];
        f_slow_y += f_bond_slow[bondForceOffset + i +   forceStride];
        f_slow_z += f_bond_slow[bondForceOffset + i + 2*forceStride];
      }
    }
    // set normal, nonbonded, and slow SOA force buffers
    d_f_normal_x[patchOffset + i] = f_normal_x;
    d_f_normal_y[patchOffset + i] = f_normal_y;
    d_f_normal_z[patchOffset + i] = f_normal_z;

    if (MAX_FORCE_NUMBER >= 1) {
      d_f_nbond_x[patchOffset + i] = f_nbond_x;
      d_f_nbond_y[patchOffset + i] = f_nbond_y;
      d_f_nbond_z[patchOffset + i] = f_nbond_z;
      if (MAX_FORCE_NUMBER == 2) {
        d_f_slow_x[patchOffset + i] = f_slow_x;
        d_f_slow_y[patchOffset + i] = f_slow_y;
        d_f_slow_z[patchOffset + i] = f_slow_z;
      }
    }

    /* Velocity updates */
    double rp = recipMass[patchOffset + i];
    vx = ((f_slow_x * dt_slow) + (f_nbond_x * dt_nbond) + (f_normal_x * dt_normal)) * scaling * rp;
    vy = ((f_slow_y * dt_slow) + (f_nbond_y * dt_nbond) + (f_normal_y * dt_normal)) * scaling * rp;
    vz = ((f_slow_z * dt_slow) + (f_nbond_z * dt_nbond) + (f_normal_z * dt_normal)) * scaling * rp;

    d_vel_x[patchOffset + i] += vx;
    d_vel_y[patchOffset + i] += vy;
    d_vel_z[patchOffset + i] += vz;
  }
}
#endif


void SequencerCUDAKernel::accumulate_force_kick(
  const int               doGlobal,						
  int                     maxForceNumber,
  const int               numPatches,
  CudaLocalRecord*        localRecords,
  const double*           f_bond,
  const double*           f_bond_nbond,
  const double*           f_bond_slow,
  int                     forceStride,
  const float4*           f_nbond,
  const float4*           f_nbond_slow,
  const CudaForce*        f_slow,
  double*                 d_f_global_x,
  double*                 d_f_global_y,
  double*                 d_f_global_z,
  double*                 d_f_normal_x,
  double*                 d_f_normal_y,
  double*                 d_f_normal_z,
  double*                 d_f_nbond_x,
  double*                 d_f_nbond_y,
  double*                 d_f_nbond_z,
  double*                 d_f_slow_x,
  double*                 d_f_slow_y,
  double*                 d_f_slow_z,
  double*                 d_vel_x,
  double*                 d_vel_y, 
  double*                 d_vel_z,
  const double*           recipMass, 
  const double            dt_normal, 
  const double            dt_nbond,
  const double            dt_slow,
  const double            scaling,
  const int*              patchUnsortOrder,
  const Lattice           lattice,
  cudaStream_t            stream
) {
  // ASSERT( 0 <= maxForceNumber && maxForceNumber <= 2 );
  if(doGlobal) {
    switch (maxForceNumber) {
    case 0:
      accumulateForceKick<0,1><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
            numPatches, localRecords,
            f_bond, f_bond_nbond, f_bond_slow, forceStride,
            f_nbond, f_nbond_slow, f_slow,
	    d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
            d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
            d_f_slow_x, d_f_slow_y, d_f_slow_z,
            patchUnsortOrder, lattice,
            d_vel_x, d_vel_y, d_vel_z, recipMass,
            dt_normal, dt_nbond, dt_slow, scaling
        );
        break;
      case 1:
        accumulateForceKick<1,1><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
            numPatches, localRecords,
            f_bond, f_bond_nbond, f_bond_slow, forceStride,
            f_nbond, f_nbond_slow, f_slow,
	    d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
            d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
            d_f_slow_x, d_f_slow_y, d_f_slow_z,
            patchUnsortOrder, lattice,
            d_vel_x, d_vel_y, d_vel_z, recipMass,
            dt_normal, dt_nbond, dt_slow, scaling
        );
        break;
      case 2:
        accumulateForceKick<2,1><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
            numPatches, localRecords,
            f_bond, f_bond_nbond, f_bond_slow, forceStride,
            f_nbond, f_nbond_slow, f_slow,
	    d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
            d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
            d_f_slow_x, d_f_slow_y, d_f_slow_z,
            patchUnsortOrder, lattice,
            d_vel_x, d_vel_y, d_vel_z, recipMass,
            dt_normal, dt_nbond, dt_slow, scaling
        );
        break;
    }
  }
  else    {
    switch (maxForceNumber) {
    case 0:
      accumulateForceKick<0,0><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
            numPatches, localRecords,
            f_bond, f_bond_nbond, f_bond_slow, forceStride,
            f_nbond, f_nbond_slow, f_slow,
	    d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
            d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
            d_f_slow_x, d_f_slow_y, d_f_slow_z,
            patchUnsortOrder, lattice,
            d_vel_x, d_vel_y, d_vel_z, recipMass,
            dt_normal, dt_nbond, dt_slow, scaling
        );
        break;
      case 1:
        accumulateForceKick<1,0><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
            numPatches, localRecords,
            f_bond, f_bond_nbond, f_bond_slow, forceStride,
            f_nbond, f_nbond_slow, f_slow,
	    d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
            d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
            d_f_slow_x, d_f_slow_y, d_f_slow_z,
            patchUnsortOrder, lattice,
            d_vel_x, d_vel_y, d_vel_z, recipMass,
            dt_normal, dt_nbond, dt_slow, scaling
        );
        break;
      case 2:
        accumulateForceKick<2,0><<<numPatches, PATCH_BLOCKS, 0, stream>>>(
            numPatches, localRecords,
            f_bond, f_bond_nbond, f_bond_slow, forceStride,
            f_nbond, f_nbond_slow, f_slow,
	    d_f_global_x, d_f_global_y, d_f_global_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
            d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
            d_f_slow_x, d_f_slow_y, d_f_slow_z,
            patchUnsortOrder, lattice,
            d_vel_x, d_vel_y, d_vel_z, recipMass,
            dt_normal, dt_nbond, dt_slow, scaling
        );
        break;
       }
    }
}

template <bool doSlow>
__global__  void copyFNBondToSOA(float4* __restrict__ f_nbond,
  float4* __restrict__                f_nbond_slow,
  double*                              f_nbond_x,
  double*                              f_nbond_y,
  double*                              f_nbond_z,
  double*                              f_slow_x,
  double*                              f_slow_y,
  double*                              f_slow_z,
  const int* __restrict__              patchIDs,
  const int* __restrict__              patchOffsets,
  const int* __restrict__              patchUnsortOrder,
  const CudaPatchRecord*  __restrict__ nbondIndexPerPatch)
{
    int unorder;
    int stride = blockDim.x;
    int pid = patchIDs[blockIdx.x];

    // nbondIndexPerPatch is organized on a per local ID on the compute
    // It's also device-wide, which means it hopefully has more patches than
    // PatchMap but that means the ordering is different. I can't use the
    // PatchID here to index nbondIndexPerPatch I need something else.
    // Each patchID from patchMap needs to map somewhere to nbondIndexPerPatch
    // Means I can't use patchIDs anymore. I need to use a different ordering
    // straight from the computes nbondIndexPerPatch is the same for
    // both multi and single PE. What changes is force storage?
    // The only thing that needs updating is the usage of the local compute index
    // instead of using the global patchID.
    // I need the local compute index
    // what do... Patches never migrate, so hopefully this datastructure will never change.
    // how do I map????
    // I can build an index at patchMap with the ordering inside the computes.
    int globalForceOffset = nbondIndexPerPatch[pid].atomStart;
    int patchOffset = patchOffsets[blockIdx.x];
    int natom = nbondIndexPerPatch[pid].numAtoms;
    for(int i = threadIdx.x; i < natom; i+= stride){
      unorder = patchUnsortOrder[patchOffset + i];
      float4 force = f_nbond[globalForceOffset + unorder];
      //        SOA                GLOBAL
      f_nbond_x[patchOffset + i] = (double)force.x;
      f_nbond_y[patchOffset + i] = (double)force.y;
      f_nbond_z[patchOffset + i] = (double)force.z;

      if(doSlow){
        // XXX Accumulation here works because each f_slow_* buffer
        // is first initialized in copyFSlowToSOA.
        float4  f_nb_slow = f_nbond_slow[globalForceOffset + unorder];
        f_slow_x[patchOffset + i] += (double)f_nb_slow.x;
        f_slow_y[patchOffset + i] += (double)f_nb_slow.y;
        f_slow_z[patchOffset + i] += (double)f_nb_slow.z;
      }
    }
}

void SequencerCUDAKernel::copy_nbond_forces(int numPatches,
  float4*                f_nbond,
  float4*                f_nbond_slow,
  double*                f_nbond_x,
  double*                f_nbond_y,
  double*                f_nbond_z,
  double*                f_slow_x,
  double*                f_slow_y,
  double*                f_slow_z,
  const int*             patchIDs,
  const int*             patchOffsets,
  const int*             patchUnsortOrder,
  const CudaPatchRecord* pr,
  const bool             doSlow,
  cudaStream_t           stream){

    if(doSlow){
      copyFNBondToSOA<1><<< numPatches, PATCH_BLOCKS, 0, stream >>>(f_nbond, f_nbond_slow, f_nbond_x,
        f_nbond_y, f_nbond_z, f_slow_x, f_slow_y, f_slow_z,
        patchIDs, patchOffsets, patchUnsortOrder, pr);
    }else {
      copyFNBondToSOA<0><<< numPatches, PATCH_BLOCKS, 0, stream >>>(f_nbond, f_nbond_slow, f_nbond_x,
        f_nbond_y, f_nbond_z, f_slow_x, f_slow_y, f_slow_z,
        patchIDs, patchOffsets, patchUnsortOrder, pr);
    }
}

#if 0
// this function will fuse copyFNBond, copyFbond, addForceToMomentum and updateVelocities (hopefully)
// XXX TODO: missing slow forces support
template<bool T_DONBOND, bool T_DOSLOW>
__global__ void fetchForcesAndUpdateVelocitiesKernel(
        const double scaling,
        double       dt_normal,
        double       dt_nbond,
        double       dt_slow,
        /* copyFbondToSOA parameters*/
        const double* __restrict__ f_bond, // forces from bond compute
        const double* __restrict__ f_bond_nbond, // forces from bond compute
        double* __restrict__ f_bond_x,
        double* __restrict__ f_bond_y,
        double* __restrict__ f_bond_z,
        double* __restrict__ f_nbond_x,
        double* __restrict__ f_nbond_y,
        double* __restrict__ f_nbond_z,
        double* __restrict__ f_slow_x,
        double* __restrict__ f_slow_y,
        double* __restrict__ f_slow_z,
        int forceStride,
        const PatchRecord* __restrict__ b_pr,
        const int* __restrict__ patchIDs,
        const int* __restrict__ patchOffsets,
        /* copyFNbondToSOA parameters*/
        const float4* __restrict__ f_nbond, // forces from bbond compute
        const float4* __restrict__ f_nbond_slow, // forces from bond compute
        const int* __restrict__ patchSortOrder,
        const CudaPatchRecord* __restrict__ nbondIndexPerPatch,
        /* copyFSlowToSOA */
        const CudaForce* __restrict__ f_slow,
        const int* __restrict__         patchPositions,
        const int* __restrict__         pencilPatchIndex,
        const int* __restrict__         patchOffsets,
        const int* __restrict__         patchIDs,
        const int* __restrict__         slow_patchIDs,
        const CudaLattice* __restrict__ lattices,
        /* addForceToMomentum */
        const double * __restrict recipMass,
        double       * __restrict vel_x,
        double       * __restrict vel_y,
        double       * __restrict vel_z,
        /* updateRigidArrays */
        const double * __restrict pos_x,
        const double * __restrict pos_y,
        const double * __restrict pos_z,
        double *       __restrict velNew_x,
        double *       __restrict velNew_y,
        double *       __restrict velNew_z,
        double *       __restrict posNew_x,
        double *       __restrict posNew_y,
        double *       __restrict posNew_z
  ){
  int order;
  int stride = blockDim.x;
  int pid = patchIDs[blockIdx.x];

  int globalForceOffset = nbondIndexPerPatch[pid].atomStart;
  int b_forceOffset  = b_pr[patchID].atomStart;

  int patchOffset = patchOffsets[blockIdx.x];
  int natom = nbondIndexPerPatch.numAtoms;

  float4 f_b;
  float4 f_nb;
  float4 f_s;
  for(int i = 0; i < threadIdx.x; i < natom; i += stride){
    f_b  = {0};
    if(T_DONBOND) f_nb = {0};
    if(T_DOSLOW)  f_s  = {0};
    // fetch the bonded forces first
    f_b.x  = f_bond[b_forceOffset + i];
    f_b.y  = f_bond[b_forceOffset + i + forceStride];
    f_b.z  = f_bond[b_forceOffset + i + 2*forceStride];

    if(T_DONBOND){
       f_nb = f_nbond[globalForceOffset + i];
       f_nb.x += f_bond_nbond[b_forceOffset + i];
       f_nb.y += f_bond_nbond[b_forceOffset + i + forceStride];
       f_nb.z += f_bond_nbond[b_forceOffset + i + 2*forceStride];
    }
    // addForceToMomentum now
    // this striding is not good
    float vx = vel_x[patchOffset + i];
    float vy = vel_y[patchOffset + i];
    float vz = vel_z[patchOffset + i];
    float rmass = recipMass[patchOffset + i];
    switch(maxForceNumber){
      // XXX TODO: Case 2 for slow forces
      case 1:
        dt_nbond *= scaling;
        vx += f_nb.x * rmass * dt_nbond;
        vy += f_nb.y * rmass * dt_nbond;
        vz += f_nb.z * rmass * dt_nbond;
      case 0:
        dt_normal *= scaling;
        vx += f_b.x * rmass * dt_nbond;
        vy += f_b.y * rmass * dt_nbond;
        vz += f_b.z * rmass * dt_nbond;
    }
    // that's it
    // updateRigidArrays

    posNew_x[patchOffset + i] = pos_x[i] + (vx * dt);
    posNew_y[patchOffset + i] = pos_y[i] + (vy * dt);
    posNew_z[patchOffset + i] = pos_z[i] + (vz * dt);
    vel_x[patchOffset + i]    = vx;
    vel_y[patchOffset + i]    = vy;
    vel_z[patchOffset + i]    = vz;
    velNew_x[patchOffset + i] = vx;
    velNew_y[patchOffset + i] = vy;
    velNew_z[patchOffset + i] = vz;
    f_normal_x[patchOffset + i] = f_b.x;
    f_normal_y[patchOffset + i] = f_b.y;
    f_normal_z[patchOffset + i] = f_b.z;

    if(T_DONBOND){
      order = patchSortOrder[patchOffset + i];
      f_nbond_x[patchOffset + order] = f_nb.x;
      f_nbond_y[patchOffset + order] = f_nb.y;
      f_nbond_z[patchOffset + order] = f_nb.z;
    }
  }
}

void SequencerCUDAKernel::fetchForcesAndUpdateVelocities(int numPatches,
        const bool  doNbond,
        const bool  doSlow,
        const double scaling,
        double       dt_normal,
        double       dt_nbond,
        double       dt_slow,
        const int    maxForceNumber,
        /* copyFbondToSOA parameters*/
        const double* __restrict__ f_bond, // forces from bond compute
        const double* __restrict__ f_bond_nbond, // forces from bond compute
        double* f_bond_x,
        double* f_bond_y,
        double* f_bond_z,
        double* f_nbond_x,
        double* f_nbond_y,
        double* f_nbond_z,
        double* f_slow_x,
        double* f_slow_y,
        double* f_slow_z,
        int forceStride,
        const PatchRecord* b_pr,
        const int* patchIDs,
        const int* patchOffsets,
        /* copyFNbondToSOA parameters*/
        const float4* f_nbond, // forces from bbond compute
        const float4* f_nbond_slow, // forces from bond compute
        const int* patchSortOrder,
        const CudaPatchRecord* nbondIndexPerPatch,
        /* copyFSlowToSOA */
        const CudaForce* f_slow,
        const int* patchPositions,
        const int* pencilPatchIndex,
        const int* patchOffsets,
        const int* patchIDs,
        const int* slow_patchIDs,
        const CudaLattice* lattices,
        /* addForceToMomentum */
        const double *  recipMass,
        double       *  vel_x,
        double       *  vel_y,
        double       *  vel_z,
        /* updateRigidArrays */
        const double *  pos_x,
        const double *  pos_y,
        const double *  pos_z,
        double *        velNew_x,
        double *        velNew_y,
        double *        velNew_z,
        double *        posNew_x,
        double *        posNew_y,
        double *        posNew_z,
        cudaStream_t    stream){

        // reduce the amount of arguments in this function
         int blocks = 128;
         int grid = numPatches;

         // XXX TODO finish this
         if(doNbond){
           if(doSlow){
             fetchForcesAndUpdateVelocities<true, true><<<numPatches, 128, 0, stream>>>();
           }else{
             fetchForcesAndUpdateVelocities<true, false><<<numPatches, 128, 0, stream>>>();
           }
         }else{
           fetchForcesAndUpdateVelocities<false, false><<<numPatches, 128, 0, stream>>>();
         }
}
#endif

// JM: I need a function to do the pairlistCheck
template<bool T_ISPERIODIC>
__global__ void pairListMarginCheckKernel(
  const int                    numPatches,
  CudaLocalRecord*             localRecords,
  const double* __restrict     pos_x, 
  const double*     __restrict pos_y, 
  const double*     __restrict pos_z,
  const double*     __restrict pos_old_x,
  const double*     __restrict pos_old_y,
  const double*     __restrict pos_old_z,
  const double3*    __restrict awayDists, // for margin check
  const Lattice                lattice,
  const Lattice                latticeOld,
  const double3*    __restrict patchMins,
  const double3*    __restrict patchMaxes,
  const double3*    __restrict patchCenter,
  const CudaMInfo*  __restrict mInfo,
  unsigned int*     __restrict tbcatomic,
  double*                      patchMaxAtomMovement,
  double*                      h_patchMaxAtomMovement,
  double*                      patchNewTolerance,
  double*                      h_patchNewTolerance,
  const double                 minSize,
  const double                 cutoff,
  const double                 sysdima,
  const double                 sysdimb,
  const double                 sysdimc,
  unsigned int*     __restrict h_marginViolations,
  unsigned int*     __restrict h_periodicCellSmall, 
  const bool                   rescalePairlistTolerance
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int pid = s_record.patchID;

  
    // So each patch has a max_CD particular to it
    // since we have a block per patch, we get the stuff here
    // we need to check eight points for each patch. Might not pay off to do that
    // we do that once per patch I guess
    double cd, max_cd, max_pd;
    __shared__ double minx, miny, minz, maxx, maxy, maxz;
    double3 corner, corner_unscaled, corner2_unscaled;
    double3 mValue;
    double3 s;

    const int stride = blockDim.x;

    const int ind = patchIndex;
    double3 min = patchMins[ind];
    double3 max = patchMaxes[ind];
    double3 aDist = awayDists[ind];

    const int start    = s_record.bufferOffset;
    const int numAtoms = s_record.numAtoms;
    double3 center = patchCenter[ind];
  
    double3 center_cur = lattice.unscale(center);
    double3 center_old = latticeOld.unscale(center);

    center_cur.x -= center_old.x;
    center_cur.y -= center_old.y;
    center_cur.z -= center_old.z;
  
    max_cd = 0.f;
    max_pd = 0.f;
    if(threadIdx.x == 0){
      if (rescalePairlistTolerance) patchNewTolerance[ind] *= (1.0f - c_pairlistShrink);

      // doMargincheck -> checks if the periodic cell has became too small:
      
      // Those values are in registers to begin with, so the overhead here is not that large
      // if this condition is set to true, simulation will stop
      mValue.x = (0.5) * (aDist.x  - cutoff / sysdima);
      mValue.y = (0.5) * (aDist.y  - cutoff / sysdimb);
      mValue.z = (0.5) * (aDist.z  - cutoff / sysdimc);
      minx = min.x - mValue.x;
      miny = min.y - mValue.y;
      minz = min.z - mValue.z;
      maxx = max.x + mValue.x;
      maxy = max.y + mValue.y;
      maxz = max.z + mValue.z;

      for(int i = 0; i < 2; i++){
        for(int j =0 ; j< 2; j++){
          for(int k = 0; k < 2; k++){
              corner.x = (i ? min.x : max.x);
              corner.y = (j ? min.y : max.y);
              corner.z = (k ? min.z : max.z);
              corner_unscaled = lattice.unscale(corner);
              corner2_unscaled = latticeOld.unscale(corner);
              corner.x = (corner_unscaled.x - corner2_unscaled.x) - center_cur.x;
              corner.y = (corner_unscaled.y - corner2_unscaled.y) - center_cur.y;
              corner.z = (corner_unscaled.z - corner2_unscaled.z) - center_cur.z;
              cd = corner.x * corner.x + 
                   corner.y * corner.y + corner.z * corner.z;
              if (cd > max_cd) max_cd = cd;
          }
        }
      }
    }
    __syncthreads();
      
    // Alrights, so we have max_cd for each patch, now we get the max_pd
    // XXX TODO: Get the number of atoms per patch
    // we need the nAtomsPerPatch
    // we need an atomStart here
    unsigned int mc = 0;
    for(int i = threadIdx.x; i < numAtoms; i += stride){
      // Here I can also check for margin violations - we need the aAwayDists
      double3 pos;
      pos.x = pos_x[start + i];
      pos.y = pos_y[start + i];
      pos.z = pos_z[start + i];
      s = lattice.scale(pos);
      if(T_ISPERIODIC){ 
        // This is true if the system in periodic in A, B and C
        // if any of these clauses are true, atom needs to migrate
        mc += ((s.x < minx || s.x >= maxx) || (s.y < miny || s.y >= maxy) || (s.z < minz || s.z >= maxz)) ? 1 : 0;
      }else{
        int xdev, ydev, zdev;
        // Ok, so if the system is not periodic, we need to access the mInfo data structure to determine migrations
        if (s.x < minx) xdev = 0;
        else if (maxx <= s.x) xdev = 2;
        else xdev = 1;

        if (s.y < miny) ydev = 0;
        else if (maxy <= s.y) ydev = 2;
        else ydev = 1;

        if (s.z < minz) zdev = 0;
        else if (maxz <= s.z) zdev = 2;
        else zdev = 1;
        
        if(xdev != 1 || ydev != 1 || zdev != 1){
          // we check if any *dev are different than zero to prevent this horrible global memory access here
          int destPatch = mInfo[ind].destPatchID[xdev][ydev][zdev];
          if(destPatch != -1 && destPatch != pid) mc += 1; // atom needs to migrate
        }
      }
      corner.x = (pos.x - pos_old_x[start + i]) - center_cur.x;
      corner.y = (pos.y - pos_old_y[start + i]) - center_cur.y;
      corner.z = (pos.z - pos_old_z[start + i]) - center_cur.z;
      cd = corner.x * corner.x + 
           corner.y * corner.y + corner.z * corner.z;
      if(cd > max_pd) max_pd = cd;
    }
    // JM NOTE: The atomic add to host memory is bad, but if you have margin violations the simulation is going badly
    //          to begin with
    if (mc != 0) {
      atomicAdd(h_marginViolations, mc);
    }

    typedef cub::BlockReduce<BigReal, PATCH_BLOCKS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    max_pd = BlockReduce(temp_storage).Reduce(max_pd, cub::Max());
    if(threadIdx.x == 0){
      max_pd = sqrt(max_cd) + sqrt(max_pd);
      // this might not be needed since I'm updating host-mapped values anyways
      patchMaxAtomMovement[ind] = max_pd;
      if(max_pd > (1.f - c_pairlistTrigger) * patchNewTolerance[ind]){
        patchNewTolerance[ind] *= (1.f + c_pairlistGrow);
      }
      if(max_pd > patchNewTolerance[ind]){
        patchNewTolerance[ind] = max_pd / (1.f - c_pairlistTrigger);
      }
      // printf("patchNewTolerance[%d] =  %lf %lf\n", ind,
      //  patchNewTolerance[ind], patchMaxAtomMovement[ind]);
      h_patchMaxAtomMovement[ind] = max_pd; // Host-mapped update
      h_patchNewTolerance[ind] = patchNewTolerance[ind];
    }

    // Checks if periodic cell has become too small and flags it
    if(threadIdx.x == 0){
      if ( ( aDist.x*sysdima < minSize*0.9999 ) ||
         ( aDist.y*sysdimb < minSize*0.9999 ) ||
         ( aDist.z*sysdimc < minSize*0.9999 ) ||
         ( mValue.x < -0.0001 )               ||
         ( mValue.y < -0.0001 )               ||
         ( mValue.z < -0.0001)) {
        *h_periodicCellSmall = 1;
      }
    }
  }
}

void SequencerCUDAKernel::PairListMarginCheck(const int numPatches, 
  CudaLocalRecord*                  localRecords,
  const double*  pos_x,
  const double*  pos_y,
  const double*  pos_z,
  const double*  pos_old_x,
  const double*  pos_old_y,
  const double*  pos_old_z,
  const double3* awayDists, // for margin check
  const Lattice lattices,
  const Lattice lattices_old,
  const double3* __restrict patchMins,
  const double3* __restrict patchMaxes,
  const double3* __restrict patchCenter,
  const CudaMInfo* __restrict mInfo,
  unsigned int* __restrict tbcatomic,
  const double pairlistTrigger,
  const double pairlistGrow,
  const double pairlistShrink,
  double* __restrict patchMaxAtomMovement,
  double* __restrict h_patchMaxAtomMovement,
  double* __restrict patchNewTolerance,
  double* __restrict h_patchNewTolerance,
  const double                 minSize,
  const double                 cutoff,
  const double                 sysdima,
  const double                 sysdimb,
  const double                 sysdimc,
  unsigned int*                h_marginViolations,
  unsigned int*               h_periodicCellSmall,
  const bool rescalePairlistTolerance,
  const bool isPeriodic,
  cudaStream_t stream){
  int grid = numPatches;

  if(!this->intConstInit){
    this->intConstInit = true;
   cudaCheck(cudaMemcpyToSymbol(c_pairlistTrigger, &pairlistTrigger, sizeof(double)));
   cudaCheck(cudaMemcpyToSymbol(c_pairlistGrow, &pairlistGrow, sizeof(double)));
   cudaCheck(cudaMemcpyToSymbol(c_pairlistShrink, &pairlistShrink, sizeof(double)));
  }

  if(isPeriodic){
    pairListMarginCheckKernel<true><<<grid, PATCH_BLOCKS, 0, stream>>>(
      numPatches, localRecords,
      pos_x, pos_y, pos_z, pos_old_x,
      pos_old_y, pos_old_z, awayDists, lattices, lattices_old,
      patchMins, patchMaxes, patchCenter, mInfo, tbcatomic, 
      patchMaxAtomMovement, h_patchMaxAtomMovement, 
      patchNewTolerance, h_patchNewTolerance, 
      minSize, cutoff, sysdima, sysdimb, sysdimc, h_marginViolations, h_periodicCellSmall,
      rescalePairlistTolerance);
  }
  else{
    pairListMarginCheckKernel<false><<<grid, PATCH_BLOCKS, 0, stream>>>(
      numPatches, localRecords, 
      pos_x, pos_y, pos_z, pos_old_x,
      pos_old_y, pos_old_z, awayDists, lattices, lattices_old, 
      patchMins, patchMaxes, patchCenter, mInfo, tbcatomic, 
      patchMaxAtomMovement, h_patchMaxAtomMovement, 
      patchNewTolerance, h_patchNewTolerance, 
      minSize, cutoff, sysdima, sysdimb, sysdimc, h_marginViolations, h_periodicCellSmall,
      rescalePairlistTolerance);
  }
}

template <bool doNbond, bool doSlow>
__global__ void copyFBondToSOA(double *f_bond,
  double *f_bond_nbond,
  double *f_bond_slow,
  double *f_bond_x,
  double *f_bond_y,
  double *f_bond_z,
  double *f_nbond_x,
  double *f_nbond_y,
  double *f_nbond_z,
  double *f_slow_x,
  double *f_slow_y,
  double *f_slow_z,
  const int forceStride,
  const PatchRecord *pr,
  const int *patchIDs,
  const int *patchOffsets)
{
  // I suppose if I work with the entire PatchMap, this should work?
  // Same thing, each block gets a patch
  // What do I need -> Forces + atomStart
  // the bonded forces are wrong here.
  // What isforceOffset and patchOffset?
  int stride       = blockDim.x;
  int patchID      = patchIDs[blockIdx.x];
  // we need to check this data structure first to make sure it is safe to access it using patchID
  // I think patchId is the correct way of indexing this datastructure. Does this change with +p?
  int natoms       = pr[patchID].numAtoms;
  int forceOffset  = pr[patchID].atomStart;
  int patchOffset  = patchOffsets[blockIdx.x];

  for(int i = threadIdx.x; i < natoms; i+=stride){
    //        LOCAL                   GLOBAL
    f_bond_x[patchOffset + i] = f_bond[forceOffset + i];
    f_bond_y[patchOffset + i] = f_bond[forceOffset + i +   forceStride];
    f_bond_z[patchOffset + i] = f_bond[forceOffset + i + 2*forceStride];

    if(doNbond){
      // XXX Accumulation here works because each f_nbond_* buffer
      // is first initialized in copyFNBondToSOA.
      f_nbond_x[patchOffset + i] += f_bond_nbond[forceOffset + i];
      f_nbond_y[patchOffset + i] += f_bond_nbond[forceOffset + i + forceStride];
      f_nbond_z[patchOffset + i] += f_bond_nbond[forceOffset + i + 2*forceStride];
    }

    if(doSlow){
      // XXX Accumulation here works because each f_slow_* buffer
      // is first initialized in copyFSlowToSOA.
      f_slow_x[patchOffset + i] += f_bond_slow[forceOffset + i];
      f_slow_y[patchOffset + i] += f_bond_slow[forceOffset + i +   forceStride];
      f_slow_z[patchOffset + i] += f_bond_slow[forceOffset + i + 2*forceStride];
    }
  }
}

void SequencerCUDAKernel::copy_bond_forces(int           numPatches,
                                           double*       f_bond,
                                           double*       f_bond_nbond,
                                           double*       f_bond_slow,
                                           double*       f_bond_x,
                                           double*       f_bond_y,
                                           double*       f_bond_z,
                                           double*       f_nbond_x,
                                           double*       f_nbond_y,
                                           double*       f_nbond_z,
                                           double*       f_slow_x,
                                           double*       f_slow_y,
                                           double*       f_slow_z,
                                           int           forceStride, //if stridedForces
                                           PatchRecord*  pr,
                                           const int*    patchIDs,
                                           const int*    patchOffsets,
                                           bool          doNbond,
                                           bool          doSlow,
                                           cudaStream_t  stream)
{
  if(doSlow){
    copyFBondToSOA<true, true><<< numPatches, PATCH_BLOCKS, 0, stream >>>(f_bond, f_bond_nbond,
      f_bond_slow, f_bond_x, f_bond_y, f_bond_z,
      f_nbond_x, f_nbond_y, f_nbond_z,
      f_slow_x, f_slow_y, f_slow_z, forceStride,
      pr, patchIDs, patchOffsets);
   }else if(doNbond){
     copyFBondToSOA<true, false><<< numPatches, PATCH_BLOCKS, 0, stream >>>(f_bond, f_bond_nbond,
       f_bond_slow, f_bond_x, f_bond_y, f_bond_z,
       f_nbond_x, f_nbond_y, f_nbond_z,
       f_slow_x, f_slow_y, f_slow_z, forceStride,
       pr, patchIDs, patchOffsets);
   }else{
     copyFBondToSOA<false, false><<< numPatches, PATCH_BLOCKS, 0, stream >>>(f_bond, f_bond_nbond,
       f_bond_slow, f_bond_x, f_bond_y, f_bond_z,
       f_nbond_x, f_nbond_y, f_nbond_z,
       f_slow_x, f_slow_y, f_slow_z, forceStride,
       pr, patchIDs, patchOffsets);
   }
}

__global__ void copyFSlowToSOA(const CudaForce* __restrict__   f_slow,
                               double*                         f_slow_x,
                               double*                         f_slow_y,
                               double*                         f_slow_z,
                               const int* __restrict__ patchOffsets,
                               const Lattice* __restrict__ lattices)
{
  int tid            = threadIdx.x;
  int stride         = blockDim.x;
  int patchOffset    = patchOffsets[blockIdx.x];
  int numAtoms       = patchOffsets[blockIdx.x + 1] - patchOffset;
  
  Lattice lat = lattices[0];
  
  for(int i = tid; i < numAtoms; i += stride){
    CudaForce f  = f_slow[patchOffset + i];

    double3 f_scaled = lat.scale_force(
      Vector((double)f.x, (double)f.y, (double)f.z));
 
    // XXX Instead of accumulating slow force (+=), set them here (=)!
    f_slow_x[patchOffset + i] = f_scaled.x;
    f_slow_y[patchOffset + i] = f_scaled.y;
    f_slow_z[patchOffset + i] = f_scaled.z;
  }
}

void SequencerCUDAKernel::copy_slow_forces(int numPatches,
  const CudaForce* f_slow,
  double*          f_slow_x,
  double*          f_slow_y,
  double*          f_slow_z,
  const int* d_patchOffsets,
  const Lattice *lattices,
  cudaStream_t     stream){

  copyFSlowToSOA<<<numPatches, PATCH_BLOCKS, 0, stream>>>(f_slow,
    f_slow_x, f_slow_y, f_slow_z, d_patchOffsets, lattices);
}

__forceinline__ __device__ void zero_cudaTensor(cudaTensor &v)
{
  v.xx = 0.0;
  v.xy = 0.0;
  v.xz = 0.0;
  v.yx = 0.0;
  v.yy = 0.0;
  v.yz = 0.0;
  v.zx = 0.0;
  v.zy = 0.0;
  v.zz = 0.0;
}

template <bool DO_VEL_RESCALING>
__global__ void addForceToMomentumKernel(const double scaling,
                                         double       dt_normal,
                                         double       dt_nbond,
                                         double       dt_slow,
                                         double       velrescaling, // for stochastic velocity rescaling
                                         const double  * __restrict recipMass,
                                         const double * __restrict f_normal_x,
                                         const double * __restrict f_normal_y,
                                         const double * __restrict f_normal_z,
                                         const double * __restrict f_nbond_x,
                                         const double * __restrict f_nbond_y,
                                         const double * __restrict f_nbond_z,
                                         const double * __restrict f_slow_x,
                                         const double * __restrict f_slow_y,
                                         const double * __restrict f_slow_z,
                                         double       * __restrict vel_x,
                                         double       * __restrict vel_y,
                                         double       * __restrict vel_z,
                                         const int    numAtoms,
                                         const int    maxForceNumber)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numAtoms) {
#if 0
    switch (maxForceNumber) {
    case 2:
      dt_slow *= scaling;
      vel_x[i] += f_slow_x[i] * recipMass[i] * dt_slow;
      vel_y[i] += f_slow_y[i] * recipMass[i] * dt_slow;
      vel_z[i] += f_slow_z[i] * recipMass[i] * dt_slow;
      // fall through because we will always have the "faster" forces
    case 1:
      dt_nbond *= scaling;
      vel_x[i] += f_nbond_x[i] * recipMass[i] * dt_nbond;
      vel_y[i] += f_nbond_y[i] * recipMass[i] * dt_nbond;
      vel_z[i] += f_nbond_z[i] * recipMass[i] * dt_nbond;
      // fall through because we will always have the "faster" forces
    case 0:
      dt_normal *= scaling;
      vel_x[i] += f_normal_x[i] * recipMass[i] * dt_normal;
      vel_y[i] += f_normal_y[i] * recipMass[i] * dt_normal;
      vel_z[i] += f_normal_z[i] * recipMass[i] * dt_normal;
    }
#else
    double vx = 0;
    double vy = 0;
    double vz = 0;
    switch (maxForceNumber) {
    case 2:
      vx += f_slow_x[i] * dt_slow;
      vy += f_slow_y[i] * dt_slow;
      vz += f_slow_z[i] * dt_slow;
      // fall through because we will always have the "faster" forces
    case 1:
      vx += f_nbond_x[i] * dt_nbond;
      vy += f_nbond_y[i] * dt_nbond;
      vz += f_nbond_z[i] * dt_nbond;
      // fall through because we will always have the "faster" forces
    case 0:
      vx += f_normal_x[i] * dt_normal;
      vy += f_normal_y[i] * dt_normal;
      vz += f_normal_z[i] * dt_normal;
    }
    vx *= scaling * recipMass[i];
    vy *= scaling * recipMass[i];
    vz *= scaling * recipMass[i];
    if (DO_VEL_RESCALING) {
      vel_x[i] = velrescaling*vel_x[i] + vx;
      vel_y[i] = velrescaling*vel_y[i] + vy;
      vel_z[i] = velrescaling*vel_z[i] + vz;
    }
    else {
      vel_x[i] += vx;
      vel_y[i] += vy;
      vel_z[i] += vz;
    }
#endif
  }
}

// JM: This sets the cudaAtom vector from the nonbonded compute
template <bool t_doNbond, bool t_doHomePatches>
__global__ void setComputePositionsKernel(
  CudaLocalRecord*                  localRecords,
  CudaPeerRecord*                   peerRecords,
  const double3* __restrict         patchCenter, 
  const int* __restrict             patchSortOrder, 
  const int                         numPatches,
  const unsigned int                devID, 
  const Lattice                     lat,
  const double                      charge_scaling, 
  const double* __restrict          d_pos_x, 
  const double* __restrict          d_pos_y, 
  const double* __restrict          d_pos_z,
  const float* __restrict           charges,
  double** __restrict               d_peer_pos_x, 
  double** __restrict               d_peer_pos_y, 
  double** __restrict               d_peer_pos_z, 
  float4* __restrict                nb_atoms,
  float4* __restrict                b_atoms,
  float4* __restrict                s_atoms
) {

  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    double3 center;
    //int soapid = globalToLocalID[record.patchID];
    center = patchCenter[patchIndex];
    double3 ucenter = lat.unscale(center);

    const int numAtoms = s_record.numAtoms;
    const int dstOffset = s_record.bufferOffset;
    const int dstOffsetNB = s_record.bufferOffsetNBPad;

    int srcDevice, srcOffset;
    const double *srcPosX, *srcPosY, *srcPosZ;
    if (t_doHomePatches) {
      srcDevice = devID;
      srcOffset = dstOffset;
      srcPosX = d_pos_x;
      srcPosY = d_pos_y;
      srcPosZ = d_pos_z;
    } else {
      srcDevice = s_record.inline_peers[0].deviceIndex;
      srcOffset = s_record.inline_peers[0].bufferOffset;
      srcPosX = d_peer_pos_x[srcDevice];
      srcPosY = d_peer_pos_y[srcDevice];
      srcPosZ = d_peer_pos_z[srcDevice];
    }

    float4 atom;
    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
      const int order = patchSortOrder[dstOffset + i]; // Should this be order or unorder?
      atom.x = srcPosX[srcOffset + order] - ucenter.x;
      atom.y = srcPosY[srcOffset + order] - ucenter.y;
      atom.z = srcPosZ[srcOffset + order] - ucenter.z;
      atom.w = charges[dstOffset + order] * charge_scaling;

      b_atoms[dstOffset + order] = atom;
      if (t_doNbond) nb_atoms[dstOffsetNB + i] = atom;
    }

    if (t_doNbond) {
      if (threadIdx.x / WARPSIZE == 0) {
        const int to_write = (((numAtoms+32-1)/32)*32) - numAtoms; // WARPSIZE

        float4 lastAtom;
        const int order = patchSortOrder[dstOffset + numAtoms - 1]; // Should this be order or unorder?
        lastAtom.x = srcPosX[srcOffset + order] - ucenter.x;
        lastAtom.y = srcPosY[srcOffset + order] - ucenter.y;
        lastAtom.z = srcPosZ[srcOffset + order] - ucenter.z;
        lastAtom.w = charges[dstOffset + order] * charge_scaling;

        if (threadIdx.x < to_write) {
          nb_atoms[dstOffsetNB+numAtoms+threadIdx.x] = lastAtom;
        }
      }
    }
    __syncthreads();
  }
}

template<bool t_doHomePatches, bool t_doFEP, bool t_doTI, bool t_doAlchDecouple, bool t_doAlchSoftCore>
__global__ void setComputePositionsKernel_PME (
  const double* __restrict          d_pos_x, 
  const double* __restrict          d_pos_y, 
  const double* __restrict          d_pos_z,
  const float* __restrict           charges,
  double** __restrict               d_peer_pos_x, 
  double** __restrict               d_peer_pos_y, 
  double** __restrict               d_peer_pos_z, 
  float** __restrict                d_peer_charge,
  int** __restrict                  d_peer_partition,
  const int* __restrict             partition,
  const double                      charge_scaling, 
  const int* __restrict             s_patchPositions,
  const int* __restrict             s_pencilPatchIndex,
  const int* __restrict             s_patchIDs,
  const Lattice                     lat,
  float4* __restrict                s_atoms,
  int                               numTotalAtoms,
  int* __restrict                   s_partition,
// NEW INPUT
  const int peerDevice,
  const int numAtoms,
  const int offset
) {

  float4 foo;

  const int pmeBufferOffset = offset;
  const int srcBufferOffset = 0; 
  const int srcDevice = peerDevice;

  // double precision calculation
  double px, py, pz, wx, wy, wz, q;
  int part;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numAtoms; i += blockDim.x * gridDim.x) {
    // this gets atoms in same order as PmeCuda code path
    if (t_doHomePatches) {
      q = (double)(charges[srcBufferOffset + i]);
      px = d_pos_x[srcBufferOffset + i];
      py = d_pos_y[srcBufferOffset + i];
      pz = d_pos_z[srcBufferOffset + i];
      if (t_doFEP || t_doTI) {
        part = partition[srcBufferOffset + i];
      }
    } else {
      q = (double)(d_peer_charge[srcDevice][srcBufferOffset + i]);
      px = d_peer_pos_x[srcDevice][srcBufferOffset + i];
      py = d_peer_pos_y[srcDevice][srcBufferOffset + i];
      pz = d_peer_pos_z[srcDevice][srcBufferOffset + i];
      if (t_doFEP || t_doTI) {
        part = d_peer_partition[srcDevice][srcBufferOffset + i];
      }
    }

    double3 w_vec = lat.scale(Vector(px, py, pz));
    wx = w_vec.x;
    wy = w_vec.y;
    wz = w_vec.z;
    wx = (wx - (floor(wx + 0.5) - 0.5));
    wy = (wy - (floor(wy + 0.5) - 0.5));
    wz = (wz - (floor(wz + 0.5) - 0.5));
    foo.x = (float) wx;
    foo.y = (float) wy;
    foo.z = (float) wz;
    foo.w = (float) (charge_scaling * q);
    foo.x = foo.x - 1.0f*(foo.x >= 1.0f);
    foo.y = foo.y - 1.0f*(foo.y >= 1.0f);
    foo.z = foo.z - 1.0f*(foo.z >= 1.0f);
    if (!t_doFEP && !t_doTI) {
      s_atoms[pmeBufferOffset + i] = foo;
    }
    else { // alchemical multiple grids
      float4 foo_zero_charge = foo;
      foo_zero_charge.w = 0.0f;
      s_partition[pmeBufferOffset + i] = part;
      /*                        grid 0      grid 1      grid 2      grid 3      grid 4
      * non-alch     (p = 0)     Y           Y           N           N           Y
      * appearing    (p = 1)     Y           N           Y           N           N
      * disappearing (p = 2)     N           Y           N           Y           N
      * Requirements of grids:
      * 1. t_doFEP || t_doTI : grid 0, grid 1
      * 2. t_doAlchDecouple : grid 2, grid 3
      * 3. t_doAlchSoftCore || t_doTI: grid 4
      * grid 4 can be s_atoms[i + 4 * numAtoms] (t_doAlchDecouple) or s_atoms[i + 2 * numAtoms] (!t_doAlchDecouple)
      * although the atoms that have zero charges in extra grids would not change in non-migration steps,
      * I still find no way to get rid of these copying, because positions of the atoms can be changed.
      * The non-zero charges may also change if they are computed from some QM engines or some new kinds of FF.
      * It seems these branchings in non-migration steps are inevitable.
      */
      switch (part) {
        // non-alch atoms
        case 0: {
          s_atoms[pmeBufferOffset + i] = foo;
          s_atoms[pmeBufferOffset + i + numTotalAtoms] = foo;
          if (t_doAlchDecouple) {
            s_atoms[pmeBufferOffset + i + 2 * numTotalAtoms] = foo_zero_charge;
            s_atoms[pmeBufferOffset + i + 3 * numTotalAtoms] = foo_zero_charge;
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[pmeBufferOffset + i + 4 * numTotalAtoms] = foo;
            }
          } else {
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[pmeBufferOffset + i + 2 * numTotalAtoms] = foo;
            }
          }
          break;
        }
        // appearing atoms
        case 1: {
          s_atoms[pmeBufferOffset + i] = foo;
          s_atoms[pmeBufferOffset + i + numTotalAtoms] = foo_zero_charge;
          if (t_doAlchDecouple) {
            s_atoms[pmeBufferOffset + i + 2 * numTotalAtoms] = foo;
            s_atoms[pmeBufferOffset + i + 3 * numTotalAtoms] = foo_zero_charge;
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[pmeBufferOffset + i + 4 * numTotalAtoms] = foo_zero_charge;
            }
          } else {
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[pmeBufferOffset + i + 2 * numTotalAtoms] = foo_zero_charge;
            }
          }
          break;
        }
        // disappearing atoms
        case 2: {
          s_atoms[pmeBufferOffset + i] = foo_zero_charge;
          s_atoms[pmeBufferOffset + i + numTotalAtoms] = foo;
          if (t_doAlchDecouple) {
            s_atoms[pmeBufferOffset + i + 2 * numTotalAtoms] = foo_zero_charge;
            s_atoms[pmeBufferOffset + i + 3 * numTotalAtoms] = foo;
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[pmeBufferOffset + i + 4 * numTotalAtoms] = foo_zero_charge;
            }
          } else {
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[pmeBufferOffset + i + 2 * numTotalAtoms] = foo_zero_charge;
            }
          }
          break;
        }
      }
    }
  }
}

void SequencerCUDAKernel::set_compute_positions(
  const int  devID,
  const bool isPmeDevice, 
  const int  nDev, 
  const int  numPatchesHomeAndProxy,
  const int  numPatchesHome,
  const bool doNbond, 
  const bool doSlow,
  const bool doFEP,
  const bool doTI,
  const bool doAlchDecouple,
  const bool doAlchSoftCore,
  const double* d_pos_x,
  const double* d_pos_y,
  const double* d_pos_z,
#ifndef NAMD_NCCL_ALLREDUCE
  double**      d_peer_pos_x, 
  double**      d_peer_pos_y, 
  double**      d_peer_pos_z, 
  float**       d_peer_charge, 
  int**         d_peer_partition,
#endif
  const float* charges,
  const int* partition,
  const double charge_scaling, 
  const double3* patchCenter, 
  const int* s_patchPositions, 
  const int* s_pencilPatchIndex, 
  const int* s_patchIDs, 
  const int* patchSortOrder, 
  const Lattice lattice,
  float4* nb_atoms,
  float4* b_atoms,
  float4* s_atoms,
  int* s_partition,
  int numTotalAtoms,
  CudaLocalRecord*                  localRecords,
  CudaPeerRecord*                   peerRecords,
  std::vector<int>& atomCounts,
  cudaStream_t stream)
{
  // Launch Local Set Compute Positions
  if(doNbond){
    setComputePositionsKernel<true, true><<<numPatchesHome, PATCH_BLOCKS, 0, stream>>>(
      localRecords, peerRecords, patchCenter, patchSortOrder,
      numPatchesHome, devID, lattice, charge_scaling,
      d_pos_x, d_pos_y, d_pos_z, charges, d_peer_pos_x, d_peer_pos_y, d_peer_pos_z, 
      nb_atoms, b_atoms, s_atoms 
    );
  } else {
    setComputePositionsKernel<false, true><<<numPatchesHome, PATCH_BLOCKS, 0, stream>>>(
      localRecords, peerRecords, patchCenter, patchSortOrder,
      numPatchesHome, devID, lattice, charge_scaling,
      d_pos_x, d_pos_y, d_pos_z, charges, d_peer_pos_x, d_peer_pos_y, d_peer_pos_z, 
      nb_atoms, b_atoms, s_atoms 
    );
  }

  // Launch Peer Set Computes
  if (nDev != 1) {
    const int numProxyPatches = numPatchesHomeAndProxy - numPatchesHome;
    if(doNbond){
      setComputePositionsKernel<true, false><<<numProxyPatches, PATCH_BLOCKS, 0, stream>>>(
        localRecords + numPatchesHome, peerRecords, patchCenter + numPatchesHome, patchSortOrder,
        numProxyPatches, devID, lattice, charge_scaling,
        d_pos_x, d_pos_y, d_pos_z, charges, d_peer_pos_x, d_peer_pos_y, d_peer_pos_z, 
        nb_atoms, b_atoms, s_atoms 
      );
    } else {
      setComputePositionsKernel<true, false><<<numProxyPatches, PATCH_BLOCKS, 0, stream>>>(
        localRecords + numPatchesHome, peerRecords, patchCenter + numPatchesHome, patchSortOrder,
        numProxyPatches, devID, lattice, charge_scaling,
        d_pos_x, d_pos_y, d_pos_z, charges, d_peer_pos_x, d_peer_pos_y, d_peer_pos_z, 
        nb_atoms, b_atoms, s_atoms
      );
    }
  }

  // Launch PME setComputes
  #define CALL(HOME, FEP, TI, ALCH_DECOUPLE, ALCH_SOFTCORE) \
    setComputePositionsKernel_PME<HOME, FEP, TI, ALCH_DECOUPLE, ALCH_SOFTCORE> \
    <<<pme_grid, PATCH_BLOCKS, 0, stream>>>( \
          d_pos_x, d_pos_y, d_pos_z, charges, \
          d_peer_pos_x, d_peer_pos_y, d_peer_pos_z, d_peer_charge, \
          d_peer_partition, partition, charge_scaling,  \
          s_patchPositions, s_pencilPatchIndex, s_patchIDs, \
          lattice, s_atoms, numTotalAtoms, s_partition, \
          i, numAtoms, offset \
    );
  // Only when PME long-range electrostaic is enabled (doSlow is true) 
  // The partition is needed for alchemical calculation.
  if(doSlow && isPmeDevice) {
    int offset = 0;
    for (int i = 0; i < nDev; i++) {
      const bool home = (i == devID);
      const int numAtoms = atomCounts[i];
      const int pme_grid = (numAtoms + PATCH_BLOCKS - 1) / PATCH_BLOCKS;
      const int options = (home << 4) + (doFEP << 3) + (doTI << 2) + (doAlchDecouple << 1) + doAlchSoftCore;

      switch (options) {
        case  0: CALL(0, 0, 0, 0, 0); break;
        case  1: CALL(0, 0, 0, 0, 1); break;
        case  2: CALL(0, 0, 0, 1, 0); break;
        case  3: CALL(0, 0, 0, 1, 1); break;
        case  4: CALL(0, 0, 1, 0, 0); break;
        case  5: CALL(0, 0, 1, 0, 1); break;
        case  6: CALL(0, 0, 1, 1, 0); break;
        case  7: CALL(0, 0, 1, 1, 1); break;
        case  8: CALL(0, 1, 0, 0, 0); break;
        case  9: CALL(0, 1, 0, 0, 1); break;
        case 10: CALL(0, 1, 0, 1, 0); break;
        case 11: CALL(0, 1, 0, 1, 1); break;
        case 12: CALL(0, 1, 1, 0, 0); break;
        case 13: CALL(0, 1, 1, 0, 1); break;
        case 14: CALL(0, 1, 1, 1, 0); break;
        case 15: CALL(0, 1, 1, 1, 1); break;
        case 16: CALL(1, 0, 0, 0, 0); break;
        case 17: CALL(1, 0, 0, 0, 1); break;
        case 18: CALL(1, 0, 0, 1, 0); break;
        case 19: CALL(1, 0, 0, 1, 1); break;
        case 20: CALL(1, 0, 1, 0, 0); break;
        case 21: CALL(1, 0, 1, 0, 1); break;
        case 22: CALL(1, 0, 1, 1, 0); break;
        case 23: CALL(1, 0, 1, 1, 1); break;
        case 24: CALL(1, 1, 0, 0, 0); break;
        case 25: CALL(1, 1, 0, 0, 1); break;
        case 26: CALL(1, 1, 0, 1, 0); break;
        case 27: CALL(1, 1, 0, 1, 1); break;
        case 28: CALL(1, 1, 1, 0, 0); break;
        case 29: CALL(1, 1, 1, 0, 1); break;
        case 30: CALL(1, 1, 1, 1, 0); break;
        case 31: CALL(1, 1, 1, 1, 1); break;
        default:
          NAMD_die("SequencerCUDAKernel::setComputePositions: no kernel called");
      }

      offset += numAtoms;
    }
  }
  #undef CALL
}

template <bool t_doFEP, bool t_doTI, bool t_doAlchDecouple, bool t_doAlchSoftCore>
__global__ void setPmePositionsKernel(
    double charge_scaling,
    const Lattice lattice,
    const double *pos_x,
    const double *pos_y,
    const double *pos_z,
    const float *charges,
    const int* partition,
    float4 *s_atoms,
    int* s_atoms_partition,
    int numAtoms)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numAtoms) {
    Lattice lat = lattice;
    float4 foo;
    double wx, wy, wz, q;
    q = (double)(charges[i]);

    double3 w_vec = lat.scale(Vector(pos_x[i], pos_y[i], pos_z[i]));

    wx = w_vec.x;
    wy = w_vec.y;
    wz = w_vec.z;
    wx = (wx - (floor(wx + 0.5) - 0.5));
    wy = (wy - (floor(wy + 0.5) - 0.5));
    wz = (wz - (floor(wz + 0.5) - 0.5));
    foo.x = (float) wx;
    foo.y = (float) wy;
    foo.z = (float) wz;
    foo.w = (float) (charge_scaling * q);
    foo.x = foo.x - 1.0f*(foo.x >= 1.0f);
    foo.y = foo.y - 1.0f*(foo.y >= 1.0f);
    foo.z = foo.z - 1.0f*(foo.z >= 1.0f);
    if (!t_doFEP && !t_doTI) {
      s_atoms[i] = foo;
    }
    else { // alchemical multiple grids
      float4 foo_zero_charge = foo;
      foo_zero_charge.w = 0.0f;
      s_atoms_partition[i] = partition[i];
      /*                        grid 0      grid 1      grid 2      grid 3      grid 4
       * non-alch     (p = 0)     Y           Y           N           N           Y
       * appearing    (p = 1)     Y           N           Y           N           N
       * disappearing (p = 2)     N           Y           N           Y           N
       * Requirements of grids:
       * 1. t_doFEP || t_doTI : grid 0, grid 1
       * 2. t_doAlchDecouple : grid 2, grid 3
       * 3. t_doAlchSoftCore || t_doTI: grid 4
       * grid 4 can be s_atoms[i + 4 * numAtoms] (t_doAlchDecouple) or s_atoms[i + 2 * numAtoms] (!t_doAlchDecouple)
       */
      switch (partition[i]) {
        // non-alch atoms
        case 0: {
          s_atoms[i] = foo;
          s_atoms[i + numAtoms] = foo;
          if (t_doAlchDecouple) {
            s_atoms[i + 2 * numAtoms] = foo_zero_charge;
            s_atoms[i + 3 * numAtoms] = foo_zero_charge;
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[i + 4 * numAtoms] = foo;
            }
          } else {
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[i + 2 * numAtoms] = foo;
            }
          }
          break;
        }
        // appearing atoms
        case 1: {
          s_atoms[i] = foo;
          s_atoms[i + numAtoms] = foo_zero_charge;
          if (t_doAlchDecouple) {
            s_atoms[i + 2 * numAtoms] = foo;
            s_atoms[i + 3 * numAtoms] = foo_zero_charge;
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[i + 4 * numAtoms] = foo_zero_charge;
            }
          } else {
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[i + 2 * numAtoms] = foo_zero_charge;
            }
          }
          break;
        }
        // disappearing atoms
        case 2: {
          s_atoms[i] = foo_zero_charge;
          s_atoms[i + numAtoms] = foo;
          if (t_doAlchDecouple) {
            s_atoms[i + 2 * numAtoms] = foo_zero_charge;
            s_atoms[i + 3 * numAtoms] = foo;
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[i + 4 * numAtoms] = foo_zero_charge;
            }
          } else {
            if (t_doAlchSoftCore || t_doTI) {
              s_atoms[i + 2 * numAtoms] = foo_zero_charge;
            }
          }
          break;
        }
      }
    }
  }
}

__global__ void maximumMoveKernel(const double maxvel2,
                                  const double * __restrict vel_x,
                                  const double * __restrict vel_y,
                                  const double * __restrict vel_z,
                                  int          *killme,
                                  const int    numAtoms)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numAtoms) {
    double vel2 = vel_x[i]*vel_x[i] + vel_y[i]*vel_y[i] + vel_z[i]*vel_z[i];
    if (vel2 > maxvel2) {
      //If this ever happens, we're already screwed, so performance does not matter
      atomicAdd(killme, 1);
    }
  }
}

void SequencerCUDAKernel::maximumMove(const double maxvel2,
                                      const double *vel_x,
                                      const double *vel_y,
                                      const double *vel_z,
                                      int          *killme,
                                      const int    numAtoms,
                                      cudaStream_t stream)
{
  //cudaCheck(cudaMemset(killme, 0, sizeof(int)));
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  maximumMoveKernel<<<grid, ATOM_BLOCKS, 0, stream>>>(
    maxvel2, vel_x, vel_y, vel_z, killme, numAtoms);
}

void SequencerCUDAKernel::addForceToMomentum(
  const double scaling,
  double       dt_normal,
  double       dt_nbond,
  double       dt_slow,
  double       velrescaling,  // for stochastic velocity rescaling
  const double *recipMass,
  const double *f_normal_x,
  const double *f_normal_y,
  const double *f_normal_z,
  const double *f_nbond_x,
  const double *f_nbond_y,
  const double *f_nbond_z,
  const double *f_slow_x,
  const double *f_slow_y,
  const double *f_slow_z,
  double       *vel_x,
  double       *vel_y,
  double       *vel_z,
  int          numAtoms,
  int          maxForceNumber,
  cudaStream_t stream)
{
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  if (velrescaling != 1.0) {
    addForceToMomentumKernel<true><<<grid, ATOM_BLOCKS, 0, stream>>>(
        scaling, dt_normal, dt_nbond, dt_slow, velrescaling,
        recipMass, f_normal_x, f_normal_y, f_normal_z,
        f_nbond_x, f_nbond_y, f_nbond_z,
        f_slow_x, f_slow_y, f_slow_z,
        vel_x, vel_y, vel_z,
        numAtoms, maxForceNumber);
  }
  else {
    addForceToMomentumKernel<false><<<grid, ATOM_BLOCKS, 0, stream>>>(
        scaling, dt_normal, dt_nbond, dt_slow, velrescaling,
        recipMass, f_normal_x, f_normal_y, f_normal_z,
        f_nbond_x, f_nbond_y, f_nbond_z,
        f_slow_x, f_slow_y, f_slow_z,
        vel_x, vel_y, vel_z,
        numAtoms, maxForceNumber);
  }
  // cudaCheck(cudaGetLastError());
}

template <bool copyPos>
__global__ void addVelocityToPositionKernel(
  const double dt,
  const double * __restrict vel_x,
  const double * __restrict vel_y,
  const double * __restrict vel_z,
  double *       __restrict pos_x,
  double *       __restrict pos_y,
  double *       __restrict pos_z,
  double *       __restrict h_pos_x, // host-mapped vectors
  double *       __restrict h_pos_y,
  double *       __restrict h_pos_z,
  int numAtoms)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  double x,y,z;
  if (i < numAtoms) {
    x = pos_x[i];
    y = pos_y[i];
    z = pos_z[i];
    x += vel_x[i] * dt;
    y += vel_y[i] * dt;
    z += vel_z[i] * dt;
    pos_x[i] = x;
    pos_y[i] = y;
    pos_z[i] = z;
    if(copyPos){
      h_pos_x[i] = x;
      h_pos_y[i] = y;
      h_pos_z[i] = z;
    }
  }
}

void SequencerCUDAKernel::addVelocityToPosition(
  const double   dt,
  const double *vel_x,
  const double *vel_y,
  const double *vel_z,
  double *pos_x,
  double *pos_y,
  double *pos_z,
  double *h_pos_x,
  double *h_pos_y,
  double *h_pos_z,
  int     numAtoms,
  bool    copyPositions,
  cudaStream_t   stream)
{
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  if(copyPositions){
    addVelocityToPositionKernel<true><<<grid, ATOM_BLOCKS, 0, stream>>>(
     dt, vel_x, vel_y, vel_z, pos_x, pos_y, pos_z,
     h_pos_x, h_pos_y, h_pos_z, numAtoms);
  }else{
    addVelocityToPositionKernel<false><<<grid, ATOM_BLOCKS, 0, stream>>>(
     dt, vel_x, vel_y, vel_z, pos_x, pos_y, pos_z,
     h_pos_x, h_pos_y, h_pos_z, numAtoms);
  }
  //cudaCheck(cudaGetLastError());
}

__global__ void updateRigidArraysKernel(
  const double dt,
  const double * __restrict vel_x,
  const double * __restrict vel_y,
  const double * __restrict vel_z,
  const double * __restrict pos_x,
  const double * __restrict pos_y,
  const double * __restrict pos_z,
  double *       __restrict velNew_x,
  double *       __restrict velNew_y,
  double *       __restrict velNew_z,
  double *       __restrict posNew_x,
  double *       __restrict posNew_y,
  double *       __restrict posNew_z,
  int numAtoms)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numAtoms) {
    double vx = vel_x[i];
    double vy = vel_y[i];
    double vz = vel_z[i];
    posNew_x[i] = pos_x[i] + (vx * dt);
    posNew_y[i] = pos_y[i] + (vy * dt);
    posNew_z[i] = pos_z[i] + (vz * dt);
    velNew_x[i] = vx;
    velNew_y[i] = vy;
    velNew_z[i] = vz;
  }
}


// JM NOTE: Optimize this kernel further to improve global memory access pattern
__global__ void centerOfMassKernel(
  const double * __restrict coor_x,
  const double * __restrict coor_y,
  const double * __restrict coor_z,
  double * __restrict cm_x, // center of mass is atom-sized
  double * __restrict cm_y,
  double * __restrict cm_z,
  const float * __restrict mass,
  const int * __restrict hydrogenGroupSize,
  const int numAtoms)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numAtoms) {
    int hgs = hydrogenGroupSize[i];
    int j;
    if (hgs > 0) {
      // missing fixed atoms
      BigReal m_cm = 0;
      BigReal reg_cm_x = 0;
      BigReal reg_cm_y = 0;
      BigReal reg_cm_z = 0;
      for ( j = i; j < (i+hgs); ++j ) {
        m_cm += mass[j];
        reg_cm_x += mass[j] * coor_x[j];
        reg_cm_y += mass[j] * coor_y[j];
        reg_cm_z += mass[j] * coor_z[j];
      }
      BigReal inv_m_cm = 1.0/m_cm;
      reg_cm_x *= inv_m_cm;
      reg_cm_y *= inv_m_cm;
      reg_cm_z *= inv_m_cm;

      for(j = i ; j < (i + hgs); j++){
         cm_x[j] = reg_cm_x;
         cm_y[j] = reg_cm_y;
         cm_z[j] = reg_cm_z;
      }
    }
  }
}

void SequencerCUDAKernel::centerOfMass(
  const double *coor_x,
  const double *coor_y,
  const double *coor_z,
  double *cm_x,
  double *cm_y,
  double *cm_z,
  const float* mass,
  const int* hydrogenGroupSize,
  const int numAtoms,
  cudaStream_t stream
){
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  //Calls kernels
  centerOfMassKernel<<<grid, ATOM_BLOCKS, 0, stream>>>(coor_x, coor_y, coor_z,
      cm_x, cm_y, cm_z, mass, hydrogenGroupSize, numAtoms);
}


void SequencerCUDAKernel::updateRigidArrays(
  const double   dt,
  const double *vel_x,
  const double *vel_y,
  const double *vel_z,
  const double *pos_x,
  const double *pos_y,
  const double *pos_z,
  double *velNew_x,
  double *velNew_y,
  double *velNew_z,
  double *posNew_x,
  double *posNew_y,
  double *posNew_z,
  int            numAtoms,
  cudaStream_t   stream)
{
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  updateRigidArraysKernel<<<grid, ATOM_BLOCKS, 0, stream>>>(
    dt, vel_x, vel_y, vel_z, pos_x, pos_y, pos_z,
    velNew_x, velNew_y, velNew_z,
    posNew_x, posNew_y, posNew_z, numAtoms);
  // cudaCheck(cudaGetLastError());
}

template<int BLOCK_SIZE>
__global__ void submitHalfKernel(
  const double * __restrict vel_x,
  const double * __restrict vel_y,
  const double * __restrict vel_z,
  const double * __restrict vcm_x,
  const double * __restrict vcm_y,
  const double * __restrict vcm_z,
  const float  * __restrict mass,
  BigReal *kineticEnergy,
  BigReal *intKineticEnergy,
  cudaTensor *virial,
  cudaTensor *intVirialNormal,
  BigReal *h_kineticEnergy,
  BigReal *h_intKineticEnergy,
  cudaTensor *h_virial,
  cudaTensor *h_intVirialNormal,
  int *hydrogenGroupSize,
  int numAtoms,
  unsigned int* tbcatomic)
{
  BigReal k = 0, intK = 0;
  cudaTensor v, intV;
  zero_cudaTensor(v);
  zero_cudaTensor(intV);
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int totaltb = gridDim.x;
  __shared__ bool isLastBlockDone;


  if(threadIdx.x == 0){
    isLastBlockDone = 0;
  }

  __syncthreads();

  if (i < numAtoms) {
    k = mass[i] *
      (vel_x[i]*vel_x[i] + vel_y[i]*vel_y[i] + vel_z[i]*vel_z[i]);
    v.xx = mass[i] * vel_x[i] * vel_x[i];
    // v.xy = mass[i] * vel_x[i] * vel_y[i];
    // v.xz = mass[i] * vel_x[i] * vel_z[i];
    v.yx = mass[i] * vel_y[i] * vel_x[i];
    v.yy = mass[i] * vel_y[i] * vel_y[i];
    // v.yz = mass[i] * vel_y[i] * vel_z[i];
    v.zx = mass[i] * vel_z[i] * vel_x[i];
    v.zy = mass[i] * vel_z[i] * vel_y[i];
    v.zz = mass[i] * vel_z[i] * vel_z[i];

#if 0
    int hgs = hydrogenGroupSize[i];
    if (hgs > 0) {
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

      for (int j = i;  j < (i+hgs);  j++) {
        BigReal dv_x = vel_x[j] - v_cm_x;
        BigReal dv_y = vel_y[j] - v_cm_y;
        BigReal dv_z = vel_z[j] - v_cm_z;
        intK += mass[j] *
          (vel_x[j] * dv_x + vel_y[j] * dv_y + vel_z[j] * dv_z);
        intV.xx += mass[j] * vel_x[j] * dv_x;
        intV.xy += mass[j] * vel_x[j] * dv_y;
        intV.xz += mass[j] * vel_x[j] * dv_z;
        intV.yx += mass[j] * vel_y[j] * dv_x;
        intV.yy += mass[j] * vel_y[j] * dv_y;
        intV.yz += mass[j] * vel_y[j] * dv_z;
        intV.zx += mass[j] * vel_z[j] * dv_x;
        intV.zy += mass[j] * vel_z[j] * dv_y;
        intV.zz += mass[j] * vel_z[j] * dv_z;
      }
    }
#else
    // JM: New version with centers of mass calculated by a separate kernel
    BigReal v_cm_x = vcm_x[i];
    BigReal v_cm_y = vcm_y[i];
    BigReal v_cm_z = vcm_z[i];
    BigReal dv_x = vel_x[i] - v_cm_x;
    BigReal dv_y = vel_y[i] - v_cm_y;
    BigReal dv_z = vel_z[i] - v_cm_z;
    intK += mass[i] *
        (vel_x[i] * dv_x + vel_y[i] * dv_y + vel_z[i] * dv_z);
    intV.xx += mass[i] * vel_x[i] * dv_x;
    intV.xy += mass[i] * vel_x[i] * dv_y;
    intV.xz += mass[i] * vel_x[i] * dv_z;
    intV.yx += mass[i] * vel_y[i] * dv_x;
    intV.yy += mass[i] * vel_y[i] * dv_y;
    intV.yz += mass[i] * vel_y[i] * dv_z;
    intV.zx += mass[i] * vel_z[i] * dv_x;
    intV.zy += mass[i] * vel_z[i] * dv_y;
    intV.zz += mass[i] * vel_z[i] * dv_z;
#endif
  }

  typedef cub::BlockReduce<BigReal, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  k    = BlockReduce(temp_storage).Sum(k);
  __syncthreads();
  v.xx = BlockReduce(temp_storage).Sum(v.xx);
  __syncthreads();
  // v.xy = BlockReduce(temp_storage).Sum(v.xy);
  // __syncthreads();
  // v.xz = BlockReduce(temp_storage).Sum(v.xz);
  // __syncthreads();
  v.yx = BlockReduce(temp_storage).Sum(v.yx);
  __syncthreads();
  v.yy = BlockReduce(temp_storage).Sum(v.yy);
  __syncthreads();
  // v.yz = BlockReduce(temp_storage).Sum(v.yz);
  // __syncthreads();
  v.zx = BlockReduce(temp_storage).Sum(v.zx);
  __syncthreads();
  v.zy = BlockReduce(temp_storage).Sum(v.zy);
  __syncthreads();
  v.zz = BlockReduce(temp_storage).Sum(v.zz);
  __syncthreads();
  intK = BlockReduce(temp_storage).Sum(intK);
  __syncthreads();
  intV.xx = BlockReduce(temp_storage).Sum(intV.xx);
  __syncthreads();
  intV.xy = BlockReduce(temp_storage).Sum(intV.xy);
  __syncthreads();
  intV.xz = BlockReduce(temp_storage).Sum(intV.xz);
  __syncthreads();
  intV.yx = BlockReduce(temp_storage).Sum(intV.yx);
  __syncthreads();
  intV.yy = BlockReduce(temp_storage).Sum(intV.yy);
  __syncthreads();
  intV.yz = BlockReduce(temp_storage).Sum(intV.yz);
  __syncthreads();
  intV.zx = BlockReduce(temp_storage).Sum(intV.zx);
  __syncthreads();
  intV.zy = BlockReduce(temp_storage).Sum(intV.zy);
  __syncthreads();
  intV.zz = BlockReduce(temp_storage).Sum(intV.zz);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int bin = blockIdx.x % ATOMIC_BINS;

    atomicAdd(&kineticEnergy[bin], k);
    atomicAdd(&virial[bin].xx, v.xx);
    // atomicAdd(&virial[bin].xy, v.xy);
    // atomicAdd(&virial[bin].xz, v.xz);
    atomicAdd(&virial[bin].yx, v.yx);
    atomicAdd(&virial[bin].yy, v.yy);
    // atomicAdd(&virial[bin].yz, v.yz);
    atomicAdd(&virial[bin].zx, v.zx);
    atomicAdd(&virial[bin].zy, v.zy);
    atomicAdd(&virial[bin].zz, v.zz);
    atomicAdd(&intKineticEnergy[bin], intK);
    atomicAdd(&intVirialNormal[bin].xx, intV.xx);
    atomicAdd(&intVirialNormal[bin].xy, intV.xy);
    atomicAdd(&intVirialNormal[bin].xz, intV.xz);
    atomicAdd(&intVirialNormal[bin].yx, intV.yx);
    atomicAdd(&intVirialNormal[bin].yy, intV.yy);
    atomicAdd(&intVirialNormal[bin].yz, intV.yz);
    atomicAdd(&intVirialNormal[bin].zx, intV.zx);
    atomicAdd(&intVirialNormal[bin].zy, intV.zy);
    atomicAdd(&intVirialNormal[bin].zz, intV.zz);

    __threadfence();
    unsigned int value = atomicInc(&tbcatomic[1], totaltb);
    isLastBlockDone = (value == (totaltb -1));
  }
  __syncthreads();

#if 1
  if(isLastBlockDone){
    if(threadIdx.x < ATOMIC_BINS){
      const int bin = threadIdx.x;

      double k = kineticEnergy[bin];
      double intK = intKineticEnergy[bin];
      cudaTensor v = virial[bin];
      cudaTensor intV = intVirialNormal[bin];

      // sets device scalars back to zero
      kineticEnergy[bin] = 0.0;
      intKineticEnergy[bin] = 0.0;
      zero_cudaTensor(virial[bin]);
      zero_cudaTensor(intVirialNormal[bin]);

      if(ATOMIC_BINS > 1){
        typedef cub::WarpReduce<double, (ATOMIC_BINS > 1 ? ATOMIC_BINS : 2)> WarpReduce;
        typedef cub::WarpReduce<cudaTensor, (ATOMIC_BINS > 1 ? ATOMIC_BINS : 2)> WarpReduceT;
        __shared__ typename WarpReduce::TempStorage tempStorage;
        __shared__ typename WarpReduceT::TempStorage tempStorageT;

        k = WarpReduce(tempStorage).Sum(k);
        intK = WarpReduce(tempStorage).Sum(intK);
        v = WarpReduceT(tempStorageT).Sum(v);
        intV = WarpReduceT(tempStorageT).Sum(intV);
      }

      if(threadIdx.x == 0){
        h_kineticEnergy[0] = k;
        h_intKineticEnergy[0] = intK;
        h_virial[0] = v;
        h_intVirialNormal[0] = intV;

        //resets atomic counter
        reset_atomic_counter(&tbcatomic[1]);
      }
    }
  }
#endif
}

void SequencerCUDAKernel::submitHalf(
  const double *vel_x,
  const double *vel_y,
  const double *vel_z,
  const double *vcm_x,
  const double *vcm_y,
  const double *vcm_z,
  const float  *mass,
  BigReal *kineticEnergy,
  BigReal *intKineticEnergy,
  cudaTensor *virial,
  cudaTensor *intVirialNormal,
  BigReal *h_kineticEnergy,
  BigReal *h_intKineticEnergy,
  cudaTensor *h_virial,
  cudaTensor *h_intVirialNormal,
  int *hydrogenGroupSize,
  int numAtoms,
  unsigned int* tbcatomic,
  cudaStream_t stream)
{
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  submitHalfKernel<ATOM_BLOCKS><<<grid, ATOM_BLOCKS, 0, stream>>>(
    vel_x, vel_y, vel_z,
    vcm_x, vcm_y, vcm_z, mass,
    kineticEnergy, intKineticEnergy,
    virial, intVirialNormal, h_kineticEnergy, h_intKineticEnergy,
    h_virial, h_intVirialNormal,
    hydrogenGroupSize, numAtoms, tbcatomic);
  //cudaCheck(cudaGetLastError());
}

__global__ void scaleCoordinateUseGroupPressureKernel(
  double * __restrict pos_x,
  double * __restrict pos_y,
  double * __restrict pos_z,
  float *mass,
  int *hydrogenGroupSize,
  cudaTensor factor,
  cudaVector origin,
  int numAtoms)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < numAtoms) {
    int hgs = hydrogenGroupSize[i];
    int j;
    if (hgs > 0) {
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
      BigReal inv_m_cm = 1.0/m_cm;
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
      for (j = i;  j < (i+hgs);  j++) {
        pos_x[j] += delta_r_cm_x;
        pos_y[j] += delta_r_cm_y;
        pos_z[j] += delta_r_cm_z;
      }
    }
  }
}

__global__ void scaleCoordinateNoGroupPressureKernel(
  double * __restrict pos_x,
  double * __restrict pos_y,
  double * __restrict pos_z,
  cudaTensor factor,
  cudaVector origin,
  int numAtoms)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < numAtoms) {
    // missing fixed atoms implementation
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

__global__ void SetAtomIndexOrderKernel(
  int *id, 
  int *idOrder, 
  int numAtoms)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < numAtoms) {
    int atomIndex = id[i];
    idOrder[atomIndex] = i;
  }
}

__global__ void scaleCoordinateUsingGCKernel(
  double* __restrict pos_x, 
  double* __restrict pos_y, 
  double* __restrict pos_z, 
  const int* __restrict idOrder, 
  const int* __restrict moleculeStartIndex,
  const int* __restrict moleculeAtom, 
  const cudaTensor factor, 
  const cudaVector origin, 
  const Lattice oldLattice,
  const Lattice newLattice,
  const char3* __restrict transform,
  const int numMolecules,
  const int numLargeMolecules)
{
  // missing fixed atoms implementation
  int startIdx, endIdx, i, j, jmapped, atomIndex;
  double3 position, r_gc, new_r_gc, delta_r_gc;
  char3 tr;
  r_gc.x = 0; r_gc.y = 0; r_gc.z = 0;
  if (blockIdx.x < numLargeMolecules){ //large molecule case
    i = blockIdx.x;
    startIdx = moleculeStartIndex[i];
    endIdx = moleculeStartIndex[i + 1];
    __shared__ double3 sh_gc;
    double inv_length = 1.0/(double)(endIdx - startIdx);
    // calculate the geometric center
    for (j = startIdx + threadIdx.x; j < endIdx; j += blockDim.x) {
      atomIndex = moleculeAtom[j];
      jmapped = idOrder[atomIndex];
      tr = transform[jmapped];
      position.x = pos_x[jmapped];
      position.y = pos_y[jmapped];
      position.z = pos_z[jmapped];
      //Unwrap the coordinate with oldLattice
      position = oldLattice.reverse_transform(position ,tr);
      r_gc.x += position.x;
      r_gc.y += position.y;
      r_gc.z += position.z;
    }
    __syncthreads();
    typedef cub::BlockReduce<double, 64> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    r_gc.x = BlockReduce(temp_storage).Sum(r_gc.x);
    __syncthreads();
    r_gc.y = BlockReduce(temp_storage).Sum(r_gc.y);
    __syncthreads();
    r_gc.z = BlockReduce(temp_storage).Sum(r_gc.z);
    __syncthreads();

    if (threadIdx.x == 0) {   
      sh_gc.x = r_gc.x * inv_length;
      sh_gc.y = r_gc.y * inv_length;
      sh_gc.z = r_gc.z * inv_length;
    }
    __syncthreads();

    // scale the geometric center with factor 
    // shift to origin
    double tx = sh_gc.x - origin.x;
    double ty = sh_gc.y - origin.y;
    double tz = sh_gc.z - origin.z;
    // apply transformation 
    new_r_gc.x = factor.xx*tx + factor.xy*ty + factor.xz*tz;
    new_r_gc.y = factor.yx*tx + factor.yy*ty + factor.yz*tz;
    new_r_gc.z = factor.zx*tx + factor.zy*ty + factor.zz*tz;
    // shift back
    new_r_gc.x += origin.x;
    new_r_gc.y += origin.y;
    new_r_gc.z += origin.z;
    // translation vector from old GC to new GC
    delta_r_gc.x = new_r_gc.x - sh_gc.x;
    delta_r_gc.y = new_r_gc.y - sh_gc.y;
    delta_r_gc.z = new_r_gc.z - sh_gc.z;

    // shift the atoms in molecule with translation vector
    for (j = startIdx + threadIdx.x; j < endIdx; j += blockDim.x) {
      atomIndex = moleculeAtom[j];
      jmapped = idOrder[atomIndex];
      tr = transform[jmapped];
      position.x = pos_x[jmapped];
      position.y = pos_y[jmapped];
      position.z = pos_z[jmapped];
      //Unwrap the coordinate with oldLattice
      position = oldLattice.reverse_transform(position, tr);
      position.x += delta_r_gc.x;
      position.y += delta_r_gc.y;
      position.z += delta_r_gc.z;
      // wrap the coordinate with new lattice parameter
      position = newLattice.apply_transform(position, tr);
      pos_x[jmapped] = position.x;
      pos_y[jmapped] = position.y;
      pos_z[jmapped] = position.z;
    }
  } else { //Small molecule
    // numLargeMolecules block has been assigned to large molecule
    i = numLargeMolecules + threadIdx.x + 
        (blockIdx.x - numLargeMolecules) * blockDim.x;

    if (i < numMolecules) {
      startIdx = moleculeStartIndex[i];
      endIdx = moleculeStartIndex[i+1];
      double inv_length = 1.0/(double)(endIdx - startIdx);

      // calculate the geometric center
      for ( j = startIdx; j < endIdx; j++ ) {
        atomIndex = moleculeAtom[j];
        jmapped = idOrder[atomIndex];
        tr = transform[jmapped];
        position.x = pos_x[jmapped];
        position.y = pos_y[jmapped];
        position.z = pos_z[jmapped];
        //Unwrap the coordinate with oldLattice
        position = oldLattice.reverse_transform(position, tr);
        r_gc.x += position.x;
        r_gc.y += position.y;
        r_gc.z += position.z;
      }

      r_gc.x *= inv_length;
      r_gc.y *= inv_length;
      r_gc.z *= inv_length;

      // scale the geometric center with factor 
      // shift to origin
      double tx = r_gc.x - origin.x;
      double ty = r_gc.y - origin.y;
      double tz = r_gc.z - origin.z;
      // apply transformation 
      new_r_gc.x = factor.xx*tx + factor.xy*ty + factor.xz*tz;
      new_r_gc.y = factor.yx*tx + factor.yy*ty + factor.yz*tz;
      new_r_gc.z = factor.zx*tx + factor.zy*ty + factor.zz*tz;
      // shift back
      new_r_gc.x += origin.x;
      new_r_gc.y += origin.y;
      new_r_gc.z += origin.z;
      // translation vector from old GC to new GC
      delta_r_gc.x = new_r_gc.x - r_gc.x;
      delta_r_gc.y = new_r_gc.y - r_gc.y;
      delta_r_gc.z = new_r_gc.z - r_gc.z;

      // shift the atoms in molecule with translation vector
      for (j = startIdx; j < endIdx; j++) {
        atomIndex = moleculeAtom[j];
        jmapped = idOrder[atomIndex];
        
        tr = transform[jmapped];
        position.x = pos_x[jmapped];
        position.y = pos_y[jmapped];
        position.z = pos_z[jmapped];
        // unwrap the coordinate with oldLattice
        position = oldLattice.reverse_transform(position, tr);
        position.x += delta_r_gc.x;
        position.y += delta_r_gc.y;
        position.z += delta_r_gc.z;
        // wrap the coordinate with new lattice parameter
        position = newLattice.apply_transform(position, tr);
        pos_x[jmapped] = position.x;
        pos_y[jmapped] = position.y;
        pos_z[jmapped] = position.z;
      }
    }
  }
}

__global__ void langevinPistonUseGroupPressureKernel(
  double * __restrict pos_x,
  double * __restrict pos_y,
  double * __restrict pos_z,
  double * __restrict vel_x,
  double * __restrict vel_y,
  double * __restrict vel_z,
  float *mass,
  int *hydrogenGroupSize,
  cudaTensor factor,
  cudaVector origin,
  double velFactor_x,
  double velFactor_y,
  double velFactor_z,
  int numAtoms)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numAtoms) {
    int hgs = hydrogenGroupSize[i];
    int j;
    if (hgs > 0) {
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
      BigReal inv_m_cm = 1.0/m_cm;
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
    }

  }
}

__global__ void langevinPistonNoGroupPressureKernel(
  double * __restrict pos_x,
  double * __restrict pos_y,
  double * __restrict pos_z,
  double * __restrict vel_x,
  double * __restrict vel_y,
  double * __restrict vel_z,
  cudaTensor factor,
  cudaVector origin,
  double velFactor_x,
  double velFactor_y,
  double velFactor_z,
  int numAtoms)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numAtoms) {
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
  }
}

void SequencerCUDAKernel::scaleCoordinateWithFactor(
  double *pos_x,
  double *pos_y,
  double *pos_z,
  float *mass,
  int *hydrogenGroupSize,
  cudaTensor factor,
  cudaVector origin,
  int useGroupPressure,
  int numAtoms,
  cudaStream_t stream)
{
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;  
  if (useGroupPressure) {
    scaleCoordinateUseGroupPressureKernel<<<grid, ATOM_BLOCKS, 0, stream>>>(
      pos_x, pos_y, pos_z, mass, hydrogenGroupSize, factor, origin, numAtoms);
  } else {
    scaleCoordinateNoGroupPressureKernel<<<grid, ATOM_BLOCKS, 0, stream>>>(
      pos_x, pos_y, pos_z, factor, origin, numAtoms);
  }
}


void SequencerCUDAKernel::SetAtomIndexOrder(
  int *id, 
  int *idOrder,
  int numAtoms,
  cudaStream_t stream)
{
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  SetAtomIndexOrderKernel<<<grid, ATOM_BLOCKS, 0, stream>>>(
    id, idOrder, numAtoms);
}

void SequencerCUDAKernel::scaleCoordinateUsingGC(
  double *pos_x,
  double *pos_y,
  double *pos_z,
  const int *idOrder,
  const int *moleculeStartIndex,
  const int *moleculeAtom,
  const cudaTensor factor,
  const cudaVector origin,
  const Lattice oldLattice,
  const Lattice newLattice,
  const char3 *transform,
  const int numMolecules,
  const int numLargeMolecules,
  cudaStream_t stream)
{
 // we want each thread to calculate geometric center for molecule with 
 // less than 128 atoms, and 1 threadblock to calculate each molecule 
 // with larger than 128 atoms
  int numThreadsPerBlock = 64;
  int grid = ((numMolecules - numLargeMolecules + numThreadsPerBlock - 1) / 
    numThreadsPerBlock) + numLargeMolecules; 
  scaleCoordinateUsingGCKernel<<<grid, numThreadsPerBlock, 0, stream>>>(
    pos_x, pos_y, pos_z, idOrder, moleculeStartIndex,
    moleculeAtom, factor, origin, oldLattice, newLattice,
    transform, numMolecules, numLargeMolecules);
    
}

void SequencerCUDAKernel::langevinPiston(
  double *pos_x,
  double *pos_y,
  double *pos_z,
  double *vel_x,
  double *vel_y,
  double *vel_z,
  float *mass,
  int *hydrogenGroupSize,
  cudaTensor factor,
  cudaVector origin,
  double velFactor_x,
  double velFactor_y,
  double velFactor_z,
  int useGroupPressure,
  int numAtoms,
  cudaStream_t stream)
{
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  #if 0
    double *h_pos_x = (double*)malloc(sizeof(double)*numAtoms);
    double *h_pos_y = (double*)malloc(sizeof(double)*numAtoms);
    double *h_pos_z = (double*)malloc(sizeof(double)*numAtoms);

    double *h_vel_x = (double*)malloc(sizeof(double)*numAtoms);
    double *h_vel_y = (double*)malloc(sizeof(double)*numAtoms);
    double *h_vel_z = (double*)malloc(sizeof(double)*numAtoms);

    copy_DtoH_sync<double>(pos_x, h_pos_x, numAtoms);
    copy_DtoH_sync<double>(pos_y, h_pos_y, numAtoms);
    copy_DtoH_sync<double>(pos_z, h_pos_z, numAtoms);

    copy_DtoH_sync<double>(vel_x, h_vel_x, numAtoms);
    copy_DtoH_sync<double>(vel_y, h_vel_y, numAtoms);
    copy_DtoH_sync<double>(vel_z, h_vel_z, numAtoms);

    fprintf(stderr, "velFactors = %lf %lf %lf\n",
      velFactor_x, velFactor_y, velFactor_z);
    for(int i = 0; i < numAtoms; i++){
      fprintf(stderr, "%lf %lf %lf %lf %lf %lf\n",
        h_pos_x[i], h_pos_y[i], h_pos_z[i],
        h_vel_x[i], h_vel_y[i], h_vel_z[i]);
    }
    free(h_pos_x);
    free(h_pos_y);
    free(h_pos_z);
    free(h_vel_x);
    free(h_vel_y);
    free(h_vel_z);
  #endif

  if (useGroupPressure) {
    langevinPistonUseGroupPressureKernel<<<grid, ATOM_BLOCKS, 0, stream>>>(
      pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, hydrogenGroupSize,
      factor, origin, velFactor_x, velFactor_y, velFactor_z, numAtoms);
  } else {
    langevinPistonNoGroupPressureKernel<<<grid, ATOM_BLOCKS, 0, stream>>>(
      pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
      factor, origin, velFactor_x, velFactor_y, velFactor_z, numAtoms);
  }
  //cudaCheck(cudaGetLastError());
#if 0
  h_pos_x = (double*)malloc(sizeof(double)*numAtoms);
  h_pos_y = (double*)malloc(sizeof(double)*numAtoms);
  h_pos_z = (double*)malloc(sizeof(double)*numAtoms);

  h_vel_x = (double*)malloc(sizeof(double)*numAtoms);
  h_vel_y = (double*)malloc(sizeof(double)*numAtoms);
  h_vel_z = (double*)malloc(sizeof(double)*numAtoms);

  copy_DtoH_sync<double>(pos_x, h_pos_x, numAtoms);
  copy_DtoH_sync<double>(pos_y, h_pos_y, numAtoms);
  copy_DtoH_sync<double>(pos_z, h_pos_z, numAtoms);

  copy_DtoH_sync<double>(vel_x, h_vel_x, numAtoms);
  copy_DtoH_sync<double>(vel_y, h_vel_y, numAtoms);
  copy_DtoH_sync<double>(vel_z, h_vel_z, numAtoms);

  for(int i = 0; i < numAtoms; i++){
    fprintf(stderr, "%lf %lf %lf %lf %lf %lf\n",
      h_pos_x[i], h_pos_y[i], h_pos_z[i],
      h_vel_x[i], h_vel_y[i], h_vel_z[i]);
  }
#endif
}

template <int BLOCK_SIZE>
__global__ void submitReduction1Kernel(
  double * __restrict pos_x,
  double * __restrict pos_y,
  double * __restrict pos_z,
  double * __restrict vel_x,
  double * __restrict vel_y,
  double * __restrict vel_z,
  float  * __restrict mass,
  // TODO: wrap these scalars in a struct
  BigReal *kineticEnergy,
  BigReal *momentum_x,
  BigReal *momentum_y,
  BigReal *momentum_z,
  BigReal *angularMomentum_x,
  BigReal *angularMomentum_y,
  BigReal *angularMomentum_z,
  BigReal origin_x,
  BigReal origin_y,
  BigReal origin_z,
  BigReal *h_kineticEnergy,
  BigReal *h_momentum_x,
  BigReal *h_momentum_y,
  BigReal *h_momentum_z,
  BigReal *h_angularMomentum_x,
  BigReal *h_angularMomentum_y,
  BigReal *h_angularMomentum_z,
  unsigned int* tbcatomic,
  int numAtoms)
{
  BigReal m_x = 0, m_y = 0, m_z = 0,
    a_x = 0, a_y = 0, a_z = 0;
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int totaltb = gridDim.x;
  __shared__ bool isLastBlockDone;

 if(threadIdx.x == 0){
   isLastBlockDone = 0;
 }
 __syncthreads();

  if (i < numAtoms) {
    // scalar kineticEnergy += mass[i] * dot_product(vel[i], vel[i])
    // k += mass[i] *
    //  (vel_x[i]*vel_x[i] + vel_y[i]*vel_y[i] + vel_z[i]*vel_z[i]);

    // vector momentum += mass[i] * vel[i]
    m_x += mass[i] * vel_x[i];
    m_y += mass[i] * vel_y[i];
    m_z += mass[i] * vel_z[i];

    // vector dpos = pos[i] - origin
    BigReal dpos_x = pos_x[i] - origin_x;
    BigReal dpos_y = pos_y[i] - origin_y;
    BigReal dpos_z = pos_z[i] - origin_z;

    // vector angularMomentum += mass[i] * cross_product(dpos, vel[i])
    a_x += mass[i] * (dpos_y*vel_z[i] - dpos_z*vel_y[i]);
    a_y += mass[i] * (dpos_z*vel_x[i] - dpos_x*vel_z[i]);
    a_z += mass[i] * (dpos_x*vel_y[i] - dpos_y*vel_x[i]);
  }
  typedef cub::BlockReduce<BigReal, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  // k   = BlockReduce(temp_storage).Sum(k);
  // __syncthreads();

  m_x = BlockReduce(temp_storage).Sum(m_x);
  __syncthreads();
  m_y = BlockReduce(temp_storage).Sum(m_y);
  __syncthreads();
  m_z = BlockReduce(temp_storage).Sum(m_z);
  __syncthreads();
  a_x = BlockReduce(temp_storage).Sum(a_x);
  __syncthreads();
  a_y = BlockReduce(temp_storage).Sum(a_y);
  __syncthreads();
  a_z = BlockReduce(temp_storage).Sum(a_z);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int bin = blockIdx.x % ATOMIC_BINS;

    // atomicAdd(&kineticEnergy[bin], k);
    atomicAdd(&momentum_x[bin], m_x);
    atomicAdd(&momentum_y[bin], m_y);
    atomicAdd(&momentum_z[bin], m_z);
    atomicAdd(&angularMomentum_x[bin], a_x);
    atomicAdd(&angularMomentum_y[bin], a_y);
    atomicAdd(&angularMomentum_z[bin], a_z);
    __threadfence();
    unsigned int value = atomicInc(&tbcatomic[0], totaltb);
    isLastBlockDone = (value == (totaltb-1));
  }

  __syncthreads();

#if 1
  if(isLastBlockDone){
    if(threadIdx.x < ATOMIC_BINS){
      const int bin = threadIdx.x;

      double m_x = momentum_x[bin];
      double m_y = momentum_y[bin];
      double m_z = momentum_z[bin];
      double a_x = angularMomentum_x[bin];
      double a_y = angularMomentum_y[bin];
      double a_z = angularMomentum_z[bin];

      // sets device scalars back to zero
      kineticEnergy[0] = 0.0;
      momentum_x[bin] = 0.0;
      momentum_y[bin] = 0.0;
      momentum_z[bin] = 0.0;
      angularMomentum_x[bin] = 0.0;
      angularMomentum_y[bin] = 0.0;
      angularMomentum_z[bin] = 0.0;

      if(ATOMIC_BINS > 1){
        typedef cub::WarpReduce<double, (ATOMIC_BINS > 1 ? ATOMIC_BINS : 2)> WarpReduce;
        __shared__ typename WarpReduce::TempStorage tempStorage;

        m_x = WarpReduce(tempStorage).Sum(m_x);
        m_y = WarpReduce(tempStorage).Sum(m_y);
        m_z = WarpReduce(tempStorage).Sum(m_z);
        a_x = WarpReduce(tempStorage).Sum(a_x);
        a_y = WarpReduce(tempStorage).Sum(a_y);
        a_z = WarpReduce(tempStorage).Sum(a_z);
      }

      if(threadIdx.x == 0){
        h_momentum_x[0] = m_x;
        h_momentum_y[0] = m_y;
        h_momentum_z[0] = m_z;
        h_angularMomentum_x[0] = a_x;
        h_angularMomentum_y[0] = a_y;
        h_angularMomentum_z[0] = a_z;

        //resets atomic counter
        reset_atomic_counter(&tbcatomic[0]);
      }
    }
  }
#endif
}


// JM: does addForcetoMomentum, maximumMove, and addVelocityToPosition
template <bool DO_VEL_RESCALING>
__global__ void velocityVerlet1Kernel(
   const int    step,
   const double scaling,
   const double dt_normal,
   const double dt_nbond,
   const double dt_slow,
   const double velrescaling,   // for stochastic velocity rescaling
   const double* __restrict recipMass,
   double* __restrict  vel_x,
   double* __restrict  vel_y,
   double* __restrict  vel_z,
   const double maxvel2,
   int* h_killme,
   double* __restrict pos_x,
   double* __restrict pos_y,
   double* __restrict pos_z,
   double* h_pos_x,
   double* h_pos_y,
   double* h_pos_z,
   double* __restrict f_normal_x,
   double* __restrict f_normal_y,
   double* __restrict f_normal_z,
   double* __restrict f_nbond_x,
   double* __restrict f_nbond_y,
   double* __restrict f_nbond_z,
   double* __restrict f_slow_x,
   double* __restrict f_slow_y,
   double* __restrict f_slow_z,
   const int numAtoms,
   const int maxForceNumber
   ){
   // fusion of addForceToMomentum, maximumMove, addVelocityToPosition
  double dt, dt_b, dt_nb, dt_s;
  int i = threadIdx.x + blockIdx.x*blockDim.x;

  if(i < numAtoms){
    dt = dt_normal;
    double velx = vel_x[i];
    double vely = vel_y[i];
    double velz = vel_z[i];
    if (DO_VEL_RESCALING) {
      velx *= velrescaling;
      vely *= velrescaling;
      velz *= velrescaling;
    }
    // JM NOTE: these need to be patch-centered
    double posx = pos_x[i];
    double posy = pos_y[i];
    double posz = pos_z[i];
    double rmass = recipMass[i];
    /* addForceToMomentum*/
    // keep velocities in registers so I can access them faster when calculating positions!
    switch(maxForceNumber){
      case 2:
        dt_s = dt_slow * scaling;
        velx += f_slow_x[i] * rmass * dt_s;
        vely += f_slow_y[i] * rmass * dt_s;
        velz += f_slow_z[i] * rmass * dt_s;
        // f_slow_x[i] = 0.0;
        // f_slow_y[i] = 0.0;
        // f_slow_z[i] = 0.0;
      case 1:
        dt_nb = dt_nbond * scaling;
        velx += f_nbond_x[i] * rmass * dt_nb;
        vely += f_nbond_y[i] * rmass * dt_nb;
        velz += f_nbond_z[i] * rmass * dt_nb;
        // f_nbond_x[i] = 0.0;
        // f_nbond_y[i] = 0.0;
        // f_nbond_z[i] = 0.0;
      case 0:
        dt_b = dt_normal * scaling;
        velx += f_normal_x[i] * rmass * dt_b;
        vely += f_normal_y[i] * rmass * dt_b;
        velz += f_normal_z[i] * rmass * dt_b;
        // f_normal_x[i] = 0.0;
        // f_normal_y[i] = 0.0;
        // f_normal_z[i] = 0.0;
    }

    // --------------------------------------------------------

    // -- MaximumMove --
    double vel2 = velx * velx + vely * vely + velz * velz;
    if(vel2 > maxvel2) atomicAdd(h_killme, 1); // stops dynamics if true, perf does not matter
    // --------------------------------------------------------

    // -- AddVelocityToPosition --

    posx += velx * dt;
    posy += vely * dt;
    posz += velz * dt;
    // ---------------------------------------------------------
    pos_x[i] = posx;
    pos_y[i] = posy;
    pos_z[i] = posz;
    vel_x[i] = velx;
    vel_y[i] = vely;
    vel_z[i] = velz;
  }

}

void SequencerCUDAKernel::velocityVerlet1(
    const int    step,
    const double scaling,
    const double dt_normal,
    const double dt_nbond,
    const double dt_slow,
    const double velrescaling,  // for stochastic velocity rescaling
    const double *recipMass,
    double*  vel_x,
    double*  vel_y,
    double*  vel_z,
    double maxvel2,
    int* h_killme,
    double* pos_x,
    double* pos_y,
    double* pos_z,
    double* h_pos_x,
    double* h_pos_y,
    double* h_pos_z,
    double* f_normal_x,
    double* f_normal_y,
    double* f_normal_z,
    double* f_nbond_x,
    double* f_nbond_y,
    double* f_nbond_z,
    double* f_slow_x,
    double* f_slow_y,
    double* f_slow_z,
    const int numAtoms,
    const int maxForceNumber,
    cudaStream_t stream)
{
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  if (velrescaling != 1) {
    // apply velocity rescaling
    velocityVerlet1Kernel<true><<<grid, ATOM_BLOCKS, 0, stream>>>(step, scaling,
        dt_normal, dt_nbond, dt_slow, velrescaling, recipMass,
        vel_x, vel_y, vel_z, maxvel2, h_killme,
        pos_x, pos_y, pos_z, h_pos_x, h_pos_y, h_pos_z,
        f_normal_x, f_normal_y, f_normal_z,
        f_nbond_x, f_nbond_y, f_nbond_z, f_slow_x, f_slow_y, f_slow_z,
        numAtoms, maxForceNumber);
  }
  else {
    // NO velocity rescaling
    velocityVerlet1Kernel<false><<<grid, ATOM_BLOCKS, 0, stream>>>(step, scaling,
        dt_normal, dt_nbond, dt_slow, velrescaling, recipMass,
        vel_x, vel_y, vel_z, maxvel2, h_killme,
        pos_x, pos_y, pos_z, h_pos_x, h_pos_y, h_pos_z,
        f_normal_x, f_normal_y, f_normal_z,
        f_nbond_x, f_nbond_y, f_nbond_z, f_slow_x, f_slow_y, f_slow_z,
        numAtoms, maxForceNumber);
  }
}

void SequencerCUDAKernel::submitReduction1(
  double *pos_x,
  double *pos_y,
  double *pos_z,
  double *vel_x,
  double *vel_y,
  double *vel_z,
  float  *mass,
  BigReal *kineticEnergy,
  BigReal *momentum_x,
  BigReal *momentum_y,
  BigReal *momentum_z,
  BigReal *angularMomentum_x,
  BigReal *angularMomentum_y,
  BigReal *angularMomentum_z,
  BigReal origin_x,
  BigReal origin_y,
  BigReal origin_z,
  BigReal *h_kineticEnergy,
  BigReal *h_momentum_x,
  BigReal *h_momentum_y,
  BigReal *h_momentum_z,
  BigReal *h_angularMomentum_x,
  BigReal *h_angularMomentum_y,
  BigReal *h_angularMomentum_z,
  unsigned int* tbcatomic,
  int numAtoms,
  cudaStream_t stream)
{
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  submitReduction1Kernel<ATOM_BLOCKS><<<grid, ATOM_BLOCKS, 0, stream>>>(
    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass,
    kineticEnergy, momentum_x, momentum_y, momentum_z,
    angularMomentum_x, angularMomentum_y, angularMomentum_z,
    origin_x, origin_y, origin_z, h_kineticEnergy, h_momentum_x, h_momentum_y,
    h_momentum_z, h_angularMomentum_x, h_angularMomentum_y, h_angularMomentum_z,
    tbcatomic, numAtoms);
  //cudaCheck(cudaGetLastError());
}

template <int BLOCK_SIZE>
__global__ void submitReduction2Kernel(
  const double * __restrict pos_x,
  const double * __restrict pos_y,
  const double * __restrict pos_z,
  const double * __restrict vel_x,
  const double * __restrict vel_y,
  const double * __restrict vel_z,
  const double * __restrict rcm_x,
  const double * __restrict rcm_y,
  const double * __restrict rcm_z,
  const double * __restrict vcm_x,
  const double * __restrict vcm_y,
  const double * __restrict vcm_z,
  const double * __restrict f_normal_x,
  const double * __restrict f_normal_y,
  const double * __restrict f_normal_z,
  const double * __restrict f_nbond_x,
  const double * __restrict f_nbond_y,
  const double * __restrict f_nbond_z,
  const double * __restrict f_slow_x,
  const double * __restrict f_slow_y,
  const double * __restrict f_slow_z,
  const float  * __restrict mass,
  const int * __restrict hydrogenGroupSize,
  BigReal    *kineticEnergy,
  BigReal    *h_kineticEnergy,
  BigReal    *intKineticEnergy,
  BigReal    *h_intKineticEnergy,
  cudaTensor *intVirialNormal,
  cudaTensor *intVirialNbond,
  cudaTensor *intVirialSlow,
  cudaTensor *h_intVirialNormal,
  cudaTensor *h_intVirialNbond,
  cudaTensor *h_intVirialSlow,
  cudaTensor* rigidVirial,
  cudaTensor* h_rigidVirial,
  unsigned int *tbcatomic,
  const int numAtoms,
  const int maxForceNumber)
{
  BigReal K = 0;
  BigReal intK = 0;
  cudaTensor intVNormal, intVNbond, intVSlow;
  zero_cudaTensor(intVNormal);
  zero_cudaTensor(intVNbond);
  zero_cudaTensor(intVSlow);
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  __shared__ bool isLastBlockDone;
  int totaltb = gridDim.x;

  if(threadIdx.x == 0){
    isLastBlockDone = false;
  }
  __syncthreads();

  if (i < numAtoms) {
#if 0
    int hgs = hydrogenGroupSize[i];
    // let's see, I have the hydrogenGroupSize, but that's not what I need;
    if (hgs > 0) {
      int j;
      BigReal m_cm = 0;
      BigReal r_cm_x = 0;
      BigReal r_cm_y = 0;
      BigReal r_cm_z = 0;
      BigReal v_cm_x = 0;
      BigReal v_cm_y = 0;
      BigReal v_cm_z = 0;
      for ( j = i; j < (i+hgs); ++j ) {
        r_mass[j -i ] = mass[j];
        r_pos[j - i].x  = pos_x[j];
        r_pos[j - i].y  = pos_y[j];
        r_pos[j - i].z  = pos_z[j];
        r_vel[j - i].x  = vel_x[j];
        r_vel[j - i].y  = vel_y[j];
        r_vel[j - i].z  = vel_z[j];
        // m_cm += mass[j];
        // r_cm_x += mass[j] * pos_x[j];
        // r_cm_y += mass[j] * pos_y[j];
        // r_cm_z += mass[j] * pos_z[j];
        // v_cm_x += mass[j] * vel_x[j];
        // v_cm_y += mass[j] * vel_y[j];
        // v_cm_z += mass[j] * vel_z[j];
        m_cm += r_mass[j - i];
        r_cm_x += r_mass[j - i] * r_pos[j-i].x;
        r_cm_y += r_mass[j - i] * r_pos[j-i].y;
        r_cm_z += r_mass[j - i] * r_pos[j-i].z;
        v_cm_x += r_mass[j - i] * r_vel[j-i].x;
        v_cm_y += r_mass[j - i] * r_vel[j-i].y;
        v_cm_z += r_mass[j - i] * r_vel[j-i].z;
      }
      BigReal inv_m_cm = 1.0/m_cm;
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
        BigReal v_x = r_vel[j-i].x;
        BigReal v_y = r_vel[j-i].y;
        BigReal v_z = r_vel[j-i].z;

        // vector dv = vel[j] - v_cm
        BigReal dv_x = v_x - v_cm_x;
        BigReal dv_y = v_y - v_cm_y;
        BigReal dv_z = v_z - v_cm_z;

        // scalar intKineticEnergy += mass[j] * dot_product(v, dv)
        //intK += mass[j] *
        //   (v_x * dv_x + v_y * dv_y + v_z * dv_z);
        intK += r_mass[j-i] *
             (v_x * dv_x + v_y * dv_y + v_z * dv_z);

        // vector dr = pos[j] - r_cm
        // BigReal dr_x = pos_x[j] - r_cm_x;
        // BigReal dr_y = pos_y[j] - r_cm_y;
        // BigReal dr_z = pos_z[j] - r_cm_z;

        BigReal dr_x = r_pos[j -i].x - r_cm_x;
        BigReal dr_y = r_pos[j -i].y - r_cm_y;
        BigReal dr_z = r_pos[j -i].z - r_cm_z;

        // tensor intVirialNormal += outer_product(f_normal[j], dr)

        // we're not going to make this function any faster if we don't fix
        // the global memory access pattern
        intVNormal.xx += f_normal_x[j] * dr_x;
        intVNormal.xy += f_normal_x[j] * dr_y;
        intVNormal.xz += f_normal_x[j] * dr_z;
        intVNormal.yx += f_normal_y[j] * dr_x;
        intVNormal.yy += f_normal_y[j] * dr_y;
        intVNormal.yz += f_normal_y[j] * dr_z;
        intVNormal.zx += f_normal_z[j] * dr_x;
        intVNormal.zy += f_normal_z[j] * dr_y;
        intVNormal.zz += f_normal_z[j] * dr_z;

        if (maxForceNumber >= 1) {
          // tensor intVirialNbond += outer_product(f_nbond[j], dr)
          intVNbond.xx += f_nbond_x[j] * dr_x;
          intVNbond.xy += f_nbond_x[j] * dr_y;
          intVNbond.xz += f_nbond_x[j] * dr_z;
          intVNbond.yx += f_nbond_y[j] * dr_x;
          intVNbond.yy += f_nbond_y[j] * dr_y;
          intVNbond.yz += f_nbond_y[j] * dr_z;
          intVNbond.zx += f_nbond_z[j] * dr_x;
          intVNbond.zy += f_nbond_z[j] * dr_y;
          intVNbond.zz += f_nbond_z[j] * dr_z;
        }

        if (maxForceNumber >= 2) {
          // tensor intVirialSlow += outer_product(f_slow[j], dr)
          intVSlow.xx += f_slow_x[j] * dr_x;
          intVSlow.xy += f_slow_x[j] * dr_y;
          intVSlow.xz += f_slow_x[j] * dr_z;
          intVSlow.yx += f_slow_y[j] * dr_x;
          intVSlow.yy += f_slow_y[j] * dr_y;
          intVSlow.yz += f_slow_y[j] * dr_z;
          intVSlow.zx += f_slow_z[j] * dr_x;
          intVSlow.zy += f_slow_z[j] * dr_y;
          intVSlow.zz += f_slow_z[j] * dr_z;
        }
      }
    }
#else
    BigReal r_cm_x = rcm_x[i];
    BigReal r_cm_y = rcm_y[i];
    BigReal r_cm_z = rcm_z[i];
    BigReal v_cm_x = vcm_x[i];
    BigReal v_cm_y = vcm_y[i];
    BigReal v_cm_z = vcm_z[i];

    BigReal v_x = vel_x[i];
    BigReal v_y = vel_y[i];
    BigReal v_z = vel_z[i];

    // vector dv = vel[j] - v_cm
    BigReal dv_x = v_x - v_cm_x;
    BigReal dv_y = v_y - v_cm_y;
    BigReal dv_z = v_z - v_cm_z;

    // scalar intKineticEnergy += mass[j] * dot_product(v, dv)
    //intK += mass[j] *
    //   (v_x * dv_x + v_y * dv_y + v_z * dv_z);
    K += mass[i] *
      (v_x * v_x + v_y*v_y + v_z*v_z);
    intK += mass[i] *
      (v_x * dv_x + v_y * dv_y + v_z * dv_z);

    // vector dr = pos[j] - r_cm
    // BigReal dr_x = pos_x[j] - r_cm_x;
    // BigReal dr_y = pos_y[j] - r_cm_y;
    // BigReal dr_z = pos_z[j] - r_cm_z;

    BigReal dr_x = pos_x[i] - r_cm_x;
    BigReal dr_y = pos_y[i] - r_cm_y;
    BigReal dr_z = pos_z[i] - r_cm_z;

    // tensor intVirialNormal += outer_product(f_normal[j], dr)

    // we're not going to make this function any faster if we don't fix
    // the global memory access pattern
    intVNormal.xx += f_normal_x[i] * dr_x;
    intVNormal.xy += f_normal_x[i] * dr_y;
    intVNormal.xz += f_normal_x[i] * dr_z;
    intVNormal.yx += f_normal_y[i] * dr_x;
    intVNormal.yy += f_normal_y[i] * dr_y;
    intVNormal.yz += f_normal_y[i] * dr_z;
    intVNormal.zx += f_normal_z[i] * dr_x;
    intVNormal.zy += f_normal_z[i] * dr_y;
    intVNormal.zz += f_normal_z[i] * dr_z;

    if (maxForceNumber >= 1) {
      // tensor intVirialNbond += outer_product(f_nbond[j], dr)
#if 0
      intVNbond.xx += f_nbond_x[i] * dr_x;
      intVNbond.xy += f_nbond_x[i] * dr_y;
      intVNbond.xz += f_nbond_x[i] * dr_z;
      intVNbond.yx += f_nbond_y[i] * dr_x;
      intVNbond.yy += f_nbond_y[i] * dr_y;
      intVNbond.yz += f_nbond_y[i] * dr_z;
      intVNbond.zx += f_nbond_z[i] * dr_x;
      intVNbond.zy += f_nbond_z[i] * dr_y;
      intVNbond.zz += f_nbond_z[i] * dr_z;
#else
      intVNormal.xx += f_nbond_x[i] * dr_x;
      intVNormal.xy += f_nbond_x[i] * dr_y;
      intVNormal.xz += f_nbond_x[i] * dr_z;
      intVNormal.yx += f_nbond_y[i] * dr_x;
      intVNormal.yy += f_nbond_y[i] * dr_y;
      intVNormal.yz += f_nbond_y[i] * dr_z;
      intVNormal.zx += f_nbond_z[i] * dr_x;
      intVNormal.zy += f_nbond_z[i] * dr_y;
      intVNormal.zz += f_nbond_z[i] * dr_z;
#endif
    }

    if (maxForceNumber >= 2) {
      // tensor intVirialSlow += outer_product(f_slow[j], dr)
#if 0
      intVSlow.xx += f_slow_x[i] * dr_x;
      intVSlow.xy += f_slow_x[i] * dr_y;
      intVSlow.xz += f_slow_x[i] * dr_z;
      intVSlow.yx += f_slow_y[i] * dr_x;
      intVSlow.yy += f_slow_y[i] * dr_y;
      intVSlow.yz += f_slow_y[i] * dr_z;
      intVSlow.zx += f_slow_z[i] * dr_x;
      intVSlow.zy += f_slow_z[i] * dr_y;
      intVSlow.zz += f_slow_z[i] * dr_z;
#else
      intVNormal.xx += f_slow_x[i] * dr_x;
      intVNormal.xy += f_slow_x[i] * dr_y;
      intVNormal.xz += f_slow_x[i] * dr_z;
      intVNormal.yx += f_slow_y[i] * dr_x;
      intVNormal.yy += f_slow_y[i] * dr_y;
      intVNormal.yz += f_slow_y[i] * dr_z;
      intVNormal.zx += f_slow_z[i] * dr_x;
      intVNormal.zy += f_slow_z[i] * dr_y;
      intVNormal.zz += f_slow_z[i] * dr_z;
#endif
    }
#endif
  }

  typedef cub::BlockReduce<BigReal, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  // XXX TODO: If we handwrite these reductions, it might avoid a
  // lot of overhead from launching CUB

  K = BlockReduce(temp_storage).Sum(K);
  __syncthreads();
  intK = BlockReduce(temp_storage).Sum(intK);
  __syncthreads();
  intVNormal.xx = BlockReduce(temp_storage).Sum(intVNormal.xx);
  __syncthreads();
  intVNormal.xy = BlockReduce(temp_storage).Sum(intVNormal.xy);
  __syncthreads();
  intVNormal.xz = BlockReduce(temp_storage).Sum(intVNormal.xz);
  __syncthreads();
  intVNormal.yx = BlockReduce(temp_storage).Sum(intVNormal.yx);
  __syncthreads();
  intVNormal.yy = BlockReduce(temp_storage).Sum(intVNormal.yy);
  __syncthreads();
  intVNormal.yz = BlockReduce(temp_storage).Sum(intVNormal.yz);
  __syncthreads();
  intVNormal.zx = BlockReduce(temp_storage).Sum(intVNormal.zx);
  __syncthreads();
  intVNormal.zy = BlockReduce(temp_storage).Sum(intVNormal.zy);
  __syncthreads();
  intVNormal.zz = BlockReduce(temp_storage).Sum(intVNormal.zz);
  __syncthreads();
#if 0
  if (maxForceNumber >= 1) {
    intVNbond.xx = BlockReduce(temp_storage).Sum(intVNbond.xx);
    __syncthreads();
    intVNbond.xy = BlockReduce(temp_storage).Sum(intVNbond.xy);
    __syncthreads();
    intVNbond.xz = BlockReduce(temp_storage).Sum(intVNbond.xz);
    __syncthreads();
    intVNbond.yx = BlockReduce(temp_storage).Sum(intVNbond.yx);
    __syncthreads();
    intVNbond.yy = BlockReduce(temp_storage).Sum(intVNbond.yy);
    __syncthreads();
    intVNbond.yz = BlockReduce(temp_storage).Sum(intVNbond.yz);
    __syncthreads();
    intVNbond.zx = BlockReduce(temp_storage).Sum(intVNbond.zx);
    __syncthreads();
    intVNbond.zy = BlockReduce(temp_storage).Sum(intVNbond.zy);
    __syncthreads();
    intVNbond.zz = BlockReduce(temp_storage).Sum(intVNbond.zz);
    __syncthreads();
  }
  if (maxForceNumber >= 2) {
    intVSlow.xx = BlockReduce(temp_storage).Sum(intVSlow.xx);
    __syncthreads();
    intVSlow.xy = BlockReduce(temp_storage).Sum(intVSlow.xy);
    __syncthreads();
    intVSlow.xz = BlockReduce(temp_storage).Sum(intVSlow.xz);
    __syncthreads();
    intVSlow.yx = BlockReduce(temp_storage).Sum(intVSlow.yx);
    __syncthreads();
    intVSlow.yy = BlockReduce(temp_storage).Sum(intVSlow.yy);
    __syncthreads();
    intVSlow.yz = BlockReduce(temp_storage).Sum(intVSlow.yz);
    __syncthreads();
    intVSlow.zx = BlockReduce(temp_storage).Sum(intVSlow.zx);
    __syncthreads();
    intVSlow.zy = BlockReduce(temp_storage).Sum(intVSlow.zy);
    __syncthreads();
    intVSlow.zz = BlockReduce(temp_storage).Sum(intVSlow.zz);
    __syncthreads();
  }
#endif


  if (threadIdx.x == 0) {
    const int bin = blockIdx.x % ATOMIC_BINS;

    atomicAdd(&kineticEnergy[bin], K);
    atomicAdd(&intKineticEnergy[bin], intK);
    atomicAdd(&intVirialNormal[bin].xx, intVNormal.xx);
    atomicAdd(&intVirialNormal[bin].xy, intVNormal.xy);
    atomicAdd(&intVirialNormal[bin].xz, intVNormal.xz);
    atomicAdd(&intVirialNormal[bin].yx, intVNormal.yx);
    atomicAdd(&intVirialNormal[bin].yy, intVNormal.yy);
    atomicAdd(&intVirialNormal[bin].yz, intVNormal.yz);
    atomicAdd(&intVirialNormal[bin].zx, intVNormal.zx);
    atomicAdd(&intVirialNormal[bin].zy, intVNormal.zy);
    atomicAdd(&intVirialNormal[bin].zz, intVNormal.zz);
#if 0
    if (maxForceNumber >= 1) {
      atomicAdd(&intVirialNbond[bin].xx, intVNbond.xx);
      atomicAdd(&intVirialNbond[bin].xy, intVNbond.xy);
      atomicAdd(&intVirialNbond[bin].xz, intVNbond.xz);
      atomicAdd(&intVirialNbond[bin].yx, intVNbond.yx);
      atomicAdd(&intVirialNbond[bin].yy, intVNbond.yy);
      atomicAdd(&intVirialNbond[bin].yz, intVNbond.yz);
      atomicAdd(&intVirialNbond[bin].zx, intVNbond.zx);
      atomicAdd(&intVirialNbond[bin].zy, intVNbond.zy);
      atomicAdd(&intVirialNbond[bin].zz, intVNbond.zz);
    }
    if (maxForceNumber >= 2) {
      atomicAdd(&intVirialSlow[bin].xx, intVSlow.xx);
      atomicAdd(&intVirialSlow[bin].xy, intVSlow.xy);
      atomicAdd(&intVirialSlow[bin].xz, intVSlow.xz);
      atomicAdd(&intVirialSlow[bin].yx, intVSlow.yx);
      atomicAdd(&intVirialSlow[bin].yy, intVSlow.yy);
      atomicAdd(&intVirialSlow[bin].yz, intVSlow.yz);
      atomicAdd(&intVirialSlow[bin].zx, intVSlow.zx);
      atomicAdd(&intVirialSlow[bin].zy, intVSlow.zy);
      atomicAdd(&intVirialSlow[bin].zz, intVSlow.zz);
    }
#endif
    __threadfence();
    unsigned int value = atomicInc(&tbcatomic[3], totaltb);
    isLastBlockDone = (value == (totaltb -1 ));
  }
  __syncthreads();
  // this function calculates the internal pressures and is a huge bottleneck if we
  // do this host-mapped update
  // How do I know if this is really the bottleneck?
  if(isLastBlockDone){
    if(threadIdx.x < ATOMIC_BINS){
      const int bin = threadIdx.x;

      double k = kineticEnergy[bin];
      double intK = intKineticEnergy[bin];
      cudaTensor intVNormal = intVirialNormal[bin];
#if 0
      cudaTensor intVNbond, intVSlow;
      if (maxForceNumber >= 1) {
        intVNbond = intVirialNbond[bin];
      }
      if (maxForceNumber >= 2) {
        intVSlow = intVirialSlow[bin];
      }
#endif
      cudaTensor rigidV = rigidVirial[bin];

      // sets device scalars back to zero
      kineticEnergy[bin] = 0.0;
      intKineticEnergy[bin] = 0.0;
      zero_cudaTensor(intVirialNormal[bin]);
      zero_cudaTensor(intVirialNbond[bin]);
      zero_cudaTensor(intVirialSlow[bin]);
      zero_cudaTensor(rigidVirial[bin]);

      if(ATOMIC_BINS > 1){
        typedef cub::WarpReduce<double, (ATOMIC_BINS > 1 ? ATOMIC_BINS : 2)> WarpReduce;
        typedef cub::WarpReduce<cudaTensor, (ATOMIC_BINS > 1 ? ATOMIC_BINS : 2)> WarpReduceT;
        __shared__ typename WarpReduce::TempStorage tempStorage;
        __shared__ typename WarpReduceT::TempStorage tempStorageT;

        k = WarpReduce(tempStorage).Sum(k);
        intK = WarpReduce(tempStorage).Sum(intK);
        intVNormal = WarpReduceT(tempStorageT).Sum(intVNormal);
        rigidV = WarpReduceT(tempStorageT).Sum(rigidV);
#if 0
        if (maxForceNumber >= 1) {
          intVNbond = WarpReduceT(tempStorageT).Sum(intVNbond);
        }
        if (maxForceNumber >= 2) {
          intVSlow = WarpReduceT(tempStorageT).Sum(intVSlow);
        }
#endif
      }

      if(threadIdx.x == 0){
        h_kineticEnergy[0] = k;
        h_intKineticEnergy[0] = intK;
        h_intVirialNormal[0] = intVNormal;
#if 0
        if (maxForceNumber >= 1) {
          h_intVirialNbond[0] = intVNbond;
        }
        if (maxForceNumber >= 2) {
          h_intVirialSlow[0] = intVSlow;
        }
#endif
        h_rigidVirial[0] = rigidV;

        //resets atomic counter
        reset_atomic_counter(&tbcatomic[3]);
      }
    }
  }
}

void SequencerCUDAKernel::submitReduction2(
  const double *pos_x,
  const double *pos_y,
  const double *pos_z,
  const double *vel_x,
  const double *vel_y,
  const double *vel_z,
  const double *rcm_x,
  const double *rcm_y,
  const double *rcm_z,
  const double *vcm_x,
  const double *vcm_y,
  const double *vcm_z,
  const double *f_normal_x,
  const double *f_normal_y,
  const double *f_normal_z,
  const double *f_nbond_x,
  const double *f_nbond_y,
  const double *f_nbond_z,
  const double *f_slow_x,
  const double *f_slow_y,
  const double *f_slow_z,
  float  *mass,
  int *hydrogenGroupSize,
  BigReal *kineticEnergy,
  BigReal *h_kineticEnergy,
  BigReal *intKineticEnergy,
  BigReal *h_intKineticEnergy,
  cudaTensor *intVirialNormal,
  cudaTensor *intVirialNbond,
  cudaTensor *intVirialSlow,
  cudaTensor *h_intVirialNormal,
  cudaTensor *h_intVirialNbond,
  cudaTensor *h_intVirialSlow,
  cudaTensor* rigidVirial,
  cudaTensor* h_rigidVirial,
  unsigned int *tbcatomic,
  int numAtoms,
  int maxForceNumber,
  cudaStream_t stream)
{

  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  submitReduction2Kernel<ATOM_BLOCKS><<<grid, ATOM_BLOCKS, 0, stream>>>(
    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
    rcm_x, rcm_y, rcm_z, vcm_x, vcm_y, vcm_z,
    f_normal_x, f_normal_y, f_normal_z,
    f_nbond_x, f_nbond_y, f_nbond_z,
    f_slow_x, f_slow_y, f_slow_z,
    mass, hydrogenGroupSize,
    kineticEnergy, h_kineticEnergy,
    intKineticEnergy, h_intKineticEnergy,
    intVirialNormal, intVirialNbond, intVirialSlow,
    h_intVirialNormal, h_intVirialNbond, h_intVirialSlow, rigidVirial,
    h_rigidVirial, tbcatomic, numAtoms, maxForceNumber);
  // cudaCheck(cudaGetLastError());
}

__global__ void langevinVelocitiesBBK1Kernel(
  BigReal timestep,
  const float * __restrict langevinParam,
  double      * __restrict vel_x,
  double      * __restrict vel_y,
  double      * __restrict vel_z,
  int numAtoms)
{
  BigReal dt = timestep * (0.001 * TIMEFACTOR);
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numAtoms) {
    BigReal dt_gamma = dt * langevinParam[i];
    BigReal scaling = 1. - 0.5 * dt_gamma;
    vel_x[i] *= scaling;
    vel_y[i] *= scaling;
    vel_z[i] *= scaling;
  }
}

void SequencerCUDAKernel::langevinVelocitiesBBK1(
  BigReal timestep,
  const float *langevinParam,
  double      *vel_x,
  double      *vel_y,
  double      *vel_z,
  int numAtoms,
  cudaStream_t stream)
{
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  langevinVelocitiesBBK1Kernel<<<grid, ATOM_BLOCKS, 0, stream>>>(
    timestep, langevinParam, vel_x, vel_y, vel_z, numAtoms);
  // cudaCheck(cudaGetLastError());
}

__global__ void langevinVelocitiesBBK2Kernel(
  BigReal timestep,
  const float * __restrict langScalVelBBK2,
  const float * __restrict langScalRandBBK2,
  const float * __restrict gaussrand_x,
  const float * __restrict gaussrand_y,
  const float * __restrict gaussrand_z,
  double * __restrict vel_x,
  double * __restrict vel_y,
  double * __restrict vel_z,
  const int numAtoms)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numAtoms) {
    vel_x[i] += gaussrand_x[i] * langScalRandBBK2[i];
    vel_y[i] += gaussrand_y[i] * langScalRandBBK2[i];
    vel_z[i] += gaussrand_z[i] * langScalRandBBK2[i];
    vel_x[i] *= langScalVelBBK2[i];
    vel_y[i] *= langScalVelBBK2[i];
    vel_z[i] *= langScalVelBBK2[i];
  }
}



void SequencerCUDAKernel::langevinVelocitiesBBK2(
  BigReal timestep,
  const float *langScalVelBBK2,
  const float *langScalRandBBK2,
  float *gaussrand_x,
  float *gaussrand_y,
  float *gaussrand_z,
  double *vel_x,
  double *vel_y,
  double *vel_z,
  const int numAtoms,
  const int numAtomsGlobal,
  const int stride,
  curandGenerator_t gen,
  cudaStream_t stream)
{

  // For some reason, if n == nAtomsDevice + 1, this gives me a misaligned address
  // Generating the full array on gaussrand without striding for multiple GPU simulations for now

  // Adding missing call to hiprandSetStream  to set the current stream for HIPRAND kernel launches
  curandCheck(curandSetStream(gen, stream));
  // array buffers have to be even length for pseudorandom normal distribution
  // choose n to be the smallest even number >= numAtoms
  // we can choose 1 larger than numAtoms since allocation is > numAtoms
  int n = (numAtomsGlobal + 1) & (~1);

  curandCheck(curandGenerateNormal(gen, gaussrand_x, n, 0, 1));
  curandCheck(curandGenerateNormal(gen, gaussrand_y, n, 0, 1));
  curandCheck(curandGenerateNormal(gen, gaussrand_z, n, 0, 1));
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  langevinVelocitiesBBK2Kernel<<<grid, ATOM_BLOCKS, 0, stream>>>(
    timestep, langScalVelBBK2, langScalRandBBK2,
    gaussrand_x + stride, gaussrand_y + stride, gaussrand_z + stride,
    vel_x, vel_y, vel_z, numAtoms);
  // cudaCheck(cudaGetLastError());
}

__global__ void reassignVelocitiesKernel(
  const BigReal timestep,
  const float * __restrict gaussrand_x,
  const float * __restrict gaussrand_y,
  const float * __restrict gaussrand_z,
  double * __restrict vel_x,
  double * __restrict vel_y,
  double * __restrict vel_z,
  const double* __restrict d_recipMass,
  const BigReal kbT,
  const int numAtoms)
{
  //! note we are NOT supporting LES (locally enhanced sampling) here
  //! hence no switch on the partition to apply tempfactor

  /* */
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < numAtoms) {
    vel_x[i]=gaussrand_x[i]*sqrt(kbT*d_recipMass[i]);
    vel_y[i]=gaussrand_y[i]*sqrt(kbT*d_recipMass[i]);
    vel_z[i]=gaussrand_z[i]*sqrt(kbT*d_recipMass[i]);
  }
  //__syncthreads();
}

void SequencerCUDAKernel::reassignVelocities(
  const BigReal timestep,
  float *gaussrand_x,
  float *gaussrand_y,
  float *gaussrand_z,
  double *vel_x,
  double *vel_y,
  double *vel_z,
  const double* d_recipMass,
  const BigReal kbT,
  const int numAtoms,
  const int numAtomsGlobal,
  const int stride,
  curandGenerator_t gen,
  cudaStream_t stream)
{

  curandCheck(curandSetStream(gen, stream));
  // array buffers have to be even length for pseudorandom normal distribution
  // choose n to be the smallest even number >= numAtoms
  // we can choose 1 larger than numAtoms since allocation is > numAtoms
  int n = (numAtomsGlobal + 1) & (~1);

  curandCheck(curandGenerateNormal(gen, gaussrand_x, n, 0, 1));
  curandCheck(curandGenerateNormal(gen, gaussrand_y, n, 0, 1));
  curandCheck(curandGenerateNormal(gen, gaussrand_z, n, 0, 1));
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  reassignVelocitiesKernel<<<grid, ATOM_BLOCKS, 0, stream>>>(
    timestep, 
    gaussrand_x + stride, gaussrand_y + stride, gaussrand_z + stride,
    vel_x, vel_y, vel_z,  d_recipMass, kbT, numAtoms);
  cudaCheck(cudaStreamSynchronize(stream));

}



__global__ void updateVelocities(const int nRattles,
  const int nSettles,
  const float invdt,
  const int * __restrict settleList,
  const CudaRattleElem * __restrict rattleList,
  const double * __restrict pos_x,
  const double * __restrict pos_y,
  const double * __restrict pos_z,
  const double * __restrict posNew_x,
  const double * __restrict posNew_y,
  const double * __restrict posNew_z,
  double * velNew_x,
  double * velNew_y,
  double * velNew_z){

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = gridDim.x*blockDim.x;

#if 0
  for(int i = tid; i <  nSettles; i += stride){
    int ig = settleList[i];
    //Now I update the position
    //Updates three atoms in the settleList
    velNew_x[ig] = (posNew_x[ig] - pos_x[ig]) * invdt;
    velNew_y[ig] = (posNew_y[ig] - pos_y[ig]) * invdt;
    velNew_z[ig] = (posNew_z[ig] - pos_z[ig]) * invdt;

    velNew_x[ig+1] = (posNew_x[ig+1] - pos_x[ig+1]) * invdt;
    velNew_y[ig+1] = (posNew_y[ig+1] - pos_y[ig+1]) * invdt;
    velNew_z[ig+1] = (posNew_z[ig+1] - pos_z[ig+1]) * invdt;

    velNew_x[ig+2] = (posNew_x[ig+2] - pos_x[ig+2]) * invdt;
    velNew_y[ig+2] = (posNew_y[ig+2] - pos_y[ig+2]) * invdt;
    velNew_z[ig+2] = (posNew_z[ig+2] - pos_z[ig+2]) * invdt;
  }
#endif

  for(int i = tid; i <  nRattles; i += stride){
    int ig = rattleList[i].ig;
    int icnt = rattleList[i].icnt;
    bool fixed[4];
    fixed[0] = false; fixed[1] = false;
    fixed[2] = false; fixed[3] = false;
    for(int j = 0; j < icnt; j++){
      //Gets two positions
      int ia = rattleList[i].params[j].ia;
      int ib = rattleList[i].params[j].ib;
      //checks if any of these positions have been updated yet
      if(!fixed[ia]){
        velNew_x[ig+ia] = (posNew_x[ig+ia] - pos_x[ig+ia]) * invdt;
        velNew_y[ig+ia] = (posNew_y[ig+ia] - pos_y[ig+ia]) * invdt;
        velNew_z[ig+ia] = (posNew_z[ig+ia] - pos_z[ig+ia]) * invdt;
        fixed[ia] = true;
      }
      if(!fixed[ib]){
        velNew_x[ig+ib] = (posNew_x[ig+ib] - pos_x[ig+ib]) * invdt;
        velNew_y[ig+ib] = (posNew_y[ig+ib] - pos_y[ig+ib]) * invdt;
        velNew_z[ig+ib] = (posNew_z[ig+ib] - pos_z[ig+ib]) * invdt;
        fixed[ib] = true;
      }
    }
  }
}

//JM: Function that adds a correction to the bonded forces
template<bool doEnergy>
__global__ void addRattleForce(int numAtoms,
      const double invdt,
      const float  * __restrict mass,
      const double * __restrict pos_x,
      const double * __restrict pos_y,
      const double * __restrict pos_z,
      double * __restrict vel_x,
      double * __restrict vel_y,
      double * __restrict vel_z,
      double * __restrict velNew_x,
      double * __restrict velNew_y,
      double * __restrict velNew_z,
      double * __restrict f_normal_x,
      double * __restrict f_normal_y,
      double * __restrict f_normal_z,
      cudaTensor* __restrict virial,
      cudaTensor* __restrict h_virial,
      unsigned int* tbcatomic){

  cudaTensor lVirial;
  // zero_cudaTensor(lVirial);
  int i = threadIdx.x + blockIdx.x*blockDim.x;

  double df[3];
  double pos[3];
  lVirial.xx = 0.0; lVirial.xy = 0.0; lVirial.xz = 0.0;
  lVirial.yx = 0.0; lVirial.yy = 0.0; lVirial.yz = 0.0;
  lVirial.zx = 0.0; lVirial.zy = 0.0; lVirial.zz = 0.0;
  //for(int i = tid; i < numAtoms; i += stride){
  if(i < numAtoms){
    df[0] = (velNew_x[i] - vel_x[i]) * ((double)mass[i] * invdt);
    df[1] = (velNew_y[i] - vel_y[i]) * ((double)mass[i] * invdt);
    df[2] = (velNew_z[i] - vel_z[i]) * ((double)mass[i] * invdt);
    if(doEnergy){
      pos[0] = pos_x[i]; pos[1] = pos_y[i]; pos[2] = pos_z[i];
      lVirial.xx += df[0] * pos[0];
      lVirial.xy += df[0] * pos[1];
      lVirial.xz += df[0] * pos[2];
      lVirial.yx += df[1] * pos[0];
      lVirial.yy += df[1] * pos[1];
      lVirial.yz += df[1] * pos[2];
      lVirial.zx += df[2] * pos[0];
      lVirial.zy += df[2] * pos[1];
      lVirial.zz += df[2] * pos[2];
    }

    //Updates force and velocities
    f_normal_x[i] += df[0];
    f_normal_y[i] += df[1];
    f_normal_z[i] += df[2];

    // instead of copying I can swap the pointers
    vel_x[i] = velNew_x[i];
    vel_y[i] = velNew_y[i];
    vel_z[i] = velNew_z[i];

  }
  //Makes sure that every thread from block has its lVirial set
  // Now do a block reduce on each position of virialTensor
  // to get consistent values
  // The virial tensor is supposed to be symmetric
  // XXX TODO: Handwrite the reductions to avoid syncthreads
  if(doEnergy){
    typedef cub::BlockReduce<BigReal, ATOM_BLOCKS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    lVirial.xx = BlockReduce(temp_storage).Sum(lVirial.xx); __syncthreads();
    // lVirial.xy = BlockReduce(temp_storage).Sum(lVirial.xy); __syncthreads();
    // lVirial.xz = BlockReduce(temp_storage).Sum(lVirial.xz); __syncthreads();
    lVirial.yx = BlockReduce(temp_storage).Sum(lVirial.yx); __syncthreads();
    lVirial.yy = BlockReduce(temp_storage).Sum(lVirial.yy); __syncthreads();
    // lVirial.yz = BlockReduce(temp_storage).Sum(lVirial.yz); __syncthreads();
    lVirial.zx = BlockReduce(temp_storage).Sum(lVirial.zx); __syncthreads();
    lVirial.zy = BlockReduce(temp_storage).Sum(lVirial.zy); __syncthreads();
    lVirial.zz = BlockReduce(temp_storage).Sum(lVirial.zz); __syncthreads();

    // Every block has a locally reduced blockVirial
    // Now every thread does an atomicAdd to get a global virial
    if(threadIdx.x == 0){
      const int bin = blockIdx.x % ATOMIC_BINS;

      atomicAdd(&virial[bin].xx, lVirial.xx);
      // atomicAdd(&virial[bin].xy, lVirial.xy);
      // atomicAdd(&virial[bin].xz, lVirial.xz);
      atomicAdd(&virial[bin].yx, lVirial.yx);
      atomicAdd(&virial[bin].yy, lVirial.yy);
      // atomicAdd(&virial[bin].yz, lVirial.yz);
      atomicAdd(&virial[bin].zx, lVirial.zx);
      atomicAdd(&virial[bin].zy, lVirial.zy);
      atomicAdd(&virial[bin].zz, lVirial.zz);
    }
  }
}


template <bool fillIndexes>
__global__ void buildRattleParams(const int size,
  const float dt,
  const float invdt,
  const int *hgi,
  //Do I need new vectors for holding the rattle forces? Let's do this for now
  const float  * __restrict mass,
  const int    * __restrict hydrogenGroupSize,
  const float  * __restrict rigidBondLength,
  const int   * __restrict atomFixed,
  CudaRattleElem*  rattleList,
  int *rattleIndexes)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = gridDim.x * blockDim.x;
  float rmass[10];
  float dsq[10];
  int   fixed[10];
  int   ial[10];
  int   ibl[10];
  int   ig, hgs, i;
  bool anyfixed;

  for(int k = tid; k < size; k += stride){
    if(!fillIndexes){
        i = rattleIndexes[k];
        if(i == -1) break;
    }else i = k;
    ig = hgi[i];
    hgs = hydrogenGroupSize[ig];
    if(hgs == 1) continue; //early bail. Sucks;
    int icnt = 0;
    anyfixed = false;
    for(int j = 0; j < hgs; j++){
      rmass[j] = (mass[ig + j] > 0.f ? __frcp_rn(mass[ig+j]) : 0.f);
      fixed[j] = atomFixed[ig+j];
      if(fixed[j]) anyfixed = true;
    }
    float tmp = rigidBondLength[ig];
    //We bail here if hydrogens are not fixed
    if(tmp > 0.f){
      if(!anyfixed) continue;
      if( !(fixed[1] && fixed[2]) ){
        dsq[icnt]   = tmp*tmp;
        ial[icnt]   = 1;
        ibl[icnt++] = 2;
      }
    }
    if(fillIndexes){
        rattleIndexes[i] = i;
        continue;
    }else{
        for(int j = 1; j < hgs; j++){
            if((tmp = rigidBondLength[ig+j]) > 0){
                if( !(fixed[0] && fixed[j])){
                    dsq[icnt]   = tmp * tmp;
                    ial[icnt]   = 0;
                    ibl[icnt++] = j;
                }
            }
        }
        // This is really bad: does an atomicadd for each rattle
        // Improve this later
        rattleList[k].ig   = ig;
        rattleList[k].icnt = icnt;
        for(int j = 0; j < icnt; j++){
            int ia = ial[j];
            int ib = ibl[j];
            rattleList[k].params[j].ia  = ia;
            rattleList[k].params[j].ib  = ib;
            rattleList[k].params[j].dsq = dsq[j];
            rattleList[k].params[j].rma = rmass[ia];
            rattleList[k].params[j].rmb = rmass[ib];
        }
    }
  }
}

void buildRattleLists(const int natoms,
    const float   dt,
    const float   invdt,
    const float  *mass,
    const int    *hydrogenGroupSize,
    const float *rigidBondLength,
    const int *atomFixed,
    int **settleList,
    size_t& settleListSize,
    int **consFailure,
    size_t& consFailureSize,
    CudaRattleElem **rattleList,
    size_t& rattleListSize,
    int &nSettles,
    int &nRattles,
    int*& d_nHG,
    int*& d_nSettles,
    int*& d_nRattles,
    int*& hgi,
    size_t& hgi_size,
    char*& d_rattleList_temp_storage,
    size_t& temp_storage_bytes,
    int*& rattleIndexes,
    size_t& rattleIndexes_size,
    bool first,
    cudaStream_t stream){
    size_t nHG = 0;
    
    ///thrust::device_vector<int> hgi(natoms);
    isWater comp(rigidBondLength, hydrogenGroupSize, atomFixed);
    //thrust::device_ptr<const int> h_dev = thrust::device_pointer_cast(hydrogenGroupSize)
    size_t st_hgi_size = (size_t)(hgi_size);
    reallocate_device<int>(&hgi, &st_hgi_size, natoms, OVERALLOC);
    hgi_size = (size_t)(st_hgi_size);

    size_t temp_length = 1;
    reallocate_device<int>(&d_nHG, &temp_length, 1);
    reallocate_device<int>(&d_nSettles, &temp_length, 1);
    reallocate_device<int>(&d_nRattles, &temp_length, 1);

    copy_DtoD<int>(hydrogenGroupSize, hgi, natoms, stream);
    size_t new_storage_bytes = 0; 
    // Removes zeroes from HGI array
    // Pass NULL to first CUB call to get temp buffer size
    cudaCheck(cub::DeviceSelect::If(NULL, new_storage_bytes, hgi, hgi, 
        d_nHG, natoms, notZero(), stream));
    reallocate_device<char>(&d_rattleList_temp_storage, &temp_storage_bytes, new_storage_bytes, OVERALLOC); 
    // DC: There were issues where reallocate wanted int types and CUB wanted unsigned ints
    new_storage_bytes = temp_storage_bytes;
    cudaCheck( cub::DeviceSelect::If(d_rattleList_temp_storage, new_storage_bytes, hgi, hgi, 
        d_nHG, natoms, notZero(), stream));

    int temp;
    copy_DtoH<int>(d_nHG, &temp, 1, stream );
    nHG = temp;

    cudaCheck(cudaStreamSynchronize(stream));
 
    // Exclusive scan to build an array of indices
    new_storage_bytes = 0; 
    cub::DeviceScan::ExclusiveSum(NULL, 
      new_storage_bytes, hgi, hgi, nHG, stream);
    reallocate_device<char>((char**) &d_rattleList_temp_storage, &temp_storage_bytes, new_storage_bytes, OVERALLOC); 
    new_storage_bytes = temp_storage_bytes;
    cub::DeviceScan::ExclusiveSum(d_rattleList_temp_storage, new_storage_bytes, 
       hgi, hgi, nHG, stream);
    
    size_t st_tempListSize = (size_t)(settleListSize);
    reallocate_device<int>(settleList, &st_tempListSize, nHG, OVERALLOC);
    settleListSize = (size_t)(st_tempListSize);

    st_tempListSize = (size_t)(rattleListSize);
    reallocate_device<CudaRattleElem>(rattleList, &st_tempListSize, nHG, OVERALLOC);
    rattleListSize = (size_t)(st_tempListSize);

    st_tempListSize = (size_t)(consFailureSize);
    reallocate_device<int>(consFailure, &st_tempListSize, nHG, OVERALLOC);
    consFailureSize = (size_t)(st_tempListSize);

    st_tempListSize = (size_t)(rattleIndexes_size);
    reallocate_device<int>(&rattleIndexes, &st_tempListSize, nHG, OVERALLOC);
    rattleIndexes_size = (size_t)(st_tempListSize);

    // Flagging arrays
    cudaCheck(cudaMemsetAsync(*rattleList,  0, sizeof(CudaRattleElem)*nHG, stream));
    cudaCheck(cudaMemsetAsync(rattleIndexes,-1, sizeof(int)*nHG, stream));
    cudaCheck(cudaMemsetAsync(*settleList, 0, sizeof(int)*nHG, stream));

    // builds SettleList with isWater functor
    new_storage_bytes = 0; 
    cub::DeviceSelect::If(NULL, new_storage_bytes, hgi, *settleList, 
         d_nSettles, nHG, comp, stream);
    reallocate_device<char>((char**) &d_rattleList_temp_storage, &temp_storage_bytes, new_storage_bytes, OVERALLOC); 
    new_storage_bytes = temp_storage_bytes;
    cub::DeviceSelect::If(d_rattleList_temp_storage, new_storage_bytes, hgi, *settleList, 
        d_nSettles, nHG, comp, stream);
    
    // alright I have the number of settles here
    copy_DtoH<int>(d_nSettles, &nSettles, 1, stream);

    //Warmup call to buildRattleParams
    buildRattleParams<true><<<128, 128, 0, stream>>>(nHG, dt, invdt,
        hgi, mass, hydrogenGroupSize,
        rigidBondLength, atomFixed, *rattleList, rattleIndexes);
    cudaCheck(cudaGetLastError());

    // Removing -1 from rattleIndexes
    new_storage_bytes = 0; 
    cub::DeviceSelect::If(NULL, new_storage_bytes, 
        rattleIndexes, rattleIndexes, d_nRattles, nHG, validRattle(), stream);
    reallocate_device<char>((char**) &d_rattleList_temp_storage, &temp_storage_bytes, new_storage_bytes, OVERALLOC); 
    new_storage_bytes = temp_storage_bytes;
    cub::DeviceSelect::If(d_rattleList_temp_storage, new_storage_bytes, 
        rattleIndexes, rattleIndexes, d_nRattles, nHG, validRattle(), stream);
    copy_DtoH<int>(d_nRattles, &nRattles, 1, stream);

    // Calculates rattleParams on rattleIndexes
    buildRattleParams<false><<<128, 128, 0, stream>>>(nRattles, dt, invdt,
        hgi, mass, hydrogenGroupSize,
        rigidBondLength, atomFixed, *rattleList, rattleIndexes);
        
    cudaCheck(cudaGetLastError());
}

// XXX TODO: JM: Memory access pattern in this function is bad, since we have to
//           access memory by the hydrogen groups. Improve it later

void SequencerCUDAKernel::rattle1(
  const bool doEnergy,
  const bool pressure, // pressure is only false for startRun1 and startRun2 right now
  const int numAtoms,
  const double dt,
  const double invdt,
  const double tol2,
  double *vel_x,
  double *vel_y,
  double *vel_z,
  double *pos_x,
  double *pos_y,
  double *pos_z,
  double *velNew_x,
  double *velNew_y,
  double *velNew_z,
  double *posNew_x,
  double *posNew_y,
  double *posNew_z,
  double *f_normal_x,
  double *f_normal_y,
  double *f_normal_z,
  const int *hydrogenGroupSize,
  const float *rigidBondLength,
  const float *mass,
  const int *atomFixed,
  int **settleList,
  size_t& settleListSize, 
  int **consFailure_d,
  size_t& consFailureSize, 
  CudaRattleElem **rattleList,
  size_t& rattleListSize,
  int *nSettle,
  int *nRattle,
  cudaTensor *virial,
  cudaTensor *h_virial,
  unsigned int* tbcatomic,
  int migration,
  SettleParameters *sp,
  bool first,
  int* consFailure,
  const WaterModel water_model,
  cudaStream_t stream)
 {
  //Calls the necessary functions to enforce rigid bonds
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  int maxiter = 10;

  if(migration || !firstRattleDone){
    // I need to allocate the water positions here
    buildRattleLists(numAtoms, dt, invdt, mass, hydrogenGroupSize, 
        rigidBondLength, atomFixed, settleList, settleListSize,
        consFailure_d, consFailureSize, rattleList, rattleListSize,
        *nSettle, *nRattle,
        d_nHG, d_nSettles, d_nRattles, hgi, hgi_size,
        d_rattleList_temp_storage, temp_storage_bytes,
        rattleIndexes, rattleIndexes_size, 
        first,stream);
    firstRattleDone = true;
  }

  if (1){
    this->updateRigidArrays(dt,
      vel_x, vel_y, vel_z,
      pos_x, pos_y, pos_z,
      velNew_x, velNew_y, velNew_z,
      posNew_x, posNew_y, posNew_z,
      numAtoms, stream);

  #if 1
    // if(*nSettle != 0){
    if(*nSettle != 0){
        Settle(doEnergy,
            numAtoms, dt, invdt, *nSettle,
            vel_x, vel_y, vel_z,
            pos_x, pos_y, pos_z,
            velNew_x, velNew_y, velNew_z,
            posNew_x, posNew_y, posNew_z,
            f_normal_x, f_normal_y, f_normal_z, virial, mass,
            hydrogenGroupSize, rigidBondLength, atomFixed,
            *settleList, sp, water_model, stream);
    }


    if(*nRattle != 0){
      MSHAKE_CUDA(doEnergy, *rattleList, *nRattle, hydrogenGroupSize, pos_x, pos_y, pos_z,
      posNew_x, posNew_y, posNew_z, velNew_x, velNew_y,velNew_z,
      f_normal_x, f_normal_y, f_normal_z, virial, mass,
      invdt, tol2, maxiter, *consFailure_d,
      consFailure, stream);
    }
  #else
      Rattle1Kernel<<<grid, ATOM_BLOCKS, 0, stream>>>(numAtoms,
        dt, invdt, *nSettle,
        vel_x, vel_y, vel_z,
        pos_x, pos_y, pos_z,
        velNew_x, velNew_y, velNew_z,
        posNew_x, posNew_y, posNew_z,
        f_normal_x, f_normal_y, f_normal_z,
        virial, mass,
        hydrogenGroupSize, rigidBondLength, atomFixed,
        *settleList, sp,
        *rattleList, *nRattle, tol2, maxiter, *consFailure_d);
  #endif


    if(invdt == 0){
      // only for the first time step
      copy_DtoD<double>(posNew_x, pos_x, numAtoms, stream);
      copy_DtoD<double>(posNew_y, pos_y, numAtoms, stream);
      copy_DtoD<double>(posNew_z, pos_z, numAtoms, stream);
      cudaCheck(cudaStreamSynchronize(stream));
    }else if (pressure == 0){ // pressure is zero for the first timesteps
      copy_DtoD<double>(velNew_x, vel_x, numAtoms, stream);
      copy_DtoD<double>(velNew_y, vel_y, numAtoms, stream);
      copy_DtoD<double>(velNew_z, vel_z, numAtoms, stream);
      cudaCheck(cudaStreamSynchronize(stream));
    }else{
      if(doEnergy){
        addRattleForce<true><<<grid, ATOM_BLOCKS, 0 ,stream>>>(numAtoms, invdt, mass,
          pos_x, pos_y, pos_z,
          vel_x, vel_y, vel_z,
          velNew_x, velNew_y, velNew_z,
          f_normal_x, f_normal_y, f_normal_z, virial, h_virial, tbcatomic);
      }else {
        addRattleForce<false><<<grid, ATOM_BLOCKS, 0 ,stream>>>(numAtoms, invdt, mass,
          pos_x, pos_y, pos_z,
          vel_x, vel_y, vel_z,
          velNew_x, velNew_y, velNew_z,
          f_normal_x, f_normal_y, f_normal_z, virial, h_virial, tbcatomic);
      }
    }
  }else{
    // Calls a fused kernel without any tensor update
    // let's calculate the number of settle blocks
    int nSettleBlocks = (*nSettle + 128 -1 ) / 128;
    int nShakeBlocks  = (*nRattle + 128 -1 ) / 128;
    // int nTotalBlocks  = (nSettleBlocks + nShakeBlocks);
    CallRattle1Kernel(
      numAtoms, dt, invdt, *nSettle,
      vel_x, vel_y, vel_z,
      pos_x, pos_y, pos_z,
      f_normal_x, f_normal_y, f_normal_z,
      mass, hydrogenGroupSize, rigidBondLength, atomFixed,
      *settleList, sp,
      *rattleList, *nRattle, tol2, maxiter, *consFailure_d, nSettleBlocks,
      nShakeBlocks, water_model, stream
    );
  }
}


template <bool T_NORMALIZED, bool T_DOENERGY>
__global__ void eFieldKernel(
  const int         numAtoms,
  const double3     eField,
  const double      eFieldOmega, 
  const double      eFieldPhi, 
  const double      t, 
  const Lattice     lat, 
  const char3*  __restrict     transform, 
  const float*  __restrict     charges,
  const double* __restrict     pos_x,
  const double* __restrict     pos_y,
  const double* __restrict     pos_z,
  double*       __restrict     f_normal_x,
  double*       __restrict     f_normal_y,
  double*       __restrict     f_normal_z,
  double3*      __restrict     d_extForce,
  cudaTensor*   __restrict     d_extVirial,
  double*       __restrict     d_extEnergy,
  double3*      __restrict     h_extForce,
  cudaTensor*   __restrict     h_extVirial,
  double*       __restrict     h_extEnergy,
  unsigned int*                tbcatomic)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  double3 eField1;
  
  // This can all be moved outside of the kernel

  double fx, fy, fz;
  double eCos = cos(eFieldOmega * t - eFieldPhi);
  double charge = charges[i];
  double3 r_extForce = {0, 0, 0};
  double  r_extEnergy = 0;
  cudaTensor r_extVirial = {0};
  int totaltb = gridDim.x;
  __shared__ bool isLastBlockDone;
  eField1.x = eCos * eField.x;
  eField1.y = eCos * eField.y;
  eField1.z = eCos * eField.z;

  if(threadIdx.x == 0) {
     isLastBlockDone = 0;
  }

  __syncthreads();

  if(i < numAtoms){
    // This can be moved outside of the kernel
    if(T_NORMALIZED){
      eField1 = Vector(lat.a_r()* (Vector)eField1, lat.b_r()* (Vector)eField1, lat.c_r()* (Vector)eField1);
    }
    
    fx = charge * eField1.x;
    fy = charge * eField1.y;
    fz = charge * eField1.z;

    f_normal_x[i] += fx;
    f_normal_y[i] += fy;
    f_normal_z[i] += fz;

    if(T_DOENERGY){
      const char3 tr = transform[i];
      double3 vpos;
      vpos.x = pos_x[i];
      vpos.y = pos_y[i];
      vpos.z = pos_z[i];

      //all threads from block calculate their contribution to energy, extForce and extVirial
      vpos = lat.reverse_transform(vpos, tr);
      double3 o = lat.origin();
      r_extEnergy -= (fx * (vpos.x - o.x)) + (fy * (vpos.y - o.y)) + (fz * (vpos.z - o.z));

      if(!T_NORMALIZED){
        r_extForce.x = fx;
        r_extForce.y = fy;
        r_extForce.z = fz;

        // outer product to calculate a virial here
        r_extVirial.xx = fx * vpos.x;
        r_extVirial.xy = fx * vpos.y;
        r_extVirial.xz = fx * vpos.z;
        r_extVirial.yx = fy * vpos.x;
        r_extVirial.yy = fy * vpos.y;
        r_extVirial.yz = fy * vpos.z;
        r_extVirial.zx = fz * vpos.x;
        r_extVirial.zz = fz * vpos.z;
        r_extVirial.zy = fz * vpos.y;
      }
    }
  }

  __syncthreads();
  if(T_DOENERGY){
      // use cub to reduce energies
      typedef cub::BlockReduce<double, ATOM_BLOCKS> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      r_extEnergy  = BlockReduce(temp_storage).Sum(r_extEnergy);
      __syncthreads();
      if(!T_NORMALIZED){
        // Do the remaining reductions
        // external force
        r_extForce.x  = BlockReduce(temp_storage).Sum(r_extForce.x);
        __syncthreads();
        r_extForce.y  = BlockReduce(temp_storage).Sum(r_extForce.y);
        __syncthreads();
        r_extForce.z  = BlockReduce(temp_storage).Sum(r_extForce.z);
        __syncthreads();
        // external virial
        r_extVirial.xx  = BlockReduce(temp_storage).Sum(r_extVirial.xx);
        __syncthreads();
        r_extVirial.xy  = BlockReduce(temp_storage).Sum(r_extVirial.xy);
        __syncthreads();
        r_extVirial.xz  = BlockReduce(temp_storage).Sum(r_extVirial.xz);
        __syncthreads();
        r_extVirial.yx  = BlockReduce(temp_storage).Sum(r_extVirial.yx);
        __syncthreads();
        r_extVirial.yy  = BlockReduce(temp_storage).Sum(r_extVirial.yy);
        __syncthreads();
        r_extVirial.yz  = BlockReduce(temp_storage).Sum(r_extVirial.yz);
        __syncthreads();
        r_extVirial.zx  = BlockReduce(temp_storage).Sum(r_extVirial.zx);
        __syncthreads();
        r_extVirial.zy  = BlockReduce(temp_storage).Sum(r_extVirial.zy);
        __syncthreads();
        r_extVirial.zz  = BlockReduce(temp_storage).Sum(r_extVirial.zz);
        __syncthreads();
      }

      // Cool now atomic add to gmem
      if(threadIdx.x == 0 ){
        const int bin = blockIdx.x % ATOMIC_BINS;

        // printf("adding %lf to d_extEnergy\n", r_extEnergy);
        atomicAdd(&d_extEnergy[bin], r_extEnergy);

        if(!T_NORMALIZED){
          // add force and virial as well
          atomicAdd(&d_extForce[bin].x, r_extForce.x);
          atomicAdd(&d_extForce[bin].y, r_extForce.y);
          atomicAdd(&d_extForce[bin].z, r_extForce.z);

          atomicAdd(&d_extVirial[bin].xx, r_extVirial.xx);
          atomicAdd(&d_extVirial[bin].xy, r_extVirial.xy);
          atomicAdd(&d_extVirial[bin].xz, r_extVirial.xz);

          atomicAdd(&d_extVirial[bin].yx, r_extVirial.yx);
          atomicAdd(&d_extVirial[bin].yy, r_extVirial.yy);
          atomicAdd(&d_extVirial[bin].yz, r_extVirial.yz);

          atomicAdd(&d_extVirial[bin].zx, r_extVirial.zx);
          atomicAdd(&d_extVirial[bin].zy, r_extVirial.zy);
          atomicAdd(&d_extVirial[bin].zz, r_extVirial.zz);
        }

        __threadfence();
        unsigned int value = atomicInc(&tbcatomic[2], totaltb);
        isLastBlockDone = (value == (totaltb -1));

      }
      __syncthreads();
      if(isLastBlockDone){
        if(threadIdx.x < ATOMIC_BINS){
          const int bin = threadIdx.x;

          double e = d_extEnergy[bin];
          double3 f;
          cudaTensor v;
          if(!T_NORMALIZED){
            f = d_extForce[bin];
            v = d_extVirial[bin];
          }

          // sets device scalars back to zero
          d_extEnergy[bin] = 0.0;
          if(!T_NORMALIZED){
            d_extForce[bin] = make_double3(0.0, 0.0, 0.0);
            zero_cudaTensor(d_extVirial[bin]);
          }

          if(ATOMIC_BINS > 1){
            typedef cub::WarpReduce<double, (ATOMIC_BINS > 1 ? ATOMIC_BINS : 2)> WarpReduce;
            typedef cub::WarpReduce<cudaTensor, (ATOMIC_BINS > 1 ? ATOMIC_BINS : 2)> WarpReduceT;
            __shared__ typename WarpReduce::TempStorage tempStorage;
            __shared__ typename WarpReduceT::TempStorage tempStorageT;

            e = WarpReduce(tempStorage).Sum(e);
            if(!T_NORMALIZED){
              f.x = WarpReduce(tempStorage).Sum(f.x);
              f.y = WarpReduce(tempStorage).Sum(f.y);
              f.z = WarpReduce(tempStorage).Sum(f.z);
              v = WarpReduceT(tempStorageT).Sum(v);
            }
          }

          if(threadIdx.x == 0){
            h_extEnergy[0] = e;
            if(!T_NORMALIZED){
              h_extForce[0] = f;
              h_extVirial[0] = v;
            }

            //resets atomic counter
            reset_atomic_counter(&(tbcatomic[2]));
          }
        }
      }
  }
}

// JM NOTE: Apply external electric field to every atom in the system equally
void SequencerCUDAKernel::apply_Efield(
  const int         numAtoms,
  const bool        normalized,
  const bool        doEnergy,
  const double3     eField,
  const double      eFieldOmega, 
  const double      eFieldPhi, 
  const double      t, 
  const Lattice     lat, 
  const char3*      transform, 
  const float*      charges,
  const double*     pos_x,
  const double*     pos_y,
  const double*     pos_z,
  double*           f_normal_x,
  double*           f_normal_y,
  double*           f_normal_z,
  double3*          d_extForce,
  cudaTensor*       d_extVirial,
  double*           d_extEnergy,
  double3*          h_extForce,
  cudaTensor*       h_extVirial,
  double*           h_extEnergy,
  unsigned int*     tbcatomic,
  cudaStream_t      stream
){
  int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  if(normalized){
    if(doEnergy) {
      eFieldKernel<true, true><<<grid, ATOM_BLOCKS, 0, stream>>>(numAtoms,
        eField, eFieldOmega, eFieldPhi, t, lat, transform, charges,
        pos_x, pos_y, pos_z,
        f_normal_x, f_normal_y, f_normal_z, d_extForce, d_extVirial,
        d_extEnergy, h_extForce, h_extVirial, h_extEnergy, tbcatomic);
    }else{
      eFieldKernel<true, false><<<grid, ATOM_BLOCKS, 0, stream>>>(numAtoms,
        eField, eFieldOmega, eFieldPhi, t, lat, transform, charges,
        pos_x, pos_y, pos_z,
        f_normal_x, f_normal_y, f_normal_z, d_extForce, d_extVirial,
        d_extEnergy, h_extForce, h_extVirial, h_extEnergy, tbcatomic);
    }
  }else{
    if(doEnergy) {
      eFieldKernel<false, true><<<grid, ATOM_BLOCKS, 0, stream>>>(numAtoms,
        eField, eFieldOmega, eFieldPhi, t, lat, transform, charges,
        pos_x, pos_y, pos_z,
        f_normal_x, f_normal_y, f_normal_z, d_extForce, d_extVirial,
        d_extEnergy, h_extForce, h_extVirial, h_extEnergy, tbcatomic);
    }else{
      eFieldKernel<false, false><<<grid, ATOM_BLOCKS, 0, stream>>>(numAtoms,
        eField, eFieldOmega, eFieldPhi, t, lat, transform, charges,
        pos_x, pos_y, pos_z,
        f_normal_x, f_normal_y, f_normal_z, d_extForce, d_extVirial,
        d_extEnergy, h_extForce, h_extVirial, h_extEnergy, tbcatomic);
    }
  }
}

SequencerCUDAKernel::SequencerCUDAKernel() { 
  firstRattleDone = false;
  intConstInit = false;

  d_nHG = NULL;
  d_nSettles = NULL;
  d_nRattles = NULL;

  hgi = NULL;
  hgi_size = 0;

  d_rattleList_temp_storage = NULL;
  temp_storage_bytes = 0;

  rattleIndexes = NULL;
  rattleIndexes_size = 0;
}

SequencerCUDAKernel::~SequencerCUDAKernel() {
  deallocate_device<int>(&d_nHG); 
  deallocate_device<int>(&d_nSettles); 
  deallocate_device<int>(&d_nRattles); 
  deallocate_device<int>(&hgi); 
  deallocate_device<int>(&rattleIndexes); 
  deallocate_device<char>(&d_rattleList_temp_storage); 
}



#if 1




void SequencerCUDAKernel::set_pme_positions(
  const int  devID,
  const bool isPmeDevice, 
  const int  nDev, 
  const int  numPatchesHomeAndProxy,
  const int  numPatchesHome,
  const bool doNbond, 
  const bool doSlow,
  const bool doFEP,
  const bool doTI,
  const bool doAlchDecouple,
  const bool doAlchSoftCore,
  const double* d_pos_x,
  const double* d_pos_y,
  const double* d_pos_z,
#ifndef NAMD_NCCL_ALLREDUCE
  double**      d_peer_pos_x, 
  double**      d_peer_pos_y, 
  double**      d_peer_pos_z, 
  float**       d_peer_charge, 
  int**         d_peer_partition,
#endif
  const float* charges,
  const int* partition,
  const double charge_scaling, 
  const double3* patchCenter, 
  const int* s_patchPositions, 
  const int* s_pencilPatchIndex, 
  const int* s_patchIDs, 
  const int* patchSortOrder, 
  const Lattice lattice,
  float4* nb_atoms,
  float4* b_atoms,
  float4* s_atoms,
  int* s_partition,
  int numTotalAtoms,
  CudaLocalRecord*                  localRecords,
  CudaPeerRecord*                   peerRecords,
  std::vector<int>& atomCounts,
  cudaStream_t stream
) {
  // Launch PME setComputes
  #define CALL(HOME, FEP, TI, ALCH_DECOUPLE, ALCH_SOFTCORE) \
    setComputePositionsKernel_PME<HOME, FEP, TI, ALCH_DECOUPLE, ALCH_SOFTCORE> \
    <<<pme_grid, PATCH_BLOCKS, 0, stream>>>( \
          d_pos_x, d_pos_y, d_pos_z, charges, \
          d_peer_pos_x, d_peer_pos_y, d_peer_pos_z, d_peer_charge, \
          d_peer_partition, partition, charge_scaling,  \
          s_patchPositions, s_pencilPatchIndex, s_patchIDs, \
          lattice, s_atoms, numTotalAtoms, s_partition, \
          i, numAtoms, offset \
    );
  // Only when PME long-range electrostaic is enabled (doSlow is true) 
  // The partition is needed for alchemical calculation.
  if(doSlow && isPmeDevice) {
    int offset = 0;
    for (int i = 0; i < nDev; i++) {
      const bool home = (i == devID);
      const int numAtoms = atomCounts[i];
      const int pme_grid = (numAtoms + PATCH_BLOCKS - 1) / PATCH_BLOCKS;
      const int options = (home << 4) + (doFEP << 3) + (doTI << 2) + (doAlchDecouple << 1) + doAlchSoftCore;

      switch (options) {
        case  0: CALL(0, 0, 0, 0, 0); break;
        case  1: CALL(0, 0, 0, 0, 1); break;
        case  2: CALL(0, 0, 0, 1, 0); break;
        case  3: CALL(0, 0, 0, 1, 1); break;
        case  4: CALL(0, 0, 1, 0, 0); break;
        case  5: CALL(0, 0, 1, 0, 1); break;
        case  6: CALL(0, 0, 1, 1, 0); break;
        case  7: CALL(0, 0, 1, 1, 1); break;
        case  8: CALL(0, 1, 0, 0, 0); break;
        case  9: CALL(0, 1, 0, 0, 1); break;
        case 10: CALL(0, 1, 0, 1, 0); break;
        case 11: CALL(0, 1, 0, 1, 1); break;
        case 12: CALL(0, 1, 1, 0, 0); break;
        case 13: CALL(0, 1, 1, 0, 1); break;
        case 14: CALL(0, 1, 1, 1, 0); break;
        case 15: CALL(0, 1, 1, 1, 1); break;
        case 16: CALL(1, 0, 0, 0, 0); break;
        case 17: CALL(1, 0, 0, 0, 1); break;
        case 18: CALL(1, 0, 0, 1, 0); break;
        case 19: CALL(1, 0, 0, 1, 1); break;
        case 20: CALL(1, 0, 1, 0, 0); break;
        case 21: CALL(1, 0, 1, 0, 1); break;
        case 22: CALL(1, 0, 1, 1, 0); break;
        case 23: CALL(1, 0, 1, 1, 1); break;
        case 24: CALL(1, 1, 0, 0, 0); break;
        case 25: CALL(1, 1, 0, 0, 1); break;
        case 26: CALL(1, 1, 0, 1, 0); break;
        case 27: CALL(1, 1, 0, 1, 1); break;
        case 28: CALL(1, 1, 1, 0, 0); break;
        case 29: CALL(1, 1, 1, 0, 1); break;
        case 30: CALL(1, 1, 1, 1, 0); break;
        case 31: CALL(1, 1, 1, 1, 1); break;
        default:
          NAMD_die("SequencerCUDAKernel::setComputePositions: no kernel called");
      }

      offset += numAtoms;
    }
  }
  #undef CALL
}

__global__ void copyForcesToHostSOAKernel(
  const int               numPatches,
  CudaLocalRecord*        localRecords,
  const int               maxForceNumber, 
  const double*           d_f_normal_x,
  const double*           d_f_normal_y,
  const double*           d_f_normal_z,
  const double*           d_f_nbond_x,
  const double*           d_f_nbond_y,
  const double*           d_f_nbond_z,
  const double*           d_f_slow_x,
  const double*           d_f_slow_y,
  const double*           d_f_slow_z,
  PatchDataSOA*           d_HostPatchDataSOA
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int offset = s_record.bufferOffset;

    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
      if (maxForceNumber >= 0) {
        d_HostPatchDataSOA[patchIndex].f_normal_x[i] = d_f_normal_x[offset + i];
        d_HostPatchDataSOA[patchIndex].f_normal_y[i] = d_f_normal_y[offset + i];
        d_HostPatchDataSOA[patchIndex].f_normal_z[i] = d_f_normal_z[offset + i];
      }
      if (maxForceNumber >= 1) {
        d_HostPatchDataSOA[patchIndex].f_saved_nbond_x[i] = d_f_nbond_x[offset + i];
        d_HostPatchDataSOA[patchIndex].f_saved_nbond_y[i] = d_f_nbond_y[offset + i];
        d_HostPatchDataSOA[patchIndex].f_saved_nbond_z[i] = d_f_nbond_z[offset + i];
      }
      if (maxForceNumber >= 2) {
        d_HostPatchDataSOA[patchIndex].f_saved_slow_x[i] = d_f_slow_x[offset + i];
        d_HostPatchDataSOA[patchIndex].f_saved_slow_y[i] = d_f_slow_y[offset + i];
        d_HostPatchDataSOA[patchIndex].f_saved_slow_z[i] = d_f_slow_z[offset + i];
      }

    }
    __syncthreads();
  }
}

void SequencerCUDAKernel::copyForcesToHostSOA(
  const int               numPatches,
  CudaLocalRecord*        localRecords,
  const int               maxForceNumber, 
  const double*           d_f_normal_x,
  const double*           d_f_normal_y,
  const double*           d_f_normal_z,
  const double*           d_f_nbond_x,
  const double*           d_f_nbond_y,
  const double*           d_f_nbond_z,
  const double*           d_f_slow_x,
  const double*           d_f_slow_y,
  const double*           d_f_slow_z,
  PatchDataSOA*           d_HostPatchDataSOA,
  cudaStream_t            stream
) {
  copyForcesToHostSOAKernel<<<numPatches, PATCH_BLOCKS, 0, stream>>>(
    numPatches,
    localRecords,
    maxForceNumber,
    d_f_normal_x,
    d_f_normal_y,
    d_f_normal_z,
    d_f_nbond_x,
    d_f_nbond_y,
    d_f_nbond_z,
    d_f_slow_x,
    d_f_slow_y,
    d_f_slow_z,
    d_HostPatchDataSOA
  );
}

__global__ void copyPositionsToHostSOAKernel(
  const int               numPatches,
  CudaLocalRecord*        localRecords,
  const double*           pos_x, 
  const double*           pos_y, 
  const double*           pos_z, 
  PatchDataSOA*           d_HostPatchDataSOA
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int offset = s_record.bufferOffset;

    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
      d_HostPatchDataSOA[patchIndex].pos_x[i] = pos_x[offset + i];
      d_HostPatchDataSOA[patchIndex].pos_y[i] = pos_y[offset + i];
      d_HostPatchDataSOA[patchIndex].pos_z[i] = pos_z[offset + i];
    }
    __syncthreads();
  }
}

void SequencerCUDAKernel::copyPositionsToHostSOA(
  const int               numPatches,
  CudaLocalRecord*        localRecords,
  const double*           pos_x, 
  const double*           pos_y, 
  const double*           pos_z, 
  PatchDataSOA*           d_HostPatchDataSOA,
  cudaStream_t            stream
) {
  copyPositionsToHostSOAKernel<<<numPatches, PATCH_BLOCKS, 0, stream>>>(
    numPatches,
    localRecords,
    pos_x, pos_y, pos_z,
    d_HostPatchDataSOA
  );
}

// Swipe from HomePatch::redistrib_lp_water_force
/* Redistribute forces from the massless lonepair charge particle onto
 * the other atoms of the water.
 *
 * This is done using the same algorithm as charmm uses for TIP4P lonepairs.
 *
 * Pass by reference the forces (O H1 H2 LP) to be modified,
 * pass by constant reference the corresponding positions.
 */
template <bool doVirial>
__device__ void redistrib_lp_water_force(
  double f_ox[3], double f_h1[3], double f_h2[3], double f_lp[3],
  const double p_ox[3], const double p_h1[3], const double p_h2[3],
  const double p_lp[3], cudaTensor& virial) {
  // Accumulate force adjustments
  double fad_ox[3] = {};
  double fad_h[3] = {};
  // Calculate the radial component of the force and add it to the oxygen
  const double r_ox_lp[3] = {p_lp[0] - p_ox[0],
                             p_lp[1] - p_ox[1],
                             p_lp[2] - p_ox[2]};
  double invlen2_r_ox_lp = rnorm3d(r_ox_lp[0], r_ox_lp[1], r_ox_lp[2]);
  invlen2_r_ox_lp *= invlen2_r_ox_lp;
  const double rad_factor =
    (f_lp[0] * r_ox_lp[0] + f_lp[1] * r_ox_lp[1] + f_lp[2] * r_ox_lp[2]) * invlen2_r_ox_lp;
  const double f_rad[3] = {r_ox_lp[0] * rad_factor,
                           r_ox_lp[1] * rad_factor,
                           r_ox_lp[2] * rad_factor};
  fad_ox[0] += f_rad[0];
  fad_ox[1] += f_rad[1];
  fad_ox[2] += f_rad[2];
  // Calculate the angular component
  const double r_hcom_ox[3] = {
    p_ox[0] - ( (p_h1[0] + p_h2[0]) * 0.5 ),
    p_ox[1] - ( (p_h1[1] + p_h2[1]) * 0.5 ),
    p_ox[2] - ( (p_h1[2] + p_h2[2]) * 0.5 )};
  const double f_ang[3] = {
    f_lp[0] - f_rad[0],
    f_lp[1] - f_rad[1],
    f_lp[2] - f_rad[2]};
  // now split this component onto the other atoms
  const double len_r_ox_lp = norm3d(r_ox_lp[0], r_ox_lp[1], r_ox_lp[2]);
  const double invlen_r_hcom_ox = rnorm3d(
    r_hcom_ox[0], r_hcom_ox[1], r_hcom_ox[2]);
  const double oxcomp =
    (norm3d(r_hcom_ox[0], r_hcom_ox[1], r_hcom_ox[2]) - len_r_ox_lp) *
    invlen_r_hcom_ox;
  const double hydcomp = 0.5 * len_r_ox_lp * invlen_r_hcom_ox;
  fad_ox[0] += (f_ang[0] * oxcomp);
  fad_ox[1] += (f_ang[1] * oxcomp);
  fad_ox[2] += (f_ang[2] * oxcomp);
  fad_h[0] += (f_ang[0] * hydcomp);
  fad_h[1] += (f_ang[1] * hydcomp);
  fad_h[2] += (f_ang[2] * hydcomp);
  if (doVirial) {
    virial.xx = fad_ox[0] * p_ox[0] + fad_h[0] * p_h1[0] + fad_h[0] * p_h2[0] - f_lp[0] * p_lp[0];
    virial.xy = fad_ox[0] * p_ox[1] + fad_h[0] * p_h1[1] + fad_h[0] * p_h2[1] - f_lp[0] * p_lp[1];
    virial.xz = fad_ox[0] * p_ox[2] + fad_h[0] * p_h1[2] + fad_h[0] * p_h2[2] - f_lp[0] * p_lp[2];
    virial.yx = fad_ox[1] * p_ox[0] + fad_h[1] * p_h1[0] + fad_h[1] * p_h2[0] - f_lp[1] * p_lp[0];
    virial.yy = fad_ox[1] * p_ox[1] + fad_h[1] * p_h1[1] + fad_h[1] * p_h2[1] - f_lp[1] * p_lp[1];
    virial.yz = fad_ox[1] * p_ox[2] + fad_h[1] * p_h1[2] + fad_h[1] * p_h2[2] - f_lp[1] * p_lp[2];
    virial.zx = fad_ox[2] * p_ox[0] + fad_h[2] * p_h1[0] + fad_h[2] * p_h2[0] - f_lp[2] * p_lp[0];
    virial.zy = fad_ox[2] * p_ox[1] + fad_h[2] * p_h1[1] + fad_h[2] * p_h2[1] - f_lp[2] * p_lp[1];
    virial.zz = fad_ox[2] * p_ox[2] + fad_h[2] * p_h1[2] + fad_h[2] * p_h2[2] - f_lp[2] * p_lp[2];
  }
  f_lp[0] = 0;
  f_lp[1] = 0;
  f_lp[2] = 0;
  f_ox[0] += fad_ox[0];
  f_ox[1] += fad_ox[1];
  f_ox[2] += fad_ox[2];
  f_h1[0] += fad_h[0];
  f_h1[1] += fad_h[1];
  f_h1[2] += fad_h[2];
  f_h2[0] += fad_h[0];
  f_h2[1] += fad_h[1];
  f_h2[2] += fad_h[2];
}

template <bool doVirial>
__global__ void redistributeTip4pForcesKernel2(
  double*        d_f_x,
  double*        d_f_y,
  double*        d_f_z,
  cudaTensor*    d_virial,
  const double*  d_pos_x,
  const double*  d_pos_y,
  const double*  d_pos_z,
  const float*   d_mass,
  const int      numAtoms)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  cudaTensor lVirial = {0};
  if (i < numAtoms) {
    if (d_mass[i] < 0.01f) {
      double f_ox[3] = {d_f_x[i-3], d_f_y[i-3], d_f_z[i-3]};
      double f_h1[3] = {d_f_x[i-2], d_f_y[i-2], d_f_z[i-2]};
      double f_h2[3] = {d_f_x[i-1], d_f_y[i-1], d_f_z[i-1]};
      double f_lp[3] = {d_f_x[i],   d_f_y[i],   d_f_z[i]};
      const double p_ox[3] = {d_pos_x[i-3], d_pos_y[i-3], d_pos_z[i-3]};
      const double p_h1[3] = {d_pos_x[i-2], d_pos_y[i-2], d_pos_z[i-2]};
      const double p_h2[3] = {d_pos_x[i-1], d_pos_y[i-1], d_pos_z[i-1]};
      const double p_lp[3] = {d_pos_x[i],   d_pos_y[i],   d_pos_z[i]};
      redistrib_lp_water_force<doVirial>(
        f_ox, f_h1, f_h2, f_lp, p_ox, p_h1, p_h2, p_lp, lVirial);
      // copy the force back
      d_f_x[i-3] = f_ox[0];
      d_f_x[i-2] = f_h1[0];
      d_f_x[i-1] = f_h2[0];
      d_f_x[i]   = f_lp[0];
      d_f_y[i-3] = f_ox[1];
      d_f_y[i-2] = f_h1[1];
      d_f_y[i-1] = f_h2[1];
      d_f_y[i]   = f_lp[1];
      d_f_z[i-3] = f_ox[2];
      d_f_z[i-2] = f_h1[2];
      d_f_z[i-1] = f_h2[2];
      d_f_z[i]   = f_lp[2];
    }
  }
  typedef cub::BlockReduce<BigReal, ATOM_BLOCKS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  if (doVirial) {
    lVirial.xx = BlockReduce(temp_storage).Sum(lVirial.xx);
    __syncthreads();
    lVirial.xy = BlockReduce(temp_storage).Sum(lVirial.xy);
    __syncthreads();
    lVirial.xz = BlockReduce(temp_storage).Sum(lVirial.xz);
    __syncthreads();
    lVirial.yx = BlockReduce(temp_storage).Sum(lVirial.yx);
    __syncthreads();
    lVirial.yy = BlockReduce(temp_storage).Sum(lVirial.yy);
    __syncthreads();
    lVirial.yz = BlockReduce(temp_storage).Sum(lVirial.yz);
    __syncthreads();
    lVirial.zx = BlockReduce(temp_storage).Sum(lVirial.zx);
    __syncthreads();
    lVirial.zy = BlockReduce(temp_storage).Sum(lVirial.zy);
    __syncthreads();
    lVirial.zz = BlockReduce(temp_storage).Sum(lVirial.zz);
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&(d_virial->xx), lVirial.xx);
      atomicAdd(&(d_virial->xy), lVirial.xy);
      atomicAdd(&(d_virial->xz), lVirial.xz);
      atomicAdd(&(d_virial->yx), lVirial.yx);
      atomicAdd(&(d_virial->yy), lVirial.yy);
      atomicAdd(&(d_virial->yz), lVirial.yz);
      atomicAdd(&(d_virial->zx), lVirial.zx);
      atomicAdd(&(d_virial->zy), lVirial.zy);
      atomicAdd(&(d_virial->zz), lVirial.zz);
    }
  }
}

void SequencerCUDAKernel::redistributeTip4pForces(
  double*        d_f_normal_x,
  double*        d_f_normal_y,
  double*        d_f_normal_z,
  double*        d_f_nbond_x,
  double*        d_f_nbond_y,
  double*        d_f_nbond_z,
  double*        d_f_slow_x,
  double*        d_f_slow_y,
  double*        d_f_slow_z,
  cudaTensor*    d_virial_normal,
  cudaTensor*    d_virial_nbond,
  cudaTensor*    d_virial_slow,
  const double*  d_pos_x,
  const double*  d_pos_y,
  const double*  d_pos_z,
  const float*   d_mass,
  const int      numAtoms,
  const int      doVirial,
  const int      maxForceNumber,
  cudaStream_t   stream
) {
  const int grid = (numAtoms + ATOM_BLOCKS - 1) / ATOM_BLOCKS;
  switch (maxForceNumber) {
    case 2:
      if (doVirial) redistributeTip4pForcesKernel2<true><<<grid, ATOM_BLOCKS, 0, stream>>>(d_f_slow_x, d_f_slow_y, d_f_slow_z, d_virial_slow, d_pos_x, d_pos_y, d_pos_z, d_mass, numAtoms);
      else redistributeTip4pForcesKernel2<false><<<grid, ATOM_BLOCKS, 0, stream>>>(d_f_slow_x, d_f_slow_y, d_f_slow_z, d_virial_slow, d_pos_x, d_pos_y, d_pos_z, d_mass, numAtoms);
    case 1:
      if (doVirial) redistributeTip4pForcesKernel2<true><<<grid, ATOM_BLOCKS, 0, stream>>>(d_f_nbond_x, d_f_nbond_y, d_f_nbond_z, d_virial_nbond, d_pos_x, d_pos_y, d_pos_z, d_mass, numAtoms);
      else redistributeTip4pForcesKernel2<false><<<grid, ATOM_BLOCKS, 0, stream>>>(d_f_nbond_x, d_f_nbond_y, d_f_nbond_z, d_virial_nbond, d_pos_x, d_pos_y, d_pos_z, d_mass, numAtoms);
    case 0:
      if (doVirial) redistributeTip4pForcesKernel2<true><<<grid, ATOM_BLOCKS, 0, stream>>>(d_f_normal_x, d_f_normal_y, d_f_normal_z, d_virial_normal, d_pos_x, d_pos_y, d_pos_z, d_mass, numAtoms);
      else redistributeTip4pForcesKernel2<false><<<grid, ATOM_BLOCKS, 0, stream>>>(d_f_normal_x, d_f_normal_y, d_f_normal_z, d_virial_normal, d_pos_x, d_pos_y, d_pos_z, d_mass, numAtoms);
  }
}

#endif


#endif // NODEGROUP_FORCE_REGISTER
