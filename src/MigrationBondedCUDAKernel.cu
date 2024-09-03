/**
 * \file
 * Tuple migration kernels and wrapper functions
 */

#ifdef NAMD_CUDA
#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include <namd_cub/cub.cuh>
#endif
#else // NAMD_HIP
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#define cub hipcub
#endif  // end NAMD_CUDA vs. NAMD_HIP

#include "HipDefines.h"

#include "MigrationBondedCUDAKernel.h"
#include "ComputeBondedCUDAKernel.h"

#ifdef NODEGROUP_FORCE_REGISTER

// Since only the downstream function from patch map is needed
// it is implemented separately
__device__ __forceinline__ int patchMap_downstream(
  const int pid1, const int pid2, const int aDim, const int bDim,
  const int cMaxIndex, const int bMaxIndex, const int aMaxIndex
) {
  int ds;
  if (pid1 == pid2) {
    ds = pid1;
  } else if (pid1 == -1 || pid2 == -1) { // -1 means notUsed
    ds = -1;
  } else {
    int3 data1, data2;
    data1.x = pid1 % aDim;
    data1.y = (pid1 / aDim) % bDim;
    data1.z = pid1 / (aDim * bDim);

    data2.x = pid2 % aDim;
    data2.y = (pid2 / aDim) % bDim;
    data2.z = pid2 / (aDim * bDim);

    int k = data1.z;
    int k2 = data2.z;
    if ( ( k ? k : cMaxIndex ) == k2 + 1 ) k = k2;

    int j = data1.y;
    int j2 = data2.y;
    if ( ( j ? j : bMaxIndex ) == j2 + 1 ) j = j2;

    int i = data1.x;
    int i2 = data2.x;
    if ( ( i ? i : aMaxIndex ) == i2 + 1 ) i = i2;

    ds = ((k*bDim)+j)*aDim + i;
  }
  return ds;
}


/**
 * \brief Computes the destination information for each tuple
 *
 * This kernel computes the destination device and patch for each tuple. It also
 * converts the indexInDevice value stored in the "index" field to the local index
 * of the atom in each patch via the migration destination values. It computes the
 * downstream patch index on that patches home device.
 *
 * The destination device and downstream patch index are stored in separate buffers
 * because they don't need to be migrated to new devices
 *
 * Since the tuple data is stored as AoS, we bulk load a batch into shared memory
 * to use coalesced access
 *
 */
template<typename T>
__global__ void computeTupleDestinationKernel(
  const int               numTuples,
  T*                      stages,
  int*                    d_device,
  int*                    d_downstreamPatchIndex,
  const int4*             migrationDestination,
  const int*              patchToGPU,
  const int*              patchIDtoHomePatchIndex,
  const int               aDim, 
  const int               bDim,
  const int               cMaxIndex, 
  const int               bMaxIndex, 
  const int               aMaxIndex
) {
  using AccessType = int32_t;
  __shared__ T s_in[MigrationBondedCUDAKernel::kNumThreads];
  AccessType* s_in_int = (AccessType*) &(s_in);

  const int numTuplesBlock = ((numTuples + blockDim.x - 1) / blockDim.x) * blockDim.x;

  for (int i = blockDim.x * blockIdx.x; i < numTuplesBlock; i += blockDim.x * gridDim.x) {
    AccessType* g_in_int = (AccessType*) (stages + i);
    const int numBytesIn = sizeof(T) * blockDim.x;
    for (int j = threadIdx.x; j < numBytesIn / sizeof(AccessType); j += blockDim.x) {
      const int tupleIdx = i + (j / (sizeof(T) / sizeof(AccessType)));
      if (tupleIdx < numTuples) {
        s_in_int[j] = g_in_int[j];
      }
    }
    __syncthreads();

    const int tupleIndex = i + threadIdx.x;
    if (tupleIndex < numTuples) {
      T elem = s_in[threadIdx.x];
      const int tmp_pid = migrationDestination[elem.index[0]].x;
      int downstream_patch = tmp_pid;
      elem.patchIDs[0] = tmp_pid;
      elem.index[0] = migrationDestination[elem.index[0]].w;

      for (int j = 1; j < T::size; j++) {
        const int tmp_pid = migrationDestination[elem.index[j]].x;

        downstream_patch = patchMap_downstream(
          downstream_patch, tmp_pid, 
          aDim, bDim, cMaxIndex, bMaxIndex, aMaxIndex
        );
        elem.patchIDs[j] = tmp_pid;
        elem.index[j] = migrationDestination[elem.index[j]].w;
      }
      
      s_in[threadIdx.x] = elem;
      d_device[tupleIndex] = patchToGPU[downstream_patch];
      d_downstreamPatchIndex[tupleIndex] = patchIDtoHomePatchIndex[downstream_patch];
    }
    __syncthreads();

    // Write tuples out
    for (int j = threadIdx.x; j < numBytesIn / sizeof(AccessType); j += blockDim.x) {
      const int tupleIdx = i + (j / (sizeof(T) / sizeof(AccessType)));
      if (tupleIdx < numTuples) {
        g_in_int[j] = s_in_int[j];
      }
    }
    __syncthreads();
  }
}

/**
 * \brief Computes the new index of non-migrating tuples
 *
 * Most tuples do not move into new patches, so we will compute their new index first
 * using a scan operation. There can be a large number of tuples per patch; more than 
 * the number of atoms per patch. So we do the scan operation in batches while still 
 * using the CUB block API. 
 * 
 */
__global__ void reserveLocalDestinationKernel(
  const int               myDeviceIndex,
  const int               numPatchesHome,
  const int*              d_device,
  const int*              d_downstreamPatchIndex,
  const int*              d_offsets,
  int*                    d_dstIndex,
  int*                    d_counts
) {
  constexpr int kNumThreads = MigrationBondedCUDAKernel::kScanThreads;
  constexpr int kItemsPerThread = MigrationBondedCUDAKernel::kScanItemsPerThread;
  constexpr int kItemsPerIteration = kNumThreads * kItemsPerThread;

  typedef cub::BlockScan<int, kNumThreads> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int patchIndex = blockIdx.x; patchIndex < numPatchesHome; patchIndex += gridDim.x) {
    const int offset = d_offsets[patchIndex];
    const int numTuples = d_offsets[patchIndex+1] - offset;
    
    const int numIterations = (numTuples + kItemsPerIteration - 1) / kItemsPerIteration;

    // We will do the scan in pieces in case there are too many tuples
    int input_prefix = 0;
    for (int iter = 0; iter < numIterations; iter++) {
      int thread_input[kItemsPerThread];
      int thread_output[kItemsPerThread];
      int sum;

      // Load the batch into registers
      for (int i = 0; i < kItemsPerThread; i++) {
        const int idx = iter * kItemsPerIteration + kItemsPerThread * threadIdx.x + i;
        if (idx < numTuples && d_downstreamPatchIndex[offset + idx] == patchIndex 
                            && d_device[offset + idx] == myDeviceIndex) {
          thread_input[i] = 1;
        } else {
          thread_input[i] = 0;
        }
      }

      // Compute the prefix sum
      BlockScan(temp_storage).ExclusiveSum(thread_input, thread_output, sum);
      __syncthreads();

      // Write the output
      for (int i = 0; i < kItemsPerThread; i++) {
        const int idx = iter * kItemsPerIteration + kItemsPerThread * threadIdx.x + i;
        if (idx < numTuples && thread_input[i]) {
          d_dstIndex[offset + idx] = thread_output[i] + input_prefix;
        } 
      }
      input_prefix += sum;
      __syncthreads();
    }
    // We then need to set the count for this patch
    if (threadIdx.x == 0) {
      d_counts[patchIndex] = input_prefix;
    }
    __syncthreads();
  }
}

/**
 * \brief Computes the destination for all tuples and computes the new index for local
 *
 * This will call the following kernels for each type type: 
 *  - computeTupleDestinationKernel() \copybrief computeTupleDestinationKernel()
 *  - reserveLocalDestinationKernel() \copybrief reserveLocalDestinationKernel()
 *
 */
void MigrationBondedCUDAKernel::computeTupleDestination(
  const int               myDeviceIndex,
  TupleCounts             count,
  const int               numPatchesHome,
  const int4*             migrationDestination,
  const int*              patchIDtoGPU,
  const int*              patchIDtoHomePatchIndex,
  const int               aDim,
  const int               bDim,
  const int               cMaxIndex, 
  const int               bMaxIndex, 
  const int               aMaxIndex,
  cudaStream_t            stream
) {
  constexpr int numThreads = MigrationBondedCUDAKernel::kNumThreads;
  int numBlocks;
  
  #define CALL(fieldName, typeName) do { \
    numBlocks = (count.fieldName + numThreads - 1) / numThreads; \
    if (numBlocks) { \
      computeTupleDestinationKernel<typeName><<<numBlocks, numThreads, 0, stream>>>( \
        count.fieldName, \
        dataSrc.fieldName, \
        d_device.fieldName(), \
        d_downstreamPatchIndex.fieldName(), \
        migrationDestination, \
        patchIDtoGPU, \
        patchIDtoHomePatchIndex, \
        aDim, bDim, \
        cMaxIndex, bMaxIndex, aMaxIndex \
      ); \
    } \
  } while (0);

  CALL(bond, CudaBondStage);
  CALL(angle, CudaAngleStage);
  CALL(dihedral, CudaDihedralStage);
  CALL(improper, CudaDihedralStage);
  CALL(modifiedExclusion, CudaExclusionStage);
  CALL(exclusion, CudaExclusionStage);
  CALL(crossterm, CudaCrosstermStage);

  #undef CALL

  #define CALL(fieldName, typeName) do { \
    numBlocks = numPatchesHome; \
    reserveLocalDestinationKernel<<<numBlocks, kScanThreads, 0, stream>>>( \
      myDeviceIndex, \
      numPatchesHome, \
      d_device.fieldName(), \
      d_downstreamPatchIndex.fieldName(), \
      d_offsets.fieldName(), \
      d_dstIndex.fieldName(), \
      d_counts.fieldName() \
    ); \
  } while (0);

  CALL(bond, CudaBondStage);
  CALL(angle, CudaAngleStage);
  CALL(dihedral, CudaDihedralStage);
  CALL(improper, CudaDihedralStage);
  CALL(modifiedExclusion, CudaExclusionStage);
  CALL(exclusion, CudaExclusionStage);
  CALL(crossterm, CudaCrosstermStage);

  #undef CALL
}

/**
 * \brief Computes the new index of migrating tuples
 *
 * Using atomic operations to compute the new index for tuples which have migrated
 * into new patches.
 *
 */
__global__ void reserveTupleDestinationKernel(
  const int               myDeviceIndex,
  const int               numPatchesHome,
  const int*              d_device,
  const int*              d_downstreamPatchIndex,
  const int*              d_offsets,
  int*                    d_dstIndex,
  int**                   peer_counts
) {
  for (int patchIndex = blockIdx.x; patchIndex < numPatchesHome; patchIndex += gridDim.x) {
    const int offset = d_offsets[patchIndex];
    const int numTuples = d_offsets[patchIndex+1] - offset;

    for (int i = threadIdx.x; i < numTuples; i += blockDim.x) {
      const int downstreamPatchIndex = d_downstreamPatchIndex[offset + i];
      const int device = d_device[offset + i];
      if (downstreamPatchIndex != patchIndex || device != myDeviceIndex) {
        const int dstIndex =
#if __CUDA_ARCH__ >= 600
          atomicAdd_system(&(peer_counts[device][downstreamPatchIndex]), 1);
#else
          // support single-GPU Maxwell
          atomicAdd(&(peer_counts[device][downstreamPatchIndex]), 1);
#endif
        d_dstIndex[offset + i] = dstIndex;
      }
    }
    __syncthreads();
  }
}

/**
 * \brief Computes the index of migrating tuples with atomics
 *
 * This will call the following kernels for each type type: 
 *  - reserveTupleDestinationKernel() \copybrief reserveTupleDestinationKernel()
 *
 */
void MigrationBondedCUDAKernel::reserveTupleDestination(
  const int               myDeviceIndex,
  const int               numPatchesHome,
  cudaStream_t            stream
) {
  constexpr int numThreads = MigrationBondedCUDAKernel::kPatchThreads;
  int numBlocks;

  #define CALL(fieldName, typeName) do { \
    numBlocks = numPatchesHome; \
    reserveTupleDestinationKernel<<<numBlocks, numThreads, 0, stream>>>( \
      myDeviceIndex, \
      numPatchesHome, \
      d_device.fieldName(), \
      d_downstreamPatchIndex.fieldName(), \
      d_offsets.fieldName(), \
      d_dstIndex.fieldName(), \
      d_peer_counts.fieldName \
    ); \
  } while (0);

  CALL(bond, CudaBondStage);
  CALL(angle, CudaAngleStage);
  CALL(dihedral, CudaDihedralStage);
  CALL(improper, CudaDihedralStage);
  CALL(modifiedExclusion, CudaExclusionStage);
  CALL(exclusion, CudaExclusionStage);
  CALL(crossterm, CudaCrosstermStage);

  #undef CALL
}

/**
 * \brief Gathers the per-device tuple counts so they can be transfers together
 *
 */
__global__ void updateTotalCount(
  const int                numPatchesHome,
  TupleIntArraysContiguous d_offsets,
  TupleCounts*             d_totalCounts
) {
  d_totalCounts[0].bond = d_offsets.bond()[numPatchesHome];
  d_totalCounts[0].angle = d_offsets.angle()[numPatchesHome];
  d_totalCounts[0].dihedral = d_offsets.dihedral()[numPatchesHome];
  d_totalCounts[0].improper = d_offsets.improper()[numPatchesHome];
  d_totalCounts[0].modifiedExclusion = d_offsets.modifiedExclusion()[numPatchesHome];
  d_totalCounts[0].exclusion = d_offsets.exclusion()[numPatchesHome];
}

/**
 * \brief Computes the patch-level offsets the tuples
 *
 * This uses the block API because for most systems the number of patches does not
 * motivate the use of the device API.
 *
 */
__global__ void threadBlockPatchScan(
  const int                numPatchesHome,
  TupleIntArraysContiguous d_counts,
  TupleIntArraysContiguous d_offsets,
  TupleCounts*             d_totalCounts
) {
  constexpr int kItemsPerThread = MigrationBondedCUDAKernel::kPatchItemsPerThread;

  typedef cub::BlockScan<int, MigrationBondedCUDAKernel::kPatchThreads> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int* input = d_counts.data + d_counts.offsets[blockIdx.x];
  int* output = d_offsets.data + d_offsets.offsets[blockIdx.x];

  int r_input[kItemsPerThread];
  int r_output[kItemsPerThread];

  for (int i = 0; i < kItemsPerThread; i++) {
    const int idx = kItemsPerThread * threadIdx.x + i;
    if (idx < numPatchesHome) {
      r_input[i] = input[idx];
    } else {
      r_input[i] = 0;
    }
  }
  __syncthreads();

  // Compute the prefix sum of solvent
  int total;
  BlockScan(temp_storage).InclusiveSum(r_input, r_output, total);
  __syncthreads();

  for (int i = 0; i < kItemsPerThread; i++) {
    const int idx = kItemsPerThread * threadIdx.x + i + 1;
    if (idx < numPatchesHome+1) {
      output[idx] = r_output[i];
    }
  }
  
  if (threadIdx.x == 0) {
    int* int_totalCounts = (int*) d_totalCounts;
    int_totalCounts[blockIdx.x] = total;
  }
}

/**
 * \brief Computes the patch-level offsets the tuples
 *
 * Depending on the number of home patches we will either use the CUB device API
 * or the CUB block API. The block API has less overhead since it is one in a single
 * kernel whereas the device API launches 2 kernels per tuple type
 *
 * If the single kernel path is taken, it will call the following kernel:
 * - threadBlockPatchScan() \copybrief threadBlockPatchScan()
 *
 */
void MigrationBondedCUDAKernel::computePatchOffsets(
  const int               numPatchesHome,
  cudaStream_t            stream
) {
  if (numPatchesHome > kMaxPatchesForSingleThreadBlock) {
    #define CALL(fieldName) do { \
      cub::DeviceScan::InclusiveSum( \
        d_patchDeviceScan_scratch, temp, \
        d_counts.fieldName(), d_offsets.fieldName()+1, numPatchesHome, stream \
      ); \
    } while (0);

    size_t temp = patchDeviceScan_alloc;
    CALL(bond);
    CALL(angle);
    CALL(dihedral);
    CALL(improper);
    CALL(modifiedExclusion);
    CALL(exclusion);
    CALL(crossterm);

    #undef CALL

    updateTotalCount<<<1, 1, 0, stream>>>(numPatchesHome, d_offsets, d_totalCount);
  } else {
    constexpr int numThreads = kPatchThreads;
    const int numBlocks = kNumTuplesTypes;

    threadBlockPatchScan<<<numBlocks, numThreads, 0, stream>>>(
      numPatchesHome,
      d_counts,
      d_offsets,
      d_totalCount
    );
  }
}

/**
 * \brief Performs actual tuple migration
 *
 * Based on previously computed tuple devices, patch index, patch offsets, and
 * tuple index, does the actual tuple migrations. The tuples are stored as AoS so
 * we try to use as coalesced memory access as possible. This does leave some 
 * threads unused.
 *
 * This copies the tuples into a scratch buffer
 *
 */
template<typename T>
__global__ void performTupleMigrationKernel(
  const int               numTuples,
  const int* const*       peer_offsets,
  T*                      src,
  T**                     dst,
  int*                    d_device,
  int*                    d_downstreamPatchIndex,
  int*                    d_dstIndex
) {
  using AccessType = int32_t;
  static_assert(sizeof(T) % sizeof(AccessType) == 0, "Tuples must be divisible by accesstype");
  constexpr int kThreadsPerTuple = sizeof(T) / sizeof(AccessType);
  constexpr int kTuplesPerWarp = WARPSIZE / kThreadsPerTuple;
  
  const int numTuplesWarp = ((numTuples + kTuplesPerWarp - 1) / kTuplesPerWarp);
  const int numWarps = (blockDim.x * gridDim.x) / WARPSIZE;
  const int wid_grid = (threadIdx.x + blockDim.x * blockIdx.x) / WARPSIZE;
  const int lid = threadIdx.x % WARPSIZE;
  const int indexInTuple = lid % kThreadsPerTuple;
  const int tupleWarpIndex = lid / kThreadsPerTuple;

  for (int i = wid_grid; i < numTuplesWarp; i += numWarps) {
    const int tupleIndex = i * kTuplesPerWarp + tupleWarpIndex;
    const int active = (lid < kThreadsPerTuple * kTuplesPerWarp);

    if (active && tupleIndex < numTuples) {
      const int device = d_device[tupleIndex];
      const int index = d_dstIndex[tupleIndex];
      const int patchIndex = d_downstreamPatchIndex[tupleIndex];

      // Hopefully this is cached to save us from lots of P2P accesses
      const int offset = peer_offsets[device][patchIndex];

      AccessType* g_src_int = (AccessType*) &(src[tupleIndex]);
      AccessType* g_dst_int = (AccessType*) &(dst[device][offset + index]);
      g_dst_int[indexInTuple] = g_src_int[indexInTuple];
    }
#if defined(NAMD_HIP)
    NAMD_WARP_SYNC(WARP_FULL_MASK);
#else
    WARP_SYNC(WARP_FULL_MASK);
#endif
  }
}


/**
 * \brief Moves tuples to their updated patches scratch buffers
 *
 * This will call the following kernels for each type type: 
 *  - performTupleMigrationKernel() \copybrief performTupleMigrationKernel()
 *
 */
void MigrationBondedCUDAKernel::performTupleMigration(
  TupleCounts             count,
  cudaStream_t            stream
) {
  constexpr int numThreads = MigrationBondedCUDAKernel::kNumThreads;
  int numBlocks;
  
  #define CALL(fieldName, typeName) do { \
    numBlocks = (count.fieldName + numThreads - 1) / numThreads; \
    if (numBlocks) { \
      performTupleMigrationKernel<typeName><<<numBlocks, numThreads, 0, stream>>>( \
        count.fieldName, \
        d_peer_offsets.fieldName, \
        dataSrc.fieldName, \
        d_peer_data.fieldName, \
        d_device.fieldName(), \
        d_downstreamPatchIndex.fieldName(), \
        d_dstIndex.fieldName() \
      ); \
    } \
  } while (0);

  CALL(bond, CudaBondStage);
  CALL(angle, CudaAngleStage);
  CALL(dihedral, CudaDihedralStage);
  CALL(improper, CudaDihedralStage);
  CALL(modifiedExclusion, CudaExclusionStage);
  CALL(exclusion, CudaExclusionStage);
  CALL(crossterm, CudaCrosstermStage);

  #undef CALL
}


template <typename T_in, typename T_out>
__device__ __forceinline__
T_out convertTuple(
  T_in& src,
  const PatchRecord* patches,
  const Lattice lattice,
  const double3* __restrict__ patchMapCenter,
  const float4* __restrict__ xyzq);


template <>
__device__ __forceinline__
CudaBond convertTuple(
  CudaBondStage& src,
  const PatchRecord* patches,
  const Lattice lattice,
  const double3* __restrict__ patchMapCenter,
  const float4* __restrict__ xyzq
) {
  CudaBond dst;

  dst.itype = src.itype;
  dst.scale = src.scale;
  dst.fepBondedType = src.fepBondedType;

  dst.i = src.index[0];
  dst.j = src.index[1];
  
  double3 center0 = patchMapCenter[src.patchIDs[0]];
  double3 center1 = patchMapCenter[src.patchIDs[1]];
  double3 diff = center0 - center1;

  double3 pos0 = make_double3(xyzq[dst.i]) + (double3) lattice.unscale(center0);
  double3 pos1 = make_double3(xyzq[dst.j]) + (double3) lattice.unscale(center1);

  double3 shiftVec = lattice.wrap_delta_scaled(pos0, pos1);

  shiftVec = shiftVec + diff;

  dst.ioffsetXYZ = make_float3(shiftVec);

  return dst;
}

template <>
__device__ __forceinline__
CudaAngle convertTuple(
  CudaAngleStage& src,
  const PatchRecord* patches,
  const Lattice lattice,
  const double3* __restrict__ patchMapCenter,
  const float4* __restrict__ xyzq) {

  CudaAngle dst;

  dst.itype = src.itype;
  dst.scale = src.scale;
  dst.fepBondedType = src.fepBondedType;

  dst.i = src.index[0];
  dst.j = src.index[1];
  dst.k = src.index[2];

  double3 center0 = patchMapCenter[src.patchIDs[0]];
  double3 center1 = patchMapCenter[src.patchIDs[1]];
  double3 center2 = patchMapCenter[src.patchIDs[2]];

  double3 diff01 = center0 - center1;
  double3 diff21 = center2 - center1;

  double3 pos0 = make_double3(xyzq[dst.i]) + (double3) lattice.unscale(center0);
  double3 pos1 = make_double3(xyzq[dst.j]) + (double3) lattice.unscale(center1);
  double3 pos2 = make_double3(xyzq[dst.k]) + (double3) lattice.unscale(center2);

  double3 shiftVec01 = lattice.wrap_delta_scaled(pos0, pos1);
  double3 shiftVec21 = lattice.wrap_delta_scaled(pos2, pos1);
  shiftVec01 = shiftVec01 + diff01;
  shiftVec21 = shiftVec21 + diff21;

  dst.ioffsetXYZ = make_float3(shiftVec01);
  dst.koffsetXYZ = make_float3(shiftVec21);

  return dst;
}

template <>
__device__ __forceinline__
CudaDihedral convertTuple(
  CudaDihedralStage& src,
  const PatchRecord* patches,
  const Lattice lattice,
  const double3* __restrict__ patchMapCenter,
  const float4* __restrict__ xyzq) {

  CudaDihedral dst;

  dst.itype = src.itype;
  dst.scale = src.scale;
  dst.fepBondedType = src.fepBondedType;

  dst.i = src.index[0];
  dst.j = src.index[1];
  dst.k = src.index[2];
  dst.l = src.index[3];

  double3 center0 = patchMapCenter[src.patchIDs[0]];
  double3 center1 = patchMapCenter[src.patchIDs[1]];
  double3 center2 = patchMapCenter[src.patchIDs[2]];
  double3 center3 = patchMapCenter[src.patchIDs[3]];

  double3 diff01 = center0 - center1;
  double3 diff12 = center1 - center2;
  double3 diff32 = center3 - center2;

  double3 pos0 = make_double3(xyzq[dst.i]) + (double3) lattice.unscale(center0);
  double3 pos1 = make_double3(xyzq[dst.j]) + (double3) lattice.unscale(center1);
  double3 pos2 = make_double3(xyzq[dst.k]) + (double3) lattice.unscale(center2);
  double3 pos3 = make_double3(xyzq[dst.l]) + (double3) lattice.unscale(center3);

  double3 shiftVec01 = lattice.wrap_delta_scaled(pos0, pos1);
  double3 shiftVec12 = lattice.wrap_delta_scaled(pos1, pos2);
  double3 shiftVec32 = lattice.wrap_delta_scaled(pos3, pos2);
  shiftVec01 = shiftVec01 + diff01;
  shiftVec12 = shiftVec12 + diff12;
  shiftVec32 = shiftVec32 + diff32;

  dst.ioffsetXYZ = make_float3(shiftVec01);
  dst.joffsetXYZ = make_float3(shiftVec12);
  dst.loffsetXYZ = make_float3(shiftVec32);

  return dst;
}

template <>
__device__ __forceinline__
CudaExclusion convertTuple(
  CudaExclusionStage& src,
  const PatchRecord* patches,
  const Lattice lattice,
  const double3* __restrict__ patchMapCenter,
  const float4* __restrict__ xyzq) {

  CudaExclusion dst;

  dst.vdwtypei = src.vdwtypei;
  dst.vdwtypej = src.vdwtypej;
  dst.pswitch = src.pswitch;

  dst.i = src.index[0];
  dst.j = src.index[1];

  double3 center0 = patchMapCenter[src.patchIDs[0]];
  double3 center1 = patchMapCenter[src.patchIDs[1]];

  double3 diff = center0 - center1;

  double3 pos0 = make_double3(xyzq[dst.i]) + (double3) lattice.unscale(center0);
  double3 pos1 = make_double3(xyzq[dst.j]) + (double3) lattice.unscale(center1);
  
  double3 shiftVec = lattice.wrap_delta_scaled(pos0, pos1);
  shiftVec = shiftVec + diff;

  dst.ioffsetXYZ = make_float3(shiftVec);

  return dst;
}

template <>
__device__ __forceinline__
CudaCrossterm convertTuple(
  CudaCrosstermStage& src,
  const PatchRecord* patches,
  const Lattice lattice,
  const double3* __restrict__ patchMapCenter,
  const float4* __restrict__ xyzq) {

  CudaCrossterm dst;

  dst.itype = src.itype;
  dst.scale = src.scale;
  dst.fepBondedType = src.fepBondedType;

  dst.i1 = src.index[0];
  dst.i2 = src.index[1];
  dst.i3 = src.index[2];
  dst.i4 = src.index[3];
  dst.i5 = src.index[4];
  dst.i6 = src.index[5];
  dst.i7 = src.index[6];
  dst.i8 = src.index[7];

  double3 center0 = patchMapCenter[src.patchIDs[0]];
  double3 center1 = patchMapCenter[src.patchIDs[1]];
  double3 center2 = patchMapCenter[src.patchIDs[2]];
  double3 center3 = patchMapCenter[src.patchIDs[3]];
  double3 center4 = patchMapCenter[src.patchIDs[4]];
  double3 center5 = patchMapCenter[src.patchIDs[5]];
  double3 center6 = patchMapCenter[src.patchIDs[6]];
  double3 center7 = patchMapCenter[src.patchIDs[7]];

  double3 diff01 = center0 - center1;
  double3 diff12 = center1 - center2;
  double3 diff23 = center2 - center3;
  double3 diff45 = center4 - center5;
  double3 diff56 = center5 - center6;
  double3 diff67 = center6 - center7;

  double3 pos0 = make_double3(xyzq[dst.i1]) + (double3) lattice.unscale(center0);
  double3 pos1 = make_double3(xyzq[dst.i2]) + (double3) lattice.unscale(center1);
  double3 pos2 = make_double3(xyzq[dst.i3]) + (double3) lattice.unscale(center2);
  double3 pos3 = make_double3(xyzq[dst.i4]) + (double3) lattice.unscale(center3);
  double3 pos4 = make_double3(xyzq[dst.i5]) + (double3) lattice.unscale(center4);
  double3 pos5 = make_double3(xyzq[dst.i6]) + (double3) lattice.unscale(center5);
  double3 pos6 = make_double3(xyzq[dst.i7]) + (double3) lattice.unscale(center6);
  double3 pos7 = make_double3(xyzq[dst.i8]) + (double3) lattice.unscale(center7);

  double3 shiftVec01 = lattice.wrap_delta_scaled(pos0, pos1);
  double3 shiftVec12 = lattice.wrap_delta_scaled(pos1, pos2);
  double3 shiftVec23 = lattice.wrap_delta_scaled(pos2, pos3);
  double3 shiftVec45 = lattice.wrap_delta_scaled(pos4, pos5);
  double3 shiftVec56 = lattice.wrap_delta_scaled(pos5, pos6);
  double3 shiftVec67 = lattice.wrap_delta_scaled(pos6, pos7);

  shiftVec01 = shiftVec01 + diff01;
  shiftVec12 = shiftVec12 + diff12;
  shiftVec23 = shiftVec23 + diff23;
  shiftVec45 = shiftVec45 + diff45;
  shiftVec56 = shiftVec56 + diff56;
  shiftVec67 = shiftVec67 + diff67;

  dst.offset12XYZ = make_float3(shiftVec01);
  dst.offset23XYZ = make_float3(shiftVec12);
  dst.offset34XYZ = make_float3(shiftVec23);
  dst.offset56XYZ = make_float3(shiftVec45);
  dst.offset67XYZ = make_float3(shiftVec56);
  dst.offset78XYZ = make_float3(shiftVec67);

  return dst;
}



template<typename T>
__device__  __forceinline__
void updateStageData(
  T&                      tuple,
  const PatchRecord*      patches,
  const int*              ids
) {
  for (int i = 0; i < T::size; i++) {
    // The patchID and ID are correct
    const int patchID = tuple.patchIDs[i];

    PatchRecord record = patches[patchID]; // This structure is indexed by global patch ID
                                           // And it contains the LOCAL offset...
    const int offset = record.atomStart;
    tuple.index[i] += offset;
  }
}

/**
 * \brief Updates the tuple stage data and constructions the real tuple data
 *
 * Similar to the other kernels, this one will bulk read tuple data to coalesce 
 * memory access. 
 *
 * The stage update is done by the updateStageData function, it will compute the
 * new indexInDevice indces of each atom based on the atom's index within its patch
 * which is currently stored in the `index` field and that patches buffer offset
 * in the device's packed buffers.
 *
 * The convertTuple function converts the stage tuple to a real tuple.
 *
 */
template <typename T_in, typename T_out>
__global__ void updateTuplesKernel(
  const int               numTuples,
  T_in*                   stages_src,
  T_in*                   stages_dst,
  T_out*                  dst,
  const int*              ids,
  const PatchRecord*      patches,
  const double3* __restrict__ patchMapCenter,
  const float4* __restrict__ xyzq,
  const Lattice       lattice
) {
  using AccessType = int32_t;

  __shared__ T_in s_in[MigrationBondedCUDAKernel::kNumThreads];
  __shared__ T_out s_out[MigrationBondedCUDAKernel::kNumThreads];

  AccessType* s_in_int = (AccessType*) &(s_in);
  AccessType* s_out_int = (AccessType*) &(s_out);

  const int numTuplesBlock = ((numTuples + blockDim.x - 1) / blockDim.x) * blockDim.x;

  for (int i = blockDim.x * blockIdx.x; i < numTuplesBlock; i += blockDim.x * gridDim.x) {
    // Load in tuples to shared memory
    AccessType* g_in_int = (AccessType*) (stages_src + i);
    const int numBytesIn = sizeof(T_in) * blockDim.x;
    for (int j = threadIdx.x; j < numBytesIn / sizeof(AccessType); j += blockDim.x) {
      const int tupleIdx = i + (j / (sizeof(T_in) / sizeof(AccessType)));
      if (tupleIdx < numTuples) {
        s_in_int[j] = g_in_int[j];
      }
    }
    __syncthreads();
    
    if (i + threadIdx.x < numTuples) {
      T_in src = s_in[threadIdx.x];
      updateStageData(src, patches, ids);
      T_out r_out = convertTuple<T_in, T_out>(src, patches, lattice, patchMapCenter, xyzq);
      s_out[threadIdx.x] = r_out;
      s_in[threadIdx.x] = src;
    }
    __syncthreads();

    // Write tuples out
    AccessType* g_out_int = (AccessType*) (stages_dst + i);
    for (int j = threadIdx.x; j < numBytesIn / sizeof(AccessType); j += blockDim.x) {
      const int tupleIdx = i + (j / (sizeof(T_in) / sizeof(AccessType)));
      if (tupleIdx < numTuples) {
        g_out_int[j] = s_in_int[j];
      }
    }

    AccessType* g_outData_int = (AccessType*) (dst + i);
    const int numBytesOut = sizeof(T_out) * blockDim.x;
    for (int j = threadIdx.x; j < numBytesOut / sizeof(AccessType); j += blockDim.x) {
      const int tupleIdx = i + (j / (sizeof(T_out) / sizeof(AccessType)));
      if (tupleIdx < numTuples) {
        g_outData_int[j] = s_out_int[j];
      }
    }
    __syncthreads();
  } 
}

/**
 * \brief Updates the tuples once they have been migrated
 *
 *
 * This will call the following kernels for each type type: 
 *  - updateTuplesKernel() \copybrief updateTuplesKernel()
 *
 */
void MigrationBondedCUDAKernel::updateTuples(
  TupleCounts             count,
  TupleData               data,
  const int*              ids,
  const PatchRecord*      patches,
  const double3*          d_patchMapCenter,
  const float4*           xyzq,
  const Lattice       lattice,
  cudaStream_t            stream
) {
  constexpr int numThreads = MigrationBondedCUDAKernel::kNumThreads;
  int numBlocks;
  
  #define CALL(fieldName, typeNameIn, typeNameOut) do { \
    numBlocks = (count.fieldName + numThreads - 1) / numThreads; \
    if (numBlocks) { \
      updateTuplesKernel<typeNameIn, typeNameOut><<<numBlocks, numThreads, 0, stream>>>( \
        count.fieldName, \
        dataDst.fieldName, \
        dataSrc.fieldName, \
        data.fieldName, \
        ids, \
        patches, \
        d_patchMapCenter, \
        xyzq, \
        lattice \
      ); \
    } \
  } while (0);

  CALL(bond, CudaBondStage, CudaBond);
  CALL(angle, CudaAngleStage, CudaAngle);
  CALL(dihedral, CudaDihedralStage, CudaDihedral);
  CALL(improper, CudaDihedralStage, CudaDihedral);
  CALL(modifiedExclusion, CudaExclusionStage, CudaExclusion);
  CALL(exclusion, CudaExclusionStage, CudaExclusion);
  CALL(crossterm, CudaCrosstermStage, CudaCrossterm);

  #undef CALL
}


void MigrationBondedCUDAKernel::copyTupleToDevice(
  TupleCounts             count,
  const int               numPatchesHome,
  TupleDataStage          h_dataStage,
  TupleIntArrays          h_counts,
  TupleIntArrays          h_offsets,
  cudaStream_t            stream
) {
  #define CALL(fieldName, typeName) do { \
    copy_HtoD<typeName>(h_dataStage.fieldName, dataDst.fieldName, count.fieldName, stream); \
    copy_HtoD<int>(h_counts.fieldName, d_counts.fieldName(), numPatchesHome, stream); \
    copy_HtoD<int>(h_offsets.fieldName, d_offsets.fieldName(), numPatchesHome+1, stream); \
  } while (0);

  CALL(bond, CudaBondStage);
  CALL(angle, CudaAngleStage);
  CALL(dihedral, CudaDihedralStage);
  CALL(improper, CudaDihedralStage);
  CALL(modifiedExclusion, CudaExclusionStage);
  CALL(exclusion, CudaExclusionStage);
  CALL(crossterm, CudaCrosstermStage);

  #undef CALL
}


bool MigrationBondedCUDAKernel::reallocateBufferSrc(TupleCounts counts) {
  bool out = false;

  constexpr float kTupleOveralloc = ComputeBondedCUDAKernel::kTupleOveralloc;

  #define CALL(fieldName, typeName) do { \
    const size_t paddedCount = ComputeBondedCUDAKernel::warpAlign(counts.fieldName); \
    if (paddedCount) { \
      out |= reallocate_device<typeName>(&(dataSrc.fieldName), &(srcAlloc.fieldName), \
                                         paddedCount, kTupleOveralloc); \
    } \
  } while (0);

  CALL(bond, CudaBondStage);
  CALL(angle, CudaAngleStage);
  CALL(dihedral, CudaDihedralStage);
  CALL(improper, CudaDihedralStage);
  CALL(modifiedExclusion, CudaExclusionStage);
  CALL(exclusion, CudaExclusionStage);
  CALL(crossterm, CudaCrosstermStage);

  #undef CALL

  const size_t totalSize = ComputeBondedCUDAKernel::warpAlign(counts.bond) 
                         + ComputeBondedCUDAKernel::warpAlign(counts.angle)
                         + ComputeBondedCUDAKernel::warpAlign(counts.dihedral)
                         + ComputeBondedCUDAKernel::warpAlign(counts.improper)
                         + ComputeBondedCUDAKernel::warpAlign(counts.modifiedExclusion)
                         + ComputeBondedCUDAKernel::warpAlign(counts.exclusion)
                         + ComputeBondedCUDAKernel::warpAlign(counts.crossterm);
  
  size_t temp = srcTotalAlloc;
  reallocate_device<int>(&(d_device.data),               &temp, totalSize, kTupleOveralloc);
  temp = srcTotalAlloc;
  reallocate_device<int>(&(d_downstreamPatchIndex.data), &temp, totalSize, kTupleOveralloc);
  temp = srcTotalAlloc;
  reallocate_device<int>(&(d_dstIndex.data),             &temp, totalSize, kTupleOveralloc);
  srcTotalAlloc = temp;

  size_t offset = 0;
  
  #define CALL(fieldName, typeName, typeIndex) do { \
    d_device.offsets[typeIndex] = offset; \
    d_dstIndex.offsets[typeIndex] = offset; \
    d_downstreamPatchIndex.offsets[typeIndex] = offset; \
    offset += (ComputeBondedCUDAKernel::warpAlign(counts.fieldName)); \
  } while (0);

  CALL(bond, CudaBondStage, 0);
  CALL(angle, CudaAngleStage, 1);
  CALL(dihedral, CudaDihedralStage, 2);
  CALL(improper, CudaDihedralStage, 3);
  CALL(modifiedExclusion, CudaExclusionStage, 4);
  CALL(exclusion, CudaExclusionStage, 5);
  CALL(crossterm, CudaCrosstermStage, 6);

  #undef CALL

  return out;
}


bool MigrationBondedCUDAKernel::reallocateBufferDst(TupleCounts counts) {
  constexpr float kTupleOveralloc = ComputeBondedCUDAKernel::kTupleOveralloc;
  bool out = false;

  #define CALL(fieldName, typeName) do { \
    const size_t paddedCount = ComputeBondedCUDAKernel::warpAlign(counts.fieldName); \
    if (paddedCount) { \
      out |= reallocate_device<typeName>(&(dataDst.fieldName), &(dstAlloc.fieldName), \
                                         paddedCount, kTupleOveralloc); \
    } \
  } while (0);

  CALL(bond, CudaBondStage);
  CALL(angle, CudaAngleStage);
  CALL(dihedral, CudaDihedralStage);
  CALL(improper, CudaDihedralStage);
  CALL(modifiedExclusion, CudaExclusionStage);
  CALL(exclusion, CudaExclusionStage);
  CALL(crossterm, CudaCrosstermStage);

  #undef CALL

  return out;
}


MigrationBondedCUDAKernel::MigrationBondedCUDAKernel() {
  #define CALL(fieldName) do { \
    srcAlloc.fieldName = 0; \
    dstAlloc.fieldName = 0; \
    dataSrc.fieldName = NULL; \
    dataDst.fieldName = NULL; \
    d_peer_data.fieldName = NULL; \
    d_peer_counts.fieldName = NULL; \
    d_peer_offsets.fieldName = NULL; \
  } while (0);

  CALL(bond);
  CALL(angle);
  CALL(dihedral);
  CALL(improper);
  CALL(modifiedExclusion);
  CALL(exclusion);
  CALL(crossterm);

  #undef CALL
  
  srcTotalAlloc = 0;
  d_device.data = NULL;
  d_downstreamPatchIndex.data = NULL;
  d_dstIndex.data = NULL;
  
  d_counts.data = NULL;
  d_offsets.data = NULL;

  patchDeviceScan_alloc = 0;
  d_patchDeviceScan_scratch = NULL;

  allocate_device<TupleCounts>(&d_totalCount, 1);
  allocate_host<TupleCounts>(&h_totalCount, 1);
}

MigrationBondedCUDAKernel::~MigrationBondedCUDAKernel() {
  #define CALL(fieldName, typeName) do { \
    if (dataSrc.fieldName != NULL) deallocate_device<typeName>(&(dataSrc.fieldName)); \
    if (dataDst.fieldName != NULL) deallocate_device<typeName>(&(dataDst.fieldName)); \
    if (d_peer_data.fieldName != NULL) deallocate_device<typeName*>(&(d_peer_data.fieldName)); \
    if (d_peer_counts.fieldName != NULL) deallocate_device<int*>(&(d_peer_counts.fieldName)); \
    if (d_peer_offsets.fieldName != NULL) deallocate_device<int*>(&(d_peer_offsets.fieldName)); \
  } while (0);

  CALL(bond, CudaBondStage);
  CALL(angle, CudaAngleStage);
  CALL(dihedral, CudaDihedralStage);
  CALL(improper, CudaDihedralStage);
  CALL(modifiedExclusion, CudaExclusionStage);
  CALL(exclusion, CudaExclusionStage);
  CALL(crossterm, CudaCrosstermStage);

  #undef CALL
  
  if (d_counts.data != NULL) deallocate_device<int>(&d_counts.data);
  if (d_offsets.data != NULL) deallocate_device<int>(&d_offsets.data);
  if (d_patchDeviceScan_scratch != NULL) deallocate_device<char>(&d_patchDeviceScan_scratch);

  if (d_device.data != NULL) deallocate_device<int>(&d_device.data);
  if (d_downstreamPatchIndex.data != NULL) deallocate_device<int>(&d_downstreamPatchIndex.data);
  if (d_dstIndex.data != NULL) deallocate_device<int>(&d_dstIndex.data);

  deallocate_device<TupleCounts>(&d_totalCount);
  deallocate_host<TupleCounts>(&h_totalCount);
}

void MigrationBondedCUDAKernel::setup(const int numDevices, const int numPatchesHome) {
  //
  // Allocate data behind the patch counters. Allocating numPatchesHome + 1 for
  // offsets
  //
  numPatchesHomePad = ((numPatchesHome + 1 + kPatchNumberPad - 1)
                      * kPatchNumberPad) / kPatchNumberPad;

  allocate_device<int>(&(d_counts.data), numPatchesHomePad * kNumTuplesTypes);
  allocate_device<int>(&(d_offsets.data), numPatchesHomePad * kNumTuplesTypes);
  cudaMemset(d_offsets.data, 0, numPatchesHomePad * kNumTuplesTypes * sizeof(int));


  #define CALL(fieldName, typeName, offset) do { \
    allocate_device<typeName*>(&(d_peer_data.fieldName), numDevices); \
    allocate_device<int*>(&(d_peer_counts.fieldName), numDevices); \
    allocate_device<int*>(&(d_peer_offsets.fieldName), numDevices); \
    d_counts.offsets[offset] = offset * numPatchesHomePad; \
    d_offsets.offsets[offset] = offset * numPatchesHomePad; \
  } while (0);

  CALL(bond, CudaBondStage, 0);
  CALL(angle, CudaAngleStage, 1);
  CALL(dihedral, CudaDihedralStage, 2);
  CALL(improper, CudaDihedralStage, 3);
  CALL(modifiedExclusion, CudaExclusionStage, 4);
  CALL(exclusion, CudaExclusionStage, 5);
  CALL(crossterm, CudaCrosstermStage, 6);

  #undef CALL


  //
  // Allocate space for CUB Device functions
  //
  size_t   temp_storage_bytes = 0;
  int* temp_ptr = NULL;
  cub::DeviceScan::InclusiveSum(
    NULL, temp_storage_bytes, 
    temp_ptr, temp_ptr, numPatchesHome
  );
  patchDeviceScan_alloc = temp_storage_bytes;
  allocate_device<char>(&d_patchDeviceScan_scratch, patchDeviceScan_alloc); 
}

void MigrationBondedCUDAKernel::clearTupleCounts(const int numPatchesHome, cudaStream_t stream) {
  cudaMemsetAsync(d_counts.data, 0, numPatchesHomePad * kNumTuplesTypes * sizeof(int), stream);
}

TupleCounts MigrationBondedCUDAKernel::fetchTupleCounts(const int numPatchesHome, cudaStream_t stream) {
    
  copy_DtoH<TupleCounts>(d_totalCount, h_totalCount, 1, stream); \

  cudaCheck(cudaStreamSynchronize(stream));
  TupleCounts newLocal;

  #define CALL(fieldName, typeName) do { \
    newLocal.fieldName = h_totalCount[0].fieldName; \
  } while (0);

  CALL(bond, CudaBondStage);
  CALL(angle, CudaAngleStage);
  CALL(dihedral, CudaDihedralStage);
  CALL(improper, CudaDihedralStage);
  CALL(modifiedExclusion, CudaExclusionStage);
  CALL(exclusion, CudaExclusionStage);
  CALL(crossterm, CudaCrosstermStage);

  #undef CALL
  
  return newLocal;
}

void MigrationBondedCUDAKernel::copyPeerDataToDevice(
  TupleDataStagePeer      h_peer_data,
  TupleIntArraysPeer      h_peer_counts,
  TupleIntArraysPeer      h_peer_offsets,
  const int               numDevices,
  cudaStream_t            stream
) {
  #define CALL(fieldName, typeName) do { \
    copy_HtoD<typeName*>(h_peer_data.fieldName, d_peer_data.fieldName, numDevices, stream); \
    copy_HtoD<int*>(h_peer_counts.fieldName, d_peer_counts.fieldName, numDevices, stream); \
    copy_HtoD<int*>(h_peer_offsets.fieldName, d_peer_offsets.fieldName, numDevices, stream); \
  } while (0);

  CALL(bond, CudaBondStage);
  CALL(angle, CudaAngleStage);
  CALL(dihedral, CudaDihedralStage);
  CALL(improper, CudaDihedralStage);
  CALL(modifiedExclusion, CudaExclusionStage);
  CALL(exclusion, CudaExclusionStage);
  CALL(crossterm, CudaCrosstermStage);

  #undef CALL
}

#endif
