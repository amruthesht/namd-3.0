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
#endif

#include "HipDefines.h"

#include "MigrationCUDAKernel.h"
#include "CudaComputeNonbondedKernel.h"
#include "CudaComputeNonbondedKernel.hip.h"

#ifdef NODEGROUP_FORCE_REGISTER

#define MAX_VALUE 2147483647
#define BIG_FLOAT 1e20
#define SMALL_FLOAT -1e20


__device__ __forceinline__ void singleBisect(
  const int               bit_pos,
  const int               bit_total,
  const double            min_dim,
  const double            max_dim,
  const double*           current_dim,
  const int               numAtoms,
  unsigned int*           s_indexBuffer,
  unsigned int            (&thread_start)[MigrationCUDAKernel::kValuesPerThread],
  unsigned int            (&thread_end)[MigrationCUDAKernel::kValuesPerThread],
  unsigned int            (&thread_out)[MigrationCUDAKernel::kValuesPerThread],
  unsigned int            (&thread_keys)[MigrationCUDAKernel::kValuesPerThread],
  unsigned int            (&thread_values)[MigrationCUDAKernel::kValuesPerThread]
) {
  constexpr int kValuesPerThread = MigrationCUDAKernel::kValuesPerThread;
  typedef cub::BlockRadixSort<unsigned int, MigrationCUDAKernel::kSortNumThreads, MigrationCUDAKernel::kValuesPerThread, unsigned int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_sort;


  for (int i = 0; i < kValuesPerThread; i++) {
    const int idx = kValuesPerThread * threadIdx.x + i;
    if (idx < numAtoms) {
        const unsigned int pos_norm = (unsigned int) ((current_dim[idx] - min_dim) / (max_dim - min_dim) * (1 << 18));
        thread_keys[i] = thread_out[i] | pos_norm;
        thread_values[i] = idx;
    } else {
        thread_keys[i] = ~0;
        thread_values[i] =  ~0;
    }
  }
  __syncthreads();

  BlockRadixSort(temp_sort).Sort(thread_keys, thread_values);
  __syncthreads();

  for (int i = 0; i < kValuesPerThread; i++) {
    const int idx = kValuesPerThread * threadIdx.x + i;
    if (thread_values[i] < numAtoms) {
      s_indexBuffer[thread_values[i]] = idx;
    }
  }
  __syncthreads();

  // Get sort index
  for (int i = 0; i < kValuesPerThread; i++) {
    const int idx = kValuesPerThread * threadIdx.x + i;
    if (idx < numAtoms) {
      const int newIndex = s_indexBuffer[idx];

      int split_factor = 32;
      int split = -1;
      while (split == -1) {
        if ( thread_start[i]/split_factor < (thread_end[i]-1)/split_factor ) {
          split = ((thread_start[i] + thread_end[i] + split_factor) / (split_factor*2)) * split_factor;
        }
        split_factor /= 2;
        if (split_factor == 1){
          split = ((thread_start[i] + thread_end[i] + split_factor) / (split_factor*2)) * split_factor;
        }
      }

      if (newIndex >= split) {
        thread_out[i] |= 1 << (30 - bit_pos);
        thread_start[i] = split;
      } else {
        thread_end[i] = split;
      }
      if (bit_pos == bit_total - 1) {
        const unsigned int pos_norm = (unsigned int) ((current_dim[idx] - min_dim) / (max_dim - min_dim) * (1 << 18));
        thread_out[i] |= pos_norm;
      }
    }
  }
  __syncthreads();
}


__global__ void computeBisect(
  const int               numPatches,
  const int               numGrids,
  const CudaLocalRecord*  localRecords,
  const double3*          patchMin,
  const double3*          patchMax,
  const double*           pos_x,
  const double*           pos_y,
  const double*           pos_z,
  int*                    sortOrder,
  int*                    sortIndex
) {
  constexpr int kValuesPerThread = MigrationCUDAKernel::kValuesPerThread;

  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  typedef cub::BlockReduce<double, MigrationCUDAKernel::kSortNumThreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ double3 s_pmin;
  __shared__ double3 s_pmax;


  __shared__ unsigned int s_indexBuffer[MigrationCUDAKernel::kMaxAtomsPerPatch];

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    #pragma unroll
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int offset = s_record.bufferOffset;

    // Compute the thread-local min/max values
    double3 pmin, pmax;
    pmin.x = BIG_FLOAT;
    pmin.y = BIG_FLOAT;
    pmin.z = BIG_FLOAT;
    pmax.x = SMALL_FLOAT;
    pmax.y = SMALL_FLOAT;
    pmax.z = SMALL_FLOAT;
    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
      pmin.x = fmin(pmin.x, pos_x[offset + i]);
      pmin.y = fmin(pmin.y, pos_y[offset + i]);
      pmin.z = fmin(pmin.z, pos_z[offset + i]);
      pmax.x = fmax(pmax.x, pos_x[offset + i]);
      pmax.y = fmax(pmax.y, pos_y[offset + i]);
      pmax.z = fmax(pmax.z, pos_z[offset + i]);
    }
    __syncthreads();
   
    // Compute the thread-block min/max values
    pmin.x = BlockReduce(temp_storage).Reduce(pmin.x, cub::Min());
    __syncthreads();
    pmin.y = BlockReduce(temp_storage).Reduce(pmin.y, cub::Min());
    __syncthreads();
    pmin.z = BlockReduce(temp_storage).Reduce(pmin.z, cub::Min());
    __syncthreads();

    pmax.x = BlockReduce(temp_storage).Reduce(pmax.x, cub::Max());
    __syncthreads();
    pmax.y = BlockReduce(temp_storage).Reduce(pmax.y, cub::Max()); 
    __syncthreads();
    pmax.z = BlockReduce(temp_storage).Reduce(pmax.z, cub::Max());
    __syncthreads();

    if (threadIdx.x == 0) {
      s_pmin = pmin;
      s_pmax = pmax;
    }
    __syncthreads();
    
    pmin = s_pmin;
    pmax = s_pmax;


    unsigned int thread_start[kValuesPerThread];
    unsigned int thread_end[kValuesPerThread];
    unsigned int thread_out[kValuesPerThread];
    unsigned int thread_keys[kValuesPerThread];
    unsigned int thread_values[kValuesPerThread];

    for (int i = 0; i < kValuesPerThread; i++) { 
      thread_out[i] = 0;
      thread_start[i] = 0;
      thread_end[i] = numAtoms;
    }

    double diff_x = pmax.x - pmin.x;
    double diff_y = pmax.y - pmin.y;
    double diff_z = pmax.z - pmin.z;
    int bit = 0;
    const int num_iters = 3;
    for (int i = 0; i < num_iters; i++) {
      if (diff_x > diff_y && diff_y > diff_z) {
        singleBisect(bit++, num_iters * 3, pmin.x, pmax.x, pos_x + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.y, pmax.y, pos_y + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.z, pmax.z, pos_z + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
      } else if (diff_x > diff_z && diff_z > diff_y) {
        singleBisect(bit++, num_iters * 3, pmin.x, pmax.x, pos_x + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.z, pmax.z, pos_z + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.y, pmax.y, pos_y + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
      } else if (diff_y > diff_x && diff_x > diff_z) {
        singleBisect(bit++, num_iters * 3, pmin.y, pmax.y, pos_y + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.x, pmax.x, pos_x + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.z, pmax.z, pos_z + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
      } else if (diff_y > diff_z && diff_z > diff_x) {
        singleBisect(bit++, num_iters * 3, pmin.y, pmax.y, pos_y + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.z, pmax.z, pos_z + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.x, pmax.x, pos_x + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
      } else if (diff_z > diff_x && diff_x > diff_y) {
        singleBisect(bit++, num_iters * 3, pmin.z, pmax.z, pos_z + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.x, pmax.x, pos_x + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.y, pmax.y, pos_y + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
      } else {
        singleBisect(bit++, num_iters * 3, pmin.z, pmax.z, pos_z + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.y, pmax.y, pos_y + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
        singleBisect(bit++, num_iters * 3, pmin.x, pmax.x, pos_x + offset, numAtoms, s_indexBuffer, 
                     thread_start, thread_end, thread_out, thread_keys, thread_values);
      }
    }

    for (int i = 0; i < kValuesPerThread; i++) {
      const int idx = kValuesPerThread * threadIdx.x + i;
      if (idx < numAtoms) {
        sortIndex[offset + idx] = thread_out[i];
        sortOrder[offset + idx] = idx;
      }
    }
    __syncthreads();

  }
}    


__device__ __forceinline__ int simple_grid(const int numGrids, const int x, const int y, const int z) {
  const int index = x + numGrids * (y + numGrids * z);
  return index;
}

__device__ __forceinline__ int snake_grid(const int numGrids, const int x, const int y, const int z) {
  int index = numGrids * numGrids * x;
  if (x % 2 == 0) {
    index += numGrids * y;
  } else {
    index += numGrids * (numGrids - 1 - y);
  }
  if (y % 2 == x % 2) {
    index += z;
  } else {
    index += (numGrids - 1 - z);
  }
  return index;
}


/**
 * \brief Computes the nonbonded index of atoms for optimal clustering
 *
 * \par 
 * Each thread-block is assigned to a patch, and does the following:
 * - First, the minimum and maximum coordinates of the patch are computed.
 *   the patchMin and patchMax don't produce the same results. I'm not sure
 *   if this is because of migration coords or something with the lattice...
 * - Seconds, the nonbonded index of each atom is computed and stored 
 *   The exact spatial hashing algorithm needs investigation
 *
 */
__global__ void computeNonbondedIndex(
  const int               numPatches,
  const int               numGrids,
  const CudaLocalRecord*  localRecords,
  const double3*          patchMin,
  const double3*          patchMax,
  const double*           pos_x,
  const double*           pos_y,
  const double*           pos_z,
  int*                    sortOrder,
  int*                    sortIndex
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  typedef cub::BlockReduce<double, MigrationCUDAKernel::kSortNumThreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ double3 s_pmin;
  __shared__ double3 s_pmax;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    #pragma unroll
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int offset = s_record.bufferOffset;

    // Compute the thread-local min/max values
    double3 pmin, pmax;
    pmin.x = BIG_FLOAT;
    pmin.y = BIG_FLOAT;
    pmin.z = BIG_FLOAT;
    pmax.x = SMALL_FLOAT;
    pmax.y = SMALL_FLOAT;
    pmax.z = SMALL_FLOAT;
    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
      pmin.x = fmin(pmin.x, pos_x[offset + i]);
      pmin.y = fmin(pmin.y, pos_y[offset + i]);
      pmin.z = fmin(pmin.z, pos_z[offset + i]);
      pmax.x = fmax(pmax.x, pos_x[offset + i]);
      pmax.y = fmax(pmax.y, pos_y[offset + i]);
      pmax.z = fmax(pmax.z, pos_z[offset + i]);
    }
    __syncthreads();
   
    // Compute the thread-block min/max values
    pmin.x = BlockReduce(temp_storage).Reduce(pmin.x, cub::Min());
    __syncthreads();
    pmin.y = BlockReduce(temp_storage).Reduce(pmin.y, cub::Min());
    __syncthreads();
    pmin.z = BlockReduce(temp_storage).Reduce(pmin.z, cub::Min());
    __syncthreads();

    pmax.x = BlockReduce(temp_storage).Reduce(pmax.x, cub::Max());
    __syncthreads();
    pmax.y = BlockReduce(temp_storage).Reduce(pmax.y, cub::Max()); 
    __syncthreads();
    pmax.z = BlockReduce(temp_storage).Reduce(pmax.z, cub::Max());
    __syncthreads();

    if (threadIdx.x == 0) {
      s_pmin = pmin;
      s_pmax = pmax;
    }
    __syncthreads();
    
    pmin = s_pmin;
    pmax = s_pmax;

    // Compute the sort index
    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
      const double x = pos_x[offset + i];
      const double y = pos_y[offset + i];
      const double z = pos_z[offset + i];

      // Compute subpatch index
      int idx_x = (int)((x - pmin.x) / (pmax.x - pmin.x) * ((double) numGrids));
      int idx_y = (int)((y - pmin.y) / (pmax.y - pmin.y) * ((double) numGrids));
      int idx_z = (int)((z - pmin.z) / (pmax.z - pmin.z) * ((double) numGrids));
      idx_x = min(max(idx_x, 0), numGrids-1);
      idx_y = min(max(idx_y, 0), numGrids-1);
      idx_z = min(max(idx_z, 0), numGrids-1);

      // Compute sort index
      const int block_index = snake_grid(numGrids, idx_x, idx_y, idx_z);
      const double z_norm = (z - pmin.z) / (pmax.z - pmin.z);

      const int reverse = ((idx_y % 2) == (idx_x % 2));
      int inner_index;
      if (reverse) {
        inner_index = (unsigned int) (z_norm * (1 << 16));
      } else {
        inner_index =  ~((unsigned int) (z_norm * (1 << 16)));
      }

      sortIndex[offset + i] = (block_index << 17) + inner_index;
      sortOrder[offset + i] = i;
    }
    __syncthreads();
  }
}

/**
 * \brief  Sorts the nonbonded sorting based on previously computed indices
 *
 * /par 
 * Uses CUB's block-level sort algorithm to generate the nonbonded ordering based on
 * previously computed spatial hashing. It will generate both a forward and backward maping
 *
 */
__global__ void sortAtomsKernel(
  const int               numPatches,
  const CudaLocalRecord*  localRecords,
  int*                    sortOrder,
  int*                    unsortOrder,
  int*                    sortIndex
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  typedef cub::BlockRadixSort<int, MigrationCUDAKernel::kSortNumThreads, MigrationCUDAKernel::kValuesPerThread, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int offset = s_record.bufferOffset;

    int thread_keys[MigrationCUDAKernel::kValuesPerThread];
    int thread_values[MigrationCUDAKernel::kValuesPerThread];
    for (int i = 0; i < MigrationCUDAKernel::kValuesPerThread; i++) {
      const int idx = MigrationCUDAKernel::kValuesPerThread * threadIdx.x + i;
      if (idx < numAtoms) {
          thread_keys[i] = sortIndex[offset + idx];
          thread_values[i] = sortOrder[offset + idx];
      } else {
          thread_keys[i] = MAX_VALUE;
          thread_values[i] =  -1;
      }
    }
    __syncthreads();

    BlockRadixSort(temp_storage).Sort(thread_keys, thread_values);
    __syncthreads();

    for (int i = 0; i < MigrationCUDAKernel::kValuesPerThread; i++) {
      const int idx = MigrationCUDAKernel::kValuesPerThread * threadIdx.x + i;
      if (thread_keys[i] != MAX_VALUE) {
        unsortOrder[offset + thread_values[i]] = idx;
        sortOrder[offset + idx] = thread_values[i];
        sortIndex[offset + idx] = thread_keys[i];
      }
    }
    __syncthreads();
  }
}


__global__ void printSortOrder(
  const int               numPatches,
  const CudaLocalRecord*  localRecords,
  int*                    sortOrder,
  const double*           pos_x,
  const double*           pos_y,
  const double*           pos_z
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

    if (patchIndex == 0) {
      if (threadIdx.x == 0) {
        for (int i = 0; i < numAtoms; i ++) {
          printf("%d %d %f %f %f\n", i, sortOrder[offset + i], pos_x[offset + i], pos_y[offset+i], pos_z[offset + i]);
        }
      }
    }
    __syncthreads();
  }
}

void MigrationCUDAKernel::sortAtoms( 
  const int               numPatches,
  const int               numAtoms,
  const CudaLocalRecord*  records,
  const double3*          patchMin,
  const double3*          patchMax,
  const double*           pos_x,
  const double*           pos_y,
  const double*           pos_z,
  int*                    sortOrder,
  int*                    unsortOrder,
  int*                    sortIndex,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = numPatches;
  
  // Knob for the spatial hashing
  const int numGrid = 4;

  computeBisect<<<numBlocks, numThreads, 0, stream>>>(
//  computeNonbondedIndex<<<numBlocks, numThreads, 0, stream>>>(
    numPatches,
    numGrid,
    records,
    patchMin,
    patchMax,
    pos_x,
    pos_y,
    pos_z,
    sortOrder,
    sortIndex
  );
  
  sortAtomsKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatches,
    records,
    sortOrder,
    unsortOrder,
    sortIndex
  );

#if 0
  printSortOrder<<<numBlocks, numThreads, 0, stream>>>(
    numPatches,
    records,
    sortOrder,
    pos_x,
    pos_y,
    pos_z
  );
#endif
}


/**
 * \brief Computes the derived quantities for SoA data
 *
 * \par 
 * Some data isn't stored in AoS data. This kernel computes quantities that are 
 * used elsewhere in calculation like reciprocal mass
 * 
 */
__global__ void calculateDerivedKernel(
  const int               numPatches,
  const CudaLocalRecord*  localRecords,
  float*                  mass,
  double*                 recipMass
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
      const float m = mass[offset + i];
      recipMass[offset + i] = (m > 0) ? 1.0f / m : 0.0;
    }
    __syncthreads();
  }
}

/**
 * \brief Computes the Langevin derived quantities for SoA data
 *
 * \par 
 * Some data isn't stored in AoS data. This kernel computes quantities that are 
 * used elsewhere in the Langevin calculation like langScalRandBBK2
 * 
 */
template<bool kDoAlch>
__global__ void calculateDerivedLangevinKernel(
  const int               numPatches,
  const CudaLocalRecord*  localRecords,
  const double            dt,
  const double            kbT,
  const double            tempFactor,
  double*                 recipMass,
  int*                    partition,
  float*                  langevinParam,
  float*                  langScalVelBBK2,
  float*                  langScalRandBBK2
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
      const double dt_gamma = dt * langevinParam[offset + i];
      const float invm = recipMass[offset + i];
      const double factor = (kDoAlch && partition[offset + i]) ? tempFactor : 1.0;
      langScalRandBBK2[offset + i] = (float) sqrt( 2.0 * dt_gamma * kbT * factor * invm);
      langScalVelBBK2[offset + i] = (float) (1.0 / (1.0 + 0.5 * dt_gamma));
    }
    __syncthreads();
  }
}



/**
 * \brief Copies AoS data into SoA buffers
 *
 * \par 
 * During migration, atomic data is stored as AoS. After migration, we must copy
 * the data to SoA buffers for the rest of the calculation. Since the AoS atomic data
 * is 128 bytes, we can do this with efficient memory access using shared memory as a 
 * buffer for the transpose
 * 
 * TODO Remove ugly if statement for writing buffers
 * 
 */
template<bool kDoAlch, bool kDoLangevin>
__global__ void copy_AoS_to_SoAKernel(
  const int               numPatches,
  const CudaLocalRecord*  localRecords,
  const FullAtom*     atomdata_AoS,
  double*                 vel_x,
  double*                 vel_y,
  double*                 vel_z,
  double*                 pos_x,
  double*                 pos_y,
  double*                 pos_z,
  float*                  mass,
  float*                  charge,
  int*                    id,
  int*                    vdwType,
  int*                    hydrogenGroupSize,
  int*                    migrationGroupSize,
  int*                    atomFixed,
  float*                  rigidBondLength,
  char3*                  transform,
  int*                    partition,
  float*                  langevinParam
) {
  constexpr int kAtomsPerBuffer = MigrationCUDAKernel::kAtomsPerBuffer;
  constexpr int kNumSoABuffers = MigrationCUDAKernel::kNumSoABuffers;

  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  // FullAtom contains "Vectors" which initialize to 99999. This isn't allow for shared memory
  // suppressing this warning as it doesn't rely on the dynamic initialization
#ifdef NAMD_CUDA
  #pragma diag_suppress static_var_with_dynamic_init
  __shared__ FullAtom s_atombuffer[kAtomsPerBuffer]; // For a 128 byte cache line
#else // NAMD_HIP
  // NVCC might allow the code above but initializations on shared memory
  // are not allowed on clang
  extern __shared__ FullAtom s_atombuffer[];
#endif

  const int warps_per_threadblock = MigrationCUDAKernel::kSortNumThreads / WARPSIZE;
  const int wid = threadIdx.x / WARPSIZE;
  const int tid = threadIdx.x % WARPSIZE;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int numAtomsPad = (s_record.numAtoms + kAtomsPerBuffer - 1) / kAtomsPerBuffer;
    const int offset = s_record.bufferOffset;

    for (int i = 0; i < numAtomsPad; i++) {
      const int atomOffset = i * kAtomsPerBuffer;
      // Load atoms. 1 warp per atom
      for (int atom_idx = wid; atom_idx < kAtomsPerBuffer; atom_idx += warps_per_threadblock) {
        if (atomOffset + atom_idx < numAtoms) {
          if (tid * 4 < sizeof(FullAtom)) { // Not all threads load in data
            int32_t *s_int = (int32_t*) &(s_atombuffer[atom_idx]);
            int32_t *g_int = (int32_t*) &(atomdata_AoS[offset + atomOffset + atom_idx]);
            s_int[tid] = g_int[tid];
          }
        }
      }
      __syncthreads();

      // Store atoms in SoA. 1 thread per atom
      for (int buffer = wid; buffer < kNumSoABuffers; buffer += warps_per_threadblock) {
        if (atomOffset + tid < numAtoms) {
          if (buffer ==  0) vel_x             [offset + atomOffset + tid] = ((FullAtom)(s_atombuffer[tid])).velocity.x;
          if (buffer ==  1) vel_y             [offset + atomOffset + tid] = s_atombuffer[tid].velocity.y;
          if (buffer ==  2) vel_z             [offset + atomOffset + tid] = s_atombuffer[tid].velocity.z;
          if (buffer ==  3) pos_x             [offset + atomOffset + tid] = s_atombuffer[tid].position.x;
          if (buffer ==  4) pos_y             [offset + atomOffset + tid] = s_atombuffer[tid].position.y;
          if (buffer ==  5) pos_z             [offset + atomOffset + tid] = s_atombuffer[tid].position.z;
          if (buffer ==  6) mass              [offset + atomOffset + tid] = s_atombuffer[tid].mass;
          if (buffer ==  7) charge            [offset + atomOffset + tid] = s_atombuffer[tid].charge;
          if (buffer ==  8) id                [offset + atomOffset + tid] = s_atombuffer[tid].id;
          if (buffer ==  9) vdwType           [offset + atomOffset + tid] = s_atombuffer[tid].vdwType;
          if (buffer == 10) hydrogenGroupSize [offset + atomOffset + tid] = s_atombuffer[tid].hydrogenGroupSize;
          if (buffer == 11) migrationGroupSize[offset + atomOffset + tid] = s_atombuffer[tid].migrationGroupSize;
          if (buffer == 12) atomFixed         [offset + atomOffset + tid] = s_atombuffer[tid].atomFixed;
          if (buffer == 13) rigidBondLength   [offset + atomOffset + tid] = s_atombuffer[tid].rigidBondLength;
          if (buffer == 14) transform         [offset + atomOffset + tid] = s_atombuffer[tid].transform;
          if (kDoAlch && 
              buffer == 15) partition         [offset + atomOffset + tid] = s_atombuffer[tid].partition;
          if (kDoLangevin && 
              buffer == 16) langevinParam     [offset + atomOffset + tid] = s_atombuffer[tid].langevinParam;
        }
      }
      __syncthreads();
    }
    __syncthreads();
  }
}


void MigrationCUDAKernel::copy_AoS_to_SoA(
  const int               numPatches,
  const bool              alchOn,
  const bool              langevinOn,
  const double            dt,
  const double            kbT,
  const double            tempFactor,
  const CudaLocalRecord*  records,
  const FullAtom*     atomdata_AoS,
  double*                 recipMass,
  double*                 vel_x,
  double*                 vel_y,
  double*                 vel_z,
  double*                 pos_x,
  double*                 pos_y,
  double*                 pos_z,
  float*                  mass,
  float*                  charge,
  int*                    id,
  int*                    vdwType,
  int*                    hydrogenGroupSize,
  int*                    migrationGroupSize,
  int*                    atomFixed,
  float*                  rigidBondLength,
  char3*                  transform,
  int*                    partition,
  float*                  langevinParam,
  float*                  langScalVelBBK2,
  float*                  langScalRandBBK2,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = numPatches;

#ifdef NAMD_CUDA
  constexpr size_t sizeatoms = 0;
#else
  constexpr size_t sizeatoms = MigrationCUDAKernel::kAtomsPerBuffer*sizeof(FullAtom);
#endif

  #define CALL(alchOn, langevinOn) do { \
    if (numBlocks) { \
      copy_AoS_to_SoAKernel<alchOn, langevinOn><<<numBlocks, numThreads, sizeatoms, stream>>>( \
        numPatches, \
        records, \
        atomdata_AoS, \
        vel_x, vel_y, vel_z, \
        pos_x, pos_y, pos_z, \
        mass, charge, \
        id, vdwType, \
        hydrogenGroupSize, \
        migrationGroupSize, \
        atomFixed, \
        rigidBondLength, \
        transform, \
        partition, \
        langevinParam \
      ); \
    } \
  } while (0);
  
  if (alchOn && langevinOn) {
    CALL(true, true);
  } else if (!alchOn && langevinOn) {
    CALL(false, true);
  } else if (alchOn && !langevinOn) {
    CALL(true, false);
  } else {
    CALL(false, false);
  }

  #undef CALL

  calculateDerivedKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatches,
    records,
    mass,
    recipMass
  );

  // This needs the recipMass
  if (langevinOn) {
    if (alchOn) {
      calculateDerivedLangevinKernel<true><<<numBlocks, numThreads, 0, stream>>>(
        numPatches,
        records,
        dt, kbT, tempFactor,
        recipMass,
        partition,
        langevinParam,
        langScalVelBBK2,
        langScalRandBBK2
      );
    } else {
      calculateDerivedLangevinKernel<false><<<numBlocks, numThreads, 0, stream>>>(
        numPatches,
        records,
        dt, kbT, tempFactor,
        recipMass,
        partition,
        langevinParam,
        langScalVelBBK2,
        langScalRandBBK2
      );
    }
  }
}


/**
 * \brief Computes the solvent index with the patch
 *
 * \par 
 * Within a patch, the atoms must be sorted such that solvent atoms are at the end of patch. This sorting
 * must be sorting.
 * 
 * We do this by using a scan to compute the index for the solute atoms. And then another scan to compute
 * the index for solvent atoms. 
 * 
 */
__global__ 
void computeSolventIndexKernel(
  const int               numPatches,
  const CudaLocalRecord*  localRecords,
  const FullAtom*     atomdata_AoS_in,
  int*                    sortIndex
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  typedef cub::BlockScan<int, MigrationCUDAKernel::kSortNumThreads> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int offsetIn = patchIndex * MigrationCUDAKernel::kMaxAtomsPerPatch;

    int thread_input[MigrationCUDAKernel::kValuesPerThread];
    int thread_soluteIndex[MigrationCUDAKernel::kValuesPerThread];
    int thread_solventIndex[MigrationCUDAKernel::kValuesPerThread];
    int numSolute;

    // Load the isWater value
    for (int i = 0; i < MigrationCUDAKernel::kValuesPerThread; i++) {
      const int idx = MigrationCUDAKernel::kValuesPerThread * threadIdx.x + i;
      if (idx < numAtoms) {
          thread_input[i] = 1 - atomdata_AoS_in[offsetIn + idx].isWater;
      } else {
          thread_input[i] = 0;
      }
    }
    __syncthreads();

    // Compute the prefix sum of solvent
    BlockScan(temp_storage).ExclusiveSum(thread_input, thread_soluteIndex, numSolute);
    __syncthreads();
    
    // Flip input data
    for (int i = 0; i < MigrationCUDAKernel::kValuesPerThread; i++) {
      const int idx = MigrationCUDAKernel::kValuesPerThread * threadIdx.x + i;
      if (idx < numAtoms) {
          thread_input[i] = 1 - thread_input[i];
      } else {
          thread_input[i] = 0;
      }
    }
    __syncthreads();

    // Compute the prefix sum of water
    BlockScan(temp_storage).ExclusiveSum(thread_input, thread_solventIndex);
    __syncthreads();

    // write index
    for (int i = 0; i < MigrationCUDAKernel::kValuesPerThread; i++) {
      const int idx = MigrationCUDAKernel::kValuesPerThread * threadIdx.x + i;
      if (idx < numAtoms) {
        if (thread_input[i]) {
          sortIndex[offsetIn + idx] = numSolute + thread_solventIndex[i];
        } else {
          sortIndex[offsetIn + idx] = thread_soluteIndex[i];
        }
      } 
    }
    __syncthreads();
  }
}

/**
 * \brief Sorts atoms into solute-solvent ordering based on previous compute indices
 *
 * \par 
 * Using the previously computed indices, this kernel moves the atomic data in AoS format
 * into the desired order. Note, this kernel copies the atomic data from the scratch buffer
 * into the true atomic AoS buffer. The actual migration data movement copies into the scratch
 * buffer and this kernel copies back.
 * 
 * Since the AoS data is ~128 bytes per atom. Each warp will move a single atom
 * 
 */
__global__ void sortSolventKernel(
  const int               numPatches,
  const CudaLocalRecord*  localRecords,
  const FullAtom*     atomdata_AoS_in,
  FullAtom*           atomdata_AoS_out,
  const int*              sortIndex
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  const int warps_per_threadblock = MigrationCUDAKernel::kSortNumThreads / WARPSIZE;
  const int wid = threadIdx.x / WARPSIZE;
  const int tid = threadIdx.x % WARPSIZE;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;

    // This was causing issues with CUDA11.3. Needed to explicitly make the offset 
    // 64-bit
#if 0
    const int64_t offsetIn = patchIndex * MigrationCUDAKernel::kMaxAtomsPerPatch;
    const int64_t offset = s_record.bufferOffset;
#else
    const int offsetIn = patchIndex * MigrationCUDAKernel::kMaxAtomsPerPatch;
    const int offset = s_record.bufferOffset;
#endif

    for (int i = wid; i < numAtoms; i+=warps_per_threadblock) {
      const int dst_idx = sortIndex[offsetIn + i];
      if (tid * 4 < sizeof(FullAtom)) { // Not all threads load in data
        int32_t *src_int = (int32_t*) &(atomdata_AoS_in[offsetIn + i]);
        int32_t *dst_int = (int32_t*) &(atomdata_AoS_out[offset + dst_idx]);
        dst_int[tid] = src_int[tid];
      }
#if defined(NAMD_HIP)
      NAMD_WARP_SYNC(WARP_FULL_MASK);
#else
      WARP_SYNC(WARP_FULL_MASK);
#endif
    }

    __syncthreads();
  }
}

void MigrationCUDAKernel::sortSolventAtoms( 
  const int               numPatches,
  const CudaLocalRecord*  records,
  const FullAtom*     atomdata_AoS_in,
  FullAtom*           atomdata_AoS_out,
  int*                    sortIndex,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = numPatches;

  computeSolventIndexKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatches, 
    records,
    atomdata_AoS_in,
    sortIndex
  );

  sortSolventKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatches,
    records,
    atomdata_AoS_in,
    atomdata_AoS_out,
    sortIndex
  );
}

/**
 * \brief Computes the migration group index
 *
 * \par 
 * Atom migration must occur at the level of migration group. Some molecules are
 * moved together. I.e. hydrogen of a water molecular are moved based on oxygen's position
 * 
 * This kernel computes the number of migration groups per patch as well as their starting
 * indices. This will be used later during migration. It does this with a scan operation
 * 
 */
__global__ void computeMigrationGroupIndexKernel(
  const int               numPatches,
  CudaLocalRecord*        localRecords,
  const int*              migrationGroupSize,
  int*                    migrationGroupIndex
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  typedef cub::BlockScan<int, MigrationCUDAKernel::kSortNumThreads> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int offset = s_record.bufferOffset;

    int thread_size[MigrationCUDAKernel::kValuesPerThread];
    int thread_index[MigrationCUDAKernel::kValuesPerThread];
    int numGroups;

    // Load the migration group size
    for (int i = 0; i < MigrationCUDAKernel::kValuesPerThread; i++) {
      const int idx = MigrationCUDAKernel::kValuesPerThread * threadIdx.x + i;
      if (idx < numAtoms) {
          thread_size[i] = migrationGroupSize[offset + idx] ? 1 : 0;
      } else {
          thread_size[i] = 0;
      }
    }
    __syncthreads();

    // Compute the prefix sum of solvent
    BlockScan(temp_storage).ExclusiveSum(thread_size, thread_index, numGroups);
    __syncthreads();

    for (int i = 0; i < MigrationCUDAKernel::kValuesPerThread; i++) {
      const int idx = MigrationCUDAKernel::kValuesPerThread * threadIdx.x + i;
      if (thread_size[i] != 0) {
        migrationGroupIndex[offset + thread_index[i]] = idx;
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      localRecords[patchIndex].numMigrationGroups = numGroups;
    }
    __syncthreads();
  }
}


void MigrationCUDAKernel::computeMigrationGroupIndex( 
  const int               numPatches,
  CudaLocalRecord*        records,
  const int*              migrationGroupSize,
  int*                    migrationGroupIndex,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = numPatches;

  computeMigrationGroupIndexKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatches, 
    records, 
    migrationGroupSize, 
    migrationGroupIndex
  );
}

/**
 * \brief Computes the transformed positions of migrated atoms
 *
 * \par 
 * When atoms move between patches, we need to do a transform on the position based
 * on the lattice. This logic is largely copied from HomePatch.C. This is only done
 * on the atoms that have actually moved between patches. This uses the numAtomsLocal
 * field of the CudaLocalRecord
 *
 * Note this kernel could likely be optimized. Threads that are not assigned to the 
 * parent of a migration group do no work. But not that many atoms actually migrate
 * 
 */
__global__ void transformMigratedPositionsKernel(
  const int               numPatches,
  const CudaLocalRecord*  localRecords,
  const double3*          patchCenter,
  FullAtom*           atomdata_AoS,
  const Lattice       lattice
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

    const int offset = patchIndex * MigrationCUDAKernel::kMaxAtomsPerPatch;
    const int numAtoms = s_record.numAtoms;
    const int numAtomsLocal = s_record.numAtomsLocal;
    const int numAtomsMigrated = numAtoms - numAtomsLocal;
    const double3 center = patchCenter[patchIndex];

    for (int i = threadIdx.x; i < numAtomsMigrated; i += blockDim.x) {
      const int startingIndex = numAtomsLocal + i;
      const int migrationSize = atomdata_AoS[offset + startingIndex].migrationGroupSize;
      const int hydrogenSize  = atomdata_AoS[offset + startingIndex].hydrogenGroupSize;

      if (migrationSize == 0) continue;

      Transform parent_transform;
      if (migrationSize != hydrogenSize) {
        double3 c_pos = make_double3(0.0, 0.0, 0.0);
        int c = 0;
        int tmp_hydrogenGroupSize = hydrogenSize;
        for (int j = 0; j < migrationSize; j+=tmp_hydrogenGroupSize) {
          c_pos = c_pos + (double3) atomdata_AoS[offset + startingIndex + j].position;
          c++;
          tmp_hydrogenGroupSize = atomdata_AoS[offset + startingIndex + j].hydrogenGroupSize;
        }
        c_pos.x = c_pos.x / ((double) c);
        c_pos.y = c_pos.y / ((double) c);
        c_pos.z = c_pos.z / ((double) c);

        parent_transform = atomdata_AoS[offset+startingIndex].transform;
        c_pos = lattice.nearest(c_pos, center, &parent_transform);

        double3 pos = atomdata_AoS[offset+startingIndex].position;
        Transform transform = atomdata_AoS[offset+startingIndex].transform;

        pos = lattice.reverse_transform(pos, transform);
        pos = lattice.apply_transform(pos, parent_transform);

        atomdata_AoS[offset+startingIndex].transform = parent_transform;  
        atomdata_AoS[offset+startingIndex].position = pos;  
      } else {

        double3 pos = atomdata_AoS[offset+startingIndex].position;
        Transform transform = atomdata_AoS[offset+startingIndex].transform;

        pos = lattice.nearest(pos, center, &transform);
        parent_transform = transform;

        atomdata_AoS[offset+startingIndex].transform = transform; 
        atomdata_AoS[offset+startingIndex].position = pos; 
      }
      for (int j = 1; j < migrationSize; j++) {
        double3 pos = atomdata_AoS[offset+startingIndex + j].position;
        Transform transform = atomdata_AoS[offset+startingIndex +j].transform;

        pos = lattice.reverse_transform(pos, transform);
        pos = lattice.apply_transform(pos, parent_transform);

        atomdata_AoS[offset+startingIndex+j].transform = parent_transform;
        atomdata_AoS[offset+startingIndex+j].position = pos; 
      }
    }
    __syncthreads();
  }
}

void MigrationCUDAKernel::transformMigratedPositions( 
  const int               numPatches,
  const CudaLocalRecord*  localRecords,
  const double3*          patchCenter,
  FullAtom*           atomdata_AoS,
  const Lattice       lattice,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = numPatches;

  transformMigratedPositionsKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatches, 
    localRecords, 
    patchCenter,
    atomdata_AoS,
    lattice
  );
}


/**
 * \brief Computes the new location of a migration group
 *
 * \par 
 * Computes the migration destination for each migration group. This info includes:
 * - Destination patch ID
 * - Destination patch local index (this is the home patches index)
 * - Destination device
 * 
 * While the information is computed at the migration group level, it is stored for each atom
 * in a migration group, so some of the later data movement can happen at the per atom level
 *
 * Note the migrationDestination structure will eventually include the atoms new index within a 
 * patch, but that is determined later
 * 
 */
__global__ void
computeMigrationDestinationKernel(
  const int               numPatches,
  CudaLocalRecord*        localRecords,
  const Lattice       lattice,
  const CudaMInfo*        mInfo,
  const int*              patchToDeviceMap,
  const int*              globalToLocalMap,
  const double3*          patchMin,
  const double3*          patchMax,
  const int*              hydrogenGroupSize,
  const int*              migrationGroupSize,
  const int*              migrationGroupIndex,
  const double*           pos_x,
  const double*           pos_y,
  const double*           pos_z,
  int4*                   migrationDestination
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
    
    // Clear migration atom count
    if (threadIdx.x == 0) {
      localRecords[patchIndex].numAtomsNew = 0;
    }
    __syncthreads();

    const int numMigrationGroups = s_record.numMigrationGroups;
    const int offset = s_record.bufferOffset;
    const double3 min = patchMin[patchIndex];
    const double3 max = patchMax[patchIndex];
    CudaMInfo migration_info = mInfo[patchIndex];

    int xdev, ydev, zdev;

    for (int i = threadIdx.x; i < numMigrationGroups; i += blockDim.x) {
      
      const int startingIndex = migrationGroupIndex[offset + i];
      const int migrationSize = migrationGroupSize[offset + startingIndex];
      const int hydrogenSize  = hydrogenGroupSize[offset + startingIndex];

      double3 pos;
      pos.x = pos_x[offset + startingIndex];
      pos.y = pos_y[offset + startingIndex];
      pos.z = pos_z[offset + startingIndex];

      if (migrationSize != hydrogenSize) {
        int c = 1;
        int tmp_hydrogenGroupSize = 1;
        for (int j = hydrogenSize; j < migrationSize; j+=tmp_hydrogenGroupSize) {
          pos.x = pos.x + pos_x[offset + startingIndex + j];
          pos.y = pos.y + pos_y[offset + startingIndex + j];
          pos.z = pos.z + pos_z[offset + startingIndex + j];
          c++;
          tmp_hydrogenGroupSize = hydrogenGroupSize[offset + startingIndex + j];
        }
        pos.x = pos.x / ((double) c);
        pos.y = pos.y / ((double) c);
        pos.z = pos.z / ((double) c);
      }

      double3 s = lattice.scale(pos);

      if (s.x < min.x) xdev = 0;
      else if (max.x <= s.x) xdev = 2;
      else xdev = 1;

      if (s.y < min.y) ydev = 0;
      else if (max.y <= s.y) ydev = 2;
      else ydev = 1;

      if (s.z < min.z) zdev = 0;
      else if (max.z <= s.z) zdev = 2;
      else zdev = 1;

      int dest_patch = migration_info.destPatchID[xdev][ydev][zdev];
      int dest_device = patchToDeviceMap[dest_patch];
      int dest_localID = globalToLocalMap[dest_patch];
      int4 dest_info;
      dest_info.x = dest_patch;
      dest_info.y = dest_device;
      dest_info.z = dest_localID;
      dest_info.w = 0;

      for (int j = 0; j < migrationSize; j++) {
        migrationDestination[offset + startingIndex + j] = dest_info;
      }
    }
    __syncthreads();
  }
}

void MigrationCUDAKernel::computeMigrationDestination(
  const int               numPatches,
  CudaLocalRecord*        localRecords,
  const Lattice       lattice,
  const CudaMInfo*        mInfo,
  const int*              patchToDeviceMap,
  const int*              globalToLocalMap,
  const double3*          patchMin,
  const double3*          patchMax,
  const int*              hydrogenGroupSize,
  const int*              migrationGroupSize,
  const int*              migrationGroupIndex,
  const double*           pos_x,
  const double*           pos_y,
  const double*           pos_z,
  int4*                   migrationDestination,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = numPatches;

  computeMigrationDestinationKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatches, 
    localRecords, 
    lattice,
    mInfo,
    patchToDeviceMap,
    globalToLocalMap,
    patchMin,
    patchMax,
    hydrogenGroupSize,
    migrationGroupSize, 
    migrationGroupIndex,
    pos_x, pos_y, pos_z,
    migrationDestination
  );
}


/**
 * \brief Updates AoS data structure before migration 
 *
 * \par 
 * Copies the necessary data from the SoA buffers to the AoS buffers
 * This is just the positions and velocities
 *
 * TODO optimize this kernel
 * 
 */
__global__ void update_AoSKernel(
  const int               numPatches,
  const CudaLocalRecord*  localRecords,
  FullAtom*           atomdata_AoS,
  const double*           vel_x,
  const double*           vel_y,
  const double*           vel_z,
  const double*           pos_x,
  const double*           pos_y,
  const double*           pos_z
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
      double3 pos;
      pos.x = pos_x[offset + i];
      pos.y = pos_y[offset + i];
      pos.z = pos_z[offset + i];
      
      double3 vel;
      vel.x = vel_x[offset + i];
      vel.y = vel_y[offset + i];
      vel.z = vel_z[offset + i];

      atomdata_AoS[offset + i].position = pos;
      atomdata_AoS[offset + i].velocity = vel;
    }
    __syncthreads();
  }
}

void MigrationCUDAKernel::update_AoS(
  const int               numPatches,
  const CudaLocalRecord*  records,
  FullAtom*           atomdata_AoS,
  const double*           vel_x,
  const double*           vel_y,
  const double*           vel_z,
  const double*           pos_x,
  const double*           pos_y,
  const double*           pos_z,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = numPatches;

  update_AoSKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatches, 
    records, 
    atomdata_AoS,
    vel_x, vel_y, vel_z,
    pos_x, pos_y, pos_z
  );
}


/**
 * \brief Copies atoms that aren't "moving" into the scratch buffers
 *
 * \par 
 * This kernel will copy the atoms that are not moving into a new patch into that
 * patches scratch buffers. This uses a scan operation to eliminate the need for 
 * atomic operations to compute atom's new location.
 * 
 * This scan will not change the order of the atoms within a migration group
 *
 * Note this will set the localID of the atom in the migrationDestination 
 * Note because AoS atomic data is ~128 bytes, each warp moves 1 atom
 *
 */
__global__ void performLocalMigrationKernel(
  const int               numPatches,
  CudaLocalRecord*        localRecords,
  const FullAtom*     atomdata_AoS_in,
  FullAtom*           atomdata_AoS_out,
  int4*                   migrationDestination
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  typedef cub::BlockScan<int, MigrationCUDAKernel::kSortNumThreads> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  __shared__ int s_local_index[MigrationCUDAKernel::kSortNumThreads * MigrationCUDAKernel::kValuesPerThread];

  const int warps_per_threadblock = blockDim.x / WARPSIZE;
  const int wid = threadIdx.x / WARPSIZE;
  const int tid = threadIdx.x % WARPSIZE;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int patchID = s_record.patchID;
    const int numAtoms = s_record.numAtoms;
    const int offset = s_record.bufferOffset;
    const int dst_offset = patchIndex * MigrationCUDAKernel::kMaxAtomsPerPatch;

    __syncthreads();
    // Load in if the atom is staying in the same patch
    int thread_input[MigrationCUDAKernel::kValuesPerThread];
    int thread_output[MigrationCUDAKernel::kValuesPerThread];
    int numLocal;
    for (int i = 0; i < MigrationCUDAKernel::kValuesPerThread; i++) {
      const int idx = MigrationCUDAKernel::kValuesPerThread * threadIdx.x + i;
      if (idx < numAtoms && migrationDestination[offset + idx].x == patchID) {
          thread_input[i] = 1;
      } else {
          thread_input[i] = 0;
      }
    }
    __syncthreads();

    // Compute the prefix sum of local
    BlockScan(temp_storage).ExclusiveSum(thread_input, thread_output, numLocal);
    __syncthreads();

    for (int i = 0; i < MigrationCUDAKernel::kValuesPerThread; i++) {
      const int idx = MigrationCUDAKernel::kValuesPerThread * threadIdx.x + i;
      if (idx < numAtoms && thread_input[i]) {
        s_local_index[thread_output[i]] = idx;
      }
    }
    if (threadIdx.x == 0) {
      localRecords[patchIndex].numAtomsLocal = numLocal;
      localRecords[patchIndex].numAtomsNew = numLocal;
    }
    __syncthreads();

    // We can do this at the atom level instead of the migrationGroup level
    // because the migrationDestinations take that into account and
    // the prefix sum means we do not change the ordering of local atoms
    for (int i = wid; i < numLocal; i += warps_per_threadblock) {
      const int src_atom_idx = s_local_index[i];
      if (tid * 4 < sizeof(FullAtom)) { // Not all threads load in data
        int32_t *src_int = (int32_t*) &(atomdata_AoS_in  [offset     + src_atom_idx]);
        int32_t *dst_int = (int32_t*) &(atomdata_AoS_out [dst_offset + i]);
        dst_int[tid] = src_int[tid];
      }
      if (tid == 0) {
        migrationDestination[offset + src_atom_idx].w = i;
      }
#if defined(NAMD_HIP)
      NAMD_WARP_SYNC(WARP_FULL_MASK);
#else
      WARP_SYNC(WARP_FULL_MASK);
#endif
    }
    __syncthreads();
  }
}

void MigrationCUDAKernel::performLocalMigration(
  const int               numPatches,
  CudaLocalRecord*        records,
  const FullAtom*     atomdata_AoS_in,
  FullAtom*           atomdata_AoS_out,
  int4*                   migrationDestination,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = numPatches;

  performLocalMigrationKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatches, 
    records, 
    atomdata_AoS_in,
    atomdata_AoS_out,
    migrationDestination
  );
}


/**
 * \brief Moves the migrating atoms into their new patches
 *
 * \par 
 * Copies the atoms which have moved into new patches
 * into those patches scratch buffers. This will use atomic operations to
 * determine the new index of each migration group. This happens at the migration
 * group level so atoms within a migration group stay in the same order.
 * 
 * Note because not many atoms move, the use of atomics isn't too expensive
 * Note because AoS atomic data is ~128 bytes, each warp moves 1 atom
 * 
 */
__global__ void performMigrationKernel(
  const int               numPatches,
  CudaLocalRecord*        localRecords,
  CudaLocalRecord**       peer_records,
  const FullAtom*     local_atomdata_AoS,
  FullAtom**          peer_atomdata_AoS,
  const int*              migrationGroupSize,
  const int*              migrationGroupIndex,
  int4*                   migrationDestination
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  const int warps_per_threadblock = blockDim.x / WARPSIZE;
  const int wid = threadIdx.x / WARPSIZE;
  const int tid = threadIdx.x % WARPSIZE;

  for (int patchIndex = blockIdx.x; patchIndex < numPatches; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numMigrationGroups = s_record.numMigrationGroups;
    const int offset = s_record.bufferOffset;
    const int patchID = s_record.patchID;

    __syncthreads();

    for (int i = wid; i < numMigrationGroups; i += warps_per_threadblock) {
      const int startingIndex = migrationGroupIndex[offset + i];

      const int size = migrationGroupSize[offset + startingIndex];

      const int4 migrationInfo = migrationDestination[offset + startingIndex];
      const int dst_patchID = migrationInfo.x;
      const int dst_device = migrationInfo.y;
      const int dst_localID = migrationInfo.z;
      const int dst_offset = dst_localID * MigrationCUDAKernel::kMaxAtomsPerPatch;

      if (dst_patchID == patchID) continue;

      // Get index via atomic operation
      int dst_atom_idx;
      if (tid == 0) {
        int* counter_index = &(peer_records[dst_device][dst_localID].numAtomsNew);
#if __CUDA_ARCH__ >= 600
        dst_atom_idx = atomicAdd_system(counter_index, size);
#else
        // support single-GPU Maxwell
        dst_atom_idx = atomicAdd(counter_index, size);
#endif
      }
      dst_atom_idx = WARP_SHUFFLE(WARP_FULL_MASK, dst_atom_idx, 0, WARPSIZE);

      
      FullAtom* remote_atomdata_AoS = peer_atomdata_AoS[dst_device];
      for (int j = 0; j < size; j++) {
        if (tid * 4 < sizeof(FullAtom)) { // Not all threads load in data
          int32_t *src_int = (int32_t*) &(local_atomdata_AoS  [offset     + startingIndex + j]);
          int32_t *dst_int = (int32_t*) &(remote_atomdata_AoS [dst_offset + dst_atom_idx + j]);
          dst_int[tid] = src_int[tid];
        }
        if (tid == 0) {
          migrationDestination[offset + startingIndex + j].w = dst_atom_idx + j;
        }
#if defined(NAMD_HIP)
        NAMD_WARP_SYNC(WARP_FULL_MASK);
#else
        WARP_SYNC(WARP_FULL_MASK);
#endif
      }
    }

    __syncthreads();
  }
}

void MigrationCUDAKernel::performMigration(
  const int               numPatches,
  CudaLocalRecord*        records,
  CudaLocalRecord**       peer_records,
  const FullAtom*     local_atomdata_AoS,
  FullAtom**          peer_atomdata_AoS,
  const int*              migrationGroupSize,
  const int*              migrationGroupIndex,
  int4*                   migrationDestination,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = numPatches;

  performMigrationKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatches, 
    records, 
    peer_records,
    local_atomdata_AoS,
    peer_atomdata_AoS,
    migrationGroupSize,
    migrationGroupIndex,
    migrationDestination
  );
}


/**
 * \brief Updates migrationDestination array based on solute-solvent order
 *
 * \par 
 * The migrationDestation structure is used later by the tuple migration, so
 * we need to keep it accurate after the solute-solvent sorting. This uses peer
 * access to update each atom's new index within its new patch. 
 *
 */
__global__ void updateMigrationDestinationKernel(
  const int               numAtomsHome,
  int4*                   migrationDestination,
  int**                   d_peer_sortSoluteIndex
) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < numAtomsHome; i += blockDim.x * gridDim.x) {
    const int4 dest = migrationDestination[i];
    const int dst_device = dest.y;
    const int dst_patchID = dest.z;
    const int dst_patchOffset = MigrationCUDAKernel::kMaxAtomsPerPatch * dst_patchID;
    const int inputIndex = dest.w;

    const int outputIndex = d_peer_sortSoluteIndex[dst_device][dst_patchOffset + inputIndex];
    migrationDestination[i].w = outputIndex;
  }
}

void MigrationCUDAKernel::updateMigrationDestination(
  const int               numAtomsHome,
  int4*                   migrationDestination,
  int**                   d_peer_sortSoluteIndex,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = (numAtomsHome + kSortNumThreads - 1) / kSortNumThreads;

  updateMigrationDestinationKernel<<<numBlocks, numThreads, 0, stream>>>(
    numAtomsHome,
    migrationDestination,
    d_peer_sortSoluteIndex
  );
}


/**
 * \brief Copies static atomic data to proxy patches
 *
 * \par 
 * Copies atomic data that does not change to proxy patches.
 *
 */
template <bool doAlch>
__global__ void copyDataToProxiesKernel(
  const int               deviceIndex,
  const int               numPatchesHome,
  const int               numPatchesHomeAndProxy,
  const CudaLocalRecord*  localRecords,
  int**                   peer_id,
  int**                   peer_vdwType,
  int**                   peer_sortOrder,
  int**                   peer_unsortOrder,
  float**                 peer_charge,
  int**                   peer_partition,
  double3**               peer_patchCenter
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  for (int patchIndex = blockIdx.x + numPatchesHome; patchIndex < numPatchesHomeAndProxy; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtoms;
    const int dstOffset = s_record.bufferOffset;
    const int srcDevice = s_record.inline_peers[0].deviceIndex;
    const int srcOffset = s_record.inline_peers[0].bufferOffset;
    const int srcPatchIndex = s_record.inline_peers[0].patchIndex;

    // TODO this data is probably on the host somewhere...
    // And it probably doesn't change????
    if (threadIdx.x == 0) {
      peer_patchCenter[deviceIndex][patchIndex] = peer_patchCenter[srcDevice][srcPatchIndex];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
      peer_id         [deviceIndex][dstOffset + i] = peer_id[srcDevice][srcOffset + i];
      peer_vdwType    [deviceIndex][dstOffset + i] = peer_vdwType[srcDevice][srcOffset + i];
      peer_sortOrder  [deviceIndex][dstOffset + i] = peer_sortOrder[srcDevice][srcOffset + i];
      peer_unsortOrder[deviceIndex][dstOffset + i] = peer_unsortOrder[srcDevice][srcOffset + i];
      peer_charge     [deviceIndex][dstOffset + i] = peer_charge[srcDevice][srcOffset + i];
      if (doAlch) {
        peer_partition[deviceIndex][dstOffset + i] = peer_partition[srcDevice][srcOffset + i];
      }
    }
    __syncthreads();
  }
}

void MigrationCUDAKernel::copyDataToProxies(
  const int               deviceIndex,
  const int               numPatchesHome,
  const int               numPatchesHomeAndProxy,
  const CudaLocalRecord*  records,
  int**                   peer_id,
  int**                   peer_vdwType,
  int**                   peer_sortOrder,
  int**                   peer_unsortOrder,
  float**                 peer_charge,
  int**                   peer_partition,
  double3**               peer_patchCenter,
  bool                    doAlch,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numPatches = numPatchesHomeAndProxy - numPatchesHome;
  const int numBlocks = numPatches;

  if (doAlch) {
    copyDataToProxiesKernel<true><<<numBlocks, numThreads, 0, stream>>>(
      deviceIndex,
      numPatchesHome,  numPatchesHomeAndProxy,
      records,
      peer_id, peer_vdwType, peer_sortOrder,
      peer_unsortOrder, peer_charge, peer_partition, peer_patchCenter
    );
  }
  else {
    copyDataToProxiesKernel<false><<<numBlocks, numThreads, 0, stream>>>(
      deviceIndex,
      numPatchesHome,  numPatchesHomeAndProxy,
      records,
      peer_id, peer_vdwType, peer_sortOrder,
      peer_unsortOrder, peer_charge, peer_partition, peer_patchCenter
    );
  }
}

/**
 * \brief Copies migrationDestination to proxy patches
 *
 * \par 
 * This copies the migrationDestination to proxies to be used in tuple migration
 * This needs to use the old bufferOffsets and atom count so it is separate from the 
 * other copyDataToProxies kernel. Additionally, we do this as a put operation to 
 * avoid a node barrier. The home patch produces this data and then it needs to write it
 * to the proxy patch; it the device which owns the proxy patch was doing a get operation,
 * we'd need to have a node barrier to make sure this data was ready to be ready. By using 
 * put operation, we can do a device local synchonization (i.e. a new kernel launch) to 
 * make sure the data is ready
 *
 */
__global__ void copyMigrationDestinationToProxiesKernel(
  const int               deviceIndex,
  const int               numPatchesHome,
  const int               numPatchesHomeAndProxy,
  const CudaLocalRecord*  localRecords,
  const CudaPeerRecord* peerRecords,
  int4**                  peer_migrationDestination
) {
  __shared__ CudaLocalRecord s_record;
  using AccessType = int32_t;
  AccessType* s_record_buffer = (AccessType*)  &s_record;

  for (int patchIndex = blockIdx.x; patchIndex < numPatchesHome; patchIndex += gridDim.x) {
    // Read in the CudaLocalRecord using multiple threads. This should 
    #pragma unroll 1
    for (int i = threadIdx.x; i < sizeof(CudaLocalRecord)/sizeof(AccessType); i += blockDim.x) {
      s_record_buffer[i] = ((AccessType*) &(localRecords[patchIndex]))[i];
    }
    __syncthreads();

    const int numAtoms = s_record.numAtomsNew;
    const int dstOffset = s_record.bufferOffsetOld;
    const int numPeerRecords = s_record.numPeerRecord;
    const int peerRecordStartIndexs = s_record.peerRecordStartIndex;

    const int inlineRemotes = min(numPeerRecords, CudaLocalRecord::num_inline_peer);

    for (int remote = 0; remote < inlineRemotes; remote++) {
      const int srcDevice = s_record.inline_peers[remote].deviceIndex;
      const int srcOffset = s_record.inline_peers[remote].bufferOffset;
      for (int i = threadIdx.x; i < numAtoms; i += blockDim.x){
        peer_migrationDestination[srcDevice][srcOffset + i] = peer_migrationDestination[deviceIndex][dstOffset + i];
      }
    }

    for (int remote = inlineRemotes; remote < numPeerRecords; remote++) {
      const int srcDevice = peerRecords[peerRecordStartIndexs + remote].deviceIndex;
      const int srcOffset = peerRecords[peerRecordStartIndexs + remote].bufferOffset;
      for (int i = threadIdx.x; i < numAtoms; i += blockDim.x){
        peer_migrationDestination[srcDevice][srcOffset + i] = peer_migrationDestination[deviceIndex][dstOffset + i];
      }
    }
    __syncthreads();
  }
}

void MigrationCUDAKernel::copyMigrationDestinationToProxies(
  const int               deviceIndex,
  const int               numPatchesHome,
  const int               numPatchesHomeAndProxy,
  const CudaLocalRecord*  records,
  const CudaPeerRecord* peerRecords,
  int4**                  peer_migrationDestination,
  cudaStream_t            stream
) {
  constexpr int numThreads = kSortNumThreads;
  const int numBlocks = numPatchesHome;

  copyMigrationDestinationToProxiesKernel<<<numBlocks, numThreads, 0, stream>>>(
    deviceIndex,
    numPatchesHome, numPatchesHomeAndProxy,
    records,
    peerRecords,
    peer_migrationDestination
  );
}

/**
 * \brief Updates the CudaLocalRecord data structure
 *
 * \par 
 * This updates the number of atoms and buffer offsets of the patches. This should only
 * be ran on the home patches since it uses peer access to update remote structures
 *
 */
__global__ void updateLocalRecordsKernel(
  const int               numPatchesHome,
  CudaLocalRecord*        localRecords,
  CudaLocalRecord**       peer_records,
  const CudaPeerRecord* peerRecords
) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < numPatchesHome; i += blockDim.x * gridDim.x) {
    const int numAtomsOld = localRecords[i].numAtoms;
    const int numAtoms = localRecords[i].numAtomsNew;
    const int numAtomsNBPad = CudaComputeNonbondedKernel::computeAtomPad(numAtoms);
    localRecords[i].numAtoms = numAtoms;
    localRecords[i].numAtomsNBPad = numAtomsNBPad;
    localRecords[i].bufferOffsetOld = localRecords[i].bufferOffset;
    localRecords[i].numAtomsNew = numAtomsOld;

    const int numPeerRecord = localRecords[i].numPeerRecord;
    const int peerRecordStartIndex = localRecords[i].peerRecordStartIndex;
    // TODO use inline remotes??

    for (int remote = 0; remote < numPeerRecord; remote++) {
      const int srcDevice = peerRecords[peerRecordStartIndex + remote].deviceIndex;
      const int srcOffset = peerRecords[peerRecordStartIndex + remote].patchIndex;
      peer_records[srcDevice][srcOffset].numAtoms = numAtoms;
      peer_records[srcDevice][srcOffset].numAtomsNBPad = numAtomsNBPad;
    }
  }
}

void MigrationCUDAKernel::updateLocalRecords(
  const int               numPatchesHome,
  CudaLocalRecord*        records,
  CudaLocalRecord**       peer_records,
  const CudaPeerRecord* peerRecords,
  cudaStream_t            stream
) {
  const int numThreads = kSortNumThreads;
  const int numBlocks = (numPatchesHome + kSortNumThreads - 1) / kSortNumThreads;

  updateLocalRecordsKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatchesHome,
    records,
    peer_records,
    peerRecords
  );

}


struct RecordToNumAtoms {
  __host__ __device__ __forceinline__
  int operator()(const CudaLocalRecord &a) const {
    return a.numAtoms;
  }
};

struct RecordTonumAtomsNBPad {
  __host__ __device__ __forceinline__
  int operator()(const CudaLocalRecord &a) const {
    return a.numAtomsNBPad;
  }
};

/**
 * \brief Updates the buffer offsets based on scratch patchOffsets
 *
 */
__global__ void updateLocalRecordsOffsetKernel(
  const int               numPatchesHomeAndProxy,
  CudaLocalRecord*        localRecords,
  int*                    patchOffsets,
  int*                    patchOffsetsNB
) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < numPatchesHomeAndProxy; i += blockDim.x * gridDim.x) {
    localRecords[i].bufferOffset = patchOffsets[i];
    localRecords[i].bufferOffsetNBPad = patchOffsetsNB[i];
  }
}

void MigrationCUDAKernel::updateLocalRecordsOffset(
  const int               numPatchesHomeAndProxy,
  CudaLocalRecord*        records,
  cudaStream_t            stream
) {
  const int numThreads = kSortNumThreads;
  const int numBlocks = (numPatchesHomeAndProxy + kSortNumThreads - 1) / kSortNumThreads;
  size_t temp = patchDeviceScan_alloc;
  
  //
  // Integration Offsets
  // 
  RecordToNumAtoms conversion_op;
  using InputIter = cub::TransformInputIterator<int, RecordToNumAtoms, CudaLocalRecord*>;
  InputIter iter(records, conversion_op);

  cub::DeviceScan::ExclusiveSum<InputIter>(
    d_patchDeviceScan_scratch, temp,
    iter, d_patchOffset_temp, numPatchesHomeAndProxy, stream
  );

  //
  // Nonbonded Offsets
  // 
  RecordTonumAtomsNBPad conversion_op_nb;
  using InputIterNB = cub::TransformInputIterator<int, RecordTonumAtomsNBPad, CudaLocalRecord*>;
  InputIterNB iterNB(records, conversion_op_nb);

  cub::DeviceScan::ExclusiveSum<InputIterNB>(
    d_patchDeviceScan_scratch, temp,
    iterNB, d_patchOffsetNB_temp, numPatchesHomeAndProxy, stream
  );

  //
  // Update AoS data
  //
  updateLocalRecordsOffsetKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatchesHomeAndProxy, 
    records,
    d_patchOffset_temp,
    d_patchOffsetNB_temp
  );
}

/**
 * \brief Updates the remote records on this device
 *
 */
__global__ void updatePeerRecordsKernel(
  const int               numPatchesHomeAndProxy,
  CudaLocalRecord*        localRecords,
  CudaLocalRecord**       peer_records,
  CudaPeerRecord*       peerRecords
) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < numPatchesHomeAndProxy; i += blockDim.x * gridDim.x) {
    const int numPeerRecord = localRecords[i].numPeerRecord;
    const int peerRecordStartIndex = localRecords[i].peerRecordStartIndex;

    for (int remote = 0; remote < numPeerRecord; remote++) {
      const int srcDevice = peerRecords[peerRecordStartIndex + remote].deviceIndex;
      const int srcOffset = peerRecords[peerRecordStartIndex + remote].patchIndex;

      const int bufferOffset = peer_records[srcDevice][srcOffset].bufferOffset;
      const int bufferOffsetNBPad = peer_records[srcDevice][srcOffset].bufferOffsetNBPad;

      peerRecords[peerRecordStartIndex + remote].bufferOffset = bufferOffset;
      peerRecords[peerRecordStartIndex + remote].bufferOffsetNBPad = bufferOffsetNBPad;

      if (remote < CudaLocalRecord::num_inline_peer) {
        localRecords[i].inline_peers[remote].bufferOffset = bufferOffset;
        localRecords[i].inline_peers[remote].bufferOffsetNBPad = bufferOffsetNBPad;
      }
    }
  }
}

void MigrationCUDAKernel::updatePeerRecords(
  const int               numPatchesHomeAndProxy,
  CudaLocalRecord*        records,
  CudaLocalRecord**       peer_records,
  CudaPeerRecord*       peerRecords,
  cudaStream_t            stream
) {
  const int numThreads = kSortNumThreads;
  const int numBlocks = (numPatchesHomeAndProxy + kSortNumThreads - 1) / kSortNumThreads;

  updatePeerRecordsKernel<<<numBlocks, numThreads, 0, stream>>>(
    numPatchesHomeAndProxy,
    records,
    peer_records,
    peerRecords
  );
}

MigrationCUDAKernel::MigrationCUDAKernel() {
  d_patchOffset_temp = NULL;
  d_patchOffsetNB_temp = NULL;

  patchDeviceScan_alloc = 0;
  d_patchDeviceScan_scratch = NULL;
}

MigrationCUDAKernel::~MigrationCUDAKernel() {
  if (d_patchOffset_temp != NULL) deallocate_device<int>(&d_patchOffset_temp);
  if (d_patchOffsetNB_temp != NULL) deallocate_device<int>(&d_patchOffsetNB_temp);
  if (d_patchDeviceScan_scratch != NULL) deallocate_device<char>(&d_patchDeviceScan_scratch);
}

void MigrationCUDAKernel::allocateScratch(const int numPatchesHomeAndProxy) {
  allocate_device<int>(&d_patchOffset_temp, numPatchesHomeAndProxy);
  allocate_device<int>(&d_patchOffsetNB_temp, numPatchesHomeAndProxy);
  
  // Fake call to CUB to get required size
  RecordToNumAtoms conversion_op;
  using InputIter = cub::TransformInputIterator<int, RecordToNumAtoms, CudaLocalRecord*>;
  InputIter iter(NULL, conversion_op);

  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum<InputIter>(
    d_temp_storage, temp_storage_bytes, 
    iter, d_patchOffset_temp, numPatchesHomeAndProxy
  );
  
  patchDeviceScan_alloc = temp_storage_bytes;
  allocate_device<char>(&d_patchDeviceScan_scratch, patchDeviceScan_alloc);
}

#endif // NODEGROUP_FORCE_REGISTER

