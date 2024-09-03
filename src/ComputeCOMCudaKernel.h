#ifndef COM_CUDA_KERNEL_H
#define COM_CUDA_KERNEL_H

#ifdef NAMD_CUDA
#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include <namd_cub/cub.cuh>
#endif
#endif // NAMD_CUDA

//JVV No reason not to do this, no?
#ifdef NAMD_HIP
//#if 0
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "HipDefines.h"
#define cub hipcub
#endif

#include "CudaUtils.h"
#include "CudaRecord.h"
#include "Lattice.h"

#ifdef NODEGROUP_FORCE_REGISTER

/*! Calculate the COM for numAtoms and store it in curCM */
template <int T_BLOCK_SIZE> 
__global__ static void computeCOMKernel(
  const int                numAtoms,
  const double             inv_group_mass,
  const Lattice            lat,
  const float * __restrict mass,
  const double* __restrict pos_x,
  const double* __restrict pos_y, 
  const double* __restrict pos_z, 
  const char3*  __restrict transform,
  const int*    __restrict atomsSOAIndex,
  double3*      __restrict d_curCM,
  double3*      __restrict curCM,
  unsigned int* __restrict tbcatomic){
  int tid = threadIdx.x + blockIdx.x * blockDim.x; 
  int totaltb = gridDim.x;
  bool isLastBlockDone = false;
  double3 cm = {0, 0, 0};

  if(tid < numAtoms){
    int SOAindex = atomsSOAIndex[tid];
    // uncoalesced memory access: too bad!
    double m = mass[SOAindex]; // Cast from float to double here
    double3 pos;
    pos.x = pos_x[SOAindex];
    pos.y = pos_y[SOAindex];
    pos.z = pos_z[SOAindex];

    // unwrap the  coordinate to calculate COM
    char3 t = transform[SOAindex];
    pos = lat.reverse_transform(pos, t);

    cm.x = pos.x * m;
    cm.y = pos.y * m;
    cm.z = pos.z * m;
  }

  // now reduce the values and add it to thread zero
#if 0
  typedef cub::BlockReduce<double, T_BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
   
  cm.x = BlockReduce(temp_storage).Sum(cm.x);
  __syncthreads();
  cm.y = BlockReduce(temp_storage).Sum(cm.y);
  __syncthreads();
  cm.z = BlockReduce(temp_storage).Sum(cm.z);
  __syncthreads();
#endif
  
  // add calculated (pos * mass) in each block
  if(threadIdx.x == 0){
    atomicAdd(&(d_curCM->x), cm.x);
    atomicAdd(&(d_curCM->y), cm.y);
    atomicAdd(&(d_curCM->z), cm.z);

    __threadfence();
    unsigned int value = atomicInc(&tbcatomic[0], totaltb);
    isLastBlockDone = (value == (totaltb -1));
  }
  __syncthreads();

  // Last block will set the host COM values
  if(isLastBlockDone){
    if(threadIdx.x == 0){
      // thread zero updates the host COM value
      curCM->x = d_curCM->x * inv_group_mass; // We calculate COM here
      curCM->y = d_curCM->y * inv_group_mass; // We calculate COM here
      curCM->z = d_curCM->z * inv_group_mass; // We calculate COM here
      // set the device values to zero
      d_curCM->x = 0.0;
      d_curCM->y = 0.0;
      d_curCM->z = 0.0; 
      //resets atomic counter
      tbcatomic[0] = 0;
      __threadfence();
    }
  }
}

/*! Calculate the COM for 2 groups of atoms and store it in 
curCM1 and curCM2.
Threads [0 , numGroup1Atoms) calculates COM of group 1
Threads [numGroup1Atoms , totalNumRestrained) 
calculates COM of group 2 */
template <int T_BLOCK_SIZE> 
__global__ static void compute2COMKernel(
  const int                numGroup1Atoms,
  const int                totalNumRestrained,
  const double             inv_group1_mass,
  const double             inv_group2_mass,
  const Lattice            lat,
  const float * __restrict mass,
  const double* __restrict pos_x,
  const double* __restrict pos_y, 
  const double* __restrict pos_z, 
  const char3*  __restrict transform,
  const int*    __restrict groupAtomsSOAIndex,
  double3*      __restrict d_curCM1,
  double3*      __restrict d_curCM2,
  double3*      __restrict curCM1,
  double3*      __restrict curCM2,
  unsigned int* __restrict tbcatomic){

  int tid = threadIdx.x + blockIdx.x * blockDim.x; 
  int totaltb = gridDim.x;
  bool isLastBlockDone = false;
  double3 com1 = {0, 0, 0};
  double3 com2 = {0, 0, 0};

  if (tid < totalNumRestrained) {
    int SOAindex = groupAtomsSOAIndex[tid];
    // uncoalesced memory access: too bad!
    double m = mass[SOAindex]; // Cast from float to double here
    double3 pos;
    pos.x = pos_x[SOAindex];
    pos.y = pos_y[SOAindex];
    pos.z = pos_z[SOAindex];

    // unwrap the coordinate to calculate COM
    char3 t = transform[SOAindex];
    pos = lat.reverse_transform(pos, t);

    if(tid < numGroup1Atoms){ 
      // threads [0 , numGroup1Atoms) calculate COM of group 1
      // we initialized the com2 to zero
      com1.x = pos.x * m;
      com1.y = pos.y * m;
      com1.z = pos.z * m;
    } else {
      // threads [numGroup1Atoms , totalNumRestrained) calculate COM of group 2
      // we initialized the com1 to zero
      com2.x = pos.x * m;
      com2.y = pos.y * m;
      com2.z = pos.z * m; 
    }
  }
  __syncthreads();

#if 0
  // now reduce the values and add it to thread zero
  typedef cub::BlockReduce<double, T_BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
   
  com1.x = BlockReduce(temp_storage).Sum(com1.x);
  __syncthreads();
  com1.y = BlockReduce(temp_storage).Sum(com1.y);
  __syncthreads();
  com1.z = BlockReduce(temp_storage).Sum(com1.z);
  __syncthreads(); 
  com2.x = BlockReduce(temp_storage).Sum(com2.x);
  __syncthreads();
  com2.y = BlockReduce(temp_storage).Sum(com2.y);
  __syncthreads();
  com2.z = BlockReduce(temp_storage).Sum(com2.z);
  __syncthreads(); 
#endif
  // add calculated (pos * mass) in each block
  if(threadIdx.x == 0){
    atomicAdd(&(d_curCM1->x), com1.x);
    atomicAdd(&(d_curCM1->y), com1.y);
    atomicAdd(&(d_curCM1->z), com1.z);
    atomicAdd(&(d_curCM2->x), com2.x);
    atomicAdd(&(d_curCM2->y), com2.y);
    atomicAdd(&(d_curCM2->z), com2.z);

    __threadfence();
    unsigned int value = atomicInc(&tbcatomic[0], totaltb);
    isLastBlockDone = (value == (totaltb -1));
  }
  __syncthreads();

  // Last block will set the host COM values
  if(isLastBlockDone){
    if(threadIdx.x == 0){
      // thread zero updates the host COM value
      curCM1->x = d_curCM1->x * inv_group1_mass; // We calculate COM here
      curCM1->y = d_curCM1->y * inv_group1_mass; // We calculate COM here
      curCM1->z = d_curCM1->z * inv_group1_mass; // We calculate COM here
      curCM2->x = d_curCM2->x * inv_group2_mass; // We calculate COM here
      curCM2->y = d_curCM2->y * inv_group2_mass; // We calculate COM here
      curCM2->z = d_curCM2->z * inv_group2_mass; // We calculate COM here
      // set the device values to zero
      d_curCM1->x = 0.0;
      d_curCM1->y = 0.0;
      d_curCM1->z = 0.0;
      d_curCM2->x = 0.0;
      d_curCM2->y = 0.0;
      d_curCM2->z = 0.0;  
      //resets atomic counter
      tbcatomic[0] = 0;
      __threadfence();
    }
  }
}

#endif // NODEGROUP_FORCE_REGISTER
#endif // COM_CUDA_KERNEL_H
