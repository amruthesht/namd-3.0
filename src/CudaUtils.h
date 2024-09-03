#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#ifdef NAMD_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // NAMD_CUDA

#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif  // NAMD_HIP

#include <stdio.h>
#include "HipDefines.h"

#ifdef NAMD_CUDA
#define WARPSIZE 32
#define BOUNDINGBOXSIZE 32
typedef unsigned int WarpMask;
#endif  // NAMD_CUDA 

#ifdef NAMD_HIP

/* JM: Definition of the warpsize: we need to be careful here since WARPSIZE is 
       a compile-time definition, but on HIP we can target both RDNA and CDNA 
       devices, which could be either wave32 or 64. 
*/ 
#ifdef NAMD_NAVI_BUILD
#define WARPSIZE 32
#else
#define WARPSIZE 64
#endif

#define BOUNDINGBOXSIZE 32
typedef unsigned int WarpMask;
#endif // NAMD_HIP

#if defined(NAMD_CUDA) || defined(NAMD_HIP)

#define FORCE_ENERGY_TABLE_SIZE 4096

#define COPY_CUDATENSOR(S,D)              \
  D.xx = S.xx; \
  D.xy = S.xy; \
  D.xz = S.xz; \
  D.yx = S.yx; \
  D.yy = S.yy; \
  D.yz = S.yz; \
  D.zx = S.zx; \
  D.zy = S.zy; \
  D.zz = S.zz

#define COPY_CUDAVECTOR(S,D) \
  D.x = S.x; \
  D.y = S.y; \
  D.z = S.z  

#define PRINT_CUDATENSOR(T, SS) \
  SS << T.xx << " " << T.xy << " " << T.xz << " " << T.yx << " " << \
        T.yy << " " << T.yz << " " << T.zx << " " << T.zy << " " << T.zz << \
        std::endl;

#ifdef SHORTREALS
typedef float	BigReal;
#else
typedef double  BigReal;
#endif

#ifdef NAMD_CUDA
#define ATOMIC_BINS 1
#else
#define ATOMIC_BINS WARPSIZE
#endif

struct cudaTensor{
  BigReal xx;
  BigReal xy;
  BigReal xz;
  BigReal yx;
  BigReal yy;
  BigReal yz;
  BigReal zx;
  BigReal zy;
  BigReal zz;

  __forceinline__ __device__
  cudaTensor& operator+=(const cudaTensor& other){
    this->xx += other.xx;
    this->xy += other.xy;
    this->xz += other.xz;
    this->yx += other.yx;
    this->yy += other.yy;
    this->yz += other.yz;
    this->zx += other.zx;
    this->zy += other.zy;
    this->zz += other.zz;
    return *this;
  }

  __forceinline__ __device__
  friend cudaTensor operator+(const cudaTensor& t1, const cudaTensor& t2) {
    cudaTensor t;
    t.xx = t1.xx + t2.xx;
    t.xy = t1.xy + t2.xy;
    t.xz = t1.xz + t2.xz;
    t.yx = t1.yx + t2.yx;
    t.yy = t1.yy + t2.yy;
    t.yz = t1.yz + t2.yz;
    t.zx = t1.zx + t2.zx;
    t.zy = t1.zy + t2.zy;
    t.zz = t1.zz + t2.zz;
    return t;
  }
};

struct cudaVector{
  BigReal x;
  BigReal y;
  BigReal z;
};

struct CudaMInfo{
  int destPatchID[3][3][3];
};


#define FEP_BONDED_CUDA_DEBUG
#ifdef FEP_BONDED_CUDA_DEBUG
#include <iostream>
#endif



#define WARP_FULL_MASK 0xffffffff
//TODO:HIP verify
//#define WARP_FULL_MASK 0xffffffffffffffff

#if (__CUDACC_VER_MAJOR__ >= 9)
#define NAMD_USE_COOPERATIVE_GROUPS
#endif

#ifdef NAMD_USE_COOPERATIVE_GROUPS
  #define WARP_SHUFFLE_XOR(MASK, VAR, LANE, SIZE) \
    __shfl_xor_sync(MASK, VAR, LANE, SIZE)
  #define WARP_SHUFFLE_UP(MASK, VAR, DELTA, SIZE) \
    __shfl_up_sync(MASK, VAR, DELTA, SIZE)
  #define WARP_SHUFFLE_DOWN(MASK, VAR, DELTA, SIZE) \
    __shfl_down_sync(MASK, VAR, DELTA, SIZE)
  #define WARP_SHUFFLE(MASK, VAR, LANE, SIZE) \
    __shfl_sync(MASK, VAR, LANE, SIZE)
  #define WARP_ALL(MASK, P)     __all_sync(MASK, P)
  #define WARP_ANY(MASK, P)     __any_sync(MASK, P)
  #define WARP_BALLOT(MASK, P)  __ballot_sync(MASK, P)
  #define WARP_SYNC(MASK)       __syncwarp(MASK)
  #define BLOCK_SYNC            __barrier_sync(0)
#else
  #define WARP_SHUFFLE_XOR(MASK, VAR, LANE, SIZE) \
    __shfl_xor(VAR, LANE, SIZE)
  #define WARP_SHUFFLE_UP(MASK, VAR, DELTA, SIZE) \
    __shfl_up(VAR, DELTA, SIZE)
  #define WARP_SHUFFLE_DOWN(MASK, VAR, DELTA, SIZE) \
    __shfl_down(VAR, DELTA, SIZE)
  #define WARP_SHUFFLE(MASK, VAR, LANE, SIZE) \
    __shfl(VAR, LANE, SIZE)
#ifdef NAMD_HIP
  // this collides with a few rocprim definitions, so we need redefine this as NAMD_WARP_* 
  #define NAMD_WARP_ALL(MASK, P)     __all(P)
  #define NAMD_WARP_ANY(MASK, P)     __any(P)
  #define NAMD_WARP_BALLOT(MASK, P)  __ballot(P)
  #define NAMD_WARP_SYNC(MASK)       
  #define BLOCK_SYNC                 __syncthreads()
#else
  #define WARP_ALL(MASK, P)     __all(P)
  #define WARP_ANY(MASK, P)     __any(P)
  #define WARP_BALLOT(MASK, P)  __ballot(P)
  #define WARP_SYNC(MASK)      
  #define BLOCK_SYNC            __syncthreads()
#endif
#endif


/*
// Define float3 + float3 operation
__host__ __device__ inline float3 operator+(const float3 a, const float3 b) {
  float3 c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  c.z = a.z + b.z;
  return c;
}
*/

//
// Cuda static assert, copied from Facebook FFT sources. Remove once nvcc has c++11
//
template <bool>
struct CudaStaticAssert;

template <>
struct CudaStaticAssert<true> {
};

#define cuda_static_assert(expr) \
  (CudaStaticAssert<(expr) != 0>())

void cudaDie(const char *msg, cudaError_t err=cudaSuccess);
void curandDie(const char *msg, int err=0);

void cudaNAMD_bug(const char *msg);

//
// Error checking wrapper for CUDA
//
#define cudaCheck(stmt) do {                                 \
	cudaError_t err = stmt;                            \
  if (err != cudaSuccess) {                          \
  	char msg[256];	\
	  sprintf(msg, "%s in file %s, function %s, line %d\n", #stmt,__FILE__,__FUNCTION__,__LINE__); \
	  cudaDie(msg, err); \
  }                                                  \
} while(0)

#define curandCheck(stmt) do {                                 \
    curandStatus_t  err = stmt;                                   \
    if (err != CURAND_STATUS_SUCCESS) {                           \
      char msg[256];                                                    \
      sprintf(msg, "%s in file %s, function %s, line %d\n", #stmt,__FILE__,__FUNCTION__,__LINE__); \
      curandDie(msg, (int)err);                                   \
    }                                                                   \
  } while(0)

#ifdef __CUDACC__
#if ( __CUDACC_VER_MAJOR__ >= 8 ) && ( !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 )
// native implementation available
#else
#if __CUDA_ARCH__ >= 600
#error using CAS implementation of double atomicAdd
#endif
//
// Double precision atomicAdd, copied from CUDA_C_Programming_Guide.pdf (ver 5.0)
//
static __device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val +
           __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif
#endif

void clear_device_array_async_T(void *data, const size_t ndata, cudaStream_t stream, const size_t sizeofT);
void clear_device_array_T(void *data, const size_t ndata, const size_t sizeofT);

template <class T>
void clear_device_array(T *data, const size_t ndata, cudaStream_t stream=0) {
  clear_device_array_async_T(data, ndata, stream, sizeof(T));
}

template <class T>
void clear_device_array_sync(T *data, const size_t ndata) {
  clear_device_array_T(data, ndata, sizeof(T));
}

void allocate_host_T(void **pp, const size_t len, const size_t sizeofT);
//----------------------------------------------------------------------------------------
//
// Allocate page-locked host memory
// pp = memory pointer
// len = length of the array
//
template <class T>
void allocate_host(T **pp, const size_t len) {
  allocate_host_T((void **)pp, len, sizeof(T));
}


void allocate_device_T(void **pp, const size_t len, const size_t sizeofT);
void allocate_device_T_managed(void **pp, const size_t len, const size_t sizeofT);
void allocate_device_T_async(void **pp, const size_t len, const size_t sizeofT, cudaStream_t stream);
//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
template <class T>
void allocate_device(T **pp, const size_t len) {
  allocate_device_T((void **)pp, len, sizeof(T));
}

template <class T>
void allocate_device_managed(T **pp, const size_t len) {
  allocate_device_T_managed((void **)pp, len, sizeof(T));
}

template <class T>
void allocate_device_async(T **pp, const size_t len, cudaStream_t stream) {
  allocate_device_T_async((void **)pp, len, sizeof(T), stream);
}

void deallocate_device_T(void **pp);
void deallocate_device_T_async(void **pp, cudaStream_t stream);
//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
template <class T>
void deallocate_device(T **pp) {
  deallocate_device_T((void **)pp);
}
template <class T>
void deallocate_device_async(T **pp, cudaStream_t stream) {
  deallocate_device_T_async((void **)pp, stream);
}

//----------------------------------------------------------------------------------------

bool reallocate_device_T(void **pp, size_t *curlen, const size_t newlen, const float fac, const size_t sizeofT);
//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate device memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will be fac*newlen
//
// returns true if reallocation happened
//
template <class T>
bool reallocate_device(T **pp, size_t *curlen, const size_t newlen, const float fac=1.0f) {
  return reallocate_device_T((void **)pp, curlen, newlen, fac, sizeof(T));
}
//----------------------------------------------------------------------------------------
bool reallocate_host_T(void **pp, size_t *curlen, const size_t newlen, const float fac, 
		       const unsigned int flag, const size_t sizeofT);
//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate pinned host memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will be fac*newlen
// flag = allocation type:
//        cudaHostAllocDefault = default type, emulates cudaMallocHost
//        cudaHostAllocMapped  = maps allocation into CUDA address space
//
// returns true if reallocation happened
//
template <class T>
bool reallocate_host(T **pp, size_t *curlen, const size_t newlen,
		     const float fac=1.0f, const unsigned int flag=cudaHostAllocDefault) {
  return reallocate_host_T((void **)pp, curlen, newlen, fac, flag, sizeof(T));
}

void deallocate_host_T(void **pp);
//----------------------------------------------------------------------------------------
//
// Deallocate page-locked host memory
// pp = memory pointer
//
template <class T>
void deallocate_host(T **pp) {
  deallocate_host_T((void **)pp);
}
//----------------------------------------------------------------------------------------

void copy_HtoD_async_T(const void *h_array, void *d_array, size_t array_len, cudaStream_t stream,
           const size_t sizeofT);
void copy_HtoD_T(const void *h_array, void *d_array, size_t array_len,
     const size_t sizeofT);
void copy_DtoH_async_T(const void *d_array, void *h_array, const size_t array_len, cudaStream_t stream,
           const size_t sizeofT);
void copy_DtoH_T(const void *d_array, void *h_array, const size_t array_len, const size_t sizeofT);

void copy_DtoD_async_T(const void *d_src, void *d_dst, const size_t array_len, cudaStream_t stream,
           const size_t sizeofT);
void copy_DtoD_T(const void *d_src, void *d_dst, const size_t array_len, const size_t sizeofT);

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
template <class T>
void copy_HtoD(const T *h_array, T *d_array, size_t array_len, cudaStream_t stream=0) {
  copy_HtoD_async_T(h_array, d_array, array_len, stream, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device using synchronous calls
//
template <class T>
void copy_HtoD_sync(const T *h_array, T *d_array, size_t array_len) {
  copy_HtoD_T(h_array, d_array, array_len, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
template <class T>
void copy_DtoH(const T *d_array, T *h_array, const size_t array_len, cudaStream_t stream=0) {
  copy_DtoH_async_T(d_array, h_array, array_len, stream, sizeof(T));
}
//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host using synchronous calls
//
template <class T>
void copy_DtoH_sync(const T *d_array, T *h_array, const size_t array_len) {
  copy_DtoH_T(d_array, h_array, array_len, sizeof(T));
}
//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Device
//
template <class T>
void copy_DtoD(const T *d_src, T *h_dst, const size_t array_len, cudaStream_t stream=0) {
  copy_DtoD_async_T(d_src, h_dst, array_len, stream, sizeof(T));
}
//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Device using synchronous calls
//
template <class T>
void copy_DtoD_sync(const T *d_src, T *h_dst, const size_t array_len) {
  copy_DtoD_T(d_src, h_dst, array_len, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies memory between two peer devices Device -> Device
//
void copy_PeerDtoD_async_T(const int src_dev, const int dst_dev,
  const void *d_src, void *d_dst, const size_t array_len, cudaStream_t stream,
  const size_t sizeofT);

template <class T>
void copy_PeerDtoD(const int src_dev, const int dst_dev,
  const T *d_src, T *d_dst, const size_t array_len, cudaStream_t stream=0) {
  copy_PeerDtoD_async_T(src_dev, dst_dev, d_src, d_dst, array_len, stream, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies 3D memory block Host -> Device
//
void copy3D_HtoD_T(void* src_data, void* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  size_t sizeofT, cudaStream_t stream);

template <class T>
void copy3D_HtoD(T* src_data, T* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  cudaStream_t stream=0) {
  copy3D_HtoD_T(src_data, dst_data,
    src_x0, src_y0, src_z0,
    src_xsize, src_ysize,
    dst_x0, dst_y0, dst_z0,
    dst_xsize, dst_ysize,
    width, height, depth,
    sizeof(T), stream);
}

//----------------------------------------------------------------------------------------
//
// Copies 3D memory block Device -> Host
//
void copy3D_DtoH_T(void* src_data, void* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  size_t sizeofT, cudaStream_t stream);

template <class T>
void copy3D_DtoH(T* src_data, T* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  cudaStream_t stream=0) {
  copy3D_DtoH_T(src_data, dst_data,
    src_x0, src_y0, src_z0,
    src_xsize, src_ysize,
    dst_x0, dst_y0, dst_z0,
    dst_xsize, dst_ysize,
    width, height, depth,
    sizeof(T), stream);
}

//----------------------------------------------------------------------------------------
//
// Copies 3D memory block Device -> Device
//
void copy3D_DtoD_T(void* src_data, void* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  size_t sizeofT, cudaStream_t stream);

template <class T>
void copy3D_DtoD(T* src_data, T* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  cudaStream_t stream=0) {
  copy3D_DtoD_T(src_data, dst_data,
    src_x0, src_y0, src_z0,
    src_xsize, src_ysize,
    dst_x0, dst_y0, dst_z0,
    dst_xsize, dst_ysize,
    width, height, depth,
    sizeof(T), stream);
}

//----------------------------------------------------------------------------------------
//
// Copies 3D memory block between two peer devices Device -> Device
//
void copy3D_PeerDtoD_T(int src_dev, int dst_dev,
  void* src_data, void* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  size_t sizeofT, cudaStream_t stream);

template <class T>
void copy3D_PeerDtoD(int src_dev, int dst_dev,
  T* src_data, T* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  cudaStream_t stream=0) {
  copy3D_PeerDtoD_T(src_dev, dst_dev,
    src_data, dst_data,
    src_x0, src_y0, src_z0,
    src_xsize, src_ysize,
    dst_x0, dst_y0, dst_z0,
    dst_xsize, dst_ysize,
    width, height, depth,
    sizeof(T), stream);
}

//----------------------------------------------------------------------------------------
//
// Utility structure for computing nonbonded interactions
//
// DMC: I needed this to be in a file which can be included from both C++ and CUDA files
struct CudaNBConstants {
  float lj_0; // denom * cutoff2 - 3.0f * switch2 * denom
  float lj_1; // denom * 2.0f
  float lj_2; // denom * -12.0f
  float lj_3; // denom *  12.0f * switch2
  float lj_4; // cutoff2
  float lj_5; // switch2
  float e_0; // roff3Inv
  float e_0_slow; // roff3Inv * (1 - slowScale)
  float e_1; // roff2Inv
  float e_2; // roffInv
  float ewald_0; // ewaldcof
  float ewald_1; // pi_ewaldcof
  float ewald_2; // ewaldcof ^ 2
  float ewald_3_slow; // ewaldcof ^ 3 * slowScale
  float slowScale; // ratio of full electrostatics to nonbonded frequency
};


#endif // NAMD_CUDA || NAMD_HIP

#endif // CUDAUTILS_H

