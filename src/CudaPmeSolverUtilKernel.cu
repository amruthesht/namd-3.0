#ifdef WIN32
#define _USE_MATH_DEFINES
#endif  

#include <math.h>
#include <stdio.h>
#ifdef NAMD_CUDA
#include <cuda.h>
#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include <namd_cub/cub.cuh>
#endif
#endif

#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#define cub hipcub
#endif

#include "HipDefines.h"
#include "CudaUtils.h"
#include "CudaPmeSolverUtilKernel.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
// CCELEC is 1/ (4 pi eps ) in AKMA units, conversion from SI
// units: CCELEC = e*e*Na / (4*pi*eps*1Kcal*1A)
//
//      parameter :: CCELEC=332.0636D0 ! old value of dubious origin
//      parameter :: CCELEC=331.843D0  ! value from 1986-1987 CRC Handbook
//                                   ! of Chemistry and Physics
//  real(chm_real), parameter ::  &
//       CCELEC_amber    = 332.0522173D0, &
//       CCELEC_charmm   = 332.0716D0   , &
//       CCELEC_discover = 332.054D0    , &
//       CCELEC_namd     = 332.0636D0   
//const double ccelec = 332.0636;
//const double half_ccelec = 0.5*ccelec;
//const float ccelec_float = 332.0636f;

/*
// Structure into which virials are stored
// NOTE: This structure is only used for computing addresses
struct Virial_t {
  double sforce_dp[27][3];
  long long int sforce_fp[27][3];
  double virmat[9];
  // Energies start here ...
};
*/

// Local structure for scalar_sum -function for energy and virial reductions
struct RecipVirial_t {
  double energy;
  double virial[6];
};


__forceinline__ __device__ double expfunc(const double x) {
  return exp(x);
}

__forceinline__ __device__ float expfunc(const float x) {
  return __expf(x);
}

//
// Performs scalar sum on data(nfft1, nfft2, nfft3)
// T = float or double
// T2 = float2 or double2
//
template <typename T, typename T2, bool calc_energy_virial, bool orderXYZ, bool doOrtho>
__global__ void scalar_sum_kernel(const int nfft1, const int nfft2, const int nfft3,
          const int size1, const int size2, const int size3,
          const T recip1x, const T recip1y, const T recip1z,
          const T recip2x, const T recip2y, const T recip2z,
          const T recip3x, const T recip3y, const T recip3z,
          const T* prefac1, const T* prefac2, const T* prefac3,
          const T pi_ewald, const T piv_inv,
          const int k2_00, const int k3_00,
          T2* data,
          double* __restrict__ energy_recip,
          double* __restrict__ virial) {
  // Shared memory required: sizeof(T)*(nfft1 + nfft2 + nfft3)
  #ifdef NAMD_CUDA
  extern __shared__ T sh_prefac[];
  #else //NAMD_HIP
  HIP_DYNAMIC_SHARED( T, sh_prefac)
  #endif

  // Create pointers to shared memory
  T* sh_prefac1 = (T *)&sh_prefac[0];
  T* sh_prefac2 = (T *)&sh_prefac[nfft1];
  T* sh_prefac3 = (T *)&sh_prefac[nfft1 + nfft2];

  // Calculate start position (k1, k2, k3) for each thread
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int k3 = tid/(size1*size2);
  tid -= k3*size1*size2;
  int k2 = tid/size1;
  int k1 = tid - k2*size1;

  // Starting position in data
  int pos = k1 + (k2 + k3*size2)*size1;

  // Move (k2, k3) to the global coordinate (k1_00 = 0 since this is the pencil direction)
  k2 += k2_00;
  k3 += k3_00;

  // Calculate limits w.r.t. global coordinates
  const int lim2 = size2 + k2_00;
  const int lim3 = size3 + k3_00;

  // Calculate increments (k1_inc, k2_inc, k3_inc)
  int tot_inc = blockDim.x*gridDim.x;
  const int k3_inc = tot_inc/(size1*size2);
  tot_inc -= k3_inc*size1*size2;
  const int k2_inc = tot_inc/size1;
  const int k1_inc = tot_inc - k2_inc*size1;

  // Set data[0] = 0 for the global (0,0,0)
  if (k1 == 0 && k2 == 0 && k3 == 0) {
    T2 zero;
    zero.x = (T)0;
    zero.y = (T)0;
    data[0] = zero;
    // Increment position
    k1 += k1_inc;
    pos += k1_inc;
    if (k1 >= size1) {
      k1 -= size1;
      k2++;
    }
    k2 += k2_inc;
    pos += k2_inc*size1;
    if (k2 >= lim2) {
      k2 -= size2;
      k3++;
    }
    k3 += k3_inc;
    pos += k3_inc*size1*size2;
  }

  // Load prefac data into shared memory
  {
    int t = threadIdx.x;
    while (t < nfft1) {
      sh_prefac1[t] = prefac1[t];
      t += blockDim.x;
    }
    t = threadIdx.x;
    while (t < nfft2) {
      sh_prefac2[t] = prefac2[t];
      t += blockDim.x;
    }
    t = threadIdx.x;
    while (t < nfft3) {
      sh_prefac3[t] = prefac3[t];
      t += blockDim.x;
    }
  }
  BLOCK_SYNC;

  double energy = 0.0;
  double virial0 = 0.0;
  double virial1 = 0.0;
  double virial2 = 0.0;
  double virial3 = 0.0;
  double virial4 = 0.0;
  double virial5 = 0.0;

  // If nfft1 is odd, set nfft1_half to impossible value so that
  // the second condition in "if ( (k1 == 0) || (k1 == nfft1_half) )" 
  // is never satisfied
  const int nfft1_half = ((nfft1 & 1) == 0) ? nfft1/2 : -1;
  const int nfft2_half = nfft2/2;
  const int nfft3_half = nfft3/2;

  while (k3 < lim3) {

    T2 q = data[pos];

    int k2s = (k2 <= nfft2_half) ? k2 : k2 - nfft2;
    int k3s = (k3 <= nfft3_half) ? k3 : k3 - nfft3;

    T m1, m2, m3;
    if (doOrtho) {
      m1 = recip1x*k1;
      m2 = recip2y*k2s;
      m3 = recip3z*k3s;
    } else {
      m1 = recip1x*k1 + recip2x*k2s + recip3x*k3s;
      m2 = recip1y*k1 + recip2y*k2s + recip3y*k3s;
      m3 = recip1z*k1 + recip2z*k2s + recip3z*k3s;
    }

    T msq = m1*m1 + m2*m2 + m3*m3;
    T msq_inv = ((T)1)/msq;

    T theta3 = sh_prefac1[k1]*sh_prefac2[k2]*sh_prefac3[k3];
    T q2 = ((T)2)*(q.x*q.x + q.y*q.y)*theta3;
    if ( (k1 == 0) || (k1 == nfft1_half) ) q2 *= ((T)0.5);
    T xp3 = expfunc(-pi_ewald*msq)*piv_inv;
    T C = xp3*msq_inv;
    T theta = theta3*C;

    if (calc_energy_virial) {
      T fac = q2*C;
      T vir = ((T)2)*(pi_ewald + msq_inv);
      energy += fac;
      virial0 += (double)(fac*(vir*m1*m1 - ((T)1)));
      virial1 += (double)(fac*vir*m1*m2);
      virial2 += (double)(fac*vir*m1*m3);
      virial3 += (double)(fac*(vir*m2*m2 - ((T)1)));
      virial4 += (double)(fac*vir*m2*m3);
      virial5 += (double)(fac*(vir*m3*m3 - ((T)1)));
    }

    q.x *= theta;
    q.y *= theta;
    data[pos] = q;
    
    // Increment position
    k1 += k1_inc;
    pos += k1_inc;
    if (k1 >= size1) {
      k1 -= size1;
      k2++;
    }
    k2 += k2_inc;
    pos += k2_inc*size1;
    if (k2 >= lim2) {
      k2 -= size2;
      k3++;
    }
    k3 += k3_inc;
    pos += k3_inc*size2*size1;
  }

  // Reduce energy and virial
  if (calc_energy_virial) {
    const int tid = threadIdx.x & (warpSize-1);
    const int base = (threadIdx.x/warpSize);
    volatile RecipVirial_t* sh_ev = (RecipVirial_t *)sh_prefac;
    // Reduce within warps
    for (int d=warpSize/2;d >= 1;d /= 2) {
      energy += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(energy), tid+d, WARPSIZE),
         WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(energy), tid+d, WARPSIZE));
      virial0 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial0), tid+d, WARPSIZE),
          WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial0), tid+d, WARPSIZE));
      virial1 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial1), tid+d, WARPSIZE),
          WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial1), tid+d, WARPSIZE));
      virial2 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial2), tid+d, WARPSIZE),
          WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial2), tid+d, WARPSIZE));
      virial3 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial3), tid+d, WARPSIZE),
          WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial3), tid+d, WARPSIZE));
      virial4 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial4), tid+d, WARPSIZE),
          WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial4), tid+d, WARPSIZE));
      virial5 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial5), tid+d, WARPSIZE),
          WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial5), tid+d, WARPSIZE));
    }
    // Reduce between warps
    // NOTE: this BLOCK_SYNC is needed because we're using a single shared memory buffer
    BLOCK_SYNC;
    if (tid == 0) {
      sh_ev[base].energy = energy;
      sh_ev[base].virial[0] = virial0;
      sh_ev[base].virial[1] = virial1;
      sh_ev[base].virial[2] = virial2;
      sh_ev[base].virial[3] = virial3;
      sh_ev[base].virial[4] = virial4;
      sh_ev[base].virial[5] = virial5;
    }
    BLOCK_SYNC;
    if (base == 0) {
      energy = (tid < blockDim.x/warpSize) ? sh_ev[tid].energy : 0.0;
      virial0 = (tid < blockDim.x/warpSize) ? sh_ev[tid].virial[0] : 0.0;
      virial1 = (tid < blockDim.x/warpSize) ? sh_ev[tid].virial[1] : 0.0;
      virial2 = (tid < blockDim.x/warpSize) ? sh_ev[tid].virial[2] : 0.0;
      virial3 = (tid < blockDim.x/warpSize) ? sh_ev[tid].virial[3] : 0.0;
      virial4 = (tid < blockDim.x/warpSize) ? sh_ev[tid].virial[4] : 0.0;
      virial5 = (tid < blockDim.x/warpSize) ? sh_ev[tid].virial[5] : 0.0;
      for (int d=warpSize/2;d >= 1;d /= 2) {
  energy += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(energy), tid+d, WARPSIZE),
           WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(energy), tid+d, WARPSIZE));
  virial0 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial0), tid+d, WARPSIZE),
            WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial0), tid+d, WARPSIZE));
  virial1 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial1), tid+d, WARPSIZE),
            WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial1), tid+d, WARPSIZE));
  virial2 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial2), tid+d, WARPSIZE),
            WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial2), tid+d, WARPSIZE));
  virial3 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial3), tid+d, WARPSIZE),
            WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial3), tid+d, WARPSIZE));
  virial4 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial4), tid+d, WARPSIZE),
            WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial4), tid+d, WARPSIZE));
  virial5 += __hiloint2double(WARP_SHUFFLE(WARP_FULL_MASK, __double2hiint(virial5), tid+d, WARPSIZE),
            WARP_SHUFFLE(WARP_FULL_MASK, __double2loint(virial5), tid+d, WARPSIZE));
      }
    }

    if (threadIdx.x == 0) {
      atomicAdd(energy_recip, energy*0.5);
      virial0 *= -0.5;
      virial1 *= -0.5;
      virial2 *= -0.5;
      virial3 *= -0.5;
      virial4 *= -0.5;
      virial5 *= -0.5;
      atomicAdd(&virial[0], virial0);
      atomicAdd(&virial[1], virial1);
      atomicAdd(&virial[2], virial2);
      atomicAdd(&virial[3], virial3);
      atomicAdd(&virial[4], virial4);
      atomicAdd(&virial[5], virial5);
    }

  }

}

template <typename T>
__forceinline__ __device__ void write_grid(const float val, const int ind,
             T* data) {
  atomicAdd(&data[ind], (T)val);
}

//
// General version for any order
//
template <typename T, int order>
__forceinline__ __device__ void calc_one_theta(const T w, T *theta) {

  theta[order-1] = ((T)0);
  theta[1] = w;
  theta[0] = ((T)1) - w;

#pragma unroll
  for (int k=3;k <= order-1;k++) {
    T div = ((T)1) / (T)(k-1);
    theta[k-1] = div*w*theta[k-2];
#pragma unroll
    for (int j=1;j <= k-2;j++) {
      theta[k-j-1] = div*((w+j)*theta[k-j-2] + (k-j-w)*theta[k-j-1]);
    }
    theta[0] = div*(((T)1) - w)*theta[0];
  }
      
  //--- one more recursion
  T div = ((T)1) / (T)(order-1);
  theta[order-1] = div*w*theta[order-2];
#pragma unroll
  for (int j=1;j <= order-2;j++) {
    theta[order-j-1] = div*((w+j)*theta[order-j-2] + (order-j-w)*theta[order-j-1]);
  }
    
  theta[0] = div*(((T)1) - w)*theta[0];
}

//
// Calculate theta and dtheta for general order bspline
//
template <typename T, typename T3, int order>
__forceinline__ __device__ void calc_theta_dtheta(T wx, T wy, T wz, T3 *theta, T3 *dtheta) {

  theta[order-1].x = ((T)0);
  theta[order-1].y = ((T)0);
  theta[order-1].z = ((T)0);
  theta[1].x = wx;
  theta[1].y = wy;
  theta[1].z = wz;
  theta[0].x = ((T)1) - wx;
  theta[0].y = ((T)1) - wy;
  theta[0].z = ((T)1) - wz;

#pragma unroll
  for (int k=3;k <= order-1;k++) {
    T div = ((T)1) / (T)(k-1);
    theta[k-1].x = div*wx*theta[k-2].x;
    theta[k-1].y = div*wy*theta[k-2].y;
    theta[k-1].z = div*wz*theta[k-2].z;
#pragma unroll
    for (int j=1;j <= k-2;j++) {
      theta[k-j-1].x = div*((wx + j)*theta[k-j-2].x + (k-j-wx)*theta[k-j-1].x);
      theta[k-j-1].y = div*((wy + j)*theta[k-j-2].y + (k-j-wy)*theta[k-j-1].y);
      theta[k-j-1].z = div*((wz + j)*theta[k-j-2].z + (k-j-wz)*theta[k-j-1].z);
    }
    theta[0].x = div*(((T)1) - wx)*theta[0].x;
    theta[0].y = div*(((T)1) - wy)*theta[0].y;
    theta[0].z = div*(((T)1) - wz)*theta[0].z;
  }

  //--- perform standard b-spline differentiation
  dtheta[0].x = -theta[0].x;
  dtheta[0].y = -theta[0].y;
  dtheta[0].z = -theta[0].z;
#pragma unroll
  for (int j=2;j <= order;j++) {
    dtheta[j-1].x = theta[j-2].x - theta[j-1].x;
    dtheta[j-1].y = theta[j-2].y - theta[j-1].y;
    dtheta[j-1].z = theta[j-2].z - theta[j-1].z;
  }
      
  //--- one more recursion
  T div = ((T)1) / (T)(order-1);
  theta[order-1].x = div*wx*theta[order-2].x;
  theta[order-1].y = div*wy*theta[order-2].y;
  theta[order-1].z = div*wz*theta[order-2].z;
#pragma unroll
  for (int j=1;j <= order-2;j++) {
    theta[order-j-1].x = div*((wx + j)*theta[order-j-2].x + (order-j-wx)*theta[order-j-1].x);
    theta[order-j-1].y = div*((wy + j)*theta[order-j-2].y + (order-j-wy)*theta[order-j-1].y);
    theta[order-j-1].z = div*((wz + j)*theta[order-j-2].z + (order-j-wz)*theta[order-j-1].z);
  }
    
  theta[0].x = div*(((T)1) - wx)*theta[0].x;
  theta[0].y = div*(((T)1) - wy)*theta[0].y;
  theta[0].z = div*(((T)1) - wz)*theta[0].z;
}

//
// Spreads the charge on the grid. Calculates theta and dtheta on the fly
// blockDim.x                   = Number of atoms each block loads
// blockDim.y*blockDim.x/order3 = Number of atoms we spread at once
//
template <typename AT, int order, bool periodicY, bool periodicZ>
__global__ void
spread_charge_kernel(const float4 *xyzq, const int ncoord,
          const int nfftx, const int nffty, const int nfftz,
          const int xsize, const int ysize, const int zsize,
          const int xdim, const int y00, const int z00,
          AT* data) {

  // Shared memory use:
  // order = 4: 1920 bytes
  // order = 6: 2688 bytes
  // order = 8: 3456 bytes
  __shared__ int sh_ix[32];
  __shared__ int sh_iy[32];
  __shared__ int sh_iz[32];
  __shared__ float sh_thetax[order*32];
  __shared__ float sh_thetay[order*32];
  __shared__ float sh_thetaz[order*32];

  // Process atoms pos to pos_end-1 (blockDim.x)
  const unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int pos_end = min((blockIdx.x+1)*blockDim.x, ncoord);

  if (pos < pos_end && threadIdx.y == 0) {

    float4 xyzqi = xyzq[pos];
    float x = xyzqi.x;
    float y = xyzqi.y;
    float z = xyzqi.z;
    float q = xyzqi.w;

    float frx = ((float)nfftx)*x;
    float fry = ((float)nffty)*y;
    float frz = ((float)nfftz)*z;

    int frxi = (int)frx;
    int fryi = (int)fry;
    int frzi = (int)frz;

    float wx = frx - (float)frxi;
    float wy = fry - (float)fryi;
    float wz = frz - (float)frzi;

    if (!periodicY && y00 == 0 && fryi >= ysize) fryi -= nffty;
    if (!periodicZ && z00 == 0 && frzi >= zsize) frzi -= nfftz;

    sh_ix[threadIdx.x] = frxi;
    sh_iy[threadIdx.x] = fryi - y00;
    sh_iz[threadIdx.x] = frzi - z00;

    float theta[order];

    calc_one_theta<float, order>(wx, theta);
#pragma unroll
    for (int i=0;i < order;i++) sh_thetax[threadIdx.x*order + i] = q*theta[i];

    calc_one_theta<float, order>(wy, theta);
#pragma unroll
    for (int i=0;i < order;i++) sh_thetay[threadIdx.x*order + i] = theta[i];

    calc_one_theta<float, order>(wz, theta);
#pragma unroll
    for (int i=0;i < order;i++) sh_thetaz[threadIdx.x*order + i] = theta[i];

  }

  BLOCK_SYNC;

  // Grid point location, values of (ix0, iy0, iz0) are in range 0..order-1
  // NOTE: Only tid=0...order*order*order-1 do any computation
  const int order3 = ((order*order*order-1)/32 + 1)*32;
  const int tid = (threadIdx.x + threadIdx.y*blockDim.x) % order3;   // 0...order3-1
  const int x0 = tid % order;
  const int y0 = (tid / order) % order;
  const int z0 = tid / (order*order);

  // Loop over atoms pos..pos_end-1
  int iadd = blockDim.x*blockDim.y/order3;
  int i = (threadIdx.x + threadIdx.y*blockDim.x)/order3;
  int iend = pos_end - blockIdx.x*blockDim.x;
  for (;i < iend;i += iadd) {
    int x = sh_ix[i] + x0;
    int y = sh_iy[i] + y0;
    int z = sh_iz[i] + z0;
      
    if (x >= nfftx) x -= nfftx;

    if (periodicY  && (y >= nffty)) y -= nffty;
    if (!periodicY && (y < 0 || y >= ysize)) continue;

    if (periodicZ  && (z >= nfftz)) z -= nfftz;
    if (!periodicZ && (z < 0 || z >= zsize)) continue;
      
    // Get position on the grid
    int ind = x + xdim*(y + ysize*(z));
    
    // Here we unroll the 6x6x6 loop with 216 threads.
    // NOTE: We use 7*32=224 threads to do this
    // Calculate interpolated charge value and store it to global memory

    if (tid < order*order*order)
      write_grid<AT>(sh_thetax[i*order+x0]*sh_thetay[i*order+y0]*sh_thetaz[i*order+z0], ind, data);
  }

}


template <typename AT, int order, bool periodicY, bool periodicZ>
__global__ void
spread_charge_kernel_v2(const float4 *xyzq, const int ncoord,
          const int nfftx, const int nffty, const int nfftz,
          const float nfftx_f, const float nffty_f, 
          const float nfftz_f,
          const int xsize, const int ysize, const int zsize,
          const int xdim, const int y00, const int z00,
          AT* data) {

  // Shared memory use:
  // order = 4: 1920 bytes
  // order = 6: 2688 bytes
  // order = 8: 3456 bytes
  __shared__ int sh_ix[WARPSIZE];
  __shared__ int sh_iy[WARPSIZE];
  __shared__ int sh_iz[WARPSIZE];
  __shared__ float sh_thetax[order*WARPSIZE];
  __shared__ float sh_thetay[order*WARPSIZE];
  __shared__ float sh_thetaz[order*WARPSIZE];

  // Process atoms pos to pos_end-1 (blockDim.x)
  const unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int pos_end = min((blockIdx.x+1)*blockDim.x, ncoord);
  // const float invorder = 1.f/order;

  if (pos < pos_end && threadIdx.y == 0) {

    float4 xyzqi = xyzq[pos];
    float x = xyzqi.x;
    float y = xyzqi.y;
    float z = xyzqi.z;
    float q = xyzqi.w;

    float frx = (nfftx_f)*x;
    float fry = (nffty_f)*y;
    float frz = (nfftz_f)*z;

    int frxi = (int)frx;
    int fryi = (int)fry;
    int frzi = (int)frz;

    float wx = frx - (float)frxi;
    float wy = fry - (float)fryi;
    float wz = frz - (float)frzi;

    if (!periodicY && y00 == 0 && fryi >= ysize) fryi -= nffty;
    if (!periodicZ && z00 == 0 && frzi >= zsize) frzi -= nfftz;

    sh_ix[threadIdx.x] = frxi;
    sh_iy[threadIdx.x] = fryi - y00;
    sh_iz[threadIdx.x] = frzi - z00;

    float theta[order];

    calc_one_theta<float, order>(wx, theta);
#pragma unroll
    for (int i=0;i < order;i++) sh_thetax[threadIdx.x*order + i] = q*theta[i];

    calc_one_theta<float, order>(wy, theta);
#pragma unroll
    for (int i=0;i < order;i++) sh_thetay[threadIdx.x*order + i] = theta[i];

    calc_one_theta<float, order>(wz, theta);
#pragma unroll
    for (int i=0;i < order;i++) sh_thetaz[threadIdx.x*order + i] = theta[i];

  }

  BLOCK_SYNC;

  // Grid point location, values of (ix0, iy0, iz0) are in range 0..order-1
  // NOTE: Only tid=0...order*order*order-1 do any computation
  const int order3 = ((order*order*order-1)/WARPSIZE + 1)*WARPSIZE;
  const int tid = (threadIdx.x + threadIdx.y*blockDim.x) % order3;   // 0...order3-1
  const int x0 = tid % order;
  const int y0 = (tid / order) % order;
  //const int y0 = (int)(tid * invorder) % order;
  const int z0 = tid / (order*order);
  //const int z0 = tid * invorder * invorder;

  // Loop over atoms pos..pos_end-1
  int iadd = blockDim.x*blockDim.y/order3;
  int i = (threadIdx.x + threadIdx.y*blockDim.x)/order3;
  int iend = pos_end - blockIdx.x*blockDim.x;
  for (;i < iend;i += iadd) {
    int x = sh_ix[i] + x0;
    int y = sh_iy[i] + y0;
    int z = sh_iz[i] + z0;
      
    if (x >= nfftx) x -= nfftx;

    if (periodicY  && (y >= nffty)) y -= nffty;
    if (!periodicY && (y < 0 || y >= ysize)) continue;

    if (periodicZ  && (z >= nfftz)) z -= nfftz;
    if (!periodicZ && (z < 0 || z >= zsize)) continue;
      
    // Get position on the grid
    int ind = x + xdim*(y + ysize*(z));
    
    // Here we unroll the 6x6x6 loop with 216 threads.
    // NOTE: We use 7*32=224 threads to do this
    // Calculate interpolated charge value and store it to global memory

    if (tid < order*order*order)
      write_grid<AT>(sh_thetax[i*order+x0]*sh_thetay[i*order+y0]*sh_thetaz[i*order+z0], ind, data);
  }

}


//-----------------------------------------------------------------------------------------
// Generic version can not be used
template <typename T> __forceinline__ __device__
void gather_force_store(const float fx, const float fy, const float fz,
      const int stride, const int pos,
      T* force) {
}

// Template specialization for "float"
template <> __forceinline__ __device__
void gather_force_store<float>(const float fx, const float fy, const float fz, 
             const int stride, const int pos, 
             float* force) {
  // Store into non-strided float XYZ array
  force[pos]          = fx;
  force[pos+stride]   = fy;
  force[pos+stride*2] = fz;
}

// Template specialization for "float3"
template <> __forceinline__ __device__
void gather_force_store<float3>(const float fx, const float fy, const float fz, 
        const int stride, const int pos, 
        float3* force) {
  // Store into non-strided "float3" array
#ifdef NAMD_HIP
  // Workaround: unlike CUDA, HIP-hcc has sizeof(float3) != sizeof(CudaForce) (and == sizeof(float4))
  // TODO-HIP: Remove when https://github.com/ROCm-Developer-Tools/HIP/issues/706 is fixed
  reinterpret_cast<float*>(force)[pos * 3 + 0] = fx;
  reinterpret_cast<float*>(force)[pos * 3 + 1] = fy;
  reinterpret_cast<float*>(force)[pos * 3 + 2] = fz;
#else
  force[pos].x = fx;
  force[pos].y = fy;
  force[pos].z = fz;
#endif
}
//-----------------------------------------------------------------------------------------

// Per atom data structure for the gather_force -kernels
template <typename T, int order>
struct gather_t {
  int ix;
  int iy;
  int iz;
  T charge;
  T thetax[order];
  T thetay[order];
  T thetaz[order];
  T dthetax[order];
  T dthetay[order];
  T dthetaz[order];
  float f1;
  float f2;
  float f3;
};

//
// Gathers forces from the grid
// blockDim.x            = Number of atoms each block loads
// blockDim.x*blockDim.y = Total number of threads per block
//
template <typename CT, typename FT, int order, bool periodicY, bool periodicZ>
__global__ void gather_force(const float4 *xyzq, const int ncoord,
              const int nfftx, const int nffty, const int nfftz,
              const int xsize, const int ysize, const int zsize,
              const int xdim, const int y00, const int z00,
              // const float recip1, const float recip2, const float recip3,
              const float* data,      // NOTE: data is used for loads when __CUDA_ARCH__ >= 350
#ifdef NAMD_CUDA
              const cudaTextureObject_t gridTexObj,
#endif
              const int stride,
              FT *force) {

  const int tid = threadIdx.x + threadIdx.y*blockDim.x; // 0...63

  // Shared memory
  __shared__ gather_t<CT, order> shmem[32];

  const int pos = blockIdx.x*blockDim.x + threadIdx.x;
  const int pos_end = min((blockIdx.x+1)*blockDim.x, ncoord);

  // Load atom data into shared memory
  if (pos < pos_end && threadIdx.y == 0) {

    float4 xyzqi = xyzq[pos];
    float x = xyzqi.x;
    float y = xyzqi.y;
    float z = xyzqi.z;
    float q = xyzqi.w;

    float frx = ((float)nfftx)*x;
    float fry = ((float)nffty)*y;
    float frz = ((float)nfftz)*z;

    int frxi = (int)frx;
    int fryi = (int)fry;
    int frzi = (int)frz;

    float wx = frx - (float)frxi;
    float wy = fry - (float)fryi;
    float wz = frz - (float)frzi;

    if (!periodicY && y00 == 0 && fryi >= ysize) fryi -= nffty;
    if (!periodicZ && z00 == 0 && frzi >= zsize) frzi -= nfftz;

    shmem[threadIdx.x].ix = frxi;
    shmem[threadIdx.x].iy = fryi - y00;
    shmem[threadIdx.x].iz = frzi - z00;
    shmem[threadIdx.x].charge = q;

    float3 theta_tmp[order];
    float3 dtheta_tmp[order];
    calc_theta_dtheta<float, float3, order>(wx, wy, wz, theta_tmp, dtheta_tmp);
    
#pragma unroll
    for (int i=0;i < order;i++) shmem[threadIdx.x].thetax[i] = theta_tmp[i].x;

#pragma unroll
    for (int i=0;i < order;i++) shmem[threadIdx.x].thetay[i] = theta_tmp[i].y;

#pragma unroll
    for (int i=0;i < order;i++) shmem[threadIdx.x].thetaz[i] = theta_tmp[i].z;

#pragma unroll
    for (int i=0;i < order;i++) shmem[threadIdx.x].dthetax[i] = dtheta_tmp[i].x;

#pragma unroll
    for (int i=0;i < order;i++) shmem[threadIdx.x].dthetay[i] = dtheta_tmp[i].y;

#pragma unroll
    for (int i=0;i < order;i++) shmem[threadIdx.x].dthetaz[i] = dtheta_tmp[i].z;

  }
  BLOCK_SYNC;

  // We divide the order x order x order cube into 8 sub-cubes.
  // These sub-cubes are taken care by a single thread
  // The size of the sub-cubes is:
  // order=4 : 2x2x2
  // order=6 : 3x3x3
  // order=8 : 4x4x4
  const int nsc = (order == 4) ? 2 : ((order == 6) ? 3 : 4);
  // Calculate the starting index on the sub-cube for this thread
  // tid = 0...63
  const int t = (tid % 8);         // sub-cube index (0...7)
  // t = (tx0 + ty0*2 + tz0*4)/nsc
  // (tx0, ty0, tz0) gives the starting index of the 3x3x3 sub-cube
  const int tz0 = (t / 4)*nsc;
  const int ty0 = ((t / 2) % 2)*nsc;
  const int tx0 = (t % 2)*nsc;

  //
  // Calculate forces for 32 atoms. We have 32*2 = 64 threads
  // Loop is iterated 4 times:
  //                         (iterations)
  // Threads 0...7   = atoms 0, 8,  16, 24
  // Threads 8...15  = atoms 1, 9,  17, 25
  // Threads 16...31 = atoms 2, 10, 18, 26
  //                ...
  // Threads 56...63 = atoms 7, 15, 23, 31
  //

  int base = tid/8;
  const int base_end = pos_end - blockIdx.x*blockDim.x;

  // Make sure to mask out any threads that are not running the loop.
  // This will happen if the number of atoms is not a multiple of 32.
#if defined(NAMD_HIP)
  WarpMask warp_mask = NAMD_WARP_BALLOT(WARP_FULL_MASK, (base < base_end) );
#else
  WarpMask warp_mask = WARP_BALLOT(WARP_FULL_MASK, (base < base_end) );
#endif

  while (base < base_end) {

    float f1 = 0.0f;
    float f2 = 0.0f;
    float f3 = 0.0f;
    int ix0 = shmem[base].ix;
    int iy0 = shmem[base].iy;
    int iz0 = shmem[base].iz;

    // Each thread calculates a nsc x nsc x nsc sub-cube
#pragma unroll
    for (int i=0;i < nsc*nsc*nsc;i++) {
      int tz = tz0 + (i/(nsc*nsc));
      int ty = ty0 + ((i/nsc) % nsc);
      int tx = tx0 + (i % nsc);

      int ix = ix0 + tx;
      int iy = iy0 + ty;
      int iz = iz0 + tz;
      if (ix >= nfftx) ix -= nfftx;

      if (periodicY  && (iy >= nffty)) iy -= nffty;
      if (!periodicY && (iy < 0 || iy >= ysize)) continue;

      if (periodicZ  && (iz >= nfftz)) iz -= nfftz;
      if (!periodicZ && (iz < 0 || iz >= zsize)) continue;

      int ind = ix + (iy + iz*ysize)*xdim;

#if __CUDA_ARCH__ >= 350 || defined(NAMD_HIP)
      float q0 = __ldg(&data[ind]);
#else
      float q0 = tex1Dfetch<float>(gridTexObj, ind);
#endif
      float thx0 = shmem[base].thetax[tx];
      float thy0 = shmem[base].thetay[ty];
      float thz0 = shmem[base].thetaz[tz];
      float dthx0 = shmem[base].dthetax[tx];
      float dthy0 = shmem[base].dthetay[ty];
      float dthz0 = shmem[base].dthetaz[tz];
      f1 += dthx0 * thy0 * thz0 * q0;
      f2 += thx0 * dthy0 * thz0 * q0;
      f3 += thx0 * thy0 * dthz0 * q0;
    }

    //-------------------------

    // Reduce
    const int i = threadIdx.x & 7;

    f1 += WARP_SHUFFLE(warp_mask, f1, i+4, 8);
    f2 += WARP_SHUFFLE(warp_mask, f2, i+4, 8);
    f3 += WARP_SHUFFLE(warp_mask, f3, i+4, 8);

    f1 += WARP_SHUFFLE(warp_mask, f1, i+2, 8);
    f2 += WARP_SHUFFLE(warp_mask, f2, i+2, 8);
    f3 += WARP_SHUFFLE(warp_mask, f3, i+2, 8);

    f1 += WARP_SHUFFLE(warp_mask, f1, i+1, 8);
    f2 += WARP_SHUFFLE(warp_mask, f2, i+1, 8);
    f3 += WARP_SHUFFLE(warp_mask, f3, i+1, 8);

    if (i == 0) {
      shmem[base].f1 = f1;
      shmem[base].f2 = f2;
      shmem[base].f3 = f3;
    }

    base += 8;
#if defined(NAMD_HIP)
    warp_mask = NAMD_WARP_BALLOT(warp_mask, (base < base_end) );
#else
    warp_mask = WARP_BALLOT(warp_mask, (base < base_end) );
#endif
  }

  // Write forces
  BLOCK_SYNC;
  if (pos < pos_end && threadIdx.y == 0) {
    float f1 = shmem[threadIdx.x].f1;
    float f2 = shmem[threadIdx.x].f2;
    float f3 = shmem[threadIdx.x].f3;
    float q = -shmem[threadIdx.x].charge; //*ccelec_float;
    // float fx = q*recip1*f1*nfftx;
    // float fy = q*recip2*f2*nffty;
    // float fz = q*recip3*f3*nfftz;
    float fx = q*f1*nfftx;
    float fy = q*f2*nffty;
    float fz = q*f3*nfftz;
    gather_force_store<FT>(fx, fy, fz, stride, pos, force);
  }

}

const int TILEDIM = 32;
const int TILEROWS = 8;

template <typename T>
__device__ __forceinline__
void transpose_xyz_yzx_device(
  const int x_in, const int y_in, const int z_in,
  const int x_out, const int y_out,
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int ysize_out, const int zsize_out,
  const T* data_in, T* data_out) {

  // Shared memory
  __shared__ T tile[TILEDIM][TILEDIM+1];

  // Read (x,y) data_in into tile (shared memory)
  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x_in < nx) && (y_in + j < ny) && (z_in < nz))
      tile[threadIdx.y + j][threadIdx.x] = data_in[x_in + (y_in + j + z_in*ysize_in)*xsize_in];

  BLOCK_SYNC;

  // Write (y,x) tile into data_out
  const int z_out = z_in;
  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x_out + j < nx) && (y_out < ny) && (z_out < nz))
      data_out[y_out + (z_out + (x_out+j)*zsize_out)*ysize_out] = tile[threadIdx.x][threadIdx.y + j];
}

//
// Transposes a 3d matrix out-of-place: data_in(x, y, z) -> data_out(y, z, x)
//
template <typename T>
__global__ void transpose_xyz_yzx_kernel(
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int ysize_out, const int zsize_out,
  const T* data_in, T* data_out) {

  int x_in = blockIdx.x * TILEDIM + threadIdx.x;
  int y_in = blockIdx.y * TILEDIM + threadIdx.y;
  int z_in = blockIdx.z           + threadIdx.z;

  int x_out = blockIdx.x * TILEDIM + threadIdx.y;
  int y_out = blockIdx.y * TILEDIM + threadIdx.x;

  transpose_xyz_yzx_device<T>(
    x_in, y_in, z_in,
    x_out, y_out,
    nx, ny, nz,
    xsize_in, ysize_in,
    ysize_out, zsize_out,
    data_in, data_out);

/*
  // Shared memory
  __shared__ T tile[TILEDIM][TILEDIM+1];

  int x = blockIdx.x * TILEDIM + threadIdx.x;
  int y = blockIdx.y * TILEDIM + threadIdx.y;
  int z = blockIdx.z           + threadIdx.z;

  // Read (x,y) data_in into tile (shared memory)
  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x < nx) && (y + j < ny) && (z < nz))
      tile[threadIdx.y + j][threadIdx.x] = data_in[x + (y + j + z*ysize_in)*xsize_in];

  BLOCK_SYNC;

  // Write (y,x) tile into data_out
  x = blockIdx.x * TILEDIM + threadIdx.y;
  y = blockIdx.y * TILEDIM + threadIdx.x;
  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x + j < nx) && (y < ny) && (z < nz))
      data_out[y + (z + (x+j)*zsize_out)*ysize_out] = tile[threadIdx.x][threadIdx.y + j];
*/
}

//
// Transposes a batch of 3d matrices out-of-place: data_in(x, y, z) -> data_out(y, z, x)
// Batch index bi is encoded in blockIdx.z, where 
// blockIdx.z = 0...nz-1 are for batch 1
// blockIdx.z = nz...2*nz-1 are for batch 2
// ...
// gridDim.z = nz*numBatches
//
template <typename T>
__global__ void batchTranspose_xyz_yzx_kernel(
  const TransposeBatch<T>* batches,
  const int ny, const int nz, 
  const int xsize_in, const int ysize_in) {

  int x_in = blockIdx.x * TILEDIM + threadIdx.x;
  int y_in = blockIdx.y * TILEDIM + threadIdx.y;
  int z_in = (blockIdx.z % nz)    + threadIdx.z;

  int x_out = blockIdx.x * TILEDIM + threadIdx.y;
  int y_out = blockIdx.y * TILEDIM + threadIdx.x;

  int bi = blockIdx.z/nz;

  TransposeBatch<T> batch = batches[bi];
  int nx        = batch.nx;
  int ysize_out = batch.ysize_out;
  int zsize_out = batch.zsize_out;
  T* data_in    = batch.data_in;
  T* data_out   = batch.data_out;

  transpose_xyz_yzx_device<T>(
    x_in, y_in, z_in,
    x_out, y_out,
    nx, ny, nz,
    xsize_in, ysize_in,
    ysize_out, zsize_out,
    data_in, data_out);

}

/*
//
// Transposes a 3d matrix out-of-place: data_in(x, y, z) -> data_out(y, z, x)
//
template <typename T>
__forceinline__ __device__
void transpose_xyz_yzx_dev(
  const int blockz,
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int xsize_out, const int ysize_out,
  const T* data_in, T* data_out) {

  // Shared memory
  __shared__ T tile[TILEDIM][TILEDIM+1];

  int x = blockIdx.x * TILEDIM + threadIdx.x;
  int y = blockIdx.y * TILEDIM + threadIdx.y;
  // int z = blockIdx.z           + threadIdx.z;
  int z = blockz               + threadIdx.z;

  // Read (x,y) data_in into tile (shared memory)
  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x < nx) && (y + j < ny) && (z < nz))
      tile[threadIdx.y + j][threadIdx.x] = data_in[x + (y + j + z*ysize_in)*xsize_in];

  BLOCK_SYNC;

  // Write (y,x) tile into data_out
  x = blockIdx.x * TILEDIM + threadIdx.y;
  y = blockIdx.y * TILEDIM + threadIdx.x;
  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x + j < nx) && (y < ny) && (z < nz))
      data_out[y + (z + (x+j)*ysize_out)*xsize_out] = tile[threadIdx.x][threadIdx.y + j];

}

//
// Transposes a 3d matrix out-of-place: data_in(x, y, z) -> data_out(y, z, x)
// (nx, ny, nz)                     = size of the transposed volume
// (xsize_in, ysize_in, zsize_in)   = size of the input data
// into nblock memory blocks
//
template <typename T>
__global__ void transpose_xyz_yzx_kernel(
  const int nblock,
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int xsize_out, const int ysize_out,
  const T* data_in, T* data_out) {

  const int iblock = blockIdx.z/nz;

  if (iblock < nblock) {
    transpose_xyz_yzx_dev(blockIdx.z % nz, nx, ny, nz,
      xsize_in, ysize_in, xsize_out, ysize_out,
      data_in, data_out);
  }

}
*/

template <typename T>
__device__ __forceinline__
void transpose_xyz_zxy_device(
  const int x_in, const int y_in, const int z_in,
  const int x_out, const int z_out,
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int zsize_out, const int xsize_out,
  const T* data_in, T* data_out) {

  // Shared memory
  __shared__ T tile[TILEDIM][TILEDIM+1];

  // Read (x,z) data_in into tile (shared memory)
  for (int k=0;k < TILEDIM;k += TILEROWS)
    if ((x_in < nx) && (y_in < ny) && (z_in + k < nz))
      tile[threadIdx.y + k][threadIdx.x] = data_in[x_in + (y_in + (z_in + k)*ysize_in)*xsize_in];

  BLOCK_SYNC;

  // Write (z,x) tile into data_out
  const int y_out = y_in;
  for (int k=0;k < TILEDIM;k += TILEROWS)
    if ((x_out + k < nx) && (y_out < ny) && (z_out < nz))
      data_out[z_out + (x_out + k + y_out*xsize_out)*zsize_out] = tile[threadIdx.x][threadIdx.y + k];
}

//
// Transposes a 3d matrix out-of-place: data_in(x, y, z) -> data_out(z, x, y)
//
template <typename T>
__global__ void transpose_xyz_zxy_kernel(
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int zsize_out, const int xsize_out,
  const T* data_in, T* data_out) {

  int x_in = blockIdx.x * TILEDIM + threadIdx.x;
  int y_in = blockIdx.z           + threadIdx.z;
  int z_in = blockIdx.y * TILEDIM + threadIdx.y;

  int x_out = blockIdx.x * TILEDIM + threadIdx.y;
  int z_out = blockIdx.y * TILEDIM + threadIdx.x;

  transpose_xyz_zxy_device<T>(
    x_in, y_in, z_in, x_out, z_out,
    nx, ny, nz,
    xsize_in, ysize_in,
    zsize_out, xsize_out,
    data_in, data_out);

}

//
// Transposes a batch of 3d matrices out-of-place: data_in(x, y, z) -> data_out(z, x, y)
// Batch index bi is encoded in blockIdx.z, where 
// blockIdx.z = 0...ny-1 are for batch 1
// blockIdx.z = ny...2*ny-1 are for batch 2
// ...
// gridDim.z = ny*numBatches
//
template <typename T>
__global__ void batchTranspose_xyz_zxy_kernel(
  const TransposeBatch<T>* batches,
  const int ny, const int nz, 
  const int xsize_in, const int ysize_in) {

  int x_in = blockIdx.x * TILEDIM + threadIdx.x;
  int y_in = (blockIdx.z % ny)    + threadIdx.z;
  int z_in = blockIdx.y * TILEDIM + threadIdx.y;

  int x_out = blockIdx.x * TILEDIM + threadIdx.y;
  int z_out = blockIdx.y * TILEDIM + threadIdx.x;

  int bi = blockIdx.z/ny;

  TransposeBatch<T> batch = batches[bi];
  int nx        = batch.nx;
  int zsize_out = batch.zsize_out;
  int xsize_out = batch.xsize_out;
  T* data_in    = batch.data_in;
  T* data_out   = batch.data_out;

  transpose_xyz_zxy_device<T>(
    x_in, y_in, z_in, x_out, z_out,
    nx, ny, nz,
    xsize_in, ysize_in,
    zsize_out, xsize_out,
    data_in, data_out);

}

// XXX should be defined in a centralized place
#undef SQRT_PI
#define SQRT_PI 1.7724538509055160273 /* mathematica 15 digits*/

template<int BLOCK_SIZE>
__global__ void selfEnergyKernel(
  double *selfEnergy,
  const float4 *atoms,
  int natoms,
  double ewaldcof
  )
{
  double qq=0; // sum q^2 over all threads
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < natoms) {
    double q = atoms[i].w;
    qq = q*q;
  }
  
  typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  qq = BlockReduce(temp_storage).Sum(qq);
  __syncthreads();
  

  if (threadIdx.x == 0) {
    // each charge already scaled by sqrt(COULOMB * scaling * dielectric_1)
    // finish scaling self energy term
    double c = -ewaldcof * (1.0 / SQRT_PI);
    atomicAdd(selfEnergy, c*qq);
  }
}

double compute_selfEnergy(
    double *d_selfEnergy,
    const float4 *d_atoms,
    int natoms,
    double ewaldcof,
    cudaStream_t stream)
{
  double selfEnergy = 0;
  const int block = 128;
  const int grid = (natoms + block - 1) / block;
  selfEnergyKernel<block><<<grid, block, 0, stream>>>(d_selfEnergy,
      d_atoms, natoms, ewaldcof);
  cudaCheck(cudaStreamSynchronize(stream));
  copy_DtoH<double>(d_selfEnergy, &selfEnergy, 1, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  return selfEnergy;
}

//#######################################################################################
//#######################################################################################
//#######################################################################################

void spread_charge(const float4 *atoms, const int numAtoms,
  const int nfftx, const int nffty, const int nfftz,
  const int xsize, const int ysize, const int zsize,
  const int xdim, const int y00, const int z00, 
  const bool periodicY, const bool periodicZ,
  float* data, const int order, cudaStream_t stream) {

  dim3 nthread, nblock;

  switch(order) {
  case 4:
    nthread.x = 32;
    nthread.y = 4;
    nthread.z = 1;
    nblock.x = (numAtoms - 1)/nthread.x + 1;
    nblock.y = 1;
    nblock.z = 1;
    if (periodicY && periodicZ)
      spread_charge_kernel<float, 4, true, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicY)
      spread_charge_kernel<float, 4, true, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicZ)
      spread_charge_kernel<float, 4, false, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    else
      spread_charge_kernel<float, 4, false, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    break;

  case 6:
    nthread.x = 32;
    nthread.y = 7;
    nthread.z = 1;
    nblock.x = (numAtoms - 1)/nthread.x + 1;
    nblock.y = 1;
    nblock.z = 1;
    if (periodicY && periodicZ)
      spread_charge_kernel<float, 6, true, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicY)
      spread_charge_kernel<float, 6, true, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicZ)
      spread_charge_kernel<float, 6, false, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    else
      spread_charge_kernel<float, 6, false, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    break;

  case 8:
    nthread.x = 32;
    nthread.y = 16;
    nthread.z = 1;
    nblock.x = (numAtoms - 1)/nthread.x + 1;
    nblock.y = 1;
    nblock.z = 1;
    if (periodicY && periodicZ)
      spread_charge_kernel<float, 8, true, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicY)
      spread_charge_kernel<float, 8, true, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicZ)
      spread_charge_kernel<float, 8, false, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    else
      spread_charge_kernel<float, 8, false, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00, data);
    break;

  default:
    char str[128];
    sprintf(str, "spread_charge, order %d not implemented",order);
    cudaNAMD_bug(str);
  }
  cudaCheck(cudaGetLastError());

}


// JM new version of spread_charge routine
void spread_charge_v2(const float4 *atoms, const int numAtoms,
  const int nfftx, const int nffty, const int nfftz,
  const float nfftx_f, const float nffty_f, const float nfftz_f,
  const int order3, 
  const int xsize, const int ysize, const int zsize,
  const int xdim, const int y00, const int z00, 
  const bool periodicY, const bool periodicZ,
  float* data, const int order, cudaStream_t stream) {

  dim3 nthread, nblock;

  switch(order) {
  case 4:
    nthread.x = 32;
    nthread.y = 4;
    nthread.z = 1;
    nblock.x = (numAtoms - 1)/nthread.x + 1;
    nblock.y = 1;
    nblock.z = 1;
    if (periodicY && periodicZ)
      spread_charge_kernel_v2<float, 4, true, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicY)
      spread_charge_kernel_v2<float, 4, true, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicZ)
      spread_charge_kernel_v2<float, 4, false, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    else
      spread_charge_kernel_v2<float, 4, false, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    break;

  case 6:
    nthread.x = WARPSIZE;
    nthread.y = order3 / WARPSIZE;
    nthread.z = 1;
    nblock.x = (numAtoms - 1)/nthread.x + 1;
    nblock.y = 1;
    nblock.z = 1;
    if (periodicY && periodicZ)
      spread_charge_kernel_v2<float, 6, true, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicY)
      spread_charge_kernel_v2<float, 6, true, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicZ)
      spread_charge_kernel_v2<float, 6, false, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    else
      spread_charge_kernel_v2<float, 6, false, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    break;

  case 8:
    nthread.x = WARPSIZE;
    nthread.y = order3 / WARPSIZE;
    nthread.z = 1;
    nblock.x = (numAtoms - 1)/nthread.x + 1;
    nblock.y = 1;
    nblock.z = 1;
    if (periodicY && periodicZ)
      spread_charge_kernel_v2<float, 8, true, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicY)
      spread_charge_kernel_v2<float, 8, true, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    else if (periodicZ)
      spread_charge_kernel_v2<float, 8, false, true> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    else
      spread_charge_kernel_v2<float, 8, false, false> <<< nblock, nthread, 0, stream >>>
        (atoms, numAtoms,
         nfftx, nffty, nfftz, nfftx_f, nffty_f, nfftz_f, xsize, ysize, zsize, xdim, y00, z00, data);
    break;

  default:
    char str[128];
    sprintf(str, "spread_charge, order %d not implemented",order);
    cudaNAMD_bug(str);
  }
  // cudaCheck(cudaGetLastError());
}


void scalar_sum(const bool orderXYZ, const int nfft1, const int nfft2, const int nfft3,
  const int size1, const int size2, const int size3, const double kappa,
  const float recip1x, const float recip1y, const float recip1z,
  const float recip2x, const float recip2y, const float recip2z,
  const float recip3x, const float recip3y, const float recip3z,
  const double volume,
  const float* prefac1, const float* prefac2, const float* prefac3,
  const int k2_00, const int k3_00,
  const bool doEnergyVirial, double* energy, double* virial, float2* data,
  cudaStream_t stream) {
#ifdef NAMD_CUDA
  int nthread = 1024;
  int nblock = 64;
#else
  int nthread = 256;
  int nblock = 256;
#endif

  int shmem_size = sizeof(float)*(nfft1 + nfft2 + nfft3);
  if (doEnergyVirial) {    
    shmem_size = max(shmem_size, (int)((nthread/WARPSIZE)*sizeof(RecipVirial_t)));
  }

  float piv_inv = (float)(1.0/(M_PI*volume));
  float fac = (float)(M_PI*M_PI/(kappa*kappa));

  if (doEnergyVirial) {
    if (orderXYZ) {
      scalar_sum_kernel<float, float2, true, true, false> <<< nblock, nthread, shmem_size, stream >>>
      (nfft1, nfft2, nfft3, size1, size2, size3,
        recip1x, recip1y, recip1z,
        recip2x, recip2y, recip2z,
        recip3x, recip3y, recip3z,
        prefac1, prefac2, prefac3,
        fac, piv_inv, k2_00, k3_00, data, energy, virial);
    } else {
      scalar_sum_kernel<float, float2, true, false, false> <<< nblock, nthread, shmem_size, stream >>>
      (nfft1, nfft2, nfft3, size1, size2, size3,
        recip1x, recip1y, recip1z,
        recip2x, recip2y, recip2z,
        recip3x, recip3y, recip3z,
        prefac1, prefac2, prefac3,
        fac, piv_inv, k2_00, k3_00, data, energy, virial);
    }
  } else {
    if (orderXYZ) {
      scalar_sum_kernel<float, float2, false, true, false> <<< nblock, nthread, shmem_size, stream >>>
      (nfft1, nfft2, nfft3, size1, size2, size3,
        recip1x, recip1y, recip1z,
        recip2x, recip2y, recip2z,
        recip3x, recip3y, recip3z,
        prefac1, prefac2, prefac3,
        fac, piv_inv, k2_00, k3_00, data, NULL, NULL);
    } else {
      scalar_sum_kernel<float, float2, false, false, false> <<< nblock, nthread, shmem_size, stream >>>
      (nfft1, nfft2, nfft3, size1, size2, size3,
        recip1x, recip1y, recip1z,
        recip2x, recip2y, recip2z,
        recip3x, recip3y, recip3z,
        prefac1, prefac2, prefac3,
        fac, piv_inv, k2_00, k3_00, data, NULL, NULL);
    }
  }
  cudaCheck(cudaGetLastError());

}

void gather_force(const float4 *atoms, const int numAtoms,
  // const float recip11, const float recip22, const float recip33,
  const int nfftx, const int nffty, const int nfftz,
  const int xsize, const int ysize, const int zsize,
  const int xdim, const int y00, const int z00, 
  const bool periodicY, const bool periodicZ,
  const float* data, const int order, float3* force, 
#ifdef NAMD_CUDA
  const cudaTextureObject_t gridTexObj,
#endif
  cudaStream_t stream) {

  dim3 nthread(32, 2, 1);
  dim3 nblock((numAtoms - 1)/nthread.x + 1, 1, 1);
  // dim3 nblock(npatch, 1, 1);

  switch(order) {
    case 4:
    if (periodicY && periodicZ)
      gather_force<float, float3, 4, true, true> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif 
        1, force);
    else if (periodicY)
      gather_force<float, float3, 4, true, false> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    else if (periodicZ)
      gather_force<float, float3, 4, false, true> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    else
      gather_force<float, float3, 4, false, false> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    break;

    case 6:
    if (periodicY && periodicZ)
      gather_force<float, float3, 6, true, true> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    else if (periodicY)
      gather_force<float, float3, 6, true, false> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    else if (periodicZ)
      gather_force<float, float3, 6, false, true> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    else
      gather_force<float, float3, 6, false, false> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    break;
 
    case 8:
    if (periodicY && periodicZ)
      gather_force<float, float3, 8, true, true> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    else if (periodicY)
      gather_force<float, float3, 8, true, false> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    else if (periodicZ)
      gather_force<float, float3, 8, false, true> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    else
      gather_force<float, float3, 8, false, false> <<< nblock, nthread, 0, stream >>>
      (atoms, numAtoms, nfftx, nffty, nfftz, xsize, ysize, zsize, xdim, y00, z00,
        // recip11, recip22, recip33,
        data,
#ifdef NAMD_CUDA
        gridTexObj,
#endif
        1, force);
    break;

    default:
    char str[128];
    sprintf(str, "gather_force, order %d not implemented",order);
    cudaNAMD_bug(str);
  }
  cudaCheck(cudaGetLastError());

}

//
// Transpose
//
void transpose_xyz_yzx(
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int ysize_out, const int zsize_out,
  const float2* data_in, float2* data_out, cudaStream_t stream) {

  dim3 numthread(TILEDIM, TILEROWS, 1);
  dim3 numblock((nx-1)/TILEDIM+1, (ny-1)/TILEDIM+1, nz);

  transpose_xyz_yzx_kernel<float2> <<< numblock, numthread, 0, stream >>>
  (nx, ny, nz, xsize_in, ysize_in,
    ysize_out, zsize_out,
    data_in, data_out);

  cudaCheck(cudaGetLastError());
}

//
// Batched transpose
//
void batchTranspose_xyz_yzx(
  const int numBatches, TransposeBatch<float2>* batches, 
  const int max_nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in, cudaStream_t stream) {

  dim3 numthread(TILEDIM, TILEROWS, 1);
  dim3 numblock((max_nx-1)/TILEDIM+1, (ny-1)/TILEDIM+1, nz*numBatches);

  batchTranspose_xyz_yzx_kernel<float2> <<< numblock, numthread, 0, stream >>>
  (batches, ny, nz, xsize_in, ysize_in);

  cudaCheck(cudaGetLastError());
}

//
// Transpose
//
void transpose_xyz_zxy(
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int zsize_out, const int xsize_out,
  const float2* data_in, float2* data_out, cudaStream_t stream) {

  dim3 numthread(TILEDIM, TILEROWS, 1);
  dim3 numblock((nx-1)/TILEDIM+1, (nz-1)/TILEDIM+1, ny);

  transpose_xyz_zxy_kernel<float2> <<< numblock, numthread, 0, stream >>>
  (nx, ny, nz, xsize_in, ysize_in,
    zsize_out, xsize_out,
    data_in, data_out);

  cudaCheck(cudaGetLastError());
}

//
// Batched transpose
//
void batchTranspose_xyz_zxy(
  const int numBatches, TransposeBatch<float2>* batches, 
  const int max_nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  cudaStream_t stream) {

  dim3 numthread(TILEDIM, TILEROWS, 1);
  dim3 numblock((max_nx-1)/TILEDIM+1, (nz-1)/TILEDIM+1, ny*numBatches);

  batchTranspose_xyz_zxy_kernel<float2> <<< numblock, numthread, 0, stream >>>
  (batches, ny, nz, xsize_in, ysize_in);

  cudaCheck(cudaGetLastError());
}

template <int block_size, bool alchDecouple>
__global__ void calcSelfEnergyFEPKernel(double* d_selfEnergy, double* d_selfEnergy_FEP, const float4* d_atoms, const int* d_partition, const int len, const double ewaldcof, const double lambda1Up, const double lambda2Up, const double lambda1Down, const double lambda2Down) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double q = 0;
  double qq1 = 0;
  double qq2 = 0;
  double scaleLambda1 = 0;
  double scaleLambda2 = 0;
  // NOTE: len is NOT the full length of d_atoms array, it's just the length of the first grid
  //       d_atoms always has at least 2 grids when the code reaches here since we are doing alchemical transformations
  if (i < len) {
    switch (d_partition[i]) {
      case 0: {
        scaleLambda1 = 1.0f;
        scaleLambda2 = 1.0f;
        q = d_atoms[i].w; // or d_atoms_grid2[i].w
        break;
      }
      case 1: {
        scaleLambda1 = lambda1Up; // lambda1 up
        scaleLambda2 = lambda2Up; // lambda2 up
        q = d_atoms[i].w;
        break;
      }
      case 2: {
        scaleLambda1 = lambda1Down; // lambda1 down
        scaleLambda2 = lambda2Down; // lambda2 down
        // the charges in the #1 grid are already scaled to 0
        // so I fetch the charges from the second grid
        q = d_atoms[i + len].w;
        break;
      }
    }
    if (alchDecouple) {
      qq1 = q * q;
      qq2 = q * q;
    } else {
      qq1 = q * q * scaleLambda1;
      qq2 = q * q * scaleLambda2;
    }
  }
  typedef cub::BlockReduce<double, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  qq1 = BlockReduce(temp_storage).Sum(qq1);
  __syncthreads();
  
  qq2 = BlockReduce(temp_storage).Sum(qq2);
  __syncthreads();
  if (threadIdx.x == 0) {
    const double c = -ewaldcof * (1.0 / SQRT_PI);
    atomicAdd(d_selfEnergy, c*qq1);
    atomicAdd(d_selfEnergy_FEP, c*qq2);
  }
}


void calcSelfEnergyFEPWrapper(double* d_selfEnergy, double* d_selfEnergy_FEP, double& h_selfEnergy, double& h_selfEnergyFEP, const float4* d_atoms, const int* d_partition, const int num_atoms, const double ewaldcof, const bool alchDecouple, const double lambda1Up, const double lambda2Up, const double lambda1Down, const double lambda2Down, cudaStream_t stream) {
  const int block_size = 128;
  const int num_blocks = int(std::ceil(double(num_atoms) / block_size));
  if (alchDecouple) {
    calcSelfEnergyFEPKernel<block_size, true><<<num_blocks, block_size, 0, stream>>>(d_selfEnergy, d_selfEnergy_FEP, d_atoms, d_partition, num_atoms, ewaldcof, lambda1Up, lambda2Up, lambda1Down, lambda2Down);
  } else {
    calcSelfEnergyFEPKernel<block_size, false><<<num_blocks, block_size, 0, stream>>>(d_selfEnergy, d_selfEnergy_FEP, d_atoms, d_partition, num_atoms, ewaldcof, lambda1Up, lambda2Up, lambda1Down, lambda2Down);
  }
  copy_DtoH<double>(d_selfEnergy, &h_selfEnergy, 1, stream);
  copy_DtoH<double>(d_selfEnergy_FEP, &h_selfEnergyFEP, 1, stream);
  cudaCheck(cudaStreamSynchronize(stream));
}

template <int block_size, bool alchDecouple>
__global__ void calcSelfEnergyTIKernel(double* d_selfEnergy, double* d_selfEnergy_TI_1, double* d_selfEnergy_TI_2, const float4* d_atoms, const int* d_partition, const int len, const double ewaldcof, const double lambda1Up, const double lambda1Down) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double q = 0;
  double qq = 0.0;
  double qq1 = 0.0;
  double qq2 = 0.0;
  double factor_ti_1 = 0.0;
  double factor_ti_2 = 0.0;
  double elecLambda1 = 0.0;
  if (i < len) {
    switch (d_partition[i]) {
      case 0: {
        elecLambda1 = 1.0;
        factor_ti_1 = 0.0;
        factor_ti_2 = 0.0;
        q = d_atoms[i].w; // or d_atoms_grid2[i].w
        break;
      }
      case 1: {
        elecLambda1 = lambda1Up;
        factor_ti_1 = 1.0;
        factor_ti_2 = 0.0;
        q = d_atoms[i].w;
        break;
      }
      case 2: {
        elecLambda1 = lambda1Down;
        factor_ti_1 = 0.0;
        factor_ti_2 = 1.0;
        q = d_atoms[i + len].w;
        break;
      }
    }
    if (alchDecouple) {
      qq = q * q;
    } else {
      qq = q * q * elecLambda1;
      qq1 = q * q * factor_ti_1;
      qq2 = q * q * factor_ti_2;
    }
  }
  typedef cub::BlockReduce<double, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  qq = BlockReduce(temp_storage).Sum(qq);
  __syncthreads();
  qq1 = BlockReduce(temp_storage).Sum(qq1);
  __syncthreads();
  qq2 = BlockReduce(temp_storage).Sum(qq2);
  __syncthreads();
  if (threadIdx.x == 0) {
    const double c = -ewaldcof * (1.0 / SQRT_PI);
    atomicAdd(d_selfEnergy, c*qq);
    atomicAdd(d_selfEnergy_TI_1, c*qq1);
    atomicAdd(d_selfEnergy_TI_2, c*qq2);
  }
  __syncthreads();
}

void calcSelfEnergyTIWrapper(double* d_selfEnergy, double* d_selfEnergy_TI_1, double* d_selfEnergy_TI_2, double& h_selfEnergy, double& h_selfEnergy_TI_1, double& h_selfEnergy_TI_2, const float4* d_atoms, const int* d_partition, const int num_atoms, const double ewaldcof, const bool alchDecouple, const double lambda1Up, const double lambda1Down, cudaStream_t stream) {
  const int block_size = 128;
  const int num_blocks = int(std::ceil(double(num_atoms) / block_size));
  if (alchDecouple) {
    calcSelfEnergyTIKernel<block_size, true><<<num_blocks, block_size, 0, stream>>>(d_selfEnergy, d_selfEnergy_TI_1, d_selfEnergy_TI_2, d_atoms, d_partition, num_atoms, ewaldcof, lambda1Up, lambda1Down);
  } else {
    calcSelfEnergyTIKernel<block_size, false><<<num_blocks, block_size, 0, stream>>>(d_selfEnergy, d_selfEnergy_TI_1, d_selfEnergy_TI_2, d_atoms, d_partition, num_atoms, ewaldcof, lambda1Up, lambda1Down);
  }
  copy_DtoH<double>(d_selfEnergy, &h_selfEnergy, 1, stream);
  copy_DtoH<double>(d_selfEnergy_TI_1, &h_selfEnergy_TI_1, 1, stream);
  copy_DtoH<double>(d_selfEnergy_TI_2, &h_selfEnergy_TI_2, 1, stream);
  cudaCheck(cudaStreamSynchronize(stream));
}

template <int NUM_ARRAYS>
__global__ void scaleAndMergeForceKernel(float3* forces, const float* factors, const int num_atoms) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_atoms) return;
  switch (NUM_ARRAYS) {
    case 2: {
      forces[i].x = forces[i].x * factors[0] + forces[i + num_atoms].x * factors[1];
      forces[i].y = forces[i].y * factors[0] + forces[i + num_atoms].y * factors[1];
      forces[i].z = forces[i].z * factors[0] + forces[i + num_atoms].z * factors[1];
      break;
    }
    case 3: {
      forces[i].x = forces[i].x * factors[0] + forces[i + num_atoms].x * factors[1] + forces[i + 2 * num_atoms].x * factors[2];
      forces[i].y = forces[i].y * factors[0] + forces[i + num_atoms].y * factors[1] + forces[i + 2 * num_atoms].y * factors[2];
      forces[i].z = forces[i].z * factors[0] + forces[i + num_atoms].z * factors[1] + forces[i + 2 * num_atoms].z * factors[2];
      break;
    }
    case 4: {
      forces[i].x = forces[i].x * factors[0] + forces[i + num_atoms].x * factors[1] + forces[i + 2 * num_atoms].x * factors[2] + forces[i + 3 * num_atoms].x * factors[3];
      forces[i].y = forces[i].y * factors[0] + forces[i + num_atoms].y * factors[1] + forces[i + 2 * num_atoms].y * factors[2] + forces[i + 3 * num_atoms].y * factors[3];
      forces[i].z = forces[i].z * factors[0] + forces[i + num_atoms].z * factors[1] + forces[i + 2 * num_atoms].z * factors[2] + forces[i + 3 * num_atoms].z * factors[3];
      break;
    }
    case 5: {
      forces[i].x = forces[i].x * factors[0] + forces[i + num_atoms].x * factors[1] + forces[i + 2 * num_atoms].x * factors[2] + forces[i + 3 * num_atoms].x * factors[3] + forces[i + 4 * num_atoms].x * factors[4];
      forces[i].y = forces[i].y * factors[0] + forces[i + num_atoms].y * factors[1] + forces[i + 2 * num_atoms].y * factors[2] + forces[i + 3 * num_atoms].y * factors[3] + forces[i + 4 * num_atoms].y * factors[4];
      forces[i].z = forces[i].z * factors[0] + forces[i + num_atoms].z * factors[1] + forces[i + 2 * num_atoms].z * factors[2] + forces[i + 3 * num_atoms].z * factors[3] + forces[i + 4 * num_atoms].z * factors[4];
      break;
    }
  }
}

void scaleAndMergeForceWrapper(float3* forces, const float* factors, const size_t num_arrays, const int num_atoms, cudaStream_t stream) {
  const int block_size = 128;
  const int num_blocks = int(std::ceil(double(num_atoms) / block_size));
  switch (num_arrays) {
    case 2: scaleAndMergeForceKernel<2><<<num_blocks, block_size, 0, stream>>>(forces, factors, num_atoms); break;
    case 3: scaleAndMergeForceKernel<3><<<num_blocks, block_size, 0, stream>>>(forces, factors, num_atoms); break;
    case 4: scaleAndMergeForceKernel<4><<<num_blocks, block_size, 0, stream>>>(forces, factors, num_atoms); break;
    case 5: scaleAndMergeForceKernel<5><<<num_blocks, block_size, 0, stream>>>(forces, factors, num_atoms); break;
  }
  cudaCheck(cudaGetLastError());
}

#endif // NAMD_CUDA

