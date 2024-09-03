#ifndef MSHAKE_H
#define MSHAKE_H
/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

//#include "InfoStream.h"
#include <stdio.h>
#include <stdlib.h>
#include "CudaUtils.h"
#include "common.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#ifdef NAMD_CUDA
#include <cuda.h>
#endif
#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif

typedef float   Real;
#ifdef SHORTREALS
typedef float   BigReal;
#else
typedef double  BigReal;
#endif

struct SettleParameters{
  double mO;
  double mH;
  double mOrmT;
  double mHrmT;
  double rra;
  double ra;
  double rb;
  double rc;
  // Haochuan: The following two variables are for four-site water models (TIP4 and SWM4)
  double r_om;
  double r_ohc;
};

// methods for rigidBonds
// we need this for the kernel

struct CudaRattleParam {
  int ia;
  int ib;
  BigReal dsq;
  BigReal rma;
  BigReal rmb;

};

struct CudaRattleElem {
  int ig;
  int icnt;
  CudaRattleParam params[4];

  __device__ bool operator==(const CudaRattleElem& other) const{
    return (this->ig == other.ig);
  }
};

//Predicates for thrust/CUB
class isWater{
public:
  const float *rigidBondLength;
  const int *hydrogenGroupSize;
  const int    *atomFixed;
  //Bunch of global memory accesses here, this sucks
  isWater(const float *rigidBondLength, const int *hydrogenGroupSize, 
      const int *atomFixed){
    this->rigidBondLength   = rigidBondLength;
    this->hydrogenGroupSize = hydrogenGroupSize;
    this->atomFixed = atomFixed;
  }
  __forceinline__ __device__
  bool operator()(const int x){
    if (rigidBondLength[x] > 0){
      int hgs = hydrogenGroupSize[x];
      for(int i = 0; i <hgs; i++){
        if(atomFixed[i]) return false;
      }
      return true;
    } 
    else return false;
  }
};

struct notZero{
  __host__ __device__ __forceinline__
  bool operator()(const int &i) const{
    return (i != 0);
  }
};
struct isEmpty{
  __device__ 
  bool operator()(CudaRattleElem elem){
    return (elem.icnt == 0);
  }
};

struct validRattle{
  __forceinline__ __host__ __device__ 
  bool operator()(const int i){
     return (i != -1);
  }
};

void Settle(
  const bool doEnergy,
  int numAtoms,
  const double dt,
  const double invdt,
  const int nSettles,
  const double *  vel_x,
  const double *  vel_y,
  const double *  vel_z,
  const double *  pos_x,
  const double *  pos_y,
  const double *  pos_z,
  double *  velNew_x,
  double *  velNew_y,
  double *  velNew_z,
  double *  posNew_x,
  double *  posNew_y,
  double *  posNew_z,
  double *  f_normal_x,
  double *  f_normal_y,
  double *  f_normal_z,
  cudaTensor*   virial,
  const float*  mass,
  const int   *  hydrogenGroupSize,
  const float  *  rigidBondLength,
  const int   *  atomFixed,
  int *  settleList,
  const SettleParameters *  sp,
  const WaterModel water_model,
  cudaStream_t stream);

__global__ void Settle_fp32( int numAtoms, float dt, float invdt, int nSettles, 
  const double * __restrict vel_x, const double * __restrict vel_y,  
  const double * __restrict vel_z, 
  const double * __restrict pos_x, const double * __restrict pos_y,  
  const double * __restrict pos_z, 
  double * __restrict velNew_x, double * __restrict velNew_y, 
  double * __restrict velNew_z, 
  double * __restrict posNew_x, double * __restrict posNew_y, 
  double * __restrict posNew_z, 
  const int   * __restrict hydrogenGroupSize, const float  * __restrict rigidBondLength,
  const int   * __restrict atomFixed, 
  int * __restrict settleList,
  const SettleParameters * __restrict sp);

  __global__ void rattlePair(int nRattlePairs,
  const double * __restrict vel_x,
  const double * __restrict vel_y,
  const double * __restrict vel_z,
  const double * __restrict pos_x,
  const double * __restrict pos_y,
  const double * __restrict pos_z,
  double * __restrict velNew_x, 
  double * __restrict velNew_y,
  double * __restrict velNew_z,
  double * __restrict posNew_x,
  double * __restrict posNew_y,
  double * __restrict posNew_z,
  const int   * __restrict hydrogenGroupSize,
  const float * __restrict rigidBondLength,
  const int   * __restrict atomFixed,
  int* consFailure);

__global__
void CheckConstraints(int* consFailure, int size);

void MSHAKE_CUDA(
  const bool doEnergy, 
  const CudaRattleElem *rattleList, 
  const int size, 
  const int *hydrogenGroupSize,
  const double *refx, 
  const double *refy, 
  const double *refz,
  double *posx, 
  double *posy, 
  double *posz,
  double *velx, 
  double *vely, 
  double *velz, 
  double *f_normal_x, 
  double *f_normal_y, 
  double *f_normal_z,
  cudaTensor* rigidVirial, 
  const float* mass, 
  const double invdt, 
  const BigReal tol2, 
  const int maxiter,
  int *consFailure_d, 
  int* consFailure, 
  cudaStream_t stream);

void CallRattle1Kernel(
  int numAtoms,
  // Settle Parameters
  const double dt,
  const double invdt,
  const int nSettles,
  double *  vel_x,
  double *  vel_y,
  double *  vel_z,
  const double *  pos_x,
  const double *  pos_y,
  const double *  pos_z,
  double *  f_normal_x,
  double *  f_normal_y,
  double *  f_normal_z,
  const float*  mass,
  const int   *  hydrogenGroupSize,
  const float *  rigidBondLength,
  const int   *  atomFixed,
  int *  settleList,
  const SettleParameters * sp,
  const CudaRattleElem *rattleList,
  const int nShakes,
  const BigReal tol2_d,
  const int maxiter_d,
  int* consFailure,
  const int nSettleBlocks,
  const int nShakeBlocks,
  const WaterModel water_model,
  cudaStream_t stream
);

#endif
#endif
