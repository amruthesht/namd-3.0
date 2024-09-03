/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/
 
#include "common.h"
#ifdef NAMD_CUDA
#include <cuda.h>
#endif  // NAMD_CUDA

#ifdef NAMD_HIP 
#include <hip/hip_runtime.h>
#endif  // NAMD_HIP

#include <stdio.h>
#include <stdlib.h>

#include "MShakeKernel.h"
#include "HipDefines.h"

typedef float   Real;
#ifdef SHORTREALS
typedef float   BigReal;
#else
typedef double  BigReal;
#endif

//__constant__ SettleParameters constSP;


__global__ void rattlePair(
  int nRattlePairs,
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
  const float  * __restrict rigidBondLength,
  const int   * __restrict atomFixed,
  CudaRattleElem    * __restrict rattlePairList,
  int* consFailure){
    
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tid; i < nRattlePairs; i+= stride){
        consFailure[i] = 0;
        int ig = rattlePairList[i].ig;
        CudaRattleParam param = rattlePairList[i].params[0];
        int a = ig + param.ia;
        int b = ig + param.ib;
        BigReal pabx = posNew_x[a] - posNew_x[b];
        BigReal paby = posNew_y[a] - posNew_y[b];
        BigReal pabz = posNew_z[a] - posNew_z[b];
        BigReal pabsq = pabx*pabx + paby*paby + pabz*pabz;
        BigReal rabsq = param.dsq;
        BigReal diffsq = rabsq - pabsq;
        BigReal rabx = pos_x[a] - pos_x[b];
        BigReal raby = pos_y[a] - pos_y[b];
        BigReal rabz = pos_z[a] - pos_z[b];
      
        BigReal refsq = rabx*rabx + raby*raby + rabz*rabz;
        BigReal rpab = rabx*pabx + raby*paby + rabz*pabz;
      
        BigReal rma = param.rma;
        BigReal rmb = param.rmb;
      
        BigReal gab;
        BigReal sqrtarg = rpab*rpab + refsq*diffsq;
        if ( sqrtarg < 0. ) {
          consFailure[i] = 1;
          gab = 0.;
        } else {
          consFailure[0] = 0;
          gab = (-rpab + sqrt(sqrtarg))/(refsq*(rma + rmb));
        }
        BigReal dpx = rabx * gab;
        BigReal dpy = raby * gab;
        BigReal dpz = rabz * gab;
        posNew_x[a] += rma * dpx;
        posNew_y[a] += rma * dpy;
        posNew_z[a] += rma * dpz;
        posNew_x[b] -= rmb * dpx;
        posNew_y[b] -= rmb * dpy;
        posNew_z[b] -= rmb * dpz;
    }
}

// Haochuan: this function does not support SWM4 and TIP4,
// and is not used anywhere, so I disable it.
#if 0
__global__ void settleInit(int waterIndex, const float *mass,
    const int *hydrogenGroupSize, const float *rigidBondLength, 
    SettleParameters *sp){
    
    //This initializes the water parameters

   //Water index points to oxygen, oatm poins to a hydrogen in the group
   float pmO, pmH, hhdist, ohdist;
   int oatm = waterIndex+1;
   oatm += 1*(mass[waterIndex] < 0.5 || mass[waterIndex+1] < 0.5);

   //Assigns values to everyone
   pmO = mass[waterIndex];
   pmH = mass[oatm];
   hhdist = rigidBondLength[waterIndex];
   ohdist = rigidBondLength[oatm];
   float rmT = 1.f / (pmO+pmH+pmH);
   sp->mOrmT = pmO * rmT;
   sp->mHrmT = pmH * rmT;
   BigReal t1 = 0.5f*pmO/pmH;
   sp->rc = 0.5f*hhdist;
   sp->ra = sqrt(ohdist*ohdist-sp->rc*sp->rc)/(1.0+t1);
   sp->rb = t1* sp->ra;
   sp->rra = 1.f / sp->ra;
}
#endif

__forceinline__ __device__ void settle1(
    const double ref[3][3],
    double pos[3][3],
    const SettleParameters *sp) {
    
    // swiped from Settle.C
    BigReal ref0x;
    BigReal ref0y;
    BigReal ref0z;
    BigReal ref1x;
    BigReal ref1y;
    BigReal ref1z;
    BigReal ref2x;
    BigReal ref2y;
    BigReal ref2z;

    BigReal pos0x;
    BigReal pos0y;
    BigReal pos0z;
    BigReal pos1x;
    BigReal pos1y;
    BigReal pos1z;
    BigReal pos2x;
    BigReal pos2y;
    BigReal pos2z;

    double mOrmT = sp->mOrmT;
    double mHrmT = sp->mHrmT;
    double rra = sp->rra;
    double ra = sp->ra;
    double rb = sp->rb;
    double rc = sp->rc;

    ref0x = ref[0][0];
    ref0y = ref[0][1];
    ref0z = ref[0][2];
    ref1x = ref[1][0];
    ref1y = ref[1][1];
    ref1z = ref[1][2];
    ref2x = ref[2][0];
    ref2y = ref[2][1];
    ref2z = ref[2][2];

    pos0x = pos[0][0];
    pos0y = pos[0][1];
    pos0z = pos[0][2];
    pos1x = pos[1][0];
    pos1y = pos[1][1];
    pos1z = pos[1][2];
    pos2x = pos[2][0];
    pos2y = pos[2][1];
    pos2z = pos[2][2];

    // vectors in the plane of the original positions
    BigReal b0x = ref1x - ref0x;
    BigReal b0y = ref1y - ref0y;
    BigReal b0z = ref1z - ref0z;

    BigReal c0x = ref2x - ref0x;
    BigReal c0y = ref2y - ref0y;
    BigReal c0z = ref2z - ref0z;

    // new center of mass
    BigReal d0x = pos0x*mOrmT + ((pos1x + pos2x)*mHrmT);
    BigReal d0y = pos0y*mOrmT + ((pos1y + pos2y)*mHrmT);
    BigReal d0z = pos0z*mOrmT + ((pos1z + pos2z)*mHrmT);

    BigReal a1x = pos0x - d0x;
    BigReal a1y = pos0y - d0y;
    BigReal a1z = pos0z - d0z;

    BigReal b1x = pos1x - d0x;
    BigReal b1y = pos1y - d0y;
    BigReal b1z = pos1z - d0z;

    BigReal c1x = pos2x - d0x;
    BigReal c1y = pos2y - d0y;
    BigReal c1z = pos2z - d0z;

    // Vectors describing transformation from original coordinate system to
    // the 'primed' coordinate system as in the diagram.
    // n0 = b0 x c0
    BigReal n0x = b0y*c0z-c0y*b0z;
    BigReal n0y = c0x*b0z-b0x*c0z;
    BigReal n0z = b0x*c0y-c0x*b0y;

    // n1 = a1 x n0
    BigReal n1x = a1y*n0z-n0y*a1z;
    BigReal n1y = n0x*a1z-a1x*n0z;
    BigReal n1z = a1x*n0y-n0x*a1y;
    // n2 = n0 x n1
    BigReal n2x = n0y*n1z-n1y*n0z;
    BigReal n2y = n1x*n0z-n0x*n1z;
    BigReal n2z = n0x*n1y-n1x*n0y;
    // Normalize n0
    //Change by rsqrtf later
    BigReal n0inv = 1.0/sqrt(n0x*n0x + n0y*n0y + n0z*n0z);
    n0x *= n0inv;
    n0y *= n0inv;
    n0z *= n0inv;

    BigReal n1inv = 1.0/sqrt(n1x*n1x + n1y*n1y + n1z*n1z);
    n1x *= n1inv;
    n1y *= n1inv;
    n1z *= n1inv;

    BigReal n2inv = 1.0/sqrt(n2x*n2x + n2y*n2y + n2z*n2z);
    n2x *= n2inv;
    n2y *= n2inv;
    n2z *= n2inv;

    //b0 = Vector(n1*b0, n2*b0, n0*b0); // note: b0.z is never referenced again
    BigReal n1b0 = n1x*b0x + n1y*b0y + n1z*b0z;
    BigReal n2b0 = n2x*b0x + n2y*b0y + n2z*b0z;

    //c0 = Vector(n1*c0, n2*c0, n0*c0); // note: c0.z is never referenced again
    BigReal n1c0 = n1x*c0x + n1y*c0y + n1z*c0z;
    BigReal n2c0 = n2x*c0x + n2y*c0y + n2z*c0z;

    BigReal A1Z = n0x*a1x + n0y*a1y + n0z*a1z;

    //b1 = Vector(n1*b1, n2*b1, n0*b1);
    BigReal n1b1 = n1x*b1x + n1y*b1y + n1z*b1z;
    BigReal n2b1 = n2x*b1x + n2y*b1y + n2z*b1z;
    BigReal n0b1 = n0x*b1x + n0y*b1y + n0z*b1z;

    //c1 = Vector(n1*c1, n2*c1, n0*c1);
    BigReal n1c1 = n1x*c1x + n1y*c1y + n1z*c1z;
    BigReal n2c1 = n2x*c1x + n2y*c1y + n2z*c1z;
    BigReal n0c1 = n0x*c1x + n0y*c1y + n0z*c1z;

    // now we can compute positions of canonical water
    BigReal sinphi = A1Z * rra;
    BigReal tmp = 1.0-sinphi*sinphi;
    BigReal cosphi = sqrt(tmp);
    BigReal sinpsi = (n0b1 - n0c1)/(2.0*rc*cosphi);
    tmp = 1.0-sinpsi*sinpsi;
    BigReal cospsi = sqrt(tmp);

    BigReal rbphi = -rb*cosphi;
    BigReal tmp1 = rc*sinpsi*sinphi;
    //BigReal tmp2 = rc*sinpsi*cosphi;

    //Vector a2(0, ra*cosphi, ra*sinphi);
    BigReal a2y = ra*cosphi;

    //Vector b2(-rc*cospsi, rbphi - tmp1, -rb*sinphi + tmp2);
    BigReal b2x = -rc*cospsi;
    BigReal b2y = rbphi - tmp1;

    //Vector c2( rc*cosphi, rbphi + tmp1, -rb*sinphi - tmp2);
    BigReal c2y = rbphi + tmp1;

    // there are no a0 terms because we've already subtracted the term off
    // when we first defined b0 and c0.
    BigReal alpha = b2x*(n1b0 - n1c0) + n2b0*b2y + n2c0*c2y;
    BigReal beta  = b2x*(n2c0 - n2b0) + n1b0*b2y + n1c0*c2y;
    BigReal gama  = n1b0*n2b1 - n1b1*n2b0 + n1c0*n2c1 - n1c1*n2c0;

    BigReal a2b2 = alpha*alpha + beta*beta;
    BigReal sintheta = (alpha*gama - beta*sqrt(a2b2 - gama*gama))/a2b2;
    BigReal costheta = sqrt(1.0 - sintheta*sintheta);

    //Vector a3( -a2y*sintheta,
    //            a2y*costheta,
    //            A1Z);
    BigReal a3x = -a2y*sintheta;
    BigReal a3y = a2y*costheta;
    BigReal a3z = A1Z;

    // Vector b3(b2x*costheta - b2y*sintheta,
    //             b2x*sintheta + b2y*costheta,
    //             n0b1);
    BigReal b3x = b2x*costheta - b2y*sintheta;
    BigReal b3y = b2x*sintheta + b2y*costheta;
    BigReal b3z = n0b1;

    // Vector c3(-b2x*costheta - c2y*sintheta,
    //           -b2x*sintheta + c2y*costheta,
    //             n0c1);
    BigReal c3x = -b2x*costheta - c2y*sintheta;
    BigReal c3y = -b2x*sintheta + c2y*costheta;
    BigReal c3z = n0c1;

    // undo the transformation; generate new normal vectors from the transpose.
    // Vector m1(n1.x, n2.x, n0.x);
    BigReal m1x = n1x;
    BigReal m1y = n2x;
    BigReal m1z = n0x;

    // Vector m2(n1.y, n2.y, n0.y);
    BigReal m2x = n1y;
    BigReal m2y = n2y;
    BigReal m2z = n0y;

    // Vector m0(n1.z, n2.z, n0.z);
    BigReal m0x = n1z;
    BigReal m0y = n2z;
    BigReal m0z = n0z;

    //pos[i*3+0] = Vector(a3*m1, a3*m2, a3*m0) + d0;
    pos0x = a3x*m1x + a3y*m1y + a3z*m1z + d0x;
    pos0y = a3x*m2x + a3y*m2y + a3z*m2z + d0y;
    pos0z = a3x*m0x + a3y*m0y + a3z*m0z + d0z;

    // pos[i*3+1] = Vector(b3*m1, b3*m2, b3*m0) + d0;
    pos1x = b3x*m1x + b3y*m1y + b3z*m1z + d0x;
    pos1y = b3x*m2x + b3y*m2y + b3z*m2z + d0y;
    pos1z = b3x*m0x + b3y*m0y + b3z*m0z + d0z;

    // pos[i*3+2] = Vector(c3*m1, c3*m2, c3*m0) + d0;
    pos2x = c3x*m1x + c3y*m1y + c3z*m1z + d0x;
    pos2y = c3x*m2x + c3y*m2y + c3z*m2z + d0y;
    pos2z = c3x*m0x + c3y*m0y + c3z*m0z + d0z;

    pos[0][0] = pos0x;
    pos[0][1] = pos0y;
    pos[0][2] = pos0z;
    pos[1][0] = pos1x;
    pos[1][1] = pos1y;
    pos[1][2] = pos1z;
    pos[2][0] = pos2x;
    pos[2][1] = pos2y;
    pos[2][2] = pos2z;
}

// swipe from HomePatch::tip4_omrepos
/* Reposition the om particle of a tip4p water
 * A little geometry shows that the appropriate position is given by
 * R_O + (1 / 2 r_ohc) * ( 0.5 (R_H1 + R_H2) - R_O )
 * Here r_om is the distance from the oxygen to Om site, and r_ohc
 * is the altitude from the oxygen to the hydrogen center of mass
 * Those quantities are precalculated upon initialization of HomePatch
 *
 * Ordering of TIP4P atoms: O, H1, H2, LP.
 */
__device__ void tip4_Om_reposition(
  double pos[4][3],
  const double r_om, const double r_ohc) {
  const double factor = r_om / r_ohc;
  pos[3][0] = pos[0][0] + (0.5 * (pos[1][0] + pos[2][0]) - pos[0][0]) * factor;
  pos[3][1] = pos[0][1] + (0.5 * (pos[1][1] + pos[2][1]) - pos[0][1]) * factor;
  pos[3][2] = pos[0][2] + (0.5 * (pos[1][2] + pos[2][2]) - pos[0][2]) * factor;
}

// swipe from HomePatch::swm4_omrepos
/* Reposition lonepair (Om) particle of Drude SWM4 water.
 * Same comments apply as to tip4_omrepos(), but the ordering of atoms
 * is different: O, D, LP, H1, H2.
 */
__device__ void swm4_Om_reposition(
  double pos[4][3],
  const double r_om, const double r_ohc) {
  const double factor = r_om / r_ohc;
  pos[2][0] = pos[0][0] + (0.5 * (pos[3][0] + pos[4][0]) - pos[0][0]) * factor;
  pos[2][1] = pos[0][1] + (0.5 * (pos[3][1] + pos[4][1]) - pos[0][1]) * factor;
  pos[2][2] = pos[0][2] + (0.5 * (pos[3][2] + pos[4][2]) - pos[0][2]) * factor;
}


void MSHAKEIterate(const int icnt, const CudaRattleElem* rattleList,
  const BigReal *refx, const BigReal *refy, const BigReal *refz,
  BigReal *posx, BigReal *posy, BigReal *posz,
  const BigReal tol2, const int maxiter,
  bool& done, bool& consFailure);

__device__ inline BigReal det_3by3(BigReal A[4][4])
{
    return
    A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])-
    A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])+
    A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}

__device__ void swap_row(BigReal A[4][4], BigReal b[4], int r1, int r2)
{
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        BigReal* p1 = &A[r1][i];
        BigReal* p2 = &A[r2][i];
        BigReal tmp = *p1;
        *p1 = *p2;
        *p2 = tmp;
    }
    BigReal tmp;
    tmp   = b[r1];
    b[r1] = b[r2];
    b[r2] = tmp;
}



__device__ void solve_4by4(BigReal lambda [4], BigReal A[4][4], BigReal sigma[4])
{
    BigReal tmp;
    #pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        //#ifdef PIVOT
        int piv_row = k;
        BigReal Max = A[k][k];

        for (int row = k + 1; row < 4; ++row)
        {
            if ((tmp = fabs(A[row][k])) > Max)
            {
                piv_row = row;
                Max = tmp;
            }
        }
        if(k != piv_row)
            swap_row(A, sigma, k, piv_row);
        //#endif
        for (int row = k + 1; row < 4; ++row)
        {
            tmp = A[row][k]/ A[k][k];
            for (int col = k+1; col < 4; col++)
                A[row][col] -= tmp * A[k][col];
            A[row][k]  = 0.;
            sigma[row]-= tmp * sigma[k];
        }
    }
    for (int row = 3; row >= 0; --row)
    {
        tmp = sigma[row];
        for (int j = 3; j > row; --j)
            tmp -= lambda[j] * A[row][j];
        lambda[row] = tmp / A[row][row];
    }
}


__device__ void solveMatrix(BigReal lambda [4], BigReal A[4][4], BigReal sigma[4], int icnt)
{
    switch(icnt)
    {
        case 1:
        {
            lambda[0] = sigma[0]/A[0][0];
            break;
        }
        case 2:
        {

            BigReal det=1./(A[0][0]*A[1][1]-A[0][1]*A[1][0]);
            lambda[0]  = ( A[1][1]*sigma[0]-A[0][1]*sigma[1])*det;
            lambda[1]  = (-A[1][0]*sigma[0]+A[0][0]*sigma[1])*det;
            break;
        }
        case 3:
        {
            BigReal det = 1./det_3by3(A);
            lambda[0] = det*((A[1][1]*A[2][2]-A[1][2]*A[2][1])*sigma[0]-
                             (A[0][1]*A[2][2]-A[0][2]*A[2][1])*sigma[1]+
                             (A[0][1]*A[1][2]-A[0][2]*A[1][1])*sigma[2]);

            lambda[1] = det*((A[1][2]*A[2][0]-A[1][0]*A[2][2])*sigma[0]+
                             (A[0][0]*A[2][2]-A[0][2]*A[2][0])*sigma[1]-
                             (A[0][0]*A[1][2]-A[0][2]*A[1][0])*sigma[2]);

            lambda[2] = det*((A[1][0]*A[2][1]-A[1][1]*A[2][0])*sigma[0]-
                             (A[0][0]*A[2][1]-A[0][1]*A[2][0])*sigma[1]+
                             (A[0][0]*A[1][1]-A[0][1]*A[1][0])*sigma[2]);
            break;
        }
        case 4:
        {
            solve_4by4(lambda, A, sigma);
            break;
        }
    }
}

//
// The choice of launch bounds is determined based on nvcc output of how many
// registers the kernel is using.  We see optimal performance using a register
// limit of 128, which corresponds to __launch_bounds__(512,1).
//
template <bool DOENERGY>
__global__
#ifdef NAMD_CUDA
__launch_bounds__(512,1)
#else
//TODO:HIP - tune for HIP
// __launch_bounds__(512,1)
#endif
void MSHAKE_CUDA_Kernel(
    const CudaRattleElem *rattleList, 
    const int size, 
    const int *hgs,
    const BigReal* __restrict__ refx_d, 
    const BigReal* __restrict__ refy_d, 
    const BigReal* __restrict__ refz_d,
    BigReal* __restrict__ posx_d, 
    BigReal* __restrict__ posy_d, 
    BigReal* __restrict__ posz_d,
    BigReal* __restrict__ velx_d, 
    BigReal* __restrict__ vely_d,
    BigReal* __restrict__ velz_d, 
    BigReal* __restrict__ f_normal_x, 
    BigReal* __restrict__ f_normal_y, 
    BigReal* __restrict__ f_normal_z,
    cudaTensor* __restrict virial, 
    const float* __restrict mass, 
    const BigReal invdt, 
    const BigReal tol2_d, 
    const int maxiter_d, 
    int* consFailure)
  {
// TODO: why is the CUDA (non-shared) path broken?
#if 1
      __shared__ BigReal sh_posx[128][4+1];
      __shared__ BigReal sh_posy[128][4+1];
      __shared__ BigReal sh_posz[128][4+1];
      __shared__ BigReal sh_refx[128][4+1];
      __shared__ BigReal sh_refy[128][4+1];
      __shared__ BigReal sh_refz[128][4+1];
#endif
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      // cudaTensor lVirial;
      // lVirial.xx = 0.0; lVirial.xy = 0.0; lVirial.xz = 0.0;
      // lVirial.yx = 0.0; lVirial.yy = 0.0; lVirial.yz = 0.0;
      // lVirial.zx = 0.0; lVirial.zy = 0.0; lVirial.zz = 0.0;
      if(idx < size)
      {
          consFailure[idx] = 0;
          int ig   = rattleList[idx].ig;
          int icnt = rattleList[idx].icnt;
          BigReal tol2 = tol2_d;
          BigReal sigma[4] = {0}; 
          BigReal lambda[4]= {0};
          BigReal A[4][4]  = {0};
          // BigReal df[3] = {0};
          BigReal pabx[4] = {0};
          BigReal rabx[4] = {0};
          BigReal paby[4] = {0};
          BigReal raby[4] = {0};
          BigReal pabz[4] = {0};
          BigReal rabz[4] = {0};
#if 0
          BigReal posx[4+1] = {0};
          BigReal posy[4+1] = {0};
          BigReal posz[4+1] = {0};
          BigReal refx[4+1] = {0};
          BigReal refy[4+1] = {0};
          BigReal refz[4+1] = {0};
#else
          BigReal* posx = &sh_posx[threadIdx.x][0];
          BigReal* posy = &sh_posy[threadIdx.x][0];
          BigReal* posz = &sh_posz[threadIdx.x][0];
          BigReal* refx = &sh_refx[threadIdx.x][0];
          BigReal* refy = &sh_refy[threadIdx.x][0];
          BigReal* refz = &sh_refz[threadIdx.x][0];
          for(int i = 0; i < 4+1; ++i)
          {
              posx[i] = 0;
              posy[i] = 0;
              posz[i] = 0;
              refx[i] = 0;
              refy[i] = 0;
              refz[i] = 0;
          }
#endif
          //load datas from global memory to local memory
          CudaRattleParam rattleParam[4]; //Does this work?
          for(int i = 0; i < 4+1; ++i)
          //for(int i = 0; i < hgs[i]; ++i)
          {
              //rattleParam[i] = rattleParam_d[4*idx+i];
              if (i < 4) rattleParam[i] = rattleList[idx].params[i];
              /*posx[i] = posx_d[i+4*idx];
              posy[i] = posy_d[i+4*idx];
              posz[i] = posz_d[i+4*idx];
              refx[i] = refx_d[i+4*idx]; 
              refx[i] = refx_d[i+4*idx]; 
              refx[i] = refx_d[i+4*idx]; 
              refy[i] = refy_d[i+4*idx];
              refz[i] = refz_d[i+4*idx];*/
              //I can pass the HGS and check if I is >= HGS[i]
              if(i < hgs[ig]){
                posx[i] = posx_d[i+ig];
                posy[i] = posy_d[i+ig];
                posz[i] = posz_d[i+ig];
                refx[i] = refx_d[i+ig]; 
                refy[i] = refy_d[i+ig];
                refz[i] = refz_d[i+ig];
              }
          }
          int loop = 0;
          int done = 1;
          //#pragma unroll
          for(int i = 0; i < 4; ++i)
          {
              int a = rattleParam[i].ia;
              int b = rattleParam[i].ib;
              pabx[i] = posx[a] - posx[b];
              paby[i] = posy[a] - posy[b];
              pabz[i] = posz[a] - posz[b];
              rabx[i] = refx[a] - refx[b];
              raby[i] = refy[a] - refy[b];
              rabz[i] = refz[a] - refz[b];
          }
          //#pragma unroll
          for(int i = 0; i < 4; ++i)
          {
              BigReal pabsq = pabx[i]*pabx[i] + paby[i]*paby[i] + pabz[i]*pabz[i];
              BigReal rabsq = rattleParam[i].dsq;
              BigReal diffsq = pabsq - rabsq;
              sigma[i] = diffsq;
              if ( fabs(diffsq) > (rabsq * tol2)  && i < icnt)
                  done = false;
          }
          int maxiter = maxiter_d;
          for(loop = 0; loop < maxiter; ++loop)
          {
              if(!done)
              {
                  //construct A
                  //#pragma unroll
                  for(int j = 0; j < 4; ++j)
                  {
                      BigReal rma = rattleParam[j].rma;
                      #pragma unroll
                      for(int i = 0; i < 4; ++i)
                      {
                          A[j][i] = 2.*(pabx[j]*rabx[i]+paby[j]*raby[i]+pabz[j]*rabz[i])*rma;
                      }
                      BigReal rmb = rattleParam[j].rmb;
                      A[j][j] += 2.*(pabx[j]*rabx[j]+paby[j]*raby[j]+pabz[j]*rabz[j])*rmb;
                  }
  
                  solveMatrix(lambda, A, sigma, icnt);
                  //#pragma unroll
                  for(int i = 0; i < 4; ++i)
                  {
                      int a = rattleParam[i].ia;
                      int b = rattleParam[i].ib;
                      BigReal rma = rattleParam[i].rma * lambda[i];
                      BigReal rmb = rattleParam[i].rmb * lambda[i];
  
                      posx[a] -= rma * rabx[i];
                      posy[a] -= rma * raby[i];
                      posz[a] -= rma * rabz[i];
                      posx[b] += rmb * rabx[i];
                      posy[b] += rmb * raby[i];
                      posz[b] += rmb * rabz[i];
                  }
              }
              else
                  break;
              done = 1;
              //#pragma unroll
              for(int i = 0; i < 4; ++i)
              {
                  int a = rattleParam[i].ia;
                  int b = rattleParam[i].ib;
                  pabx[i] = posx[a] - posx[b];
                  paby[i] = posy[a] - posy[b];
                  pabz[i] = posz[a] - posz[b];
                  BigReal pabsq = pabx[i]*pabx[i] + paby[i]*paby[i] + pabz[i]*pabz[i];
                  BigReal rabsq = rattleParam[i].dsq;
                  BigReal diffsq = pabsq - rabsq;
                  sigma[i] = diffsq;
                  if ( fabs(diffsq) > (rabsq * tol2) && i < icnt)
                      done = 0;
              }
          }
          if(loop == maxiter)
              consFailure[idx] = 1;
          else
          {
              #pragma unroll
              for(int i = 0; i < hgs[ig]; ++i)
              {
                posx_d[i+ig] = posx[i];
                posy_d[i+ig] = posy[i];
                posz_d[i+ig] = posz[i];    
#if 1
                // Velocity update
                // double vx  = velx_d[i + ig];
                // double vy  = vely_d[i + ig];
                // double vz  = velz_d[i + ig];
                double vnx = (posx[i] - refx[i]) * invdt;
                double vny = (posy[i] - refy[i]) * invdt;
                double vnz = (posz[i] - refz[i]) * invdt;
                // velx_d[i+ig] = (posx[i] - refx[i]) * invdt;
                // vely_d[i+ig] = (posy[i] - refy[i]) * invdt;
                // velz_d[i+ig] = (posz[i] - refz[i]) * invdt;
                // velnew update
                velx_d[i + ig] = vnx;
                vely_d[i + ig] = vny;
                velz_d[i + ig] = vnz;
#endif
              }
              consFailure[idx] = 0;
              //printf("Loop is %d %d\n", loop, consFailure[idx]);
          }
      }
  }

// XXX TODO: Fix this, it's horrible
__global__
void CheckConstraints(int* consFailure, int* consFailure_h, int size)
{
    __shared__ int result[128];
    int idx = threadIdx.x;
    result[idx] = 0;
    for(int i = idx; i < size; i += 128)
        result[idx] += consFailure[i];
    __syncthreads();
    if(idx < 64)
        result[idx] += result[idx+64];
    __syncthreads();
    if(idx < 32)
        result[idx] += result[idx+32];
    __syncthreads();
    if(idx < 16)
        result[idx] += result[idx+16];
    __syncthreads();
    if(idx < 8)
        result[idx] += result[idx+8];
    __syncthreads();
    if(idx < 4)
        result[idx] += result[idx+4];
    __syncthreads();
    if(idx < 2)
        result[idx] += result[idx+2];
    __syncthreads();
    if(idx < 1)
        result[idx] += result[idx+1];
    __syncthreads();
    if(idx == 0)
    {
        consFailure_h[0] += result[0];
        //printf("Constraints %d\n", onsFailure[0]);
    }
}

void MSHAKE_CUDA(
    const bool doEnergy, 
    const CudaRattleElem* rattleList, 
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
    const float *mass,
    const double invdt, 
    const BigReal tol2, 
    const int maxiter,
    int* consFailure_d,
    int* consFailure,
    cudaStream_t stream)
{
     if(size == 0){ 
         //fprintf(stderr, "No rattles, returning\n");
         return;
     }

    int gridDim = int(size/128)+(size%128==0 ? 0 : 1);
    int blockDim = 128;
     
    if(doEnergy){
        MSHAKE_CUDA_Kernel<true><<< gridDim, blockDim, 0, stream>>>(rattleList, size,
           hydrogenGroupSize, 
           refx, refy, refz, posx, posy, posz, velx, vely, velz, f_normal_x, 
           f_normal_y, f_normal_z, rigidVirial, mass, invdt, 
           tol2, maxiter, consFailure_d);
    }else {
        MSHAKE_CUDA_Kernel<false><<< gridDim, blockDim, 0, stream>>>(rattleList, size,
            hydrogenGroupSize, 
            refx, refy, refz, posx, posy, posz, velx, vely, velz, 
            f_normal_x, f_normal_y, f_normal_z, rigidVirial, mass, invdt, 
            tol2, maxiter, consFailure_d);
    }

    // cudaCheck(cudaGetLastError());
    // JM NOTE: We can check the consFailure flag for every migration step
    // CheckConstraints<<<1,128, 0, stream>>>(consFailure_d, consFailure, size);
    // I can't do this here
    // why do I need this variable?????
    //done = !consFailure;
}

//
// Perform a warp-wide read on 3-element clusters from global memory
// into shared memory.
// a - global memory buffer
// buf - shared memory buffer, dimension BLOCK*3
// list - index offset to beginning of each 3-element cluster
// n - total number of clusters
//
// We apply this to improve reading of water (positions and velocities).
// Data in global memory is stored in SOA form: pos_x, pos_y, pos_z, etc,
// so we call this routine six times to read positions and velocities.
// Reading is coalesced whenever water molecules are all consecutive.
// Although waters are sorted to the end of each patch, there will
// generally be gaps between patches due to solute.
//
template <int VECTOR_SIZE>
__device__ __forceinline__ void ld_vecN(const double *a, double buf[][VECTOR_SIZE], int *list, int n)
{
  int laneid = threadIdx.x % warpSize;
  int warpid = threadIdx.x / warpSize;  
#ifdef NAMD_CUDA
  __syncwarp();
#else
  //TODO:HIP verify for HIP
  //__all(1);
#endif

  for (int j = 0; j < VECTOR_SIZE; j++) {
    int offset = laneid + j*warpSize;
    int ii = offset % VECTOR_SIZE;
    int jj = warpid*warpSize + offset/VECTOR_SIZE;
    if (blockIdx.x*blockDim.x + jj < n) {
      int idx = list[jj] + ii;
      buf[jj][ii] = a[idx];
    }
  }
#ifdef NAMD_CUDA
  __syncwarp();
#else
  //TODO:HIP verify for HIP
  //__all(1);
#endif

}

// Overload for float
template <int VECTOR_SIZE>
__device__ __forceinline__ void ld_vecN(const float *a, float buf[][VECTOR_SIZE], int *list, int n)
{
  int laneid = threadIdx.x % warpSize;
  int warpid = threadIdx.x / warpSize;
#ifdef NAMD_CUDA
  __syncwarp();
#else
  //TODO:HIP verify for HIP
  //__all(1);
#endif

  for (int j = 0; j < VECTOR_SIZE; j++) {
    int offset = laneid + j*warpSize;
    int ii = offset % VECTOR_SIZE;
    int jj = warpid*warpSize + offset/VECTOR_SIZE;
    if (blockIdx.x*blockDim.x + jj < n) {
      int idx = list[jj] + ii;
      buf[jj][ii] = a[idx];
    }
  }
#ifdef NAMD_CUDA
  __syncwarp();
#else
  //TODO:HIP verify for HIP
  //__all(1);
#endif

}

//
// Same as previous, only this time we write to global memory from
// shared memory.
//
template <int VECTOR_SIZE>
__device__ __forceinline__ void st_vecN(double *a, double buf[][VECTOR_SIZE], int *list, int n)
{
  int laneid = threadIdx.x % warpSize;
  int warpid = threadIdx.x / warpSize;  
#ifdef NAMD_CUDA
  __syncwarp();
#else
  //TODO:HIP verify for HIP
  //__all(1);
#endif  
  for (int j = 0; j < VECTOR_SIZE; j++) {
    int offset = laneid + j*warpSize;
    int ii = offset % VECTOR_SIZE;
    int jj = warpid*warpSize + offset/VECTOR_SIZE;
    if (blockIdx.x*blockDim.x + jj < n) {
      int idx = list[jj] + ii;
      a[idx] = buf[jj][ii];
    }
  } 
#ifdef NAMD_CUDA
  __syncwarp();
#else
  //TODO:HIP verify for HIP
  //__all(1);
#endif
}

//
// Same as previous, but accumulates the value instead of copying
//
template <int VECTOR_SIZE>
__device__ __forceinline__ void acc_vecN(double* __restrict__ a, double buf[][VECTOR_SIZE], int *list, int n)
{
  int laneid = threadIdx.x % warpSize;
  int warpid = threadIdx.x / warpSize;  
#ifdef NAMD_CUDA
  __syncwarp();
#else
  //TODO:HIP verify for HIP
  //__all(1);
#endif
  for (int j = 0; j < VECTOR_SIZE; j++) {
    int offset = laneid + j*warpSize;
    int ii = offset % VECTOR_SIZE;
    int jj = warpid*warpSize + offset/VECTOR_SIZE;
    if (blockIdx.x*blockDim.x + jj < n) {
      int idx = list[jj] + ii;
      a[idx] += buf[jj][ii];
    }
  } 
#ifdef NAMD_CUDA
  __syncwarp();
#else
  //TODO:HIP verify for HIP
  //__all(1);
#endif
}

//
// The choice of launch bounds is determined based on nvcc output of how many
// registers the kernel is using.  We see optimal performance using a register
// limit of 128, which corresponds to __launch_bounds__(512,1).
//
template <bool DOENERGY, int BLOCK,
          int WATER_GROUP_SIZE,
          WaterModel WATER_MODEL>
__global__
#ifdef NAMD_CUDA
__launch_bounds__(512,1)
#else
//TODO:HIP tune 
// __launch_bounds__(512,1)
#endif	
void SettleKernel(
  int numAtoms,
  const double dt,
  const double invdt,
  int nSettles,
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
  double * __restrict f_normal_x, 
  double * __restrict f_normal_y,
  double * __restrict f_normal_z,
  cudaTensor* __restrict virial, 
  const float* __restrict mass,   
  const int   * __restrict hydrogenGroupSize,
  const float  * __restrict rigidBondLength,
  const int   * __restrict atomFixed,
  const int    * __restrict settleList,
  const SettleParameters * __restrict sp)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  double ref[WATER_GROUP_SIZE][3] = {0};
  double pos[WATER_GROUP_SIZE][3] = {0};
  double vel[WATER_GROUP_SIZE][3] = {0};

  __shared__ int list[BLOCK];
  __shared__ double buf[BLOCK][WATER_GROUP_SIZE];
  if (tid < nSettles) {
    list[threadIdx.x] = settleList[tid];
  }
  __syncthreads();
  // Do a warp-wide read on pos* and vel* from global memory to shmem
  //BEGIN OF FIRST PART OF RATTLE1: Settle_SIMD invocation
  ld_vecN<WATER_GROUP_SIZE>(pos_x, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) ref[i][0] = buf[threadIdx.x][i];
  ld_vecN<WATER_GROUP_SIZE>(pos_y, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) ref[i][1] = buf[threadIdx.x][i];
  ld_vecN<WATER_GROUP_SIZE>(pos_z, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) ref[i][2] = buf[threadIdx.x][i];
  ld_vecN<WATER_GROUP_SIZE>(vel_x, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) vel[i][0] = buf[threadIdx.x][i];
  ld_vecN<WATER_GROUP_SIZE>(vel_y, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) vel[i][1] = buf[threadIdx.x][i];
  ld_vecN<WATER_GROUP_SIZE>(vel_z, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) vel[i][2] = buf[threadIdx.x][i];

  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) {
    pos[i][0] = ref[i][0] + vel[i][0]*dt;
    pos[i][1] = ref[i][1] + vel[i][1]*dt;
    pos[i][2] = ref[i][2] + vel[i][2]*dt;
  }

  switch (WATER_MODEL) {
    case WaterModel::SWM4: {
      // SWM4 ordering:  O D LP H1 H2
      // do swap(O,LP) and call settle with subarray O H1 H2
      double lp_ref[3];
      double lp_pos[3];
      #pragma unroll
      for (int i = 0; i < 3; ++i) {
        lp_ref[i] = ref[2][i];
        lp_pos[i] = pos[2][i];
      }
      #pragma unroll
      for (int i = 0; i < 3; ++i) {
        ref[2][i] = ref[0][i];
        pos[2][i] = pos[0][i];
      }
      settle1(ref+2, pos+2, sp);
      // swap back after we return
      #pragma unroll
      for (int i = 0; i < 3; ++i) {
        ref[0][i] = ref[2][i];
        pos[0][i] = pos[2][i];
      }
      #pragma unroll
      for (int i = 0; i < 3; ++i) {
        ref[2][i] = lp_ref[i];
        pos[2][i] = lp_pos[i];
      }
      swm4_Om_reposition(pos, sp->r_om, sp->r_ohc);
      break;
    }
    case WaterModel::TIP4: {
      settle1(ref, pos, sp);
      tip4_Om_reposition(pos, sp->r_om, sp->r_ohc);
      break;
    }
    default: {
      settle1(ref, pos, sp);
    }
  }

  // Update position and velocities with stuff calculated by settle1
#if 1
  // Do a warp-wide write on pos* and vel* to global memory from shmem
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = (pos[i][0] - ref[i][0])*invdt;
  st_vecN<WATER_GROUP_SIZE>(velNew_x, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = (pos[i][1] - ref[i][1])*invdt;
  st_vecN<WATER_GROUP_SIZE>(velNew_y, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = (pos[i][2] - ref[i][2])*invdt;
  st_vecN<WATER_GROUP_SIZE>(velNew_z, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = pos[i][0];
  st_vecN<WATER_GROUP_SIZE>(posNew_x, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = pos[i][1];
  st_vecN<WATER_GROUP_SIZE>(posNew_y, buf, list, nSettles);
  #pragma unroll
  for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = pos[i][2];
  st_vecN<WATER_GROUP_SIZE>(posNew_z, buf, list, nSettles);
#else
    // force+velocity update
  if (tid < nSettles) {
    double vnx, vny, vnz;
#pragma unroll
    for(int i = 0 ; i < 3; i++){
        vnx = (pos[i][0] - ref[i][0]) * invdt;
        vny = (pos[i][1] - ref[i][1]) * invdt;
        vnz = (pos[i][2] - ref[i][2]) * invdt;

        df[0] = (vnx - vel[i][0]) * mass[i] * invdt;
        df[1] = (vny - vel[i][1]) * mass[i] * invdt;
        df[2] = (vnz - vel[i][2]) * mass[i] * invdt;

        velNew_x[ig+i] = vnx;
        velNew_y[ig+i] = vny;
        velNew_z[ig+i] = vnx;
        posNew_x[ig+i] = pos[i][0];
        posNew_y[ig+i] = pos[i][1];
        posNew_z[ig+i] = pos[i][2];
        f_normal_x[ig+i] += df[0];
        f_normal_y[ig+i] += df[1];
        f_normal_z[ig+i] += df[2];

        if (DOENERGY){
            lVirial.xx +=  df[0] * pos[i][0];
            // lVirial.xy +=  df[0] * pos[i][1];
            // lVirial.xz +=  df[0] * pos[i][2];
            lVirial.yx +=  df[1] * pos[i][0];
            lVirial.yy +=  df[1] * pos[i][1];
            // lVirial.yz +=  df[1] * pos[i][2];
            lVirial.zx +=  df[2] * pos[i][0];
            lVirial.zy +=  df[2] * pos[i][1];
            lVirial.zz +=  df[2] * pos[i][2];
        }
    }
  }
    if(DOENERGY){
        typedef cub::BlockReduce<BigReal, 128> BlockReduce;
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
          atomicAdd(&(virial->xx), lVirial.xx);
          atomicAdd(&(virial->xy), lVirial.yx);
          atomicAdd(&(virial->xz), lVirial.zx);
          atomicAdd(&(virial->yx), lVirial.yx);
          atomicAdd(&(virial->yy), lVirial.yy);
          atomicAdd(&(virial->yz), lVirial.zy);
          atomicAdd(&(virial->zx), lVirial.zx);
          atomicAdd(&(virial->zy), lVirial.zy);
          atomicAdd(&(virial->zz), lVirial.zz);
        }
      }   
#endif
  // }
}

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
    const float *  rigidBondLength,
    const int   *  atomFixed, 
    int *  settleList,
    const SettleParameters * sp,
    const WaterModel water_model,
    cudaStream_t stream)
{

#ifdef NAMD_CUDA
    const int blocks = 128;
#else
    const int blocks = 128;
#endif
    int grid = (nSettles + blocks - 1) / blocks;
#define CALL(DOENERGY, WATER_MODEL) \
  SettleKernel<DOENERGY, blocks, getWaterModelGroupSize(WATER_MODEL), WATER_MODEL> \
  <<< grid, blocks, 0, stream >>> \
  (numAtoms, dt, invdt, nSettles, \
    vel_x, vel_y, vel_z, \
    pos_x, pos_y, pos_z, \
    velNew_x, velNew_y, velNew_z, \
    posNew_x, posNew_y, posNew_z, \
    f_normal_x, f_normal_y, f_normal_z, virial, mass, \
    hydrogenGroupSize, rigidBondLength, atomFixed, \
    settleList, sp \
  );
  switch (water_model) {
    case WaterModel::SWM4: {
      if ( doEnergy) CALL(true, WaterModel::SWM4);
      if (!doEnergy) CALL(false, WaterModel::SWM4);
      break;
    }
    case WaterModel::TIP4: {
      if ( doEnergy) CALL(true, WaterModel::TIP4);
      if (!doEnergy) CALL(false, WaterModel::TIP4);
      break;
    }
    default: {
      if ( doEnergy) CALL(true, WaterModel::TIP3);
      if (!doEnergy) CALL(false, WaterModel::TIP3);
    }
  }
#undef CALL
  cudaCheck(cudaStreamSynchronize(stream));
}

// Fused kernel for rigid bonds constraints
// Each block will do a single operation -> either SETTLE or MSHAKE.
template <int WATER_GROUP_SIZE,
          WaterModel WATER_MODEL>
#ifdef NAMD_CUDA
__launch_bounds__(512,1)
#else
//TODO:HIP tune
// __launch_bounds__(512,1)
#endif
__global__ void Rattle1Kernel(
  int                                   numAtoms,
  // Settle Parameters
  const double                          dt,
  const double                          invdt,
  const int                             nSettles,
  double * __restrict                   vel_x,
  double * __restrict                   vel_y,
  double * __restrict                   vel_z,
  const double * __restrict             pos_x,
  const double * __restrict             pos_y,
  const double * __restrict             pos_z,
  double * __restrict                   f_normal_x,
  double * __restrict                   f_normal_y,
  double * __restrict                   f_normal_z,
  const float * __restrict              mass,
  const int   * __restrict              hydrogenGroupSize,
  const float * __restrict              rigidBondLength,
  const int   * __restrict              atomFixed,
  int * __restrict                      settleList,
  const SettleParameters *              sp,
  const CudaRattleElem * __restrict     rattleList,
  const int                             nShakes,
  const BigReal                         tol2_d,
  const int                             maxiter_d,
  int*                                  consFailure,
  const int                             nSettleBlocks
){
  // Great, first step is fetching value
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  bool isSettle = blockIdx.x < nSettleBlocks;
  __shared__ int list[128];
  __shared__ double buf[128][WATER_GROUP_SIZE];
#ifndef NAMD_CUDA
  __shared__ BigReal sh_posx[128][4+1];
  __shared__ BigReal sh_posy[128][4+1];
  __shared__ BigReal sh_posz[128][4+1];
  __shared__ BigReal sh_refx[128][4+1];
  __shared__ BigReal sh_refy[128][4+1];
  __shared__ BigReal sh_refz[128][4+1];
#endif

  if( isSettle ){
    // SETTLE CODEPATH
    list[threadIdx.x] = settleList[tid];
    __syncthreads();
    double ref[WATER_GROUP_SIZE][3] = {0};
    double pos[WATER_GROUP_SIZE][3] = {0};
    double vel[WATER_GROUP_SIZE][3] = {0};
    float tmp_mass[WATER_GROUP_SIZE];

    ld_vecN<WATER_GROUP_SIZE>(pos_x, buf, list, nSettles);
    #pragma unroll
    for (int i = 0; i < WATER_GROUP_SIZE; ++i) ref[i][0] = buf[threadIdx.x][i];
    ld_vecN<WATER_GROUP_SIZE>(pos_y, buf, list, nSettles);
    #pragma unroll
    for (int i = 0; i < WATER_GROUP_SIZE; ++i) ref[i][1] = buf[threadIdx.x][i];
    ld_vecN<WATER_GROUP_SIZE>(pos_z, buf, list, nSettles);
    #pragma unroll
    for (int i = 0; i < WATER_GROUP_SIZE; ++i) ref[i][2] = buf[threadIdx.x][i];
    ld_vecN<WATER_GROUP_SIZE>(vel_x, buf, list, nSettles);
    #pragma unroll
    for (int i = 0; i < WATER_GROUP_SIZE; ++i) vel[i][0] = buf[threadIdx.x][i];
    ld_vecN<WATER_GROUP_SIZE>(vel_y, buf, list, nSettles);
    #pragma unroll
    for (int i = 0; i < WATER_GROUP_SIZE; ++i) vel[i][1] = buf[threadIdx.x][i];
    ld_vecN<WATER_GROUP_SIZE>(vel_z, buf, list, nSettles);
    #pragma unroll
    for (int i = 0; i < WATER_GROUP_SIZE; ++i) vel[i][2] = buf[threadIdx.x][i];
    #pragma unroll
    for (int i = 0; i < WATER_GROUP_SIZE; ++i) {
      pos[i][0] = ref[i][0] + vel[i][0]*dt;
      pos[i][1] = ref[i][1] + vel[i][1]*dt;
      pos[i][2] = ref[i][2] + vel[i][2]*dt;
    }
    switch (WATER_MODEL) {
      case WaterModel::SWM4: {
        // SWM4 ordering:  O D LP H1 H2
        // do swap(O,LP) and call settle with subarray O H1 H2
        double lp_ref[3];
        double lp_pos[3];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
          lp_ref[i] = ref[2][i];
          lp_pos[i] = pos[2][i];
        }
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
          ref[2][i] = ref[0][i];
          pos[2][i] = pos[0][i];
        }
        settle1(ref+2, pos+2, sp);
        // swap back after we return
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
          ref[0][i] = ref[2][i];
          pos[0][i] = pos[2][i];
        }
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
          ref[2][i] = lp_ref[i];
          pos[2][i] = lp_pos[i];
        }
        swm4_Om_reposition(pos, sp->r_om, sp->r_ohc);
        break;
      }
      case WaterModel::TIP4: {
        settle1(ref, pos, sp);
        tip4_Om_reposition(pos, sp->r_om, sp->r_ohc);
        break;
      }
      default: {
        settle1(ref, pos, sp);
      }
    }
    // load the mass to calculate the force
    if (WATER_MODEL != WaterModel::TIP3) {
      __shared__ float mass_buf[128][WATER_GROUP_SIZE];
      ld_vecN<WATER_GROUP_SIZE>(mass, mass_buf, list, nSettles);
      #pragma unroll
      for (int i = 0; i < WATER_GROUP_SIZE; ++i) tmp_mass[i] = mass_buf[threadIdx.x][i];
    }
    // ------ Update the atomic velocities along X
    #pragma unroll
    for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = (pos[i][0] - ref[i][0])*invdt;

    st_vecN<WATER_GROUP_SIZE>(vel_x, buf, list, nSettles);

    // buf here holds the new velocities. now we calculate the force contributions
    // maybe it's this mass?
    if (WATER_MODEL != WaterModel::TIP3) {
      #pragma unroll
      for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = (buf[threadIdx.x][i] - vel[i][0]) * (tmp_mass[i]*invdt);
    } else {
      buf[threadIdx.x][0] = (buf[threadIdx.x][0] - vel[0][0]) * (sp->mO*invdt);
      buf[threadIdx.x][1] = (buf[threadIdx.x][1] - vel[1][0]) * (sp->mH*invdt);
      buf[threadIdx.x][2] = (buf[threadIdx.x][2] - vel[2][0]) * (sp->mH*invdt);
    }
    acc_vecN<WATER_GROUP_SIZE>(f_normal_x, buf, list, nSettles);

    // ------ Update the atomic velocities along Y
    #pragma unroll
    for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = (pos[i][1] - ref[i][1])*invdt;
    st_vecN<WATER_GROUP_SIZE>(vel_y, buf, list, nSettles);
    if (WATER_MODEL != WaterModel::TIP3) {
      #pragma unroll
      for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = (buf[threadIdx.x][i] - vel[i][1]) * (tmp_mass[i]*invdt);
    } else {
      buf[threadIdx.x][0] = (buf[threadIdx.x][0] - vel[0][1]) * (sp->mO*invdt);
      buf[threadIdx.x][1] = (buf[threadIdx.x][1] - vel[1][1]) * (sp->mH*invdt);
      buf[threadIdx.x][2] = (buf[threadIdx.x][2] - vel[2][1]) * (sp->mH*invdt);
    }
    acc_vecN<WATER_GROUP_SIZE>(f_normal_y, buf, list, nSettles);


    // ------ Update the atomic velocities along Z
    #pragma unroll
    for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = (pos[i][2] - ref[i][2])*invdt;
    st_vecN<WATER_GROUP_SIZE>(vel_z, buf, list, nSettles);
    if (WATER_MODEL != WaterModel::TIP3) {
      #pragma unroll
      for (int i = 0; i < WATER_GROUP_SIZE; ++i) buf[threadIdx.x][i] = (buf[threadIdx.x][i] - vel[i][2]) * (tmp_mass[i]*invdt);
    } else {
      buf[threadIdx.x][0] = (buf[threadIdx.x][0] - vel[0][2]) * (sp->mO*invdt);
      buf[threadIdx.x][1] = (buf[threadIdx.x][1] - vel[1][2]) * (sp->mH*invdt);
      buf[threadIdx.x][2] = (buf[threadIdx.x][2] - vel[2][2]) * (sp->mH*invdt);
    }
    acc_vecN<WATER_GROUP_SIZE>(f_normal_z, buf, list, nSettles);

  }else{
    // Remaining threadblocks do MSHAKE.
    // instead of tid - nSettle, IDX needs to be the first threadBlock that does MSHAKE instead of settle
    int idx = tid - blockDim.x * nSettleBlocks; // Numbers of blocks that did SETTLE
    // Ok, now we follow the logic of MSHAKE_CUDA_KERNEL
    if(idx < nShakes){
      // This needs to increase monotonically
      // if (threadIdx.x == 0) printf("tid %d idx %d\n", tid, idx);
      // consFailure[idx] = 0;
      int ig   = rattleList[idx].ig;
      int icnt = rattleList[idx].icnt;
      BigReal tol2 = tol2_d;
      BigReal sigma[4] = {0};
      BigReal lambda[4]= {0};
      BigReal A[4][4]  = {0};
      BigReal velx[4+1] = {0};
      BigReal vely[4+1] = {0};
      BigReal velz[4+1] = {0};
      BigReal pabx[4] = {0};
      BigReal rabx[4] = {0};
      BigReal paby[4] = {0};
      BigReal raby[4] = {0};
      BigReal pabz[4] = {0};
      BigReal rabz[4] = {0};
      BigReal df[3]   = {0};
#ifdef NAMD_CUDA
      BigReal posx[4+1] = {0};
      BigReal posy[4+1] = {0};
      BigReal posz[4+1] = {0};
      BigReal refx[4+1] = {0};
      BigReal refy[4+1] = {0};
      BigReal refz[4+1] = {0};
#else
      BigReal* posx = &sh_posx[threadIdx.x][0];
      BigReal* posy = &sh_posy[threadIdx.x][0];
      BigReal* posz = &sh_posz[threadIdx.x][0];
      BigReal* refx = &sh_refx[threadIdx.x][0];
      BigReal* refy = &sh_refy[threadIdx.x][0];
      BigReal* refz = &sh_refz[threadIdx.x][0];
      for(int i = 0; i < 4+1; ++i)
      {
          posx[i] = 0;
          posy[i] = 0;
          posz[i] = 0;
          refx[i] = 0;
          refy[i] = 0;
          refz[i] = 0;
      }
#endif
      //load datas from global memory to local memory
      CudaRattleParam rattleParam[4];
#pragma unroll
      for(int i = 0; i < 4+1; ++i)
      //for(int i = 0; i < hgs[i]; ++i)
      {
          //rattleParam[i] = rattleParam_d[4*idx+i];
          if (i < 4) rattleParam[i] = rattleList[idx].params[i];
          /*posx[i] = posx_d[i+4*idx];
          posy[i] = posy_d[i+4*idx];
          posz[i] = posz_d[i+4*idx];
          refx[i] = refx_d[i+4*idx];
          refx[i] = refx_d[i+4*idx];
          refx[i] = refx_d[i+4*idx];
          refy[i] = refy_d[i+4*idx];
          refz[i] = refz_d[i+4*idx];*/
          //I can pass the HGS and check if I is >= HGS[i]
          if(i < hydrogenGroupSize[ig]){
            refx[i] = pos_x[i+ig];
            refy[i] = pos_y[i+ig];
            refz[i] = pos_z[i+ig];
            velx[i] = vel_x[i + ig];
            vely[i] = vel_y[i + ig];
            velz[i] = vel_z[i + ig];
            posx[i] = refx[i] + (velx[i] * dt);
            posy[i] = refy[i] + (vely[i] * dt);
            posz[i] = refz[i] + (velz[i] * dt);
          }
      }
      int loop = 0;
      int done = 1;
#pragma unroll
      for(int i = 0; i < 4; ++i)
      {
          int a = rattleParam[i].ia;
          int b = rattleParam[i].ib;
          pabx[i] = posx[a] - posx[b];
          paby[i] = posy[a] - posy[b];
          pabz[i] = posz[a] - posz[b];
          rabx[i] = refx[a] - refx[b];
          raby[i] = refy[a] - refy[b];
          rabz[i] = refz[a] - refz[b];
      }

#pragma unroll
      for(int i = 0; i < 4; ++i)
      {
          BigReal pabsq = pabx[i]*pabx[i] + paby[i]*paby[i] + pabz[i]*pabz[i];
          BigReal rabsq = rattleParam[i].dsq;
          BigReal diffsq = pabsq - rabsq;
          sigma[i] = diffsq;
          if ( fabs(diffsq) > (rabsq * tol2)  && i < icnt)
              done = false;
      }
      int maxiter = maxiter_d;
      for(loop = 0; loop < maxiter; ++loop)
      {
          if(!done)
          {
              //construct A
#pragma unroll
              for(int j = 0; j < 4; ++j)
              {
                  BigReal rma = rattleParam[j].rma;
                  #pragma unroll
                  for(int i = 0; i < 4; ++i)
                  {
                      A[j][i] = 2.*(pabx[j]*rabx[i]+paby[j]*raby[i]+pabz[j]*rabz[i])*rma;
                  }
                  BigReal rmb = rattleParam[j].rmb;
                  A[j][j] += 2.*(pabx[j]*rabx[j]+paby[j]*raby[j]+pabz[j]*rabz[j])*rmb;
              }

              solveMatrix(lambda, A, sigma, icnt);
#pragma unroll
              for(int i = 0; i < 4; ++i)
              {
                  int a = rattleParam[i].ia;
                  int b = rattleParam[i].ib;
                  BigReal rma = rattleParam[i].rma * lambda[i];
                  BigReal rmb = rattleParam[i].rmb * lambda[i];

                  posx[a] -= rma * rabx[i];
                  posy[a] -= rma * raby[i];
                  posz[a] -= rma * rabz[i];
                  posx[b] += rmb * rabx[i];
                  posy[b] += rmb * raby[i];
                  posz[b] += rmb * rabz[i];
              }
          }
          else
              break;
          done = 1;
#pragma unroll
          for(int i = 0; i < 4; ++i)
          {
              int a = rattleParam[i].ia;
              int b = rattleParam[i].ib;
              pabx[i] = posx[a] - posx[b];
              paby[i] = posy[a] - posy[b];
              pabz[i] = posz[a] - posz[b];
              BigReal pabsq = pabx[i]*pabx[i] + paby[i]*paby[i] + pabz[i]*pabz[i];
              BigReal rabsq = rattleParam[i].dsq;
              BigReal diffsq = pabsq - rabsq;
              sigma[i] = diffsq;
              if ( fabs(diffsq) > (rabsq * tol2) && i < icnt)
                  done = 0;
          }
      }
      if(loop == maxiter)
          consFailure[idx] = 1;
      else
      {

//           double m = mass[ig];
          for(int i = 0; i < hydrogenGroupSize[ig]; ++i)
          {
            // posNew_x[i+ig] = posx[i];
            // posNew_y[i+ig] = posy[i];
            // posNew_z[i+ig] = posz[i];
    #if 1
            // Velocity update
            // double vx  = velx_d[i + ig];
            // double vy  = vely_d[i + ig];
            // double vz  = velz_d[i + ig];
            const double m = mass[i + ig];
            double vnx = (posx[i] - refx[i]) * invdt;
            double vny = (posy[i] - refy[i]) * invdt;
            double vnz = (posz[i] - refz[i]) * invdt;
            df[0] = (vnx - velx[i]) * (m * invdt);
            df[1] = (vny - vely[i]) * (m * invdt);
            df[2] = (vnz - velz[i]) * (m * invdt);


            // Updates velocity and force
            vel_x[i + ig] = vnx;
            vel_y[i + ig] = vny;
            vel_z[i + ig] = vnz;

            f_normal_x[i + ig] += df[0];
            f_normal_y[i + ig] += df[1];
            f_normal_z[i + ig] += df[2];
//             m = h_m;
            // velx_d[i+ig] = (posx[i] - refx[i]) * invdt;
            // vely_d[i+ig] = (posy[i] - refy[i]) * invdt;
            // velz_d[i+ig] = (posz[i] - refz[i]) * invdt;
            // velnew update
            // velNew_x[i + ig] = vnx;
            // velNew_y[i + ig] = vny;
            // velNew_z[i + ig] = vnz;
    #endif
          }
          // consFailure[idx] = 0;
          //printf("Loop is %d %d\n", loop, consFailure[idx]);
      }
    }
  }
}

void CallRattle1Kernel(
  int                                   numAtoms,
  // Settle Parameters
  const double                          dt,
  const double                          invdt,
  const int                             nSettles,
  double * __restrict                   vel_x,
  double * __restrict                   vel_y,
  double * __restrict                   vel_z,
  const double * __restrict             pos_x,
  const double * __restrict             pos_y,
  const double * __restrict             pos_z,
  double * __restrict                   f_normal_x,
  double * __restrict                   f_normal_y,
  double * __restrict                   f_normal_z,
  const float * __restrict              mass,
  const int   * __restrict              hydrogenGroupSize,
  const float * __restrict              rigidBondLength,
  const int   * __restrict              atomFixed,
  int * __restrict                      settleList,
  const SettleParameters *              sp,
  const CudaRattleElem * __restrict     rattleList,
  const int                             nShakes,
  const BigReal                         tol2_d,
  const int                             maxiter_d,
  int*                                  consFailure,
  const int                             nSettleBlocks,
  const int                             nShakeBlocks,
  const WaterModel                      water_model,
  cudaStream_t                          stream
) {
  const int nTotalBlocks  = (nSettleBlocks + nShakeBlocks);
#define CALL(WATER_MODEL) \
    Rattle1Kernel<getWaterModelGroupSize(WATER_MODEL), WATER_MODEL> \
    <<<nTotalBlocks, 128, 0, stream>>> \
    (numAtoms, dt, invdt, nSettles, \
     vel_x, vel_y, vel_z, \
     pos_x, pos_y, pos_z, \
     f_normal_x, f_normal_y, f_normal_z, \
     mass, hydrogenGroupSize, rigidBondLength, atomFixed, \
     settleList, sp, \
     rattleList, nShakes, tol2_d, maxiter_d, consFailure, nSettleBlocks);
    switch (water_model) {
      case WaterModel::SWM4: CALL(WaterModel::SWM4); break;
      case WaterModel::TIP4: CALL(WaterModel::TIP4); break;
      case WaterModel::TIP3: CALL(WaterModel::TIP3); break;
    }
#undef CALL
  cudaStreamSynchronize(stream);
}
