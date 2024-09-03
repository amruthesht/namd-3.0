#ifdef WIN32
#define _USE_MATH_DEFINES
#define __thread __declspec(thread)
#endif  // WIN32

#ifdef NAMD_HIP
#include <hipcub/hipcub.hpp>
#define cub hipcub
#endif //  NAMD_HIP

#ifdef NAMD_CUDA
#include <cuda.h>
#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include <namd_cub/cub.cuh>
#endif
#endif  // NAMD_CUDA

#include <math.h>

#include "ComputeBondedCUDAKernel.h"
#include "CudaComputeNonbondedInteractions.h"

#ifdef FUTURE_CUDADEVICE
#include "CudaDevice.h"
#else
#include "DeviceCUDA.h"
extern __thread DeviceCUDA *deviceCUDA;
#endif

#ifndef USE_TABLE_ARRAYS
__device__ __forceinline__
float4 sampleTableTex(cudaTextureObject_t tex, float k) {
#if defined(NAMD_CUDA)
  return tex1D<float4>(tex, k);
#else
  const int tableSize = FORCE_ENERGY_TABLE_SIZE;
  const float x = k * (float)tableSize - 0.5f;
  const float f = floorf(x);
  const float a = x - f;
  const unsigned int i = (unsigned int)f;
  const int i0 = i < tableSize - 1 ? i : tableSize - 1;
  const int i1 = i0 + 1;
  const float4 t0 = tex1Dfetch<float4>(tex, i0);
  const float4 t1 = tex1Dfetch<float4>(tex, i1);
  return make_float4(
    a * (t1.x - t0.x) + t0.x,
    a * (t1.y - t0.y) + t0.y,
    a * (t1.z - t0.z) + t0.z,
    a * (t1.w - t0.w) + t0.w);
#endif
}
#endif

__device__ __forceinline__
float4 tableLookup(const float4* table, const float k)
{
  const int tableSize = FORCE_ENERGY_TABLE_SIZE;
  const float x = k * static_cast<float>(tableSize) - 0.5f;
  const float f = floorf(x);
  const float a = x - f;
  const int i = static_cast<int>(f);
  const int i0 = max(0, min(tableSize - 1, i));
  const int i1 = max(0, min(tableSize - 1, i + 1));
  const float4 t0 = __ldg(&table[i0]);
  const float4 t1 = __ldg(&table[i1]);
  return make_float4(
    a * (t1.x - t0.x) + t0.x,
    a * (t1.y - t0.y) + t0.y,
    a * (t1.z - t0.z) + t0.z,
    a * (t1.w - t0.w) + t0.w);
}

// global variable for alchemical transformation
namespace AlchBondedCUDA {
  // Alchemical parameters and lambdas
  __constant__ CudaAlchParameters  alchParams;
  __constant__ CudaAlchLambdas     alchLambdas;
}


template <typename T>
__forceinline__ __device__
void convertForces(const float x, const float y, const float z,
  T &Tx, T &Ty, T &Tz) {

  Tx = (T)(x);
  Ty = (T)(y);
  Tz = (T)(z);
}

template <typename T>
__forceinline__ __device__
void calcComponentForces(
  float fij,
  const float dx, const float dy, const float dz,
  T &fxij, T &fyij, T &fzij) {

  fxij = (T)(fij*dx);
  fyij = (T)(fij*dy);
  fzij = (T)(fij*dz);

}

template <typename T>
__forceinline__ __device__
void calcComponentForces(
  float fij1,
  const float dx1, const float dy1, const float dz1,
  float fij2,
  const float dx2, const float dy2, const float dz2,
  T &fxij, T &fyij, T &fzij) {

  fxij = (T)(fij1*dx1 + fij2*dx2);
  fyij = (T)(fij1*dy1 + fij2*dy2);
  fzij = (T)(fij1*dz1 + fij2*dz2);
}

__forceinline__ __device__
int warpAggregatingAtomicInc(int* counter) {
#if BOUNDINGBOXSIZE == 64
  WarpMask mask = __ballot(1);
  int total = __popcll(mask);
  int prefix = __popcll(mask & cub::LaneMaskLt());
  int firstLane = __ffsll(mask) - 1;
#else
  WarpMask mask = __ballot(1);
  int total = __popc(mask);
  int prefix = __popc(mask & cub::LaneMaskLt());
  int firstLane = __ffs(mask) - 1;
#endif
  int start = 0;
  if (prefix == 0) {
    start = atomicAdd(counter, total);
  }
  start = WARP_SHUFFLE(mask, start, firstLane, WARPSIZE);
  return start + prefix;
}

template <typename T>
__forceinline__ __device__
void storeForces(const T fx, const T fy, const T fz,
     const int ind, const int stride,
     T* force,
     T* forceList, int* forceListCounter,
     int* forceListStarts, int* forceListNexts) {

#if defined(USE_BONDED_FORCE_ATOMIC_STORE)
#if defined(NAMD_HIP)
  // Try to decrease conflicts between lanes if there are repeting ind
  WarpMask mask = NAMD_WARP_BALLOT(WARP_FULL_MASK, 1);
  if (mask == ~(WarpMask)0) { // All lanes are active (may be not true for the last block)
    const int laneId = threadIdx.x % WARPSIZE;
    const int prevLaneInd = WARP_SHUFFLE(WARP_FULL_MASK, ind, laneId - 1, WARPSIZE);
    const bool isHead = laneId == 0 || ind != prevLaneInd;
    if (!NAMD_WARP_ALL(WARP_FULL_MASK, isHead)) {
      // There are segments of repeating ind
      typedef cub::WarpReduce<T> WarpReduce;
      __shared__ typename WarpReduce::TempStorage temp_storage;
      const T sumfx = WarpReduce(temp_storage).HeadSegmentedSum(fx, isHead);
      const T sumfy = WarpReduce(temp_storage).HeadSegmentedSum(fy, isHead);
      const T sumfz = WarpReduce(temp_storage).HeadSegmentedSum(fz, isHead);
      if (isHead) {
        atomicAdd(&force[ind           ], sumfx);
        atomicAdd(&force[ind + stride  ], sumfy);
        atomicAdd(&force[ind + stride*2], sumfz);
      }
      return;
    }
  }
  // Not all lanes are active (the last block) or there is no repeating ind
  atomicAdd(&force[ind           ], fx);
  atomicAdd(&force[ind + stride  ], fy);
  atomicAdd(&force[ind + stride*2], fz);
#else
  atomicAdd(&force[ind           ], fx);
  atomicAdd(&force[ind + stride  ], fy);
  atomicAdd(&force[ind + stride*2], fz);
#endif
#else
  const int newPos = warpAggregatingAtomicInc(forceListCounter);
  forceListNexts[newPos] = atomicExch(&forceListStarts[ind], newPos);
  forceList[newPos * 3 + 0] = fx;
  forceList[newPos * 3 + 1] = fy;
  forceList[newPos * 3 + 2] = fz;
#endif
}

//
// Calculates bonds
//
template <typename T, bool doEnergy, bool doVirial, bool doFEP, bool doTI>
__device__ void bondForce(
  const int index,
  const CudaBond* __restrict__ bondList,
  const CudaBondValue* __restrict__ bondValues,
  const float4* __restrict__ xyzq,
  const int stride,
  const float3 lata, const float3 latb, const float3 latc,
  T* __restrict__ force, double &energy,
  T* __restrict__ forceList, int* forceListCounter,
  int* forceListStarts, int* forceListNexts,
#ifdef WRITE_FULL_VIRIALS
  ComputeBondedCUDAKernel::BondedVirial<double>& virial,
#else
  ComputeBondedCUDAKernel::BondedVirial* __restrict__ virial,
#endif
  double &energy_F, double &energy_TI_1, double &energy_TI_2
  ) {

  CudaBond bl = bondList[index];
  CudaBondValue bondValue = bondValues[bl.itype];
//   if (bondValue.x0 == 0.0f) return;

  float shx = bl.ioffsetXYZ.x*lata.x + bl.ioffsetXYZ.y*latb.x + bl.ioffsetXYZ.z*latc.x;
  float shy = bl.ioffsetXYZ.x*lata.y + bl.ioffsetXYZ.y*latb.y + bl.ioffsetXYZ.z*latc.y;
  float shz = bl.ioffsetXYZ.x*lata.z + bl.ioffsetXYZ.y*latb.z + bl.ioffsetXYZ.z*latc.z;

  float4 xyzqi = xyzq[bl.i];
  float4 xyzqj = xyzq[bl.j];

  float xij = xyzqi.x + shx - xyzqj.x;
  float yij = xyzqi.y + shy - xyzqj.y;
  float zij = xyzqi.z + shz - xyzqj.z;

  float r = sqrtf(xij*xij + yij*yij + zij*zij);

  float db = r - bondValue.x0;
  if (bondValue.x1) {
    // in this case, the bond represents a harmonic wall potential
    // where x0 is the lower wall and x1 is the upper
    db = (r > bondValue.x1 ? r - bondValue.x1 : (r > bondValue.x0 ? 0 : db));
  }
  float fij = db * bondValue.k * bl.scale;
 
  // Define a temporary variable to store the energy
  // used by both alchemical and non-alchemical route
  float energy_tmp = 0.0f;
  if (doEnergy) {
    energy_tmp += fij * db;
  }

  // Alchemical route
  // JM: Get this shit off here
  float alch_scale = 1.0f;
  if (doFEP || doTI) {
    switch (bl.fepBondedType) {
      case 1: {
        alch_scale *= (AlchBondedCUDA::alchLambdas).bondLambda1;
        if (doTI) {
          energy_TI_1 += (double)(energy_tmp);
        }
        if (doFEP) {
          energy_F += (double)(energy_tmp * ((AlchBondedCUDA::alchLambdas).bondLambda12 - (AlchBondedCUDA::alchLambdas).bondLambda1));
        }
        break;
      }
      case 2: {
        alch_scale *= (AlchBondedCUDA::alchLambdas).bondLambda2;
        if (doTI) {
          energy_TI_2 += (double)(energy_tmp);
        }
        if (doFEP) {
          energy_F += (double)(energy_tmp * ((AlchBondedCUDA::alchLambdas).bondLambda22 - (AlchBondedCUDA::alchLambdas).bondLambda2));
        }
        break;
      }
    }
  }

  if (doEnergy) {
    if (doFEP || doTI) {
      energy += (double)(energy_tmp * alch_scale);
    } else {
      energy += (double)(energy_tmp);
    }
  }
  fij *= -2.0f/r;
  // XXX TODO: Get this off here
  // XXX TODO: Decide on a templated parameter nomenclature
  if (doFEP || doTI) {
    fij *= alch_scale;
  }
  
  T T_fxij, T_fyij, T_fzij;
  calcComponentForces<T>(fij, xij, yij, zij, T_fxij, T_fyij, T_fzij);
  
  // Store forces
  storeForces<T>(T_fxij, T_fyij, T_fzij, bl.i, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);
  storeForces<T>(-T_fxij, -T_fyij, -T_fzij, bl.j, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);

  // Store virial
  if (doVirial) {
#ifdef WRITE_FULL_VIRIALS
    float fxij = fij*xij;
    float fyij = fij*yij;
    float fzij = fij*zij;
    virial.xx = (fxij*xij);
    virial.xy = (fxij*yij);
    virial.xz = (fxij*zij);
    virial.yx = (fyij*xij);
    virial.yy = (fyij*yij);
    virial.yz = (fyij*zij);
    virial.zx = (fzij*xij);
    virial.zy = (fzij*yij);
    virial.zz = (fzij*zij);
#endif
  }
}

//
// Calculates modified exclusions
//
template <typename T, bool doEnergy, bool doVirial, bool doElect, bool doFEP, bool doTI, bool doTable>
__device__ void modifiedExclusionForce(
  const int index,
  const CudaExclusion* __restrict__ exclusionList,
  const bool doSlow,
  const float one_scale14,                // 1 - scale14
  const int vdwCoefTableWidth,
#if defined(USE_TABLE_ARRAYS) ||  __CUDA_ARCH__ >= 350
  const float2* __restrict__ vdwCoefTable,
#else
  cudaTextureObject_t vdwCoefTableTex, 
#endif
#ifdef USE_TABLE_ARRAYS
  const float4* __restrict__ forceTable, const float4* __restrict__ energyTable,
#else
  cudaTextureObject_t forceTableTex, cudaTextureObject_t energyTableTex,
#endif
  const float4* __restrict__ xyzq,
  const int stride,
  const float3 lata, const float3 latb, const float3 latc,
  const float cutoff2,
  const CudaNBConstants nbConstants,
  double &energyVdw,
  T* __restrict__ forceNbond, double &energyNbond,
  T* __restrict__ forceSlow, double &energySlow,
  T* __restrict__ forceList, int* forceListCounter,
  int* forceListStartsNbond, int* forceListStartsSlow, int* forceListNexts,
#ifdef WRITE_FULL_VIRIALS
  ComputeBondedCUDAKernel::BondedVirial<double>& virialNbond,
  ComputeBondedCUDAKernel::BondedVirial<double>& virialSlow,
#else
  ComputeBondedCUDAKernel::BondedVirial* __restrict__ virialNbond,
  ComputeBondedCUDAKernel::BondedVirial* __restrict__ virialSlow,
#endif
  double &energyVdw_F, double &energyVdw_TI_1, double &energyVdw_TI_2,
  double &energyNbond_F, double &energyNbond_TI_1, double &energyNbond_TI_2,
  double &energySlow_F, double &energySlow_TI_1, double &energySlow_TI_2
  ) {

  CudaExclusion bl = exclusionList[index];

  float shx = bl.ioffsetXYZ.x*lata.x + bl.ioffsetXYZ.y*latb.x + bl.ioffsetXYZ.z*latc.x;
  float shy = bl.ioffsetXYZ.x*lata.y + bl.ioffsetXYZ.y*latb.y + bl.ioffsetXYZ.z*latc.y;
  float shz = bl.ioffsetXYZ.x*lata.z + bl.ioffsetXYZ.y*latb.z + bl.ioffsetXYZ.z*latc.z;

  float4 xyzqi = xyzq[bl.i];
  float4 xyzqj = xyzq[bl.j];

  float xij = xyzqi.x + shx - xyzqj.x;
  float yij = xyzqi.y + shy - xyzqj.y;
  float zij = xyzqi.z + shz - xyzqj.z;

  float r2 = xij*xij + yij*yij + zij*zij;
  if (r2 < cutoff2) {

    float rinv = rsqrtf(r2);

    float qq;
    if (doElect) qq = one_scale14 * xyzqi.w * xyzqj.w;

    int vdwIndex = bl.vdwtypei + bl.vdwtypej*vdwCoefTableWidth;
#if defined(USE_TABLE_ARRAYS) || __CUDA_ARCH__ >= 350 
    float2 ljab = __ldg(&vdwCoefTable[vdwIndex]);
#else
    float2 ljab = tex1Dfetch<float2>(vdwCoefTableTex, vdwIndex);
#endif

    // Alchemical route
    float myElecLambda = 1.0f;
    float myElecLambda2 = 1.0f;
    float myVdwLambda = 1.0f;
    float myVdwLambda2 = 1.0f;
    float myVdwShift = 0.0f;
    float myVdwShift2 = 0.0f;
    double alch_vdw_energy;
    double alch_vdw_energy_2;
    float alch_vdw_force = 0.0f;
    float alch_vdw_dUdl = 0.0f;
    double* p_energyVdw_TI = NULL;
    double* p_energyNbond_TI = NULL;
    double* p_energySlow_TI = NULL;
    float p0Factor;
    if (doFEP || doTI) {
      p0Factor = 0.0f;
      /*lambda values 'up' are for atoms scaled up with lambda (partition 1)*/
      float elecLambdaUp        = (AlchBondedCUDA::alchLambdas).elecLambdaUp;
      float vdwLambdaUp         = (AlchBondedCUDA::alchLambdas).vdwLambdaUp;
      float vdwShiftUp          = (AlchBondedCUDA::alchParams).alchVdwShiftCoeff * (1.0f - vdwLambdaUp);
      float elecLambda2Up       = (AlchBondedCUDA::alchLambdas).elecLambda2Up;
      float vdwLambda2Up        = (AlchBondedCUDA::alchLambdas).vdwLambda2Up;
      float vdwShift2Up         = (AlchBondedCUDA::alchParams).alchVdwShiftCoeff * (1.0f - vdwLambda2Up);
      /*lambda values 'down' are for atoms scaled down with lambda (partition 2)*/
      float elecLambdaDown      = (AlchBondedCUDA::alchLambdas).elecLambdaDown;
      float vdwLambdaDown       = (AlchBondedCUDA::alchLambdas).vdwLambdaDown;
      float vdwShiftDown        = (AlchBondedCUDA::alchParams).alchVdwShiftCoeff * (1.0f - vdwLambdaDown);
      float elecLambda2Down     = (AlchBondedCUDA::alchLambdas).elecLambda2Down;
      float vdwLambda2Down      = (AlchBondedCUDA::alchLambdas).vdwLambda2Down;
      float vdwShift2Down       = (AlchBondedCUDA::alchParams).alchVdwShiftCoeff * (1.0f - vdwLambda2Down);
      switch (bl.pswitch) {
        case 0: myVdwLambda     = 1.0f;
                myVdwLambda2    = 1.0f;
                myElecLambda    = 1.0f;
                myElecLambda2   = 1.0f;
                p0Factor = 1.0f;
                break;
        case 1: myVdwLambda     = vdwLambdaUp;
                myVdwLambda2    = vdwLambda2Up;
                myElecLambda    = elecLambdaUp;
                myElecLambda2   = elecLambda2Up;
                myVdwShift      = vdwShiftUp;
                myVdwShift2     = vdwShift2Up;
                p_energyVdw_TI  = &(energyVdw_TI_1);
                p_energyNbond_TI= &(energyNbond_TI_1);
                p_energySlow_TI = &(energySlow_TI_1);
                break;
        case 2: myVdwLambda     = vdwLambdaDown;
                myVdwLambda2    = vdwLambda2Down;
                myElecLambda    = elecLambdaDown;
                myElecLambda2   = elecLambda2Down;
                myVdwShift      = vdwShiftDown;
                myVdwShift2     = vdwShift2Down;
                p_energyVdw_TI  = &(energyVdw_TI_2);
                p_energyNbond_TI= &(energyNbond_TI_2);
                p_energySlow_TI = &(energySlow_TI_2);
                break;
        default: myVdwLambda    = 0.0f;
                 myVdwLambda2   = 0.0f;
                 myElecLambda   = 0.0f;
                 myElecLambda2  = 0.0f;
                 break;
      }
      if (bl.pswitch != 99 && bl.pswitch != 0) {
        // Common part of FEP and TI
        const float& switchOn2    = (AlchBondedCUDA::alchParams).switchDist2;
        const float switchfactor  = 1.0f / ((cutoff2 - switchOn2) * (cutoff2 - switchOn2) * (cutoff2 - switchOn2));
        const float switchmul     = (r2 > switchOn2 ? (switchfactor * (cutoff2 - r2) * (cutoff2 - r2) * (cutoff2 - 3.0f * switchOn2 + 2.0f * r2)) : 1.0f);
        const float switchmul2    = (r2 > switchOn2 ? (switchfactor * (cutoff2 - r2) * 12.0f * (r2 - switchOn2)) : 0.0f);
        // A = ljab.x ; B = ljab.y
        const float r2_1 = 1.0f / (r2 + myVdwShift);
        const float r2_2 = 1.0f / (r2 + myVdwShift2);
        const float r6_1 = r2_1 * r2_1 * r2_1;
        const float r6_2 = r2_2 * r2_2 * r2_2;
        const float U1   = ljab.x * r6_1 * r6_1 - ljab.y * r6_1;
        const float U2   = ljab.x * r6_2 * r6_2 - ljab.y * r6_2;
        // rinv is already calculated above
        if (doEnergy) {
          alch_vdw_energy   = double(myVdwLambda  * switchmul * U1);
          if (doFEP) {
            alch_vdw_energy_2 = double(myVdwLambda2 * switchmul * U2);
          }
        }
        alch_vdw_force = myVdwLambda  * (switchmul * (12.0f * U1 + 6.0f * ljab.y * r6_1) * r2_1 + switchmul2 * U1);
        if (doTI) {
          alch_vdw_dUdl = switchmul * (U1 + myVdwLambda * (AlchBondedCUDA::alchParams).alchVdwShiftCoeff * (6.0f * U1 + 3.0f * ljab.y * r6_1) * r2_1);
        }
      } else {
        if (doEnergy) {
          alch_vdw_energy = 0.0;
          alch_vdw_energy_2 = 0.0;
        }
      }
    }

    float4 fi;
    float4 ei;
    if (doTable) {
#ifdef NAMD_HIP
#ifdef USE_TABLE_ARRAYS
      fi = tableLookup(forceTable, rinv);
#else
      fi = sampleTableTex(forceTableTex, rinv);
#endif
#else
      fi = tex1D<float4>(forceTableTex, rinv);
#endif
    }

    if (doEnergy && doTable) {
#ifdef NAMD_HIP
#ifdef USE_TABLE_ARRAYS
      ei = tableLookup(energyTable, rinv);
#else
      ei = sampleTableTex(energyTableTex, rinv);
#endif
#else
      ei = tex1D<float4>(energyTableTex, rinv);
#endif
      if (doFEP || doTI) {
        energyVdw += (double)((ljab.x * ei.z + ljab.y * ei.y) * myElecLambda * p0Factor);
        energyVdw += alch_vdw_energy;
        if (doFEP) {
          energyVdw_F += (double)((ljab.x * ei.z + ljab.y * ei.y) * myElecLambda2 * p0Factor);
          energyVdw_F += alch_vdw_energy_2;
        }
        if (doTI) {
          if (p_energyVdw_TI != NULL) {
            (*p_energyVdw_TI) += alch_vdw_dUdl;
          }
        }
      } else {
        energyVdw += (double)(ljab.x * ei.z + ljab.y * ei.y);
      }
      if (doElect) {
        energyNbond += qq * ei.x * myElecLambda;
        if (doFEP) {
          energyNbond_F += qq * ei.x * myElecLambda2;
        }
        if (doTI) {
          if (p_energyNbond_TI != NULL) (*p_energyNbond_TI) += qq * ei.x;
        }
        if (doSlow) {
          energySlow  += qq * ei.w * myElecLambda;
          if (doFEP) {
            energySlow_F += qq * ei.w * myElecLambda2;
          }
          if (doTI) {
            if (p_energySlow_TI != NULL) (*p_energySlow_TI) += qq * ei.w;
          }
        }
      }
    }

    float fNbond;
    float fSlow;
    if (doTable) {
      if (doElect) {
        fNbond = -(ljab.x * fi.z + ljab.y * fi.y + qq * fi.x);
      } else {
        fNbond = -(ljab.x * fi.z + ljab.y * fi.y);
      }
    } else {
      float e_vdw, e_elec, e_slow;
      if (doEnergy) {
        e_vdw = 0.0f;
        e_elec = 0.0f;
        e_slow = 0.0f;
      }

      cudaModExclForceMagCalc_VdwEnergySwitch_PMEC1<doEnergy>(
        doSlow, doElect, r2, rinv, qq, ljab, 
        nbConstants, fNbond, fSlow, e_vdw, e_elec, e_slow);
        
      if (doEnergy) {
        energyVdw += (double) (e_vdw);
        energyNbond += (double) (e_elec * myElecLambda); 
        energySlow += (double) (e_slow * myElecLambda);
      }
    }
    // fNbond = fFast + fVdw
    if (doFEP || doTI) {
      if (bl.pswitch != 0) {
        fNbond -= -(ljab.x * fi.z + ljab.y * fi.y);
        fNbond = fNbond * myElecLambda + alch_vdw_force * float(bl.pswitch == 1 || bl.pswitch == 2);
      }
      // if pswitch == 0, myElecLambda = 1.0
    }
    T T_fxij, T_fyij, T_fzij;
    calcComponentForces<T>(fNbond, xij, yij, zij, T_fxij, T_fyij, T_fzij);
    storeForces<T>(T_fxij, T_fyij, T_fzij, bl.i, stride, forceNbond, forceList, forceListCounter, forceListStartsNbond, forceListNexts);
    storeForces<T>(-T_fxij, -T_fyij, -T_fzij, bl.j, stride, forceNbond, forceList, forceListCounter, forceListStartsNbond, forceListNexts);

    if (doSlow && doElect) {
      if (doTable) {
        fSlow = -qq * fi.w;
      }
      if (doFEP || doTI) {
        fSlow *= myElecLambda;
      }
      T T_fxij, T_fyij, T_fzij;
      calcComponentForces<T>(fSlow, xij, yij, zij, T_fxij, T_fyij, T_fzij);
      storeForces<T>(T_fxij, T_fyij, T_fzij, bl.i, stride, forceSlow, forceList, forceListCounter, forceListStartsSlow, forceListNexts);
      storeForces<T>(-T_fxij, -T_fyij, -T_fzij, bl.j, stride, forceSlow, forceList, forceListCounter, forceListStartsSlow, forceListNexts);
    }

    // Store virial
    if (doVirial) {
#ifdef WRITE_FULL_VIRIALS
      float fxij = fNbond*xij;
      float fyij = fNbond*yij;
      float fzij = fNbond*zij;
      virialNbond.xx = (fxij*xij);
      virialNbond.xy = (fxij*yij);
      virialNbond.xz = (fxij*zij);
      virialNbond.yx = (fyij*xij);
      virialNbond.yy = (fyij*yij);
      virialNbond.yz = (fyij*zij);
      virialNbond.zx = (fzij*xij);
      virialNbond.zy = (fzij*yij);
      virialNbond.zz = (fzij*zij);
#endif
    }

    // Store virial
    if (doVirial && doSlow && doElect) {
#ifdef WRITE_FULL_VIRIALS
      float fxij = fSlow*xij;
      float fyij = fSlow*yij;
      float fzij = fSlow*zij;
      virialSlow.xx = (fxij*xij);
      virialSlow.xy = (fxij*yij);
      virialSlow.xz = (fxij*zij);
      virialSlow.yx = (fyij*xij);
      virialSlow.yy = (fyij*yij);
      virialSlow.yz = (fyij*zij);
      virialSlow.zx = (fzij*xij);
      virialSlow.zy = (fzij*yij);
      virialSlow.zz = (fzij*zij);
#endif
    }

  }
}

//
// Calculates exclusions. Here doSlow = true
//
// doFull = doFullElectrostatics = doSlow
template <typename T, bool doEnergy, bool doVirial, bool doFEP, bool doTI>
__device__ void exclusionForce(
  const int index,
  const CudaExclusion* __restrict__ exclusionList,
  const float r2_delta, const int r2_delta_expc,

#if __CUDA_ARCH__ >= 350 || defined(NAMD_HIP) || defined(USE_TABLE_ARRAYS)
  const float* __restrict__ r2_table,
  const float4* __restrict__ exclusionTable,
#else
  cudaTextureObject_t r2_table_tex,
  cudaTextureObject_t exclusionTableTex,
#endif
  const float4* __restrict__ xyzq,
  const int stride,
  const float3 lata, const float3 latb, const float3 latc,
  const float cutoff2,
  T* __restrict__ forceSlow, double &energySlow,
  T* __restrict__ forceList, int* forceListCounter,
  int* forceListStartsSlow, int* forceListNexts,
#ifdef WRITE_FULL_VIRIALS
  ComputeBondedCUDAKernel::BondedVirial<double>& virialSlow,
#else
  ComputeBondedCUDAKernel::BondedVirial* __restrict__ virialSlow,
#endif
  double &energy_F, double &energy_TI_1, double &energy_TI_2
  ) {

  CudaExclusion bl = exclusionList[index];

  float shx = bl.ioffsetXYZ.x*lata.x + bl.ioffsetXYZ.y*latb.x + bl.ioffsetXYZ.z*latc.x;
  float shy = bl.ioffsetXYZ.x*lata.y + bl.ioffsetXYZ.y*latb.y + bl.ioffsetXYZ.z*latc.y;
  float shz = bl.ioffsetXYZ.x*lata.z + bl.ioffsetXYZ.y*latb.z + bl.ioffsetXYZ.z*latc.z;

  float4 xyzqi = xyzq[bl.i];
  float4 xyzqj = xyzq[bl.j];

  float xij = xyzqi.x + shx - xyzqj.x;
  float yij = xyzqi.y + shy - xyzqj.y;
  float zij = xyzqi.z + shz - xyzqj.z;

  float r2 = xij*xij + yij*yij + zij*zij;
  if (r2 < cutoff2) {
    r2 += r2_delta;

    union { float f; int i; } r2i;
    r2i.f = r2;
    int table_i = (r2i.i >> 17) + r2_delta_expc;  // table_i >= 0

#if __CUDA_ARCH__ >= 350 || defined(NAMD_HIP) || defined(USE_TABLE_ARRAYS)
#ifdef NAMD_HIP
    float r2_table_val = r2_table[table_i];
#else
    float r2_table_val = __ldg(&r2_table[table_i]);
#endif
#else
    float r2_table_val = tex1Dfetch<float>(r2_table_tex, table_i);
#endif
    float diffa = r2 - r2_table_val;
    float qq = xyzqi.w * xyzqj.w;

#if __CUDA_ARCH__ >= 350 || defined(NAMD_HIP) || defined(USE_TABLE_ARRAYS)
#ifdef NAMD_HIP
    float4 slow = exclusionTable[table_i];
#else
    float4 slow = __ldg(&exclusionTable[table_i]);
#endif
#else
    float4 slow = tex1Dfetch<float4>(exclusionTableTex, table_i);
#endif
    // Alchemical route
    float myElecLambda;
    float myElecLambda2;
    double* p_energy_TI;
    if (doFEP || doTI) {
      myElecLambda = 1.0f;
      myElecLambda2 = 1.0f;
      if (doTI) p_energy_TI = NULL;
      /*lambda values 'up' are for atoms scaled up with lambda (partition 1)*/
      float elecLambdaUp        = (AlchBondedCUDA::alchLambdas).elecLambdaUp;
      float elecLambda2Up       = (AlchBondedCUDA::alchLambdas).elecLambda2Up;
      /*lambda values 'down' are for atoms scaled down with lambda (partition 2)*/
      float elecLambdaDown      = (AlchBondedCUDA::alchLambdas).elecLambdaDown;
      float elecLambda2Down     = (AlchBondedCUDA::alchLambdas).elecLambda2Down;
      switch (bl.pswitch) {
        case 0: myElecLambda = 1.0f;
                myElecLambda2 = 1.0f;
                break;
        case 1: myElecLambda = elecLambdaUp;
                myElecLambda2 = elecLambda2Up;
                p_energy_TI = &(energy_TI_1);
                break;
        case 2: myElecLambda = elecLambdaDown;
                myElecLambda2 = elecLambda2Down;
                p_energy_TI = &(energy_TI_2);
                break;
        default: myElecLambda = 0.0f;
                 myElecLambda2 = 0.0f;
                 break;
      }
    }
    if (doEnergy) {
      double energy_slow_tmp = (double)(qq*(((diffa * (1.0f/6.0f)*slow.x + 0.25f*slow.y ) * diffa + 0.5f*slow.z ) * diffa + slow.w));
      if (doFEP || doTI) {
        energySlow += energy_slow_tmp * myElecLambda;
        if (doFEP) {
          energy_F += energy_slow_tmp * myElecLambda2;
        }
        if (doTI) {
          if (p_energy_TI != NULL) {
            (*p_energy_TI) += energy_slow_tmp;
          }
        }
      } else {
        energySlow += energy_slow_tmp;
      }
    }

    float fSlow = -qq*((diffa * slow.x + slow.y) * diffa + slow.z);
    if (doFEP || doTI) {
      fSlow *= myElecLambda;
    }

    T T_fxij, T_fyij, T_fzij;
    calcComponentForces<T>(fSlow, xij, yij, zij, T_fxij, T_fyij, T_fzij);
    storeForces<T>(T_fxij, T_fyij, T_fzij, bl.i, stride, forceSlow, forceList, forceListCounter, forceListStartsSlow, forceListNexts);
    storeForces<T>(-T_fxij, -T_fyij, -T_fzij, bl.j, stride, forceSlow, forceList, forceListCounter, forceListStartsSlow, forceListNexts);

    // Store virial
    if (doVirial) {
#ifdef WRITE_FULL_VIRIALS
      float fxij = fSlow*xij;
      float fyij = fSlow*yij;
      float fzij = fSlow*zij;
      virialSlow.xx = (fxij*xij);
      virialSlow.xy = (fxij*yij);
      virialSlow.xz = (fxij*zij);
      virialSlow.yx = (fyij*xij);
      virialSlow.yy = (fyij*yij);
      virialSlow.yz = (fyij*zij);
      virialSlow.zx = (fzij*xij);
      virialSlow.zy = (fzij*yij);
      virialSlow.zz = (fzij*zij);
#endif
    }
  }
}

template <typename T, bool doEnergy, bool doVirial, bool doFEP, bool doTI>
__device__ void angleForce(const int index,
  const CudaAngle* __restrict__ angleList,
  const CudaAngleValue* __restrict__ angleValues,
  const float4* __restrict__ xyzq,
  const int stride,
  const float3 lata, const float3 latb, const float3 latc,
  T* __restrict__ force, double &energy,
  T* __restrict__ forceList, int* forceListCounter,
  int* forceListStarts, int* forceListNexts,
#ifdef WRITE_FULL_VIRIALS
  ComputeBondedCUDAKernel::BondedVirial<double>& virial,
#else
  ComputeBondedCUDAKernel::BondedVirial* __restrict__ virial,
#endif
  double &energy_F, double &energy_TI_1, double &energy_TI_2
  ) {

  CudaAngle al = angleList[index];

  float ishx = al.ioffsetXYZ.x*lata.x + al.ioffsetXYZ.y*latb.x + al.ioffsetXYZ.z*latc.x;
  float ishy = al.ioffsetXYZ.x*lata.y + al.ioffsetXYZ.y*latb.y + al.ioffsetXYZ.z*latc.y;
  float ishz = al.ioffsetXYZ.x*lata.z + al.ioffsetXYZ.y*latb.z + al.ioffsetXYZ.z*latc.z;

  float kshx = al.koffsetXYZ.x*lata.x + al.koffsetXYZ.y*latb.x + al.koffsetXYZ.z*latc.x;
  float kshy = al.koffsetXYZ.x*lata.y + al.koffsetXYZ.y*latb.y + al.koffsetXYZ.z*latc.y;
  float kshz = al.koffsetXYZ.x*lata.z + al.koffsetXYZ.y*latb.z + al.koffsetXYZ.z*latc.z;

  float xij = xyzq[al.i].x + ishx - xyzq[al.j].x;
  float yij = xyzq[al.i].y + ishy - xyzq[al.j].y;
  float zij = xyzq[al.i].z + ishz - xyzq[al.j].z;

  float xkj = xyzq[al.k].x + kshx - xyzq[al.j].x;
  float ykj = xyzq[al.k].y + kshy - xyzq[al.j].y;
  float zkj = xyzq[al.k].z + kshz - xyzq[al.j].z;

  float rij_inv = rsqrtf(xij*xij + yij*yij + zij*zij);
  float rkj_inv = rsqrtf(xkj*xkj + ykj*ykj + zkj*zkj);

  float xijr = xij*rij_inv;
  float yijr = yij*rij_inv;
  float zijr = zij*rij_inv;
  float xkjr = xkj*rkj_inv;
  float ykjr = ykj*rkj_inv;
  float zkjr = zkj*rkj_inv;
  float cos_theta = xijr*xkjr + yijr*ykjr + zijr*zkjr;

  CudaAngleValue angleValue = angleValues[al.itype];
  angleValue.k *= al.scale;

  float diff;
  if (angleValue.normal == 1) {
    // Restrict values of cos_theta to the interval [-0.999 ... 0.999]
    cos_theta = min(0.999f, max(-0.999f, cos_theta));
    float theta = acosf(cos_theta);
    diff = theta - angleValue.theta0;
  } else {
    diff = cos_theta - angleValue.theta0;
  }

  float energy_tmp = 0.0f;
  if (doEnergy) {
    energy_tmp += angleValue.k * diff * diff;
  }

  // Alchemical route
  float alch_scale;
  // Be careful: alch_scale_energy_fep is 0 here!
  float alch_scale_energy_fep;
  double* p_energy_TI;
  // NOTE: On account of the Urey-Bradley terms, energy calculation is not
  //       finished so we cannot add energy_tmp to energy_TI_* and energy_F here!
  //       but we still need the scaling factor for forces,
  //       so the code here is a little bit different from the bondForce.
  //       calculate a scale factor for FEP energy first, and then scale it to
  //       the final result after finishing energy evaluation, and also use 
  //       a pointer point to the correct energy_TI_x term.
  if (doFEP || doTI) {
    alch_scale = 1.0f;
    alch_scale_energy_fep = 0.0f;
    if (doTI) p_energy_TI = NULL;
    switch (al.fepBondedType) {
      case 1: {
        alch_scale *= (AlchBondedCUDA::alchLambdas).bondLambda1;
        if (doEnergy) {
          if (doTI) {
            p_energy_TI = &(energy_TI_1);
          }
          if (doFEP) {
            alch_scale_energy_fep = (AlchBondedCUDA::alchLambdas).bondLambda12 - (AlchBondedCUDA::alchLambdas).bondLambda1;
          }
        }
        break;
      }
      case 2: {
        alch_scale *= (AlchBondedCUDA::alchLambdas).bondLambda2;
        if (doEnergy) {
          if (doTI) {
            p_energy_TI = &(energy_TI_2);
          }
          if (doFEP) {
            alch_scale_energy_fep = (AlchBondedCUDA::alchLambdas).bondLambda22 - (AlchBondedCUDA::alchLambdas).bondLambda2;
          }
        }
        break;
      }
    }
  }

  if (angleValue.normal == 1) {
    float inv_sin_theta = rsqrtf(1.0f - cos_theta*cos_theta);
    if (inv_sin_theta > 1.0e6) {
      diff = (diff < 0.0f) ? 1.0f : -1.0f;
    } else {
      diff *= -inv_sin_theta;
    }
  }
  diff *= -2.0f*angleValue.k;

  // Do alchemical scaling
  if (doFEP || doTI) {
    diff *= alch_scale;
  }

  float dtxi = rij_inv*(xkjr - cos_theta*xijr);
  float dtxj = rkj_inv*(xijr - cos_theta*xkjr);
  float dtyi = rij_inv*(ykjr - cos_theta*yijr);
  float dtyj = rkj_inv*(yijr - cos_theta*ykjr);
  float dtzi = rij_inv*(zkjr - cos_theta*zijr);
  float dtzj = rkj_inv*(zijr - cos_theta*zkjr);

  T T_dtxi, T_dtyi, T_dtzi;
  T T_dtxj, T_dtyj, T_dtzj;
  calcComponentForces<T>(diff, dtxi, dtyi, dtzi, T_dtxi, T_dtyi, T_dtzi);
  calcComponentForces<T>(diff, dtxj, dtyj, dtzj, T_dtxj, T_dtyj, T_dtzj);
  T T_dtxk = -T_dtxi - T_dtxj;
  T T_dtyk = -T_dtyi - T_dtyj;
  T T_dtzk = -T_dtzi - T_dtzj;
  storeForces<T>(T_dtxk, T_dtyk, T_dtzk, al.j, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);

  if (angleValue.k_ub) {
    float xik = xij - xkj;
    float yik = yij - ykj;
    float zik = zij - zkj;
    float rik_inv = rsqrtf(xik*xik + yik*yik + zik*zik);
    float rik = 1.0f/rik_inv;
    float diff_ub = rik - angleValue.r_ub;
    if (doEnergy) {
      energy_tmp += angleValue.k_ub * diff_ub * diff_ub;
    }
    diff_ub *= -2.0f*angleValue.k_ub*rik_inv;
    if (doFEP || doTI) {
      diff_ub *= alch_scale;
    }
    T T_dxik, T_dyik, T_dzik;
    calcComponentForces<T>(diff_ub, xik, yik, zik, T_dxik, T_dyik, T_dzik);
    T_dtxi += T_dxik;
    T_dtyi += T_dyik;
    T_dtzi += T_dzik;
    T_dtxj -= T_dxik;
    T_dtyj -= T_dyik;
    T_dtzj -= T_dzik;
  }



  if (doEnergy) {
    if (doFEP || doTI) {
      energy += double(energy_tmp * alch_scale);
      if (doTI) {
        if (p_energy_TI != NULL) {
          *(p_energy_TI) += double(energy_tmp);
        }
      }
      if (doFEP) {
        energy_F += double(energy_tmp * alch_scale_energy_fep);
      }
    } else {
      energy += double(energy_tmp);
    }
  }

  storeForces<T>(T_dtxi, T_dtyi, T_dtzi, al.i, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);
  storeForces<T>(T_dtxj, T_dtyj, T_dtzj, al.k, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);

  // Store virial
  if (doVirial) {
#ifdef WRITE_FULL_VIRIALS
    float fxi = ((float)T_dtxi);
    float fyi = ((float)T_dtyi);
    float fzi = ((float)T_dtzi);
    float fxk = ((float)T_dtxj);
    float fyk = ((float)T_dtyj);
    float fzk = ((float)T_dtzj);
    virial.xx = (fxi*xij) + (fxk*xkj);
    virial.xy = (fxi*yij) + (fxk*ykj);
    virial.xz = (fxi*zij) + (fxk*zkj);
    virial.yx = (fyi*xij) + (fyk*xkj);
    virial.yy = (fyi*yij) + (fyk*ykj);
    virial.yz = (fyi*zij) + (fyk*zkj);
    virial.zx = (fzi*xij) + (fzk*xkj);
    virial.zy = (fzi*yij) + (fzk*ykj);
    virial.zz = (fzi*zij) + (fzk*zkj);
#endif
  }
}

//
// Dihedral computation
//
// Out: df, e
//
template <bool doEnergy>
__forceinline__ __device__
void diheComp(const CudaDihedralValue* dihedralValues, int ic,
  const float sin_phi, const float cos_phi, const float scale, float& df, double& e) {

  df = 0.0f;
  if (doEnergy) e = 0.0;

  float phi = atan2f(sin_phi, cos_phi);

  bool lrep = true;
  while (lrep) {
    CudaDihedralValue dihedralValue = dihedralValues[ic];
    dihedralValue.k *= scale;

    // Last dihedral has n low bit set to 0
    lrep = (dihedralValue.n & 1);
    dihedralValue.n >>= 1;
    if (dihedralValue.n) {
      float nf = dihedralValue.n;
      float x = nf*phi - dihedralValue.delta;
      if (doEnergy) {
        float sin_x, cos_x;
        sincosf(x, &sin_x, &cos_x);
        e += (double)(dihedralValue.k*(1.0f + cos_x));
        df += (double)(nf*dihedralValue.k*sin_x);
      } else {
        float sin_x = sinf(x);
        df += (double)(nf*dihedralValue.k*sin_x);
      }
    } else {
      float diff = phi - dihedralValue.delta;
      if (diff < -(float)(M_PI)) diff += (float)(2.0*M_PI);
      if (diff >  (float)(M_PI)) diff -= (float)(2.0*M_PI);
      if (doEnergy) {
        e += (double)(dihedralValue.k*diff*diff);
      }
      df -= 2.0f*dihedralValue.k*diff;
    }
    ic++;
  }

}


template <typename T, bool doEnergy, bool doVirial, bool doFEP, bool doTI>
__device__ void diheForce(const int index,
  const CudaDihedral* __restrict__ diheList,
  const CudaDihedralValue* __restrict__ dihedralValues,
  const float4* __restrict__ xyzq,
  const int stride,
  const float3 lata, const float3 latb, const float3 latc,
  T* __restrict__ force, double &energy,
  T* __restrict__ forceList, int* forceListCounter, int* forceListStarts, int* forceListNexts,
#ifdef WRITE_FULL_VIRIALS
  ComputeBondedCUDAKernel::BondedVirial<double>& virial,
#else
  ComputeBondedCUDAKernel::BondedVirial* __restrict__ virial,
#endif
  double &energy_F, double &energy_TI_1, double &energy_TI_2
  ) {

  CudaDihedral dl = diheList[index];

  float ishx = dl.ioffsetXYZ.x*lata.x + dl.ioffsetXYZ.y*latb.x + dl.ioffsetXYZ.z*latc.x;
  float ishy = dl.ioffsetXYZ.x*lata.y + dl.ioffsetXYZ.y*latb.y + dl.ioffsetXYZ.z*latc.y;
  float ishz = dl.ioffsetXYZ.x*lata.z + dl.ioffsetXYZ.y*latb.z + dl.ioffsetXYZ.z*latc.z;

  float jshx = dl.joffsetXYZ.x*lata.x + dl.joffsetXYZ.y*latb.x + dl.joffsetXYZ.z*latc.x;
  float jshy = dl.joffsetXYZ.x*lata.y + dl.joffsetXYZ.y*latb.y + dl.joffsetXYZ.z*latc.y;
  float jshz = dl.joffsetXYZ.x*lata.z + dl.joffsetXYZ.y*latb.z + dl.joffsetXYZ.z*latc.z;

  float lshx = dl.loffsetXYZ.x*lata.x + dl.loffsetXYZ.y*latb.x + dl.loffsetXYZ.z*latc.x;
  float lshy = dl.loffsetXYZ.x*lata.y + dl.loffsetXYZ.y*latb.y + dl.loffsetXYZ.z*latc.y;
  float lshz = dl.loffsetXYZ.x*lata.z + dl.loffsetXYZ.y*latb.z + dl.loffsetXYZ.z*latc.z;

  float xij = xyzq[dl.i].x + ishx - xyzq[dl.j].x;
  float yij = xyzq[dl.i].y + ishy - xyzq[dl.j].y;
  float zij = xyzq[dl.i].z + ishz - xyzq[dl.j].z;

  float xjk = xyzq[dl.j].x + jshx - xyzq[dl.k].x;
  float yjk = xyzq[dl.j].y + jshy - xyzq[dl.k].y;
  float zjk = xyzq[dl.j].z + jshz - xyzq[dl.k].z;

  float xlk = xyzq[dl.l].x + lshx - xyzq[dl.k].x;
  float ylk = xyzq[dl.l].y + lshy - xyzq[dl.k].y;
  float zlk = xyzq[dl.l].z + lshz - xyzq[dl.k].z;

  // A=F^G, B=H^G.
  float ax = yij*zjk - zij*yjk;
  float ay = zij*xjk - xij*zjk;
  float az = xij*yjk - yij*xjk;
  float bx = ylk*zjk - zlk*yjk;
  float by = zlk*xjk - xlk*zjk;
  float bz = xlk*yjk - ylk*xjk;

  float ra2 = ax*ax + ay*ay + az*az;
  float rb2 = bx*bx + by*by + bz*bz;
  float rg = sqrtf(xjk*xjk + yjk*yjk + zjk*zjk);

  //    if((ra2 <= rxmin2) .or. (rb2 <= rxmin2) .or. (rg <= rxmin)) then
  //          nlinear = nlinear + 1
  //       endif

  float rgr = 1.0f / rg;
  float ra2r = 1.0f / ra2;
  float rb2r = 1.0f / rb2;
  float rabr = sqrtf(ra2r*rb2r);

  float cos_phi = (ax*bx + ay*by + az*bz)*rabr;
  //
  // Note that sin(phi).G/|G|=B^A/(|A|.|B|)
  // which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
  float sin_phi = rg*rabr*(ax*xlk + ay*ylk + az*zlk);
  //
  //     Energy and derivative contributions.

  float df;
  double e;
  diheComp<doEnergy>(dihedralValues, dl.itype, sin_phi, cos_phi, dl.scale, df, e);

  // Alchemical transformation
  float alch_scale;
  if (doFEP || doTI) {
    alch_scale = 1.0f;
    switch (dl.fepBondedType) {
      case 1: {
        alch_scale *= (AlchBondedCUDA::alchLambdas).bondLambda1;
        if (doEnergy) {
          if (doTI) {
            energy_TI_1 += (double)(e);
          }
          if (doFEP) {
            energy_F += (double)(e * ((AlchBondedCUDA::alchLambdas).bondLambda12 - (AlchBondedCUDA::alchLambdas).bondLambda1));
          }
        }
        break;
      }
      case 2: {
        alch_scale *= (AlchBondedCUDA::alchLambdas).bondLambda2;
        if (doEnergy) {
          if (doTI) {
            energy_TI_2 += (double)(e);
          }
          if (doFEP) {
            energy_F += (double)(e * ((AlchBondedCUDA::alchLambdas).bondLambda22 - (AlchBondedCUDA::alchLambdas).bondLambda2));
          }
        }
        break;
      }
    }
  }

  if (doEnergy) {
    if (doFEP || doTI) {
      energy += e * alch_scale;
    } else {
      energy += e;
    }
  }

  //
  //     Compute derivatives wrt catesian coordinates.
  //
  // GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
  //  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)

  float fg = xij*xjk + yij*yjk + zij*zjk;
  float hg = xlk*xjk + ylk*yjk + zlk*zjk;
  ra2r *= df;
  rb2r *= df;
  float fga = fg*ra2r*rgr;
  float hgb = hg*rb2r*rgr;
  float gaa = ra2r*rg;
  float gbb = rb2r*rg;
  // DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.

  // Remember T is long long int
  // Don't try to scale T_fix and similar variables directly by float
  if (doFEP || doTI) {
    fga *= alch_scale;
    hgb *= alch_scale;
    gaa *= alch_scale;
    gbb *= alch_scale;
  }

  // Store forces
  T T_fix, T_fiy, T_fiz;
  calcComponentForces<T>(-gaa, ax, ay, az, T_fix, T_fiy, T_fiz);
  storeForces<T>(T_fix, T_fiy, T_fiz, dl.i, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);

  T dgx, dgy, dgz;
  calcComponentForces<T>(fga, ax, ay, az, -hgb, bx, by, bz, dgx, dgy, dgz);
  T T_fjx = dgx - T_fix;
  T T_fjy = dgy - T_fiy;
  T T_fjz = dgz - T_fiz;
  storeForces<T>(T_fjx, T_fjy, T_fjz, dl.j, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);

  T dhx, dhy, dhz;
  calcComponentForces<T>(gbb, bx, by, bz, dhx, dhy, dhz);
  T T_fkx = -dhx - dgx;
  T T_fky = -dhy - dgy;
  T T_fkz = -dhz - dgz;
  storeForces<T>(T_fkx, T_fky, T_fkz, dl.k, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);
  storeForces<T>(dhx, dhy, dhz, dl.l, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);

  // Store virial
  if (doVirial) {
#ifdef WRITE_FULL_VIRIALS
    float fix = ((float)T_fix);
    float fiy = ((float)T_fiy);
    float fiz = ((float)T_fiz);
    float fjx = ((float)dgx);
    float fjy = ((float)dgy);
    float fjz = ((float)dgz);
    float flx = ((float)dhx);
    float fly = ((float)dhy);
    float flz = ((float)dhz);
    virial.xx = (fix*xij) + (fjx*xjk) + (flx*xlk);
    virial.xy = (fix*yij) + (fjx*yjk) + (flx*ylk);
    virial.xz = (fix*zij) + (fjx*zjk) + (flx*zlk);
    virial.yx = (fiy*xij) + (fjy*xjk) + (fly*xlk);
    virial.yy = (fiy*yij) + (fjy*yjk) + (fly*ylk);
    virial.yz = (fiy*zij) + (fjy*zjk) + (fly*zlk);
    virial.zx = (fiz*xij) + (fjz*xjk) + (flz*xlk);
    virial.zy = (fiz*yij) + (fjz*yjk) + (flz*ylk);
    virial.zz = (fiz*zij) + (fjz*zjk) + (flz*zlk);
#endif
  }

}

__device__ __forceinline__ float3 cross(const float3 v1, const float3 v2) {
 return make_float3(
  v1.y*v2.z-v2.y*v1.z,
  v2.x*v1.z-v1.x*v2.z,
  v1.x*v2.y-v2.x*v1.y
  );
}

__device__ __forceinline__ float dot(const float3 v1, const float3 v2) {
  return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

//
// Calculates crossterms
//
template <typename T, bool doEnergy, bool doVirial, bool doFEP, bool doTI>
__device__ void crosstermForce(
  const int index,
  const CudaCrossterm* __restrict__ crosstermList,
  const CudaCrosstermValue* __restrict__ crosstermValues,
  const float4* __restrict__ xyzq,
  const int stride,
  const float3 lata, const float3 latb, const float3 latc,
  T* __restrict__ force, double &energy,
  T* __restrict__ forceList, int* forceListCounter, int* forceListStarts, int* forceListNexts,
#ifdef WRITE_FULL_VIRIALS
  ComputeBondedCUDAKernel::BondedVirial<double>& virial,
#else
  ComputeBondedCUDAKernel::BondedVirial* __restrict__ virial,
#endif
  double &energy_F, double &energy_TI_1, double &energy_TI_2
  ) {

  CudaCrossterm cl = crosstermList[index];

  // ----------------------------------------------------------------------------
  // Angle between 1 - 2 - 3 - 4
  //
  // 1 - 2
  float3 sh12 = make_float3(
    cl.offset12XYZ.x*lata.x + cl.offset12XYZ.y*latb.x + cl.offset12XYZ.z*latc.x,
    cl.offset12XYZ.x*lata.y + cl.offset12XYZ.y*latb.y + cl.offset12XYZ.z*latc.y,
    cl.offset12XYZ.x*lata.z + cl.offset12XYZ.y*latb.z + cl.offset12XYZ.z*latc.z);

  float4 xyzq1 = xyzq[cl.i1];
  float4 xyzq2 = xyzq[cl.i2];

  float3 r12 = make_float3(
    xyzq1.x + sh12.x - xyzq2.x,
    xyzq1.y + sh12.y - xyzq2.y,
    xyzq1.z + sh12.z - xyzq2.z);

  // 2 - 3
  float3 sh23 = make_float3(
    cl.offset23XYZ.x*lata.x + cl.offset23XYZ.y*latb.x + cl.offset23XYZ.z*latc.x,
    cl.offset23XYZ.x*lata.y + cl.offset23XYZ.y*latb.y + cl.offset23XYZ.z*latc.y,
    cl.offset23XYZ.x*lata.z + cl.offset23XYZ.y*latb.z + cl.offset23XYZ.z*latc.z);

  float4 xyzq3 = xyzq[cl.i3];

  float3 r23 = make_float3(
    xyzq2.x + sh23.x - xyzq3.x,
    xyzq2.y + sh23.y - xyzq3.y,
    xyzq2.z + sh23.z - xyzq3.z);

  // 3 - 4
  float3 sh34 = make_float3(
    cl.offset34XYZ.x*lata.x + cl.offset34XYZ.y*latb.x + cl.offset34XYZ.z*latc.x,
    cl.offset34XYZ.x*lata.y + cl.offset34XYZ.y*latb.y + cl.offset34XYZ.z*latc.y,
    cl.offset34XYZ.x*lata.z + cl.offset34XYZ.y*latb.z + cl.offset34XYZ.z*latc.z);

  float4 xyzq4 = xyzq[cl.i4];

  float3 r34 = make_float3(
    xyzq3.x + sh34.x - xyzq4.x,
    xyzq3.y + sh34.y - xyzq4.y,
    xyzq3.z + sh34.z - xyzq4.z);

  // Calculate the cross products
  float3 A = cross(r12, r23);
  float3 B = cross(r23, r34);
  float3 C = cross(r23, A);

  // Calculate the inverse distances
  float inv_rA = rsqrtf(dot(A, A));
  float inv_rB = rsqrtf(dot(B, B));
  float inv_rC = rsqrtf(dot(C, C));

  //  Calculate the sin and cos
  float cos_phi = dot(A, B)*(inv_rA*inv_rB);
  float sin_phi = dot(C, B)*(inv_rC*inv_rB);

  float phi = -atan2f(sin_phi,cos_phi);

  // ----------------------------------------------------------------------------
  // Angle between 5 - 6 - 7 - 8
  //

  // 5 - 6
  float3 sh56 = make_float3(
    cl.offset56XYZ.x*lata.x + cl.offset56XYZ.y*latb.x + cl.offset56XYZ.z*latc.x,
    cl.offset56XYZ.x*lata.y + cl.offset56XYZ.y*latb.y + cl.offset56XYZ.z*latc.y,
    cl.offset56XYZ.x*lata.z + cl.offset56XYZ.y*latb.z + cl.offset56XYZ.z*latc.z);

  float4 xyzq5 = xyzq[cl.i5];
  float4 xyzq6 = xyzq[cl.i6];

  float3 r56 = make_float3(
    xyzq5.x + sh56.x - xyzq6.x,
    xyzq5.y + sh56.y - xyzq6.y,
    xyzq5.z + sh56.z - xyzq6.z);

  // 6 - 7
  float3 sh67 = make_float3(
    cl.offset67XYZ.x*lata.x + cl.offset67XYZ.y*latb.x + cl.offset67XYZ.z*latc.x,
    cl.offset67XYZ.x*lata.y + cl.offset67XYZ.y*latb.y + cl.offset67XYZ.z*latc.y,
    cl.offset67XYZ.x*lata.z + cl.offset67XYZ.y*latb.z + cl.offset67XYZ.z*latc.z);

  float4 xyzq7 = xyzq[cl.i7];

  float3 r67 = make_float3(
    xyzq6.x + sh67.x - xyzq7.x,
    xyzq6.y + sh67.y - xyzq7.y,
    xyzq6.z + sh67.z - xyzq7.z);

  // 7 - 8
  float3 sh78 = make_float3(
    cl.offset78XYZ.x*lata.x + cl.offset78XYZ.y*latb.x + cl.offset78XYZ.z*latc.x,
    cl.offset78XYZ.x*lata.y + cl.offset78XYZ.y*latb.y + cl.offset78XYZ.z*latc.y,
    cl.offset78XYZ.x*lata.z + cl.offset78XYZ.y*latb.z + cl.offset78XYZ.z*latc.z);

  float4 xyzq8 = xyzq[cl.i8];

  float3 r78 = make_float3(
    xyzq7.x + sh78.x - xyzq8.x,
    xyzq7.y + sh78.y - xyzq8.y,
    xyzq7.z + sh78.z - xyzq8.z);

  // Calculate the cross products
  float3 D = cross(r56, r67);
  float3 E = cross(r67, r78);
  float3 F = cross(r67, D);
  
  // Calculate the inverse distances
  float inv_rD = rsqrtf(dot(D, D));
  float inv_rE = rsqrtf(dot(E, E));
  float inv_rF = rsqrtf(dot(F, F));

  //  Calculate the sin and cos
  float cos_psi = dot(D, E)*(inv_rD*inv_rE);
  float sin_psi = dot(F, E)*(inv_rF*inv_rE);

  float psi = -atan2f(sin_psi,cos_psi);

  // ----------------------------------------------------------------------------
  // Calculate the energy

  const float inv_h = (float)( (CudaCrosstermValue::dim) / (2.0*M_PI) );

  // Shift angles
  phi = fmod(phi + (float)M_PI, 2.0f*(float)M_PI);
  psi = fmod(psi + (float)M_PI, 2.0f*(float)M_PI);

  // distance measured in grid points between angle and smallest value
  float phi_h = phi * inv_h;
  float psi_h = psi * inv_h;

  // find smallest numbered grid point in stencil
  int iphi = (int)floor(phi_h);
  int ipsi = (int)floor(psi_h);

  const CudaCrosstermValue& cp = crosstermValues[cl.itype];

  // Load coefficients
  float4 c[4];
  c[0] = cp.c[iphi][ipsi][0];
  c[1] = cp.c[iphi][ipsi][1];
  c[2] = cp.c[iphi][ipsi][2];
  c[3] = cp.c[iphi][ipsi][3];

  float dphi = phi_h - (float)iphi;
  float dpsi = psi_h - (float)ipsi;

  float alch_scale;
  float alch_scale_energy_fep;
  double* p_energy_TI;
  if (doFEP || doTI) {
    alch_scale = 1.0f;
    alch_scale_energy_fep = 0.0f;
    if (doTI) p_energy_TI = NULL;
    switch (cl.fepBondedType) {
      case 1: {
        alch_scale *= (AlchBondedCUDA::alchLambdas).bondLambda1;
        if (doEnergy) {
          if (doTI) {
            p_energy_TI = &(energy_TI_1);
          }
          if (doFEP) {
            alch_scale_energy_fep = (AlchBondedCUDA::alchLambdas).bondLambda12 - (AlchBondedCUDA::alchLambdas).bondLambda1;
          }
        }
        break;
      }
      case 2: {
        alch_scale *= (AlchBondedCUDA::alchLambdas).bondLambda2;
        if (doEnergy) {
          if (doTI) {
            p_energy_TI = &(energy_TI_2);
          }
          if (doFEP) {
            alch_scale_energy_fep = (AlchBondedCUDA::alchLambdas).bondLambda22 - (AlchBondedCUDA::alchLambdas).bondLambda2;
          }
        }
        break;
      }
    }
  }

  if (doEnergy) {
    float energyf =          c[3].x + dphi*( c[3].y + dphi*( c[3].z + dphi*c[3].w ) );
    energyf = energyf*dpsi + c[2].x + dphi*( c[2].y + dphi*( c[2].z + dphi*c[2].w ) );
    energyf = energyf*dpsi + c[1].x + dphi*( c[1].y + dphi*( c[1].z + dphi*c[1].w ) );
    energyf = energyf*dpsi + c[0].x + dphi*( c[0].y + dphi*( c[0].z + dphi*c[0].w ) );
    if (doFEP || doTI) {
      energy += energyf * cl.scale * alch_scale;
      if (doFEP) {
        energy_F += energyf * cl.scale * alch_scale_energy_fep;
      }
      if (doTI) {
        if (p_energy_TI != NULL) {
          (*p_energy_TI) += energyf * cl.scale;
        }
      }
    } else {
      energy += energyf * cl.scale;
    }
  }

  float dEdphi =         3.0f*(c[0].w + dpsi*( c[1].w + dpsi*( c[2].w + dpsi*c[3].w ) ));
  dEdphi = dEdphi*dphi + 2.0f*(c[0].z + dpsi*( c[1].z + dpsi*( c[2].z + dpsi*c[3].z ) ));
  dEdphi = dEdphi*dphi +      (c[0].y + dpsi*( c[1].y + dpsi*( c[2].y + dpsi*c[3].y ) ));  // 13 muls
  dEdphi *= cl.scale*inv_h;

  float dEdpsi =         3.0f*(c[3].x + dphi*( c[3].y + dphi*( c[3].z + dphi*c[3].w ) ));
  dEdpsi = dEdpsi*dpsi + 2.0f*(c[2].x + dphi*( c[2].y + dphi*( c[2].z + dphi*c[2].w ) ));
  dEdpsi = dEdpsi*dpsi +      (c[1].x + dphi*( c[1].y + dphi*( c[1].z + dphi*c[1].w ) ));  // 13 muls
  dEdpsi *= cl.scale*inv_h;

  // float normCross1 = dot(A, A);
  float square_r23 = dot(r23, r23);
  float norm_r23 = sqrtf(square_r23);
  float inv_square_r23 = 1.0f/square_r23;
  float ff1 = dEdphi*norm_r23*inv_rA*inv_rA;
  float ff2 = -dot(r12, r23)*inv_square_r23;
  float ff3 = -dot(r34, r23)*inv_square_r23;
  float ff4 = -dEdphi*norm_r23*inv_rB*inv_rB;

  // NOTE: The reason why scaling ff1 and ff4 is enough:
  //       f1, f4 are already scaled, and in following t1's formula:
  //       first term (t1.x) : ff2*f1.x - ff3*f4.x
  //                                ^          ^
  //                               scaled     scaled
  //       so t1.x is scaled, and also t1.y and t1.z => t1 is scaled
  //       then let's look at f2.x:
  //       f2.x = t1.x - f1.x
  //               ^      ^
  //              scaled scaled => f2.x is scaled
  //       and also f2.y and f2.z are scaled => f2 is scaled => similarly f3 is scaled
  //       As a result scaling ff1 and ff4 is enough. DONT scale ff2 and ff3!
  if (doFEP || doTI) {
    ff1 *= alch_scale;
    ff4 *= alch_scale;
  }

  float3 f1 = make_float3(ff1*A.x, ff1*A.y, ff1*A.z);
  float3 f4 = make_float3(ff4*B.x, ff4*B.y, ff4*B.z);
  float3 t1 = make_float3( ff2*f1.x - ff3*f4.x, ff2*f1.y - ff3*f4.y, ff2*f1.z - ff3*f4.z );
  float3 f2 = make_float3(  t1.x - f1.x,  t1.y - f1.y,  t1.z - f1.z);
  float3 f3 = make_float3( -t1.x - f4.x, -t1.y - f4.y, -t1.z - f4.z);

  T T_f1x, T_f1y, T_f1z;
  T T_f2x, T_f2y, T_f2z;
  T T_f3x, T_f3y, T_f3z;
  T T_f4x, T_f4y, T_f4z;
  convertForces<T>(f1.x, f1.y, f1.z, T_f1x, T_f1y, T_f1z);
  convertForces<T>(f2.x, f2.y, f2.z, T_f2x, T_f2y, T_f2z);
  convertForces<T>(f3.x, f3.y, f3.z, T_f3x, T_f3y, T_f3z);
  convertForces<T>(f4.x, f4.y, f4.z, T_f4x, T_f4y, T_f4z);
  storeForces<T>(T_f1x, T_f1y, T_f1z, cl.i1, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);
  storeForces<T>(T_f2x, T_f2y, T_f2z, cl.i2, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);
  storeForces<T>(T_f3x, T_f3y, T_f3z, cl.i3, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);
  storeForces<T>(T_f4x, T_f4y, T_f4z, cl.i4, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);

  float square_r67 = dot(r67, r67);
  float norm_r67 = sqrtf(square_r67);
  float inv_square_r67 = 1.0f/(square_r67);
  ff1 = dEdpsi*norm_r67*inv_rD*inv_rD;
  ff2 = -dot(r56, r67)*inv_square_r67;
  ff3 = -dot(r78, r67)*inv_square_r67;
  ff4 = -dEdpsi*norm_r67*inv_rE*inv_rE;

  if (doFEP || doTI) {
    ff1 *= alch_scale;
    ff4 *= alch_scale;
  }

  float3 f5 = make_float3( ff1*D.x, ff1*D.y, ff1*D.z );
  float3 f8 = make_float3( ff4*E.x, ff4*E.y, ff4*E.z );
  float3 t2 = make_float3( ff2*f5.x - ff3*f8.x, ff2*f5.y - ff3*f8.y, ff2*f5.z - ff3*f8.z );
  float3 f6 = make_float3( t2.x - f5.x,  t2.y - f5.y,  t2.z - f5.z );
  float3 f7 = make_float3(-t2.x - f8.x, -t2.y - f8.y, -t2.z - f8.z );

  T T_f5x, T_f5y, T_f5z;
  T T_f6x, T_f6y, T_f6z;
  T T_f7x, T_f7y, T_f7z;
  T T_f8x, T_f8y, T_f8z;
  convertForces<T>(f5.x, f5.y, f5.z, T_f5x, T_f5y, T_f5z);
  convertForces<T>(f6.x, f6.y, f6.z, T_f6x, T_f6y, T_f6z);
  convertForces<T>(f7.x, f7.y, f7.z, T_f7x, T_f7y, T_f7z);
  convertForces<T>(f8.x, f8.y, f8.z, T_f8x, T_f8y, T_f8z);
  storeForces<T>(T_f5x, T_f5y, T_f5z, cl.i5, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);
  storeForces<T>(T_f6x, T_f6y, T_f6z, cl.i6, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);
  storeForces<T>(T_f7x, T_f7y, T_f7z, cl.i7, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);
  storeForces<T>(T_f8x, T_f8y, T_f8z, cl.i8, stride, force, forceList, forceListCounter, forceListStarts, forceListNexts);

  // Store virial
  if (doVirial) {
#ifdef WRITE_FULL_VIRIALS
    float3 s12 = make_float3( f1.x + f2.x, f1.y + f2.y, f1.z + f2.z );
    float3 s56 = make_float3( f5.x + f6.x, f5.y + f6.y, f5.z + f6.z );
    virial.xx = f1.x*r12.x + s12.x*r23.x - f4.x*r34.x + f5.x*r56.x + s56.x*r67.x - f8.x*r78.x;
    virial.xy = f1.x*r12.y + s12.x*r23.y - f4.x*r34.y + f5.x*r56.y + s56.x*r67.y - f8.x*r78.y;
    virial.xz = f1.x*r12.z + s12.x*r23.z - f4.x*r34.z + f5.x*r56.z + s56.x*r67.z - f8.x*r78.z;
    virial.yx = f1.y*r12.x + s12.y*r23.x - f4.y*r34.x + f5.y*r56.x + s56.y*r67.x - f8.y*r78.x;
    virial.yy = f1.y*r12.y + s12.y*r23.y - f4.y*r34.y + f5.y*r56.y + s56.y*r67.y - f8.y*r78.y;
    virial.yz = f1.y*r12.z + s12.y*r23.z - f4.y*r34.z + f5.y*r56.z + s56.y*r67.z - f8.y*r78.z;
    virial.zx = f1.z*r12.x + s12.z*r23.x - f4.z*r34.x + f5.z*r56.x + s56.z*r67.x - f8.z*r78.x;
    virial.zy = f1.z*r12.y + s12.z*r23.y - f4.z*r34.y + f5.z*r56.y + s56.z*r67.y - f8.z*r78.y;
    virial.zz = f1.z*r12.z + s12.z*r23.z - f4.z*r34.z + f5.z*r56.z + s56.z*r67.z - f8.z*r78.z;
  }
#endif

}

#ifndef NAMD_CUDA
#define BONDEDFORCESKERNEL_NUM_WARP 2
#else
#define BONDEDFORCESKERNEL_NUM_WARP 4
#endif
//
// Calculates all forces in a single kernel call
//
template <typename T, bool doEnergy, bool doVirial, bool doFEP, bool doTI>
__global__ void bondedForcesKernel(
  const int start,

  const int numBonds,
  const CudaBond* __restrict__ bonds,
  const CudaBondValue* __restrict__ bondValues,

  const int numAngles,
  const CudaAngle* __restrict__ angles,
  const CudaAngleValue* __restrict__ angleValues,

  const int numDihedrals,
  const CudaDihedral* __restrict__ dihedrals,
  const CudaDihedralValue* __restrict__ dihedralValues,

  const int numImpropers,
  const CudaDihedral* __restrict__ impropers,
  const CudaDihedralValue* __restrict__ improperValues,

  const int numExclusions,
  const CudaExclusion* __restrict__ exclusions,

  const int numCrossterms,
  const CudaCrossterm* __restrict__ crossterms,
  const CudaCrosstermValue* __restrict__ crosstermValues,

  const float cutoff2,
  const float r2_delta, const int r2_delta_expc,

  const float* __restrict__ r2_table,
  const float4* __restrict__ exclusionTable,
  
#if !defined(USE_TABLE_ARRAYS)  
  cudaTextureObject_t r2_table_tex,
  cudaTextureObject_t exclusionTableTex,
#endif

  const float4* __restrict__ xyzq,
  const int stride,
  const float3 lata, const float3 latb, const float3 latc,
  T* __restrict__ force,
  T* __restrict__ forceSlow,
  T* __restrict__ forceList,
  int* forceListCounter,
  int* forceListStarts,
  int* forceListStartsSlow,
  int* forceListNexts,
  double* __restrict__ energies_virials) {

  // Thread-block index
  int indexTB = start + blockIdx.x;

  const int numBondsTB     = (numBonds + blockDim.x - 1)/blockDim.x;
  const int numAnglesTB    = (numAngles + blockDim.x - 1)/blockDim.x;
  const int numDihedralsTB = (numDihedrals + blockDim.x - 1)/blockDim.x;
  const int numImpropersTB = (numImpropers + blockDim.x - 1)/blockDim.x;
  const int numExclusionsTB= (numExclusions + blockDim.x - 1)/blockDim.x;
  const int numCrosstermsTB= (numCrossterms + blockDim.x - 1)/blockDim.x;

  // Each thread computes single bonded interaction.
  // Each thread block computes single bonded type
  double energy;
  double energy_F;
  double energy_TI_1;
  double energy_TI_2;
  int energyIndex;
  int energyIndex_F;
  int energyIndex_TI_1;
  int energyIndex_TI_2;

  if (doEnergy) {
    energy = 0.0;
    energyIndex = -1;
    if (doFEP) {
      energy_F = 0.0;
      energyIndex_F = -1;
    }
    if (doTI) {
      energy_TI_1 = 0.0;
      energy_TI_2 = 0.0;
      energyIndex_TI_1 = -1;
      energyIndex_TI_2 = -1;
    }
  }

#ifdef WRITE_FULL_VIRIALS
  ComputeBondedCUDAKernel::BondedVirial<double> virial;
  int virialIndex;
  if (doVirial) {
    virial.xx = 0.0;
    virial.xy = 0.0;
    virial.xz = 0.0;
    virial.yx = 0.0;
    virial.yy = 0.0;
    virial.yz = 0.0;
    virial.zx = 0.0;
    virial.zy = 0.0;
    virial.zz = 0.0;
    virialIndex = ComputeBondedCUDAKernel::normalVirialIndex_XX;
  }
#endif

  if (indexTB < numBondsTB) {
    int i = threadIdx.x + indexTB*blockDim.x;
    if (doEnergy) {
      energyIndex = ComputeBondedCUDAKernel::energyIndex_BOND;
      if (doFEP) {
        energyIndex_F = ComputeBondedCUDAKernel::energyIndex_BOND_F;
      }
      if (doTI) {
        energyIndex_TI_1 = ComputeBondedCUDAKernel::energyIndex_BOND_TI_1;
        energyIndex_TI_2 = ComputeBondedCUDAKernel::energyIndex_BOND_TI_2;
      }
    }
    if (i < numBonds) {
      bondForce<T, doEnergy, doVirial, doFEP, doTI>
      (i, bonds, bondValues, xyzq,
        stride, lata, latb, latc,
        force, energy,
        forceList, forceListCounter, forceListStarts, forceListNexts,
        virial,
        energy_F, energy_TI_1, energy_TI_2);
    }
    goto reduce;
  }
  indexTB -= numBondsTB;

  if (indexTB < numAnglesTB) {
    int i = threadIdx.x + indexTB*blockDim.x;
    if (doEnergy) {
      energyIndex = ComputeBondedCUDAKernel::energyIndex_ANGLE;
      if (doFEP) {
        energyIndex_F = ComputeBondedCUDAKernel::energyIndex_ANGLE_F;
      }
      if (doTI) {
        energyIndex_TI_1 = ComputeBondedCUDAKernel::energyIndex_ANGLE_TI_1;
        energyIndex_TI_2 = ComputeBondedCUDAKernel::energyIndex_ANGLE_TI_2;
      }
    }
    if (i < numAngles) {
      angleForce<T, doEnergy, doVirial, doFEP, doTI>
      (i, angles, angleValues, xyzq, stride,
        lata, latb, latc,
        force, energy,
        forceList, forceListCounter, forceListStarts, forceListNexts,
        virial,
        energy_F, energy_TI_1, energy_TI_2);
    }
    goto reduce;
  }
  indexTB -= numAnglesTB;

  if (indexTB < numDihedralsTB) {
    int i = threadIdx.x + indexTB*blockDim.x;
    if (doEnergy) {
      energyIndex = ComputeBondedCUDAKernel::energyIndex_DIHEDRAL;
      if (doFEP) {
        energyIndex_F = ComputeBondedCUDAKernel::energyIndex_DIHEDRAL_F;
      }
      if (doTI) {
        energyIndex_TI_1 = ComputeBondedCUDAKernel::energyIndex_DIHEDRAL_TI_1;
        energyIndex_TI_2 = ComputeBondedCUDAKernel::energyIndex_DIHEDRAL_TI_2;
      }
    }
    if (doVirial) {
      virialIndex = ComputeBondedCUDAKernel::amdDiheVirialIndex_XX;
    }
    if (i < numDihedrals) {
      diheForce<T, doEnergy, doVirial, doFEP, doTI>
      (i, dihedrals, dihedralValues, xyzq, stride,
        lata, latb, latc,
        force, energy,
        forceList, forceListCounter, forceListStarts, forceListNexts,
        virial,
        energy_F, energy_TI_1, energy_TI_2);
    }
    goto reduce;
  }
  indexTB -= numDihedralsTB;

  if (indexTB < numImpropersTB) {
    int i = threadIdx.x + indexTB*blockDim.x;
    if (doEnergy) {
      energyIndex = ComputeBondedCUDAKernel::energyIndex_IMPROPER;
      if (doFEP) {
        energyIndex_F = ComputeBondedCUDAKernel::energyIndex_IMPROPER_F;
      }
      if (doTI) {
        energyIndex_TI_1 = ComputeBondedCUDAKernel::energyIndex_IMPROPER_TI_1;
        energyIndex_TI_2 = ComputeBondedCUDAKernel::energyIndex_IMPROPER_TI_2;
      }
    }
    if (i < numImpropers) {
      diheForce<T, doEnergy, doVirial, doFEP, doTI>
      (i, impropers, improperValues, xyzq, stride,
        lata, latb, latc,
        force, energy,
        forceList, forceListCounter, forceListStarts, forceListNexts,
        virial,
        energy_F, energy_TI_1, energy_TI_2);
    }
    goto reduce;
  }
  indexTB -= numImpropersTB;

  if (indexTB < numExclusionsTB) {
    int i = threadIdx.x + indexTB*blockDim.x;
    if (doEnergy) {
      energyIndex = ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW;
      if (doFEP) {
        energyIndex_F = ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_F;
      }
      if (doTI) {
        energyIndex_TI_1 = ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_TI_1;
        energyIndex_TI_2 = ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_TI_2;
      }
    }
    if (doVirial) {
      virialIndex = ComputeBondedCUDAKernel::slowVirialIndex_XX;
    }
    if (i < numExclusions) {
      exclusionForce<T, doEnergy, doVirial, doFEP, doTI>
      (i, exclusions, r2_delta, r2_delta_expc,
#if __CUDA_ARCH__ >= 350 || defined(NAMD_HIP) || defined(USE_TABLE_ARRAYS)
        r2_table, exclusionTable,
#else
        r2_table_tex, exclusionTableTex,
#endif
        xyzq, stride, lata, latb, latc, cutoff2,
        forceSlow, energy,
        forceList, forceListCounter, forceListStartsSlow, forceListNexts,
        virial,
        energy_F, energy_TI_1, energy_TI_2);
    }
    goto reduce;
  }
  indexTB -= numExclusionsTB;

  if (indexTB < numCrosstermsTB) {
    int i = threadIdx.x + indexTB*blockDim.x;
    if (doEnergy) {
      energyIndex = ComputeBondedCUDAKernel::energyIndex_CROSSTERM;
      if (doFEP) {
        energyIndex_F = ComputeBondedCUDAKernel::energyIndex_CROSSTERM_F;
      }
      if (doTI) {
        energyIndex_TI_1 = ComputeBondedCUDAKernel::energyIndex_CROSSTERM_TI_1;
        energyIndex_TI_2 = ComputeBondedCUDAKernel::energyIndex_CROSSTERM_TI_2;
      }
    }
    if (doVirial) {
      virialIndex = ComputeBondedCUDAKernel::amdDiheVirialIndex_XX;
    }
    if (i < numCrossterms) {
      crosstermForce<T, doEnergy, doVirial, doFEP, doTI>
      (i, crossterms, crosstermValues,
        xyzq, stride, lata, latb, latc,
        force, energy,
        forceList, forceListCounter, forceListStarts, forceListNexts,
        virial,
        energy_F, energy_TI_1, energy_TI_2);
    }
    goto reduce;
  }
  // indexTB -= numCrosstermsTB;

  reduce:

  // Write energies to global memory
  if (doEnergy) {
    // energyIndex is constant within thread-block
    __shared__ double shEnergy[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergy_TI_1[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergy_TI_2[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergy_F[BONDEDFORCESKERNEL_NUM_WARP];
#pragma unroll
    for (int i=WARPSIZE/2;i >= 1;i/=2) {
      energy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energy, i, WARPSIZE);

      if (doFEP) {
        energy_F += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energy_F, i, WARPSIZE);
      }
      if (doTI) {
        energy_TI_1 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energy_TI_1, i, WARPSIZE);
        energy_TI_2 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energy_TI_2, i, WARPSIZE);
      }
    }
    int laneID = (threadIdx.x & (WARPSIZE - 1));
    int warpID = threadIdx.x / WARPSIZE;
    if (laneID == 0) {
      shEnergy[warpID] = energy;
      if (doFEP) {
        shEnergy_F[warpID] = energy_F;
      }
      if (doTI) {
        shEnergy_TI_1[warpID] = energy_TI_1;
        shEnergy_TI_2[warpID] = energy_TI_2;
      }
    }
    BLOCK_SYNC;
    if (warpID == 0) {
      energy = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergy[laneID] : 0.0;
      if (doFEP) {
        energy_F = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergy_F[laneID] : 0.0;
      }
      if (doTI) {
        energy_TI_1 = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergy_TI_1[laneID] : 0.0;
        energy_TI_2 = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergy_TI_2[laneID] : 0.0;
      }
#pragma unroll
      for (int i=WARPSIZE/2;i >= 1;i/=2) {
        energy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energy, i, WARPSIZE);

        if (doFEP) {
          energy_F += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energy_F, i, WARPSIZE);
        }
        if (doTI) {
          energy_TI_1 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energy_TI_1, i, WARPSIZE);
          energy_TI_2 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energy_TI_2, i, WARPSIZE);
        }
      }
      if (laneID == 0) {
        const int bin = blockIdx.x % ATOMIC_BINS;
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + energyIndex], energy);
        if (doFEP) {
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + energyIndex_F], energy_F);
        }
        if (doTI) {
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + energyIndex_TI_1], energy_TI_1);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + energyIndex_TI_2], energy_TI_2);
        }
      }
    }
  }

  // Write virials to global memory
#ifdef WRITE_FULL_VIRIALS
  if (doVirial) {
#pragma unroll
    for (int i=WARPSIZE/2;i >= 1;i/=2) {
      virial.xx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.xx, i, WARPSIZE);
      virial.xy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.xy, i, WARPSIZE);
      virial.xz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.xz, i, WARPSIZE);
      virial.yx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.yx, i, WARPSIZE);
      virial.yy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.yy, i, WARPSIZE);
      virial.yz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.yz, i, WARPSIZE);
      virial.zx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.zx, i, WARPSIZE);
      virial.zy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.zy, i, WARPSIZE);
      virial.zz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.zz, i, WARPSIZE);
    }
    __shared__ ComputeBondedCUDAKernel::BondedVirial<double> shVirial[BONDEDFORCESKERNEL_NUM_WARP];
    int laneID = (threadIdx.x & (WARPSIZE - 1));
    int warpID = threadIdx.x / WARPSIZE;
    if (laneID == 0) {
      shVirial[warpID] = virial;
    }
    BLOCK_SYNC;

    if (warpID == 0) {
      virial.xx = 0.0;
      virial.xy = 0.0;
      virial.xz = 0.0;
      virial.yx = 0.0;
      virial.yy = 0.0;
      virial.yz = 0.0;
      virial.zx = 0.0;
      virial.zy = 0.0;
      virial.zz = 0.0;
      if (laneID < BONDEDFORCESKERNEL_NUM_WARP) virial = shVirial[laneID];
#pragma unroll
      for (int i=WARPSIZE/2;i >= 1;i/=2) {
        virial.xx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.xx, i, WARPSIZE);
        virial.xy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.xy, i, WARPSIZE);
        virial.xz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.xz, i, WARPSIZE);
        virial.yx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.yx, i, WARPSIZE);
        virial.yy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.yy, i, WARPSIZE);
        virial.yz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.yz, i, WARPSIZE);
        virial.zx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.zx, i, WARPSIZE);
        virial.zy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.zy, i, WARPSIZE);
        virial.zz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virial.zz, i, WARPSIZE);
      }   

      if (laneID == 0) {
        const int bin = blockIdx.x % ATOMIC_BINS;
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + virialIndex + 0], virial.xx);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + virialIndex + 1], virial.xy);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + virialIndex + 2], virial.xz);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + virialIndex + 3], virial.yx);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + virialIndex + 4], virial.yy);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + virialIndex + 5], virial.yz);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + virialIndex + 6], virial.zx);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + virialIndex + 7], virial.zy);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + virialIndex + 8], virial.zz);
      }
    }
  }
#endif

}

template <typename T, bool doEnergy, bool doVirial, bool doElect, bool doFEP, bool doTI, bool doTable>
__global__ void modifiedExclusionForcesKernel(
  const int start,
  const int numModifiedExclusions,
  const CudaExclusion* __restrict__ modifiedExclusions,
  const bool doSlow,
  const float one_scale14,                // 1 - scale14
  const float cutoff2,
  const CudaNBConstants nbConstants,
  const int vdwCoefTableWidth,
  const float2* __restrict__ vdwCoefTable,
#if !defined(USE_TABLE_ARRAYS)
  cudaTextureObject_t vdwCoefTableTex, 
#endif
#if defined(USE_TABLE_ARRAYS) && defined(NAMD_HIP)
  const float4* __restrict__ modifiedExclusionForceTable, 
  const float4* __restrict__ modifiedExclusionEnergyTable,
#else
  cudaTextureObject_t modifiedExclusionForceTableTex, 
  cudaTextureObject_t modifiedExclusionEnergyTableTex,
#endif
  const float4* __restrict__ xyzq,
  const int stride,
  const float3 lata, const float3 latb, const float3 latc,
  T* __restrict__ forceNbond, T* __restrict__ forceSlow,
  T* __restrict__ forceList,
  int* forceListCounter,
  int* forceListStartsNbond,
  int* forceListStartsSlow,
  int* forceListNexts,
  double* __restrict__ energies_virials) {

  // index
  int i = threadIdx.x + (start + blockIdx.x)*blockDim.x;

  double energyVdw, energyNbond, energySlow;
  // Alchemical energies
  double energyVdw_F, energyVdw_TI_1, energyVdw_TI_2;
  double energyNbond_F, energyNbond_TI_1, energyNbond_TI_2;
  double energySlow_F, energySlow_TI_1, energySlow_TI_2;
  if (doEnergy) {
    energyVdw = 0.0;
    if (doFEP) {
      energyVdw_F = 0.0;
    }
    if (doTI) {
      energyVdw_TI_1 = 0.0;
      energyVdw_TI_2 = 0.0;
    }
    if (doElect) {
      energyNbond = 0.0;
      energySlow = 0.0;
      if (doFEP) {
        energyNbond_F = 0.0;
        energySlow_F = 0.0;
      }
      if (doTI) {
        energyNbond_TI_1 = 0.0;
        energyNbond_TI_2 = 0.0;
        energySlow_TI_1 = 0.0;
        energySlow_TI_2 = 0.0;
      }
    }
  }

#ifdef WRITE_FULL_VIRIALS
  ComputeBondedCUDAKernel::BondedVirial<double> virialNbond;
  ComputeBondedCUDAKernel::BondedVirial<double> virialSlow;
  if (doVirial) {
    virialNbond.xx = 0.0;
    virialNbond.xy = 0.0;
    virialNbond.xz = 0.0;
    virialNbond.yx = 0.0;
    virialNbond.yy = 0.0;
    virialNbond.yz = 0.0;
    virialNbond.zx = 0.0;
    virialNbond.zy = 0.0;
    virialNbond.zz = 0.0;
    if (doElect) {
      virialSlow.xx = 0.0;
      virialSlow.xy = 0.0;
      virialSlow.xz = 0.0;
      virialSlow.yx = 0.0;
      virialSlow.yy = 0.0;
      virialSlow.yz = 0.0;
      virialSlow.zx = 0.0;
      virialSlow.zy = 0.0;
      virialSlow.zz = 0.0;
    }
  }
#endif

  if (i < numModifiedExclusions)
  {
    modifiedExclusionForce<T, doEnergy, doVirial, doElect, doFEP, doTI, doTable>
    (i, modifiedExclusions, doSlow, one_scale14, vdwCoefTableWidth,
#if __CUDA_ARCH__ >= 350 || defined(USE_TABLE_ARRAYS)
      vdwCoefTable,
#else
      vdwCoefTableTex,
#endif
      // for modified exclusions, we do regular tables if HIP and USE_FORCE_TABLES, otherwise it's textures
#if defined(USE_TABLE_ARRAYS) && defined(NAMD_HIP)
      // tables
      modifiedExclusionForceTable,
      modifiedExclusionEnergyTable,
#else
      // if CUDA build, non-tables, fall back to texture force/energy tables
      modifiedExclusionForceTableTex, 
      modifiedExclusionEnergyTableTex,
#endif
      xyzq, stride, lata, latb, latc, cutoff2, nbConstants,
      energyVdw, forceNbond, energyNbond,
      forceSlow, energySlow,
      forceList, forceListCounter, forceListStartsNbond, forceListStartsSlow, forceListNexts,
      virialNbond, virialSlow,
      energyVdw_F, energyVdw_TI_1, energyVdw_TI_2,
      energyNbond_F, energyNbond_TI_1, energyNbond_TI_2,
      energySlow_F, energySlow_TI_1, energySlow_TI_2);
  }

  // Write energies to global memory
  if (doEnergy) {
    __shared__ double shEnergyVdw[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergyNbond[(doElect) ? BONDEDFORCESKERNEL_NUM_WARP : 1];
    __shared__ double shEnergySlow[(doElect) ? BONDEDFORCESKERNEL_NUM_WARP : 1];
    __shared__ double shEnergyVdw_F[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergyVdw_TI_1[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergyVdw_TI_2[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergyNbond_F[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergyNbond_TI_1[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergyNbond_TI_2[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergySlow_F[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergySlow_TI_1[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ double shEnergySlow_TI_2[BONDEDFORCESKERNEL_NUM_WARP];
#pragma unroll
    for (int i=WARPSIZE/2;i >= 1;i/=2) {
      energyVdw   += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw, i, WARPSIZE);

      if (doFEP) {
        energyVdw_F += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw_F, i, WARPSIZE);
      }
      if (doTI) {
        energyVdw_TI_1 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw_TI_1, i, WARPSIZE);
        energyVdw_TI_2 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw_TI_2, i, WARPSIZE);
      }
      if (doElect) {
        energyNbond += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyNbond, i, WARPSIZE);
        energySlow  += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow, i, WARPSIZE);
        if (doFEP) {
          energyNbond_F += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyNbond_F, i, WARPSIZE);
          energySlow_F += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow_F, i, WARPSIZE);
        }
        if (doTI) {
          energyNbond_TI_1 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyNbond_TI_1, i, WARPSIZE);
          energyNbond_TI_2 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyNbond_TI_2, i, WARPSIZE);
          energySlow_TI_1 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow_TI_1, i, WARPSIZE);
          energySlow_TI_2 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow_TI_2, i, WARPSIZE);
        }
      }
    }
    int laneID = (threadIdx.x & (WARPSIZE - 1));
    int warpID = threadIdx.x / WARPSIZE;
    if (laneID == 0) {
      shEnergyVdw[warpID]   = energyVdw;
      if (doFEP) {
        shEnergyVdw_F[warpID] = energyVdw_F;
      }
      if (doTI) {
        shEnergyVdw_TI_1[warpID] = energyVdw_TI_1;
        shEnergyVdw_TI_2[warpID] = energyVdw_TI_2;
      }
      if (doElect) {
        shEnergyNbond[warpID] = energyNbond;
        shEnergySlow[warpID]  = energySlow;
        if (doFEP) {
          shEnergyNbond_F[warpID] = energyNbond_F;
          shEnergySlow_F[warpID] = energySlow_F;
        }
        if (doTI) {
          shEnergyNbond_TI_1[warpID] = energyNbond_TI_1;
          shEnergyNbond_TI_2[warpID] = energyNbond_TI_2;
          shEnergySlow_TI_1[warpID] = energySlow_TI_1;
          shEnergySlow_TI_2[warpID] = energySlow_TI_2;
        }
      }
    }
    BLOCK_SYNC;
    if (warpID == 0) {
      energyVdw   = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergyVdw[laneID] : 0.0;
      if (doFEP) {
        energyVdw_F = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergyVdw_F[laneID] : 0.0;
      }
      if (doTI) {
        energyVdw_TI_1 = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergyVdw_TI_1[laneID] : 0.0;
        energyVdw_TI_2 = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergyVdw_TI_2[laneID] : 0.0;
      }
      if (doElect) {
        energyNbond = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergyNbond[laneID] : 0.0;
        energySlow  = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergySlow[laneID] : 0.0;
        if (doFEP) {
          energyNbond_F = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergyNbond_F[laneID] : 0.0;
          energySlow_F = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergySlow_F[laneID] : 0.0;
        }
        if (doTI) {
          energyNbond_TI_1 = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergyNbond_TI_1[laneID] : 0.0;
          energyNbond_TI_2 = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergyNbond_TI_2[laneID] : 0.0;
          energySlow_TI_1 = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergySlow_TI_1[laneID] : 0.0;
          energySlow_TI_2 = (laneID < BONDEDFORCESKERNEL_NUM_WARP) ? shEnergySlow_TI_2[laneID] : 0.0;
        }
      }
#pragma unroll
      for (int i=WARPSIZE/2;i >= 1;i/=2) {
        energyVdw   += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw, i, WARPSIZE);

        if (doFEP) {
          energyVdw_F += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw_F, i, WARPSIZE);
        }
        if (doTI) {
          energyVdw_TI_1 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw_TI_1, i, WARPSIZE);
          energyVdw_TI_2 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw_TI_2, i, WARPSIZE);
        }
        if (doElect) {
          energyNbond += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyNbond, i, WARPSIZE);
          energySlow  += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow, i, WARPSIZE);
          if (doFEP) {
            energyNbond_F += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyNbond_F, i, WARPSIZE);
            energySlow_F += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow_F, i, WARPSIZE);
          }
          if (doTI) {
            energyNbond_TI_1 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyNbond_TI_1, i, WARPSIZE);
            energyNbond_TI_2 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyNbond_TI_2, i, WARPSIZE);
            energySlow_TI_1 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow_TI_1, i, WARPSIZE);
            energySlow_TI_2 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow_TI_2, i, WARPSIZE);
          }
        }
      }
      if (laneID == 0) {
        const int bin = blockIdx.x % ATOMIC_BINS;
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_LJ],         energyVdw);
        if (doFEP) {
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_LJ_F],     energyVdw_F);
        }
        if (doTI) {
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_LJ_TI_1],  energyVdw_TI_1);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_LJ_TI_2],  energyVdw_TI_2);
        }
        if (doElect) {
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_ELECT],      energyNbond);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW], energySlow);
          if (doFEP) {
            atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_ELECT_F],     energyNbond_F);
            atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_F],     energySlow_F);
          }
          if (doTI) {
            atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_ELECT_TI_1],  energyNbond_TI_1);
            atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_ELECT_TI_2],  energyNbond_TI_2);
            atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_TI_1],  energySlow_TI_1);
            atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::energyIndex_ELECT_SLOW_TI_2],  energySlow_TI_2);
          }
        }
      }
    }
  }

  // Write virials to global memory
#ifdef WRITE_FULL_VIRIALS
  if (doVirial) {
#pragma unroll
    for (int i=WARPSIZE/2;i >= 1;i/=2) {
      virialNbond.xx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.xx, i, WARPSIZE);
      virialNbond.xy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.xy, i, WARPSIZE);
      virialNbond.xz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.xz, i, WARPSIZE);
      virialNbond.yx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.yx, i, WARPSIZE);
      virialNbond.yy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.yy, i, WARPSIZE);
      virialNbond.yz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.yz, i, WARPSIZE);
      virialNbond.zx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.zx, i, WARPSIZE);
      virialNbond.zy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.zy, i, WARPSIZE);
      virialNbond.zz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.zz, i, WARPSIZE);
      if (doElect && doSlow) {
        virialSlow.xx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.xx, i, WARPSIZE);
        virialSlow.xy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.xy, i, WARPSIZE);
        virialSlow.xz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.xz, i, WARPSIZE);
        virialSlow.yx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.yx, i, WARPSIZE);
        virialSlow.yy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.yy, i, WARPSIZE);
        virialSlow.yz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.yz, i, WARPSIZE);
        virialSlow.zx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.zx, i, WARPSIZE);
        virialSlow.zy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.zy, i, WARPSIZE);
        virialSlow.zz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.zz, i, WARPSIZE);
      }
    }
    __shared__ ComputeBondedCUDAKernel::BondedVirial<double> shVirialNbond[BONDEDFORCESKERNEL_NUM_WARP];
    __shared__ ComputeBondedCUDAKernel::BondedVirial<double> shVirialSlow[(doElect) ? BONDEDFORCESKERNEL_NUM_WARP : 1];
    int laneID = (threadIdx.x & (WARPSIZE - 1));
    int warpID = threadIdx.x / WARPSIZE;
    if (laneID == 0) {
      shVirialNbond[warpID] = virialNbond;
      if (doElect && doSlow) {
        shVirialSlow[warpID] = virialSlow;
      }
    }
    BLOCK_SYNC;

    virialNbond.xx = 0.0;
    virialNbond.xy = 0.0;
    virialNbond.xz = 0.0;
    virialNbond.yx = 0.0;
    virialNbond.yy = 0.0;
    virialNbond.yz = 0.0;
    virialNbond.zx = 0.0;
    virialNbond.zy = 0.0;
    virialNbond.zz = 0.0;
    if (doElect && doSlow) {
      virialSlow.xx = 0.0;
      virialSlow.xy = 0.0;
      virialSlow.xz = 0.0;
      virialSlow.yx = 0.0;
      virialSlow.yy = 0.0;
      virialSlow.yz = 0.0;
      virialSlow.zx = 0.0;
      virialSlow.zy = 0.0;
      virialSlow.zz = 0.0;
    }

    if (warpID == 0) {
      if (laneID < BONDEDFORCESKERNEL_NUM_WARP) {
        virialNbond = shVirialNbond[laneID];
        if (doElect && doSlow) {
          virialSlow = shVirialSlow[laneID];
        }
      }
#pragma unroll
      for (int i=WARPSIZE/2;i >= 1;i/=2) {
        virialNbond.xx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.xx, i, WARPSIZE);
        virialNbond.xy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.xy, i, WARPSIZE);
        virialNbond.xz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.xz, i, WARPSIZE);
        virialNbond.yx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.yx, i, WARPSIZE);
        virialNbond.yy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.yy, i, WARPSIZE);
        virialNbond.yz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.yz, i, WARPSIZE);
        virialNbond.zx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.zx, i, WARPSIZE);
        virialNbond.zy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.zy, i, WARPSIZE);
        virialNbond.zz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialNbond.zz, i, WARPSIZE);
        if (doElect && doSlow) {
          virialSlow.xx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.xx, i, WARPSIZE);
          virialSlow.xy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.xy, i, WARPSIZE);
          virialSlow.xz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.xz, i, WARPSIZE);
          virialSlow.yx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.yx, i, WARPSIZE);
          virialSlow.yy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.yy, i, WARPSIZE);
          virialSlow.yz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.yz, i, WARPSIZE);
          virialSlow.zx += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.zx, i, WARPSIZE);
          virialSlow.zy += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.zy, i, WARPSIZE);
          virialSlow.zz += WARP_SHUFFLE_XOR(WARP_FULL_MASK, virialSlow.zz, i, WARPSIZE);
        }
      }

      if (laneID == 0)
      {
        const int bin = blockIdx.x % ATOMIC_BINS;
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::nbondVirialIndex_XX], virialNbond.xx);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::nbondVirialIndex_XY], virialNbond.xy);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::nbondVirialIndex_XZ], virialNbond.xz);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::nbondVirialIndex_YX], virialNbond.yx);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::nbondVirialIndex_YY], virialNbond.yy);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::nbondVirialIndex_YZ], virialNbond.yz);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::nbondVirialIndex_ZX], virialNbond.zx);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::nbondVirialIndex_ZY], virialNbond.zy);
        atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::nbondVirialIndex_ZZ], virialNbond.zz);
        if (doElect && doSlow) {
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::slowVirialIndex_XX], virialSlow.xx);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::slowVirialIndex_XY], virialSlow.xy);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::slowVirialIndex_XZ], virialSlow.xz);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::slowVirialIndex_YX], virialSlow.yx);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::slowVirialIndex_YY], virialSlow.yy);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::slowVirialIndex_YZ], virialSlow.yz);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::slowVirialIndex_ZX], virialSlow.zx);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::slowVirialIndex_ZY], virialSlow.zy);
          atomicAdd(&energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + ComputeBondedCUDAKernel::slowVirialIndex_ZZ], virialSlow.zz);
        }
      }
    }
  }
#endif

}

template <typename T>
__global__ void gatherBondedForcesKernel(
  const int start,
  const int atomStorageSize,
  const int stride,
  const T* __restrict__ forceList,
  const int* __restrict__ forceListStarts,
  const int* __restrict__ forceListNexts,
  T* __restrict__ force) {

  int i = threadIdx.x + (start + blockIdx.x) * blockDim.x;

  if (i < atomStorageSize) {
    T fx = 0;
    T fy = 0;
    T fz = 0;
    int pos = forceListStarts[i];
    while (pos != -1) {
      fx += forceList[pos * 3 + 0];
      fy += forceList[pos * 3 + 1];
      fz += forceList[pos * 3 + 2];
      pos = forceListNexts[pos];
    }
    force[i             ] = fx;
    force[i + stride    ] = fy;
    force[i + stride * 2] = fz;
  }
}

__global__ void reduceBondedBinsKernel(double* energies_virials) {
  const int bin = threadIdx.x;

  typedef cub::WarpReduce<double, (ATOMIC_BINS > 1 ? ATOMIC_BINS : 2)> WarpReduce;
  __shared__ typename WarpReduce::TempStorage tempStorage;

  double v = WarpReduce(tempStorage).Sum(energies_virials[bin * ComputeBondedCUDAKernel::energies_virials_SIZE + blockIdx.x]);
  if (threadIdx.x == 0) {
    energies_virials[blockIdx.x] = v;
  }
}

// ##############################################################################################
// ##############################################################################################
// ##############################################################################################

//
// Class constructor
//
ComputeBondedCUDAKernel::ComputeBondedCUDAKernel(int deviceID, CudaNonbondedTables& cudaNonbondedTables) :
deviceID(deviceID), cudaNonbondedTables(cudaNonbondedTables) {

  cudaCheck(cudaSetDevice(deviceID));

  tupleData = NULL;
  tupleDataSize = 0;

  bonds = NULL;
  angles = NULL;
  dihedrals = NULL;
  impropers = NULL;
  modifiedExclusions = NULL;
  exclusions = NULL;

  numBonds = 0;
  numAngles = 0;
  numDihedrals = 0;
  numImpropers = 0;
  numModifiedExclusions = 0;
  numExclusions = 0;
  numCrossterms = 0;

  bondValues = NULL;
  angleValues = NULL;
  dihedralValues = NULL;
  improperValues = NULL;
  crosstermValues = NULL;

  xyzq = NULL;
  xyzqSize = 0;

  forces = NULL;
  forcesSize = 0;

  forceList = NULL;
  forceListStarts = NULL;
  forceListNexts = NULL;
  forceListSize = 0;
  forceListStartsSize = 0;
  forceListNextsSize = 0;
  allocate_device<int>(&forceListCounter, 1);

  allocate_device<double>(&energies_virials, ATOMIC_BINS * energies_virials_SIZE);
}

//
// Class destructor
//
ComputeBondedCUDAKernel::~ComputeBondedCUDAKernel() {
  cudaCheck(cudaSetDevice(deviceID));

  deallocate_device<double>(&energies_virials);
  // deallocate_device<BondedVirial>(&virial);

  if (tupleData != NULL) deallocate_device<char>(&tupleData);
  if (xyzq != NULL) deallocate_device<float4>(&xyzq);
  if (forces != NULL) deallocate_device<FORCE_TYPE>(&forces);

  if (forceList != NULL) deallocate_device<FORCE_TYPE>(&forceList);
  if (forceListCounter != NULL) deallocate_device<int>(&forceListCounter);
  if (forceListStarts != NULL) deallocate_device<int>(&forceListStarts);
  if (forceListNexts != NULL) deallocate_device<int>(&forceListNexts);

  if (bondValues != NULL) deallocate_device<CudaBondValue>(&bondValues);
  if (angleValues != NULL) deallocate_device<CudaAngleValue>(&angleValues);
  if (dihedralValues != NULL) deallocate_device<CudaDihedralValue>(&dihedralValues);
  if (improperValues != NULL) deallocate_device<CudaDihedralValue>(&improperValues);
  if (crosstermValues != NULL) deallocate_device<CudaCrosstermValue>(&crosstermValues);
}

void ComputeBondedCUDAKernel::setupBondValues(int numBondValues, CudaBondValue* h_bondValues) {
  allocate_device<CudaBondValue>(&bondValues, numBondValues);
  copy_HtoD_sync<CudaBondValue>(h_bondValues, bondValues, numBondValues);
}

void ComputeBondedCUDAKernel::setupAngleValues(int numAngleValues, CudaAngleValue* h_angleValues) {
  allocate_device<CudaAngleValue>(&angleValues, numAngleValues);
  copy_HtoD_sync<CudaAngleValue>(h_angleValues, angleValues, numAngleValues);
}

void ComputeBondedCUDAKernel::setupDihedralValues(int numDihedralValues, CudaDihedralValue* h_dihedralValues) {
  allocate_device<CudaDihedralValue>(&dihedralValues, numDihedralValues);
  copy_HtoD_sync<CudaDihedralValue>(h_dihedralValues, dihedralValues, numDihedralValues);
}

void ComputeBondedCUDAKernel::setupImproperValues(int numImproperValues, CudaDihedralValue* h_improperValues) {
  allocate_device<CudaDihedralValue>(&improperValues, numImproperValues);
  copy_HtoD_sync<CudaDihedralValue>(h_improperValues, improperValues, numImproperValues);
}

void ComputeBondedCUDAKernel::setupCrosstermValues(int numCrosstermValues, CudaCrosstermValue* h_crosstermValues) {
  allocate_device<CudaCrosstermValue>(&crosstermValues, numCrosstermValues);
  copy_HtoD_sync<CudaCrosstermValue>(h_crosstermValues, crosstermValues, numCrosstermValues);
}

void ComputeBondedCUDAKernel::updateCudaAlchParameters(const CudaAlchParameters* h_cudaAlchParameters, cudaStream_t stream) {
  cudaCheck(cudaMemcpyToSymbolAsync(AlchBondedCUDA::alchParams, h_cudaAlchParameters, 1*sizeof(AlchBondedCUDA::alchParams), 0, cudaMemcpyHostToDevice, stream));
}

void ComputeBondedCUDAKernel::updateCudaAlchLambdas(const CudaAlchLambdas* h_cudaAlchLambdas, cudaStream_t stream) {
  cudaCheck(cudaMemcpyToSymbolAsync(AlchBondedCUDA::alchLambdas, h_cudaAlchLambdas, 1*sizeof(AlchBondedCUDA::alchLambdas), 0, cudaMemcpyHostToDevice, stream));
}

void ComputeBondedCUDAKernel::updateCudaAlchFlags(const CudaAlchFlags& h_cudaAlchFlags) {
  alchFlags = h_cudaAlchFlags;
}

//
// Update bonded lists
//
void ComputeBondedCUDAKernel::setTupleCounts(
  const TupleCounts count
) {
  numBonds              = count.bond;
  numAngles             = count.angle;
  numDihedrals          = count.dihedral;
  numImpropers          = count.improper;
  numModifiedExclusions = count.modifiedExclusion;
  numExclusions         = count.exclusion;
  numCrossterms         = count.crossterm;
}

TupleCounts ComputeBondedCUDAKernel::getTupleCounts() {
  TupleCounts count;

  count.bond              = numBonds;
  count.angle             = numAngles;
  count.dihedral          = numDihedrals;
  count.improper          = numImpropers;
  count.modifiedExclusion = numModifiedExclusions;
  count.exclusion         = numExclusions;
  count.crossterm         = numCrossterms;
  
  return count;
}

TupleData ComputeBondedCUDAKernel::getData() {
  TupleData data;
  data.bond = bonds;
  data.angle = angles;
  data.dihedral = dihedrals;
  data.improper = impropers;
  data.modifiedExclusion = modifiedExclusions;
  data.exclusion = exclusions;
  data.crossterm = crossterms;

  return data;
}

size_t ComputeBondedCUDAKernel::reallocateTupleBuffer(
  const TupleCounts countIn,
  cudaStream_t
) {
  const int numBondsWA              = warpAlign(countIn.bond);
  const int numAnglesWA             = warpAlign(countIn.angle);
  const int numDihedralsWA          = warpAlign(countIn.dihedral);
  const int numImpropersWA          = warpAlign(countIn.improper);
  const int numModifiedExclusionsWA = warpAlign(countIn.modifiedExclusion);
  const int numExclusionsWA         = warpAlign(countIn.exclusion);
  const int numCrosstermsWA         = warpAlign(countIn.crossterm);

  const size_t sizeTot = numBondsWA*sizeof(CudaBond) 
                    + numAnglesWA*sizeof(CudaAngle) 
                    + numDihedralsWA*sizeof(CudaDihedral)
                    + numImpropersWA*sizeof(CudaDihedral)
                    + numModifiedExclusionsWA*sizeof(CudaExclusion) 
                    + numExclusionsWA*sizeof(CudaExclusion) 
                    + numCrosstermsWA*sizeof(CudaCrossterm);

  reallocate_device<char>(&tupleData, &tupleDataSize, sizeTot, kTupleOveralloc);

  // Setup pointers
  size_t pos = 0;
  bonds = (CudaBond *)&tupleData[pos];
  pos += numBondsWA*sizeof(CudaBond);

  angles = (CudaAngle* )&tupleData[pos];
  pos += numAnglesWA*sizeof(CudaAngle);

  dihedrals = (CudaDihedral* )&tupleData[pos];
  pos += numDihedralsWA*sizeof(CudaDihedral);

  impropers = (CudaDihedral* )&tupleData[pos];
  pos += numImpropersWA*sizeof(CudaDihedral);

  modifiedExclusions = (CudaExclusion* )&tupleData[pos];
  pos += numModifiedExclusionsWA*sizeof(CudaExclusion);

  exclusions = (CudaExclusion* )&tupleData[pos];
  pos += numExclusionsWA*sizeof(CudaExclusion);

  crossterms = (CudaCrossterm* )&tupleData[pos];
  pos += numCrosstermsWA*sizeof(CudaCrossterm);

  return sizeTot;
}

void ComputeBondedCUDAKernel::update(
  const int numBondsIn,
  const int numAnglesIn,
  const int numDihedralsIn,
  const int numImpropersIn,
  const int numModifiedExclusionsIn,
  const int numExclusionsIn,
  const int numCrosstermsIn,
  const char* h_tupleData,
  cudaStream_t stream) {

  numBonds              = numBondsIn;
  numAngles             = numAnglesIn;
  numDihedrals          = numDihedralsIn;
  numImpropers          = numImpropersIn;
  numModifiedExclusions = numModifiedExclusionsIn;
  numExclusions         = numExclusionsIn;
  numCrossterms         = numCrosstermsIn;
  
  const size_t sizeTot = reallocateTupleBuffer(getTupleCounts(), stream); 

  copy_HtoD<char>(h_tupleData, tupleData, sizeTot, stream);
}

void ComputeBondedCUDAKernel::updateAtomBuffer(
  const int atomStorageSize,
  cudaStream_t stream
) {
  reallocate_device<float4>(&xyzq, &xyzqSize, atomStorageSize, 1.4f);
}

//
// Return stride for force array
//
int ComputeBondedCUDAKernel::getForceStride(const int atomStorageSize) {
#ifdef USE_STRIDED_FORCE
  // Align stride to 256 bytes
  return ((atomStorageSize*sizeof(FORCE_TYPE) - 1)/256 + 1)*256/sizeof(FORCE_TYPE);
#else
  return 1;
#endif
}

//
// Return size of single force array
//
int ComputeBondedCUDAKernel::getForceSize(const int atomStorageSize) {
#ifdef USE_STRIDED_FORCE
  return (3*getForceStride(atomStorageSize));
#else
  return (3*atomStorageSize);
#endif
}

//
// Return size of the all force arrays
//
int ComputeBondedCUDAKernel::getAllForceSize(const int atomStorageSize, const bool doSlow) {

  int forceSize = getForceSize(atomStorageSize);

  if (numModifiedExclusions > 0 || numExclusions > 0) {
    if (doSlow) {
      // All three force arrays [normal, nbond, slow]
      forceSize *= 3;
    } else {
      // Two force arrays [normal, nbond]
      forceSize *= 2;
    }
  }

  return forceSize;
}

//
// Compute bonded forces
//
void ComputeBondedCUDAKernel::bondedForce(
  const double scale14, const int atomStorageSize,
  const bool doEnergy, const bool doVirial, const bool doSlow,
  const bool doTable,
  const float3 lata, const float3 latb, const float3 latc,
  const float cutoff2, const float r2_delta, const int r2_delta_expc,
  const CudaNBConstants nbConstants,
  const float4* h_xyzq, FORCE_TYPE* h_forces, 
  double *h_energies_virials, bool atomsChanged, bool CUDASOAintegrate, 
  bool useDeviceMigration, cudaStream_t stream) {

  int forceStorageSize = getAllForceSize(atomStorageSize, true);
  int forceCopySize = getAllForceSize(atomStorageSize, doSlow);
  int forceStride = getForceStride(atomStorageSize);

  int forceSize   = getForceSize(atomStorageSize);
  int startNbond  = forceSize;
  int startSlow   = 2*forceSize;
  // Re-allocate coordinate and force arrays if neccessary
  reallocate_device<float4>(&xyzq, &xyzqSize, atomStorageSize, 1.4f);
  reallocate_device<FORCE_TYPE>(&forces, &forcesSize, forceStorageSize, 1.4f);

#if !defined(USE_BONDED_FORCE_ATOMIC_STORE)
  //                       function               stores
  // numBonds              bondForce              2
  // numAngles             angleForce             3
  // numDihedrals          diheForce              4
  // numImpropers          diheForce              4
  // numExclusions         exclusionForce         2
  // numCrossterms         crosstermForce         8
  // numModifiedExclusions modifiedExclusionForce 4
  int listSize = 3 * (numBonds * 2 + numAngles * 3 + numDihedrals * 4 + numImpropers * 4 + numExclusions * 2 + numCrossterms * 8 + numModifiedExclusions * 4);
  reallocate_device<FORCE_TYPE>(&forceList, &forceListSize, listSize, 1.4f);
  reallocate_device<int>(&forceListNexts, &forceListNextsSize, listSize, 1.4f);
  reallocate_device<int>(&forceListStarts, &forceListStartsSize, 3 * atomStorageSize, 1.4f);
  int* forceListStartsNbond = forceListStarts + atomStorageSize;
  int* forceListStartsSlow = forceListStarts + 2 * atomStorageSize;

  clear_device_array<int>(forceListCounter, 1, stream);
  cudaCheck(cudaMemsetAsync(forceListStarts, -1, sizeof(int) * 3 * atomStorageSize, stream));
#else
  int* forceListStartsNbond = NULL;
  int* forceListStartsSlow = NULL;
#endif

#ifdef NODEGROUP_FORCE_REGISTER
  if(CUDASOAintegrate){
    if(atomsChanged && !useDeviceMigration) copy_HtoD<float4>(h_xyzq, xyzq, atomStorageSize, stream);
  }else copy_HtoD<float4>(h_xyzq, xyzq, atomStorageSize, stream);
#else
  copy_HtoD<float4>(h_xyzq, xyzq, atomStorageSize, stream);
#endif
#if defined(USE_BONDED_FORCE_ATOMIC_STORE)
  clear_device_array<FORCE_TYPE>(forces, forceCopySize, stream);
#endif
  if (doEnergy || doVirial ) {
    clear_device_array<double>(energies_virials, ATOMIC_BINS * energies_virials_SIZE, stream);
  }

  // Check if we are doing alchemical free energy calculation
  // Is checking alchOn required?
  const bool doFEP  = (alchFlags.alchOn) && (alchFlags.alchFepOn);
  const bool doTI   = (alchFlags.alchOn) && (alchFlags.alchThermIntOn);

  float one_scale14 = (float)(1.0 - scale14);

  // If doSlow = false, these exclusions are not calculated
  int numExclusionsDoSlow = doSlow ? numExclusions : 0;

  int nthread =  BONDEDFORCESKERNEL_NUM_WARP * WARPSIZE;

  int numBondsTB     = (numBonds + nthread - 1)/nthread;
  int numAnglesTB    = (numAngles + nthread - 1)/nthread;
  int numDihedralsTB = (numDihedrals + nthread - 1)/nthread;
  int numImpropersTB = (numImpropers + nthread - 1)/nthread;
  int numExclusionsTB= (numExclusionsDoSlow + nthread - 1)/nthread;
  int numCrosstermsTB= (numCrossterms + nthread - 1)/nthread;

  int nblock = numBondsTB + numAnglesTB + numDihedralsTB + numImpropersTB + 
  numExclusionsTB + numCrosstermsTB;
  int shmem_size = 0;

  // printf("%d %d %d %d %d %d nblock %d\n",
  //   numBonds, numAngles, numDihedrals, numImpropers, numModifiedExclusions, numExclusions, nblock);

  int max_nblock = deviceCUDA->getMaxNumBlocks();

  int start = 0;
  while (start < nblock)
  {
    int nleft = nblock - start;
    int nblock_use = min(max_nblock, nleft);

#if defined(USE_TABLE_ARRAYS) && defined(NAMD_HIP)
  #define TABLE_PARAMS cudaNonbondedTables.get_r2_table(), \
    cudaNonbondedTables.getExclusionTable()
#else
  #define TABLE_PARAMS cudaNonbondedTables.get_r2_table_tex(), \
    cudaNonbondedTables.getExclusionTableTex()
#endif  

#if defined(USE_TABLE_ARRAYS)
#define CALL(DOENERGY, DOVIRIAL, DOFEP, DOTI) \
    bondedForcesKernel<FORCE_TYPE, DOENERGY, DOVIRIAL, DOFEP, DOTI> \
    <<< nblock_use, nthread, shmem_size, stream >>> \
    (start, numBonds, bonds, bondValues, \
    numAngles, angles, angleValues, \
    numDihedrals, dihedrals, dihedralValues, \
    numImpropers, impropers, improperValues, \
    numExclusionsDoSlow, exclusions, \
    numCrossterms, crossterms, crosstermValues, \
    cutoff2, \
    r2_delta, r2_delta_expc, \
    TABLE_PARAMS, \
    xyzq, forceStride, \
    lata, latb, latc, \
    forces, &forces[startSlow], \
    forceList, forceListCounter, forceListStarts, forceListStartsSlow, forceListNexts, \
    energies_virials);
#else
#define CALL(DOENERGY, DOVIRIAL, DOFEP, DOTI) \
  bondedForcesKernel<FORCE_TYPE, DOENERGY, DOVIRIAL, DOFEP, DOTI> \
  <<< nblock_use, nthread, shmem_size, stream >>> \
  (start, numBonds, bonds, bondValues, \
    numAngles, angles, angleValues, \
    numDihedrals, dihedrals, dihedralValues, \
    numImpropers, impropers, improperValues, \
    numExclusionsDoSlow, exclusions, \
    numCrossterms, crossterms, crosstermValues, \
    cutoff2, \
    r2_delta, r2_delta_expc, \
    cudaNonbondedTables.get_r2_table(), cudaNonbondedTables.getExclusionTable(), \
    cudaNonbondedTables.get_r2_table_tex(), cudaNonbondedTables.getExclusionTableTex(), \
    xyzq, forceStride, \
    lata, latb, latc, \
    forces, &forces[startSlow], \
    forceList, forceListCounter, forceListStarts, forceListStartsSlow, forceListNexts, \
    energies_virials);
#endif
    if (!doEnergy && !doVirial && !doFEP && !doTI) CALL(0, 0, 0, 0);
    if (!doEnergy &&  doVirial && !doFEP && !doTI) CALL(0, 1, 0, 0);
    if ( doEnergy && !doVirial && !doFEP && !doTI) CALL(1, 0, 0, 0);
    if ( doEnergy &&  doVirial && !doFEP && !doTI) CALL(1, 1, 0, 0);

    if (!doEnergy && !doVirial &&  doFEP && !doTI) CALL(0, 0, 1, 0);
    if (!doEnergy &&  doVirial &&  doFEP && !doTI) CALL(0, 1, 1, 0);
    if ( doEnergy && !doVirial &&  doFEP && !doTI) CALL(1, 0, 1, 0);
    if ( doEnergy &&  doVirial &&  doFEP && !doTI) CALL(1, 1, 1, 0);

    if (!doEnergy && !doVirial && !doFEP && doTI) CALL(0, 0, 0, 1);
    if (!doEnergy &&  doVirial && !doFEP && doTI) CALL(0, 1, 0, 1);
    if ( doEnergy && !doVirial && !doFEP && doTI) CALL(1, 0, 0, 1);
    if ( doEnergy &&  doVirial && !doFEP && doTI) CALL(1, 1, 0, 1);

    // Can't enable both FEP and TI, don't expand the following functions.
#if 0
    if (!doEnergy && !doVirial &&  doFEP && doTI) CALL(0, 0, 1, 1);
    if (!doEnergy &&  doVirial &&  doFEP && doTI) CALL(0, 1, 1, 1);
    if ( doEnergy && !doVirial &&  doFEP && doTI) CALL(1, 0, 1, 1);
    if ( doEnergy &&  doVirial &&  doFEP && doTI) CALL(1, 1, 1, 1);
#endif

#undef CALL
#undef TABLE_PARAMS
    cudaCheck(cudaGetLastError());

    start += nblock_use;
  }

  nthread = BONDEDFORCESKERNEL_NUM_WARP * WARPSIZE;
  nblock = (numModifiedExclusions + nthread - 1)/nthread;

  bool doElect = (one_scale14 == 0.0f) ? false : true;

  start = 0;
  while (start < nblock)
  {
    int nleft = nblock - start;
    int nblock_use = min(max_nblock, nleft);

#if defined(USE_TABLE_ARRAYS) && defined(NAMD_HIP)  
#define TABLE_PARAMS \
    cudaNonbondedTables.getExclusionVdwCoefTable(), \
    cudaNonbondedTables.getModifiedExclusionForceTable(), \
    cudaNonbondedTables.getModifiedExclusionEnergyTable()
#else 
#define TABLE_PARAMS \
    cudaNonbondedTables.getExclusionVdwCoefTable(), \
    cudaNonbondedTables.getExclusionVdwCoefTableTex(), \
    cudaNonbondedTables.getModifiedExclusionForceTableTex(), \
    cudaNonbondedTables.getModifiedExclusionEnergyTableTex()
#endif

#define CALL(DOENERGY, DOVIRIAL, DOELECT, DOFEP, DOTI, DOTABLE) \
  modifiedExclusionForcesKernel<FORCE_TYPE, DOENERGY, DOVIRIAL, DOELECT, DOFEP, DOTI, DOTABLE> \
  <<< nblock_use, nthread, shmem_size, stream >>> \
  (start, numModifiedExclusions, modifiedExclusions, \
    doSlow, one_scale14, cutoff2, nbConstants, \
    cudaNonbondedTables.getVdwCoefTableWidth(), \
    TABLE_PARAMS, \
    xyzq, forceStride, lata, latb, latc, \
    &forces[startNbond], &forces[startSlow], \
    forceList, forceListCounter, forceListStartsNbond, forceListStartsSlow, forceListNexts, \
    energies_virials);
    
    

    if (!doEnergy && !doVirial && !doElect && !doFEP && !doTI && doTable) CALL(0, 0, 0, 0, 0, 1);
    if (!doEnergy &&  doVirial && !doElect && !doFEP && !doTI && doTable) CALL(0, 1, 0, 0, 0, 1);
    if ( doEnergy && !doVirial && !doElect && !doFEP && !doTI && doTable) CALL(1, 0, 0, 0, 0, 1);
    if ( doEnergy &&  doVirial && !doElect && !doFEP && !doTI && doTable) CALL(1, 1, 0, 0, 0, 1);

    if (!doEnergy && !doVirial &&  doElect && !doFEP && !doTI && doTable) CALL(0, 0, 1, 0, 0, 1);
    if (!doEnergy &&  doVirial &&  doElect && !doFEP && !doTI && doTable) CALL(0, 1, 1, 0, 0, 1);
    if ( doEnergy && !doVirial &&  doElect && !doFEP && !doTI && doTable) CALL(1, 0, 1, 0, 0, 1);
    if ( doEnergy &&  doVirial &&  doElect && !doFEP && !doTI && doTable) CALL(1, 1, 1, 0, 0, 1);

    if (!doEnergy && !doVirial && !doElect &&  doFEP && !doTI && doTable) CALL(0, 0, 0, 1, 0, 1);
    if (!doEnergy &&  doVirial && !doElect &&  doFEP && !doTI && doTable) CALL(0, 1, 0, 1, 0, 1);
    if ( doEnergy && !doVirial && !doElect &&  doFEP && !doTI && doTable) CALL(1, 0, 0, 1, 0, 1);
    if ( doEnergy &&  doVirial && !doElect &&  doFEP && !doTI && doTable) CALL(1, 1, 0, 1, 0, 1);

    if (!doEnergy && !doVirial &&  doElect &&  doFEP && !doTI && doTable) CALL(0, 0, 1, 1, 0, 1);
    if (!doEnergy &&  doVirial &&  doElect &&  doFEP && !doTI && doTable) CALL(0, 1, 1, 1, 0, 1);
    if ( doEnergy && !doVirial &&  doElect &&  doFEP && !doTI && doTable) CALL(1, 0, 1, 1, 0, 1);
    if ( doEnergy &&  doVirial &&  doElect &&  doFEP && !doTI && doTable) CALL(1, 1, 1, 1, 0, 1);

    if (!doEnergy && !doVirial && !doElect && !doFEP &&  doTI && doTable) CALL(0, 0, 0, 0, 1, 1);
    if (!doEnergy &&  doVirial && !doElect && !doFEP &&  doTI && doTable) CALL(0, 1, 0, 0, 1, 1);
    if ( doEnergy && !doVirial && !doElect && !doFEP &&  doTI && doTable) CALL(1, 0, 0, 0, 1, 1);
    if ( doEnergy &&  doVirial && !doElect && !doFEP &&  doTI && doTable) CALL(1, 1, 0, 0, 1, 1);

    if (!doEnergy && !doVirial &&  doElect && !doFEP &&  doTI && doTable) CALL(0, 0, 1, 0, 1, 1);
    if (!doEnergy &&  doVirial &&  doElect && !doFEP &&  doTI && doTable) CALL(0, 1, 1, 0, 1, 1);
    if ( doEnergy && !doVirial &&  doElect && !doFEP &&  doTI && doTable) CALL(1, 0, 1, 0, 1, 1);
    if ( doEnergy &&  doVirial &&  doElect && !doFEP &&  doTI && doTable) CALL(1, 1, 1, 0, 1, 1);

    // doTable disabled is only supported by no FEP/ no TI
    if (!doEnergy && !doVirial && !doElect && !doFEP && !doTI && !doTable) CALL(0, 0, 0, 0, 0, 0);
    if (!doEnergy &&  doVirial && !doElect && !doFEP && !doTI && !doTable) CALL(0, 1, 0, 0, 0, 0);
    if ( doEnergy && !doVirial && !doElect && !doFEP && !doTI && !doTable) CALL(1, 0, 0, 0, 0, 0);
    if ( doEnergy &&  doVirial && !doElect && !doFEP && !doTI && !doTable) CALL(1, 1, 0, 0, 0, 0);

    if (!doEnergy && !doVirial &&  doElect && !doFEP && !doTI && !doTable) CALL(0, 0, 1, 0, 0, 0);
    if (!doEnergy &&  doVirial &&  doElect && !doFEP && !doTI && !doTable) CALL(0, 1, 1, 0, 0, 0);
    if ( doEnergy && !doVirial &&  doElect && !doFEP && !doTI && !doTable) CALL(1, 0, 1, 0, 0, 0);
    if ( doEnergy &&  doVirial &&  doElect && !doFEP && !doTI && !doTable) CALL(1, 1, 1, 0, 0, 0);

    // Can't enable both FEP and TI, don't expand the following functions.
#if 0
    if (!doEnergy && !doVirial && !doElect &&  doFEP &&  doTI) CALL(0, 0, 0, 1, 1);
    if (!doEnergy &&  doVirial && !doElect &&  doFEP &&  doTI) CALL(0, 1, 0, 1, 1);
    if ( doEnergy && !doVirial && !doElect &&  doFEP &&  doTI) CALL(1, 0, 0, 1, 1);
    if ( doEnergy &&  doVirial && !doElect &&  doFEP &&  doTI) CALL(1, 1, 0, 1, 1);

    if (!doEnergy && !doVirial &&  doElect &&  doFEP &&  doTI) CALL(0, 0, 1, 1, 1);
    if (!doEnergy &&  doVirial &&  doElect &&  doFEP &&  doTI) CALL(0, 1, 1, 1, 1);
    if ( doEnergy && !doVirial &&  doElect &&  doFEP &&  doTI) CALL(1, 0, 1, 1, 1);
    if ( doEnergy &&  doVirial &&  doElect &&  doFEP &&  doTI) CALL(1, 1, 1, 1, 1);
#endif

#undef CALL
    cudaCheck(cudaGetLastError());

    start += nblock_use;
  }
#if !defined(USE_BONDED_FORCE_ATOMIC_STORE)
  nthread = BONDEDFORCESKERNEL_NUM_WARP * WARPSIZE;
  nblock = (atomStorageSize + nthread - 1)/nthread;

  start = 0;
  while (start < nblock)
  {
    int nleft = nblock - start;
    int nblock_use = min(max_nblock, nleft);

    // cudaCheck(hipDeviceSynchronize());
    // auto t0 = std::chrono::high_resolution_clock::now();

    gatherBondedForcesKernel<FORCE_TYPE><<<nblock_use, nthread, 0, stream>>>(
      start, atomStorageSize, forceStride,
      forceList, forceListStarts, forceListNexts,
      forces);
    gatherBondedForcesKernel<FORCE_TYPE><<<nblock_use, nthread, 0, stream>>>(
      start, atomStorageSize, forceStride,
      forceList, forceListStartsNbond, forceListNexts,
      &forces[startNbond]);
    if (doSlow) {
      gatherBondedForcesKernel<FORCE_TYPE><<<nblock_use, nthread, 0, stream>>>(
        start, atomStorageSize, forceStride,
        forceList, forceListStartsSlow, forceListNexts,
        &forces[startSlow]);
    }
    cudaCheck(cudaGetLastError());

    // cudaCheck(hipStreamSynchronize(stream));
    // auto t1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff1 = t1 - t0;
    // std::cout << "gatherBondedForcesKernel";
    // std::cout << " " << std::setprecision(3) << diff1.count() * 1e3 << " ms" << std::endl;

    start += nblock_use;
  }
#endif

#ifdef NODEGROUP_FORCE_REGISTER
  if((atomsChanged && !useDeviceMigration) || !CUDASOAintegrate){ 
    copy_DtoH<double>(forces, h_forces, forceCopySize, stream);    
#if 0
      // XXX TODO: ERASE THIS AFTERWARDS
      // this is not numAtoms, this is something else
      // will print the force inside the compute and afterwards
      FILE* pos_nb_atoms = fopen("compute_b_dforce.txt", "w");
      //fprintf(pos_nb_atoms, "Calculating %d positions\n", tlKernel.getCudaPatchesSize());
      // positions are wrong here.
      //for(int i = 0; i < atomStorageSize; i++){
      for(int i = 29415; i < 29895; i++){
        fprintf(pos_nb_atoms, "%2.4lf\n", h_forcesDP[i]);
      }
      fclose(pos_nb_atoms);
#endif
  }
  
#else
  copy_DtoH<double>(forces, h_forces, forceCopySize, stream);
#endif
  if (doEnergy || doVirial) {
    if (ATOMIC_BINS > 1) {
      // Reduce energies_virials[ATOMIC_BINS][energies_virials_SIZE] in-place (results are in energies_virials[0])
      reduceBondedBinsKernel<<<energies_virials_SIZE, ATOMIC_BINS, 0, stream>>>(energies_virials);
    }
    // virial copy, is this necessary?
    copy_DtoH<double>(energies_virials, h_energies_virials, energies_virials_SIZE, stream);
  }
}


