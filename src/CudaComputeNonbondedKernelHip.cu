#if defined(NAMD_HIP)
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#define cub hipcub
#endif // NAMD_HIP

#include "HipDefines.h"
#include "CudaComputeNonbondedKernel.hip.h"
#include "CudaTileListKernel.hip.h"
#include "DeviceCUDA.h"
#include "CudaComputeNonbondedInteractions.h"

#if defined(NAMD_HIP)

#ifdef WIN32
#define __thread __declspec(thread)
#endif
extern __thread DeviceCUDA *deviceCUDA;

#define NONBONDKERNEL_NUM_WARP 1
#define REDUCENONBONDEDVIRIALKERNEL_NUM_WARP 4
#define REDUCEVIRIALENERGYKERNEL_NUM_WARP 4
#define REDUCEGBISENERGYKERNEL_NUM_WARP 4
#define __ldg *
#define NONBONDKERNEL_SWEEPS_PER_TILE WARPSIZE/BOUNDINGBOXSIZE

#define OVERALLOC 1.2f

void NAMD_die(const char *);
void NAMD_bug(const char *);

#define MAX_CONST_EXCLUSIONS 2048  // cache size is 8k
__constant__ unsigned int constExclusions[MAX_CONST_EXCLUSIONS];

//FEP parameters
__constant__ AlchData alchflags;

struct fast_float3
{
    typedef float __attribute__((ext_vector_type(2))) Native_float2_;

    union
    {
        struct __attribute__((packed)) { Native_float2_ dxy; float dz; };
        struct { float x, y, z; };
    };

    __host__ __device__
    fast_float3() = default;

    __host__ __device__
    fast_float3(float x_, float y_, float z_) : dxy{ x_, y_ }, dz{ z_ } {}

    __host__ __device__
    fast_float3(Native_float2_ xy_, float z_) : dxy{ xy_ }, dz{ z_ } {}

    __host__ __device__
    operator float3() const
    {
        return float3{ x, y, z };
    }

    __forceinline__ __host__ __device__
    fast_float3& operator=(float4 y)
    {
      this->x = y.x;
      this->y = y.y;
      this->z = y.z;
      return *this;
    }
};
static_assert(sizeof(fast_float3) == 12);

__forceinline__ __host__ __device__
fast_float3 operator*(fast_float3 x, fast_float3 y)
{
    return fast_float3{ x.dxy * y.dxy, x.dz * y.dz };
}

__forceinline__ __host__ __device__
fast_float3 operator*(fast_float3 x, float y)
{
    return fast_float3{ x.dxy * y, x.dz * y };
}

__forceinline__ __host__ __device__
fast_float3 operator*(float x, fast_float3 y)
{
    return fast_float3{ x * y.dxy, x * y.dz };
}

__forceinline__ __host__ __device__
fast_float3 operator+(fast_float3 x, fast_float3 y)
{
    return fast_float3{ x.dxy + y.dxy, x.dz + y.dz };
}

__forceinline__ __host__ __device__
fast_float3 operator-(fast_float3 x, fast_float3 y)
{
    return fast_float3{ x.dxy - y.dxy, x.dz - y.dz };
}


static __forceinline__ __host__ __device__ fast_float3 make_fast_float3(const float x)
{
    return fast_float3{ x, x, x };
}

static __forceinline__ __host__ __device__ fast_float3 make_fast_float3(const float3 x)
{
    return fast_float3{ x.x, x.y, x.z };
}

static __forceinline__ __host__ __device__ fast_float3 make_fast_float3(const float4 x)
{
    return fast_float3{ x.x, x.y, x.z };
}

static __forceinline__ __host__ __device__ float norm2(fast_float3 a)
{
    fast_float3 b = a * a;
    return (b.x + b.y + b.z);
}

static __forceinline__ __host__ __device__ float4 make_float4(fast_float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.f);
}


#ifndef USE_TABLE_ARRAYS
__device__ __forceinline__
float4 sampleTableTex(cudaTextureObject_t tex, float k) {
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
}

__device__ __forceinline__
float4 sampleTableTexInv(cudaTextureObject_t tex, float k) {
  const int tableSize = FORCE_ENERGY_TABLE_SIZE;
  const float x = k * (float)tableSize - 0.5f;
  const float f = floorf(x);
  const float a = x - f;
  const unsigned int i = (unsigned int)f;
  const int i0 = i < tableSize - 1 ? i : tableSize - 1;
  const int i1 = i0 + 1;
  const float4 t0 = tex1Dfetch<float4>(tex, i0);
  const float4 t1 = tex1Dfetch<float4>(tex, i1);
  float4 t2 = (t1 - t0) * a + t0;
  return make_float4(t2.z, t2.y, t2.x, t2.w);
}

#endif



// HIP implementation of tex1D has lower performance than this custom implementation of
// linear filtering

//#define USE_TABLE_ARRAYS

// Fast loads here
template<typename T>
static __forceinline__ __device__ const T& fast_load(const T* buffer, unsigned int idx, unsigned int offset = 0)
{
    return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T)) + offset * static_cast<unsigned int>(sizeof(T)));
}

__device__ __forceinline__
float4 tableLookup(const float4* table, const float k)
{
  const int tableSize = FORCE_ENERGY_TABLE_SIZE;
  const float x = k * static_cast<float>(tableSize) - 0.5f;
  const float f = floorf(x);
  const float a = x - f;
  const int i = static_cast<int>(f);
  const int i0 = i < tableSize - 1 ? i : tableSize - 1;
  const int i1 = i0 + 1;
  // const float4 t0 = __ldg(&table[i0]);
  // const float4 t1 = __ldg(&table[i1]);
  const float4 t0 = fast_load(table, i0);
  const float4 t1 = fast_load(table, i1);
  return make_float4(
    a * (t1.x - t0.x) + t0.x,
    a * (t1.y - t0.y) + t0.y,
    a * (t1.z - t0.z) + t0.z,
    a * (t1.w - t0.w) + t0.w);
}

__device__ __forceinline__
float4 tableLookupInv(const float4* table, const float k)
{
  constexpr int tableSize = FORCE_ENERGY_TABLE_SIZE;
  const float x = k * static_cast<float>(tableSize) - 0.5f;
  const float f = floorf(x);
  const float a = x - f;
  const int i = static_cast<int>(f);
  const int i0 = i < tableSize - 1 ? i : tableSize - 1;
  const int i1 = i0 + 1;
  // const float4 t0 = __ldg(&table[i0]);
  // const float4 t1 = __ldg(&table[i1]);
  const float4 t0 = fast_load(table, i0);
  const float4 t1 = fast_load(table, i1);
  float4 t2 = (t1 - t0) * a + t0;
  return make_float4(t2.z, t2.y, t2.x, t2.w);
}


template<bool doEnergy, bool doSlow>
__device__ __forceinline__
void calcForceEnergy(const float r2, const float qi, const float qj,
  const float dx, const float dy, const float dz,
  const int vdwtypei, const int vdwtypej, 
#ifdef USE_TABLE_ARRAYS
  const float2* __restrict__ vdwCoefTable,
  const float4* __restrict__ forceTable, 
  const float4* __restrict__ energyTable,
#else
  cudaTextureObject_t vdwCoefTableTex, 
  cudaTextureObject_t forceTableTex, 
  cudaTextureObject_t energyTableTex,
#endif
  fast_float3& iforce, fast_float3& iforceSlow, fast_float3& jforce, fast_float3& jforceSlow,
  float& energyVdw, float& energyElec, float& energySlow) {

  int vdwIndex = vdwtypej + vdwtypei;

#if defined(USE_TABLE_ARRAYS)
  // float2 ljab = vdwCoefTable[vdwIndex];
  float2 ljab = fast_load(vdwCoefTable, vdwIndex);
#else 
  float2 ljab = tex1Dfetch<float2>(vdwCoefTableTex, vdwIndex);
#endif
  float rinv = __frsqrt_rn(r2);
  float4 ei;

#if defined(USE_TABLE_ARRAYS)
  float4 fi = tableLookupInv(forceTable, rinv);
  if (doEnergy) ei = tableLookup(energyTable, rinv);
#else
   float4 fi = sampleTableTexInv(forceTableTex, rinv);
   if (doEnergy) ei = sampleTableTex(energyTableTex, rinv);
#endif

  float fSlow = qi * qj;
  fast_float3 f3(fi.x, fi.y, fi.z); // no waster registers, we already allocated stuff
  fast_float3 lj(ljab.x, ljab.y, fSlow);
  // we want to operate on reverse data here, the order of the float4 needs to be 
  // fast_float3 fi = f.z, f.y, f.x;
  // fast_float3 lj   = ljab.x, ljab.y, fSlow
  // so if we do fi = fi * lj;
  // f is the norm(fi) // x y z
  // fSlow = fi.w
  // float f = ljab.x * fi.z + ljab.y * fi.y + fSlow * fi.x;
  f3 = f3 * lj;
  float f = f3.x + f3.y + f3.z;

  if (doEnergy) {
    energyVdw  += ljab.x * ei.z + ljab.y * ei.y;
    energyElec += fSlow * ei.x;
    if (doSlow) energySlow += fSlow * ei.w;
  }
  if (doSlow) fSlow *= fi.w;

  // dx, dy, and dz are already store in contiguous registers -> wrap them in a fast_float3 and operate
  fast_float3 dr(dx, dy, dz);
  fast_float3 fr = dr *f;
  // float fx = dx * f;
  // float fy = dy * f;
  // float fz = dz * f;
  // iforce.x += fr.x;
  // iforce.y += fr.y;
  // iforce.z += fr.z;
  // jforce.x -= fr.x;
  // jforce.y -= fr.y;
  // jforce.z -= fr.z;
  iforce = iforce + fr;
  jforce = jforce - fr;

  if (doSlow) {
    fast_float3 sr = dr*fSlow;
    // float fxSlow = dx * fSlow;
    // float fySlow = dy * fSlow;
    // float fzSlow = dz * fSlow;
    // iforceSlow.x += sr.x;
    // iforceSlow.y += sr.y;
    // iforceSlow.z += sr.z;
    // jforceSlow.x -= sr.x;
    // jforceSlow.y -= sr.y;
    // jforceSlow.z -= sr.z;
    iforceSlow = iforceSlow + sr;
    jforceSlow = jforceSlow - sr;
  }
}

template<bool doEnergy, bool doSlow>
__device__ __forceinline__
void calcForceEnergyMath(const float r2, const float qi, const float qj,
  const float dx, const float dy, const float dz,
  const int vdwtypei, const int vdwtypej, 
#if defined(USE_TABLE_ARRAYS)
  const float2* __restrict__ vdwCoefTable,
#else
  cudaTextureObject_t vdwCoefTableTex, 
#endif
  fast_float3& iforce, fast_float3& iforceSlow, fast_float3& jforce, fast_float3& jforceSlow,
  float& energyVdw, float& energyElec, float& energySlow,
  const CudaNBConstants nbConstants) {

  int vdwIndex = vdwtypej + vdwtypei;

#if defined(USE_TABLE_ARRAYS)
  float2 ljab = vdwCoefTable[vdwIndex];
#else 
  float2 ljab = tex1Dfetch<float2>(vdwCoefTableTex, vdwIndex);
#endif
 

  float rinv = rsqrtf(r2);
  float f, fSlow;
  float charge = qi * qj;

  cudaNBForceMagCalc_VdwEnergySwitch_PMEC1<doEnergy, doSlow>(
    r2, rinv, charge, ljab, nbConstants, 
    f, fSlow, energyVdw, energyElec, energySlow);  
  // dx, dy, and dz are already store in contiguous registers -> wrap them in a fast_float3 and operate
  fast_float3 ff(dx, dy, dz);

  ff = ff * f;
  // float fx = dx * f;
  // float fy = dy * f;
  // float fz = dz * f;
  fast_float3 iff = make_fast_float3(iforce);
  fast_float3 jff = make_fast_float3(jforce);
  iff =  iff + ff;
  iforce.x = iff.x;
  iforce.y = iff.y;
  iforce.z = iff.z;
  // iforce.x += fx;
  // iforce.y += fy;
  // iforce.z += fz;
  jff = jff - ff;
  jforce.x = jff.x;
  jforce.y = jff.y;
  jforce.z = jff.z;
  // jforce.x -= fx;
  // jforce.y -= fy;
  // jforce.z -= fz;
  if (doSlow) {
    float fxSlow = dx * fSlow;
    float fySlow = dy * fSlow;
    float fzSlow = dz * fSlow;
    iforceSlow.x += fxSlow;
    iforceSlow.y += fySlow;
    iforceSlow.z += fzSlow;
    jforceSlow.x -= fxSlow;
    jforceSlow.y -= fySlow;
    jforceSlow.z -= fzSlow;
  }
}

/* JM: Special __device__ function to compute VDW forces for alchemy. 
 * Partially swiped from ComputeNonbondedFEP.C
 */
template<bool doEnergy, bool doSlow, bool shift, bool vdwForceSwitch>
__device__ __forceinline__
void calcForceEnergyFEP(const float r2, const float qi, const float qj,
  const float dx, const float dy, const float dz,
  const int vdwtypei, const int vdwtypej, 
  char p1, char p2,
  /*const AlchData &alchflags, */
 #ifdef USE_TABLE_ARRAYS
  const float2* __restrict__ vdwCoefTable, 
  const float4* __restrict__ forceTable, 
  const float4* __restrict__ energyTable,
#else
  cudaTextureObject_t vdwCoefTableTex,
  cudaTextureObject_t forceTableTex, 
  cudaTextureObject_t energyTableTex,
#endif
  fast_float3& iforce, fast_float3& iforceSlow, fast_float3& jforce, fast_float3& jforceSlow,
  float& energyVdw, float &energyVdw_s, float& energyElec, float& energySlow, 
  float& energyElec_s, float& energySlow_s) {
  

  int vdwIndex = vdwtypej + vdwtypei;
#if defined(USE_TABLE_ARRAYS)
  float2 ljab = fast_load(vdwCoefTable, vdwIndex);
#else
  float2 ljab = tex1Dfetch<float2>(vdwCoefTableTex, vdwIndex); //ljab.x is A and ljab.y is B
#endif
  
  float myVdwLambda = 0.0f;
  float myVdwLambda2 = 0.0f;
  float myElecLambda = 0.0f;
  float myElecLambda2 = 0.0f;
  float rinv = __frsqrt_rn(r2);
  float f;
  float alch_vdw_energy = 0.0f;
  float alch_vdw_energy_2 = 0.0f;
  float alch_vdw_force = 0.0f;
  float fSlow = qi * qj;
  float4 ei;
 
#ifdef USE_TABLE_ARRAYS
  float4 fi = tableLookup(forceTable, rinv);
  if (doEnergy) ei = tableLookup(energyTable, rinv);
#else
   float4 fi = sampleTableTex(forceTableTex, rinv);
   if (doEnergy) ei = sampleTableTex(energyTableTex, rinv);
#endif


  //John said that there is a better way to avoid divergences here
  //alch: true if => 1-0, 1-1, 2-0, 2-2
  //dec:  true if => 1-1, 2-2 && decouple
  //up: true if => 1-0 && 1,1
  //down: true if => 2-0, && 2,2
  int ref  = (p1 == 0 && p2 == 0);
  int alch = (!ref && !(p1 == 1  && p2 ==2) && !(p1 == 2 && p2 == 1));
  int dec  = (alch &&  (p1 == p2) && alchflags.alchDecouple);
  int up   = (alch &&  (p1 == 1 || p2 == 1) && !dec);
  int down = (alch &&  (p1 == 2 || p2 == 2) && !dec);
  
  float r2_1, r2_2;
  f = (fSlow * fi.x);

/*---------------   VDW SPECIAL ALCH FORCES (Swiped from ComputeNonbondedFEP.C)  ---------------*/

  myVdwLambda   = alchflags.vdwLambdaUp*(up)   + alchflags.vdwLambdaDown*(down)   + 1.f*(ref || dec);
  myVdwLambda2  = alchflags.vdwLambda2Up*(up)  + alchflags.vdwLambda2Down*(down)  + 1.f*(ref || dec);
  myElecLambda  = alchflags.elecLambdaUp*(up)  + alchflags.elecLambdaDown*(down)  + 1.f*(ref || dec);
  myElecLambda2 = alchflags.elecLambda2Up*(up) + alchflags.elecLambda2Down*(down) + 1.f*(ref || dec);

  if (alch) {
    if (vdwForceSwitch) {
      // force switching
      float switchdist6_1, switchdist6_2;
      const float cutoff6 = alchflags.cutoff2 * alchflags.cutoff2 * alchflags.cutoff2;
      // const float
      //Templated parameter. No control divergence here
      if (shift) {
        const float myVdwShift = alchflags.vdwShiftUp*up + alchflags.vdwShiftDown*(!up);
        const float myVdwShift2 = alchflags.vdwShift2Up*up + alchflags.vdwShift2Down*(!up);
        r2_1 = __fdividef(1.f,(r2 + myVdwShift));
        r2_2 = __fdividef(1.f,(r2 + myVdwShift2));
        switchdist6_1 = alchflags.switchdist2 + myVdwShift;
        switchdist6_1 = switchdist6_1 * switchdist6_1 * switchdist6_1;
        switchdist6_2 = alchflags.switchdist2 + myVdwShift2;
        switchdist6_2 = switchdist6_2 * switchdist6_2 * switchdist6_2;
      } else {
        r2_1 = rinv*rinv;
        r2_2 = rinv*rinv;
        switchdist6_1 = alchflags.switchdist2 * alchflags.switchdist2 * alchflags.switchdist2;
        switchdist6_2 = switchdist6_1;
      }
      const float r6_1 = r2_1*r2_1*r2_1;
      const float r6_2 = r2_2*r2_2*r2_2;
      if (r2 <= alchflags.switchdist2) {
        const float U1 = ljab.x*r6_1*r6_1 - ljab.y*r6_1; // NB: unscaled, shorthand only!
        const float U2 = ljab.x*r6_2*r6_2 - ljab.y*r6_2;
        // A == ljab.x, B == ljab.y
        const float dU_1 = -ljab.x / (cutoff6 * switchdist6_1) - (-ljab.y * rsqrtf(cutoff6 * switchdist6_1));
        const float dU_2 = -ljab.x / (cutoff6 * switchdist6_2) - (-ljab.y * rsqrtf(cutoff6 * switchdist6_2));
        alch_vdw_energy   = myVdwLambda  * (U1 + dU_1);
        alch_vdw_energy_2 = myVdwLambda2 * (U2 + dU_2);

        //Multiplied by -1.0 to match CPU values
        alch_vdw_force    =-1.f*myVdwLambda*((12.f*U1 + 6.f*ljab.y*r6_1)*r2_1);
      } else {
        const float r3_1 = sqrtf(r6_1);
        const float r3_2 = sqrtf(r6_2);
        const float inv_cutoff6 = 1.0f / cutoff6;
        const float inv_cutoff3 = rsqrtf(cutoff6);
        // A == ljab.x, B == ljab.y
        const float k_vdwa_1 = ljab.x / (1.0f - switchdist6_1 * inv_cutoff6);
        const float k_vdwb_1 = ljab.y / (1.0f - sqrtf(switchdist6_1 * inv_cutoff6));
        const float k_vdwa_2 = ljab.x / (1.0f - switchdist6_2 * inv_cutoff6);
        const float k_vdwb_2 = ljab.y / (1.0f - sqrtf(switchdist6_2 * inv_cutoff6));
        const float tmpa_1 = r6_1 - inv_cutoff6;
        const float tmpb_1 = r3_1 - inv_cutoff3;
        const float tmpa_2 = r6_2 - inv_cutoff6;
        const float tmpb_2 = r3_2 - inv_cutoff3;
        alch_vdw_energy   = myVdwLambda  * (k_vdwa_1 * tmpa_1 * tmpa_1 - k_vdwb_1 * tmpb_1 * tmpb_1);
        alch_vdw_energy_2 = myVdwLambda2 * (k_vdwa_2 * tmpa_2 * tmpa_2 - k_vdwb_2 * tmpb_2 * tmpb_2);
        //Multiplied by -1.0 to match CPU values
        alch_vdw_force = -1.0f * myVdwLambda * (6.0f * r2_1 *  (2.0f * k_vdwa_1 * tmpa_1 * r6_1 - k_vdwb_1 * tmpb_1 * r3_1));
      } // r2 <= alchflags.switchdist2
    } else {
      // potential switching
      const float diff = alchflags.cutoff2 - r2;

      const float switchmul  = (alchflags.switchfactor*(diff)*(diff)*(alchflags.cutoff2 - 3.f*alchflags.switchdist2 + 2.f*r2))*(r2 > alchflags.switchdist2)  + (1.f)*(r2 <= alchflags.switchdist2);
      const float switchmul2 = (12.f*alchflags.switchfactor*(diff)*(r2 - alchflags.switchdist2))*(r2 > alchflags.switchdist2) +  (0.f) * (r2 <= alchflags.switchdist2);

      //Templated parameter. No control divergence here
      if(shift){
        //This templated parameter lets me get away with not making 2 divisions. But for myVdwShift != 0, how do I do this?
        const float myVdwShift = alchflags.vdwShiftUp*up + alchflags.vdwShiftDown*(!up);
        const float myVdwShift2 = alchflags.vdwShift2Up*up + alchflags.vdwShift2Down*(!up);
        //r2_1 = 1.0/(r2 + myVdwShift);
        //r2_2 = 1.0/(r2 + myVdwShift2);
        r2_1 = __fdividef(1.f,(r2 + myVdwShift));
        r2_2 = __fdividef(1.f,(r2 + myVdwShift2));
      }else{
        r2_1 = rinv*rinv;
        r2_2 = rinv*rinv;
      }

      const float r6_1 = r2_1*r2_1*r2_1;
      const float r6_2 = r2_2*r2_2*r2_2;
      const float U1 = ljab.x*r6_1*r6_1 - ljab.y*r6_1; // NB: unscaled, shorthand only!
      const float U2 = ljab.x*r6_2*r6_2 - ljab.y*r6_2;
      alch_vdw_energy   = myVdwLambda*switchmul*U1;
      alch_vdw_energy_2 = myVdwLambda2*switchmul*U2;

      //Multiplied by -1.0 to match CPU values
      alch_vdw_force    =-1.f*myVdwLambda*((switchmul*(12.f*U1 + 6.f*ljab.y*r6_1)*r2_1+ switchmul2*U1));
    } // vdwForceSwitch
  }

/*-----------------------------------------------------------*/

  if (doEnergy){
    //All energies should be scaled by the corresponding lambda
    energyVdw    += (ljab.x * ei.z + ljab.y * ei.y)*(ref || dec) + alch_vdw_energy*(alch && !dec);
    energyElec   += (fSlow * ei.x)*myElecLambda;
    energyVdw_s  += (ljab.x * ei.z + ljab.y * ei.y)*(ref || dec) + alch_vdw_energy_2*(alch && !dec);
    energyElec_s += (fSlow * ei.x)*myElecLambda2;
    if (doSlow){
      energySlow   += (fSlow * ei.w)*myElecLambda;
      energySlow_s += (fSlow * ei.w)*myElecLambda2;
    }
  }

  if (doSlow) fSlow *= fi.w;

  //We should include the regular VDW forces if not dealing with alch pairs
  f = (f + ((ljab.x * fi.z + ljab.y * fi.y)*(!alch || dec)))*myElecLambda 
	+ alch_vdw_force*(alch && !dec); 

  float fx = dx * f;
  float fy = dy * f;
  float fz = dz * f;

  iforce.x += fx;
  iforce.y += fy;
  iforce.z += fz;
  jforce.x -= fx;
  jforce.y -= fy;
  jforce.z -= fz;
   
  if (doSlow) {
    /*There's stuff that needs to be added here, when FAST AND NOSHORT macros are on*/
    fSlow = myElecLambda*fSlow; 
    float fxSlow = dx * fSlow;
    float fySlow = dy * fSlow;
    float fzSlow = dz * fSlow;
    iforceSlow.x += fxSlow;
    iforceSlow.y += fySlow;
    iforceSlow.z += fzSlow;
    jforceSlow.x -= fxSlow;
    jforceSlow.y -= fySlow;
    jforceSlow.z -= fzSlow;
  }
}

/* JM: Special __device__ function to compute VDW forces for TI. 
 */

template<bool doEnergy, bool doSlow, bool shift, bool vdwForceSwitch>
__device__ __forceinline__
void calcForceEnergyTI(const float r2, const float qi, const float qj,
  const float dx, const float dy, const float dz,
  const int vdwtypei, const int vdwtypej, 
  char p1, char p2,
#ifdef USE_TABLE_ARRAYS
  const float2* __restrict__ vdwCoefTable, 
  const float4* __restrict__ forceTable, 
  const float4* __restrict__ energyTable,
#else
  cudaTextureObject_t vdwCoefTableTex, 
  cudaTextureObject_t forceTableTex, 
  cudaTextureObject_t energyTableTex,
#endif
  fast_float3& iforce, fast_float3& iforceSlow, fast_float3& jforce, fast_float3& jforceSlow,
  float& energyVdw,  float& energyVdw_ti_1,  float& energyVdw_ti_2,  
  float& energyElec, float& energyElec_ti_1, float& energyElec_ti_2,
  float& energySlow, float& energySlow_ti_1, float& energySlow_ti_2) {
  
 int vdwIndex = vdwtypej + vdwtypei;
#if defined(USE_TABLE_ARRAYS)
  float2 ljab = fast_load(vdwCoefTable, vdwIndex);
#else
  float2 ljab = tex1Dfetch<float2>(vdwCoefTableTex, vdwIndex); //ljab.x is A and ljab.y is B
#endif

  /*  JM: For TI, we have to deal ALCH1 OR ALCH2 during ComputeNonbondedBase2
   *  ALCH1 for appearing terms;
   *  ALCH2 for dissapearing terms;
   *  Instead of the _s energy terms, we need the to calculate:
   * 
   *  vdwEnergy_ti_1 and _2 for VDW energies. For those we need to add the special terms calculated on
   *  ComputeNonbondedTI.C
   * 
   * elecEnergy_ti_1 and _2 for electrostatic energy. No correction needed here though.
   * 
   */
  
  float myVdwLambda = 0.0f;
  float myElecLambda = 0.0f;
  float rinv = __frsqrt_rn(r2);
  float f;
  float alch_vdw_energy = 0.0f;
  float alch_vdw_force  = 0.0f;
  float alch_vdw_dUdl   = 0.0f;
  float fSlow = qi * qj;
  float4 ei;
#if defined(USE_TABLE_ARRAYS)
  float4 fi = tableLookup(forceTable, rinv);
  if (doEnergy) ei = tableLookup(energyTable, rinv);
#else
   float4 fi = sampleTableTex(forceTableTex, rinv);
   if (doEnergy) ei = sampleTableTex(energyTableTex, rinv);
#endif
  
  //John said that there is a better way to avoid divergences here
  //alch: true if => 1-0, 1-1, 2-0, 2-2
  //dec:  true if => 1-1, 2-2 && decouple
  //up: true if => 1-0 && 1,1
  //down: true if => 2-0, && 2,2
  int ref  = (p1 == 0 && p2 == 0);
  int alch = (!ref && !(p1 == 1  && p2 ==2) && !(p1 == 2 && p2 == 1));
  int dec  = (alch &&  (p1 == p2) && alchflags.alchDecouple);
  int up   = (alch &&  (p1 == 1 || p2 == 1) && !dec);
  int down = (alch &&  (p1 == 2 || p2 == 2) && !dec);
  
  float r2_1;
  f = (fSlow * fi.x);

/*---------------   VDW SPECIAL ALCH STUFF (Swiped from ComputeNonbondedTI.C)  ---------------*/
  myVdwLambda   = alchflags.vdwLambdaUp*(up)   + alchflags.vdwLambdaDown*(down)   + 1.f*(ref || dec);
  myElecLambda  = alchflags.elecLambdaUp*(up)  + alchflags.elecLambdaDown*(down)  + 1.f*(ref || dec);
  if(alch){
    if (vdwForceSwitch) {
      const float cutoff6 = alchflags.cutoff2 * alchflags.cutoff2 * alchflags.cutoff2;
      float switchdist6;
      if (shift) {
        const float myVdwShift = alchflags.vdwShiftUp*up + alchflags.vdwShiftDown*(!up);
        r2_1 = __fdividef(1.f,(r2 + myVdwShift));
        switchdist6 = alchflags.switchdist2 + myVdwShift;
        switchdist6 = switchdist6 * switchdist6 * switchdist6;
      } else {
        r2_1 = rinv*rinv;
        switchdist6 = alchflags.switchdist2 * alchflags.switchdist2 * alchflags.switchdist2;
      }
      const float r6_1 = r2_1*r2_1*r2_1;
      if (r2 <= alchflags.switchdist2) {
        const float U    = ljab.x*r6_1*r6_1 - ljab.y*r6_1;
        const float dU =  -ljab.x / (cutoff6 * switchdist6) - (-ljab.y * rsqrtf(cutoff6 * switchdist6));
        alch_vdw_force = -1.f*(myVdwLambda*((12.f*U + 6.f*ljab.y*r6_1)*r2_1));
        alch_vdw_energy = myVdwLambda * (U + dU);
        alch_vdw_dUdl = U + myVdwLambda * alchflags.alchVdwShiftCoeff * (6.f*U + 3.f*ljab.y*r6_1)*r2_1 + dU;
      } else {
        const float r3_1 = sqrtf(r6_1);
        const float inv_cutoff6 = 1.0f / cutoff6;
        const float inv_cutoff3 = sqrtf(inv_cutoff6);
        const float k_vdwa_1 = ljab.x / (1.0f - switchdist6 * inv_cutoff6);
        const float k_vdwb_1 = ljab.y / (1.0f - sqrtf(switchdist6 * inv_cutoff6));
        const float tmpa_1 = r6_1 - inv_cutoff6;
        const float tmpb_1 = r3_1 - inv_cutoff3;
        const float U = k_vdwa_1 * tmpa_1 * tmpa_1 - k_vdwb_1 * tmpb_1 * tmpb_1;
        alch_vdw_force = -1.0f * myVdwLambda * (6.0f * r2_1 * (2.0f * k_vdwa_1 * tmpa_1 * r6_1 - k_vdwb_1 * tmpb_1 * r3_1));
        alch_vdw_energy = myVdwLambda * U;
        alch_vdw_dUdl = U + myVdwLambda * alchflags.alchVdwShiftCoeff * (3.0f * r2_1 * (2.0f * k_vdwa_1 * tmpa_1 * r6_1 - k_vdwb_1 * tmpb_1 * r3_1));
      } // r2 <= alchflags.switchdist2
    } else {
      // potential switching
      const float diff = alchflags.cutoff2 - r2;
      const float switchmul  = (r2 > alchflags.switchdist2 ? alchflags.switchfactor*(diff)*(diff) \
                              *(alchflags.cutoff2 - 3.f*alchflags.switchdist2 + 2.f*r2) : 1.f);

      const float switchmul2 = (r2 > alchflags.switchdist2 ?          \
                                12.f*alchflags.switchfactor*(diff)       \
                                *(r2 - alchflags.switchdist2) : 0.f);
      //Templated parameter. No control divergence here
      if(shift){
        const float myVdwShift = alchflags.vdwShiftUp*up + alchflags.vdwShiftDown*(!up);
        r2_1 = __fdividef(1.f,(r2 + myVdwShift));
      }else r2_1 = rinv*rinv;

      const float r6_1 = r2_1*r2_1*r2_1;
      const float U    = ljab.x*r6_1*r6_1 - ljab.y*r6_1; // NB: unscaled! for shorthand only!
      alch_vdw_energy = myVdwLambda*switchmul*U;
      //Multiplied by -1.0 to match CPU values
      alch_vdw_force = -1.f*(myVdwLambda*(switchmul*(12.f*U + 6.f*ljab.y*r6_1)*r2_1 \
                                    + switchmul2*U));
      alch_vdw_dUdl = (switchmul*(U + myVdwLambda*alchflags.alchVdwShiftCoeff \
                                    *(6.f*U + 3.f*ljab.y*r6_1)*r2_1));
    } // vdwForceSwitch
  }
  /*-------------------------------------------------------------------------*/

  if (doEnergy){
    //All energies should be scaled by the corresponding lambda
    energyVdw       += (ljab.x * ei.z + ljab.y * ei.y)*(ref || dec) + alch_vdw_energy*(alch && !dec);
    energyElec      += (fSlow * ei.x)*myElecLambda;
    if(alch){
      energyVdw_ti_1  += alch_vdw_dUdl*up;
      energyVdw_ti_2  += alch_vdw_dUdl*down;
      energyElec_ti_1 += (fSlow * ei.x)*up;
      energyElec_ti_2 += (fSlow * ei.x)*down;
    }
    if (doSlow){
      energySlow      += (fSlow * ei.w)*myElecLambda;
      if(alch){
        energySlow_ti_1 += (fSlow * ei.w)*up;
        energySlow_ti_2 += (fSlow * ei.w)*down;
      }
    }
  }

  if (doSlow) fSlow *= fi.w;
  //We should include the regular VDW forces if not dealing with alch pairs
  f = (f + ((ljab.x * fi.z + ljab.y * fi.y)*(ref || dec)))*myElecLambda 
	+ alch_vdw_force*(alch && !dec);

  float fx = dx * f;
  float fy = dy * f;
  float fz = dz * f;

  iforce.x += fx;
  iforce.y += fy;
  iforce.z += fz;
  jforce.x -= fx;
  jforce.y -= fy;
  jforce.z -= fz;
   
  if (doSlow) {
    /*There's stuff that needs to be added here, when FAST AND NOSHORT macros are on*/
    fSlow = myElecLambda*fSlow;  /* FAST(NOSHORT(+alch_vdw_force))*/ //Those should also be zeroed 
    float fxSlow = dx * fSlow;
    float fySlow = dy * fSlow;
    float fzSlow = dz * fSlow;
    iforceSlow.x += fxSlow;
    iforceSlow.y += fySlow;
    iforceSlow.z += fzSlow;
    jforceSlow.x -= fxSlow;
    jforceSlow.y -= fySlow;
    jforceSlow.z -= fzSlow;
  }
}

template<bool doSlow, typename T>
__device__ __forceinline__
void storeForces(const int pos, const T force, const T forceSlow,
                 float* __restrict__ devForces_x, 
                 float* __restrict__ devForces_y, 
                 float* __restrict__ devForces_z,
                 float* __restrict__ devForcesSlow_x, 
                 float* __restrict__ devForcesSlow_y, 
                 float* __restrict__ devForcesSlow_z)
{
#if defined(NAMD_HIP) && ((HIP_VERSION_MAJOR == 3) && (HIP_VERSION_MINOR > 3) || (HIP_VERSION_MAJOR > 3))
  if (force.x != 0.0f || force.y != 0.0f || force.z != 0.0f) {
    atomicAdd(&devForces_x[pos], force.x);
    atomicAdd(&devForces_y[pos], force.y);
    atomicAdd(&devForces_z[pos], force.z);
  }
  if (doSlow) {
    if (forceSlow.x != 0.0f || forceSlow.y != 0.0f || forceSlow.z != 0.0f) {
      atomicAdd(&devForcesSlow_x[pos], forceSlow.x);
      atomicAdd(&devForcesSlow_y[pos], forceSlow.y);
      atomicAdd(&devForcesSlow_z[pos], forceSlow.z);
    }
  }
#else
  atomicAdd(&devForces_x[pos], force.x);
  atomicAdd(&devForces_y[pos], force.y);
  atomicAdd(&devForces_z[pos], force.z);
  if (doSlow) {
    atomicAdd(&devForcesSlow_x[pos], forceSlow.x);
    atomicAdd(&devForcesSlow_y[pos], forceSlow.y);
    atomicAdd(&devForcesSlow_z[pos], forceSlow.z);
  }
#endif
}

template<bool doSlow>
__device__ __forceinline__
void storeForces(const int pos, const float4 force, const float4 forceSlow,
                 float* __restrict__ devForces_x, 
                 float* __restrict__ devForces_y, 
                 float* __restrict__ devForces_z,
                 float* __restrict__ devForcesSlow_x, 
                 float* __restrict__ devForcesSlow_y, 
                 float* __restrict__ devForcesSlow_z)
{
#if defined(NAMD_HIP) && ((HIP_VERSION_MAJOR == 3) && (HIP_VERSION_MINOR > 3) || (HIP_VERSION_MAJOR > 3))
  if (force.x != 0.0f || force.y != 0.0f || force.z != 0.0f) {
    atomicAdd(&devForces_x[pos], force.x);
    atomicAdd(&devForces_y[pos], force.y);
    atomicAdd(&devForces_z[pos], force.z);
  }
  if (doSlow) {
    if (forceSlow.x != 0.0f || forceSlow.y != 0.0f || forceSlow.z != 0.0f) {
      atomicAdd(&devForcesSlow_x[pos], forceSlow.x);
      atomicAdd(&devForcesSlow_y[pos], forceSlow.y);
      atomicAdd(&devForcesSlow_z[pos], forceSlow.z);
    }
  }
#else
  atomicAdd(&devForces_x[pos], force.x);
  atomicAdd(&devForces_y[pos], force.y);
  atomicAdd(&devForces_z[pos], force.z);
  if (doSlow) {
    atomicAdd(&devForcesSlow_x[pos], forceSlow.x);
    atomicAdd(&devForcesSlow_y[pos], forceSlow.y);
    atomicAdd(&devForcesSlow_z[pos], forceSlow.z);
  }
#endif
}

//#define USE_NEW_EXCL_METHOD

//
// Returns the lower estimate for the distance between a bounding box and a set of atoms
//
__device__ __forceinline__ float distsq(const BoundingBox a, const float4 b) {
  float dx = max(0.0f, fabsf(a.x - b.x) - a.wx);
  float dy = max(0.0f, fabsf(a.y - b.y) - a.wy);
  float dz = max(0.0f, fabsf(a.z - b.z) - a.wz);
  float r2 = dx*dx + dy*dy + dz*dz;
  return r2;
}

#define LARGE_FLOAT (float)(1.0e10)

//
// Nonbonded force kernel
//
template <bool doEnergy, bool doVirial, bool doSlow, bool doPairlist, bool doAlch, bool doFEP, bool doTI, bool doStreaming, bool doTable, bool doAlchVdwForceSwitching>
__global__ void
__launch_bounds__(WARPSIZE*NONBONDKERNEL_NUM_WARP,
  doPairlist ? (10) : (doEnergy ? (10) : (10) ))
nonbondedForceKernel(
  const int start, const int numTileLists,
  const TileList* __restrict__ tileLists, TileExcl* __restrict__ tileExcls,
  const int* __restrict__ tileJatomStart,
  const int vdwCoefTableWidth,
#if defined(USE_TABLE_ARRAYS) 
  const float2* __restrict__ vdwCoefTable,
#else
  cudaTextureObject_t vdwCoefTableTex,
#endif
  const int* __restrict__ vdwTypes,
  const float3 lata, const float3 latb, const float3 latc,
  const float4* __restrict__ xyzq, const float cutoff2, const CudaNBConstants nbConstants, 
#ifdef USE_TABLE_ARRAYS
  const float4* __restrict__ forceTable, const float4* __restrict__ energyTable,
#else
  cudaTextureObject_t forceTableTex, cudaTextureObject_t energyTableTex,
#endif
  // ----------
  // doPairlist
  float plcutoff2, const PatchPairRecord* __restrict__ patchPairs,
  const int* __restrict__ atomIndex,
  const int2* __restrict__ exclIndexMaxDiff, const unsigned int* __restrict__ overflowExclusions,
  unsigned int* __restrict__ tileListDepth, int* __restrict__ tileListOrder,
  int* __restrict__ jtiles, TileListStat* __restrict__ tileListStat,
  const BoundingBox* __restrict__ boundingBoxes,
#ifdef USE_NEW_EXCL_METHOD
  const int* __restrict__ minmaxExclAtom,
#endif
  // ----------
  float * __restrict__ devForce_x,
  float * __restrict__ devForce_y,
  float * __restrict__ devForce_z,
  float * __restrict__ devForce_w,
  float * __restrict__ devForceSlow_x,
  float * __restrict__ devForceSlow_y,
  float * __restrict__ devForceSlow_z,
  float * __restrict__ devForceSlow_w,                     
  // ---- USE_STREAMING_FORCES ----
  const int numPatches,
  unsigned int* __restrict__ patchNumCount,
  const CudaPatchRecord* __restrict__ cudaPatches,
  float4* __restrict__ mapForces, float4* __restrict__ mapForcesSlow,
  int* __restrict__ mapPatchReadyQueue,
  int* __restrict__ outputOrder,
  // ------------------------------
  TileListVirialEnergy* __restrict__ virialEnergy,
  // ---- doAlch ----
  char* __restrict__ p
  ) {

  // Single warp takes care of one list of tiles
  // for (int itileList = (threadIdx.x + blockDim.x*blockIdx.x)/WARPSIZE;itileList < numTileLists;itileList += blockDim.x*gridDim.x/WARPSIZE)
  // int itileList = start + threadIdx.x/WARPSIZE + blockDim.x/WARPSIZE*blockIdx.x;
  // The line above is the CUDA-exclusive version. HIPification in this case is selecting b const CudaNBConstants nbConstants, ased on specific parameters
  // In the hip scheme: Each warp gets an itile and 2 jtiles
  // if NONBONDEDKERNEL_NUM_WARP == 1, each warp does 2 tiles, so we do 2* blockIdx
  // int itileList = start + (threadIdx.x/WARPSIZE + NONBONDKERNEL_NUM_WARP*blockIdx.x);
  // fetches 1 tile
  int itileList = start + (WARPSIZE == 64 ? blockIdx.x : (threadIdx.x/BOUNDINGBOXSIZE + NONBONDKERNEL_NUM_WARP*blockIdx.x));
  if (itileList < numTileLists)
  {
    fast_float3 iforce, iforceSlow;
    fast_float3 jforce, jforceSlow;
    float energyVdw, energyElec, energySlow;
    //FEP energies
    float energyVdw_s, energyElec_s, energySlow_s;
    //TI energies
    float energyVdw_ti_1, energyVdw_ti_2, energyElec_ti_1, energyElec_ti_2, energySlow_ti_1, energySlow_ti_2;
    int nexcluded;
    unsigned int itileListLen;
    int2 patchInd;
    int2 patchNumList;
    char part1, part2, p2;
    bool doShift = doAlch && (alchflags.alchVdwShiftCoeff != 0.0f);
    __shared__ float4 s_xyzq[NONBONDKERNEL_NUM_WARP][BOUNDINGBOXSIZE];
    __shared__ int s_vdwtypej[NONBONDKERNEL_NUM_WARP][BOUNDINGBOXSIZE];
    __shared__ float4 s_jforce[NONBONDKERNEL_SWEEPS_PER_TILE][BOUNDINGBOXSIZE];
    __shared__ float4 s_jforceSlow[NONBONDKERNEL_SWEEPS_PER_TILE][BOUNDINGBOXSIZE];
    __shared__ int s_jatomIndex[NONBONDKERNEL_NUM_WARP][BOUNDINGBOXSIZE];

    // Warp index (0...WARPSIZE-1)
    // JM: For BoundingBoxSize32 and WARPSIZE == 64, each warp gets two tile lists!

    const int wid = threadIdx.x % BOUNDINGBOXSIZE;
    const int iwarp  = 0; // one workgroup does two sweeps
    const int isweep = threadIdx.x / BOUNDINGBOXSIZE; // 2 sweeps for 32-sized bounding boxes 
    constexpr int nsweep = NONBONDKERNEL_SWEEPS_PER_TILE;

    // Start computation
    {
      TileList tmp = tileLists[itileList];
      int iatomStart = tmp.iatomStart;
      int jtileStart = tmp.jtileStart;
      int jtileEnd   = tmp.jtileEnd;
      patchInd     = tmp.patchInd;
      patchNumList = tmp.patchNumList;

      float shx = tmp.offsetXYZ.x*lata.x + tmp.offsetXYZ.y*latb.x + tmp.offsetXYZ.z*latc.x;
      float shy = tmp.offsetXYZ.x*lata.y + tmp.offsetXYZ.y*latb.y + tmp.offsetXYZ.z*latc.y;
      float shz = tmp.offsetXYZ.x*lata.z + tmp.offsetXYZ.y*latb.z + tmp.offsetXYZ.z*latc.z;

      // DH - set zeroShift flag if magnitude of shift vector is zero
      bool zeroShift = ! (shx*shx + shy*shy + shz*shz > 0);

      int iatomSize, iatomFreeSize, jatomSize, jatomFreeSize;
      if (doPairlist) {
        PatchPairRecord PPStmp = patchPairs[itileList];
        iatomSize     = PPStmp.iatomSize;
        iatomFreeSize = PPStmp.iatomFreeSize;
        jatomSize     = PPStmp.jatomSize;
        jatomFreeSize = PPStmp.jatomFreeSize;
      }

      // Write to global memory here to avoid register spilling
      if (doVirial) {
        if (wid == 0) {
          virialEnergy[itileList].shx = shx;
          virialEnergy[itileList].shy = shy;
          virialEnergy[itileList].shz = shz;
        }
      }

      // Load i-atom data (and shift coordinates)
      fast_float3 xyz_i;
      float4 xyzq_i = xyzq[iatomStart + wid];
      if (doAlch) part1 =  p[iatomStart + wid];
      xyzq_i.x += shx;
      xyzq_i.y += shy;
      xyzq_i.z += shz;
      xyz_i.x = xyzq_i.x; // no wasted registers here, just aliasing as float4s are vectorized
      xyz_i.y = xyzq_i.y;
      xyz_i.z = xyzq_i.z;
      int vdwtypei = vdwTypes[iatomStart + wid]*vdwCoefTableWidth;

      // Load i-atom data (and shift coordinates)
      BoundingBox boundingBoxI;
      if (doPairlist) {
        boundingBoxI = boundingBoxes[iatomStart/BOUNDINGBOXSIZE];
        boundingBoxI.x += shx;
        boundingBoxI.y += shy;
        boundingBoxI.z += shz;
      }

      // Get i-atom global index
#ifdef USE_NEW_EXCL_METHOD
      int iatomIndex, minExclAtom, maxExclAtom;
#else
      int iatomIndex;
#endif
      if (doPairlist) {
#ifdef USE_NEW_EXCL_METHOD
        iatomIndex = atomIndex[iatomStart + wid];
        int2 tmp = minmaxExclAtom[iatomStart + wid];
        minExclAtom = tmp.x;
        maxExclAtom = tmp.y;
#else
        iatomIndex = atomIndex[iatomStart + wid];
#endif
      }

      // i-forces in registers
      // float3 iforce;
      iforce.x = 0.0f;
      iforce.y = 0.0f;
      iforce.z = 0.0f;

      // float3 iforceSlow;
      if (doSlow) {
        iforceSlow.x = 0.0f;
        iforceSlow.y = 0.0f;
        iforceSlow.z = 0.0f;
      }

      // float energyVdw, energyElec, energySlow;
      if (doEnergy) {
        energyVdw       = 0.0f;
        energyVdw_s     = 0.0f;
        energyVdw_ti_1  = 0.0f;
        energyVdw_ti_2  = 0.0f;
        energyElec      = 0.0f;
        energyElec_ti_1 = 0.0f;
        energyElec_ti_2 = 0.0f;
        energyElec_s    = 0.0f;
        if (doSlow){
          energySlow      = 0.0f;
          energySlow_s    = 0.0f;
          energySlow_ti_1 = 0.0f;
          energySlow_ti_2 = 0.0f;
        }
      }

      // Number of exclusions
      // NOTE: Lowest bit is used as indicator bit for tile pairs:
      //       bit 0 tile has no atoms within pairlist cutoff
      //       bit 1 tile has atoms within pairlist cutoff
      // int nexcluded;
      if (doPairlist) nexcluded = 0;

      // Number of i loops and free atoms
      int nfreei;
      if (doPairlist) {
        int nloopi = min(iatomSize - iatomStart, BOUNDINGBOXSIZE);
        nfreei = max(iatomFreeSize - iatomStart, 0);
        if (wid >= nloopi) {
          xyzq_i.x = -LARGE_FLOAT;
          xyzq_i.y = -LARGE_FLOAT;
          xyzq_i.z = -LARGE_FLOAT;
        }
      }

      // tile list stuff
      // int itileListLen;
      // int minJatomStart;
      if (doPairlist) {
        // minJatomStart = tileJatomStart[jtileStart];
        itileListLen = 0;
      }

      // Exclusion index and maxdiff
      int iexclIndex, iexclMaxdiff;
      if (doPairlist) {
        int2 tmp = exclIndexMaxDiff[iatomStart + wid];
        iexclIndex   = tmp.x;
        iexclMaxdiff = tmp.y;
      }
      // fetch two tiles at once here
      for (int jtile=jtileStart;jtile <= jtileEnd;jtile++) {
        // Load j-atom starting index and exclusion mask
        int jatomStart = tileJatomStart[jtile];

        float4 xyzq_j = xyzq[jatomStart + wid];
        NAMD_WARP_SYNC(WARP_FULL_MASK);        
        if (doAlch) p2 =  p[jatomStart + wid];

        // Check for early bail
        if (doPairlist) {
          float r2bb = distsq(boundingBoxI, xyzq_j);
          if (NAMD_WARP_ALL(WARP_FULL_MASK, r2bb > plcutoff2)) continue;
        }
        WarpMask excl = (doPairlist) ? 0 : tileExcls[jtile].excl[wid];
        int vdwtypej = vdwTypes[jatomStart + wid];
        s_vdwtypej[iwarp][wid] = vdwtypej;

        // Get i-atom global index
        if (doPairlist) {
          s_jatomIndex[iwarp][wid] = atomIndex[jatomStart + wid];
        }

        // Number of j loops and free atoms
        int nfreej;
        if (doPairlist) {
          int nloopj = min(jatomSize - jatomStart, BOUNDINGBOXSIZE);
          nfreej = max(jatomFreeSize - jatomStart, 0);
          //if (nfreei == 0 && nfreej == 0) continue;
          if (wid >= nloopj) {
            xyzq_j.x = LARGE_FLOAT;
            xyzq_j.y = LARGE_FLOAT;
            xyzq_j.z = LARGE_FLOAT;
          }
        }
        s_xyzq[iwarp][wid] = xyzq_j;        

        // DH - self requires that zeroShift is also set
        const bool self = zeroShift && (iatomStart == jatomStart);
        const int modval = (self) ? 2*BOUNDINGBOXSIZE-1 : BOUNDINGBOXSIZE-1;

        s_jforce[isweep][wid] = make_float4(0.0f, 0.0f, 0.0f, 0.f);
        if (doSlow)
          s_jforceSlow[isweep][wid] = make_float4(0.0f, 0.0f, 0.0f, 0.f);
        NAMD_WARP_SYNC(WARP_FULL_MASK);

        if (doPairlist) {
          // Build pair list
          // NOTE: Pairlist update, we must also include the diagonal since this is used
          //       in GBIS phase 2.
          // Clear the lowest (indicator) bit
          nexcluded &= (~1);

          // For self tiles, do the diagonal term (t=0).
          // NOTE: No energies are computed here, since this self-diagonal term is only for GBIS phase 2
          if (self && !isweep) {
            int j = (0 + wid) & modval;
            xyzq_j = s_xyzq[iwarp][j];
            float dx = xyzq_j.x - xyzq_i.x;
            float dy = xyzq_j.y - xyzq_i.y;
            float dz = xyzq_j.z - xyzq_i.z;
            float r2 = dx*dx + dy*dy + dz*dz;

            if (j < BOUNDINGBOXSIZE && r2 < plcutoff2) {
              // We have atom pair within the pairlist cutoff => Set indicator bit
              nexcluded |= 1;
            }
          }
          for (int t = (self && !isweep) ? isweep+nsweep : isweep;t < BOUNDINGBOXSIZE; t+=nsweep) {
            int j = (t + wid) & modval;

            // NOTE: __shfl() operation can give non-sense here because j may be >= WARPSIZE.
            //       However, if (j < WARPSIZE ..) below makes sure that these non-sense
            //       results are not used
            if (doAlch) part2 = WARP_SHUFFLE(WARP_FULL_MASK, p2, j, WARPSIZE);

            excl >>= nsweep;// moves it two sweeps
            if (j < BOUNDINGBOXSIZE) {            
              xyzq_j = s_xyzq[iwarp][j];
              float dx = xyzq_j.x - xyzq_i.x;
              float dy = xyzq_j.y - xyzq_i.y;
              float dz = xyzq_j.z - xyzq_i.z;
              float r2 = dx*dx + dy*dy + dz*dz;
              if (r2 < plcutoff2) {
                // We have atom pair within the pairlist cutoff => Set indicator bit
                nexcluded |= 1;
                if (j < nfreej || wid < nfreei) {
                  bool excluded = false;
                  int indexdiff = s_jatomIndex[iwarp][j] - iatomIndex;                
                  if ( abs(indexdiff) <= iexclMaxdiff) {
                    indexdiff += iexclIndex;
                    int indexword = ((unsigned int) indexdiff) >> 5;
                    {
                      indexword = overflowExclusions[indexword];
                    }

                    excluded = ((indexword & (1<<(indexdiff&31))) != 0);
                  }
                  if (excluded) nexcluded += 2;
                  if (!excluded) excl |= (WarpMask)1 << (BOUNDINGBOXSIZE-(nsweep - isweep)); // good luck understanding this
                  if(doAlch){
                    if(!excluded && r2 < cutoff2){
                      jforce = float4(s_jforce[isweep][j]);
		      if(doSlow) jforceSlow = float4(s_jforceSlow[isweep][j]);
                      if(doShift){
                        if(doFEP){
                          calcForceEnergyFEP<doEnergy, doSlow, true, doAlchVdwForceSwitching>(
                            r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                            vdwtypei, s_vdwtypej[iwarp][j], part1, part2,
#ifdef USE_TABLE_ARRAYS
                            vdwCoefTable, 
#else
                            vdwCoefTableTex,
#endif
#ifdef USE_TABLE_ARRAYS
                            forceTable, energyTable,
#else
                            forceTableTex, energyTableTex,
#endif
                            iforce, iforceSlow,
                            jforce, jforceSlow,
                            energyVdw, energyVdw_s,
                            energyElec, energySlow, energyElec_s, energySlow_s);
                        }else{
                          calcForceEnergyTI<doEnergy, doSlow, true, doAlchVdwForceSwitching>(
                            r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                            vdwtypei, s_vdwtypej[iwarp][j], part1, part2,
#ifdef USE_TABLE_ARRAYS
                            vdwCoefTable, 
#else
                            vdwCoefTableTex,
#endif
#ifdef USE_TABLE_ARRAYS
                            forceTable, energyTable,
#else
                            forceTableTex, energyTableTex,
#endif
                            iforce, iforceSlow,
                            jforce, jforceSlow,
                            energyVdw, energyVdw_ti_1, 
                            energyVdw_ti_2, energyElec, energyElec_ti_1, energyElec_ti_2, 
                            energySlow, energySlow_ti_1, energySlow_ti_2);
                        }//if doFEP
                      }else{
                        if(doFEP){
                          calcForceEnergyFEP<doEnergy, doSlow, false, doAlchVdwForceSwitching>(
                            r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                            vdwtypei, s_vdwtypej[iwarp][j], part1, part2,
#ifdef USE_TABLE_ARRAYS
                            vdwCoefTable, 
#else
                            vdwCoefTableTex,
#endif
#ifdef USE_TABLE_ARRAYS
                            forceTable, energyTable,
#else
                            forceTableTex, energyTableTex,
#endif
                            iforce, iforceSlow,
                            jforce, jforceSlow,
                            energyVdw, energyVdw_s,
                            energyElec, energySlow, energyElec_s, energySlow_s);
                        }else{
                          calcForceEnergyTI<doEnergy, doSlow, false, doAlchVdwForceSwitching>(
                            r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                            vdwtypei, s_vdwtypej[iwarp][j], part1, part2,
#ifdef USE_TABLE_ARRAYS
                            vdwCoefTable, 
#else
                            vdwCoefTableTex,
#endif
#ifdef USE_TABLE_ARRAYS
                            forceTable, energyTable,
#else
                            forceTableTex, energyTableTex,
#endif
                            iforce, iforceSlow,
                            jforce, jforceSlow,
                            energyVdw, energyVdw_ti_1, 
                            energyVdw_ti_2, energyElec, energyElec_ti_1, energyElec_ti_2, 
                            energySlow, energySlow_ti_1, energySlow_ti_2);
                        }
                      }//if doShift
                      s_jforce[isweep][j] = make_float4(jforce.x, jforce.y, jforce.z, 1.f);
                      if(doSlow) s_jforceSlow[isweep][j] = make_float4(jforceSlow.x, jforceSlow.y, jforceSlow.z, 1.f);
                    }//if !excluded && r2 < cutoff2
                  }else{
                    if (!excluded && r2 < cutoff2) {
                      jforce = float4(s_jforce[isweep][j]);
		      if(doSlow) jforceSlow = float4(s_jforceSlow[isweep][j]);
		      if(doTable){
                        calcForceEnergy<doEnergy, doSlow>(
                           r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                           vdwtypei, s_vdwtypej[iwarp][j],
#ifdef USE_TABLE_ARRAYS
                           vdwCoefTable,
#else
                           vdwCoefTableTex,
#endif
#ifdef USE_TABLE_ARRAYS
                           forceTable, energyTable,
#else
                           forceTableTex, energyTableTex,
#endif
                           iforce, iforceSlow,
                           jforce, jforceSlow,
                           energyVdw,
                           energyElec, energySlow);
                      } else {
                        calcForceEnergyMath<doEnergy, doSlow>(
                           r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                           vdwtypei, s_vdwtypej[iwarp][j],
#ifdef USE_TABLE_ARRAYS
                           vdwCoefTable,
#else
                           vdwCoefTableTex,
#endif
                           iforce, iforceSlow,
                           jforce, jforceSlow, 
                           energyVdw,
                           energyElec, energySlow, nbConstants);

                      }
                      s_jforce[isweep][j]                = make_float4(jforce.x, jforce.y, jforce.z, 1.f);
                      if(doSlow) s_jforceSlow[isweep][j] = make_float4(jforceSlow.x, jforceSlow.y, jforceSlow.z, 1.f);
                    } 
                  } 
                }
              }
            }
            NAMD_WARP_SYNC(WARP_FULL_MASK);            
         } // t
       } else {
          // Just compute forces
          if (self) {
            // Clear the first bit
            excl = excl & (~(WarpMask)1);
          } 
          excl >>= isweep;
          int ind_sweep = isweep + wid;
#pragma unroll 4
          for (int t = 0; t < BOUNDINGBOXSIZE;t+= nsweep) {
            if (doAlch) {
              int j = (ind_sweep + t) & (BOUNDINGBOXSIZE-1);
              part2 = WARP_SHUFFLE(WARP_FULL_MASK, p2, j, WARPSIZE);
            }
            if ((excl & 1)) {
              fast_float3 xyz_j;
              int j = (ind_sweep + t) & (BOUNDINGBOXSIZE-1);
              xyzq_j = (float4)s_xyzq[iwarp][j];
              xyz_j.x = xyzq_j.x;
              xyz_j.y = xyzq_j.y;
              xyz_j.z = xyzq_j.z;
              fast_float3 dr = xyz_j - xyz_i;
              // float dx = xyzq_j.x - xyzq_i.x;
              // float dy = xyzq_j.y - xyzq_i.y;
              // float dz = xyzq_j.z - xyzq_i.z;
              // float r2 = dx*dx + dy*dy + dz*dz;              
              float dx = dr.x; 
              float dy = dr.y; 
              float dz = dr.z; 
              float r2 = norm2(dr);
              if(doAlch){
                  if(r2 < cutoff2){ // (r2 < cutoff2)
		      jforce = float4(s_jforce[isweep][j]);
		      if(doSlow) jforceSlow = float4(s_jforceSlow[isweep][j]);
                      if(doShift){
                            if (doFEP){
                                calcForceEnergyFEP<doEnergy, doSlow, true, doAlchVdwForceSwitching>(
                                  r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                                  vdwtypei, s_vdwtypej[iwarp][j], part1, part2,
#ifdef USE_TABLE_ARRAYS
                                  vdwCoefTable, 
#else
                                  vdwCoefTableTex,
#endif
#ifdef USE_TABLE_ARRAYS
                                  forceTable, energyTable,
#else
                                  forceTableTex, energyTableTex,
#endif
                                  iforce, iforceSlow,
                                  jforce, jforceSlow,
                                  energyVdw, energyVdw_s,
                                  energyElec, energySlow, energyElec_s, energySlow_s);
                            }else{
                                calcForceEnergyTI<doEnergy, doSlow, true, doAlchVdwForceSwitching>(
                                  r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                                  vdwtypei, s_vdwtypej[iwarp][j], part1, part2,
#ifdef USE_TABLE_ARRAYS
                                  vdwCoefTable, 
#else
                                  vdwCoefTableTex,
#endif
#ifdef USE_TABLE_ARRAYS
                                  forceTable, energyTable,
#else
                                  forceTableTex, energyTableTex,
#endif
                                  iforce, iforceSlow,
                                  jforce, jforceSlow,
                                  energyVdw, energyVdw_ti_1, 
                                  energyVdw_ti_2, energyElec, energyElec_ti_1, energyElec_ti_2, 
                                  energySlow, energySlow_ti_1, energySlow_ti_2);
                            }//if doFEP
                      }else{
                        if(doFEP){
                          calcForceEnergyFEP<doEnergy, doSlow, false, doAlchVdwForceSwitching>(
                            r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                            vdwtypei, s_vdwtypej[iwarp][j], part1, part2,
#ifdef USE_TABLE_ARRAYS
                            vdwCoefTable, 
#else
                            vdwCoefTableTex,
#endif
#ifdef USE_TABLE_ARRAYS
                            forceTable, energyTable,
#else
                            forceTableTex, energyTableTex,
#endif
                            iforce, iforceSlow,
                            jforce, jforceSlow,
                            energyVdw, energyVdw_s,
                            energyElec, energySlow, energyElec_s, energySlow_s);
                        }else{
                          calcForceEnergyTI<doEnergy, doSlow, false, doAlchVdwForceSwitching>(
                            r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                            vdwtypei, s_vdwtypej[iwarp][j], part1, part2,
#ifdef USE_TABLE_ARRAYS
                            vdwCoefTable,
#else
                            vdwCoefTableTex,
#endif
#ifdef USE_TABLE_ARRAYS
                            forceTable, energyTable,
#else
                            forceTableTex, energyTableTex,
#endif
                            iforce, iforceSlow,
                            jforce, jforceSlow,
                            energyVdw, energyVdw_ti_1, 
                            energyVdw_ti_2, energyElec, energyElec_ti_1, energyElec_ti_2, 
                            energySlow, energySlow_ti_1, energySlow_ti_2);
                        }//if doFEP
                      }//doShift 
                      s_jforce[isweep][j] = make_float4(jforce.x, jforce.y, jforce.z, 1.f);
                     if(doSlow) s_jforceSlow[isweep][j] = make_float4(jforceSlow.x, jforceSlow.y, jforceSlow.z, 1.f);
                }//r2 < cutoff
              }else {
                if (r2 < cutoff2) {
		  jforce = float4(s_jforce[isweep][j]);
		  if(doSlow) jforceSlow = float4(s_jforceSlow[isweep][j]);
	          if (doTable) {
                    calcForceEnergy<doEnergy, doSlow>(
                      r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                      vdwtypei, s_vdwtypej[iwarp][j],
#ifdef USE_TABLE_ARRAYS
                      vdwCoefTable, 
#else
                      vdwCoefTableTex,
#endif
#ifdef USE_TABLE_ARRAYS
                      forceTable, energyTable,
#else
                      forceTableTex, energyTableTex,
#endif
                      iforce, iforceSlow,
                      jforce, jforceSlow,
                      energyVdw, energyElec, energySlow);
		  }
		  else{
                    calcForceEnergyMath<doEnergy, doSlow>(
                      r2, xyzq_i.w, xyzq_j.w, dx, dy, dz,
                      vdwtypei, s_vdwtypej[iwarp][j],
#ifdef USE_TABLE_ARRAYS
                      vdwCoefTable,
#else
                      vdwCoefTableTex,
#endif
                      iforce, iforceSlow,
                      jforce, jforceSlow,
                      energyVdw,
                      energyElec, energySlow, nbConstants);
		  }
		  s_jforce[isweep][j] = make_float4(jforce.x, jforce.y, jforce.z, 0.f);
                  if(doSlow) s_jforceSlow[isweep][j] = make_float4(jforceSlow.x, jforceSlow.y, jforceSlow.z, 0.f);
                }
              }//doAlch 
            } 
	    excl >>= nsweep;
            NAMD_WARP_SYNC(WARP_FULL_MASK);
          } // t
        }

        // Write j-forces - shuffle them before storing to global memory
        jforce = s_jforce[isweep][wid];
        if(doSlow) jforceSlow = s_jforceSlow[isweep][wid];
        float4 jforceAccum = make_float4(jforce.x, jforce.y, jforce.z, 1.f);
        float4 jforceSlowAccum;
        if (doSlow) jforceSlowAccum = make_float4(jforceSlow.x, jforceSlow.y, jforceSlow.z, 1.f);
#if 0
        jforce.x += WARP_SHUFFLE(WARP_FULL_MASK, jforce.x, wid + BOUNDINGBOXSIZE, WARPSIZE);
        jforce.y += WARP_SHUFFLE(WARP_FULL_MASK, jforce.y, wid + BOUNDINGBOXSIZE, WARPSIZE);
        jforce.z += WARP_SHUFFLE(WARP_FULL_MASK, jforce.z, wid + BOUNDINGBOXSIZE, WARPSIZE);
        if(doSlow){
          jforceSlow.x += WARP_SHUFFLE(WARP_FULL_MASK, jforceSlow.x, wid + BOUNDINGBOXSIZE, WARPSIZE);
          jforceSlow.y += WARP_SHUFFLE(WARP_FULL_MASK, jforceSlow.y, wid + BOUNDINGBOXSIZE, WARPSIZE);
          jforceSlow.z += WARP_SHUFFLE(WARP_FULL_MASK, jforceSlow.z, wid + BOUNDINGBOXSIZE, WARPSIZE);
        }
#else
        for(int k = 1 ; k < nsweep; k++){
          jforceAccum.x += WARP_SHUFFLE(WARP_FULL_MASK, jforce.x, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
          jforceAccum.y += WARP_SHUFFLE(WARP_FULL_MASK, jforce.y, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
          jforceAccum.z += WARP_SHUFFLE(WARP_FULL_MASK, jforce.z, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
          if(doSlow){
            jforceSlowAccum.x += WARP_SHUFFLE(WARP_FULL_MASK, jforceSlow.x, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
            jforceSlowAccum.y += WARP_SHUFFLE(WARP_FULL_MASK, jforceSlow.y, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
            jforceSlowAccum.z += WARP_SHUFFLE(WARP_FULL_MASK, jforceSlow.z, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
          }
        }
#endif
        if(!isweep){
          storeForces<doSlow, float4>(jatomStart + wid, jforceAccum, jforceSlowAccum,
                            devForce_x, devForce_y, devForce_z,
                            devForceSlow_x, devForceSlow_y, devForceSlow_z);
        }
        // Write exclusions
        if (doPairlist) {

          // We have to calculate the global flags of this part here, which is 
          WarpMask res_excl = excl;
          for(int k = 1 ; k < nsweep; k++){
            // shuffle the integers to make sure we have complete flags
            res_excl |= WARP_SHUFFLE(WARP_FULL_MASK, excl, wid + BOUNDINGBOXSIZE*k, WARPSIZE);
          }
          // after that, everyone grabs the value from the lower warp
          int anyexcl = (65536 | NAMD_WARP_ANY(WARP_FULL_MASK, res_excl != 0));
          // Mark this jtile as non-empty:
          //  VdW:      1 if tile has atom pairs within pairlist cutoff and some these atoms interact
          //  GBIS: 65536 if tile has atom pairs within pairlist cutoff but not necessary interacting (i.e. these atoms are fixed or excluded)
          if (!isweep && anyexcl){ // lower threads in warp updates the values
            if (wid) jtiles[jtile] = anyexcl;
            // Store exclusions            
            tileExcls[jtile].excl[wid] = res_excl;
            // itileListLen:
            // lower 16 bits number of tiles with atom pairs within pairlist cutoff that interact
            // upper 16 bits number of tiles with atom pairs within pairlist cutoff (but not necessary interacting)
            // low sweep has the correct value for this, so all good
            itileListLen += anyexcl;
            // NOTE, this minJatomStart is only stored once for the first tile list entry
            // minJatomStart = min(minJatomStart, jatomStart);
          }
        }

      } // jtile

#if 1
      // Write i-forces
      // shfl before writing - lower tile gets high tile value
      float3 iforceAccum = iforce;
      float3 iforceSlowAccum = iforceSlow;
      for(int k = 1 ; k < nsweep; k++){
        iforceAccum.x += WARP_SHUFFLE(WARP_FULL_MASK, iforce.x, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
        iforceAccum.y += WARP_SHUFFLE(WARP_FULL_MASK, iforce.y, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
        iforceAccum.z += WARP_SHUFFLE(WARP_FULL_MASK, iforce.z, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
        if(doSlow){
          iforceSlowAccum.x += WARP_SHUFFLE(WARP_FULL_MASK, iforceSlow.x, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
          iforceSlowAccum.y += WARP_SHUFFLE(WARP_FULL_MASK, iforceSlow.y, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
          iforceSlowAccum.z += WARP_SHUFFLE(WARP_FULL_MASK, iforceSlow.z, wid+BOUNDINGBOXSIZE*k, WARPSIZE);
        }
      }
      if(!isweep){
        storeForces<doSlow, float3>(iatomStart + wid, iforceAccum, iforceSlowAccum,
          devForce_x, devForce_y, devForce_z,
          devForceSlow_x, devForceSlow_y, devForceSlow_z);
      }
#endif
    }
    // Done with computation

    // Save pairlist stuff
    if (doPairlist) {
      if (wid == 0 && isweep == 0){
        // minJatomStart is in range [0 ... atomStorageSize-1]
        //int atom0 = (minJatomStart)/WARPSIZE;
        // int atom0 = 0;
        // int storageOffset = atomStorageSize/WARPSIZE;
        // int itileListLen = 0;
        // for (int jtile=jtileStart;jtile <= jtileEnd;jtile++) itileListLen += jtiles[jtile];
        // Store 0 if itileListLen == 0
        // tileListDepth[itileList] = (itileListLen > 0)*(itileListLen*storageOffset + atom0);
        // race condition here if we have the warp fetching two tiles from the same list
        tileListDepth[itileList] = itileListLen;
        tileListOrder[itileList] = itileList;
        // Number of active tilelists with tile with atom pairs within pairlist cutoff that interact
        if ((itileListLen & 65535) > 0) atomicAdd(&tileListStat->numTileLists, 1);
        // Number of active tilelists with tiles with atom pairs within pairlist cutoff (but not necessary interacting)
        if (itileListLen > 0) atomicAdd(&tileListStat->numTileListsGBIS, 1);
        // NOTE: always numTileListsGBIS >= numTileLists
      }

      typedef cub::WarpReduce<int> WarpReduceInt;
      __shared__ typename WarpReduceInt::TempStorage tempStorage[NONBONDKERNEL_NUM_WARP];
      const int warpId = threadIdx.x / WARPSIZE;
      // Remove indicator bit
      nexcluded >>= 1;
      volatile int nexcludedWarp = WarpReduceInt(tempStorage[warpId]).Sum(nexcluded);
      if (threadIdx.x % WARPSIZE == 0){
        atomicAdd(&tileListStat->numExcluded, nexcludedWarp);
      }
    }

    if (doVirial) {
      typedef cub::WarpReduce<float> WarpReduce;
      __shared__ typename WarpReduce::TempStorage tempStorage[NONBONDKERNEL_NUM_WARP];
      const int warpId = threadIdx.x / WARPSIZE;
      volatile float iforcexSum = WarpReduce(tempStorage[warpId]).Sum(iforce.x);
      NAMD_WARP_SYNC(WARP_FULL_MASK);
      volatile float iforceySum = WarpReduce(tempStorage[warpId]).Sum(iforce.y);
      NAMD_WARP_SYNC(WARP_FULL_MASK);
      volatile float iforcezSum = WarpReduce(tempStorage[warpId]).Sum(iforce.z);
      NAMD_WARP_SYNC(WARP_FULL_MASK);
      if (wid == 0) {
        virialEnergy[itileList].forcex = iforcexSum;
        virialEnergy[itileList].forcey = iforceySum;
        virialEnergy[itileList].forcez = iforcezSum;
      }

      if (doSlow) {
        iforcexSum = WarpReduce(tempStorage[warpId]).Sum(iforceSlow.x);
        NAMD_WARP_SYNC(WARP_FULL_MASK);
        iforceySum = WarpReduce(tempStorage[warpId]).Sum(iforceSlow.y);
        NAMD_WARP_SYNC(WARP_FULL_MASK);
        iforcezSum = WarpReduce(tempStorage[warpId]).Sum(iforceSlow.z);
        NAMD_WARP_SYNC(WARP_FULL_MASK);
        if (wid == 0) {
          virialEnergy[itileList].forceSlowx = iforcexSum;
          virialEnergy[itileList].forceSlowy = iforceySum;
          virialEnergy[itileList].forceSlowz = iforcezSum;
        }
      }
    }

    // Reduce energy
    if (doEnergy) {
      // NOTE: We must hand write these warp-wide reductions to avoid excess register spillage
      //       (Why does CUB suck here?)
#pragma unroll
      for (int i=WARPSIZE/2;i >= 1;i/=2) {
        energyVdw += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw, i, WARPSIZE);
        energyElec += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyElec, i, WARPSIZE);
        if(doFEP) energyVdw_s += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw_s, i, WARPSIZE);
        if(doFEP) energyElec_s += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyElec_s, i, WARPSIZE);
        if(doTI){
           energyVdw_ti_1  += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw_ti_1, i, WARPSIZE);
           energyVdw_ti_2  += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyVdw_ti_2, i, WARPSIZE);
           energyElec_ti_1 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyElec_ti_1, i, WARPSIZE);
           energyElec_ti_2 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energyElec_ti_2, i, WARPSIZE);
        }
        if (doSlow){ 
          energySlow += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow, i, WARPSIZE);
          if(doFEP) energySlow_s += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow_s, i, WARPSIZE);
          if(doTI){
            energySlow_ti_1 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow_ti_1, i, WARPSIZE);
            energySlow_ti_2 += WARP_SHUFFLE_XOR(WARP_FULL_MASK, energySlow_ti_2, i, WARPSIZE);
          }
        }
      }

      if (threadIdx.x % WARPSIZE == 0) {
        virialEnergy[itileList].energyVdw  = energyVdw;
        virialEnergy[itileList].energyElec = energyElec;
        if (doFEP) virialEnergy[itileList].energyVdw_s  = energyVdw_s;
        if (doFEP) virialEnergy[itileList].energyElec_s = energyElec_s;
        if(doTI){
           virialEnergy[itileList].energyVdw_ti_1   = energyVdw_ti_1;
           virialEnergy[itileList].energyVdw_ti_2   = energyVdw_ti_2;
           virialEnergy[itileList].energyElec_ti_1  = energyElec_ti_1;
           virialEnergy[itileList].energyElec_ti_2  = energyElec_ti_2;
        }
        if (doSlow) {
          virialEnergy[itileList].energySlow = energySlow;
          if(doFEP) virialEnergy[itileList].energySlow_s = energySlow_s;
          if (doTI){
            virialEnergy[itileList].energySlow_ti_1 = energySlow_ti_1;
            virialEnergy[itileList].energySlow_ti_2 = energySlow_ti_2;
          }
        }
      }
    }
    // XXX TODO: Disable streaming and see what happens
    // Let's try to set
    if (doStreaming) {
      // Make sure devForces and devForcesSlow have been written into device memory
      NAMD_WARP_SYNC(WARP_FULL_MASK);
      __threadfence();

      int patchDone[2] = {false, false};
      const int wid = threadIdx.x % WARPSIZE;
      if (wid == 0) {
        int patchCountOld0 = atomicInc(&patchNumCount[patchInd.x], (unsigned int)(patchNumList.x-1));
        patchDone[0] = (patchCountOld0 + 1 == patchNumList.x);
        if (patchInd.x != patchInd.y) {
          int patchCountOld1 = atomicInc(&patchNumCount[patchInd.y], (unsigned int)(patchNumList.y-1));
          patchDone[1] = (patchCountOld1 + 1 == patchNumList.y);
        }
      }

      patchDone[0] = NAMD_WARP_ANY(WARP_FULL_MASK, patchDone[0]);
      patchDone[1] = NAMD_WARP_ANY(WARP_FULL_MASK, patchDone[1]);

      if (patchDone[0]) {
        // Patch 1 is done, write onto host-mapped memory
        CudaPatchRecord patch = cudaPatches[patchInd.x];
        int start = patch.atomStart;
        int end   = start + patch.numAtoms;
        for (int i=start+wid;i < end;i+=WARPSIZE) {
          mapForces[i] = make_float4(devForce_x[i],
              devForce_y[i], devForce_z[i], devForce_w[i]);
          if (doSlow) {
            mapForcesSlow[i] = make_float4(devForceSlow_x[i],
                devForceSlow_y[i], devForceSlow_z[i], devForceSlow_w[i]);
          }
        }
      }
      if (patchDone[1]) {
        // Patch 2 is done
        CudaPatchRecord patch = cudaPatches[patchInd.y];
        int start = patch.atomStart;
        int end   = start + patch.numAtoms;
        for (int i=start+wid;i < end;i+=WARPSIZE) {
          mapForces[i] = make_float4(devForce_x[i],
              devForce_y[i], devForce_z[i], devForce_w[i]);
          if (doSlow) {
            mapForcesSlow[i] = make_float4(devForceSlow_x[i],
                devForceSlow_y[i], devForceSlow_z[i], devForceSlow_w[i]);
          }
        }
      }

      if (patchDone[0] || patchDone[1]) {
        // Make sure mapForces and mapForcesSlow are up-to-date
        NAMD_WARP_SYNC(WARP_FULL_MASK);
        __threadfence_system();
        // Add patch into "patchReadyQueue"
        if (wid == 0) {
          if (patchDone[0]) {
            int ind = atomicAdd(&tileListStat->patchReadyQueueCount, 1);
            // int ind = atomicInc((unsigned int *)&mapPatchReadyQueue[numPatches], numPatches-1);
            mapPatchReadyQueue[ind] = patchInd.x;
          }
          if (patchDone[1]) {
            int ind = atomicAdd(&tileListStat->patchReadyQueueCount, 1);
            // int ind = atomicInc((unsigned int *)&mapPatchReadyQueue[numPatches], numPatches-1);
            mapPatchReadyQueue[ind] = patchInd.y;
          }
        }
      }
    }

    if (doStreaming && outputOrder != NULL && threadIdx.x % WARPSIZE == 0) {
      int index = atomicAdd(&tileListStat->outputOrderIndex, 1);
      outputOrder[index] = itileList;
    }
  } // if (itileList < numTileLists)
}

//
// Finish up - reduce virials from nonbonded kernel
//
__global__ void reduceNonbondedVirialKernel(const bool doSlow,
  const int atomStorageSize,
  const float4* __restrict__ xyzq,
  const float4* __restrict__ devForces, const float4* __restrict__ devForcesSlow,
  VirialEnergy* __restrict__ virialEnergy) {

  for (int ibase = blockIdx.x*blockDim.x;ibase < atomStorageSize;ibase += blockDim.x*gridDim.x)
  {
    int i = ibase + threadIdx.x;

    // Set to zero to avoid nan*0
    float4 pos;
    pos.x = 0.0f;
    pos.y = 0.0f;
    pos.z = 0.0f;
    float4 force, forceSlow;
    force.x = 0.0f;
    force.y = 0.0f;
    force.z = 0.0f;
    forceSlow.x = 0.0f;
    forceSlow.y = 0.0f;
    forceSlow.z = 0.0f;
    if (i < atomStorageSize) {
      pos = xyzq[i];
      force = devForces[i];
      if (doSlow) forceSlow = devForcesSlow[i];
    }
    // Reduce across the entire thread block
    float vxxt = force.x*pos.x;
    float vxyt = force.x*pos.y;
    float vxzt = force.x*pos.z;
    float vyxt = force.y*pos.x;
    float vyyt = force.y*pos.y;
    float vyzt = force.y*pos.z;
    float vzxt = force.z*pos.x;
    float vzyt = force.z*pos.y;
    float vzzt = force.z*pos.z;

    const int bin = blockIdx.x % ATOMIC_BINS;

    typedef cub::BlockReduce<float, REDUCENONBONDEDVIRIALKERNEL_NUM_WARP*WARPSIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    float vxx = BlockReduce(tempStorage).Sum(vxxt); BLOCK_SYNC;
    float vxy = BlockReduce(tempStorage).Sum(vxyt); BLOCK_SYNC;
    float vxz = BlockReduce(tempStorage).Sum(vxzt); BLOCK_SYNC;
    float vyx = BlockReduce(tempStorage).Sum(vyxt); BLOCK_SYNC;
    float vyy = BlockReduce(tempStorage).Sum(vyyt); BLOCK_SYNC;
    float vyz = BlockReduce(tempStorage).Sum(vyzt); BLOCK_SYNC;
    float vzx = BlockReduce(tempStorage).Sum(vzxt); BLOCK_SYNC;
    float vzy = BlockReduce(tempStorage).Sum(vzyt); BLOCK_SYNC;
    float vzz = BlockReduce(tempStorage).Sum(vzzt); BLOCK_SYNC;
    if (threadIdx.x == 0) {
      atomicAdd(&virialEnergy[bin].virial[0], (double)vxx);
      atomicAdd(&virialEnergy[bin].virial[1], (double)vxy);
      atomicAdd(&virialEnergy[bin].virial[2], (double)vxz);
      atomicAdd(&virialEnergy[bin].virial[3], (double)vyx);
      atomicAdd(&virialEnergy[bin].virial[4], (double)vyy);
      atomicAdd(&virialEnergy[bin].virial[5], (double)vyz);
      atomicAdd(&virialEnergy[bin].virial[6], (double)vzx);
      atomicAdd(&virialEnergy[bin].virial[7], (double)vzy);
      atomicAdd(&virialEnergy[bin].virial[8], (double)vzz);
    }

    if (doSlow) {
      // if (isnan(forceSlow.x) || isnan(forceSlow.y) || isnan(forceSlow.z))
      float vxxSlowt = forceSlow.x*pos.x;
      float vxySlowt = forceSlow.x*pos.y;
      float vxzSlowt = forceSlow.x*pos.z;
      float vyxSlowt = forceSlow.y*pos.x;
      float vyySlowt = forceSlow.y*pos.y;
      float vyzSlowt = forceSlow.y*pos.z;
      float vzxSlowt = forceSlow.z*pos.x;
      float vzySlowt = forceSlow.z*pos.y;
      float vzzSlowt = forceSlow.z*pos.z;
      float vxxSlow = BlockReduce(tempStorage).Sum(vxxSlowt); BLOCK_SYNC;
      float vxySlow = BlockReduce(tempStorage).Sum(vxySlowt); BLOCK_SYNC;
      float vxzSlow = BlockReduce(tempStorage).Sum(vxzSlowt); BLOCK_SYNC;
      float vyxSlow = BlockReduce(tempStorage).Sum(vyxSlowt); BLOCK_SYNC;
      float vyySlow = BlockReduce(tempStorage).Sum(vyySlowt); BLOCK_SYNC;
      float vyzSlow = BlockReduce(tempStorage).Sum(vyzSlowt); BLOCK_SYNC;
      float vzxSlow = BlockReduce(tempStorage).Sum(vzxSlowt); BLOCK_SYNC;
      float vzySlow = BlockReduce(tempStorage).Sum(vzySlowt); BLOCK_SYNC;
      float vzzSlow = BlockReduce(tempStorage).Sum(vzzSlowt); BLOCK_SYNC;
      if (threadIdx.x == 0) {
        atomicAdd(&virialEnergy[bin].virialSlow[0], (double)vxxSlow);
        atomicAdd(&virialEnergy[bin].virialSlow[1], (double)vxySlow);
        atomicAdd(&virialEnergy[bin].virialSlow[2], (double)vxzSlow);
        atomicAdd(&virialEnergy[bin].virialSlow[3], (double)vyxSlow);
        atomicAdd(&virialEnergy[bin].virialSlow[4], (double)vyySlow);
        atomicAdd(&virialEnergy[bin].virialSlow[5], (double)vyzSlow);
        atomicAdd(&virialEnergy[bin].virialSlow[6], (double)vzxSlow);
        atomicAdd(&virialEnergy[bin].virialSlow[7], (double)vzySlow);
        atomicAdd(&virialEnergy[bin].virialSlow[8], (double)vzzSlow);
      }
    }
  
  }
}

__global__ void reduceVirialEnergyKernel(
  const bool doEnergy, const bool doVirial, const bool doSlow,
  const int numTileLists,
  const TileListVirialEnergy* __restrict__ tileListVirialEnergy,
  VirialEnergy* __restrict__ virialEnergy) {

  for (int ibase = blockIdx.x*blockDim.x;ibase < numTileLists;ibase += blockDim.x*gridDim.x)
  {
    int itileList = ibase + threadIdx.x;
    TileListVirialEnergy ve;
    if (itileList < numTileLists) {
      ve = tileListVirialEnergy[itileList];
    } else {
      // Set to zero to avoid nan*0
      if (doVirial) {
        ve.shx = 0.0f;
        ve.shy = 0.0f;
        ve.shz = 0.0f;
        ve.forcex = 0.0f;
        ve.forcey = 0.0f;
        ve.forcez = 0.0f;
        ve.forceSlowx = 0.0f;
        ve.forceSlowy = 0.0f;
        ve.forceSlowz = 0.0f;
      }
      if (doEnergy) {
         ve.energyVdw    = 0.0;
         ve.energyVdw_s  = 0.0;
         ve.energyElec   = 0.0;
         ve.energySlow   = 0.0;
         ve.energyElec_s = 0.0;
         ve.energySlow_s = 0.0;
         
         /* TI stuff */
         ve.energyVdw_ti_1 = 0.0;
         ve.energyVdw_ti_2 = 0.0;
         ve.energyElec_ti_1 = 0.0;
         ve.energyElec_ti_2 = 0.0;
         ve.energySlow_ti_1 = 0.0;
         ve.energySlow_ti_2 = 0.0;
        // ve.energyGBIS = 0.0;
      }
    }

    const int bin = blockIdx.x % ATOMIC_BINS;

    if (doVirial) {
      typedef cub::BlockReduce<float, REDUCEVIRIALENERGYKERNEL_NUM_WARP*WARPSIZE> BlockReduce;
      __shared__ typename BlockReduce::TempStorage tempStorage;
      float vxxt = ve.forcex*ve.shx;
      float vxyt = ve.forcex*ve.shy;
      float vxzt = ve.forcex*ve.shz;
      float vyxt = ve.forcey*ve.shx;
      float vyyt = ve.forcey*ve.shy;
      float vyzt = ve.forcey*ve.shz;
      float vzxt = ve.forcez*ve.shx;
      float vzyt = ve.forcez*ve.shy;
      float vzzt = ve.forcez*ve.shz;
      float vxx = BlockReduce(tempStorage).Sum(vxxt); BLOCK_SYNC;
      float vxy = BlockReduce(tempStorage).Sum(vxyt); BLOCK_SYNC;
      float vxz = BlockReduce(tempStorage).Sum(vxzt); BLOCK_SYNC;
      float vyx = BlockReduce(tempStorage).Sum(vyxt); BLOCK_SYNC;
      float vyy = BlockReduce(tempStorage).Sum(vyyt); BLOCK_SYNC;
      float vyz = BlockReduce(tempStorage).Sum(vyzt); BLOCK_SYNC;
      float vzx = BlockReduce(tempStorage).Sum(vzxt); BLOCK_SYNC;
      float vzy = BlockReduce(tempStorage).Sum(vzyt); BLOCK_SYNC;
      float vzz = BlockReduce(tempStorage).Sum(vzzt); BLOCK_SYNC;
      if (threadIdx.x == 0) {
        atomicAdd(&virialEnergy[bin].virial[0], (double)vxx);
        atomicAdd(&virialEnergy[bin].virial[1], (double)vxy);
        atomicAdd(&virialEnergy[bin].virial[2], (double)vxz);
        atomicAdd(&virialEnergy[bin].virial[3], (double)vyx);
        atomicAdd(&virialEnergy[bin].virial[4], (double)vyy);
        atomicAdd(&virialEnergy[bin].virial[5], (double)vyz);
        atomicAdd(&virialEnergy[bin].virial[6], (double)vzx);
        atomicAdd(&virialEnergy[bin].virial[7], (double)vzy);
        atomicAdd(&virialEnergy[bin].virial[8], (double)vzz);
      }

      if (doSlow) {
        typedef cub::BlockReduce<float, REDUCEVIRIALENERGYKERNEL_NUM_WARP*WARPSIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage tempStorage;
        float vxxt = ve.forceSlowx*ve.shx;
        float vxyt = ve.forceSlowx*ve.shy;
        float vxzt = ve.forceSlowx*ve.shz;
        float vyxt = ve.forceSlowy*ve.shx;
        float vyyt = ve.forceSlowy*ve.shy;
        float vyzt = ve.forceSlowy*ve.shz;
        float vzxt = ve.forceSlowz*ve.shx;
        float vzyt = ve.forceSlowz*ve.shy;
        float vzzt = ve.forceSlowz*ve.shz;
        float vxx = BlockReduce(tempStorage).Sum(vxxt); BLOCK_SYNC;
        float vxy = BlockReduce(tempStorage).Sum(vxyt); BLOCK_SYNC;
        float vxz = BlockReduce(tempStorage).Sum(vxzt); BLOCK_SYNC;
        float vyx = BlockReduce(tempStorage).Sum(vyxt); BLOCK_SYNC;
        float vyy = BlockReduce(tempStorage).Sum(vyyt); BLOCK_SYNC;
        float vyz = BlockReduce(tempStorage).Sum(vyzt); BLOCK_SYNC;
        float vzx = BlockReduce(tempStorage).Sum(vzxt); BLOCK_SYNC;
        float vzy = BlockReduce(tempStorage).Sum(vzyt); BLOCK_SYNC;
        float vzz = BlockReduce(tempStorage).Sum(vzzt); BLOCK_SYNC;
        if (threadIdx.x == 0) {
          atomicAdd(&virialEnergy[bin].virialSlow[0], (double)vxx);
          atomicAdd(&virialEnergy[bin].virialSlow[1], (double)vxy);
          atomicAdd(&virialEnergy[bin].virialSlow[2], (double)vxz);
          atomicAdd(&virialEnergy[bin].virialSlow[3], (double)vyx);
          atomicAdd(&virialEnergy[bin].virialSlow[4], (double)vyy);
          atomicAdd(&virialEnergy[bin].virialSlow[5], (double)vyz);
          atomicAdd(&virialEnergy[bin].virialSlow[6], (double)vzx);
          atomicAdd(&virialEnergy[bin].virialSlow[7], (double)vzy);
          atomicAdd(&virialEnergy[bin].virialSlow[8], (double)vzz);
        }
      }
    }

    if (doEnergy) {
      typedef cub::BlockReduce<double, REDUCEVIRIALENERGYKERNEL_NUM_WARP*WARPSIZE> BlockReduce;
      /* Maybe we should guard the TI and FEP energies, since those are not to be calculated on regular MDs */
      __shared__ typename BlockReduce::TempStorage tempStorage;
      double energyVdw       = BlockReduce(tempStorage).Sum(ve.energyVdw); BLOCK_SYNC;
      double energyVdw_s     = BlockReduce(tempStorage).Sum(ve.energyVdw_s); BLOCK_SYNC;
      double energyElec      = BlockReduce(tempStorage).Sum(ve.energyElec); BLOCK_SYNC;
      double energyElec_s    = BlockReduce(tempStorage).Sum(ve.energyElec_s); BLOCK_SYNC;
      double energyVdw_ti_1  = BlockReduce(tempStorage).Sum(ve.energyVdw_ti_1); BLOCK_SYNC;
      double energyVdw_ti_2  = BlockReduce(tempStorage).Sum(ve.energyVdw_ti_2); BLOCK_SYNC;
      double energyElec_ti_1 = BlockReduce(tempStorage).Sum(ve.energyElec_ti_1); BLOCK_SYNC;
      double energyElec_ti_2 = BlockReduce(tempStorage).Sum(ve.energyElec_ti_2); BLOCK_SYNC;
      if (threadIdx.x == 0){
          atomicAdd(&virialEnergy[bin].energyVdw,       energyVdw);
          atomicAdd(&virialEnergy[bin].energyVdw_s,     energyVdw_s);
          atomicAdd(&virialEnergy[bin].energyElec,      energyElec);
          atomicAdd(&virialEnergy[bin].energyElec_s,    energyElec_s);
          atomicAdd(&virialEnergy[bin].energyVdw_ti_1,  energyVdw_ti_1);
          atomicAdd(&virialEnergy[bin].energyVdw_ti_2,  energyVdw_ti_2);
          atomicAdd(&virialEnergy[bin].energyElec_ti_1, energyElec_ti_1);
          atomicAdd(&virialEnergy[bin].energyElec_ti_2, energyElec_ti_2);
      }
      if (doSlow) {
        double energySlow      = BlockReduce(tempStorage).Sum(ve.energySlow); BLOCK_SYNC;
        double energySlow_s    = BlockReduce(tempStorage).Sum(ve.energySlow_s); BLOCK_SYNC;
        double energySlow_ti_1 = BlockReduce(tempStorage).Sum(ve.energySlow_ti_1); BLOCK_SYNC;
        double energySlow_ti_2 = BlockReduce(tempStorage).Sum(ve.energySlow_ti_2); BLOCK_SYNC;
        if (threadIdx.x == 0) {
          atomicAdd(&virialEnergy[bin].energySlow,      energySlow);
          atomicAdd(&virialEnergy[bin].energySlow_s,    energySlow_s);
          atomicAdd(&virialEnergy[bin].energySlow_ti_1, energySlow_ti_1);
          atomicAdd(&virialEnergy[bin].energySlow_ti_2, energySlow_ti_2);
        }
      }
    }

  }

}

__global__ void reduceGBISEnergyKernel(const int numTileLists,
  const TileListVirialEnergy* __restrict__ tileListVirialEnergy,
  VirialEnergy* __restrict__ virialEnergy) {

  for (int ibase = blockIdx.x*blockDim.x;ibase < numTileLists;ibase += blockDim.x*gridDim.x)
  {
    int itileList = ibase + threadIdx.x;
    double energyGBISt = 0.0;
    if (itileList < numTileLists) {
      energyGBISt = tileListVirialEnergy[itileList].energyGBIS;
    }

    const int bin = blockIdx.x % ATOMIC_BINS;

    typedef cub::BlockReduce<double, REDUCEGBISENERGYKERNEL_NUM_WARP*WARPSIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    double energyGBIS = BlockReduce(tempStorage).Sum(energyGBISt); BLOCK_SYNC;
    if (threadIdx.x == 0) atomicAdd(&virialEnergy[bin].energyGBIS, energyGBIS);
  }
}

__global__ void reduceNonbondedBinsKernel(
  const bool doVirial,
  const bool doEnergy,
  const bool doSlow,
  const bool doGBIS,
  VirialEnergy* __restrict__ virialEnergy) {

  const int bin = threadIdx.x;

  typedef cub::WarpReduce<double, (ATOMIC_BINS > 1 ? ATOMIC_BINS : 2)> WarpReduce;
  __shared__ typename WarpReduce::TempStorage tempStorage;

  if (doVirial) {
    double vxx = WarpReduce(tempStorage).Sum(virialEnergy[bin].virial[0]);
    double vxy = WarpReduce(tempStorage).Sum(virialEnergy[bin].virial[1]);
    double vxz = WarpReduce(tempStorage).Sum(virialEnergy[bin].virial[2]);
    double vyx = WarpReduce(tempStorage).Sum(virialEnergy[bin].virial[3]);
    double vyy = WarpReduce(tempStorage).Sum(virialEnergy[bin].virial[4]);
    double vyz = WarpReduce(tempStorage).Sum(virialEnergy[bin].virial[5]);
    double vzx = WarpReduce(tempStorage).Sum(virialEnergy[bin].virial[6]);
    double vzy = WarpReduce(tempStorage).Sum(virialEnergy[bin].virial[7]);
    double vzz = WarpReduce(tempStorage).Sum(virialEnergy[bin].virial[8]);
    if (threadIdx.x == 0) {
      virialEnergy->virial[0] = vxx;
      virialEnergy->virial[1] = vxy;
      virialEnergy->virial[2] = vxz;
      virialEnergy->virial[3] = vyx;
      virialEnergy->virial[4] = vyy;
      virialEnergy->virial[5] = vyz;
      virialEnergy->virial[6] = vzx;
      virialEnergy->virial[7] = vzy;
      virialEnergy->virial[8] = vzz;
    }

    if (doSlow) {
      double vxxSlow = WarpReduce(tempStorage).Sum(virialEnergy[bin].virialSlow[0]);
      double vxySlow = WarpReduce(tempStorage).Sum(virialEnergy[bin].virialSlow[1]);
      double vxzSlow = WarpReduce(tempStorage).Sum(virialEnergy[bin].virialSlow[2]);
      double vyxSlow = WarpReduce(tempStorage).Sum(virialEnergy[bin].virialSlow[3]);
      double vyySlow = WarpReduce(tempStorage).Sum(virialEnergy[bin].virialSlow[4]);
      double vyzSlow = WarpReduce(tempStorage).Sum(virialEnergy[bin].virialSlow[5]);
      double vzxSlow = WarpReduce(tempStorage).Sum(virialEnergy[bin].virialSlow[6]);
      double vzySlow = WarpReduce(tempStorage).Sum(virialEnergy[bin].virialSlow[7]);
      double vzzSlow = WarpReduce(tempStorage).Sum(virialEnergy[bin].virialSlow[8]);
      if (threadIdx.x == 0) {
        virialEnergy->virialSlow[0] = vxxSlow;
        virialEnergy->virialSlow[1] = vxySlow;
        virialEnergy->virialSlow[2] = vxzSlow;
        virialEnergy->virialSlow[3] = vyxSlow;
        virialEnergy->virialSlow[4] = vyySlow;
        virialEnergy->virialSlow[5] = vyzSlow;
        virialEnergy->virialSlow[6] = vzxSlow;
        virialEnergy->virialSlow[7] = vzySlow;
        virialEnergy->virialSlow[8] = vzzSlow;
      }
    }
  }

  if (doEnergy) {
    double energyVdw       = WarpReduce(tempStorage).Sum(virialEnergy[bin].energyVdw);
    double energyVdw_s     = WarpReduce(tempStorage).Sum(virialEnergy[bin].energyVdw_s);
    double energyElec      = WarpReduce(tempStorage).Sum(virialEnergy[bin].energyElec);
    double energyElec_s    = WarpReduce(tempStorage).Sum(virialEnergy[bin].energyElec_s);
    double energyVdw_ti_1  = WarpReduce(tempStorage).Sum(virialEnergy[bin].energyVdw_ti_1);
    double energyVdw_ti_2  = WarpReduce(tempStorage).Sum(virialEnergy[bin].energyVdw_ti_2);
    double energyElec_ti_1 = WarpReduce(tempStorage).Sum(virialEnergy[bin].energyElec_ti_1);
    double energyElec_ti_2 = WarpReduce(tempStorage).Sum(virialEnergy[bin].energyElec_ti_2);
    if (threadIdx.x == 0) {
      virialEnergy->energyVdw =       energyVdw;
      virialEnergy->energyVdw_s =     energyVdw_s;
      virialEnergy->energyElec =      energyElec;
      virialEnergy->energyElec_s =    energyElec_s;
      virialEnergy->energyVdw_ti_1 =  energyVdw_ti_1;
      virialEnergy->energyVdw_ti_2 =  energyVdw_ti_2;
      virialEnergy->energyElec_ti_1 = energyElec_ti_1;
      virialEnergy->energyElec_ti_2 = energyElec_ti_2;
    }
    if (doSlow) {
      double energySlow      = WarpReduce(tempStorage).Sum(virialEnergy[bin].energySlow);
      double energySlow_s    = WarpReduce(tempStorage).Sum(virialEnergy[bin].energySlow_s);
      double energySlow_ti_1 = WarpReduce(tempStorage).Sum(virialEnergy[bin].energySlow_ti_1);
      double energySlow_ti_2 = WarpReduce(tempStorage).Sum(virialEnergy[bin].energySlow_ti_2);
      if (threadIdx.x == 0) {
        virialEnergy->energySlow =      energySlow;
        virialEnergy->energySlow_s =    energySlow_s;
        virialEnergy->energySlow_ti_1 = energySlow_ti_1;
        virialEnergy->energySlow_ti_2 = energySlow_ti_2;
      }
    }
    if (doGBIS) {
      double energyGBIS = WarpReduce(tempStorage).Sum(virialEnergy[bin].energyGBIS);
      if (threadIdx.x == 0) {
        virialEnergy->energyGBIS = energyGBIS;
      }
    }
  }
}

// ##############################################################################################
// ##############################################################################################
// ##############################################################################################

CudaComputeNonbondedKernel::CudaComputeNonbondedKernel(int deviceID, CudaNonbondedTables& cudaNonbondedTables,
  bool doStreaming) : deviceID(deviceID), cudaNonbondedTables(cudaNonbondedTables), doStreaming(doStreaming) {
  
  cudaCheck(cudaSetDevice(deviceID));

  overflowExclusions = NULL;
  overflowExclusionsSize = 0;

  exclIndexMaxDiff = NULL;
  exclIndexMaxDiffSize = 0;

  atomIndex = NULL;
  atomIndexSize = 0;

  vdwTypes = NULL;
  vdwTypesSize = 0;

  patchNumCount = NULL;
  patchNumCountSize = 0;

  patchReadyQueue = NULL;
  patchReadyQueueSize = 0;

  force_x = force_y = force_z = force_w = NULL;
  forceSize = 0;
  forceSlow_x = forceSlow_y = forceSlow_z = forceSlow_w = NULL;
  forceSlowSize = 0;
}

void CudaComputeNonbondedKernel::reallocate_forceSOA(int atomStorageSize)
{
#if 0
  reallocate_device<float>(&force_x, &forceSize, atomStorageSize, 1.4f);
  reallocate_device<float>(&force_y, &forceSize, atomStorageSize, 1.4f);
  reallocate_device<float>(&force_z, &forceSize, atomStorageSize, 1.4f);
  reallocate_device<float>(&force_w, &forceSize, atomStorageSize, 1.4f);
  reallocate_device<float>(&forceSlow_x, &forceSlowSize, atomStorageSize, 1.4f);
  reallocate_device<float>(&forceSlow_y, &forceSlowSize, atomStorageSize, 1.4f);
  reallocate_device<float>(&forceSlow_z, &forceSlowSize, atomStorageSize, 1.4f);
  reallocate_device<float>(&forceSlow_w, &forceSlowSize, atomStorageSize, 1.4f);  
#else
  reallocate_device<float>(&force_x, &forceSize, atomStorageSize*8, 1.4f);
  force_y = force_x + atomStorageSize;
  force_z = force_y + atomStorageSize;
  force_w = force_z + atomStorageSize;
  forceSlow_x = force_w + atomStorageSize;
  forceSlow_y = forceSlow_x + atomStorageSize;
  forceSlow_z = forceSlow_y + atomStorageSize;
  forceSlow_w = forceSlow_z + atomStorageSize;
#endif
}

CudaComputeNonbondedKernel::~CudaComputeNonbondedKernel() {
  cudaCheck(cudaSetDevice(deviceID));
  if (overflowExclusions != NULL) deallocate_device<unsigned int>(&overflowExclusions);
  if (exclIndexMaxDiff != NULL) deallocate_device<int2>(&exclIndexMaxDiff);
  if (atomIndex != NULL) deallocate_device<int>(&atomIndex);
  if (vdwTypes != NULL) deallocate_device<int>(&vdwTypes);
  if (patchNumCount != NULL) deallocate_device<unsigned int>(&patchNumCount);
  if (patchReadyQueue != NULL) deallocate_host<int>(&patchReadyQueue);
#if 0
  if (force_x != NULL) deallocate_device<float>(&force_x);
  if (force_y != NULL) deallocate_device<float>(&force_y);
  if (force_z != NULL) deallocate_device<float>(&force_z);
  if (force_w != NULL) deallocate_device<float>(&force_w);
  if (forceSlow_x != NULL) deallocate_device<float>(&forceSlow_x);
  if (forceSlow_y != NULL) deallocate_device<float>(&forceSlow_y);
  if (forceSlow_z != NULL) deallocate_device<float>(&forceSlow_z);
  if (forceSlow_w != NULL) deallocate_device<float>(&forceSlow_w);  
#else
  if (force_x != NULL) deallocate_device<float>(&force_x);
#endif
}

void CudaComputeNonbondedKernel::updateVdwTypesExcl(const int atomStorageSize, const int* h_vdwTypes,
  const int2* h_exclIndexMaxDiff, const int* h_atomIndex, cudaStream_t stream) {

  reallocate_device<int>(&vdwTypes, &vdwTypesSize, atomStorageSize, OVERALLOC);
  reallocate_device<int2>(&exclIndexMaxDiff, &exclIndexMaxDiffSize, atomStorageSize, OVERALLOC);
  reallocate_device<int>(&atomIndex, &atomIndexSize, atomStorageSize, OVERALLOC);

  copy_HtoD<int>(h_vdwTypes, vdwTypes, atomStorageSize, stream);
  copy_HtoD<int2>(h_exclIndexMaxDiff, exclIndexMaxDiff, atomStorageSize, stream);
  copy_HtoD<int>(h_atomIndex, atomIndex, atomStorageSize, stream);
}

int* CudaComputeNonbondedKernel::getPatchReadyQueue() {
  if (!doStreaming) {
    NAMD_die("CudaComputeNonbondedKernel::getPatchReadyQueue() called on non-streaming kernel");
  }
  return patchReadyQueue;
}

template <int doSlow>
__global__ void transposeForcesKernel(float4 *f, float4 *fSlow,
                                      float *fx, float *fy, float *fz, float *fw,
                                      float *fSlowx, float *fSlowy, float *fSlowz, float *fSloww,
                                      int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n) {
    f[tid] = make_float4(fx[tid], fy[tid], fz[tid], fw[tid]);
    fx[tid] = 0.f; fy[tid] = 0.f; fz[tid] = 0.f; fw[tid] = 0.f;
    if (doSlow) {
      fSlow[tid] = make_float4(fSlowx[tid], fSlowy[tid], fSlowz[tid], fSloww[tid]);
      fSlowx[tid] = 0.f; fSlowy[tid] = 0.f; fSlowz[tid] = 0.f; fSloww[tid] = 0.f;
    }
  }
}



void CudaComputeNonbondedKernel::nonbondedForce(CudaTileListKernel& tlKernel,
  const int atomStorageSize, const bool atomsChanged, const bool doMinimize, 
  const bool doPairlist,
  const bool doEnergy, const bool doVirial, const bool doSlow, const bool doAlch, 
  const bool doAlchVdwForceSwitching,
  const bool doFEP, const bool doTI, const bool doTable,
  const float3 lata, const float3 latb, const float3 latc,
  const float4* h_xyzq, const float cutoff2, 
  const CudaNBConstants     nbConstants, 
  float4* d_forces, float4* d_forcesSlow,
  float4* h_forces, float4* h_forcesSlow, AlchData *srcFlags, 
  bool lambdaWindowUpdated, char *part,
  bool CUDASOAintegrator,  bool useDeviceMigration,
  cudaStream_t stream) {  

#ifdef NODEGROUP_FORCE_REGISTER
  if (!atomsChanged && !CUDASOAintegrator) copy_HtoD<float4>(h_xyzq, tlKernel.get_xyzq(), atomStorageSize, stream);
#else
  if (!doPairlist) copy_HtoD<float4>(h_xyzq, tlKernel.get_xyzq(), atomStorageSize, stream);
#endif

  if (doAlch){
    // Copy partition to device. This is not necessary if both CUDASOAintegrator and useDeviceMigration
    // are true.
    if (doPairlist && (!CUDASOAintegrator || !useDeviceMigration)) { 
      copy_HtoD< char>(part, tlKernel.get_part(), atomStorageSize, stream);
    }
    //Copies flags to constant memory
    if(lambdaWindowUpdated) cudaCheck(cudaMemcpyToSymbol(alchflags, srcFlags, sizeof(AlchData)));
  }

  // XXX TODO: Get rid of the clears
  if(1){
     if (doStreaming)  tlKernel.clearTileListStat(stream);
     if(atomsChanged || doMinimize){
      clear_device_array<float>(force_x, atomStorageSize*4, stream);
      if(doSlow) clear_device_array<float>(forceSlow_x, atomStorageSize*4, stream);
     }
  }

  // --- streaming ----
  float4* m_forces = NULL;
  float4* m_forcesSlow = NULL;
  int* m_patchReadyQueue = NULL;
  int numPatches = 0;
  unsigned long long *calculatedPairs;
  unsigned long long  *skippedPairs;
  unsigned int* patchNumCountPtr = NULL;

  if (doStreaming) {
    numPatches = tlKernel.getNumPatches();
    if (reallocate_device<unsigned int>(&patchNumCount, &patchNumCountSize, numPatches)) {
      // If re-allocated, clear array
      clear_device_array<unsigned int>(patchNumCount, numPatches, stream);
    }
    patchNumCountPtr = patchNumCount;
    bool re = reallocate_host<int>(&patchReadyQueue, &patchReadyQueueSize, numPatches, cudaHostAllocMapped);
    if (re) {
      // If re-allocated, re-set to "-1"
      for (int i=0;i < numPatches;i++) patchReadyQueue[i] = -1;
    }
    cudaCheck(cudaHostGetDevicePointer((void**)&m_patchReadyQueue, patchReadyQueue, 0));
    cudaCheck(cudaHostGetDevicePointer((void**)&m_forces, h_forces, 0));
    cudaCheck(cudaHostGetDevicePointer((void**)&m_forcesSlow, h_forcesSlow, 0));
  }
  // -----------------

  if (doVirial || doEnergy) {
    tlKernel.setTileListVirialEnergyLength(tlKernel.getNumTileLists());
  }

  int shMemSize = 0;

  int* outputOrderPtr = tlKernel.getOutputOrder();

  int nwarp = NONBONDKERNEL_NUM_WARP;
  int nthread = WARPSIZE*nwarp;
  int start = 0;

#define APVERSION
#undef APVERSION

#ifdef APVERSION
#else
  int options = doEnergy + (doVirial << 1) + (doSlow << 2) +
    (doPairlist << 3) + (doAlch << 4) + (doFEP << 5) + (doTI << 6) + (doStreaming << 7) + (doTable << 8) + (doAlchVdwForceSwitching << 9);
#endif

  while (start < tlKernel.getNumTileLists()) {

    int nleft = tlKernel.getNumTileLists() - start;
    int nblock = min(deviceCUDA->getMaxNumBlocks(), (nleft-1)/nwarp+1);

#ifdef APVERSION
    
#ifdef USE_TABLE_ARRAYS
  #define VDW_TABLE_PARAMS cudaNonbondedTables.getVdwCoefTable()
  #define TABLE_PARAMS \
    cudaNonbondedTables.getForceTable(), cudaNonbondedTables.getEnergyTable()
#else
  #define VDW_TABLE_PARAMS cudaNonbondedTables.getVdwCoefTableTex()
  #define TABLE_PARAMS \
    cudaNonbondedTables.getForceTableTex(), cudaNonbondedTables.getEnergyTableTex()
#endif


#define CALL(DOENERGY, DOVIRIAL, DOSLOW, DOPAIRLIST, DOALCH, DOFEP, DOTI, DOSTREAMING, DOALCHWDWFORCESWITCHING) \
    nonbondedForceKernel<DOENERGY, DOVIRIAL, DOSLOW, DOPAIRLIST, DOALCH, DOFEP, DOTI, DOSTREAMING, DOALCHWDWFORCESWITCHING > \
  <<< nblock, nthread, shMemSize, stream >>>  \
  (start, tlKernel.getNumTileLists(), tlKernel.getTileLists(), tlKernel.getTileExcls(), tlKernel.getTileJatomStart(), \
    cudaNonbondedTables.getVdwCoefTableWidth(), \
    VDW_TABLE_PARAMS, \
    vdwTypes, lata, latb, latc, tlKernel.get_xyzq(), cutoff2, nbConstants, \
    TABLE_PARAMS, \
    tlKernel.get_plcutoff2(), tlKernel.getPatchPairs(), atomIndex, exclIndexMaxDiff, overflowExclusions, \
    tlKernel.getTileListDepth(), tlKernel.getTileListOrder(), tlKernel.getJtiles(), tlKernel.getTileListStatDevPtr(), \
    tlKernel.getBoundingBoxes(), d_forces, d_forcesSlow, \
    force_x, force_y, force_z, force_w, \
    forceSlow_x, forceSlow_y, forceSlow_z, forceSlow_w, \
    numPatches, patchNumCountPtr, tlKernel.getCudaPatches(), m_forces, m_forcesSlow, m_patchReadyQueue, \
    outputOrderPtr, tlKernel.getTileListVirialEnergy(), tlKernel.get_part(), calculatedPairs, skippedPairs); called=true

    bool called = false;
    if (doStreaming) {
      if(!doAlch){
      	if (!doEnergy && !doVirial && !doSlow && !doPairlist) CALL(0, 0, 0, 0, 0, 0, 0, 1, 0);
      	if (!doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(0, 0, 1, 0, 0, 0, 0, 1, 0);
      	if (!doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(0, 1, 0, 0, 0, 0, 0, 1, 0);
      	if (!doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(0, 1, 1, 0, 0, 0, 0, 1, 0);
      	if ( doEnergy && !doVirial && !doSlow && !doPairlist) CALL(1, 0, 0, 0, 0, 0, 0, 1, 0);
      	if ( doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(1, 0, 1, 0, 0, 0, 0, 1, 0);
      	if ( doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(1, 1, 0, 0, 0, 0, 0, 1, 0);
      	if ( doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(1, 1, 1, 0, 0, 0, 0, 1, 0);
      

      	if (!doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(0, 0, 0, 1, 0, 0, 0, 1, 0);
      	if (!doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(0, 0, 1, 1, 0, 0, 0, 1, 0);
      	if (!doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(0, 1, 0, 1, 0, 0, 0, 1, 0);
      	if (!doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(0, 1, 1, 1, 0, 0, 0, 1, 0);
      	if ( doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(1, 0, 0, 1, 0, 0, 0, 1, 0);
      	if ( doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(1, 0, 1, 1, 0, 0, 0, 1, 0);
      	if ( doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(1, 1, 0, 1, 0, 0, 0, 1, 0);
      	if ( doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(1, 1, 1, 1, 0, 0, 0, 1, 0);
      }else{
	if(doFEP){
          if (doAlchVdwForceSwitching) {
            if (!doEnergy && !doVirial && !doSlow && !doPairlist) CALL(0, 0, 0, 0, 1, 1, 0, 1, 1);
            if (!doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(0, 0, 1, 0, 1, 1, 0, 1, 1);
            if (!doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(0, 1, 0, 0, 1, 1, 0, 1, 1);
            if (!doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(0, 1, 1, 0, 1, 1, 0, 1, 1);
            if ( doEnergy && !doVirial && !doSlow && !doPairlist) CALL(1, 0, 0, 0, 1, 1, 0, 1, 1);
            if ( doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(1, 0, 1, 0, 1, 1, 0, 1, 1);
            if ( doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(1, 1, 0, 0, 1, 1, 0, 1, 1);
            if ( doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(1, 1, 1, 0, 1, 1, 0, 1, 1);

            if (!doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(0, 0, 0, 1, 1, 1, 0, 1, 1);
            if (!doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(0, 0, 1, 1, 1, 1, 0, 1, 1);
            if (!doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(0, 1, 0, 1, 1, 1, 0, 1, 1);
            if (!doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(0, 1, 1, 1, 1, 1, 0, 1, 1);
            if ( doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(1, 0, 0, 1, 1, 1, 0, 1, 1);
            if ( doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(1, 0, 1, 1, 1, 1, 0, 1, 1);
            if ( doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(1, 1, 0, 1, 1, 1, 0, 1, 1);
            if ( doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(1, 1, 1, 1, 1, 1, 0, 1, 1);
          } else {
            if (!doEnergy && !doVirial && !doSlow && !doPairlist) CALL(0, 0, 0, 0, 1, 1, 0, 1, 0);
            if (!doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(0, 0, 1, 0, 1, 1, 0, 1, 0);
            if (!doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(0, 1, 0, 0, 1, 1, 0, 1, 0);
            if (!doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(0, 1, 1, 0, 1, 1, 0, 1, 0);
            if ( doEnergy && !doVirial && !doSlow && !doPairlist) CALL(1, 0, 0, 0, 1, 1, 0, 1, 0);
            if ( doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(1, 0, 1, 0, 1, 1, 0, 1, 0);
            if ( doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(1, 1, 0, 0, 1, 1, 0, 1, 0);
            if ( doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(1, 1, 1, 0, 1, 1, 0, 1, 0);

            if (!doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(0, 0, 0, 1, 1, 1, 0, 1, 0);
            if (!doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(0, 0, 1, 1, 1, 1, 0, 1, 0);
            if (!doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(0, 1, 0, 1, 1, 1, 0, 1, 0);
            if (!doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(0, 1, 1, 1, 1, 1, 0, 1, 0);
            if ( doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(1, 0, 0, 1, 1, 1, 0, 1, 0);
            if ( doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(1, 0, 1, 1, 1, 1, 0, 1, 0);
            if ( doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(1, 1, 0, 1, 1, 1, 0, 1, 0);
            if ( doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(1, 1, 1, 1, 1, 1, 0, 1, 0);
          }
        }else{
          if (doAlchVdwForceSwitching) {
            if (!doEnergy && !doVirial && !doSlow && !doPairlist) CALL(0, 0, 0, 0, 1, 0, 1, 1, 1);
            if (!doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(0, 0, 1, 0, 1, 0, 1, 1, 1);
            if (!doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(0, 1, 0, 0, 1, 0, 1, 1, 1);
            if (!doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(0, 1, 1, 0, 1, 0, 1, 1, 1);
            if ( doEnergy && !doVirial && !doSlow && !doPairlist) CALL(1, 0, 0, 0, 1, 0, 1, 1, 1);
            if ( doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(1, 0, 1, 0, 1, 0, 1, 1, 1);
            if ( doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(1, 1, 0, 0, 1, 0, 1, 1, 1);
            if ( doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(1, 1, 1, 0, 1, 0, 1, 1, 1);

            if (!doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(0, 0, 0, 1, 1, 0, 1, 1, 1);
            if (!doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(0, 0, 1, 1, 1, 0, 1, 1, 1);
            if (!doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(0, 1, 0, 1, 1, 0, 1, 1, 1);
            if (!doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(0, 1, 1, 1, 1, 0, 1, 1, 1);
            if ( doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(1, 0, 0, 1, 1, 0, 1, 1, 1);
            if ( doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(1, 0, 1, 1, 1, 0, 1, 1, 1);
            if ( doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(1, 1, 0, 1, 1, 0, 1, 1, 1);
            if ( doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(1, 1, 1, 1, 1, 0, 1, 1, 1);
          } else {
            if (!doEnergy && !doVirial && !doSlow && !doPairlist) CALL(0, 0, 0, 0, 1, 0, 1, 1, 0);
            if (!doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(0, 0, 1, 0, 1, 0, 1, 1, 0);
            if (!doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(0, 1, 0, 0, 1, 0, 1, 1, 0);
            if (!doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(0, 1, 1, 0, 1, 0, 1, 1, 0);
            if ( doEnergy && !doVirial && !doSlow && !doPairlist) CALL(1, 0, 0, 0, 1, 0, 1, 1, 0);
            if ( doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(1, 0, 1, 0, 1, 0, 1, 1, 0);
            if ( doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(1, 1, 0, 0, 1, 0, 1, 1, 0);
            if ( doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(1, 1, 1, 0, 1, 0, 1, 1, 0);

            if (!doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(0, 0, 0, 1, 1, 0, 1, 1, 0);
            if (!doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(0, 0, 1, 1, 1, 0, 1, 1, 0);
            if (!doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(0, 1, 0, 1, 1, 0, 1, 1, 0);
            if (!doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(0, 1, 1, 1, 1, 0, 1, 1, 0);
            if ( doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(1, 0, 0, 1, 1, 0, 1, 1, 0);
            if ( doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(1, 0, 1, 1, 1, 0, 1, 1, 0);
            if ( doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(1, 1, 0, 1, 1, 0, 1, 1, 0);
            if ( doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(1, 1, 1, 1, 1, 0, 1, 1, 0);
          }
        }
      }
    } 
    else {
      if(!doAlch){
      	if (!doEnergy && !doVirial && !doSlow && !doPairlist) CALL(0, 0, 0, 0, 0, 0, 0, 0, 0);
      	if (!doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(0, 0, 1, 0, 0, 0, 0, 0, 0);
      	if (!doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(0, 1, 0, 0, 0, 0, 0, 0, 0);
      	if (!doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(0, 1, 1, 0, 0, 0, 0, 0, 0);
      	if ( doEnergy && !doVirial && !doSlow && !doPairlist) CALL(1, 0, 0, 0, 0, 0, 0, 0, 0);
      	if ( doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(1, 0, 1, 0, 0, 0, 0, 0, 0);
      	if ( doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(1, 1, 0, 0, 0, 0, 0, 0, 0);
      	if ( doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(1, 1, 1, 0, 0, 0, 0, 0, 0);
      

      	if (!doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(0, 0, 0, 1, 0, 0, 0, 0, 0);
      	if (!doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(0, 0, 1, 1, 0, 0, 0, 0, 0);
      	if (!doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(0, 1, 0, 1, 0, 0, 0, 0, 0);
      	if (!doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(0, 1, 1, 1, 0, 0, 0, 0, 0);
      	if ( doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(1, 0, 0, 1, 0, 0, 0, 0, 0);
      	if ( doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(1, 0, 1, 1, 0, 0, 0, 0, 0);
      	if ( doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(1, 1, 0, 1, 0, 0, 0, 0, 0);
      	if ( doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(1, 1, 1, 1, 0, 0, 0, 0, 0);
      }else{
	if(doFEP){
          if (doAlchVdwForceSwitching) {
            if (!doEnergy && !doVirial && !doSlow && !doPairlist) CALL(0, 0, 0, 0, 1, 1, 0, 0, 1);
            if (!doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(0, 0, 1, 0, 1, 1, 0, 0, 1);
            if (!doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(0, 1, 0, 0, 1, 1, 0, 0, 1);
            if (!doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(0, 1, 1, 0, 1, 1, 0, 0, 1);
            if ( doEnergy && !doVirial && !doSlow && !doPairlist) CALL(1, 0, 0, 0, 1, 1, 0, 0, 1);
            if ( doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(1, 0, 1, 0, 1, 1, 0, 0, 1);
            if ( doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(1, 1, 0, 0, 1, 1, 0, 0, 1);
            if ( doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(1, 1, 1, 0, 1, 1, 0, 0, 1);

            if (!doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(0, 0, 0, 1, 1, 1, 0, 0, 1);
            if (!doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(0, 0, 1, 1, 1, 1, 0, 0, 1);
            if (!doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(0, 1, 0, 1, 1, 1, 0, 0, 1);
            if (!doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(0, 1, 1, 1, 1, 1, 0, 0, 1);
            if ( doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(1, 0, 0, 1, 1, 1, 0, 0, 1);
            if ( doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(1, 0, 1, 1, 1, 1, 0, 0, 1);
            if ( doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(1, 1, 0, 1, 1, 1, 0, 0, 1);
            if ( doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(1, 1, 1, 1, 1, 1, 0, 0, 1);
          } else {
            if (!doEnergy && !doVirial && !doSlow && !doPairlist) CALL(0, 0, 0, 0, 1, 1, 0, 0, 0);
            if (!doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(0, 0, 1, 0, 1, 1, 0, 0, 0);
            if (!doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(0, 1, 0, 0, 1, 1, 0, 0, 0);
            if (!doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(0, 1, 1, 0, 1, 1, 0, 0, 0);
            if ( doEnergy && !doVirial && !doSlow && !doPairlist) CALL(1, 0, 0, 0, 1, 1, 0, 0, 0);
            if ( doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(1, 0, 1, 0, 1, 1, 0, 0, 0);
            if ( doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(1, 1, 0, 0, 1, 1, 0, 0, 0);
            if ( doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(1, 1, 1, 0, 1, 1, 0, 0, 0);

            if (!doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(0, 0, 0, 1, 1, 1, 0, 0, 0);
            if (!doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(0, 0, 1, 1, 1, 1, 0, 0, 0);
            if (!doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(0, 1, 0, 1, 1, 1, 0, 0, 0);
            if (!doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(0, 1, 1, 1, 1, 1, 0, 0, 0);
            if ( doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(1, 0, 0, 1, 1, 1, 0, 0, 0);
            if ( doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(1, 0, 1, 1, 1, 1, 0, 0, 0);
            if ( doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(1, 1, 0, 1, 1, 1, 0, 0, 0);
            if ( doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(1, 1, 1, 1, 1, 1, 0, 0, 0);
          }
        }else{
          if (doAlchVdwForceSwitching) {
            if (!doEnergy && !doVirial && !doSlow && !doPairlist) CALL(0, 0, 0, 0, 1, 0, 1, 0, 1);
            if (!doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(0, 0, 1, 0, 1, 0, 1, 0, 1);
            if (!doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(0, 1, 0, 0, 1, 0, 1, 0, 1);
            if (!doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(0, 1, 1, 0, 1, 0, 1, 0, 1);
            if ( doEnergy && !doVirial && !doSlow && !doPairlist) CALL(1, 0, 0, 0, 1, 0, 1, 0, 1);
            if ( doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(1, 0, 1, 0, 1, 0, 1, 0, 1);
            if ( doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(1, 1, 0, 0, 1, 0, 1, 0, 1);
            if ( doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(1, 1, 1, 0, 1, 0, 1, 0, 1);

            if (!doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(0, 0, 0, 1, 1, 0, 1, 0, 1);
            if (!doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(0, 0, 1, 1, 1, 0, 1, 0, 1);
            if (!doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(0, 1, 0, 1, 1, 0, 1, 0, 1);
            if (!doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(0, 1, 1, 1, 1, 0, 1, 0, 1);
            if ( doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(1, 0, 0, 1, 1, 0, 1, 0, 1);
            if ( doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(1, 0, 1, 1, 1, 0, 1, 0, 1);
            if ( doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(1, 1, 0, 1, 1, 0, 1, 0, 1);
            if ( doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(1, 1, 1, 1, 1, 0, 1, 0, 1);
          } else {
            if (!doEnergy && !doVirial && !doSlow && !doPairlist) CALL(0, 0, 0, 0, 1, 0, 1, 0, 0);
            if (!doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(0, 0, 1, 0, 1, 0, 1, 0, 0);
            if (!doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(0, 1, 0, 0, 1, 0, 1, 0, 0);
            if (!doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(0, 1, 1, 0, 1, 0, 1, 0, 0);
            if ( doEnergy && !doVirial && !doSlow && !doPairlist) CALL(1, 0, 0, 0, 1, 0, 1, 0, 0);
            if ( doEnergy && !doVirial &&  doSlow && !doPairlist) CALL(1, 0, 1, 0, 1, 0, 1, 0, 0);
            if ( doEnergy &&  doVirial && !doSlow && !doPairlist) CALL(1, 1, 0, 0, 1, 0, 1, 0, 0);
            if ( doEnergy &&  doVirial &&  doSlow && !doPairlist) CALL(1, 1, 1, 0, 1, 0, 1, 0, 0);

            if (!doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(0, 0, 0, 1, 1, 0, 1, 0, 0);
            if (!doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(0, 0, 1, 1, 1, 0, 1, 0, 0);
            if (!doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(0, 1, 0, 1, 1, 0, 1, 0, 0);
            if (!doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(0, 1, 1, 1, 1, 0, 1, 0, 0);
            if ( doEnergy && !doVirial && !doSlow &&  doPairlist) CALL(1, 0, 0, 1, 1, 0, 1, 0, 0);
            if ( doEnergy && !doVirial &&  doSlow &&  doPairlist) CALL(1, 0, 1, 1, 1, 0, 1, 0, 0);
            if ( doEnergy &&  doVirial && !doSlow &&  doPairlist) CALL(1, 1, 0, 1, 1, 0, 1, 0, 0);
            if ( doEnergy &&  doVirial &&  doSlow &&  doPairlist) CALL(1, 1, 1, 1, 1, 0, 1, 0, 0);
          }
        }//if doFEP
      }//if doAlch
    }//if doStreaming

    if (!called) {
      NAMD_die("CudaComputeNonbondedKernel::nonbondedForce, none of the kernels called");
    }

#else

#ifdef USE_TABLE_ARRAYS
  #define VDW_TABLE_PARAMS cudaNonbondedTables.getVdwCoefTable()
  #define TABLE_PARAMS \
    cudaNonbondedTables.getForceTable(), cudaNonbondedTables.getEnergyTable()
#else
  #define VDW_TABLE_PARAMS cudaNonbondedTables.getVdwCoefTableTex()
  #define TABLE_PARAMS \
    cudaNonbondedTables.getForceTableTex(), cudaNonbondedTables.getEnergyTableTex()
#endif

#define CALL(DOENERGY, DOVIRIAL, DOSLOW, DOPAIRLIST, DOALCH, DOFEP, DOTI, DOSTREAMING, DOTABLE, DOALCHWDWFORCESWITCHING) \
    nonbondedForceKernel<DOENERGY, DOVIRIAL, DOSLOW, DOPAIRLIST, DOALCH, DOFEP, DOTI, DOSTREAMING, DOTABLE, DOALCHWDWFORCESWITCHING> \
  <<< nblock, nthread, shMemSize, stream >>>  \
  (start, tlKernel.getNumTileLists(), tlKernel.getTileLists(), tlKernel.getTileExcls(), tlKernel.getTileJatomStart(), \
    cudaNonbondedTables.getVdwCoefTableWidth(), \
    VDW_TABLE_PARAMS, \
    vdwTypes, lata, latb, latc, tlKernel.get_xyzq(), cutoff2, nbConstants, \
    TABLE_PARAMS,\
    tlKernel.get_plcutoff2(), tlKernel.getPatchPairs(), atomIndex, exclIndexMaxDiff, overflowExclusions, \
    tlKernel.getTileListDepth(), tlKernel.getTileListOrder(), tlKernel.getJtiles(), tlKernel.getTileListStatDevPtr(), \
    tlKernel.getBoundingBoxes(), \
    force_x, force_y, force_z, force_w, \
    forceSlow_x, forceSlow_y, forceSlow_z, forceSlow_w, \
    numPatches, patchNumCountPtr, tlKernel.getCudaPatches(), m_forces, m_forcesSlow, m_patchReadyQueue, \
    outputOrderPtr, tlKernel.getTileListVirialEnergy(), tlKernel.get_part())

#ifdef DEBUG
    char errmsg[256];
#endif

    switch (options) {
      case  0: CALL(0, 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
      case  1: CALL(1, 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
      case  2: CALL(0, 1, 0, 0, 0, 0, 0, 0, 0, 0); break;
      case  3: CALL(1, 1, 0, 0, 0, 0, 0, 0, 0, 0); break;
      case  4: CALL(0, 0, 1, 0, 0, 0, 0, 0, 0, 0); break;
      case  5: CALL(1, 0, 1, 0, 0, 0, 0, 0, 0, 0); break;
      case  6: CALL(0, 1, 1, 0, 0, 0, 0, 0, 0, 0); break;
      case  7: CALL(1, 1, 1, 0, 0, 0, 0, 0, 0, 0); break;
      case  8: CALL(0, 0, 0, 1, 0, 0, 0, 0, 0, 0); break;
      case  9: CALL(1, 0, 0, 1, 0, 0, 0, 0, 0, 0); break;
      case  10: CALL(0, 1, 0, 1, 0, 0, 0, 0, 0, 0); break;
      case  11: CALL(1, 1, 0, 1, 0, 0, 0, 0, 0, 0); break;
      case  12: CALL(0, 0, 1, 1, 0, 0, 0, 0, 0, 0); break;
      case  13: CALL(1, 0, 1, 1, 0, 0, 0, 0, 0, 0); break;
      case  14: CALL(0, 1, 1, 1, 0, 0, 0, 0, 0, 0); break;
      case  15: CALL(1, 1, 1, 1, 0, 0, 0, 0, 0, 0); break;

#if 0
      case  16: CALL(0, 0, 0, 0, 1, 0, 0, 0, 0, 0); break;
      case  17: CALL(1, 0, 0, 0, 1, 0, 0, 0, 0, 0); break;
      case  18: CALL(0, 1, 0, 0, 1, 0, 0, 0, 0, 0); break;
      case  19: CALL(1, 1, 0, 0, 1, 0, 0, 0, 0, 0); break;
      case  20: CALL(0, 0, 1, 0, 1, 0, 0, 0, 0, 0); break;
      case  21: CALL(1, 0, 1, 0, 1, 0, 0, 0, 0, 0); break;
      case  22: CALL(0, 1, 1, 0, 1, 0, 0, 0, 0, 0); break;
      case  23: CALL(1, 1, 1, 0, 1, 0, 0, 0, 0, 0); break;
      case  24: CALL(0, 0, 0, 1, 1, 0, 0, 0, 0, 0); break;
      case  25: CALL(1, 0, 0, 1, 1, 0, 0, 0, 0, 0); break;
      case  26: CALL(0, 1, 0, 1, 1, 0, 0, 0, 0, 0); break;
      case  27: CALL(1, 1, 0, 1, 1, 0, 0, 0, 0, 0); break;
      case  28: CALL(0, 0, 1, 1, 1, 0, 0, 0, 0, 0); break;
      case  29: CALL(1, 0, 1, 1, 1, 0, 0, 0, 0, 0); break;
      case  30: CALL(0, 1, 1, 1, 1, 0, 0, 0, 0, 0); break;
      case  31: CALL(1, 1, 1, 1, 1, 0, 0, 0, 0, 0); break;

      case  32: CALL(0, 0, 0, 0, 0, 1, 0, 0, 0, 0); break;
      case  33: CALL(1, 0, 0, 0, 0, 1, 0, 0, 0, 0); break;
      case  34: CALL(0, 1, 0, 0, 0, 1, 0, 0, 0, 0); break;
      case  35: CALL(1, 1, 0, 0, 0, 1, 0, 0, 0, 0); break;
      case  36: CALL(0, 0, 1, 0, 0, 1, 0, 0, 0, 0); break;
      case  37: CALL(1, 0, 1, 0, 0, 1, 0, 0, 0, 0); break;
      case  38: CALL(0, 1, 1, 0, 0, 1, 0, 0, 0, 0); break;
      case  39: CALL(1, 1, 1, 0, 0, 1, 0, 0, 0, 0); break;
      case  40: CALL(0, 0, 0, 1, 0, 1, 0, 0, 0, 0); break;
      case  41: CALL(1, 0, 0, 1, 0, 1, 0, 0, 0, 0); break;
      case  42: CALL(0, 1, 0, 1, 0, 1, 0, 0, 0, 0); break;
      case  43: CALL(1, 1, 0, 1, 0, 1, 0, 0, 0, 0); break;
      case  44: CALL(0, 0, 1, 1, 0, 1, 0, 0, 0, 0); break;
      case  45: CALL(1, 0, 1, 1, 0, 1, 0, 0, 0, 0); break;
      case  46: CALL(0, 1, 1, 1, 0, 1, 0, 0, 0, 0); break;
      case  47: CALL(1, 1, 1, 1, 0, 1, 0, 0, 0, 0); break;

      case  48: CALL(0, 0, 0, 0, 1, 1, 0, 0, 0, 0); break;
      case  49: CALL(1, 0, 0, 0, 1, 1, 0, 0, 0, 0); break;
      case  50: CALL(0, 1, 0, 0, 1, 1, 0, 0, 0, 0); break;
      case  51: CALL(1, 1, 0, 0, 1, 1, 0, 0, 0, 0); break;
      case  52: CALL(0, 0, 1, 0, 1, 1, 0, 0, 0, 0); break;
      case  53: CALL(1, 0, 1, 0, 1, 1, 0, 0, 0, 0); break;
      case  54: CALL(0, 1, 1, 0, 1, 1, 0, 0, 0, 0); break;
      case  55: CALL(1, 1, 1, 0, 1, 1, 0, 0, 0, 0); break;
      case  56: CALL(0, 0, 0, 1, 1, 1, 0, 0, 0, 0); break;
      case  57: CALL(1, 0, 0, 1, 1, 1, 0, 0, 0, 0); break;
      case  58: CALL(0, 1, 0, 1, 1, 1, 0, 0, 0, 0); break;
      case  59: CALL(1, 1, 0, 1, 1, 1, 0, 0, 0, 0); break;
      case  60: CALL(0, 0, 1, 1, 1, 1, 0, 0, 0, 0); break;
      case  61: CALL(1, 0, 1, 1, 1, 1, 0, 0, 0, 0); break;
      case  62: CALL(0, 1, 1, 1, 1, 1, 0, 0, 0, 0); break;
      case  63: CALL(1, 1, 1, 1, 1, 1, 0, 0, 0, 0); break;

      case  64: CALL(0, 0, 0, 0, 0, 0, 1, 0, 0, 0); break;
      case  65: CALL(1, 0, 0, 0, 0, 0, 1, 0, 0, 0); break;
      case  66: CALL(0, 1, 0, 0, 0, 0, 1, 0, 0, 0); break;
      case  67: CALL(1, 1, 0, 0, 0, 0, 1, 0, 0, 0); break;
      case  68: CALL(0, 0, 1, 0, 0, 0, 1, 0, 0, 0); break;
      case  69: CALL(1, 0, 1, 0, 0, 0, 1, 0, 0, 0); break;
      case  70: CALL(0, 1, 1, 0, 0, 0, 1, 0, 0, 0); break;
      case  71: CALL(1, 1, 1, 0, 0, 0, 1, 0, 0, 0); break;
      case  72: CALL(0, 0, 0, 1, 0, 0, 1, 0, 0, 0); break;
      case  73: CALL(1, 0, 0, 1, 0, 0, 1, 0, 0, 0); break;
      case  74: CALL(0, 1, 0, 1, 0, 0, 1, 0, 0, 0); break;
      case  75: CALL(1, 1, 0, 1, 0, 0, 1, 0, 0, 0); break;
      case  76: CALL(0, 0, 1, 1, 0, 0, 1, 0, 0, 0); break;
      case  77: CALL(1, 0, 1, 1, 0, 0, 1, 0, 0, 0); break;
      case  78: CALL(0, 1, 1, 1, 0, 0, 1, 0, 0, 0); break;
      case  79: CALL(1, 1, 1, 1, 0, 0, 1, 0, 0, 0); break;

      case  80: CALL(0, 0, 0, 0, 1, 0, 1, 0, 0, 0); break;
      case  81: CALL(1, 0, 0, 0, 1, 0, 1, 0, 0, 0); break;
      case  82: CALL(0, 1, 0, 0, 1, 0, 1, 0, 0, 0); break;
      case  83: CALL(1, 1, 0, 0, 1, 0, 1, 0, 0, 0); break;
      case  84: CALL(0, 0, 1, 0, 1, 0, 1, 0, 0, 0); break;
      case  85: CALL(1, 0, 1, 0, 1, 0, 1, 0, 0, 0); break;
      case  86: CALL(0, 1, 1, 0, 1, 0, 1, 0, 0, 0); break;
      case  87: CALL(1, 1, 1, 0, 1, 0, 1, 0, 0, 0); break;
      case  88: CALL(0, 0, 0, 1, 1, 0, 1, 0, 0, 0); break;
      case  89: CALL(1, 0, 0, 1, 1, 0, 1, 0, 0, 0); break;
      case  90: CALL(0, 1, 0, 1, 1, 0, 1, 0, 0, 0); break;
      case  91: CALL(1, 1, 0, 1, 1, 0, 1, 0, 0, 0); break;
      case  92: CALL(0, 0, 1, 1, 1, 0, 1, 0, 0, 0); break;
      case  93: CALL(1, 0, 1, 1, 1, 0, 1, 0, 0, 0); break;
      case  94: CALL(0, 1, 1, 1, 1, 0, 1, 0, 0, 0); break;
      case  95: CALL(1, 1, 1, 1, 1, 0, 1, 0, 0, 0); break;

      case  96: CALL(0, 0, 0, 0, 0, 1, 1, 0, 0, 0); break;
      case  97: CALL(1, 0, 0, 0, 0, 1, 1, 0, 0, 0); break;
      case  98: CALL(0, 1, 0, 0, 0, 1, 1, 0, 0, 0); break;
      case  99: CALL(1, 1, 0, 0, 0, 1, 1, 0, 0, 0); break;
      case  100: CALL(0, 0, 1, 0, 0, 1, 1, 0, 0, 0); break;
      case  101: CALL(1, 0, 1, 0, 0, 1, 1, 0, 0, 0); break;
      case  102: CALL(0, 1, 1, 0, 0, 1, 1, 0, 0, 0); break;
      case  103: CALL(1, 1, 1, 0, 0, 1, 1, 0, 0, 0); break;
      case  104: CALL(0, 0, 0, 1, 0, 1, 1, 0, 0, 0); break;
      case  105: CALL(1, 0, 0, 1, 0, 1, 1, 0, 0, 0); break;
      case  106: CALL(0, 1, 0, 1, 0, 1, 1, 0, 0, 0); break;
      case  107: CALL(1, 1, 0, 1, 0, 1, 1, 0, 0, 0); break;
      case  108: CALL(0, 0, 1, 1, 0, 1, 1, 0, 0, 0); break;
      case  109: CALL(1, 0, 1, 1, 0, 1, 1, 0, 0, 0); break;
      case  110: CALL(0, 1, 1, 1, 0, 1, 1, 0, 0, 0); break;
      case  111: CALL(1, 1, 1, 1, 0, 1, 1, 0, 0, 0); break;

      case  112: CALL(0, 0, 0, 0, 1, 1, 1, 0, 0, 0); break;
      case  113: CALL(1, 0, 0, 0, 1, 1, 1, 0, 0, 0); break;
      case  114: CALL(0, 1, 0, 0, 1, 1, 1, 0, 0, 0); break;
      case  115: CALL(1, 1, 0, 0, 1, 1, 1, 0, 0, 0); break;
      case  116: CALL(0, 0, 1, 0, 1, 1, 1, 0, 0, 0); break;
      case  117: CALL(1, 0, 1, 0, 1, 1, 1, 0, 0, 0); break;
      case  118: CALL(0, 1, 1, 0, 1, 1, 1, 0, 0, 0); break;
      case  119: CALL(1, 1, 1, 0, 1, 1, 1, 0, 0, 0); break;
      case  120: CALL(0, 0, 0, 1, 1, 1, 1, 0, 0, 0); break;
      case  121: CALL(1, 0, 0, 1, 1, 1, 1, 0, 0, 0); break;
      case  122: CALL(0, 1, 0, 1, 1, 1, 1, 0, 0, 0); break;
      case  123: CALL(1, 1, 0, 1, 1, 1, 1, 0, 0, 0); break;
      case  124: CALL(0, 0, 1, 1, 1, 1, 1, 0, 0, 0); break;
      case  125: CALL(1, 0, 1, 1, 1, 1, 1, 0, 0, 0); break;
      case  126: CALL(0, 1, 1, 1, 1, 1, 1, 0, 0, 0); break;
      case  127: CALL(1, 1, 1, 1, 1, 1, 1, 0, 0, 0); break;

#endif
      case  128: CALL(0, 0, 0, 0, 0, 0, 0, 1, 0, 0); break;
      case  129: CALL(1, 0, 0, 0, 0, 0, 0, 1, 0, 0); break;
      case  130: CALL(0, 1, 0, 0, 0, 0, 0, 1, 0, 0); break;
      case  131: CALL(1, 1, 0, 0, 0, 0, 0, 1, 0, 0); break;
      case  132: CALL(0, 0, 1, 0, 0, 0, 0, 1, 0, 0); break;
      case  133: CALL(1, 0, 1, 0, 0, 0, 0, 1, 0, 0); break;
      case  134: CALL(0, 1, 1, 0, 0, 0, 0, 1, 0, 0); break;
      case  135: CALL(1, 1, 1, 0, 0, 0, 0, 1, 0, 0); break;
      case  136: CALL(0, 0, 0, 1, 0, 0, 0, 1, 0, 0); break;
      case  137: CALL(1, 0, 0, 1, 0, 0, 0, 1, 0, 0); break;
      case  138: CALL(0, 1, 0, 1, 0, 0, 0, 1, 0, 0); break;
      case  139: CALL(1, 1, 0, 1, 0, 0, 0, 1, 0, 0); break;
      case  140: CALL(0, 0, 1, 1, 0, 0, 0, 1, 0, 0); break;
      case  141: CALL(1, 0, 1, 1, 0, 0, 0, 1, 0, 0); break;
      case  142: CALL(0, 1, 1, 1, 0, 0, 0, 1, 0, 0); break;
      case  143: CALL(1, 1, 1, 1, 0, 0, 0, 1, 0, 0); break;

#if 0
      case  144: CALL(0, 0, 0, 0, 1, 0, 0, 1, 0, 0); break;
      case  145: CALL(1, 0, 0, 0, 1, 0, 0, 1, 0, 0); break;
      case  146: CALL(0, 1, 0, 0, 1, 0, 0, 1, 0, 0); break;
      case  147: CALL(1, 1, 0, 0, 1, 0, 0, 1, 0, 0); break;
      case  148: CALL(0, 0, 1, 0, 1, 0, 0, 1, 0, 0); break;
      case  149: CALL(1, 0, 1, 0, 1, 0, 0, 1, 0, 0); break;
      case  150: CALL(0, 1, 1, 0, 1, 0, 0, 1, 0, 0); break;
      case  151: CALL(1, 1, 1, 0, 1, 0, 0, 1, 0, 0); break;
      case  152: CALL(0, 0, 0, 1, 1, 0, 0, 1, 0, 0); break;
      case  153: CALL(1, 0, 0, 1, 1, 0, 0, 1, 0, 0); break;
      case  154: CALL(0, 1, 0, 1, 1, 0, 0, 1, 0, 0); break;
      case  155: CALL(1, 1, 0, 1, 1, 0, 0, 1, 0, 0); break;
      case  156: CALL(0, 0, 1, 1, 1, 0, 0, 1, 0, 0); break;
      case  157: CALL(1, 0, 1, 1, 1, 0, 0, 1, 0, 0); break;
      case  158: CALL(0, 1, 1, 1, 1, 0, 0, 1, 0, 0); break;
      case  159: CALL(1, 1, 1, 1, 1, 0, 0, 1, 0, 0); break;

      case  160: CALL(0, 0, 0, 0, 0, 1, 0, 1, 0, 0); break;
      case  161: CALL(1, 0, 0, 0, 0, 1, 0, 1, 0, 0); break;
      case  162: CALL(0, 1, 0, 0, 0, 1, 0, 1, 0, 0); break;
      case  163: CALL(1, 1, 0, 0, 0, 1, 0, 1, 0, 0); break;
      case  164: CALL(0, 0, 1, 0, 0, 1, 0, 1, 0, 0); break;
      case  165: CALL(1, 0, 1, 0, 0, 1, 0, 1, 0, 0); break;
      case  166: CALL(0, 1, 1, 0, 0, 1, 0, 1, 0, 0); break;
      case  167: CALL(1, 1, 1, 0, 0, 1, 0, 1, 0, 0); break;
      case  168: CALL(0, 0, 0, 1, 0, 1, 0, 1, 0, 0); break;
      case  169: CALL(1, 0, 0, 1, 0, 1, 0, 1, 0, 0); break;
      case  170: CALL(0, 1, 0, 1, 0, 1, 0, 1, 0, 0); break;
      case  171: CALL(1, 1, 0, 1, 0, 1, 0, 1, 0, 0); break;
      case  172: CALL(0, 0, 1, 1, 0, 1, 0, 1, 0, 0); break;
      case  173: CALL(1, 0, 1, 1, 0, 1, 0, 1, 0, 0); break;
      case  174: CALL(0, 1, 1, 1, 0, 1, 0, 1, 0, 0); break;
      case  175: CALL(1, 1, 1, 1, 0, 1, 0, 1, 0, 0); break;

      case  176: CALL(0, 0, 0, 0, 1, 1, 0, 1, 0, 0); break;
      case  177: CALL(1, 0, 0, 0, 1, 1, 0, 1, 0, 0); break;
      case  178: CALL(0, 1, 0, 0, 1, 1, 0, 1, 0, 0); break;
      case  179: CALL(1, 1, 0, 0, 1, 1, 0, 1, 0, 0); break;
      case  180: CALL(0, 0, 1, 0, 1, 1, 0, 1, 0, 0); break;
      case  181: CALL(1, 0, 1, 0, 1, 1, 0, 1, 0, 0); break;
      case  182: CALL(0, 1, 1, 0, 1, 1, 0, 1, 0, 0); break;
      case  183: CALL(1, 1, 1, 0, 1, 1, 0, 1, 0, 0); break;
      case  184: CALL(0, 0, 0, 1, 1, 1, 0, 1, 0, 0); break;
      case  185: CALL(1, 0, 0, 1, 1, 1, 0, 1, 0, 0); break;
      case  186: CALL(0, 1, 0, 1, 1, 1, 0, 1, 0, 0); break;
      case  187: CALL(1, 1, 0, 1, 1, 1, 0, 1, 0, 0); break;
      case  188: CALL(0, 0, 1, 1, 1, 1, 0, 1, 0, 0); break;
      case  189: CALL(1, 0, 1, 1, 1, 1, 0, 1, 0, 0); break;
      case  190: CALL(0, 1, 1, 1, 1, 1, 0, 1, 0, 0); break;
      case  191: CALL(1, 1, 1, 1, 1, 1, 0, 1, 0, 0); break;

      case  192: CALL(0, 0, 0, 0, 0, 0, 1, 1, 0, 0); break;
      case  193: CALL(1, 0, 0, 0, 0, 0, 1, 1, 0, 0); break;
      case  194: CALL(0, 1, 0, 0, 0, 0, 1, 1, 0, 0); break;
      case  195: CALL(1, 1, 0, 0, 0, 0, 1, 1, 0, 0); break;
      case  196: CALL(0, 0, 1, 0, 0, 0, 1, 1, 0, 0); break;
      case  197: CALL(1, 0, 1, 0, 0, 0, 1, 1, 0, 0); break;
      case  198: CALL(0, 1, 1, 0, 0, 0, 1, 1, 0, 0); break;
      case  199: CALL(1, 1, 1, 0, 0, 0, 1, 1, 0, 0); break;
      case  200: CALL(0, 0, 0, 1, 0, 0, 1, 1, 0, 0); break;
      case  201: CALL(1, 0, 0, 1, 0, 0, 1, 1, 0, 0); break;
      case  202: CALL(0, 1, 0, 1, 0, 0, 1, 1, 0, 0); break;
      case  203: CALL(1, 1, 0, 1, 0, 0, 1, 1, 0, 0); break;
      case  204: CALL(0, 0, 1, 1, 0, 0, 1, 1, 0, 0); break;
      case  205: CALL(1, 0, 1, 1, 0, 0, 1, 1, 0, 0); break;
      case  206: CALL(0, 1, 1, 1, 0, 0, 1, 1, 0, 0); break;
      case  207: CALL(1, 1, 1, 1, 0, 0, 1, 1, 0, 0); break;

      case  208: CALL(0, 0, 0, 0, 1, 0, 1, 1, 0, 0); break;
      case  209: CALL(1, 0, 0, 0, 1, 0, 1, 1, 0, 0); break;
      case  210: CALL(0, 1, 0, 0, 1, 0, 1, 1, 0, 0); break;
      case  211: CALL(1, 1, 0, 0, 1, 0, 1, 1, 0, 0); break;
      case  212: CALL(0, 0, 1, 0, 1, 0, 1, 1, 0, 0); break;
      case  213: CALL(1, 0, 1, 0, 1, 0, 1, 1, 0, 0); break;
      case  214: CALL(0, 1, 1, 0, 1, 0, 1, 1, 0, 0); break;
      case  215: CALL(1, 1, 1, 0, 1, 0, 1, 1, 0, 0); break;
      case  216: CALL(0, 0, 0, 1, 1, 0, 1, 1, 0, 0); break;
      case  217: CALL(1, 0, 0, 1, 1, 0, 1, 1, 0, 0); break;
      case  218: CALL(0, 1, 0, 1, 1, 0, 1, 1, 0, 0); break;
      case  219: CALL(1, 1, 0, 1, 1, 0, 1, 1, 0, 0); break;
      case  220: CALL(0, 0, 1, 1, 1, 0, 1, 1, 0, 0); break;
      case  221: CALL(1, 0, 1, 1, 1, 0, 1, 1, 0, 0); break;
      case  222: CALL(0, 1, 1, 1, 1, 0, 1, 1, 0, 0); break;
      case  223: CALL(1, 1, 1, 1, 1, 0, 1, 1, 0, 0); break;

      case  224: CALL(0, 0, 0, 0, 0, 1, 1, 1, 0, 0); break;
      case  225: CALL(1, 0, 0, 0, 0, 1, 1, 1, 0, 0); break;
      case  226: CALL(0, 1, 0, 0, 0, 1, 1, 1, 0, 0); break;
      case  227: CALL(1, 1, 0, 0, 0, 1, 1, 1, 0, 0); break;
      case  228: CALL(0, 0, 1, 0, 0, 1, 1, 1, 0, 0); break;
      case  229: CALL(1, 0, 1, 0, 0, 1, 1, 1, 0, 0); break;
      case  230: CALL(0, 1, 1, 0, 0, 1, 1, 1, 0, 0); break;
      case  231: CALL(1, 1, 1, 0, 0, 1, 1, 1, 0, 0); break;
      case  232: CALL(0, 0, 0, 1, 0, 1, 1, 1, 0, 0); break;
      case  233: CALL(1, 0, 0, 1, 0, 1, 1, 1, 0, 0); break;
      case  234: CALL(0, 1, 0, 1, 0, 1, 1, 1, 0, 0); break;
      case  235: CALL(1, 1, 0, 1, 0, 1, 1, 1, 0, 0); break;
      case  236: CALL(0, 0, 1, 1, 0, 1, 1, 1, 0, 0); break;
      case  237: CALL(1, 0, 1, 1, 0, 1, 1, 1, 0, 0); break;
      case  238: CALL(0, 1, 1, 1, 0, 1, 1, 1, 0, 0); break;
      case  239: CALL(1, 1, 1, 1, 0, 1, 1, 1, 0, 0); break;

      case  240: CALL(0, 0, 0, 0, 1, 1, 1, 1, 0, 0); break;
      case  241: CALL(1, 0, 0, 0, 1, 1, 1, 1, 0, 0); break;
      case  242: CALL(0, 1, 0, 0, 1, 1, 1, 1, 0, 0); break;
      case  243: CALL(1, 1, 0, 0, 1, 1, 1, 1, 0, 0); break;
      case  244: CALL(0, 0, 1, 0, 1, 1, 1, 1, 0, 0); break;
      case  245: CALL(1, 0, 1, 0, 1, 1, 1, 1, 0, 0); break;
      case  246: CALL(0, 1, 1, 0, 1, 1, 1, 1, 0, 0); break;
      case  247: CALL(1, 1, 1, 0, 1, 1, 1, 1, 0, 0); break;
      case  248: CALL(0, 0, 0, 1, 1, 1, 1, 1, 0, 0); break;
      case  249: CALL(1, 0, 0, 1, 1, 1, 1, 1, 0, 0); break;
      case  250: CALL(0, 1, 0, 1, 1, 1, 1, 1, 0, 0); break;
      case  251: CALL(1, 1, 0, 1, 1, 1, 1, 1, 0, 0); break;
      case  252: CALL(0, 0, 1, 1, 1, 1, 1, 1, 0, 0); break;
      case  253: CALL(1, 0, 1, 1, 1, 1, 1, 1, 0, 0); break;
      case  254: CALL(0, 1, 1, 1, 1, 1, 1, 1, 0, 0); break;
      case  255: CALL(1, 1, 1, 1, 1, 1, 1, 1, 0, 0); break;

#endif
      case  256: CALL(0, 0, 0, 0, 0, 0, 0, 0, 1, 0); break;
      case  257: CALL(1, 0, 0, 0, 0, 0, 0, 0, 1, 0); break;
      case  258: CALL(0, 1, 0, 0, 0, 0, 0, 0, 1, 0); break;
      case  259: CALL(1, 1, 0, 0, 0, 0, 0, 0, 1, 0); break;
      case  260: CALL(0, 0, 1, 0, 0, 0, 0, 0, 1, 0); break;
      case  261: CALL(1, 0, 1, 0, 0, 0, 0, 0, 1, 0); break;
      case  262: CALL(0, 1, 1, 0, 0, 0, 0, 0, 1, 0); break;
      case  263: CALL(1, 1, 1, 0, 0, 0, 0, 0, 1, 0); break;
      case  264: CALL(0, 0, 0, 1, 0, 0, 0, 0, 1, 0); break;
      case  265: CALL(1, 0, 0, 1, 0, 0, 0, 0, 1, 0); break;
      case  266: CALL(0, 1, 0, 1, 0, 0, 0, 0, 1, 0); break;
      case  267: CALL(1, 1, 0, 1, 0, 0, 0, 0, 1, 0); break;
      case  268: CALL(0, 0, 1, 1, 0, 0, 0, 0, 1, 0); break;
      case  269: CALL(1, 0, 1, 1, 0, 0, 0, 0, 1, 0); break;
      case  270: CALL(0, 1, 1, 1, 0, 0, 0, 0, 1, 0); break;
      case  271: CALL(1, 1, 1, 1, 0, 0, 0, 0, 1, 0); break;

#if 0
      case  272: CALL(0, 0, 0, 0, 1, 0, 0, 0, 1, 0); break;
      case  273: CALL(1, 0, 0, 0, 1, 0, 0, 0, 1, 0); break;
      case  274: CALL(0, 1, 0, 0, 1, 0, 0, 0, 1, 0); break;
      case  275: CALL(1, 1, 0, 0, 1, 0, 0, 0, 1, 0); break;
      case  276: CALL(0, 0, 1, 0, 1, 0, 0, 0, 1, 0); break;
      case  277: CALL(1, 0, 1, 0, 1, 0, 0, 0, 1, 0); break;
      case  278: CALL(0, 1, 1, 0, 1, 0, 0, 0, 1, 0); break;
      case  279: CALL(1, 1, 1, 0, 1, 0, 0, 0, 1, 0); break;
      case  280: CALL(0, 0, 0, 1, 1, 0, 0, 0, 1, 0); break;
      case  281: CALL(1, 0, 0, 1, 1, 0, 0, 0, 1, 0); break;
      case  282: CALL(0, 1, 0, 1, 1, 0, 0, 0, 1, 0); break;
      case  283: CALL(1, 1, 0, 1, 1, 0, 0, 0, 1, 0); break;
      case  284: CALL(0, 0, 1, 1, 1, 0, 0, 0, 1, 0); break;
      case  285: CALL(1, 0, 1, 1, 1, 0, 0, 0, 1, 0); break;
      case  286: CALL(0, 1, 1, 1, 1, 0, 0, 0, 1, 0); break;
      case  287: CALL(1, 1, 1, 1, 1, 0, 0, 0, 1, 0); break;

      case  288: CALL(0, 0, 0, 0, 0, 1, 0, 0, 1, 0); break;
      case  289: CALL(1, 0, 0, 0, 0, 1, 0, 0, 1, 0); break;
      case  290: CALL(0, 1, 0, 0, 0, 1, 0, 0, 1, 0); break;
      case  291: CALL(1, 1, 0, 0, 0, 1, 0, 0, 1, 0); break;
      case  292: CALL(0, 0, 1, 0, 0, 1, 0, 0, 1, 0); break;
      case  293: CALL(1, 0, 1, 0, 0, 1, 0, 0, 1, 0); break;
      case  294: CALL(0, 1, 1, 0, 0, 1, 0, 0, 1, 0); break;
      case  295: CALL(1, 1, 1, 0, 0, 1, 0, 0, 1, 0); break;
      case  296: CALL(0, 0, 0, 1, 0, 1, 0, 0, 1, 0); break;
      case  297: CALL(1, 0, 0, 1, 0, 1, 0, 0, 1, 0); break;
      case  298: CALL(0, 1, 0, 1, 0, 1, 0, 0, 1, 0); break;
      case  299: CALL(1, 1, 0, 1, 0, 1, 0, 0, 1, 0); break;
      case  300: CALL(0, 0, 1, 1, 0, 1, 0, 0, 1, 0); break;
      case  301: CALL(1, 0, 1, 1, 0, 1, 0, 0, 1, 0); break;
      case  302: CALL(0, 1, 1, 1, 0, 1, 0, 0, 1, 0); break;
      case  303: CALL(1, 1, 1, 1, 0, 1, 0, 0, 1, 0); break;

#endif
      case  304: CALL(0, 0, 0, 0, 1, 1, 0, 0, 1, 0); break;
      case  305: CALL(1, 0, 0, 0, 1, 1, 0, 0, 1, 0); break;
      case  306: CALL(0, 1, 0, 0, 1, 1, 0, 0, 1, 0); break;
      case  307: CALL(1, 1, 0, 0, 1, 1, 0, 0, 1, 0); break;
      case  308: CALL(0, 0, 1, 0, 1, 1, 0, 0, 1, 0); break;
      case  309: CALL(1, 0, 1, 0, 1, 1, 0, 0, 1, 0); break;
      case  310: CALL(0, 1, 1, 0, 1, 1, 0, 0, 1, 0); break;
      case  311: CALL(1, 1, 1, 0, 1, 1, 0, 0, 1, 0); break;
      case  312: CALL(0, 0, 0, 1, 1, 1, 0, 0, 1, 0); break;
      case  313: CALL(1, 0, 0, 1, 1, 1, 0, 0, 1, 0); break;
      case  314: CALL(0, 1, 0, 1, 1, 1, 0, 0, 1, 0); break;
      case  315: CALL(1, 1, 0, 1, 1, 1, 0, 0, 1, 0); break;
      case  316: CALL(0, 0, 1, 1, 1, 1, 0, 0, 1, 0); break;
      case  317: CALL(1, 0, 1, 1, 1, 1, 0, 0, 1, 0); break;
      case  318: CALL(0, 1, 1, 1, 1, 1, 0, 0, 1, 0); break;
      case  319: CALL(1, 1, 1, 1, 1, 1, 0, 0, 1, 0); break;

#if 0
      case  320: CALL(0, 0, 0, 0, 0, 0, 1, 0, 1, 0); break;
      case  321: CALL(1, 0, 0, 0, 0, 0, 1, 0, 1, 0); break;
      case  322: CALL(0, 1, 0, 0, 0, 0, 1, 0, 1, 0); break;
      case  323: CALL(1, 1, 0, 0, 0, 0, 1, 0, 1, 0); break;
      case  324: CALL(0, 0, 1, 0, 0, 0, 1, 0, 1, 0); break;
      case  325: CALL(1, 0, 1, 0, 0, 0, 1, 0, 1, 0); break;
      case  326: CALL(0, 1, 1, 0, 0, 0, 1, 0, 1, 0); break;
      case  327: CALL(1, 1, 1, 0, 0, 0, 1, 0, 1, 0); break;
      case  328: CALL(0, 0, 0, 1, 0, 0, 1, 0, 1, 0); break;
      case  329: CALL(1, 0, 0, 1, 0, 0, 1, 0, 1, 0); break;
      case  330: CALL(0, 1, 0, 1, 0, 0, 1, 0, 1, 0); break;
      case  331: CALL(1, 1, 0, 1, 0, 0, 1, 0, 1, 0); break;
      case  332: CALL(0, 0, 1, 1, 0, 0, 1, 0, 1, 0); break;
      case  333: CALL(1, 0, 1, 1, 0, 0, 1, 0, 1, 0); break;
      case  334: CALL(0, 1, 1, 1, 0, 0, 1, 0, 1, 0); break;
      case  335: CALL(1, 1, 1, 1, 0, 0, 1, 0, 1, 0); break;

#endif
      case  336: CALL(0, 0, 0, 0, 1, 0, 1, 0, 1, 0); break;
      case  337: CALL(1, 0, 0, 0, 1, 0, 1, 0, 1, 0); break;
      case  338: CALL(0, 1, 0, 0, 1, 0, 1, 0, 1, 0); break;
      case  339: CALL(1, 1, 0, 0, 1, 0, 1, 0, 1, 0); break;
      case  340: CALL(0, 0, 1, 0, 1, 0, 1, 0, 1, 0); break;
      case  341: CALL(1, 0, 1, 0, 1, 0, 1, 0, 1, 0); break;
      case  342: CALL(0, 1, 1, 0, 1, 0, 1, 0, 1, 0); break;
      case  343: CALL(1, 1, 1, 0, 1, 0, 1, 0, 1, 0); break;
      case  344: CALL(0, 0, 0, 1, 1, 0, 1, 0, 1, 0); break;
      case  345: CALL(1, 0, 0, 1, 1, 0, 1, 0, 1, 0); break;
      case  346: CALL(0, 1, 0, 1, 1, 0, 1, 0, 1, 0); break;
      case  347: CALL(1, 1, 0, 1, 1, 0, 1, 0, 1, 0); break;
      case  348: CALL(0, 0, 1, 1, 1, 0, 1, 0, 1, 0); break;
      case  349: CALL(1, 0, 1, 1, 1, 0, 1, 0, 1, 0); break;
      case  350: CALL(0, 1, 1, 1, 1, 0, 1, 0, 1, 0); break;
      case  351: CALL(1, 1, 1, 1, 1, 0, 1, 0, 1, 0); break;

#if 0
      case  352: CALL(0, 0, 0, 0, 0, 1, 1, 0, 1, 0); break;
      case  353: CALL(1, 0, 0, 0, 0, 1, 1, 0, 1, 0); break;
      case  354: CALL(0, 1, 0, 0, 0, 1, 1, 0, 1, 0); break;
      case  355: CALL(1, 1, 0, 0, 0, 1, 1, 0, 1, 0); break;
      case  356: CALL(0, 0, 1, 0, 0, 1, 1, 0, 1, 0); break;
      case  357: CALL(1, 0, 1, 0, 0, 1, 1, 0, 1, 0); break;
      case  358: CALL(0, 1, 1, 0, 0, 1, 1, 0, 1, 0); break;
      case  359: CALL(1, 1, 1, 0, 0, 1, 1, 0, 1, 0); break;
      case  360: CALL(0, 0, 0, 1, 0, 1, 1, 0, 1, 0); break;
      case  361: CALL(1, 0, 0, 1, 0, 1, 1, 0, 1, 0); break;
      case  362: CALL(0, 1, 0, 1, 0, 1, 1, 0, 1, 0); break;
      case  363: CALL(1, 1, 0, 1, 0, 1, 1, 0, 1, 0); break;
      case  364: CALL(0, 0, 1, 1, 0, 1, 1, 0, 1, 0); break;
      case  365: CALL(1, 0, 1, 1, 0, 1, 1, 0, 1, 0); break;
      case  366: CALL(0, 1, 1, 1, 0, 1, 1, 0, 1, 0); break;
      case  367: CALL(1, 1, 1, 1, 0, 1, 1, 0, 1, 0); break;

      case  368: CALL(0, 0, 0, 0, 1, 1, 1, 0, 1, 0); break;
      case  369: CALL(1, 0, 0, 0, 1, 1, 1, 0, 1, 0); break;
      case  370: CALL(0, 1, 0, 0, 1, 1, 1, 0, 1, 0); break;
      case  371: CALL(1, 1, 0, 0, 1, 1, 1, 0, 1, 0); break;
      case  372: CALL(0, 0, 1, 0, 1, 1, 1, 0, 1, 0); break;
      case  373: CALL(1, 0, 1, 0, 1, 1, 1, 0, 1, 0); break;
      case  374: CALL(0, 1, 1, 0, 1, 1, 1, 0, 1, 0); break;
      case  375: CALL(1, 1, 1, 0, 1, 1, 1, 0, 1, 0); break;
      case  376: CALL(0, 0, 0, 1, 1, 1, 1, 0, 1, 0); break;
      case  377: CALL(1, 0, 0, 1, 1, 1, 1, 0, 1, 0); break;
      case  378: CALL(0, 1, 0, 1, 1, 1, 1, 0, 1, 0); break;
      case  379: CALL(1, 1, 0, 1, 1, 1, 1, 0, 1, 0); break;
      case  380: CALL(0, 0, 1, 1, 1, 1, 1, 0, 1, 0); break;
      case  381: CALL(1, 0, 1, 1, 1, 1, 1, 0, 1, 0); break;
      case  382: CALL(0, 1, 1, 1, 1, 1, 1, 0, 1, 0); break;
      case  383: CALL(1, 1, 1, 1, 1, 1, 1, 0, 1, 0); break;

#endif
      case  384: CALL(0, 0, 0, 0, 0, 0, 0, 1, 1, 0); break;
      case  385: CALL(1, 0, 0, 0, 0, 0, 0, 1, 1, 0); break;
      case  386: CALL(0, 1, 0, 0, 0, 0, 0, 1, 1, 0); break;
      case  387: CALL(1, 1, 0, 0, 0, 0, 0, 1, 1, 0); break;
      case  388: CALL(0, 0, 1, 0, 0, 0, 0, 1, 1, 0); break;
      case  389: CALL(1, 0, 1, 0, 0, 0, 0, 1, 1, 0); break;
      case  390: CALL(0, 1, 1, 0, 0, 0, 0, 1, 1, 0); break;
      case  391: CALL(1, 1, 1, 0, 0, 0, 0, 1, 1, 0); break;
      case  392: CALL(0, 0, 0, 1, 0, 0, 0, 1, 1, 0); break;
      case  393: CALL(1, 0, 0, 1, 0, 0, 0, 1, 1, 0); break;
      case  394: CALL(0, 1, 0, 1, 0, 0, 0, 1, 1, 0); break;
      case  395: CALL(1, 1, 0, 1, 0, 0, 0, 1, 1, 0); break;
      case  396: CALL(0, 0, 1, 1, 0, 0, 0, 1, 1, 0); break;
      case  397: CALL(1, 0, 1, 1, 0, 0, 0, 1, 1, 0); break;
      case  398: CALL(0, 1, 1, 1, 0, 0, 0, 1, 1, 0); break;
      case  399: CALL(1, 1, 1, 1, 0, 0, 0, 1, 1, 0); break;

#if 0
      case  400: CALL(0, 0, 0, 0, 1, 0, 0, 1, 1, 0); break;
      case  401: CALL(1, 0, 0, 0, 1, 0, 0, 1, 1, 0); break;
      case  402: CALL(0, 1, 0, 0, 1, 0, 0, 1, 1, 0); break;
      case  403: CALL(1, 1, 0, 0, 1, 0, 0, 1, 1, 0); break;
      case  404: CALL(0, 0, 1, 0, 1, 0, 0, 1, 1, 0); break;
      case  405: CALL(1, 0, 1, 0, 1, 0, 0, 1, 1, 0); break;
      case  406: CALL(0, 1, 1, 0, 1, 0, 0, 1, 1, 0); break;
      case  407: CALL(1, 1, 1, 0, 1, 0, 0, 1, 1, 0); break;
      case  408: CALL(0, 0, 0, 1, 1, 0, 0, 1, 1, 0); break;
      case  409: CALL(1, 0, 0, 1, 1, 0, 0, 1, 1, 0); break;
      case  410: CALL(0, 1, 0, 1, 1, 0, 0, 1, 1, 0); break;
      case  411: CALL(1, 1, 0, 1, 1, 0, 0, 1, 1, 0); break;
      case  412: CALL(0, 0, 1, 1, 1, 0, 0, 1, 1, 0); break;
      case  413: CALL(1, 0, 1, 1, 1, 0, 0, 1, 1, 0); break;
      case  414: CALL(0, 1, 1, 1, 1, 0, 0, 1, 1, 0); break;
      case  415: CALL(1, 1, 1, 1, 1, 0, 0, 1, 1, 0); break;

      case  416: CALL(0, 0, 0, 0, 0, 1, 0, 1, 1, 0); break;
      case  417: CALL(1, 0, 0, 0, 0, 1, 0, 1, 1, 0); break;
      case  418: CALL(0, 1, 0, 0, 0, 1, 0, 1, 1, 0); break;
      case  419: CALL(1, 1, 0, 0, 0, 1, 0, 1, 1, 0); break;
      case  420: CALL(0, 0, 1, 0, 0, 1, 0, 1, 1, 0); break;
      case  421: CALL(1, 0, 1, 0, 0, 1, 0, 1, 1, 0); break;
      case  422: CALL(0, 1, 1, 0, 0, 1, 0, 1, 1, 0); break;
      case  423: CALL(1, 1, 1, 0, 0, 1, 0, 1, 1, 0); break;
      case  424: CALL(0, 0, 0, 1, 0, 1, 0, 1, 1, 0); break;
      case  425: CALL(1, 0, 0, 1, 0, 1, 0, 1, 1, 0); break;
      case  426: CALL(0, 1, 0, 1, 0, 1, 0, 1, 1, 0); break;
      case  427: CALL(1, 1, 0, 1, 0, 1, 0, 1, 1, 0); break;
      case  428: CALL(0, 0, 1, 1, 0, 1, 0, 1, 1, 0); break;
      case  429: CALL(1, 0, 1, 1, 0, 1, 0, 1, 1, 0); break;
      case  430: CALL(0, 1, 1, 1, 0, 1, 0, 1, 1, 0); break;
      case  431: CALL(1, 1, 1, 1, 0, 1, 0, 1, 1, 0); break;

#endif
      case  432: CALL(0, 0, 0, 0, 1, 1, 0, 1, 1, 0); break;
      case  433: CALL(1, 0, 0, 0, 1, 1, 0, 1, 1, 0); break;
      case  434: CALL(0, 1, 0, 0, 1, 1, 0, 1, 1, 0); break;
      case  435: CALL(1, 1, 0, 0, 1, 1, 0, 1, 1, 0); break;
      case  436: CALL(0, 0, 1, 0, 1, 1, 0, 1, 1, 0); break;
      case  437: CALL(1, 0, 1, 0, 1, 1, 0, 1, 1, 0); break;
      case  438: CALL(0, 1, 1, 0, 1, 1, 0, 1, 1, 0); break;
      case  439: CALL(1, 1, 1, 0, 1, 1, 0, 1, 1, 0); break;
      case  440: CALL(0, 0, 0, 1, 1, 1, 0, 1, 1, 0); break;
      case  441: CALL(1, 0, 0, 1, 1, 1, 0, 1, 1, 0); break;
      case  442: CALL(0, 1, 0, 1, 1, 1, 0, 1, 1, 0); break;
      case  443: CALL(1, 1, 0, 1, 1, 1, 0, 1, 1, 0); break;
      case  444: CALL(0, 0, 1, 1, 1, 1, 0, 1, 1, 0); break;
      case  445: CALL(1, 0, 1, 1, 1, 1, 0, 1, 1, 0); break;
      case  446: CALL(0, 1, 1, 1, 1, 1, 0, 1, 1, 0); break;
      case  447: CALL(1, 1, 1, 1, 1, 1, 0, 1, 1, 0); break;

#if 0
      case  448: CALL(0, 0, 0, 0, 0, 0, 1, 1, 1, 0); break;
      case  449: CALL(1, 0, 0, 0, 0, 0, 1, 1, 1, 0); break;
      case  450: CALL(0, 1, 0, 0, 0, 0, 1, 1, 1, 0); break;
      case  451: CALL(1, 1, 0, 0, 0, 0, 1, 1, 1, 0); break;
      case  452: CALL(0, 0, 1, 0, 0, 0, 1, 1, 1, 0); break;
      case  453: CALL(1, 0, 1, 0, 0, 0, 1, 1, 1, 0); break;
      case  454: CALL(0, 1, 1, 0, 0, 0, 1, 1, 1, 0); break;
      case  455: CALL(1, 1, 1, 0, 0, 0, 1, 1, 1, 0); break;
      case  456: CALL(0, 0, 0, 1, 0, 0, 1, 1, 1, 0); break;
      case  457: CALL(1, 0, 0, 1, 0, 0, 1, 1, 1, 0); break;
      case  458: CALL(0, 1, 0, 1, 0, 0, 1, 1, 1, 0); break;
      case  459: CALL(1, 1, 0, 1, 0, 0, 1, 1, 1, 0); break;
      case  460: CALL(0, 0, 1, 1, 0, 0, 1, 1, 1, 0); break;
      case  461: CALL(1, 0, 1, 1, 0, 0, 1, 1, 1, 0); break;
      case  462: CALL(0, 1, 1, 1, 0, 0, 1, 1, 1, 0); break;
      case  463: CALL(1, 1, 1, 1, 0, 0, 1, 1, 1, 0); break;

#endif
      case  464: CALL(0, 0, 0, 0, 1, 0, 1, 1, 1, 0); break;
      case  465: CALL(1, 0, 0, 0, 1, 0, 1, 1, 1, 0); break;
      case  466: CALL(0, 1, 0, 0, 1, 0, 1, 1, 1, 0); break;
      case  467: CALL(1, 1, 0, 0, 1, 0, 1, 1, 1, 0); break;
      case  468: CALL(0, 0, 1, 0, 1, 0, 1, 1, 1, 0); break;
      case  469: CALL(1, 0, 1, 0, 1, 0, 1, 1, 1, 0); break;
      case  470: CALL(0, 1, 1, 0, 1, 0, 1, 1, 1, 0); break;
      case  471: CALL(1, 1, 1, 0, 1, 0, 1, 1, 1, 0); break;
      case  472: CALL(0, 0, 0, 1, 1, 0, 1, 1, 1, 0); break;
      case  473: CALL(1, 0, 0, 1, 1, 0, 1, 1, 1, 0); break;
      case  474: CALL(0, 1, 0, 1, 1, 0, 1, 1, 1, 0); break;
      case  475: CALL(1, 1, 0, 1, 1, 0, 1, 1, 1, 0); break;
      case  476: CALL(0, 0, 1, 1, 1, 0, 1, 1, 1, 0); break;
      case  477: CALL(1, 0, 1, 1, 1, 0, 1, 1, 1, 0); break;
      case  478: CALL(0, 1, 1, 1, 1, 0, 1, 1, 1, 0); break;
      case  479: CALL(1, 1, 1, 1, 1, 0, 1, 1, 1, 0); break;

#if 0
      case  480: CALL(0, 0, 0, 0, 0, 1, 1, 1, 1, 0); break;
      case  481: CALL(1, 0, 0, 0, 0, 1, 1, 1, 1, 0); break;
      case  482: CALL(0, 1, 0, 0, 0, 1, 1, 1, 1, 0); break;
      case  483: CALL(1, 1, 0, 0, 0, 1, 1, 1, 1, 0); break;
      case  484: CALL(0, 0, 1, 0, 0, 1, 1, 1, 1, 0); break;
      case  485: CALL(1, 0, 1, 0, 0, 1, 1, 1, 1, 0); break;
      case  486: CALL(0, 1, 1, 0, 0, 1, 1, 1, 1, 0); break;
      case  487: CALL(1, 1, 1, 0, 0, 1, 1, 1, 1, 0); break;
      case  488: CALL(0, 0, 0, 1, 0, 1, 1, 1, 1, 0); break;
      case  489: CALL(1, 0, 0, 1, 0, 1, 1, 1, 1, 0); break;
      case  490: CALL(0, 1, 0, 1, 0, 1, 1, 1, 1, 0); break;
      case  491: CALL(1, 1, 0, 1, 0, 1, 1, 1, 1, 0); break;
      case  492: CALL(0, 0, 1, 1, 0, 1, 1, 1, 1, 0); break;
      case  493: CALL(1, 0, 1, 1, 0, 1, 1, 1, 1, 0); break;
      case  494: CALL(0, 1, 1, 1, 0, 1, 1, 1, 1, 0); break;
      case  495: CALL(1, 1, 1, 1, 0, 1, 1, 1, 1, 0); break;

      case  496: CALL(0, 0, 0, 0, 1, 1, 1, 1, 1, 0); break;
      case  497: CALL(1, 0, 0, 0, 1, 1, 1, 1, 1, 0); break;
      case  498: CALL(0, 1, 0, 0, 1, 1, 1, 1, 1, 0); break;
      case  499: CALL(1, 1, 0, 0, 1, 1, 1, 1, 1, 0); break;
      case  500: CALL(0, 0, 1, 0, 1, 1, 1, 1, 1, 0); break;
      case  501: CALL(1, 0, 1, 0, 1, 1, 1, 1, 1, 0); break;
      case  502: CALL(0, 1, 1, 0, 1, 1, 1, 1, 1, 0); break;
      case  503: CALL(1, 1, 1, 0, 1, 1, 1, 1, 1, 0); break;
      case  504: CALL(0, 0, 0, 1, 1, 1, 1, 1, 1, 0); break;
      case  505: CALL(1, 0, 0, 1, 1, 1, 1, 1, 1, 0); break;
      case  506: CALL(0, 1, 0, 1, 1, 1, 1, 1, 1, 0); break;
      case  507: CALL(1, 1, 0, 1, 1, 1, 1, 1, 1, 0); break;
      case  508: CALL(0, 0, 1, 1, 1, 1, 1, 1, 1, 0); break;
      case  509: CALL(1, 0, 1, 1, 1, 1, 1, 1, 1, 0); break;
      case  510: CALL(0, 1, 1, 1, 1, 1, 1, 1, 1, 0); break;
      case  511: CALL(1, 1, 1, 1, 1, 1, 1, 1, 1, 0); break;
#endif
      /*
       * Haochuan: the calls starting from 512 to 1023 were generated by the following python script
       * #!/usr/bin/env python3
       * def gen_call(option: int):
       *     doEnergy = option & 1
       *     doVirial = (option >> 1) & 1
       *     doSlow = (option >> 2) & 1
       *     doPairlist = (option >> 3) & 1
       *     doAlch = (option >> 4) & 1
       *     doFEP = (option >> 5) & 1
       *     doTI = (option >> 6) & 1
       *     doStreaming = (option >> 7) & 1
       *     doTable = (option >> 8) & 1
       *     doAlchVdwForceSwitching = (option >> 9) & 1
       *     incompatible = False
       *     incompatible = incompatible | (doFEP and doTI)
       *     incompatible = incompatible | (doAlch and ((not doFEP) and (not doTI)))
       *     incompatible = incompatible | ((not doAlch) and (doFEP or doTI or doAlchVdwForceSwitching))
       *     incompatible = incompatible | ((not doTable) and (doAlch or doTI or doFEP or doAlchVdwForceSwitching))
       *     if incompatible:
       *         pass
       *         print(f'      // case {option}: CALL({doEnergy}, {doVirial}, {doSlow}, {doPairlist}, {doAlch}, {doFEP}, {doTI}, {doStreaming}, {doTable}, {doAlchVdwForceSwitching}); break;')
       *     else:
       *         print(f'      case {option}: CALL({doEnergy}, {doVirial}, {doSlow}, {doPairlist}, {doAlch}, {doFEP}, {doTI}, {doStreaming}, {doTable}, {doAlchVdwForceSwitching}); break;')
       *     return
       *
       * for i in range(512, 1024):
       *     gen_call(i)
       *
       */
      // case 512: CALL(0, 0, 0, 0, 0, 0, 0, 0, 0, 1); break;
      // case 513: CALL(1, 0, 0, 0, 0, 0, 0, 0, 0, 1); break;
      // case 514: CALL(0, 1, 0, 0, 0, 0, 0, 0, 0, 1); break;
      // case 515: CALL(1, 1, 0, 0, 0, 0, 0, 0, 0, 1); break;
      // case 516: CALL(0, 0, 1, 0, 0, 0, 0, 0, 0, 1); break;
      // case 517: CALL(1, 0, 1, 0, 0, 0, 0, 0, 0, 1); break;
      // case 518: CALL(0, 1, 1, 0, 0, 0, 0, 0, 0, 1); break;
      // case 519: CALL(1, 1, 1, 0, 0, 0, 0, 0, 0, 1); break;
      // case 520: CALL(0, 0, 0, 1, 0, 0, 0, 0, 0, 1); break;
      // case 521: CALL(1, 0, 0, 1, 0, 0, 0, 0, 0, 1); break;
      // case 522: CALL(0, 1, 0, 1, 0, 0, 0, 0, 0, 1); break;
      // case 523: CALL(1, 1, 0, 1, 0, 0, 0, 0, 0, 1); break;
      // case 524: CALL(0, 0, 1, 1, 0, 0, 0, 0, 0, 1); break;
      // case 525: CALL(1, 0, 1, 1, 0, 0, 0, 0, 0, 1); break;
      // case 526: CALL(0, 1, 1, 1, 0, 0, 0, 0, 0, 1); break;
      // case 527: CALL(1, 1, 1, 1, 0, 0, 0, 0, 0, 1); break;
      // case 528: CALL(0, 0, 0, 0, 1, 0, 0, 0, 0, 1); break;
      // case 529: CALL(1, 0, 0, 0, 1, 0, 0, 0, 0, 1); break;
      // case 530: CALL(0, 1, 0, 0, 1, 0, 0, 0, 0, 1); break;
      // case 531: CALL(1, 1, 0, 0, 1, 0, 0, 0, 0, 1); break;
      // case 532: CALL(0, 0, 1, 0, 1, 0, 0, 0, 0, 1); break;
      // case 533: CALL(1, 0, 1, 0, 1, 0, 0, 0, 0, 1); break;
      // case 534: CALL(0, 1, 1, 0, 1, 0, 0, 0, 0, 1); break;
      // case 535: CALL(1, 1, 1, 0, 1, 0, 0, 0, 0, 1); break;
      // case 536: CALL(0, 0, 0, 1, 1, 0, 0, 0, 0, 1); break;
      // case 537: CALL(1, 0, 0, 1, 1, 0, 0, 0, 0, 1); break;
      // case 538: CALL(0, 1, 0, 1, 1, 0, 0, 0, 0, 1); break;
      // case 539: CALL(1, 1, 0, 1, 1, 0, 0, 0, 0, 1); break;
      // case 540: CALL(0, 0, 1, 1, 1, 0, 0, 0, 0, 1); break;
      // case 541: CALL(1, 0, 1, 1, 1, 0, 0, 0, 0, 1); break;
      // case 542: CALL(0, 1, 1, 1, 1, 0, 0, 0, 0, 1); break;
      // case 543: CALL(1, 1, 1, 1, 1, 0, 0, 0, 0, 1); break;
      // case 544: CALL(0, 0, 0, 0, 0, 1, 0, 0, 0, 1); break;
      // case 545: CALL(1, 0, 0, 0, 0, 1, 0, 0, 0, 1); break;
      // case 546: CALL(0, 1, 0, 0, 0, 1, 0, 0, 0, 1); break;
      // case 547: CALL(1, 1, 0, 0, 0, 1, 0, 0, 0, 1); break;
      // case 548: CALL(0, 0, 1, 0, 0, 1, 0, 0, 0, 1); break;
      // case 549: CALL(1, 0, 1, 0, 0, 1, 0, 0, 0, 1); break;
      // case 550: CALL(0, 1, 1, 0, 0, 1, 0, 0, 0, 1); break;
      // case 551: CALL(1, 1, 1, 0, 0, 1, 0, 0, 0, 1); break;
      // case 552: CALL(0, 0, 0, 1, 0, 1, 0, 0, 0, 1); break;
      // case 553: CALL(1, 0, 0, 1, 0, 1, 0, 0, 0, 1); break;
      // case 554: CALL(0, 1, 0, 1, 0, 1, 0, 0, 0, 1); break;
      // case 555: CALL(1, 1, 0, 1, 0, 1, 0, 0, 0, 1); break;
      // case 556: CALL(0, 0, 1, 1, 0, 1, 0, 0, 0, 1); break;
      // case 557: CALL(1, 0, 1, 1, 0, 1, 0, 0, 0, 1); break;
      // case 558: CALL(0, 1, 1, 1, 0, 1, 0, 0, 0, 1); break;
      // case 559: CALL(1, 1, 1, 1, 0, 1, 0, 0, 0, 1); break;
      // case 560: CALL(0, 0, 0, 0, 1, 1, 0, 0, 0, 1); break;
      // case 561: CALL(1, 0, 0, 0, 1, 1, 0, 0, 0, 1); break;
      // case 562: CALL(0, 1, 0, 0, 1, 1, 0, 0, 0, 1); break;
      // case 563: CALL(1, 1, 0, 0, 1, 1, 0, 0, 0, 1); break;
      // case 564: CALL(0, 0, 1, 0, 1, 1, 0, 0, 0, 1); break;
      // case 565: CALL(1, 0, 1, 0, 1, 1, 0, 0, 0, 1); break;
      // case 566: CALL(0, 1, 1, 0, 1, 1, 0, 0, 0, 1); break;
      // case 567: CALL(1, 1, 1, 0, 1, 1, 0, 0, 0, 1); break;
      // case 568: CALL(0, 0, 0, 1, 1, 1, 0, 0, 0, 1); break;
      // case 569: CALL(1, 0, 0, 1, 1, 1, 0, 0, 0, 1); break;
      // case 570: CALL(0, 1, 0, 1, 1, 1, 0, 0, 0, 1); break;
      // case 571: CALL(1, 1, 0, 1, 1, 1, 0, 0, 0, 1); break;
      // case 572: CALL(0, 0, 1, 1, 1, 1, 0, 0, 0, 1); break;
      // case 573: CALL(1, 0, 1, 1, 1, 1, 0, 0, 0, 1); break;
      // case 574: CALL(0, 1, 1, 1, 1, 1, 0, 0, 0, 1); break;
      // case 575: CALL(1, 1, 1, 1, 1, 1, 0, 0, 0, 1); break;
      // case 576: CALL(0, 0, 0, 0, 0, 0, 1, 0, 0, 1); break;
      // case 577: CALL(1, 0, 0, 0, 0, 0, 1, 0, 0, 1); break;
      // case 578: CALL(0, 1, 0, 0, 0, 0, 1, 0, 0, 1); break;
      // case 579: CALL(1, 1, 0, 0, 0, 0, 1, 0, 0, 1); break;
      // case 580: CALL(0, 0, 1, 0, 0, 0, 1, 0, 0, 1); break;
      // case 581: CALL(1, 0, 1, 0, 0, 0, 1, 0, 0, 1); break;
      // case 582: CALL(0, 1, 1, 0, 0, 0, 1, 0, 0, 1); break;
      // case 583: CALL(1, 1, 1, 0, 0, 0, 1, 0, 0, 1); break;
      // case 584: CALL(0, 0, 0, 1, 0, 0, 1, 0, 0, 1); break;
      // case 585: CALL(1, 0, 0, 1, 0, 0, 1, 0, 0, 1); break;
      // case 586: CALL(0, 1, 0, 1, 0, 0, 1, 0, 0, 1); break;
      // case 587: CALL(1, 1, 0, 1, 0, 0, 1, 0, 0, 1); break;
      // case 588: CALL(0, 0, 1, 1, 0, 0, 1, 0, 0, 1); break;
      // case 589: CALL(1, 0, 1, 1, 0, 0, 1, 0, 0, 1); break;
      // case 590: CALL(0, 1, 1, 1, 0, 0, 1, 0, 0, 1); break;
      // case 591: CALL(1, 1, 1, 1, 0, 0, 1, 0, 0, 1); break;
      // case 592: CALL(0, 0, 0, 0, 1, 0, 1, 0, 0, 1); break;
      // case 593: CALL(1, 0, 0, 0, 1, 0, 1, 0, 0, 1); break;
      // case 594: CALL(0, 1, 0, 0, 1, 0, 1, 0, 0, 1); break;
      // case 595: CALL(1, 1, 0, 0, 1, 0, 1, 0, 0, 1); break;
      // case 596: CALL(0, 0, 1, 0, 1, 0, 1, 0, 0, 1); break;
      // case 597: CALL(1, 0, 1, 0, 1, 0, 1, 0, 0, 1); break;
      // case 598: CALL(0, 1, 1, 0, 1, 0, 1, 0, 0, 1); break;
      // case 599: CALL(1, 1, 1, 0, 1, 0, 1, 0, 0, 1); break;
      // case 600: CALL(0, 0, 0, 1, 1, 0, 1, 0, 0, 1); break;
      // case 601: CALL(1, 0, 0, 1, 1, 0, 1, 0, 0, 1); break;
      // case 602: CALL(0, 1, 0, 1, 1, 0, 1, 0, 0, 1); break;
      // case 603: CALL(1, 1, 0, 1, 1, 0, 1, 0, 0, 1); break;
      // case 604: CALL(0, 0, 1, 1, 1, 0, 1, 0, 0, 1); break;
      // case 605: CALL(1, 0, 1, 1, 1, 0, 1, 0, 0, 1); break;
      // case 606: CALL(0, 1, 1, 1, 1, 0, 1, 0, 0, 1); break;
      // case 607: CALL(1, 1, 1, 1, 1, 0, 1, 0, 0, 1); break;
      // case 608: CALL(0, 0, 0, 0, 0, 1, 1, 0, 0, 1); break;
      // case 609: CALL(1, 0, 0, 0, 0, 1, 1, 0, 0, 1); break;
      // case 610: CALL(0, 1, 0, 0, 0, 1, 1, 0, 0, 1); break;
      // case 611: CALL(1, 1, 0, 0, 0, 1, 1, 0, 0, 1); break;
      // case 612: CALL(0, 0, 1, 0, 0, 1, 1, 0, 0, 1); break;
      // case 613: CALL(1, 0, 1, 0, 0, 1, 1, 0, 0, 1); break;
      // case 614: CALL(0, 1, 1, 0, 0, 1, 1, 0, 0, 1); break;
      // case 615: CALL(1, 1, 1, 0, 0, 1, 1, 0, 0, 1); break;
      // case 616: CALL(0, 0, 0, 1, 0, 1, 1, 0, 0, 1); break;
      // case 617: CALL(1, 0, 0, 1, 0, 1, 1, 0, 0, 1); break;
      // case 618: CALL(0, 1, 0, 1, 0, 1, 1, 0, 0, 1); break;
      // case 619: CALL(1, 1, 0, 1, 0, 1, 1, 0, 0, 1); break;
      // case 620: CALL(0, 0, 1, 1, 0, 1, 1, 0, 0, 1); break;
      // case 621: CALL(1, 0, 1, 1, 0, 1, 1, 0, 0, 1); break;
      // case 622: CALL(0, 1, 1, 1, 0, 1, 1, 0, 0, 1); break;
      // case 623: CALL(1, 1, 1, 1, 0, 1, 1, 0, 0, 1); break;
      // case 624: CALL(0, 0, 0, 0, 1, 1, 1, 0, 0, 1); break;
      // case 625: CALL(1, 0, 0, 0, 1, 1, 1, 0, 0, 1); break;
      // case 626: CALL(0, 1, 0, 0, 1, 1, 1, 0, 0, 1); break;
      // case 627: CALL(1, 1, 0, 0, 1, 1, 1, 0, 0, 1); break;
      // case 628: CALL(0, 0, 1, 0, 1, 1, 1, 0, 0, 1); break;
      // case 629: CALL(1, 0, 1, 0, 1, 1, 1, 0, 0, 1); break;
      // case 630: CALL(0, 1, 1, 0, 1, 1, 1, 0, 0, 1); break;
      // case 631: CALL(1, 1, 1, 0, 1, 1, 1, 0, 0, 1); break;
      // case 632: CALL(0, 0, 0, 1, 1, 1, 1, 0, 0, 1); break;
      // case 633: CALL(1, 0, 0, 1, 1, 1, 1, 0, 0, 1); break;
      // case 634: CALL(0, 1, 0, 1, 1, 1, 1, 0, 0, 1); break;
      // case 635: CALL(1, 1, 0, 1, 1, 1, 1, 0, 0, 1); break;
      // case 636: CALL(0, 0, 1, 1, 1, 1, 1, 0, 0, 1); break;
      // case 637: CALL(1, 0, 1, 1, 1, 1, 1, 0, 0, 1); break;
      // case 638: CALL(0, 1, 1, 1, 1, 1, 1, 0, 0, 1); break;
      // case 639: CALL(1, 1, 1, 1, 1, 1, 1, 0, 0, 1); break;
      // case 640: CALL(0, 0, 0, 0, 0, 0, 0, 1, 0, 1); break;
      // case 641: CALL(1, 0, 0, 0, 0, 0, 0, 1, 0, 1); break;
      // case 642: CALL(0, 1, 0, 0, 0, 0, 0, 1, 0, 1); break;
      // case 643: CALL(1, 1, 0, 0, 0, 0, 0, 1, 0, 1); break;
      // case 644: CALL(0, 0, 1, 0, 0, 0, 0, 1, 0, 1); break;
      // case 645: CALL(1, 0, 1, 0, 0, 0, 0, 1, 0, 1); break;
      // case 646: CALL(0, 1, 1, 0, 0, 0, 0, 1, 0, 1); break;
      // case 647: CALL(1, 1, 1, 0, 0, 0, 0, 1, 0, 1); break;
      // case 648: CALL(0, 0, 0, 1, 0, 0, 0, 1, 0, 1); break;
      // case 649: CALL(1, 0, 0, 1, 0, 0, 0, 1, 0, 1); break;
      // case 650: CALL(0, 1, 0, 1, 0, 0, 0, 1, 0, 1); break;
      // case 651: CALL(1, 1, 0, 1, 0, 0, 0, 1, 0, 1); break;
      // case 652: CALL(0, 0, 1, 1, 0, 0, 0, 1, 0, 1); break;
      // case 653: CALL(1, 0, 1, 1, 0, 0, 0, 1, 0, 1); break;
      // case 654: CALL(0, 1, 1, 1, 0, 0, 0, 1, 0, 1); break;
      // case 655: CALL(1, 1, 1, 1, 0, 0, 0, 1, 0, 1); break;
      // case 656: CALL(0, 0, 0, 0, 1, 0, 0, 1, 0, 1); break;
      // case 657: CALL(1, 0, 0, 0, 1, 0, 0, 1, 0, 1); break;
      // case 658: CALL(0, 1, 0, 0, 1, 0, 0, 1, 0, 1); break;
      // case 659: CALL(1, 1, 0, 0, 1, 0, 0, 1, 0, 1); break;
      // case 660: CALL(0, 0, 1, 0, 1, 0, 0, 1, 0, 1); break;
      // case 661: CALL(1, 0, 1, 0, 1, 0, 0, 1, 0, 1); break;
      // case 662: CALL(0, 1, 1, 0, 1, 0, 0, 1, 0, 1); break;
      // case 663: CALL(1, 1, 1, 0, 1, 0, 0, 1, 0, 1); break;
      // case 664: CALL(0, 0, 0, 1, 1, 0, 0, 1, 0, 1); break;
      // case 665: CALL(1, 0, 0, 1, 1, 0, 0, 1, 0, 1); break;
      // case 666: CALL(0, 1, 0, 1, 1, 0, 0, 1, 0, 1); break;
      // case 667: CALL(1, 1, 0, 1, 1, 0, 0, 1, 0, 1); break;
      // case 668: CALL(0, 0, 1, 1, 1, 0, 0, 1, 0, 1); break;
      // case 669: CALL(1, 0, 1, 1, 1, 0, 0, 1, 0, 1); break;
      // case 670: CALL(0, 1, 1, 1, 1, 0, 0, 1, 0, 1); break;
      // case 671: CALL(1, 1, 1, 1, 1, 0, 0, 1, 0, 1); break;
      // case 672: CALL(0, 0, 0, 0, 0, 1, 0, 1, 0, 1); break;
      // case 673: CALL(1, 0, 0, 0, 0, 1, 0, 1, 0, 1); break;
      // case 674: CALL(0, 1, 0, 0, 0, 1, 0, 1, 0, 1); break;
      // case 675: CALL(1, 1, 0, 0, 0, 1, 0, 1, 0, 1); break;
      // case 676: CALL(0, 0, 1, 0, 0, 1, 0, 1, 0, 1); break;
      // case 677: CALL(1, 0, 1, 0, 0, 1, 0, 1, 0, 1); break;
      // case 678: CALL(0, 1, 1, 0, 0, 1, 0, 1, 0, 1); break;
      // case 679: CALL(1, 1, 1, 0, 0, 1, 0, 1, 0, 1); break;
      // case 680: CALL(0, 0, 0, 1, 0, 1, 0, 1, 0, 1); break;
      // case 681: CALL(1, 0, 0, 1, 0, 1, 0, 1, 0, 1); break;
      // case 682: CALL(0, 1, 0, 1, 0, 1, 0, 1, 0, 1); break;
      // case 683: CALL(1, 1, 0, 1, 0, 1, 0, 1, 0, 1); break;
      // case 684: CALL(0, 0, 1, 1, 0, 1, 0, 1, 0, 1); break;
      // case 685: CALL(1, 0, 1, 1, 0, 1, 0, 1, 0, 1); break;
      // case 686: CALL(0, 1, 1, 1, 0, 1, 0, 1, 0, 1); break;
      // case 687: CALL(1, 1, 1, 1, 0, 1, 0, 1, 0, 1); break;
      // case 688: CALL(0, 0, 0, 0, 1, 1, 0, 1, 0, 1); break;
      // case 689: CALL(1, 0, 0, 0, 1, 1, 0, 1, 0, 1); break;
      // case 690: CALL(0, 1, 0, 0, 1, 1, 0, 1, 0, 1); break;
      // case 691: CALL(1, 1, 0, 0, 1, 1, 0, 1, 0, 1); break;
      // case 692: CALL(0, 0, 1, 0, 1, 1, 0, 1, 0, 1); break;
      // case 693: CALL(1, 0, 1, 0, 1, 1, 0, 1, 0, 1); break;
      // case 694: CALL(0, 1, 1, 0, 1, 1, 0, 1, 0, 1); break;
      // case 695: CALL(1, 1, 1, 0, 1, 1, 0, 1, 0, 1); break;
      // case 696: CALL(0, 0, 0, 1, 1, 1, 0, 1, 0, 1); break;
      // case 697: CALL(1, 0, 0, 1, 1, 1, 0, 1, 0, 1); break;
      // case 698: CALL(0, 1, 0, 1, 1, 1, 0, 1, 0, 1); break;
      // case 699: CALL(1, 1, 0, 1, 1, 1, 0, 1, 0, 1); break;
      // case 700: CALL(0, 0, 1, 1, 1, 1, 0, 1, 0, 1); break;
      // case 701: CALL(1, 0, 1, 1, 1, 1, 0, 1, 0, 1); break;
      // case 702: CALL(0, 1, 1, 1, 1, 1, 0, 1, 0, 1); break;
      // case 703: CALL(1, 1, 1, 1, 1, 1, 0, 1, 0, 1); break;
      // case 704: CALL(0, 0, 0, 0, 0, 0, 1, 1, 0, 1); break;
      // case 705: CALL(1, 0, 0, 0, 0, 0, 1, 1, 0, 1); break;
      // case 706: CALL(0, 1, 0, 0, 0, 0, 1, 1, 0, 1); break;
      // case 707: CALL(1, 1, 0, 0, 0, 0, 1, 1, 0, 1); break;
      // case 708: CALL(0, 0, 1, 0, 0, 0, 1, 1, 0, 1); break;
      // case 709: CALL(1, 0, 1, 0, 0, 0, 1, 1, 0, 1); break;
      // case 710: CALL(0, 1, 1, 0, 0, 0, 1, 1, 0, 1); break;
      // case 711: CALL(1, 1, 1, 0, 0, 0, 1, 1, 0, 1); break;
      // case 712: CALL(0, 0, 0, 1, 0, 0, 1, 1, 0, 1); break;
      // case 713: CALL(1, 0, 0, 1, 0, 0, 1, 1, 0, 1); break;
      // case 714: CALL(0, 1, 0, 1, 0, 0, 1, 1, 0, 1); break;
      // case 715: CALL(1, 1, 0, 1, 0, 0, 1, 1, 0, 1); break;
      // case 716: CALL(0, 0, 1, 1, 0, 0, 1, 1, 0, 1); break;
      // case 717: CALL(1, 0, 1, 1, 0, 0, 1, 1, 0, 1); break;
      // case 718: CALL(0, 1, 1, 1, 0, 0, 1, 1, 0, 1); break;
      // case 719: CALL(1, 1, 1, 1, 0, 0, 1, 1, 0, 1); break;
      // case 720: CALL(0, 0, 0, 0, 1, 0, 1, 1, 0, 1); break;
      // case 721: CALL(1, 0, 0, 0, 1, 0, 1, 1, 0, 1); break;
      // case 722: CALL(0, 1, 0, 0, 1, 0, 1, 1, 0, 1); break;
      // case 723: CALL(1, 1, 0, 0, 1, 0, 1, 1, 0, 1); break;
      // case 724: CALL(0, 0, 1, 0, 1, 0, 1, 1, 0, 1); break;
      // case 725: CALL(1, 0, 1, 0, 1, 0, 1, 1, 0, 1); break;
      // case 726: CALL(0, 1, 1, 0, 1, 0, 1, 1, 0, 1); break;
      // case 727: CALL(1, 1, 1, 0, 1, 0, 1, 1, 0, 1); break;
      // case 728: CALL(0, 0, 0, 1, 1, 0, 1, 1, 0, 1); break;
      // case 729: CALL(1, 0, 0, 1, 1, 0, 1, 1, 0, 1); break;
      // case 730: CALL(0, 1, 0, 1, 1, 0, 1, 1, 0, 1); break;
      // case 731: CALL(1, 1, 0, 1, 1, 0, 1, 1, 0, 1); break;
      // case 732: CALL(0, 0, 1, 1, 1, 0, 1, 1, 0, 1); break;
      // case 733: CALL(1, 0, 1, 1, 1, 0, 1, 1, 0, 1); break;
      // case 734: CALL(0, 1, 1, 1, 1, 0, 1, 1, 0, 1); break;
      // case 735: CALL(1, 1, 1, 1, 1, 0, 1, 1, 0, 1); break;
      // case 736: CALL(0, 0, 0, 0, 0, 1, 1, 1, 0, 1); break;
      // case 737: CALL(1, 0, 0, 0, 0, 1, 1, 1, 0, 1); break;
      // case 738: CALL(0, 1, 0, 0, 0, 1, 1, 1, 0, 1); break;
      // case 739: CALL(1, 1, 0, 0, 0, 1, 1, 1, 0, 1); break;
      // case 740: CALL(0, 0, 1, 0, 0, 1, 1, 1, 0, 1); break;
      // case 741: CALL(1, 0, 1, 0, 0, 1, 1, 1, 0, 1); break;
      // case 742: CALL(0, 1, 1, 0, 0, 1, 1, 1, 0, 1); break;
      // case 743: CALL(1, 1, 1, 0, 0, 1, 1, 1, 0, 1); break;
      // case 744: CALL(0, 0, 0, 1, 0, 1, 1, 1, 0, 1); break;
      // case 745: CALL(1, 0, 0, 1, 0, 1, 1, 1, 0, 1); break;
      // case 746: CALL(0, 1, 0, 1, 0, 1, 1, 1, 0, 1); break;
      // case 747: CALL(1, 1, 0, 1, 0, 1, 1, 1, 0, 1); break;
      // case 748: CALL(0, 0, 1, 1, 0, 1, 1, 1, 0, 1); break;
      // case 749: CALL(1, 0, 1, 1, 0, 1, 1, 1, 0, 1); break;
      // case 750: CALL(0, 1, 1, 1, 0, 1, 1, 1, 0, 1); break;
      // case 751: CALL(1, 1, 1, 1, 0, 1, 1, 1, 0, 1); break;
      // case 752: CALL(0, 0, 0, 0, 1, 1, 1, 1, 0, 1); break;
      // case 753: CALL(1, 0, 0, 0, 1, 1, 1, 1, 0, 1); break;
      // case 754: CALL(0, 1, 0, 0, 1, 1, 1, 1, 0, 1); break;
      // case 755: CALL(1, 1, 0, 0, 1, 1, 1, 1, 0, 1); break;
      // case 756: CALL(0, 0, 1, 0, 1, 1, 1, 1, 0, 1); break;
      // case 757: CALL(1, 0, 1, 0, 1, 1, 1, 1, 0, 1); break;
      // case 758: CALL(0, 1, 1, 0, 1, 1, 1, 1, 0, 1); break;
      // case 759: CALL(1, 1, 1, 0, 1, 1, 1, 1, 0, 1); break;
      // case 760: CALL(0, 0, 0, 1, 1, 1, 1, 1, 0, 1); break;
      // case 761: CALL(1, 0, 0, 1, 1, 1, 1, 1, 0, 1); break;
      // case 762: CALL(0, 1, 0, 1, 1, 1, 1, 1, 0, 1); break;
      // case 763: CALL(1, 1, 0, 1, 1, 1, 1, 1, 0, 1); break;
      // case 764: CALL(0, 0, 1, 1, 1, 1, 1, 1, 0, 1); break;
      // case 765: CALL(1, 0, 1, 1, 1, 1, 1, 1, 0, 1); break;
      // case 766: CALL(0, 1, 1, 1, 1, 1, 1, 1, 0, 1); break;
      // case 767: CALL(1, 1, 1, 1, 1, 1, 1, 1, 0, 1); break;
      // case 768: CALL(0, 0, 0, 0, 0, 0, 0, 0, 1, 1); break;
      // case 769: CALL(1, 0, 0, 0, 0, 0, 0, 0, 1, 1); break;
      // case 770: CALL(0, 1, 0, 0, 0, 0, 0, 0, 1, 1); break;
      // case 771: CALL(1, 1, 0, 0, 0, 0, 0, 0, 1, 1); break;
      // case 772: CALL(0, 0, 1, 0, 0, 0, 0, 0, 1, 1); break;
      // case 773: CALL(1, 0, 1, 0, 0, 0, 0, 0, 1, 1); break;
      // case 774: CALL(0, 1, 1, 0, 0, 0, 0, 0, 1, 1); break;
      // case 775: CALL(1, 1, 1, 0, 0, 0, 0, 0, 1, 1); break;
      // case 776: CALL(0, 0, 0, 1, 0, 0, 0, 0, 1, 1); break;
      // case 777: CALL(1, 0, 0, 1, 0, 0, 0, 0, 1, 1); break;
      // case 778: CALL(0, 1, 0, 1, 0, 0, 0, 0, 1, 1); break;
      // case 779: CALL(1, 1, 0, 1, 0, 0, 0, 0, 1, 1); break;
      // case 780: CALL(0, 0, 1, 1, 0, 0, 0, 0, 1, 1); break;
      // case 781: CALL(1, 0, 1, 1, 0, 0, 0, 0, 1, 1); break;
      // case 782: CALL(0, 1, 1, 1, 0, 0, 0, 0, 1, 1); break;
      // case 783: CALL(1, 1, 1, 1, 0, 0, 0, 0, 1, 1); break;
      // case 784: CALL(0, 0, 0, 0, 1, 0, 0, 0, 1, 1); break;
      // case 785: CALL(1, 0, 0, 0, 1, 0, 0, 0, 1, 1); break;
      // case 786: CALL(0, 1, 0, 0, 1, 0, 0, 0, 1, 1); break;
      // case 787: CALL(1, 1, 0, 0, 1, 0, 0, 0, 1, 1); break;
      // case 788: CALL(0, 0, 1, 0, 1, 0, 0, 0, 1, 1); break;
      // case 789: CALL(1, 0, 1, 0, 1, 0, 0, 0, 1, 1); break;
      // case 790: CALL(0, 1, 1, 0, 1, 0, 0, 0, 1, 1); break;
      // case 791: CALL(1, 1, 1, 0, 1, 0, 0, 0, 1, 1); break;
      // case 792: CALL(0, 0, 0, 1, 1, 0, 0, 0, 1, 1); break;
      // case 793: CALL(1, 0, 0, 1, 1, 0, 0, 0, 1, 1); break;
      // case 794: CALL(0, 1, 0, 1, 1, 0, 0, 0, 1, 1); break;
      // case 795: CALL(1, 1, 0, 1, 1, 0, 0, 0, 1, 1); break;
      // case 796: CALL(0, 0, 1, 1, 1, 0, 0, 0, 1, 1); break;
      // case 797: CALL(1, 0, 1, 1, 1, 0, 0, 0, 1, 1); break;
      // case 798: CALL(0, 1, 1, 1, 1, 0, 0, 0, 1, 1); break;
      // case 799: CALL(1, 1, 1, 1, 1, 0, 0, 0, 1, 1); break;
      // case 800: CALL(0, 0, 0, 0, 0, 1, 0, 0, 1, 1); break;
      // case 801: CALL(1, 0, 0, 0, 0, 1, 0, 0, 1, 1); break;
      // case 802: CALL(0, 1, 0, 0, 0, 1, 0, 0, 1, 1); break;
      // case 803: CALL(1, 1, 0, 0, 0, 1, 0, 0, 1, 1); break;
      // case 804: CALL(0, 0, 1, 0, 0, 1, 0, 0, 1, 1); break;
      // case 805: CALL(1, 0, 1, 0, 0, 1, 0, 0, 1, 1); break;
      // case 806: CALL(0, 1, 1, 0, 0, 1, 0, 0, 1, 1); break;
      // case 807: CALL(1, 1, 1, 0, 0, 1, 0, 0, 1, 1); break;
      // case 808: CALL(0, 0, 0, 1, 0, 1, 0, 0, 1, 1); break;
      // case 809: CALL(1, 0, 0, 1, 0, 1, 0, 0, 1, 1); break;
      // case 810: CALL(0, 1, 0, 1, 0, 1, 0, 0, 1, 1); break;
      // case 811: CALL(1, 1, 0, 1, 0, 1, 0, 0, 1, 1); break;
      // case 812: CALL(0, 0, 1, 1, 0, 1, 0, 0, 1, 1); break;
      // case 813: CALL(1, 0, 1, 1, 0, 1, 0, 0, 1, 1); break;
      // case 814: CALL(0, 1, 1, 1, 0, 1, 0, 0, 1, 1); break;
      // case 815: CALL(1, 1, 1, 1, 0, 1, 0, 0, 1, 1); break;
      case 816: CALL(0, 0, 0, 0, 1, 1, 0, 0, 1, 1); break;
      case 817: CALL(1, 0, 0, 0, 1, 1, 0, 0, 1, 1); break;
      case 818: CALL(0, 1, 0, 0, 1, 1, 0, 0, 1, 1); break;
      case 819: CALL(1, 1, 0, 0, 1, 1, 0, 0, 1, 1); break;
      case 820: CALL(0, 0, 1, 0, 1, 1, 0, 0, 1, 1); break;
      case 821: CALL(1, 0, 1, 0, 1, 1, 0, 0, 1, 1); break;
      case 822: CALL(0, 1, 1, 0, 1, 1, 0, 0, 1, 1); break;
      case 823: CALL(1, 1, 1, 0, 1, 1, 0, 0, 1, 1); break;
      case 824: CALL(0, 0, 0, 1, 1, 1, 0, 0, 1, 1); break;
      case 825: CALL(1, 0, 0, 1, 1, 1, 0, 0, 1, 1); break;
      case 826: CALL(0, 1, 0, 1, 1, 1, 0, 0, 1, 1); break;
      case 827: CALL(1, 1, 0, 1, 1, 1, 0, 0, 1, 1); break;
      case 828: CALL(0, 0, 1, 1, 1, 1, 0, 0, 1, 1); break;
      case 829: CALL(1, 0, 1, 1, 1, 1, 0, 0, 1, 1); break;
      case 830: CALL(0, 1, 1, 1, 1, 1, 0, 0, 1, 1); break;
      case 831: CALL(1, 1, 1, 1, 1, 1, 0, 0, 1, 1); break;
      // case 832: CALL(0, 0, 0, 0, 0, 0, 1, 0, 1, 1); break;
      // case 833: CALL(1, 0, 0, 0, 0, 0, 1, 0, 1, 1); break;
      // case 834: CALL(0, 1, 0, 0, 0, 0, 1, 0, 1, 1); break;
      // case 835: CALL(1, 1, 0, 0, 0, 0, 1, 0, 1, 1); break;
      // case 836: CALL(0, 0, 1, 0, 0, 0, 1, 0, 1, 1); break;
      // case 837: CALL(1, 0, 1, 0, 0, 0, 1, 0, 1, 1); break;
      // case 838: CALL(0, 1, 1, 0, 0, 0, 1, 0, 1, 1); break;
      // case 839: CALL(1, 1, 1, 0, 0, 0, 1, 0, 1, 1); break;
      // case 840: CALL(0, 0, 0, 1, 0, 0, 1, 0, 1, 1); break;
      // case 841: CALL(1, 0, 0, 1, 0, 0, 1, 0, 1, 1); break;
      // case 842: CALL(0, 1, 0, 1, 0, 0, 1, 0, 1, 1); break;
      // case 843: CALL(1, 1, 0, 1, 0, 0, 1, 0, 1, 1); break;
      // case 844: CALL(0, 0, 1, 1, 0, 0, 1, 0, 1, 1); break;
      // case 845: CALL(1, 0, 1, 1, 0, 0, 1, 0, 1, 1); break;
      // case 846: CALL(0, 1, 1, 1, 0, 0, 1, 0, 1, 1); break;
      // case 847: CALL(1, 1, 1, 1, 0, 0, 1, 0, 1, 1); break;
      case 848: CALL(0, 0, 0, 0, 1, 0, 1, 0, 1, 1); break;
      case 849: CALL(1, 0, 0, 0, 1, 0, 1, 0, 1, 1); break;
      case 850: CALL(0, 1, 0, 0, 1, 0, 1, 0, 1, 1); break;
      case 851: CALL(1, 1, 0, 0, 1, 0, 1, 0, 1, 1); break;
      case 852: CALL(0, 0, 1, 0, 1, 0, 1, 0, 1, 1); break;
      case 853: CALL(1, 0, 1, 0, 1, 0, 1, 0, 1, 1); break;
      case 854: CALL(0, 1, 1, 0, 1, 0, 1, 0, 1, 1); break;
      case 855: CALL(1, 1, 1, 0, 1, 0, 1, 0, 1, 1); break;
      case 856: CALL(0, 0, 0, 1, 1, 0, 1, 0, 1, 1); break;
      case 857: CALL(1, 0, 0, 1, 1, 0, 1, 0, 1, 1); break;
      case 858: CALL(0, 1, 0, 1, 1, 0, 1, 0, 1, 1); break;
      case 859: CALL(1, 1, 0, 1, 1, 0, 1, 0, 1, 1); break;
      case 860: CALL(0, 0, 1, 1, 1, 0, 1, 0, 1, 1); break;
      case 861: CALL(1, 0, 1, 1, 1, 0, 1, 0, 1, 1); break;
      case 862: CALL(0, 1, 1, 1, 1, 0, 1, 0, 1, 1); break;
      case 863: CALL(1, 1, 1, 1, 1, 0, 1, 0, 1, 1); break;
      // case 864: CALL(0, 0, 0, 0, 0, 1, 1, 0, 1, 1); break;
      // case 865: CALL(1, 0, 0, 0, 0, 1, 1, 0, 1, 1); break;
      // case 866: CALL(0, 1, 0, 0, 0, 1, 1, 0, 1, 1); break;
      // case 867: CALL(1, 1, 0, 0, 0, 1, 1, 0, 1, 1); break;
      // case 868: CALL(0, 0, 1, 0, 0, 1, 1, 0, 1, 1); break;
      // case 869: CALL(1, 0, 1, 0, 0, 1, 1, 0, 1, 1); break;
      // case 870: CALL(0, 1, 1, 0, 0, 1, 1, 0, 1, 1); break;
      // case 871: CALL(1, 1, 1, 0, 0, 1, 1, 0, 1, 1); break;
      // case 872: CALL(0, 0, 0, 1, 0, 1, 1, 0, 1, 1); break;
      // case 873: CALL(1, 0, 0, 1, 0, 1, 1, 0, 1, 1); break;
      // case 874: CALL(0, 1, 0, 1, 0, 1, 1, 0, 1, 1); break;
      // case 875: CALL(1, 1, 0, 1, 0, 1, 1, 0, 1, 1); break;
      // case 876: CALL(0, 0, 1, 1, 0, 1, 1, 0, 1, 1); break;
      // case 877: CALL(1, 0, 1, 1, 0, 1, 1, 0, 1, 1); break;
      // case 878: CALL(0, 1, 1, 1, 0, 1, 1, 0, 1, 1); break;
      // case 879: CALL(1, 1, 1, 1, 0, 1, 1, 0, 1, 1); break;
      // case 880: CALL(0, 0, 0, 0, 1, 1, 1, 0, 1, 1); break;
      // case 881: CALL(1, 0, 0, 0, 1, 1, 1, 0, 1, 1); break;
      // case 882: CALL(0, 1, 0, 0, 1, 1, 1, 0, 1, 1); break;
      // case 883: CALL(1, 1, 0, 0, 1, 1, 1, 0, 1, 1); break;
      // case 884: CALL(0, 0, 1, 0, 1, 1, 1, 0, 1, 1); break;
      // case 885: CALL(1, 0, 1, 0, 1, 1, 1, 0, 1, 1); break;
      // case 886: CALL(0, 1, 1, 0, 1, 1, 1, 0, 1, 1); break;
      // case 887: CALL(1, 1, 1, 0, 1, 1, 1, 0, 1, 1); break;
      // case 888: CALL(0, 0, 0, 1, 1, 1, 1, 0, 1, 1); break;
      // case 889: CALL(1, 0, 0, 1, 1, 1, 1, 0, 1, 1); break;
      // case 890: CALL(0, 1, 0, 1, 1, 1, 1, 0, 1, 1); break;
      // case 891: CALL(1, 1, 0, 1, 1, 1, 1, 0, 1, 1); break;
      // case 892: CALL(0, 0, 1, 1, 1, 1, 1, 0, 1, 1); break;
      // case 893: CALL(1, 0, 1, 1, 1, 1, 1, 0, 1, 1); break;
      // case 894: CALL(0, 1, 1, 1, 1, 1, 1, 0, 1, 1); break;
      // case 895: CALL(1, 1, 1, 1, 1, 1, 1, 0, 1, 1); break;
      // case 896: CALL(0, 0, 0, 0, 0, 0, 0, 1, 1, 1); break;
      // case 897: CALL(1, 0, 0, 0, 0, 0, 0, 1, 1, 1); break;
      // case 898: CALL(0, 1, 0, 0, 0, 0, 0, 1, 1, 1); break;
      // case 899: CALL(1, 1, 0, 0, 0, 0, 0, 1, 1, 1); break;
      // case 900: CALL(0, 0, 1, 0, 0, 0, 0, 1, 1, 1); break;
      // case 901: CALL(1, 0, 1, 0, 0, 0, 0, 1, 1, 1); break;
      // case 902: CALL(0, 1, 1, 0, 0, 0, 0, 1, 1, 1); break;
      // case 903: CALL(1, 1, 1, 0, 0, 0, 0, 1, 1, 1); break;
      // case 904: CALL(0, 0, 0, 1, 0, 0, 0, 1, 1, 1); break;
      // case 905: CALL(1, 0, 0, 1, 0, 0, 0, 1, 1, 1); break;
      // case 906: CALL(0, 1, 0, 1, 0, 0, 0, 1, 1, 1); break;
      // case 907: CALL(1, 1, 0, 1, 0, 0, 0, 1, 1, 1); break;
      // case 908: CALL(0, 0, 1, 1, 0, 0, 0, 1, 1, 1); break;
      // case 909: CALL(1, 0, 1, 1, 0, 0, 0, 1, 1, 1); break;
      // case 910: CALL(0, 1, 1, 1, 0, 0, 0, 1, 1, 1); break;
      // case 911: CALL(1, 1, 1, 1, 0, 0, 0, 1, 1, 1); break;
      // case 912: CALL(0, 0, 0, 0, 1, 0, 0, 1, 1, 1); break;
      // case 913: CALL(1, 0, 0, 0, 1, 0, 0, 1, 1, 1); break;
      // case 914: CALL(0, 1, 0, 0, 1, 0, 0, 1, 1, 1); break;
      // case 915: CALL(1, 1, 0, 0, 1, 0, 0, 1, 1, 1); break;
      // case 916: CALL(0, 0, 1, 0, 1, 0, 0, 1, 1, 1); break;
      // case 917: CALL(1, 0, 1, 0, 1, 0, 0, 1, 1, 1); break;
      // case 918: CALL(0, 1, 1, 0, 1, 0, 0, 1, 1, 1); break;
      // case 919: CALL(1, 1, 1, 0, 1, 0, 0, 1, 1, 1); break;
      // case 920: CALL(0, 0, 0, 1, 1, 0, 0, 1, 1, 1); break;
      // case 921: CALL(1, 0, 0, 1, 1, 0, 0, 1, 1, 1); break;
      // case 922: CALL(0, 1, 0, 1, 1, 0, 0, 1, 1, 1); break;
      // case 923: CALL(1, 1, 0, 1, 1, 0, 0, 1, 1, 1); break;
      // case 924: CALL(0, 0, 1, 1, 1, 0, 0, 1, 1, 1); break;
      // case 925: CALL(1, 0, 1, 1, 1, 0, 0, 1, 1, 1); break;
      // case 926: CALL(0, 1, 1, 1, 1, 0, 0, 1, 1, 1); break;
      // case 927: CALL(1, 1, 1, 1, 1, 0, 0, 1, 1, 1); break;
      // case 928: CALL(0, 0, 0, 0, 0, 1, 0, 1, 1, 1); break;
      // case 929: CALL(1, 0, 0, 0, 0, 1, 0, 1, 1, 1); break;
      // case 930: CALL(0, 1, 0, 0, 0, 1, 0, 1, 1, 1); break;
      // case 931: CALL(1, 1, 0, 0, 0, 1, 0, 1, 1, 1); break;
      // case 932: CALL(0, 0, 1, 0, 0, 1, 0, 1, 1, 1); break;
      // case 933: CALL(1, 0, 1, 0, 0, 1, 0, 1, 1, 1); break;
      // case 934: CALL(0, 1, 1, 0, 0, 1, 0, 1, 1, 1); break;
      // case 935: CALL(1, 1, 1, 0, 0, 1, 0, 1, 1, 1); break;
      // case 936: CALL(0, 0, 0, 1, 0, 1, 0, 1, 1, 1); break;
      // case 937: CALL(1, 0, 0, 1, 0, 1, 0, 1, 1, 1); break;
      // case 938: CALL(0, 1, 0, 1, 0, 1, 0, 1, 1, 1); break;
      // case 939: CALL(1, 1, 0, 1, 0, 1, 0, 1, 1, 1); break;
      // case 940: CALL(0, 0, 1, 1, 0, 1, 0, 1, 1, 1); break;
      // case 941: CALL(1, 0, 1, 1, 0, 1, 0, 1, 1, 1); break;
      // case 942: CALL(0, 1, 1, 1, 0, 1, 0, 1, 1, 1); break;
      // case 943: CALL(1, 1, 1, 1, 0, 1, 0, 1, 1, 1); break;
      case 944: CALL(0, 0, 0, 0, 1, 1, 0, 1, 1, 1); break;
      case 945: CALL(1, 0, 0, 0, 1, 1, 0, 1, 1, 1); break;
      case 946: CALL(0, 1, 0, 0, 1, 1, 0, 1, 1, 1); break;
      case 947: CALL(1, 1, 0, 0, 1, 1, 0, 1, 1, 1); break;
      case 948: CALL(0, 0, 1, 0, 1, 1, 0, 1, 1, 1); break;
      case 949: CALL(1, 0, 1, 0, 1, 1, 0, 1, 1, 1); break;
      case 950: CALL(0, 1, 1, 0, 1, 1, 0, 1, 1, 1); break;
      case 951: CALL(1, 1, 1, 0, 1, 1, 0, 1, 1, 1); break;
      case 952: CALL(0, 0, 0, 1, 1, 1, 0, 1, 1, 1); break;
      case 953: CALL(1, 0, 0, 1, 1, 1, 0, 1, 1, 1); break;
      case 954: CALL(0, 1, 0, 1, 1, 1, 0, 1, 1, 1); break;
      case 955: CALL(1, 1, 0, 1, 1, 1, 0, 1, 1, 1); break;
      case 956: CALL(0, 0, 1, 1, 1, 1, 0, 1, 1, 1); break;
      case 957: CALL(1, 0, 1, 1, 1, 1, 0, 1, 1, 1); break;
      case 958: CALL(0, 1, 1, 1, 1, 1, 0, 1, 1, 1); break;
      case 959: CALL(1, 1, 1, 1, 1, 1, 0, 1, 1, 1); break;
      // case 960: CALL(0, 0, 0, 0, 0, 0, 1, 1, 1, 1); break;
      // case 961: CALL(1, 0, 0, 0, 0, 0, 1, 1, 1, 1); break;
      // case 962: CALL(0, 1, 0, 0, 0, 0, 1, 1, 1, 1); break;
      // case 963: CALL(1, 1, 0, 0, 0, 0, 1, 1, 1, 1); break;
      // case 964: CALL(0, 0, 1, 0, 0, 0, 1, 1, 1, 1); break;
      // case 965: CALL(1, 0, 1, 0, 0, 0, 1, 1, 1, 1); break;
      // case 966: CALL(0, 1, 1, 0, 0, 0, 1, 1, 1, 1); break;
      // case 967: CALL(1, 1, 1, 0, 0, 0, 1, 1, 1, 1); break;
      // case 968: CALL(0, 0, 0, 1, 0, 0, 1, 1, 1, 1); break;
      // case 969: CALL(1, 0, 0, 1, 0, 0, 1, 1, 1, 1); break;
      // case 970: CALL(0, 1, 0, 1, 0, 0, 1, 1, 1, 1); break;
      // case 971: CALL(1, 1, 0, 1, 0, 0, 1, 1, 1, 1); break;
      // case 972: CALL(0, 0, 1, 1, 0, 0, 1, 1, 1, 1); break;
      // case 973: CALL(1, 0, 1, 1, 0, 0, 1, 1, 1, 1); break;
      // case 974: CALL(0, 1, 1, 1, 0, 0, 1, 1, 1, 1); break;
      // case 975: CALL(1, 1, 1, 1, 0, 0, 1, 1, 1, 1); break;
      case 976: CALL(0, 0, 0, 0, 1, 0, 1, 1, 1, 1); break;
      case 977: CALL(1, 0, 0, 0, 1, 0, 1, 1, 1, 1); break;
      case 978: CALL(0, 1, 0, 0, 1, 0, 1, 1, 1, 1); break;
      case 979: CALL(1, 1, 0, 0, 1, 0, 1, 1, 1, 1); break;
      case 980: CALL(0, 0, 1, 0, 1, 0, 1, 1, 1, 1); break;
      case 981: CALL(1, 0, 1, 0, 1, 0, 1, 1, 1, 1); break;
      case 982: CALL(0, 1, 1, 0, 1, 0, 1, 1, 1, 1); break;
      case 983: CALL(1, 1, 1, 0, 1, 0, 1, 1, 1, 1); break;
      case 984: CALL(0, 0, 0, 1, 1, 0, 1, 1, 1, 1); break;
      case 985: CALL(1, 0, 0, 1, 1, 0, 1, 1, 1, 1); break;
      case 986: CALL(0, 1, 0, 1, 1, 0, 1, 1, 1, 1); break;
      case 987: CALL(1, 1, 0, 1, 1, 0, 1, 1, 1, 1); break;
      case 988: CALL(0, 0, 1, 1, 1, 0, 1, 1, 1, 1); break;
      case 989: CALL(1, 0, 1, 1, 1, 0, 1, 1, 1, 1); break;
      case 990: CALL(0, 1, 1, 1, 1, 0, 1, 1, 1, 1); break;
      case 991: CALL(1, 1, 1, 1, 1, 0, 1, 1, 1, 1); break;
      // case 992: CALL(0, 0, 0, 0, 0, 1, 1, 1, 1, 1); break;
      // case 993: CALL(1, 0, 0, 0, 0, 1, 1, 1, 1, 1); break;
      // case 994: CALL(0, 1, 0, 0, 0, 1, 1, 1, 1, 1); break;
      // case 995: CALL(1, 1, 0, 0, 0, 1, 1, 1, 1, 1); break;
      // case 996: CALL(0, 0, 1, 0, 0, 1, 1, 1, 1, 1); break;
      // case 997: CALL(1, 0, 1, 0, 0, 1, 1, 1, 1, 1); break;
      // case 998: CALL(0, 1, 1, 0, 0, 1, 1, 1, 1, 1); break;
      // case 999: CALL(1, 1, 1, 0, 0, 1, 1, 1, 1, 1); break;
      // case 1000: CALL(0, 0, 0, 1, 0, 1, 1, 1, 1, 1); break;
      // case 1001: CALL(1, 0, 0, 1, 0, 1, 1, 1, 1, 1); break;
      // case 1002: CALL(0, 1, 0, 1, 0, 1, 1, 1, 1, 1); break;
      // case 1003: CALL(1, 1, 0, 1, 0, 1, 1, 1, 1, 1); break;
      // case 1004: CALL(0, 0, 1, 1, 0, 1, 1, 1, 1, 1); break;
      // case 1005: CALL(1, 0, 1, 1, 0, 1, 1, 1, 1, 1); break;
      // case 1006: CALL(0, 1, 1, 1, 0, 1, 1, 1, 1, 1); break;
      // case 1007: CALL(1, 1, 1, 1, 0, 1, 1, 1, 1, 1); break;
      // case 1008: CALL(0, 0, 0, 0, 1, 1, 1, 1, 1, 1); break;
      // case 1009: CALL(1, 0, 0, 0, 1, 1, 1, 1, 1, 1); break;
      // case 1010: CALL(0, 1, 0, 0, 1, 1, 1, 1, 1, 1); break;
      // case 1011: CALL(1, 1, 0, 0, 1, 1, 1, 1, 1, 1); break;
      // case 1012: CALL(0, 0, 1, 0, 1, 1, 1, 1, 1, 1); break;
      // case 1013: CALL(1, 0, 1, 0, 1, 1, 1, 1, 1, 1); break;
      // case 1014: CALL(0, 1, 1, 0, 1, 1, 1, 1, 1, 1); break;
      // case 1015: CALL(1, 1, 1, 0, 1, 1, 1, 1, 1, 1); break;
      // case 1016: CALL(0, 0, 0, 1, 1, 1, 1, 1, 1, 1); break;
      // case 1017: CALL(1, 0, 0, 1, 1, 1, 1, 1, 1, 1); break;
      // case 1018: CALL(0, 1, 0, 1, 1, 1, 1, 1, 1, 1); break;
      // case 1019: CALL(1, 1, 0, 1, 1, 1, 1, 1, 1, 1); break;
      // case 1020: CALL(0, 0, 1, 1, 1, 1, 1, 1, 1, 1); break;
      // case 1021: CALL(1, 0, 1, 1, 1, 1, 1, 1, 1, 1); break;
      // case 1022: CALL(0, 1, 1, 1, 1, 1, 1, 1, 1, 1); break;
      // case 1023: CALL(1, 1, 1, 1, 1, 1, 1, 1, 1, 1); break;
      default: {
#ifdef DEBUG
        sprintf(errmsg,
            "doEnergy=%d  doVirial=%d  doSlow=%d  doPairlist=%d  "
            "doAlch=%d  doFEP=%d  doTI=%d  doStreaming=%d doTable=%d" 
            "\noptions = %d\n",
            doEnergy, doVirial, doSlow, doPairlist, doAlch, doFEP, doTI,
            doStreaming, doTable, options);
        NAMD_die(errmsg);
#else
        std::string call_options;
        call_options += "doEnergy = " + std::to_string(int(doEnergy));
        call_options += ", doVirial = " + std::to_string(int(doVirial));
        call_options += ", doSlow = " + std::to_string(int(doSlow));
        call_options += ", doPairlist = " + std::to_string(int(doPairlist));
        call_options += ", doAlch = " + std::to_string(int(doAlch));
        call_options += ", doFEP = " + std::to_string(int(doFEP));
        call_options += ", doTI = " + std::to_string(int(doTI));
        call_options += ", doStreaming = " + std::to_string(int(doStreaming));
        // call_options += ", doTable = " + std::to_string(int(doTable));
        call_options += ", doAlchVdwForceSwitching = " + std::to_string(int(doAlchVdwForceSwitching));
        const std::string error = "CudaComputeNonbondedKernel::nonbondedForce, none of the kernels called. Options are:\n" + call_options;
        NAMD_bug(error.c_str());
#endif
      }
    }

#endif

#undef CALL
#undef TABLE_PARAMS
    cudaCheck(cudaGetLastError());

    start += nblock*nwarp;
  }

  if ( doVirial || ! doStreaming ){
    int block = 128;
    int grid = (atomStorageSize + block - 1)/block;
    if (doSlow)
      transposeForcesKernel<1><<<grid, block, 0, stream>>>(d_forces, d_forcesSlow,
                      force_x, force_y, force_z, force_w,
                      forceSlow_x, forceSlow_y, forceSlow_z, forceSlow_w,
                      atomStorageSize);
    else
      transposeForcesKernel<0><<<grid, block, 0, stream>>>(d_forces, d_forcesSlow,
                      force_x, force_y, force_z, force_w,
                      forceSlow_x, forceSlow_y, forceSlow_z, forceSlow_w,
                      atomStorageSize);
  }
}

//
// Perform virial and energy reductions for non-bonded force calculation
//
void CudaComputeNonbondedKernel::reduceVirialEnergy(CudaTileListKernel& tlKernel,
  const int atomStorageSize, const bool doEnergy, const bool doVirial, const bool doSlow, const bool doGBIS,
  float4* d_forces, float4* d_forcesSlow,
  VirialEnergy* d_virialEnergy, cudaStream_t stream) {

  if (doEnergy || doVirial) {
    clear_device_array<VirialEnergy>(d_virialEnergy, ATOMIC_BINS, stream);
  }

  if (doVirial)
  {
    int nthread = REDUCENONBONDEDVIRIALKERNEL_NUM_WARP*WARPSIZE;
    int nblock = min(deviceCUDA->getMaxNumBlocks(), (atomStorageSize-1)/nthread+1);
    reduceNonbondedVirialKernel <<< nblock, nthread, 0, stream >>>
    (doSlow, atomStorageSize, tlKernel.get_xyzq(), d_forces, d_forcesSlow, d_virialEnergy);
    cudaCheck(cudaGetLastError());
  }

  if (doVirial || doEnergy)
  {
    int nthread = REDUCEVIRIALENERGYKERNEL_NUM_WARP*WARPSIZE;
    int nblock = min(deviceCUDA->getMaxNumBlocks(), (tlKernel.getTileListVirialEnergyLength()-1)/nthread+1);
    reduceVirialEnergyKernel <<< nblock, nthread, 0, stream >>>
    (doEnergy, doVirial, doSlow, tlKernel.getTileListVirialEnergyLength(), tlKernel.getTileListVirialEnergy(), d_virialEnergy);
    cudaCheck(cudaGetLastError());
  }  

  if (doGBIS && doEnergy)
  {
    int nthread = REDUCEGBISENERGYKERNEL_NUM_WARP*WARPSIZE;
    int nblock = min(deviceCUDA->getMaxNumBlocks(), (tlKernel.getTileListVirialEnergyGBISLength()-1)/nthread+1);
    reduceGBISEnergyKernel <<< nblock, nthread, 0, stream >>>
    (tlKernel.getTileListVirialEnergyGBISLength(), tlKernel.getTileListVirialEnergy(), d_virialEnergy);
    cudaCheck(cudaGetLastError());
  }

  if (ATOMIC_BINS > 1)
  {
    // Reduce d_virialEnergy[ATOMIC_BINS] in-place (results are in d_virialEnergy[0])
    reduceNonbondedBinsKernel<<<1, ATOMIC_BINS, 0, stream>>>(doVirial, doEnergy, doSlow, doGBIS, d_virialEnergy);
  }
}

void CudaComputeNonbondedKernel::bindExclusions(int numExclusions, unsigned int* exclusion_bits) {
  reallocate_device<unsigned int>(&overflowExclusions, &overflowExclusionsSize, numExclusions);
  copy_HtoD_sync<unsigned int>(exclusion_bits, overflowExclusions, numExclusions);
}


void CudaComputeNonbondedKernel::setExclusionsByAtom(int2* h_data, const int num_atoms) {
  // Global data structure shouldn't be reallocated
  if (d_exclusionsByAtom == NULL) allocate_device<int2>(&d_exclusionsByAtom, num_atoms);
  copy_HtoD_sync<int2>(h_data, d_exclusionsByAtom, num_atoms);

}


template<bool kDoAlch>
__global__ void updateVdwTypesExclKernel(
  const int numPatches,
  const CudaLocalRecord* localRecords,
  const int* global_vdwTypes,
  const int* global_id,
  const int* patchSortOrder,
  const int2* exclusionsByAtom,
  const int* global_partition,
  int* vdwTypes,
  int* atomIndex,
  int2* exclusions,
  char* part
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

    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
      const int order = patchSortOrder[offset + i];
      const int id = global_id[offset + order];
      vdwTypes  [offsetNB + i]   = global_vdwTypes[offset + order];
      atomIndex [offsetNB + i]   = id;
      exclusions[offsetNB + i].x = exclusionsByAtom[id].y;
      exclusions[offsetNB + i].y = exclusionsByAtom[id].x;
      if (kDoAlch) {
        part    [offsetNB + i]   = global_partition[offset + order];
      }
    }
    __syncthreads();
  }
}


void CudaComputeNonbondedKernel::updateVdwTypesExclOnGPU(CudaTileListKernel& tlKernel,
  const int numPatches, const int atomStorageSize, const bool alchOn,
  CudaLocalRecord* localRecords,
  const int* d_vdwTypes, const int* d_id, const int* d_sortOrder, 
  const int* d_partition,
  cudaStream_t stream
) {
  reallocate_device<int>(&vdwTypes, &vdwTypesSize, atomStorageSize, OVERALLOC);
  reallocate_device<int2>(&exclIndexMaxDiff, &exclIndexMaxDiffSize, atomStorageSize, OVERALLOC);
  reallocate_device<int>(&atomIndex, &atomIndexSize, atomStorageSize, OVERALLOC);
 
  const int numBlocks = numPatches;
  const int numThreads = 512;
  
  if (alchOn) {
    updateVdwTypesExclKernel<true><<<numBlocks, numThreads, 0, stream>>>(
      numPatches, localRecords,
      d_vdwTypes, d_id, d_sortOrder, d_exclusionsByAtom, d_partition,
      vdwTypes, atomIndex, exclIndexMaxDiff, tlKernel.get_part()
    );
  } else {
    updateVdwTypesExclKernel<false><<<numBlocks, numThreads, 0, stream>>>(
      numPatches, localRecords,
      d_vdwTypes, d_id, d_sortOrder, d_exclusionsByAtom, d_partition,
      vdwTypes, atomIndex, exclIndexMaxDiff, tlKernel.get_part()
    );
  }
}

#endif  // NAMD_HIP
