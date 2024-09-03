#ifndef AVXTILESKERNEL_H
#define AVXTILESKERNEL_H

#include "ComputeNonbondedUtil.h" // WMB: for KNL_TABLE_FACTOR
#include "AVXTileLists.h"

#ifdef NAMD_AVXTILES
#include <immintrin.h>


FORCEINLINE void getOmpSimdTableI(const __m512 &r_1, __m512i &table_int,
				    __m512 &tableDiff, __m512 &rTableDiff) {
  // table_r_1 = table_r_1 > KNL_TABLE_MAX_R_1 ? KNL_TABLE_MAX_R_1 : r_1
  const __m512 maxv = _mm512_set1_ps(KNL_TABLE_MAX_R_1);
  const __mmask16 tmask = _mm512_cmplt_ps_mask(maxv, r_1);
  const __m512 table_r_1 = _mm512_mask_mov_ps(r_1, tmask, maxv);
  // table_f = (KNL_TABLE_FACTOR-2) * knl_table_r_1;
  const __m512 table_f = _mm512_mul_ps(_mm512_set1_ps(KNL_TABLE_FACTOR-2),
				       table_r_1);
  // table_int = floor(table_f)
  table_int = _mm512_cvttps_epi32(table_f);
  // tableDiff = table_f - table_int
  tableDiff = _mm512_sub_ps(table_f, _mm512_cvtepi32_ps(table_int));
  // rTableDiff = 1. - tableDiff;
  rTableDiff = _mm512_sub_ps(_mm512_set1_ps(1.f), tableDiff);
}

// Full electrostatics portion

template<bool doEnergy>
FORCEINLINE void forceEnergySlow512(const __mmask16 r2mask,
				      const __m512 &kqq,
				      const float * __restrict__ slowTable, 
				      const float * __restrict__ slowEtable,
				      const __m512i &table_int,
				      const __m512 &tableDiff, 
				      const __m512 &rTableDiff,
				      __m512 &forceSlow, __m512 &energySlow) {
  // Load interpolation values
  const __m512 t0 = (__m512)_mm512_mask_i32logather_pd(_mm512_undefined_pd(),
    r2mask, table_int, slowTable, _MM_SCALE_4);
  const __m512i table_int2 = _mm512_shuffle_i32x4(table_int, table_int, 238);
  const __mmask16 r2mask2 = r2mask >> 8;
  const __m512 t1 = (__m512)_mm512_mask_i32logather_pd(_mm512_undefined_pd(),
    r2mask2, table_int2, slowTable, _MM_SCALE_4);
  const __m512i t4 = _mm512_set_epi32(31,29,27,25,23,21,19,17,
				      15,13,11,9,7,5,3,1);
  const __m512 tabSlowP1 = _mm512_permutex2var_ps(t0, t4, t1);
  const __m512i t6 = _mm512_set_epi32(30,28,26,24,22,20,18,16,
				      14,12,10,8,6,4,2,0);
  const __m512 tabSlow = _mm512_permutex2var_ps(t0, t6, t1);

  // forceSlow = kqq * (-tabSlow * rTableDiff - tabSlowP1 * tableDiff)
  forceSlow = _mm512_mul_ps(kqq, _mm512_fnmsub_ps(tabSlow, rTableDiff,
    _mm512_mul_ps(tabSlowP1, tableDiff)));
  if (doEnergy) {
    // Load interpolation values for energy
    const __m512 t10 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), r2mask, table_int, slowEtable, _MM_SCALE_4);
    const __m512 t11 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), r2mask2, table_int2, slowEtable, _MM_SCALE_4);
    const __m512i t4 = _mm512_set_epi32(31,29,27,25,23,21,19,17,
					15,13,11,9,7,5,3,1);
    const __m512 tabSlowEp1 = _mm512_permutex2var_ps(t10, t4, t11);
    const __m512i t6 = _mm512_set_epi32(30,28,26,24,22,20,18,16,
					14,12,10,8,6,4,2,0);
    const __m512 tabSlowE = _mm512_permutex2var_ps(t10, t6, t11);

    // eSlow = tabSlowE * rTableDiff + tabSlowEp1 * tableDiff
    const __m512 eSlow = _mm512_fmadd_ps(tabSlowE, rTableDiff,
					 _mm512_mul_ps(tabSlowEp1, tableDiff));
    // energySlow = -kqq*eSlow + energySlow
    energySlow = _mm512_mask_mov_ps(energySlow, r2mask,
				    _mm512_fnmadd_ps(kqq, eSlow, energySlow));
  }
}

// Interpolation mode 2 (c1 splitting) and 3 variants

template<bool doEnergy, bool doSlow, int iMode>
FORCEINLINE void forceEnergyInterp2(const __m512 &r2, const __m512 &kqq,
				      const __m512i &type_i,
				      const __m512i &type_j, __m512 &force,
				      __m512 &forceSlow, __m512 &energyVdw,
				      __m512 &energyElec, __m512 &energySlow,
				      const __mmask16 r2mask, 
				      const float scaling, const float c1,
				      const float c3, const float switchOn2,
				      const float cutoff2,
				      const float mInvCut3,
				      const float cutUnder3,
				      const float * __restrict__ fastTable,
				      const float * __restrict__ energyTable,
				      const float * __restrict__ slowTable,
				      const float * __restrict__ slowEtable,
				      const float * __restrict__ ljTable,
				      const int ljWidth) {

  // lj_i = (type_i*ljWidth + type_j) * 2
  const __m512i lj_i = _mm512_slli_epi32(_mm512_add_epi32(
    _mm512_mullo_epi32(type_i,_mm512_set1_epi32(ljWidth)), type_j), 1);
  // Load A and B values from ljTable
  const __m512 t0 = (__m512)_mm512_mask_i32logather_pd(_mm512_undefined_pd(),
    r2mask, lj_i, ljTable, _MM_SCALE_8);
  const __m512i lj_i2 = _mm512_shuffle_i32x4(lj_i, lj_i, 238);
  const __mmask16 r2mask2 = r2mask >> 8;
  const __m512 t1 = (__m512)_mm512_mask_i32logather_pd(_mm512_undefined_pd(),
    r2mask2, lj_i2, ljTable, _MM_SCALE_8);
  const __m512i t4 = _mm512_set_epi32(31,29,27,25,23,21,19,17,
				      15,13,11,9,7,5,3,1);
  const __m512 B = _mm512_permutex2var_ps(t0, t4, t1);
  const __m512i t6 = _mm512_set_epi32(30,28,26,24,22,20,18,16,
				      14,12,10,8,6,4,2,0);
  const __m512 A = _mm512_permutex2var_ps(t0, t6, t1);

  // r_1 = 1./sqrt(r2)
  const __m512 r_1 = _mm512_invsqrt_ps(r2);
  __m512 tableDiff, rTableDiff;
  __m512i table_int;
  if (iMode == 3 || doSlow)
    getOmpSimdTableI(r_1, table_int, tableDiff, rTableDiff);

  // Get interpolation values for fast
  __m512 tabFast, tabFastp1, tabEnergy, tabEnergyp1;
  __m512 tabSlow, tabSlowp1, tabSlowE, tabSlowEp1;
  if (iMode == 3) {
    const __m512 t0 = (__m512)_mm512_mask_i32logather_pd(_mm512_undefined_pd(),
      r2mask, table_int, fastTable, _MM_SCALE_4);
    const __m512i table_int2 = _mm512_shuffle_i32x4(table_int, table_int, 238);
    const __mmask16 r2mask2 = r2mask >> 8;
    const __m512 t1 = (__m512)_mm512_mask_i32logather_pd(_mm512_undefined_pd(),
      r2mask2, table_int2, fastTable, _MM_SCALE_4);
    const __m512i t4 = _mm512_set_epi32(31,29,27,25,23,21,19,17,
					15,13,11,9,7,5,3,1);
    tabFastp1 = _mm512_permutex2var_ps(t0, t4, t1);
    const __m512i t6 = _mm512_set_epi32(30,28,26,24,22,20,18,16,
					14,12,10,8,6,4,2,0);
    tabFast = _mm512_permutex2var_ps(t0, t6, t1);

    if (doEnergy) {
      const __m512 t10 = (__m512)_mm512_mask_i32logather_pd(
        _mm512_undefined_pd(), r2mask, table_int, energyTable, _MM_SCALE_4);
      const __m512 t11 = (__m512)_mm512_mask_i32logather_pd(
        _mm512_undefined_pd(), r2mask2, table_int2, energyTable, _MM_SCALE_4);
      const __m512i t4 = _mm512_set_epi32(31,29,27,25,23,21,19,17,
					  15,13,11,9,7,5,3,1);
      tabEnergyp1 = _mm512_permutex2var_ps(t10, t4, t11);
      const __m512i t6 = _mm512_set_epi32(30,28,26,24,22,20,18,16,
					  14,12,10,8,6,4,2,0);
      tabEnergy = _mm512_permutex2var_ps(t10, t6, t11);
    }
  }

  // r_2 = r_1 * r_1
  const __m512 r_2 = _mm512_mul_ps(r_1, r_1);
  // r_6 = r_2 * r_2 * r_2
  const __m512 r_6 = _mm512_mul_ps(r_2, _mm512_mul_ps(r_2, r_2));
  // r_12 = r_6 * r_6;
  const __m512 r_12 = _mm512_mul_ps(r_6, r_6);
  // c2 = cutoff2 - r2
  const __m512 c2 = _mm512_sub_ps(_mm512_set1_ps(cutoff2), r2);
  // c4 = (-2. * c2 + c3) * c2
  const __m512 c4 = _mm512_mul_ps(_mm512_fnmadd_ps(_mm512_set1_ps(2.f), c2, 
						   _mm512_set1_ps(c3)), c2);
  // switchVal = r2 > switchOn2 ? c2*c4*c1 : 1.
  const __mmask16 switchMask = _mm512_cmplt_ps_mask(_mm512_set1_ps(switchOn2),
						    r2);
  const __m512 switchVal = _mm512_mask_mov_ps(_mm512_set1_ps(1.f), switchMask,
    _mm512_mul_ps(c2, _mm512_mul_ps(c4, _mm512_set1_ps(c1))));
  // dSwitchVal = r2 > switchOn2 ? 2.*c1*(c2*c2 - c4) : 0.
  const __m512 dSwitchVal = _mm512_mask_mov_ps(_mm512_setzero_ps(),
    switchMask, _mm512_mul_ps(_mm512_set1_ps(2.f),
     _mm512_mul_ps(_mm512_set1_ps(c1), _mm512_fmsub_ps(c2,c2,c4))));
  // r2SwitchVal = switchVal * r_2
  const __m512 r2SwitchVal = _mm512_mul_ps(switchVal, r_2);
  // vdwAgradient = (-6.*r2SwitchVal + dSwitchVal) * r_12
  const __m512 vdwAgradient = _mm512_mul_ps(_mm512_fnmadd_ps(
    _mm512_set1_ps(6.f), r2SwitchVal, dSwitchVal), r_12);
  // vdwBgradient = (-3. * r2SwitchVal + dSwitchVal) * r_6
  const __m512 vdwBgradient = _mm512_mul_ps(_mm512_fnmadd_ps(
    _mm512_set1_ps(3.f), r2SwitchVal, dSwitchVal), r_6);
  // vdwB = scaling * (A*vdwAgradient - B*vdwBgradient)
  const __m512 vdwB = _mm512_mul_ps(_mm512_set1_ps(scaling), 
    _mm512_fmsub_ps(A,vdwAgradient, _mm512_mul_ps(B, vdwBgradient)));

  __m512 ffast;
  // ffast = kqq * (r_2*r_1+mInvCut3);
  if (iMode == 2) ffast = _mm512_mul_ps(kqq, _mm512_fmadd_ps(r_2,r_1,
    _mm512_set1_ps(mInvCut3)));
  // ffast = kqq * (tabFast*rTableDiff+tabFastp1*tableDiff)
  else ffast = _mm512_mul_ps(kqq, _mm512_fmadd_ps(tabFast, rTableDiff,
    _mm512_mul_ps(tabFastp1, tableDiff)));

  if (doEnergy) {
    __m512 efast;
    if (iMode == 2) {
      // efast = r2*mInvCut3 + cutUnder3
      efast = _mm512_fmadd_ps(r2,_mm512_set1_ps(mInvCut3),
			      _mm512_set1_ps(cutUnder3));
      // efast = efast*0.5f - r_1
      efast = _mm512_fmsub_ps(efast,_mm512_set1_ps(0.5f), r_1);
      // efast = tabEnergy*rTableDiff + tabEnergyp1*tableDiff
    } else
      efast = _mm512_fmadd_ps(tabEnergy, rTableDiff,
			      _mm512_mul_ps(tabEnergyp1, tableDiff));

    // vdwTerm = A*r_12 - B*r_6
    const __m512 vdwTerm = _mm512_fmsub_ps(A, r_12, _mm512_mul_ps(B, r_6));
    // energyVdw = switchVal * vdwTerm + energyVdw
    energyVdw = _mm512_mask_mov_ps(energyVdw, r2mask,
      _mm512_fmadd_ps(switchVal, vdwTerm, energyVdw));
    // energyElec = -kqq*efast + energyElec
    energyElec = _mm512_mask_mov_ps(energyElec, r2mask,
				    _mm512_fnmadd_ps(kqq, efast, energyElec));
  }

  // force = vdwB - ffast
  force = _mm512_sub_ps(vdwB, ffast);
  if (doSlow)
    forceEnergySlow512<doEnergy>(r2mask, kqq, slowTable, slowEtable, table_int,
				 tableDiff, rTableDiff, forceSlow, energySlow);
}

// Interpolation mode 1 variant of "fast" calculations
// -- Supports arithmetic VDW mixing and C1 splitting

template<bool doEnergy, bool doSlow>
FORCEINLINE void forceEnergyInterp1(const __m512 &r2, const __m512 &kqq,
				      __m512 &force, __m512 &forceSlow,
				      __m512 &energyVdw, __m512 &energyElec,
				      __m512 &energySlow,
				      const __mmask16 r2mask, const float c1,
				      const float c3, const float switchOn2,
				      const float cutoff2,
				      const float mInvCut3, 
				      const float cutUnder3,
				      const float * __restrict__ slowTable,
				      const float * __restrict__ slowEtable,
				      const __m512 &eps4i, const __m512 &eps4j,
				      const __m512 &sigmaI, 
				      const __m512 &sigmaJ) {

  // eps_ij = sqrt(eps4i * eps4j)
  const __m512 eps_ij = _mm512_sqrt_ps(_mm512_mul_ps(eps4i, eps4j));
  // sigma_ij = 0.5f * (sigmaI + sigmaJ)
  __m512 sigma_ij = _mm512_mul_ps(_mm512_set1_ps(0.5f),
				  _mm512_add_ps(sigmaI, sigmaJ));
  // sigma_ij *= sigma_ij * sigma_ij
  sigma_ij = _mm512_mul_ps(sigma_ij, _mm512_mul_ps(sigma_ij, sigma_ij));
  // sigma_ij *= sigma_ij
  sigma_ij = _mm512_mul_ps(sigma_ij, sigma_ij);
  // B = sigma_ij * eps_ij
  const __m512 B(_mm512_mul_ps(sigma_ij, eps_ij));
  // A = B * sigma_ij
  const __m512 A(_mm512_mul_ps(B, sigma_ij));
  
  // r_1 = 1./sqrt(r2)
  const __m512 r_1 = _mm512_invsqrt_ps(r2);
  __m512 tableDiff, rTableDiff;;
  __m512i table_int;
  if (doSlow)
    getOmpSimdTableI(r_1, table_int, tableDiff, rTableDiff);

  // r_2 = r_1 * r_1
  const __m512 r_2 = _mm512_mul_ps(r_1, r_1);
  // r_6 = r_2 * r_2 * r_2
  const __m512 r_6 = _mm512_mul_ps(r_2, _mm512_mul_ps(r_2, r_2));
  // r_12 = r_6 * r_6
  const __m512 r_12 = _mm512_mul_ps(r_6, r_6);
  // c2 = cutoff2 - r2
  const __m512 c2 = _mm512_sub_ps(_mm512_set1_ps(cutoff2), r2);
  // c4 = (-2. * c2 + c3) * c2
  const __m512 c4 = _mm512_mul_ps(_mm512_fnmadd_ps(_mm512_set1_ps(2.f), c2,
						   _mm512_set1_ps(c3)), c2);
  // switchVal = r2 > switchOn2 ? c2*c4*c1 : 1.
  const __mmask16 switchMask = _mm512_cmplt_ps_mask(_mm512_set1_ps(switchOn2),
						    r2);
  const __m512 switchVal = _mm512_mask_mov_ps(_mm512_set1_ps(1.f), switchMask,
    _mm512_mul_ps(c2,_mm512_mul_ps(c4,_mm512_set1_ps(c1))));
  // dSwitchVal = r2 > switchOn2 ? 2.*c1*(c2*c2 - c4) : 0.
  const __m512 dSwitchVal = _mm512_mask_mov_ps(_mm512_setzero_ps(),
    switchMask, _mm512_mul_ps(_mm512_set1_ps(2.f),
    _mm512_mul_ps(_mm512_set1_ps(c1), _mm512_fmsub_ps(c2,c2,c4))));
  // r2SwitchVal = switchVal * r_2
  const __m512 r2SwitchVal = _mm512_mul_ps(switchVal, r_2);
  // vdwAgradient = (-6.*r2SwitchVal + dSwitchVal) * r_12
  const __m512 vdwAgradient = _mm512_mul_ps(_mm512_fnmadd_ps(
    _mm512_set1_ps(6.f), r2SwitchVal, dSwitchVal), r_12);
  // vdwBgradient = (-3. * r2SwitchVal + dSwitchVal) * r_6
  const __m512 vdwBgradient = _mm512_mul_ps(_mm512_fnmadd_ps(
    _mm512_set1_ps(3.f), r2SwitchVal, dSwitchVal), r_6);
  // vdwB = scaling * (A*vdwAgradient - B*vdwBgradient)
  const __m512 vdwB = _mm512_mul_ps(_mm512_set1_ps(2.f),
    _mm512_fmsub_ps(A, vdwAgradient, _mm512_mul_ps(B, vdwBgradient)));
  // ffast = kqq * (r_2*r_1+mInvCut3);
  const __m512 ffast = _mm512_mul_ps(kqq, _mm512_fmadd_ps(r_2, r_1,
    _mm512_set1_ps(mInvCut3)));

  if (doEnergy) {
    // efast = r2*mInvCut3 + cutUnder3
    __m512 efast = _mm512_fmadd_ps(r2,_mm512_set1_ps(mInvCut3),
				   _mm512_set1_ps(cutUnder3));
    // efast = efast*0.5f - r_1
    efast = _mm512_fmsub_ps(efast, _mm512_set1_ps(0.5f), r_1);
    // vdwTerm = A*r_12 - B*r_6
    const __m512 vdwTerm = _mm512_fmsub_ps(A, r_12, _mm512_mul_ps(B, r_6));
    // energyVdw = switchVal * vdwTerm + energyVdw
    energyVdw = _mm512_mask_mov_ps(energyVdw, r2mask,
      _mm512_fmadd_ps(switchVal, vdwTerm, energyVdw));
    // energyElec = -kqq*efast + energyElec
    energyElec = _mm512_mask_mov_ps(energyElec, r2mask,
      _mm512_fnmadd_ps(kqq, efast, energyElec));
  }

  // force = vdwB - ffast
  force = _mm512_sub_ps(vdwB, ffast);
  if (doSlow)
    forceEnergySlow512<doEnergy>(r2mask, kqq, slowTable, slowEtable, table_int,
				 tableDiff, rTableDiff, forceSlow, energySlow);
}

#endif // NAMD_AVXTILES
#endif // AVXTILELISTS_H
