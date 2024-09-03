#ifndef CUDACOMPUTENONBONDEDINTERACTIONS_H
#define CUDACOMPUTENONBONDEDINTERACTIONS_H

#include "CudaUtils.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)

/*
 *  CudaNBConstants: defined in CudaUtils.h
 *
 *  float lj_0; // denom * cutoff2 - 3.0f * switch2 * denom
 *  float lj_1; // denom * 2.0f
 *  float lj_2; // denom * -12.0f
 *  float lj_3; // denom *  12.0f * switch2
 *  float lj_4; // cutoff2
 *  float lj_5; // switch2
 *  float e_0; // roff3Inv
 *  float e_0_slow; // roff3Inv * (1 - slowScale)
 *  float e_1; // roff2Inv
 *  float e_2; // roffInv
 *  float ewald_0; // ewaldcof
 *  float ewald_1; // pi_ewaldcof
 *  float ewald_2; // ewaldcof ^ 2
 *  float ewald_3_slow; // ewaldcof ^ 3 * slowScale
 *  float slowScale; // ratio of full electrostatics to nonbonded frequency
 */ 


/*
 * Computes the Van der Waals interaction between two particles with 
 * energy switching.
 * 
 * ljab is loaded from the vdw coef table, so this function can be used by 
 * both the nonbonded and modified exclusion kernels; 
 *
 */
template<bool doEnergy>
__device__ __forceinline__
void cudaNBForce_Vdw_EnergySwitch(const float r2, const float rinv6, const float rinv8,
  const float2 ljab, const CudaNBConstants c,
  float& f_vdw, float& energyVdw) {

  const float ab_r6 = ljab.x * rinv6 - ljab.y;
  const float w = ab_r6 * rinv6;
  const float dw_r = (ljab.x * rinv6 + ab_r6) * -6.0f * rinv8;

  float e_vdw;

  if (r2 > c.lj_5) {
    const float delta_r = (c.lj_4 - r2);
    const float s = delta_r * delta_r * (c.lj_0 + c.lj_1 * r2);
    const float ds_r = delta_r * (c.lj_3 + c.lj_2 * r2);
    f_vdw = w * ds_r + dw_r * s;
    if (doEnergy) e_vdw = w * s;
  } else {
    f_vdw = dw_r;
    if (doEnergy) e_vdw = w;
  }

  if (doEnergy) energyVdw += e_vdw;
}

/*
 * Computes the electrostatic interaction between two particles with the PME 
 * correction term and a C1 splitting function
 *
 * This will take the slowScale into account so it is only usable when
 * the fast and slow force buffers are combine
 *
 */
template<bool doEnergy>
__device__ __forceinline__
void cudaNBForce_PMESlowAndFast_C1(const float r2, const float rinv, 
  const float rinv2, const float rinv3,
  const float charge,
  const CudaNBConstants c, 
  float& f_elec, float& energyElec, float& energySlow
) {
  const float elec_fast = -1.0f * rinv3 + c.e_0;
  const float r = sqrtf(r2);
  const float elec_a = r * c.ewald_0;
  const float elec_exp = expf(-1.0f * elec_a * elec_a);
  const float elec_b = erfcf(elec_a);
  const float corr_grad = elec_b + c.ewald_1 * r * elec_exp;

  const float elec_slow = -1.0f * c.e_0 * r;
  const float scor_grad = (elec_slow - (corr_grad - 1.0f) * rinv2)*rinv;

  if (doEnergy) {
    float slow_energy = 0.5f * c.e_2 * (3.0f - r2 * c.e_1);
    float fast_energy = rinv - slow_energy;
    energyElec += charge * fast_energy;
    const float corr_energy = elec_b;
    const float scor_energy = slow_energy + (corr_energy - 1.0f) * rinv;
    energySlow += charge * scor_energy;
  }
  f_elec = charge * (elec_fast + scor_grad * c.slowScale);
}

/*
 * Computes the electrostatic interaction between two particles without the PME 
 * correction term and with a C1 splitting function
 *
 * This is used on non-PME timesteps
 *
 */
template<bool doEnergy>
__device__ __forceinline__
void cudaNBForce_PMEFast_C1(const float r2, const float rinv, 
  const float rinv2, const float rinv3,
  const float charge,
  const CudaNBConstants c,
  float& f_elec, float& energyElec
) {
  const float elec_fast = -1.0f * rinv3 + c.e_0;
  f_elec = charge * elec_fast;

  if (doEnergy) {
    float slow_energy = 0.5f * c.e_2 * (3.0f - r2 * c.e_1);
    float fast_energy = rinv - slow_energy;
    energyElec += charge * fast_energy;
  }
}

/*
 * Computes the PME correction term and with a C1 splitting function
 *
 * This is used on PME timesteps
 *
 */
template<bool doEnergy>
__device__ __forceinline__
void cudaNBForce_PMESlow_C1(const float r2, const float rinv, 
  const float rinv2, const float rinv3,
  const float charge,
  const CudaNBConstants c,
  float& fSlow, float& energySlow
) {
  fSlow = charge;

  const float r = sqrtf(r2);
  const float elec_a = r * c.ewald_0;
  // very expensive
  const float elec_b = erfcf(elec_a);
  const float elec_exp = expf(-1.0f * elec_a * elec_a);
  const float corr_grad = elec_b + c.ewald_1 * r * elec_exp;

  const float elec_slow = -1.0f * c.e_0 * r;
  const float scor_grad = (elec_slow - (corr_grad - 1.0f) * rinv2)*rinv;

  if (doEnergy) {
    float slow_energy = 0.5f * c.e_2 * (3.0f - r2 * c.e_1);
    const float corr_energy = elec_b;
    const float scor_energy = slow_energy + (corr_energy - 1.0f) * rinv;
    energySlow += fSlow * scor_energy;
  }

  fSlow *= scor_grad;
}

/*
 * Computes the modified electrostatic interaction between two particles with a C1 splitting function
 *
 * This is used by the modifiedExclusionForce kernel, it corresponds to just
 * computing the force from the "slow" table instead of the "scor" table.
 *
 */
template<bool doEnergy>
__device__ __forceinline__
void cudaModExclForce_PMESlow_C1(const float r2, const float rinv, 
  const float rinv2, const float rinv3,
  const float charge,
  const CudaNBConstants c,
  float& f_elec, float& energySlow
) {
  const float elec_slow = -1.0f * c.e_0;
  f_elec = charge * elec_slow;

  if (doEnergy) {
    float slow_energy = 0.5f * c.e_2 * (3.0f - r2 * c.e_1);
    energySlow += charge * slow_energy;
  }
}

/*
 * Computes the nonbonded interaction between two particles combining Van der Waals 
 * with energy switching and PME corrected electrostatics with a C1 splitting function
 *
 */
template<bool doEnergy, bool doSlow>
__device__ __forceinline__
void cudaNBForceMagCalc_VdwEnergySwitch_PMEC1(const float r2, const float rinv, 
  const float charge, const float2 ljab, const CudaNBConstants c, 
  float& f, float& fSlow,
  float& energyVdw, float& energyElec, float& energySlow) {

  const float rinv2 = rinv * rinv;
  const float rinv3 = rinv * rinv2;
  const float rinv6 = rinv3 * rinv3;
  const float rinv8 = rinv6 * rinv2;

  float f_vdw;
  cudaNBForce_Vdw_EnergySwitch<doEnergy>(r2, rinv6, rinv8, ljab, c, f_vdw, energyVdw);

  // Electrostatics
  float f_elec;
  if (!doSlow) {
    cudaNBForce_PMEFast_C1<doEnergy>(r2, rinv, rinv2, rinv3, charge, c, f_elec, energyElec);
  } else {
    cudaNBForce_PMEFast_C1<doEnergy>(r2, rinv, rinv2, rinv3, charge, c, f_elec, energyElec);
    cudaNBForce_PMESlow_C1<doEnergy>(r2, rinv, rinv2, rinv3, charge, c, fSlow, energySlow);
  }
  f = f_elec + f_vdw;
}

/*
 * Computes the modified exclusion force between two particles combining Van der Waals 
 * with energy switching and PME corrected electrostatics with a C1 splitting function
 *
 */
template<bool doEnergy>
__device__ __forceinline__
void cudaModExclForceMagCalc_VdwEnergySwitch_PMEC1(
  const int doSlow, const int doElec,
  const float r2, const float rinv, 
  const float charge, const float2 ljab, const CudaNBConstants c, 
  float& f, float& fSlow,
  float& energyVdw, float& energyElec, float& energySlow) {

  const float rinv2 = rinv * rinv;
  const float rinv3 = rinv * rinv2;
  const float rinv6 = rinv3 * rinv3;
  const float rinv8 = rinv6 * rinv2;

  float f_vdw;
  cudaNBForce_Vdw_EnergySwitch<doEnergy>(r2, rinv6, rinv8, ljab, c, f_vdw, energyVdw);
  // Sign corrections. The force tables are flipped and exclusions kernel uses xyz.i - xyz.j while
  // the nonbonded kernel uses xyz.j - xyz.i
  f = -1.0f * f_vdw;

  // Electrostatics
  if (doElec) {
    float f_elec;
    cudaNBForce_PMEFast_C1<doEnergy>(r2, rinv, rinv2, rinv3, charge, c, f_elec, energyElec);

    f += f_elec;
    energyElec *= -1.0f;

    if (doSlow) {
      cudaModExclForce_PMESlow_C1<doEnergy>(r2, rinv, rinv2, rinv3, charge, c, fSlow, energySlow);
      energySlow *= -1.0f;
    }
  }
}  


#endif  // NAMD_CUDA
#endif  // CUDACOMPUTENONBONDEDINTERACTIONS_H

