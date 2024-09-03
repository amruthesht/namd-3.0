#ifndef SMDKERNEL_H
#define SMDKERNEL_H

#include "Lattice.h"
#include "CudaUtils.h"
#include "CudaRecord.h"

#ifdef NODEGROUP_FORCE_REGISTER

/*! Calculate SMD force and virial on a group of atoms.
  This kernel is optimized for all SMD group size. */
void computeSMDForce(
  const Lattice     &lat, 
  const double      inv_group_mass,
  const double      spring_constant,
  const double      transverse_spring_constant, 
  const double      velocity, 
  const double3     direction,
  const int         doEnergy, 
  const int         currentTime,
  const double3     origCM,  
  const float*      d_mass, 
  const double*     d_pos_x, 
  const double*     d_pos_y,
  const double*     d_pos_z, 
  const char3*      d_transform, 
  double *          d_f_normal_x, 
  double *          d_f_normal_y, 
  double *          d_f_normal_z, 
  const int         numSMDAtoms, 
  const int*        d_smdAtomsSOAIndex,
  double3*          d_curCM, 
  double3*          h_curCM,
  cudaTensor*       d_extVirial,
  double*           h_extEnergy,  
  double3*          h_extForce, 
  cudaTensor*       h_extVirial, 
  unsigned int*     d_tbcatomic, 
  cudaStream_t      stream);

#endif // NODEGROUP_FORCE_REGISTER
#endif // SMDKERNEL_H
