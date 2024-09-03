#ifndef RESTRAINTSKERNEL_H
#define RESTRAINTSKERNEL_H

#include "Lattice.h"
#include "CudaUtils.h"
#include "CudaRecord.h"

#ifdef NODEGROUP_FORCE_REGISTER

void computeRestrainingForce(
  const int doEnergy, 
  const int doVirial, 
  const int currentTime, 
  const int nConstrainedAtoms,
  const int consExp, 
  const double consScaling, 
  const bool movConsOn, 
  const bool rotConsOn,
  const bool selConsOn,
  const bool spheConsOn, 
  const bool consSelectX,
  const bool consSelectY,
  const bool consSelectZ, 
  const double   rotVel, 
  const double3  rotAxis,
  const double3  rotPivot,
  const double3  moveVel,
  const double3  spheConsCenter,
  const int*     d_constrainedSOA, 
  const int*     d_constrainedID, 
  const double*  d_pos_x,
  const double*  d_pos_y, 
  const double*  d_pos_z,
  const double*  d_k, 
  const double*  d_cons_x,
  const double*  d_cons_y,
  const double*  d_cons_z,
  double*        f_normal_x, 
  double*        f_normal_y,
  double*        f_normal_z,
  double* d_bcEnergy,
  double* h_bcEnergy,
  double3* d_netForce, 
  double3* h_netForce, 
  const Lattice* lat, 
  cudaTensor* d_virial, 
  cudaTensor* h_virial, 
  cudaTensor rotationMatrix, 
  unsigned int* d_tbcatomic, 
  cudaStream_t stream
);

#endif // NODEGROUP_FORCE_REGISTER
#endif // RESTRAINTSKERNEL_H
