#ifndef GROUP_RESTRAINTS_2GROUP_CUDA_KERNEL_H
#define GROUP_RESTRAINTS_2GROUP_CUDA_KERNEL_H

#include "Lattice.h"
#include "CudaUtils.h"
#include "CudaRecord.h"

#ifdef NODEGROUP_FORCE_REGISTER

/*! Compute restraint force, energy, and virial 
    applied to group 2, due to restraining COM of 
    group 2 to the COM of group 1 */
void computeGroupRestraint_2Group(
    const int         useMagnitude,
    const int         doEnergy,
    const int         doVirial, 
    const int         numRestrainedGroup1,
    const int         totalNumRestrained,
    const int         restraintExp,
    const double      restraintK,
    const double3     resCenterVec,
    const double3     resDirection,
    const double      inv_group1_mass,
    const double      inv_group2_mass,
    const int*        d_groupAtomsSOAIndex,
    const Lattice     &lat,
    const char3*      d_transform,
    const float*      d_mass,
    const double*     d_pos_x,
    const double*     d_pos_y,
    const double*     d_pos_z,
    double*           d_f_normal_x,
    double*           d_f_normal_y,
    double*           d_f_normal_z,
    cudaTensor*       d_virial,
    cudaTensor*       h_extVirial,
    double*           h_resEnergy,
    double3*          h_resForce,
    double3*          h_group1COM,
    double3*          h_group2COM,
    double3*          h_diffCOM,
    double3*          d_group1COM,
    double3*          d_group2COM,
    unsigned int*     d_tbcatomic,
    cudaStream_t      stream);

#endif // NODEGROUP_FORCE_REGISTER
#endif // GROUP_RESTRAINTS_2GROUP_CUDA_KERNEL_H
