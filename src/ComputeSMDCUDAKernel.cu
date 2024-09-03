#ifdef NAMD_CUDA
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

#include "ComputeSMDCUDAKernel.h"
#include "ComputeCOMCudaKernel.h"
#include "HipDefines.h"

#ifdef NODEGROUP_FORCE_REGISTER


/*! Calculate SMD force and virial for large atom group (numSMDAtoms > 1024)
  Multiple thread block will be called to do this operation.
  The current COM (curCOM) must be calculated and pssed to this function. */
template<bool T_DOENERGY>
__global__ void computeSMDForceWithCOMKernel(
  const int                numSMDAtoms,
  const Lattice            lat,  
  const double             inv_group_mass,
  const double             k,
  const double             k2, 
  const double             velocity, 
  const double3            direction, 
  const int                currentTime,
  const double3            origCM, 
  const float *  __restrict mass,
  const double*  __restrict pos_x,
  const double*  __restrict pos_y, 
  const double*  __restrict pos_z, 
  const char3*   __restrict transform, 
  double*        __restrict f_normal_x, 
  double*        __restrict f_normal_y,
  double*        __restrict f_normal_z, 
  const int*     __restrict smdAtomsSOAIndex,
  cudaTensor*    __restrict d_virial, 
  const double3* __restrict curCOM,
  double*        __restrict h_extEnergy,  
  double3*       __restrict h_extForce, 
  cudaTensor*    __restrict h_extVirial,
  unsigned int*  __restrict tbcatomic)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  int totaltb = gridDim.x;
  bool isLastBlockDone = 0;

  double3 group_f = {0, 0, 0};
  double energy = 0.0;
  double3 pos = {0, 0, 0};
  double3 f = {0, 0, 0};
  cudaTensor r_virial;
  r_virial.xx = 0.0; r_virial.xy = 0.0; r_virial.xz = 0.0;
  r_virial.yx = 0.0; r_virial.yy = 0.0; r_virial.yz = 0.0;
  r_virial.zx = 0.0; r_virial.zy = 0.0; r_virial.zz = 0.0;
  int SOAindex;
  if(tid < numSMDAtoms){
    // First -> recalculate center of mass.
    // Only thread zero is doing this
    SOAindex = smdAtomsSOAIndex[tid];

    // uncoalesced memory access: too bad!
    double m = mass[SOAindex]; // Cast from float to double here
    pos.x = pos_x[SOAindex];
    pos.y = pos_y[SOAindex];
    pos.z = pos_z[SOAindex];
  
    // calculate the distance difference along direction
    double3 diffCOM;
    diffCOM.x = curCOM->x - origCM.x;
    diffCOM.y = curCOM->y - origCM.y;
    diffCOM.z = curCOM->z - origCM.z; 
    double diff = diffCOM.x*direction.x + diffCOM.y*direction.y + 
      diffCOM.z*direction.z;

    // Ok so we've calculated the new center of mass, now we can calculate the bias
    double preFactor = (velocity*currentTime - diff);
    group_f.x = k*preFactor*direction.x + k2*(diff*direction.x - diffCOM.x);
    group_f.y = k*preFactor*direction.y + k2*(diff*direction.y - diffCOM.y);
    group_f.z = k*preFactor*direction.z + k2*(diff*direction.z - diffCOM.z);

    // calculate the force on each atom
    f.x = group_f.x * m * inv_group_mass;
    f.y = group_f.y * m * inv_group_mass;
    f.z = group_f.z * m * inv_group_mass;

    // apply the bias
    f_normal_x[SOAindex] += f.x ;
    f_normal_y[SOAindex] += f.y ;
    f_normal_z[SOAindex] += f.z ;
    if(T_DOENERGY){
      // energy for restraint along the direction
      energy = 0.5*k*preFactor*preFactor; 
      // energy for transverse restraint
      energy += 0.5*k2*(diffCOM.x*diffCOM.x + diffCOM.y*diffCOM.y +
        diffCOM.z*diffCOM.z - diff*diff);
      // unwrap coordinates before calculating the virial
      char3 t = transform[SOAindex];
      pos = lat.reverse_transform(pos, t);
      r_virial.xx = f.x * pos.x;
      r_virial.xy = f.x * pos.y;
      r_virial.xz = f.x * pos.z;
      r_virial.yx = f.y * pos.x;
      r_virial.yy = f.y * pos.y;
      r_virial.yz = f.y * pos.z;
      r_virial.zx = f.z * pos.x;
      r_virial.zy = f.z * pos.y;
      r_virial.zz = f.z * pos.z;
    }
  }
  __syncthreads();

  if(T_DOENERGY){
    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    r_virial.xx = BlockReduce(temp_storage).Sum(r_virial.xx);
    __syncthreads();
    r_virial.xy = BlockReduce(temp_storage).Sum(r_virial.xy);
    __syncthreads();
    r_virial.xz = BlockReduce(temp_storage).Sum(r_virial.xz);
    __syncthreads();

    r_virial.yx = BlockReduce(temp_storage).Sum(r_virial.yx);
    __syncthreads();
    r_virial.yy = BlockReduce(temp_storage).Sum(r_virial.yy);
    __syncthreads();
    r_virial.yz = BlockReduce(temp_storage).Sum(r_virial.yz);
    __syncthreads();

    r_virial.zx = BlockReduce(temp_storage).Sum(r_virial.zx);
    __syncthreads();
    r_virial.zy = BlockReduce(temp_storage).Sum(r_virial.zy);
    __syncthreads();
    r_virial.zz = BlockReduce(temp_storage).Sum(r_virial.zz);
    __syncthreads();

    if(threadIdx.x == 0){
      atomicAdd(&(d_virial->xx), r_virial.xx);
      atomicAdd(&(d_virial->xy), r_virial.xy);
      atomicAdd(&(d_virial->xz), r_virial.xz);
      
      atomicAdd(&(d_virial->yx), r_virial.yx);
      atomicAdd(&(d_virial->yy), r_virial.yy);
      atomicAdd(&(d_virial->yz), r_virial.yz);
      
      atomicAdd(&(d_virial->zx), r_virial.zx);
      atomicAdd(&(d_virial->zy), r_virial.zy);
      atomicAdd(&(d_virial->zz), r_virial.zz);

      __threadfence();
      unsigned int value = atomicInc(&tbcatomic[0], totaltb);
      isLastBlockDone = (value == (totaltb -1));
    }

    __syncthreads();
    // Last block will set the host values
    if(isLastBlockDone){
      if(threadIdx.x == 0){
        h_extEnergy[0] = energy;
        h_extForce->x  = group_f.x;
        h_extForce->y  = group_f.y;
        h_extForce->z  = group_f.z;

        h_extVirial->xx = d_virial->xx;
        h_extVirial->xy = d_virial->xy;
        h_extVirial->xz = d_virial->xz;
        h_extVirial->yx = d_virial->yx;
        h_extVirial->yy = d_virial->yy;
        h_extVirial->yz = d_virial->yz;
        h_extVirial->zx = d_virial->zx;
        h_extVirial->zy = d_virial->zy;
        h_extVirial->zz = d_virial->zz;

        //reset the device virial value
        d_virial->xx = 0;
        d_virial->xy = 0;
        d_virial->xz = 0;
        
        d_virial->yx = 0;
        d_virial->yy = 0;
        d_virial->yz = 0;

        d_virial->zx = 0;
        d_virial->zy = 0;
        d_virial->zz = 0;
        //resets atomic counter
        tbcatomic[0] = 0;
        __threadfence();
      }
    }
  }
}


/*! Calculate SMD force, virial, and COM for small atom group (numSMDAtoms <= 1024)
  Single thread block will be called to do this operation.
  The current COM will be calculated and stored in h_curCM. */
template<bool T_DOENERGY>
__global__ void computeSMDForceKernel(
  const int                numSMDAtoms,
  const Lattice            lat,  
  const double             inv_group_mass,
  const double             k,
  const double             k2, 
  const double             velocity, 
  const double3            direction, 
  const int                currentTime,
  const double3            origCM, 
  const float * __restrict mass,
  const double* __restrict pos_x,
  const double* __restrict pos_y, 
  const double* __restrict pos_z, 
  const char3*  __restrict transform, 
  double*       __restrict f_normal_x, 
  double*       __restrict f_normal_y,
  double*       __restrict f_normal_z, 
  const int*    __restrict smdAtomsSOAIndex,
  double3*      __restrict h_curCM,
  double*       __restrict h_extEnergy, 
  double3*      __restrict h_extForce, 
  cudaTensor*   __restrict h_extVirial)
{
  __shared__ double3 group_f;
  __shared__ double energy;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double m = 0;
  double3 cm = {0, 0, 0};
  double3 pos = {0, 0, 0};
  double3 f = {0, 0, 0};
  cudaTensor r_virial;
  r_virial.xx = 0.0; r_virial.xy = 0.0; r_virial.xz = 0.0;
  r_virial.yx = 0.0; r_virial.yy = 0.0; r_virial.yz = 0.0;
  r_virial.zx = 0.0; r_virial.zy = 0.0; r_virial.zz = 0.0;
  int SOAindex;
  if(tid < numSMDAtoms){
    // First -> recalculate center of mass.
    // Only thread zero is doing this
    SOAindex = smdAtomsSOAIndex[tid];

    // uncoalesced memory access: too bad!
    m = mass[SOAindex]; // Cast from float to double here
    pos.x = pos_x[SOAindex];
    pos.y = pos_y[SOAindex];
    pos.z = pos_z[SOAindex];

    // unwrap the  coordinate to calculate COM
    char3 t = transform[SOAindex];
    pos = lat.reverse_transform(pos, t);

    cm.x = pos.x * m;
    cm.y = pos.y * m;
    cm.z = pos.z * m;
  }

  // now reduce the values and add it to thread zero
  typedef cub::BlockReduce<double, 1024> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
   
  cm.x = BlockReduce(temp_storage).Sum(cm.x);
  __syncthreads();
  cm.y = BlockReduce(temp_storage).Sum(cm.y);
  __syncthreads();
  cm.z = BlockReduce(temp_storage).Sum(cm.z);
  __syncthreads();
  
  // Calculate group force and acceleration
  if(threadIdx.x == 0){
    cm.x *= inv_group_mass; // calculates the current center of mass
    cm.y *= inv_group_mass; // calculates the current center of mass
    cm.z *= inv_group_mass; // calculates the current center of mass

    // calculate the distance difference along direction
    double3 diffCOM;
    diffCOM.x = cm.x - origCM.x;
    diffCOM.y = cm.y - origCM.y;
    diffCOM.z = cm.z - origCM.z; 
    double diff = diffCOM.x*direction.x + diffCOM.y*direction.y + 
      diffCOM.z*direction.z;

    // Ok so we've calculated the new center of mass, now we can calculate the bias
    double preFactor = (velocity*currentTime - diff);
    group_f.x = k*preFactor*direction.x + k2*(diff*direction.x - diffCOM.x);
    group_f.y = k*preFactor*direction.y + k2*(diff*direction.y - diffCOM.y);
    group_f.z = k*preFactor*direction.z + k2*(diff*direction.z - diffCOM.z);
    if(T_DOENERGY) {
      // energy for restraint along the direction
      energy = 0.5*k*preFactor*preFactor; 
      // energy for transverse restraint
      energy += 0.5*k2*(diffCOM.x*diffCOM.x + diffCOM.y*diffCOM.y +
        diffCOM.z*diffCOM.z - diff*diff);
    } 
  }
  __syncthreads();

  if(tid < numSMDAtoms){
    // calculate the force on each atom
    f.x = group_f.x * m * inv_group_mass;
    f.y = group_f.y * m * inv_group_mass;
    f.z = group_f.z * m * inv_group_mass;

    // apply the bias
    f_normal_x[SOAindex] += f.x ;
    f_normal_y[SOAindex] += f.y ;
    f_normal_z[SOAindex] += f.z ;
    if(T_DOENERGY){
      r_virial.xx = f.x * pos.x;
      r_virial.xy = f.x * pos.y;
      r_virial.xz = f.x * pos.z;
      r_virial.yx = f.y * pos.x;
      r_virial.yy = f.y * pos.y;
      r_virial.yz = f.y * pos.z;
      r_virial.zx = f.z * pos.x;
      r_virial.zy = f.z * pos.y;
      r_virial.zz = f.z * pos.z;
    }
  }
  if(T_DOENERGY){
    r_virial.xx = BlockReduce(temp_storage).Sum(r_virial.xx);
    __syncthreads();
    r_virial.xy = BlockReduce(temp_storage).Sum(r_virial.xy);
    __syncthreads();
    r_virial.xz = BlockReduce(temp_storage).Sum(r_virial.xz);
    __syncthreads();

    r_virial.yx = BlockReduce(temp_storage).Sum(r_virial.yx);
    __syncthreads();
    r_virial.yy = BlockReduce(temp_storage).Sum(r_virial.yy);
    __syncthreads();
    r_virial.yz = BlockReduce(temp_storage).Sum(r_virial.yz);
    __syncthreads();

    r_virial.zx = BlockReduce(temp_storage).Sum(r_virial.zx);
    __syncthreads();
    r_virial.zy = BlockReduce(temp_storage).Sum(r_virial.zy);
    __syncthreads();
    r_virial.zz = BlockReduce(temp_storage).Sum(r_virial.zz);
    __syncthreads();

    if(threadIdx.x == 0){
      // thread zero updates the value
      h_curCM->x = cm.x; // update current center of mass
      h_curCM->y = cm.y; // update current center of mass
      h_curCM->z = cm.z; // update current center of mass
      
      h_extEnergy[0] = energy;    // bias energy
      h_extForce->x  = group_f.x; // bias force
      h_extForce->y  = group_f.y;
      h_extForce->z  = group_f.z;

      h_extVirial->xx = r_virial.xx;
      h_extVirial->xy = r_virial.xy;
      h_extVirial->xz = r_virial.xz;
      h_extVirial->yx = r_virial.yx;
      h_extVirial->yy = r_virial.yy;
      h_extVirial->yz = r_virial.yz;
      h_extVirial->zx = r_virial.zx;
      h_extVirial->zy = r_virial.zy;
      h_extVirial->zz = r_virial.zz;
    }
  }
}

/*! Compute SMD force and virial on group of atoms */
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
  double*          h_extEnergy,  
  double3*          h_extForce, 
  cudaTensor*       h_extVirial, 
  unsigned int*     d_tbcatomic, 
  cudaStream_t      stream)
{
  if (numSMDAtoms > 1024) {
    const int blocks = 128; 
    const int grid = (numSMDAtoms + blocks - 1) / blocks;
    //first calculate the COM for SMD group and store it in h_curCM 
    computeCOMKernel<128><<<grid, blocks, 0, stream>>>(
      numSMDAtoms,
      inv_group_mass,
      lat,
      d_mass,
      d_pos_x,
      d_pos_y, 
      d_pos_z, 
      d_transform, 
      d_smdAtomsSOAIndex,
      d_curCM,
      h_curCM,
      d_tbcatomic);

      if(doEnergy){
        computeSMDForceWithCOMKernel<true><<<grid, blocks, 0, stream>>>(
          numSMDAtoms,
          lat,
          inv_group_mass,
          spring_constant,
          transverse_spring_constant, 
          velocity, 
          direction, 
          currentTime,
          origCM, 
          d_mass,
          d_pos_x,
          d_pos_y, 
          d_pos_z, 
          d_transform, 
          d_f_normal_x, 
          d_f_normal_y,
          d_f_normal_z, 
          d_smdAtomsSOAIndex,
          d_extVirial,
          h_curCM,
          h_extEnergy,  
          h_extForce, 
          h_extVirial,
          d_tbcatomic);
      } else {
        computeSMDForceWithCOMKernel<false><<<grid, blocks, 0, stream>>>(
          numSMDAtoms,
          lat,
          inv_group_mass,
          spring_constant,
          transverse_spring_constant, 
          velocity, 
          direction, 
          currentTime,
          origCM, 
          d_mass,
          d_pos_x,
          d_pos_y, 
          d_pos_z, 
          d_transform, 
          d_f_normal_x, 
          d_f_normal_y,
          d_f_normal_z, 
          d_smdAtomsSOAIndex,
          d_extVirial,
          h_curCM,
          h_extEnergy,   
          h_extForce, 
          h_extVirial,
          d_tbcatomic);
      }
  } else {
    // SMD is usually comprised of a small number of atoms. So we can get away with launching 
    //  a single threadblock
    const int blocks = 1024;
    const int grid = 1;
  
    if(doEnergy){
      computeSMDForceKernel<true><<<grid, blocks, 0, stream>>>(
        numSMDAtoms,
        lat,
        inv_group_mass,
        spring_constant,
        transverse_spring_constant, 
        velocity, 
        direction, 
        currentTime,
        origCM, 
        d_mass,
        d_pos_x,
        d_pos_y, 
        d_pos_z, 
        d_transform, 
        d_f_normal_x, 
        d_f_normal_y,
        d_f_normal_z, 
        d_smdAtomsSOAIndex,
        h_curCM,
        h_extEnergy,   
        h_extForce, 
        h_extVirial);
    }else{
      computeSMDForceKernel<false><<<grid, blocks, 0, stream>>>( 
        numSMDAtoms, 
        lat, 
        inv_group_mass, 
        spring_constant,
        transverse_spring_constant, 
        velocity, 
        direction, 
        currentTime,
        origCM, 
        d_mass, 
        d_pos_x,
        d_pos_y, 
        d_pos_z, 
        d_transform,
        d_f_normal_x, 
        d_f_normal_y,
        d_f_normal_z, 
        d_smdAtomsSOAIndex,
        h_curCM,
        h_extEnergy,  
        h_extForce, 
        h_extVirial);
    }
  }
}

#endif // NODEGROUP_FORCE_REGISTER
