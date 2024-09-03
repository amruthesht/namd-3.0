#ifdef NAMD_CUDA
#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include <namd_cub/cub.cuh>
#endif
#else // NAMD_HIP
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#define cub hipcub
#endif // end NAMD_CUDA vs. NAMD_HIP

#include "HipDefines.h"

#include "ComputeRestraintsCUDAKernel.h"

#ifdef NODEGROUP_FORCE_REGISTER


#define PI	3.141592653589793


// Host function to update the rotation matrix
void vec_rotation_matrix(double angle, double3 v, cudaTensor& m){

  double mag, s, c;
  double xs, ys, zs, one_c;
  s = sin(angle * PI/180.0);
  c = cos(angle * PI/180.0);
  xs = v.x * s;
  ys = v.y * s;
  zs = v.z * s;
  one_c = 1.0 - c;

  mag = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
  
  if( mag == 0.0){
    // Return a 3x3 identity matrix
    m.xx = 1.0;
    m.xy = 0.0;
    m.xz = 0.0;
    m.yx = 0.0;
    m.yy = 1.0;
    m.yz = 0.0;
    m.zx = 0.0;
    m.zy = 0.0;
    m.zz = 1.0;
  }

  m.xx = (one_c * (v.x * v.x) ) + c;
  m.xy = (one_c * (v.x * v.y) ) - zs;
  m.xz = (one_c * (v.z * v.x) ) + ys;
  
  m.yx = (one_c * (v.x * v.y) ) + zs;
  m.yy = (one_c * (v.y * v.y) ) + c;
  m.yz = (one_c * (v.y * v.z) ) - xs;
  
  m.zx = (one_c * (v.z * v.x) ) - ys;
  m.zy = (one_c * (v.y * v.z) ) + xs;
  m.zz = (one_c * (v.z * v.z) ) + c;
}


template<bool T_DOENERGY>
__global__ void computeRestrainingForceKernel(
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
  const int*    __restrict d_constrainedSOA, 
  const int*    __restrict d_constrainedID, 
  const double* __restrict d_pos_x,
  const double* __restrict d_pos_y, 
  const double* __restrict d_pos_z,
  const double* __restrict d_k,
  const double* __restrict d_cons_x,
  const double* __restrict d_cons_y,
  const double* __restrict d_cons_z,
  double*       __restrict f_normal_x, 
  double*       __restrict f_normal_y,
  double*       __restrict f_normal_z,
  double*       __restrict d_bcEnergy,
  double*       __restrict h_bcEnergy,
  double3*      __restrict d_netForce, 
  double3*      __restrict h_netForce,
  cudaTensor*   __restrict d_virial, 
  cudaTensor*   __restrict h_virial,
  const Lattice            lat, 
  unsigned int* __restrict tbcatomic,
  cudaTensor  rotMatrix
)
{
  
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  
  int totaltb = gridDim.x;
  bool isLastBlockDone;

  if(threadIdx.x == 0){
    isLastBlockDone = 0;
  }

  __syncthreads();
  
  double energy = 0;
  double3 r_netForce = {0, 0, 0};
  cudaTensor r_virial;
  r_virial.xx = 0.0; r_virial.xy = 0.0; r_virial.xz = 0.0;
  r_virial.yx = 0.0; r_virial.yy = 0.0; r_virial.yz = 0.0;
  r_virial.zx = 0.0; r_virial.zy = 0.0; r_virial.zz = 0.0;

  if(tid < nConstrainedAtoms){

    // Index of the constrained atom in the SOA data structure
    int soaID = d_constrainedSOA[tid];

    // Atomic fixed positions
    double ref_x = d_cons_x[tid];
    double ref_y = d_cons_y[tid];
    double ref_z = d_cons_z[tid];

    // JM: BAD BAD BAD -> UNCOALESCED GLOBAL MEMORY ACCESS
    double pos_x = d_pos_x[soaID];
    double pos_y = d_pos_y[soaID];
    double pos_z = d_pos_z[soaID];

    
    double k = d_k[tid];
    k *= consScaling;
    // I can just store consScaling * k here instead of doing the math

    if(movConsOn){
      ref_x += currentTime * moveVel.x;
      ref_y += currentTime * moveVel.y;
      ref_z += currentTime * moveVel.z;
    }

    else if(rotConsOn){

      // do a matrix-vector operation

      double rx = ref_x - rotPivot.x;
      double ry = ref_y - rotPivot.y;
      double rz = ref_z - rotPivot.z;

      ref_x = rotMatrix.xx * rx + rotMatrix.xy * ry + rotMatrix.xz * rz;
      ref_y = rotMatrix.yx * rx + rotMatrix.yy * ry + rotMatrix.yz * rz;
      ref_z = rotMatrix.zx * rx + rotMatrix.zy * ry + rotMatrix.zz * rz;
    }

    // END moving and rotationg contraints

    if(spheConsOn){
      // JM: This code sucks, but maybe it's not a very common use-case, so let's go with it for now
      double3 diff;
      diff.x = ref_x - spheConsCenter.x;
      diff.y = ref_y - spheConsCenter.y;
      diff.z = ref_z - spheConsCenter.z;
      // length of refCtr
      double refRad = sqrt(diff.x * diff.x + diff.y*diff.y + diff.z * diff.z); // Whoops
      
      // Reusing diff here as relPos: first let's store global position - spherical center
      diff = lat.delta(Vector(pos_x, pos_y, pos_z), spheConsCenter);

      refRad *= rsqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z); // 2x-whoops
      // now we recalculate refPos:
      ref_x = spheConsCenter.x + diff.x * refRad;
      ref_y = spheConsCenter.y + diff.y * refRad;
      ref_z = spheConsCenter.z + diff.z * refRad;
    }

    // Calculating the RIJ vector as lattice.delta(ref, pos);
    double3 rij;

    rij = lat.delta(Vector(ref_x, ref_y, ref_z), Vector(pos_x, pos_y, pos_z));
    double3 vpos;
    vpos.x = ref_x - rij.x;
    vpos.y = ref_y - rij.y;
    vpos.z = ref_z - rij.z;

    if(selConsOn){
      rij.x *= (1.0 * consSelectX);
      rij.y *= (1.0 * consSelectY);
      rij.z *= (1.0 * consSelectZ);
    }

    double r2 = rij.x * rij.x + rij.y*rij.y + rij.z*rij.z;
    double r  = sqrt(r2); // 3x-whoops


    if (r > .0){
      double value = k * (pow(r, consExp)); // NOTE: this consExp is an int, so it might be better to just do a loop
      if (T_DOENERGY) energy = value;
      value *= consExp;
      value /= r2;
      rij.x *= value;
      rij.y *= value;
      rij.z *= value;

      // JM: BAD BAD BAD ->UNCOALESCED GLOBAL MEMORY ACCESS
      f_normal_x[soaID] += rij.x;
      f_normal_y[soaID] += rij.y;
      f_normal_z[soaID] += rij.z;
      r_netForce.x = rij.x;
      r_netForce.y = rij.y;
      r_netForce.z = rij.z;
      
      // Now we calculate the virial contribution
      // JM: is this virial symmetrical? 
      r_virial.xx = rij.x * vpos.x;
      r_virial.xy = rij.x * vpos.y;
      r_virial.xz = rij.x * vpos.z;
      r_virial.yx = rij.y * vpos.x;
      r_virial.yy = rij.y * vpos.y;
      r_virial.yz = rij.y * vpos.z;
      r_virial.zx = rij.z * vpos.x;
      r_virial.zy = rij.z * vpos.y;
      r_virial.zz = rij.z * vpos.z;
    }
  }

#if 1
  if(T_DOENERGY){
    // Reduce energy and virials
    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    energy  = BlockReduce(temp_storage).Sum(energy);
    __syncthreads();

    r_netForce.x  = BlockReduce(temp_storage).Sum(r_netForce.x);
    __syncthreads();
    r_netForce.y  = BlockReduce(temp_storage).Sum(r_netForce.y);
    __syncthreads();
    r_netForce.z  = BlockReduce(temp_storage).Sum(r_netForce.z);
    __syncthreads();
    r_virial.xx  = BlockReduce(temp_storage).Sum(r_virial.xx);
    __syncthreads();
    r_virial.xy  = BlockReduce(temp_storage).Sum(r_virial.xy);
    __syncthreads();
    r_virial.xz  = BlockReduce(temp_storage).Sum(r_virial.xz);
    __syncthreads();
    r_virial.yx  = BlockReduce(temp_storage).Sum(r_virial.yx);
    __syncthreads();
    r_virial.yy  = BlockReduce(temp_storage).Sum(r_virial.yy);
    __syncthreads();
    r_virial.yz  = BlockReduce(temp_storage).Sum(r_virial.yz);
    __syncthreads();
    r_virial.zx  = BlockReduce(temp_storage).Sum(r_virial.zx);
    __syncthreads();
    r_virial.zy  = BlockReduce(temp_storage).Sum(r_virial.zy);
    __syncthreads();
    r_virial.zz  = BlockReduce(temp_storage).Sum(r_virial.zz);
    __syncthreads();

    if(threadIdx.x == 0){
      atomicAdd(d_bcEnergy, energy);
      atomicAdd(&(d_netForce->x), r_netForce.x);
      atomicAdd(&(d_netForce->y), r_netForce.y);
      atomicAdd(&(d_netForce->z), r_netForce.z);
      
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
      unsigned int value = atomicInc(tbcatomic, totaltb);
      isLastBlockDone = (value == (totaltb -1));
    }
  }
#endif

  __syncthreads();

  if(isLastBlockDone){
    if(threadIdx.x == 0){
      //updates to host-mapped mem
      h_bcEnergy[0]  = d_bcEnergy[0];      
      h_netForce->x = d_netForce->x;
      h_netForce->y = d_netForce->y;
      h_netForce->z = d_netForce->z;
      
      h_virial->xx = d_virial->xx;
      h_virial->xy = d_virial->xy;
      h_virial->xz = d_virial->xz;
      
      h_virial->yx = d_virial->yx;
      h_virial->yy = d_virial->yy;
      h_virial->yz = d_virial->yz;
      
      h_virial->zx = d_virial->zx;
      h_virial->zy = d_virial->zy;
      h_virial->zz = d_virial->zz;
    
      d_bcEnergy[0] = 0;
      d_netForce->x = 0;
      d_netForce->y = 0;
      d_netForce->z = 0;
      
      d_virial->xx = 0;
      d_virial->xy = 0;
      d_virial->xz = 0;
      
      d_virial->yx = 0;
      d_virial->yy = 0;
      d_virial->yz = 0;

      d_virial->zx = 0;
      d_virial->zy = 0;
      d_virial->zz = 0;
      
      tbcatomic[0] = 0;
      __threadfence();
    }
  }
}

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
  double*        d_f_normal_x, 
  double*        d_f_normal_y,
  double*        d_f_normal_z,
  double*  d_bcEnergy,
  double*  h_bcEnergy,
  double3* d_netForce, 
  double3* h_netForce, 
  const Lattice* lat, 
  cudaTensor* d_virial, 
  cudaTensor* h_virial, 
  cudaTensor  rotationMatrix, 
  unsigned int* d_tbcatomic, 
  cudaStream_t stream
){

  const int blocks = 128; 
  const int grid = (nConstrainedAtoms + blocks - 1) / blocks;
  
  // we calculate the rotational matrix for this timestep on the host, hopefully this is fast enough
  vec_rotation_matrix(rotVel * currentTime, rotAxis, rotationMatrix);

  if(doEnergy || doVirial){
    computeRestrainingForceKernel<true> <<<grid, blocks, 0, stream >>>(
      currentTime, 
      nConstrainedAtoms,
      consExp, 
      consScaling, 
      movConsOn, 
      rotConsOn,
      selConsOn,
      spheConsOn, 
      consSelectX,
      consSelectY,
      consSelectZ, 
      rotVel, 
      rotAxis,
      rotPivot,
      moveVel,
      spheConsCenter,
      d_constrainedSOA, 
      d_constrainedID, 
      d_pos_x,
      d_pos_y, 
      d_pos_z,
      d_k,
      d_cons_x,
      d_cons_y,
      d_cons_z,
      d_f_normal_x, 
      d_f_normal_y,
      d_f_normal_z,
      d_bcEnergy,
      h_bcEnergy,
      d_netForce, 
      h_netForce,
      d_virial, 
      h_virial,
      *lat, 
      d_tbcatomic,
      rotationMatrix);
    
  }else {
    computeRestrainingForceKernel <false> <<<grid, blocks, 0, stream>>>( 
      currentTime, 
      nConstrainedAtoms, 
      consExp, 
      consScaling,
      movConsOn, 
      rotConsOn, 
      selConsOn, 
      spheConsOn, 
      consSelectX, 
      consSelectY, 
      consSelectZ,
      rotVel, 
      rotAxis, 
      rotPivot, 
      moveVel, 
      spheConsCenter, 
      d_constrainedSOA, 
      d_constrainedID,
      d_pos_x, 
      d_pos_y, 
      d_pos_z, 
      d_k, 
      d_cons_x, 
      d_cons_y, 
      d_cons_z, 
      d_f_normal_x, 
      d_f_normal_y, 
      d_f_normal_z, 
      d_bcEnergy, 
      h_bcEnergy, 
      d_netForce, 
      h_netForce, 
      d_virial, 
      h_virial, 
      *lat, 
      d_tbcatomic, 
      rotationMatrix);
  }
}

#endif // NODEGROUP_FORCE_REGISTER
