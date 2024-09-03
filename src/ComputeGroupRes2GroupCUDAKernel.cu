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

#include "ComputeGroupRes2GroupCUDAKernel.h"
#include "ComputeCOMCudaKernel.h"
#include "HipDefines.h"

#ifdef NODEGROUP_FORCE_REGISTER

/*! Compute restraint force, virial, and energy applied to large
    group 2 (atoms >= 1024), due to restraining COM of group 2 
    (h_group2COM) to the COM of the group 1 (h_group1COM). 
    To use this function, the COM of the group 1 and 2 
    must be calculated and passed to this function as h_group1COM 
    and h_group2COM. 
    This function also calculates the distance from COM of the
    group 1 to COM of the group 2. */
template<int T_DOENERGY, int T_DOVIRIAL, int T_USEMAGNITUDE>
__global__ void computeLargeGroupRestraint2GroupsKernel( 
    const int         numRestrainedGroup1,
    const int         totalNumRestrained,
    const int         restraintExp,
    const double      restraintK,
    const double3     resCenterVec,
    const double3     resDirection,
    const double      inv_group1_mass,
    const double      inv_group2_mass,
    const int*      __restrict    groupAtomsSOAIndex,
    const Lattice                 lat,
    const char3*    __restrict    transform,
    const float*    __restrict    mass,
    const double*   __restrict    pos_x,
    const double*   __restrict    pos_y,
    const double*   __restrict    pos_z,
    double*         __restrict    f_normal_x,
    double*         __restrict    f_normal_y,
    double*         __restrict    f_normal_z,
    cudaTensor*     __restrict    d_virial,
    cudaTensor*     __restrict    h_extVirial,
    double*         __restrict    h_resEnergy,
    double3*        __restrict    h_resForce,
    const double3*  __restrict    h_group1COM,
    const double3*  __restrict    h_group2COM,
    double3*        __restrict    h_diffCOM,
    unsigned int*   __restrict    d_tbcatomic)
{
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int totaltb = gridDim.x;
    bool isLastBlockDone = false;

    int SOAindex;
    double m = 0;
    double energy = 0.0;
    double3 diffCOM = {0, 0, 0};
    double3 group_f = {0, 0, 0};
    double3 pos = {0, 0, 0};
    double3 f = {0, 0, 0};
    cudaTensor r_virial;
    r_virial.xx = 0.0; r_virial.xy = 0.0; r_virial.xz = 0.0;
    r_virial.yx = 0.0; r_virial.yy = 0.0; r_virial.yz = 0.0;
    r_virial.zx = 0.0; r_virial.zy = 0.0; r_virial.zz = 0.0;
   
    if(tIdx < totalNumRestrained) {
        SOAindex = groupAtomsSOAIndex[tIdx];
        m = mass[SOAindex];
        // Here for consistency with ComputeGroupRes1, we calculate 
        // distance from com1 to com2 along specific restraint dimention,
        // so force is acting on group 2
        diffCOM.x = (h_group2COM->x - h_group1COM->x) * resDirection.x;
        diffCOM.y = (h_group2COM->y - h_group1COM->y) * resDirection.y;
        diffCOM.z = (h_group2COM->z - h_group1COM->z) * resDirection.z;
        // Calculate the minimum image distance
        diffCOM = lat.delta_from_diff(diffCOM);

        if (T_USEMAGNITUDE) {
            // Calculate the difference from equilibrium restraint distance
            double comVal = sqrt(diffCOM.x*diffCOM.x + diffCOM.y*diffCOM.y + diffCOM.z*diffCOM.z);
            double centerVal = sqrt(resCenterVec.x*resCenterVec.x + resCenterVec.y*resCenterVec.y +
                resCenterVec.z*resCenterVec.z);  
            
            double distDiff = (comVal - centerVal);
            double distSqDiff = distDiff * distDiff;
            double invCOMVal = 1.0 / comVal; 

            // Calculate energy and force on group of atoms
            if(distSqDiff > 0.0f) { // To avoid numerical error
                // Energy = k * (r - r_eq)^n
                energy = restraintK * distSqDiff;
                for (int n = 2; n < restraintExp; n += 2) {
                    energy *= distSqDiff;
                }
                // Force = -k * n * (r - r_eq)^(n-1)
                double force = -energy * restraintExp / distDiff;
                // calculate force along COM difference
                group_f.x = force * diffCOM.x * invCOMVal;
                group_f.y = force * diffCOM.y * invCOMVal;
                group_f.z = force * diffCOM.z * invCOMVal;
            }
        } else {
            // Calculate the difference from equilibrium restraint distance vector
            // along specific restraint dimention
            double3 resDist;
            resDist.x = (diffCOM.x - resCenterVec.x) * resDirection.x;
            resDist.y = (diffCOM.y - resCenterVec.y) * resDirection.y;
            resDist.z = (diffCOM.z - resCenterVec.z) * resDirection.z;
            // Wrap the distance difference (diffCOM - resCenterVec) 
            resDist = lat.delta_from_diff(resDist); 

            double distSqDiff = resDist.x*resDist.x + resDist.y*resDist.y + resDist.z*resDist.z;

            // Calculate energy and force on group of atoms
            if(distSqDiff > 0.0f) { // To avoid numerical error
                // Energy = k * (r - r_eq)^n
                energy = restraintK * distSqDiff;
                for (int n = 2; n < restraintExp; n += 2) {
                    energy *= distSqDiff;
                }
                // Force = -k * n * (r - r_eq)^(n-1) x (r - r_eq)/|r - r_eq| 
                double force = -energy * restraintExp / distSqDiff;
                group_f.x = force * resDist.x;
                group_f.y = force * resDist.y;
                group_f.z = force * resDist.z;
            }
        }

        // calculate the force on each atom of the group
        if (tIdx < numRestrainedGroup1) {
            // threads [0 , numGroup1Atoms) calculate force for group 1
            // We use negative because force is calculated for group 2
            f.x = -group_f.x * m * inv_group1_mass;
            f.y = -group_f.y * m * inv_group1_mass;
            f.z = -group_f.z * m * inv_group1_mass;
        } else {
            // threads [numGroup1Atoms , totalNumRestrained) calculate force for group 2
            f.x = group_f.x * m * inv_group2_mass;
            f.y = group_f.y * m * inv_group2_mass;
            f.z = group_f.z * m * inv_group2_mass;
        }
        // apply the bias to each atom in group
        f_normal_x[SOAindex] += f.x;
        f_normal_y[SOAindex] += f.y;
        f_normal_z[SOAindex] += f.z;
        // Virial is based on applied force on each atom
        if(T_DOVIRIAL) {
            // positions must be unwraped for virial calculation
            pos.x = pos_x[SOAindex];
            pos.y = pos_y[SOAindex];
            pos.z = pos_z[SOAindex];
            char3 tr = transform[SOAindex];
            pos = lat.reverse_transform(pos, tr);
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

    if(T_DOENERGY || T_DOVIRIAL) {
        if(T_DOVIRIAL) {
            // Reduce virial values in the thread block
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
        }
    
        if(threadIdx.x == 0) {
            if(T_DOVIRIAL) {
                // thread 0 adds the reduced virial values into device memory
                atomicAdd(&(d_virial->xx), r_virial.xx);
                atomicAdd(&(d_virial->xy), r_virial.xy);
                atomicAdd(&(d_virial->xz), r_virial.xz);
                
                atomicAdd(&(d_virial->yx), r_virial.yx);
                atomicAdd(&(d_virial->yy), r_virial.yy);
                atomicAdd(&(d_virial->yz), r_virial.yz);
                
                atomicAdd(&(d_virial->zx), r_virial.zx);
                atomicAdd(&(d_virial->zy), r_virial.zy);
                atomicAdd(&(d_virial->zz), r_virial.zz);
            } 
            __threadfence();
            unsigned int value = atomicInc(&d_tbcatomic[0], totaltb);
            isLastBlockDone = (value == (totaltb -1));
        }

        __syncthreads();

        if(isLastBlockDone) {
        // Thread 0 of the last block will set the host values
            if(threadIdx.x == 0) {
                if(T_DOENERGY) {
                    h_resEnergy[0] = energy;     // restraint energy for each group, needed for output
                    h_diffCOM->x = diffCOM.x;    // distance between COM of two restrained groups 
                    h_diffCOM->y = diffCOM.y;    // distance between COM of two restrained groups 
                    h_diffCOM->z = diffCOM.z;    // distance between COM of two restrained groups 
                    h_resForce->x = group_f.x;   // restraint force on group 2
                    h_resForce->y = group_f.y;   // restraint force on group 2
                    h_resForce->z = group_f.z;   // restraint force on group 2
                }
                if(T_DOVIRIAL) {
                    // Add virial values to host memory. 
                    // We use add,since we have with multiple restraints group 
                    h_extVirial->xx += d_virial->xx;
                    h_extVirial->xy += d_virial->xy;
                    h_extVirial->xz += d_virial->xz;
                    h_extVirial->yx += d_virial->yx;
                    h_extVirial->yy += d_virial->yy;
                    h_extVirial->yz += d_virial->yz;
                    h_extVirial->zx += d_virial->zx;
                    h_extVirial->zy += d_virial->zy;
                    h_extVirial->zz += d_virial->zz;

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
                }
                //resets atomic counter
                d_tbcatomic[0] = 0;
                __threadfence();
            }
        }
    }
}


/*! Compute restraint force, virial, and energy applied to small
    groups (atoms < 1024), due to restraining COM of group 2 
    (h_group2COM) to the COM of the group 1 (h_group1COM). 
    This function also calculates the distance from COM of the
    group 1 to COM of the group 2. */
template<int T_DOENERGY, int T_DOVIRIAL, int T_USEMAGNITUDE>
__global__ void computeSmallGroupRestraint2GroupsKernel(
    const int         numRestrainedGroup1,
    const int         totalNumRestrained,
    const int         restraintExp,
    const double      restraintK,
    const double3     resCenterVec,
    const double3     resDirection,
    const double      inv_group1_mass,
    const double      inv_group2_mass,
    const int*      __restrict    groupAtomsSOAIndex,
    const Lattice                 lat,
    const char3*    __restrict    transform,
    const float*    __restrict    mass,
    const double*   __restrict    pos_x,
    const double*   __restrict    pos_y,
    const double*   __restrict    pos_z,
    double*         __restrict    f_normal_x,
    double*         __restrict    f_normal_y,
    double*         __restrict    f_normal_z,
    cudaTensor*     __restrict    h_extVirial,
    double*         __restrict    h_resEnergy,
    double3*        __restrict    h_resForce,
    double3*        __restrict    h_diffCOM)
{
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double3 sh_com1;
    __shared__ double3 sh_com2;

    double m = 0;
    double energy = 0.0;
    double3 com1 = {0, 0, 0};
    double3 com2 = {0, 0, 0};
    double3 diffCOM = {0, 0, 0};
    double3 group_f = {0, 0, 0};
    double3 pos = {0, 0, 0};
    double3 f = {0, 0, 0};
    cudaTensor r_virial;
    r_virial.xx = 0.0; r_virial.xy = 0.0; r_virial.xz = 0.0;
    r_virial.yx = 0.0; r_virial.yy = 0.0; r_virial.yz = 0.0;
    r_virial.zx = 0.0; r_virial.zy = 0.0; r_virial.zz = 0.0;
    int SOAindex;

    if(tIdx < totalNumRestrained){
        // First -> recalculate center of mass.
        SOAindex = groupAtomsSOAIndex[tIdx];
    
        m = mass[SOAindex]; // Cast from float to double here
        pos.x = pos_x[SOAindex];
        pos.y = pos_y[SOAindex];
        pos.z = pos_z[SOAindex];
    
        // unwrap the  coordinate to calculate COM
        char3 tr = transform[SOAindex];
        pos = lat.reverse_transform(pos, tr);
        
        if (tIdx < numRestrainedGroup1) {
            // we initialized the com2 to zero
            com1.x = pos.x * m;
            com1.y = pos.y * m;
            com1.z = pos.z * m;
        } else {
            // we initialized the com1 to zero
            com2.x = pos.x * m;
            com2.y = pos.y * m;
            com2.z = pos.z * m;
        }
    }
      
    // reduce the (mass * position) values for group 1 and 2 in the thread block
    typedef cub::BlockReduce<double, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
        
    com1.x = BlockReduce(temp_storage).Sum(com1.x);
    __syncthreads();
    com1.y = BlockReduce(temp_storage).Sum(com1.y);
    __syncthreads();
    com1.z = BlockReduce(temp_storage).Sum(com1.z);
    __syncthreads();
    com2.x = BlockReduce(temp_storage).Sum(com2.x);
    __syncthreads();
    com2.y = BlockReduce(temp_storage).Sum(com2.y);
    __syncthreads();
    com2.z = BlockReduce(temp_storage).Sum(com2.z);
    __syncthreads();
           
    // Thread 0 calculates the COM of group 1 and 2
    if(threadIdx.x == 0){
        sh_com1.x = com1.x * inv_group1_mass; // calculates the COM of group 1
        sh_com1.y = com1.y * inv_group1_mass; // calculates the COM of group 1
        sh_com1.z = com1.z * inv_group1_mass; // calculates the COM of group 1
        sh_com2.x = com2.x * inv_group2_mass; // calculates the COM of group 2
        sh_com2.y = com2.y * inv_group2_mass; // calculates the COM of group 2
        sh_com2.z = com2.z * inv_group2_mass; // calculates the COM of group 2
    }
    __syncthreads();

    if(tIdx < totalNumRestrained) {
        // Here for consistency with distanceZ, we calculate 
        // distance from com1 to com2 along specific restraint dimention,
        // so force is acting on group 2
        diffCOM.x = (sh_com2.x - sh_com1.x) * resDirection.x;
        diffCOM.y = (sh_com2.y - sh_com1.y) * resDirection.y; 
        diffCOM.z = (sh_com2.z - sh_com1.z) * resDirection.z;
        // Calculate the minimum image distance 
        diffCOM = lat.delta_from_diff(diffCOM);
        
        if (T_USEMAGNITUDE) {
            // Calculate the difference from equilibrium restraint distance
            double comVal = sqrt(diffCOM.x*diffCOM.x + diffCOM.y*diffCOM.y + diffCOM.z*diffCOM.z);
            double centerVal = sqrt(resCenterVec.x*resCenterVec.x + resCenterVec.y*resCenterVec.y +
                resCenterVec.z*resCenterVec.z);  
            
            double distDiff = (comVal - centerVal);
            double distSqDiff = distDiff * distDiff;
            double invCOMVal = 1.0 / comVal; 

            // Calculate energy and force on group of atoms
            if(distSqDiff > 0.0f) { // To avoid numerical error
                // Energy = k * (r - r_eq)^n
                energy = restraintK * distSqDiff;
                for (int n = 2; n < restraintExp; n += 2) {
                    energy *= distSqDiff;
                }
                // Force = -k * n * (r - r_eq)^(n-1)
                double force = -energy * restraintExp / distDiff;
                // calculate force along COM difference
                group_f.x = force * diffCOM.x * invCOMVal;
                group_f.y = force * diffCOM.y * invCOMVal;
                group_f.z = force * diffCOM.z * invCOMVal;
            }
        } else {
            // Calculate the difference from equilibrium restraint distance vector
            // along specific restraint dimention
            double3 resDist;
            resDist.x = (diffCOM.x - resCenterVec.x) * resDirection.x;
            resDist.y = (diffCOM.y - resCenterVec.y) * resDirection.y;
            resDist.z = (diffCOM.z - resCenterVec.z) * resDirection.z; 
            // Wrap the distance difference (diffCOM - resCenterVec) 
            resDist = lat.delta_from_diff(resDist); 

            double distSqDiff = resDist.x*resDist.x + resDist.y*resDist.y + resDist.z*resDist.z;

            // Calculate energy and force on group of atoms
            if(distSqDiff > 0.0f) { // To avoid numerical error
                // Energy = k * (r - r_eq)^n
                energy = restraintK * distSqDiff;
                for (int n = 2; n < restraintExp; n += 2) {
                    energy *= distSqDiff;
                }
                // Force = -k * n * (r - r_eq)^(n-1) x (r - r_eq)/|r - r_eq| 
                double force = -energy * restraintExp / distSqDiff;
                group_f.x = force * resDist.x;
                group_f.y = force * resDist.y;
                group_f.z = force * resDist.z;
            }
        }
      
        // calculate the force on each atom of the group
        if (tIdx < numRestrainedGroup1) {
            // threads [0 , numGroup1Atoms) calculate force for group 1
            // We use negative because force is calculated for group 2
            f.x = -group_f.x * m * inv_group1_mass;
            f.y = -group_f.y * m * inv_group1_mass;
            f.z = -group_f.z * m * inv_group1_mass;
        } else {
            // threads [numGroup1Atoms , totalNumRestrained) calculate force for group 2
            f.x = group_f.x * m * inv_group2_mass;
            f.y = group_f.y * m * inv_group2_mass;
            f.z = group_f.z * m * inv_group2_mass;
        }

        // apply the bias to each atom in group
        f_normal_x[SOAindex] += f.x ;
        f_normal_y[SOAindex] += f.y ;
        f_normal_z[SOAindex] += f.z ;
        // Virial is based on applied force on each atom
        if(T_DOVIRIAL){
            // positions must be unwraped for virial calculation
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

    if(T_DOENERGY || T_DOVIRIAL) {
        if(T_DOVIRIAL){
            // Reduce virial values in the thread block
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
        }
      
        // thread zero updates the restraints energy and force
        if(threadIdx.x == 0){
            if(T_DOVIRIAL){
                // Add virial values to host memory. 
                // We use add,since we have with multiple restraints group 
                h_extVirial->xx += r_virial.xx;
                h_extVirial->xy += r_virial.xy;
                h_extVirial->xz += r_virial.xz;
                h_extVirial->yx += r_virial.yx;
                h_extVirial->yy += r_virial.yy;
                h_extVirial->yz += r_virial.yz;
                h_extVirial->zx += r_virial.zx;
                h_extVirial->zy += r_virial.zy;
                h_extVirial->zz += r_virial.zz;
            }
            if (T_DOENERGY) {
                h_resEnergy[0] = energy;     // restraint energy for each group, needed for output
                h_diffCOM->x = diffCOM.x;    // distance between two COM of restrained groups 
                h_diffCOM->y = diffCOM.y;    // distance between two COM of restrained groups  
                h_diffCOM->z = diffCOM.z;    // distance between two COM of restrained groups 
                h_resForce->x = group_f.x;   // restraint force on group 
                h_resForce->y = group_f.y;   // restraint force on group 
                h_resForce->z = group_f.z;   // restraint force on group 
            }
        }
    }
} 

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
    cudaStream_t      stream)
{
    int options = doEnergy + (doVirial << 1) + (useMagnitude << 2);

    if (totalNumRestrained > 1024) {
        const int blocks = 128; 
        const int grid = (totalNumRestrained + blocks - 1) / blocks;
        // first calculate the COM for restraint groups and store it in
        // h_group1COM and h_group2COM 
        compute2COMKernel<128><<<grid, blocks, 0, stream>>>(
            numRestrainedGroup1,
            totalNumRestrained,
            inv_group1_mass,
            inv_group2_mass,
            lat,
            d_mass,
            d_pos_x,
            d_pos_y, 
            d_pos_z, 
            d_transform, 
            d_groupAtomsSOAIndex,
            d_group1COM,
            d_group2COM,
            h_group1COM,
            h_group2COM,
            d_tbcatomic);
      
        #define CALL_LARGE_GROUP_RES(DOENERGY, DOVIRIAL, USEMAGNITUDE) \
        computeLargeGroupRestraint2GroupsKernel<DOENERGY, DOVIRIAL, USEMAGNITUDE>\
        <<<grid, blocks, 0, stream>>>( \
            numRestrainedGroup1, totalNumRestrained, \
            restraintExp, restraintK, resCenterVec, resDirection, \
            inv_group1_mass, inv_group2_mass, d_groupAtomsSOAIndex, \
            lat, d_transform, d_mass, d_pos_x, d_pos_y, d_pos_z, \
            d_f_normal_x, d_f_normal_y, d_f_normal_z,  d_virial, \
            h_extVirial, h_resEnergy, h_resForce, h_group1COM, \
            h_group2COM, h_diffCOM, d_tbcatomic); 
    
        switch(options) {
            case 0: CALL_LARGE_GROUP_RES(0, 0, 0); break;
            case 1: CALL_LARGE_GROUP_RES(1, 0, 0); break; 
            case 2: CALL_LARGE_GROUP_RES(0, 1, 0); break; 
            case 3: CALL_LARGE_GROUP_RES(1, 1, 0); break;  
            case 4: CALL_LARGE_GROUP_RES(0, 0, 1); break;
            case 5: CALL_LARGE_GROUP_RES(1, 0, 1); break; 
            case 6: CALL_LARGE_GROUP_RES(0, 1, 1); break; 
            case 7: CALL_LARGE_GROUP_RES(1, 1, 1); break;  
        }
    
        #undef CALL_LARGE_GROUP_RES 

    } else {
        // For small group of restrained atom, we can just launch
        //  a single threadblock
        const int blocks = 1024;
        const int grid = 1;

        #define CALL_SMALL_GROUP_RES(DOENERGY, DOVIRIAL, USEMAGNITUDE) \
        computeSmallGroupRestraint2GroupsKernel<DOENERGY, DOVIRIAL, USEMAGNITUDE>\
        <<<grid, blocks, 0, stream>>>( \
            numRestrainedGroup1, totalNumRestrained, \
            restraintExp, restraintK, resCenterVec, resDirection, \
            inv_group1_mass, inv_group2_mass, d_groupAtomsSOAIndex, \
            lat, d_transform, d_mass, d_pos_x, d_pos_y, d_pos_z, \
            d_f_normal_x, d_f_normal_y, d_f_normal_z, \
            h_extVirial, h_resEnergy, h_resForce, h_diffCOM); 
    
        switch(options) {
            case 0: CALL_SMALL_GROUP_RES(0, 0, 0); break;
            case 1: CALL_SMALL_GROUP_RES(1, 0, 0); break; 
            case 2: CALL_SMALL_GROUP_RES(0, 1, 0); break; 
            case 3: CALL_SMALL_GROUP_RES(1, 1, 0); break;  
            case 4: CALL_SMALL_GROUP_RES(0, 0, 1); break;
            case 5: CALL_SMALL_GROUP_RES(1, 0, 1); break; 
            case 6: CALL_SMALL_GROUP_RES(0, 1, 1); break; 
            case 7: CALL_SMALL_GROUP_RES(1, 1, 1); break;   
        }
    
        #undef CALL_SMALL_GROUP_RES
    } 

}

#endif // NODEGROUP_FORCE_REGISTER
