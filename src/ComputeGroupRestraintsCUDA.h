#ifndef COMPUTE_GROUP_RESTRAINTS_H
#define COMPUTE_GROUP_RESTRAINTS_H

#ifdef NAMD_CUDA
#include <cuda.h>
#endif

#include <vector>
#include "PatchMap.h"
#include "AtomMap.h"
#include "CudaUtils.h"
#include "Lattice.h"
#include "GroupRestraintsParam.h"
#include "HipDefines.h"

#if defined NAMD_CUDA || defined NAMD_HIP

#ifdef NODEGROUP_FORCE_REGISTER
/*! GroupRestraintsCUDA class
    Stores data that are required to calculate one harmonic group restraints.
    Allocate memory in host and device for one harmonic group restraints. */
class GroupRestraintsCUDA {
public:
    GroupRestraintsCUDA(const GroupRestraintParam *param);
    
    ~GroupRestraintsCUDA();

    /*! Update the global index to local position in SOA data structure  for a group */
    void updateAtoms(std::vector<AtomMap*> &atomMapsList,
                    std::vector<CudaLocalRecord> &localRecords,
                    const int *h_globalToLocalID);

    /*! Compute harmonic restriant energy, force, and virial for a group */
    void doForce(
        const int timeStep,
        const int doEnergy,
        const int doVirial,
        const int doOutput,  
        const Lattice &lat, 
        const char3*  d_transform, 
        const float*  d_mass, 
        const double* d_pos_x,
        const double* d_pos_y,
        const double* d_pos_z,
        double*       d_f_normal_x,
        double*       d_f_normal_y,
        double*       d_f_normal_z,
        cudaTensor*   d_virial,
        double*       h_extEnergy, 
        double3*      h_extForce, 
        cudaTensor*   h_extVirial, 
        cudaStream_t  stream);

  private:
    bool calcGroup1COM;       /**< To check if we need to calculate the COM for group 1 */
    bool useDistMagnitude;    /**< To check if we need to restraint using distance magnitude or vector */
    const char *groupName;    /**< Restraint group name */ 
    int      restraintExp;    /**< Restraint restraint exponent*/
    double   restraintK;      /**< Restraint restraint force constant */
    double   inv_group1_mass; /**< invers group mass for restrained atom group 1 */
    double   inv_group2_mass; /**< invers group mass for restrained atom group 2 */
    double3  resDirection;    /**< Distance component which group of atoms are restrained */
    double3  resCenterVec;    /**< Restraint center or equilibrium vector */

    double3* h_diffCOM;       /**< Host-mapped distance between COM of restrained atoms in group 1 and 2 */
    double3* h_group1COM;     /**< Host-mapped center of mass of restrained atoms in group 1 */
    double3* h_group2COM;     /**< Host-mapped center of mass of restrained atoms in group 2 */
    double3* d_group1COM;     /**< Device version of center of mass of restrained atoms in group 1 */
    double3* d_group2COM;     /**< Device version of center of mass of restrained atoms in group 2 */
    double3* h_resForce;      /**< Host-mapped Force term applied to the center of mass of the group */
    double* h_resEnergy;      /**< Host-mapped energy term from restraint */
    

    std::vector<HomePatch*> *patchList;        /**< Pointer to global data structure for homePatches */
    std::vector<int> groupAtomsSOAIndex;       /**< Stores the SOA position of atoms involved in restraining of group 1 and 2 */
    int* d_groupAtomsSOAIndex;                 /**< Device-map SOA position of atoms involved in restraining of group 1 and 2 */
    int numRestrainedGroup1;                   /**< Number of restrained atoms in the group 1 */
    int numRestrainedGroup2;                   /**< Number of restrained atoms in the group 2 */
    int totalNumRestrained;                    /**< Total number of restrained atoms in the group 1 and 2 */
    const GroupRestraintParam *resParam;       /**< Restraint parameter */

    unsigned int* d_tbcatomic; /**< Scalar that contains the thread block counter for flagging the execution of the last threadblock*/

};

/*! ComputeGroupRestraintsCUDA class
    Manages all restraints group. */
class ComputeGroupRestraintsCUDA {
public:
    
    ComputeGroupRestraintsCUDA(const int ouputFreq,
            const GroupRestraintList &resList);
    
    ~ComputeGroupRestraintsCUDA();
  
    /*! Update the global index to local position in SOA data structure for all groups */
    void updateAtoms(std::vector<AtomMap*> &atomMapsList,
                    std::vector<CudaLocalRecord> &localRecords,
                    const int *h_globalToLocalID);
   
    /*! Compute harmonic restriant energy, force, and virial for all groups */                
    void doForce(
        const int timeStep, 
        const int doEnergy,
        const int doVirial, 
        const Lattice &lat, 
        const char3*  d_transform, 
        const float*  d_mass, 
        const double* d_pos_x,
        const double* d_pos_y,
        const double* d_pos_z,
        double*       d_f_normal_x,
        double*       d_f_normal_y,
        double*       d_f_normal_z,
        cudaTensor*   d_virial,
        double*       h_extEnergy, 
        double3*      h_extForce, 
        cudaTensor*   h_extVirial, 
        cudaStream_t  stream);

private:
    int gResOutputFreq; /**< Frequency of outputing group restraints data */
    /**< vector storing cuda parameters for a restraint group */
    std::vector<GroupRestraintsCUDA*> restraintsCUDAList;
};

#endif // NODEGROUP_FORCE_REGISTER

#endif // NAMD_CUDA

#endif // COMPUTE_GROUP_RESTRAINTS_H
