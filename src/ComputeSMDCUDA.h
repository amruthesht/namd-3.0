#ifndef SMDCUDA_H
#define SMDCUDA_H

#if defined NAMD_CUDA
#include <cuda.h>
#endif

#include <vector>
#include "PatchMap.h"
#include "AtomMap.h"
#include "Lattice.h"
#include "CudaUtils.h"
#include "HipDefines.h"

#if defined NAMD_CUDA || defined NAMD_HIP
#ifdef NODEGROUP_FORCE_REGISTER
class ComputeSMDCUDA{
  public:
    ComputeSMDCUDA(
        std::vector<HomePatch*> &patchList, 
        double springConstant, 
        double transverseSpringConstant, 
        double velocity,
        double3 direction,
        int outputFrequency,
        int firsttimestep,
        const char* filename, 
        int numAtoms );
    
    ~ComputeSMDCUDA();

    void updateAtoms(
      std::vector<AtomMap*> &atomMapsList,
      std::vector<CudaLocalRecord> &localRecords,
      const int* h_globalToLocalID);

    void doForce(
      const int timeStep, 
      const Lattice &lat, 
      const bool doEnergy, 
      const float*  d_mass, 
      const double* d_pos_x,
      const double* d_pos_y,
      const double* d_pos_z,
      const char3*  d_transform, 
      double* f_normal_x,
      double* f_normal_y,
      double* f_normal_z,
      cudaTensor* d_extVirial,
      double*     h_extEnergy,   
      double3*    h_extForce, 
      cudaTensor* h_extVirial, 
      cudaStream_t stream
    );

  private:
    double   springConstant;
    double   transverseSpringConstant;
    double   velocity;
    double   inv_group_mass; /**< invers group mass for SMD group*/
    double3  direction;
    double3  origCOM; /**<  Original center of mass of all atoms in SMD */
    double3* curCOM;  /**<  Host-mapped center of mass of all atoms in SMD */
    double3* d_curCOM; /**< Device version of current center of mass*/
    
    std::vector<HomePatch*> *patchList; /**<  Pointer to global data structure for homePatches */
    std::vector<int> smdAtomsGlobalIndex; /**< Holds the Global atom ID (PDB position) of SMD atoms */
    std::vector<int> smdAtomsSOAIndex; /**< Holds the SOA position of atoms involved in SMD */
    int* d_smdAtomsSOAIndex;
        
    // JM: SMD needs to handle IO, so I need to have a pointer to the file somehow
    int firstTimeStep;
    const char* filename;
    int outputFrequency;
    int numAtoms;
    int numSMDAtoms;

    unsigned int* d_tbcatomic; /**< Scalar that contains the thread block counter for flagging the execution of the last threadblock*/

    void parseAtoms(); /**< Function swiped from GlobalMasterSMD.C */
};
#endif // NODEGROUP_FORCE_REGISTER

#endif // NAMD_CUDA

#endif // SMDCUDA_H
