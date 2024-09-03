#ifndef RESTRAINTSCUDA_H
#define RESTRAINTSCUDA_H

#ifdef NAMD_CUDA
#include <cuda.h>
#endif
#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif

#include <vector>
#include "PatchMap.h"
#include "AtomMap.h"
#include "Lattice.h"
#include "CudaUtils.h"
#include "CudaRecord.h"
#include "HipDefines.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#ifdef NODEGROUP_FORCE_REGISTER

class ComputeRestraintsCUDA {

public:
  ComputeRestraintsCUDA( 
    std::vector<HomePatch*> &patchList, 
    std::vector<AtomMap*> &atomMapList,
    cudaStream_t stream);

    
  void updateRestrainedAtoms(
    std::vector<AtomMap*> &atomMapsLists,
    std::vector<CudaLocalRecord> &localRecords,
    const int* h_globalToLocalID
  );
  
  ~ComputeRestraintsCUDA();
  
  void doForce(
    const Lattice *lat, 
    const bool doEnergy, 
    const bool doVirial, 
    const int timeStep, 
    double* d_pos_x,
    double* d_pos_y, 
    double* d_pos_z, 
    double* f_normal_x, 
    double* f_normal_y,
    double* f_normal_z, 
    double* d_bcEnergy, 
    double* h_bcEnergy,
    double3* d_netForce,
    double3* h_netForce,
    cudaTensor* d_virial, 
    cudaTensor* h_virial
  );

private:

  int nConstrainedAtoms;
  int  consExp;    /**<  Exponent for energy functions from simParameter? */
  bool movConsOn;  /**< Are movement constraints On? */
  bool rotConsOn;  /**< Are rotational constraints on? */
  bool selConsOn;  /**< Are selective constraints on? */
  bool spheConsOn; /**< Are spherical constraints on? */

  bool consSelectX;
  bool consSelectY;
  bool consSelectZ;
  // double consScaling;  // always read latest value from SimParameters
  double rotVel;
  
  double3 rotAxis; 
  double3 rotPivot;
  double3 moveVel;
  double3 spheConsCenter;

  cudaTensor rotationMatrix;

  // This can be only on the host, since 
  std::vector<int> h_constrainedID; /**< Contains the global IDs of the constrained atoms */
  
  std::vector<double> h_k;      /**< Contains force constant */
  std::vector<double> h_cons_x; /**< Constrained atomic positions: X */
  std::vector<double> h_cons_y; /**< Constrained atomic positions: Y */
  std::vector<double> h_cons_z; /**< Constrained atomic positions: Z */

  int* h_constrainedSOA; /**< Contains SOA index of constrained atoms*/
  int* d_constrainedSOA; // Device version of h_constrainedSOA
  int* d_constrainedID; // Device version of h_constrainedID
  double* d_cons_x; /**< Device Constrained atomic positions: X */
  double* d_cons_y; /**< Device Constrained atomic positions: Y */
  double* d_cons_z; /**< Device Constrained atomic positions: Z */
  double* d_k; /**< Device Constrained force constant */

  unsigned int* d_tbcatomic; /**< Scalar that contains the thread block counter for flagging the execution of the last threadblock*/

  cudaStream_t stream;
};

#endif // NODEGROUP_FORCE_REGISTER
#endif // NAMD_CUDA
#endif // RESTRAINTSCUDA_H

