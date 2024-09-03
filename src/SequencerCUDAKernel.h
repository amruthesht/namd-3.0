#ifndef SEQUENCERCUDAKERNEL_H
#define SEQUENCERCUDAKERNEL_H

#ifdef NAMD_CUDA
#include <curand.h>
#endif

#ifdef NAMD_HIP
#include <hiprand/hiprand.h>
#endif

#include <vector>

#include "NamdTypes.h"
#include "CudaUtils.h"
#include "MShakeKernel.h"
#include "CudaTileListKernel.h" // for CudaPatchRecord
#include "CudaTileListKernel.hip.h" // for CudaPatchRecord
#include "CudaRecord.h"
#include "Lattice.h"
#include "HipDefines.h"

#ifdef NODEGROUP_FORCE_REGISTER

#define TIMEFACTOR 48.88821
#define NOT_AVAILABLE 999

#define PATCH_BLOCKS 512
#define ATOM_BLOCKS 128


class SequencerCUDAKernel{

public:
  SettleParameters *sp;
  bool firstRattleDone;
  bool intConstInit;
  int nSettle, nRattle;
 
// Size 1
  int* d_nHG;
  int* d_nSettles;
  int* d_nRattles;

  int* hgi;
  size_t hgi_size;

  // Other list sizes are bounded by the number of atoms, ok to use 32-bit int.
  // However, storage size in bytes could overflow 32-bit int, so use size_t.
  char* d_rattleList_temp_storage;
  size_t temp_storage_bytes;

  int* rattleIndexes;
  size_t rattleIndexes_size;

  SequencerCUDAKernel();
  ~SequencerCUDAKernel(); 

  void addForceToMomentum(
    const double scaling,
    double       dt_normal,
    double       dt_nbond,
    double       dt_slow,
    double       velrescaling,
    const double  *recipMass,
    const double *f_normal_x,
    const double *f_normal_y,
    const double *f_normal_z,
    const double *f_nbond_x, 
    const double *f_nbond_y,
    const double *f_nbond_z,
    const double *f_slow_x,  
    const double *f_slow_y,
    const double *f_slow_z,
    double       *vel_x,
    double       *vel_y,
    double       *vel_z,
    int          numAtoms,
    int          maxForceNumber,
    cudaStream_t stream);
  void maximumMove(
    const double maxvel2,
    const double *vel_x,
    const double *vel_y,
    const double *vel_z,
    int          *killme,
    const int    numAtoms,
    cudaStream_t stream);
  void addVelocityToPosition(
    const double   dt,
    const double *vel_x,
    const double *vel_y,
    const double *vel_z,
    double *pos_x,
    double *pos_y,
    double *pos_z,
    double *h_pos_x,
    double *h_pos_y,
    double *h_pos_z, 
    int            numAtoms,
    bool copyPositions, 
    cudaStream_t   stream);
  void velocityVerlet1(
    const int    step, 
    const double scaling, 
    const double dt_normal, 
    const double dt_nbond, 
    const double dt_slow,
    const double velrescaling,
    const double* recipMass,  
    double*  vel_x,
    double*  vel_y, 
    double*  vel_z, 
    const double maxvel2,
    int* h_killme,
    double* pos_x,
    double* pos_y, 
    double* pos_z,
    double* h_pos_x, 
    double* h_pos_y,
    double* h_pos_z, 
    double* f_normal_x, 
    double* f_normal_y, 
    double* f_normal_z,
    double* f_nbond_x, 
    double* f_nbond_y, 
    double* f_nbond_z,
    double* f_slow_x, 
    double* f_slow_y, 
    double* f_slow_z,  
    const int numAtoms, 
    const int maxForceNumber, 
    cudaStream_t stream);


  void centerOfMass(
    const double *coor_x, 
    const double *coor_y,
    const double *coor_z, 
    double *cm_x,
    double *cm_y,
    double *cm_z, 
    const float* mass, 
    const int* hydrogenGroupSize, 
    const int numAtoms, 
    cudaStream_t stream
  );

  void updateRigidArrays(
    const double  dt,
    const double *vel_x,
    const double *vel_y,
    const double *vel_z,
    const double *pos_x,
    const double *pos_y,
    const double *pos_z,
    double* velNew_x, 
    double* velNew_y,
    double* velNew_z,
    double* posNew_x,
    double* posNew_y,
    double* posNew_z,
    int     numAtoms,
    cudaStream_t stream);
  void submitHalf(
    const double *vel_x,
    const double *vel_y,
    const double *vel_z,
    const double *vcm_x,
    const double *vcm_y,
    const double *vcm_z, 
    const float  *mass,
    BigReal *kineticEnergy,
    BigReal *intKineticEnergy,
    cudaTensor *virial,
    cudaTensor *intVirialNormal,
    BigReal *h_kineticEnergy,
    BigReal *h_intKineticEnergy,
    cudaTensor *h_virial,
    cudaTensor *h_intVirialNormal,
    int *hydrogenGroupSize,
    int numAtoms,
    unsigned int* tbcatomic,
    cudaStream_t stream);

  void scaleCoordinateWithFactor(
    double *pos_x,
    double *pos_y,
    double *pos_z,
    float *mass,
    int *hydrogenGroupSize,
    cudaTensor factor,
    cudaVector origin,
    int useGroupPressure,
    int numAtoms,
    cudaStream_t stream);

  // Maps the global atom index to local
  void SetAtomIndexOrder(
    int *id, 
    int *idOrder,
    int numAtoms,
    cudaStream_t stream);
  
  // scale the coordinate using Molecule's geometric center
  void scaleCoordinateUsingGC(
    double *pos_x,
    double *pos_y,
    double *pos_z,
    const int *idOrder,
    const int *moleculeStartIndex,
    const int *moleculeAtom,
    const cudaTensor factor,
    const cudaVector origin,
    const Lattice oldLattice,
    const Lattice newLattice,
    const char3 *transform,
    const int numMolecules,
    const int numLargeMolecules,
    cudaStream_t stream);

  void langevinPiston(
    double *pos_x,
    double *pos_y,
    double *pos_z,
    double *vel_x,
    double *vel_y,
    double *vel_z,
    float *mass,
    int *hydrogenGroupSize,
    cudaTensor factor,
    cudaVector origin,
    double velFactor_x,
    double velFactor_y,
    double velFactor_z,
    int useGroupPressure,
    int numAtoms,
    cudaStream_t stream);
  void submitReduction1(
    double *pos_x,
    double *pos_y,
    double *pos_z,
    double *vel_x,
    double *vel_y,
    double *vel_z,
    float  *mass,
    BigReal *kineticEnergy,
    BigReal *momentum_x,
    BigReal *momentum_y,
    BigReal *momentum_z,
    BigReal *angularMomentum_x,
    BigReal *angularMomentum_y,
    BigReal *angularMomentum_z,
    BigReal origin_x,
    BigReal origin_y,
    BigReal origin_z,
    BigReal *h_kineticEnergy,
    BigReal *h_momentum_x,
    BigReal *h_momentum_y,
    BigReal *h_momentum_z,
    BigReal *h_angularMomentum_x, 
    BigReal *h_angularMomentum_y,
    BigReal *h_angularMomentum_z,
    unsigned int* tbcatomic, 
    int numAtoms,
    cudaStream_t stream);
  void submitReduction2(
    const double *pos_x,
    const double *pos_y,
    const double *pos_z,
    const double *vel_x,
    const double *vel_y,
    const double *vel_z,
    const double *rcm_x,
    const double *rcm_y,
    const double *rcm_z,
    const double *vcm_x,
    const double *vcm_y,
    const double *vcm_z, 
    const double *f_normal_x,
    const double *f_normal_y,
    const double *f_normal_z,
    const double *f_nbond_x, 
    const double *f_nbond_y,
    const double *f_nbond_z,
    const double *f_slow_x,  
    const double *f_slow_y,
    const double *f_slow_z,  
    float  *mass,
    int *hydrogenGroupSize,
    BigReal *kineticEnergy, 
    BigReal *h_kineticEnergy, 
    BigReal *intKineticEnergy,
    BigReal *h_intKineticEnergy,
    cudaTensor *intVirialNormal,
    cudaTensor *intVirialNbond,
    cudaTensor *intVirialSlow,
    cudaTensor *h_intVirialNormal,
    cudaTensor *h_intVirialNbond, 
    cudaTensor *h_intVirialSlow,
    cudaTensor *rigidVirial, 
    cudaTensor *h_rigidVirial, 
    unsigned int* tbcatomic, 
    int numAtoms,
    int maxForceNumber,
    cudaStream_t stream);
  void langevinVelocitiesBBK1(
    BigReal timestep,
    const float *langevinParam,
    double      *vel_x,
    double      *vel_y,
    double      *vel_z,
    int numAtoms,
    cudaStream_t stream);
  void langevinVelocitiesBBK2(
    BigReal timestep,
    const float *langScalVelBBK2,
    const float *langScalRandBBK2,
    float *gaussrand_x,
    float *gaussrand_y,
    float *gaussrand_z,
    double *vel_x,
    double *vel_y,
    double *vel_z,
    const int numAtoms,
    const int numAtomsGlobal, 
    const int stride, 
    curandGenerator_t gen,
    cudaStream_t stream);

  void reassignVelocities(
    const BigReal timestep,
    float *gaussrand_x,
    float *gaussrand_y,
    float *gaussrand_z,
    double *vel_x,
    double *vel_y,
    double *vel_z,
    const double *d_recipMass,
    const BigReal kbT,
    const int numAtoms,
    const int numAtomsGlobal, 
    const int stride, 
    curandGenerator_t gen,
    cudaStream_t stream);

  void rattle1(const bool doEnergy, 
    const bool pressure, 
    const int numAtoms, 
    const double dt, 
    const double invdt,
    const double tol2, 
    double *vel_x,
    double *vel_y,
    double *vel_z,
    double *pos_x,
    double *pos_y,
    double *pos_z,
    double *velNew_x,
    double *velNew_y,
    double *velNew_z,
    double *posNew_x, 
    double *posNew_y,
    double *posNew_z,
    double *f_normal_x,
    double *f_normal_y,
    double *f_normal_z,
    const int   *hydrogenGroupSize,
    const float *rigidBondLength,
    const float *mass, 
    const int   *atomFixed,
    int **settleList, 
    size_t& settleListSize, 
    int **consFailure, 
    size_t& consFailureSize, 
    CudaRattleElem **rattleList,
    size_t& rattleListSize,
    int *nSettle,
    int *nRattle,
    cudaTensor *virial,
    cudaTensor *h_virial,
    unsigned int* tbcatomic, 
    int migration,
    SettleParameters *sp, 
    bool first,
    int* h_consFailure,
    const WaterModel water_model,
    cudaStream_t stream);
    
    void copy_nbond_forces(int numPatches, float4 *f_nbond,
      float4*          f_nbond_slow, 
      double*                f_nbond_x, 
      double*                f_nbond_y,
      double*                f_nbond_z,
      double*                f_slow_x, 
      double*                f_slow_y,
      double*                f_slow_z,
      const int*             patchIDS,
      const int*             patchoffsets,
      const int*             patchUnsortOrder,
      const CudaPatchRecord* nbondIndexPerPatch,
      const bool doSlow, 
      cudaStream_t stream);
    
    void copy_bond_forces(int numPatches,
        double *f_bond,
        double *f_bond_nbond,
        double *f_bond_slow,
        double *f_bond_x,
        double *f_bond_y, 
        double *f_bond_z, 
        double *f_nbond_x, 
        double *f_nbond_y,
        double *f_nbond_z, 
        double *f_slow_x, 
        double *f_slow_y,
        double *f_slow_z,
        int forceStride, //if stridedForces
        PatchRecord *pr, 
        const int *patchIDs, 
        const int *patchOffsets,
        bool doNbond, 
        bool doSlow, 
        cudaStream_t stream);
        
   void copy_slow_forces(int numPatches,
      const CudaForce* f_slow, 
      double* f_slow_x, 
      double* f_slow_y, 
      double* f_slow_z, 
      const int* d_patchOffsets,
      const Lattice *lattices,
      cudaStream_t stream);

   void accumulateForceToSOA(
      const int               doGlobal,			     
      const int               maxForceNumber,
      const int               numPatches,
      const int               nDevices,
      CudaLocalRecord*        localRecords,
      const double*           f_bond,
      const double*           f_bond_nbond,
      const double*           f_bond_slow,
      int                     forceStride,
      const float4*           f_nbond,
      const float4*           f_nbond_slow,
      const CudaForce*        f_slow,
      double*                 d_f_global_x,
      double*                 d_f_global_y,
      double*                 d_f_global_z,
      double*                 d_f_normal_x,
      double*                 d_f_normal_y,
      double*                 d_f_normal_z,
      double*                 d_f_nbond_x,
      double*                 d_f_nbond_y,
      double*                 d_f_nbond_z,
      double*                 d_f_slow_x,
      double*                 d_f_slow_y,
      double*                 d_f_slow_z,
      const int*              patchUnsortOrder,
      const Lattice           lattice,
      unsigned int**          deviceQueues,
      unsigned int*           queueCounters,
      unsigned int*           tbcatomic, 
      cudaStream_t            stream);
 
  void accumulate_force_kick(
      const int               doGlobal,			     
      int                     maxForceNumber,
      const int               numPatches,
      CudaLocalRecord*        localRecords,
      const double*           f_bond,
      const double*           f_bond_nbond,
      const double*           f_bond_slow,
      int                     forceStride,
      const float4*           f_nbond,
      const float4*           f_nbond_slow,
      const CudaForce*        f_slow,
      double*                 d_f_global_x,
      double*                 d_f_global_y,
      double*                 d_f_global_z,
      double*                 d_f_normal_x,
      double*                 d_f_normal_y,
      double*                 d_f_normal_z,
      double*                 d_f_nbond_x,
      double*                 d_f_nbond_y,
      double*                 d_f_nbond_z,
      double*                 d_f_slow_x,
      double*                 d_f_slow_y,
      double*                 d_f_slow_z,
      double*                 d_vel_x,
      double*                 d_vel_y, 
      double*                 d_vel_z,
      const double*           recipMass, 
      const double            dt_normal, 
      const double            dt_nbond,
      const double            dt_slow,
      const double            scaling,
      const int*              patchUnsortOrder,
      const Lattice           lattice,
      cudaStream_t            stream);

   void set_compute_positions(
     const int devID, 
     const bool isPmeDevice, 
     const int nDev, 
     const int numPatchesHomeAndProxy, 
     const int numPatchesHome, 
     const bool doNbond,
     const bool doSlow, 
     const bool doFEP,
     const bool doTI,
     const bool doAlchDecouple,
     const bool doAlchSoftCore,
     const double* pos_x,
     const double* pos_y, 
     const double* pos_z, 
#ifndef NAMD_NCCL_ALLREDUCE
     double**      peer_pos_x, 
     double**      peer_pos_y, 
     double**      peer_pos_z, 
     float**       d_peer_charge, 
     int**         d_peer_partition,
#endif
     const float* charges,
     const int* partition,
     const double  charge_scaling,
     const double3* patchCenter, 
     const int* s_patchPositions, 
     const int* s_pencilPatchIndex,
     const int* s_patchIDs, 
     const int* patchSortOrder,
     const Lattice lattice, 
     float4*  nb_atoms,
     float4*  b_atoms,
     float4*  s_atoms,
     int* s_partition,
     int numTotalAtoms,
     CudaLocalRecord*                  localRecords,
     CudaPeerRecord*                   peerRecords,
     std::vector<int>& atomCounts,
     cudaStream_t stream);

  void set_pme_positions(
    const int  devID,
    const bool isPmeDevice, 
    const int  nDev, 
    const int  numPatchesHomeAndProxy,
    const int  numPatchesHome,
    const bool doNbond, 
    const bool doSlow,
    const bool doFEP,
    const bool doTI,
    const bool doAlchDecouple,
    const bool doAlchSoftCore,
    const double* d_pos_x,
    const double* d_pos_y,
    const double* d_pos_z,
  #ifndef NAMD_NCCL_ALLREDUCE
    double**      d_peer_pos_x, 
    double**      d_peer_pos_y, 
    double**      d_peer_pos_z, 
    float**       d_peer_charge, 
    int**         d_peer_partition,
  #endif
    const float* charges,
    const int* partition,
    const double charge_scaling, 
    const double3* patchCenter, 
    const int* s_patchPositions, 
    const int* s_pencilPatchIndex, 
    const int* s_patchIDs, 
    const int* patchSortOrder, 
    const Lattice lattice,
    float4* nb_atoms,
    float4* b_atoms,
    float4* s_atoms,
    int* s_partition,
    int numTotalAtoms,
    CudaLocalRecord*                  localRecords,
    CudaPeerRecord*                   peerRecords,
    std::vector<int>& atomCounts,
    cudaStream_t stream);
   
  void PairListMarginCheck(const int numPatches,
    CudaLocalRecord*           localRecords,
    const double*              pos_x,
    const double*              pos_y,
    const double*              pos_z,
    const double*              pos_old_x,
    const double*              pos_old_y,
    const double*              pos_old_z,
    const double3*             awayDists, // for margin check
    const Lattice              lattice, 
    const Lattice              lattice_old,
    const double3*             patchMins, 
    const double3*             patchMaxes,
    const double3*             patchCenter, 
    const CudaMInfo*           mInfo, 
    unsigned int*              tbcatomic,
    const double               pairlistTrigger,
    const double               pairlistGrow, 
    const double               pairlistShrink, 
    double*                    patchMaxAtomMovement,
    double*                    h_patchMaxAtomMovement, 
    double*                    patchNewTolerance,
    double*                    h_patchNewTolerance,
    const double               minSize, 
    const double               cutoff,
    const double               sysdima, 
    const double               sysdimb,
    const double               sysdimc,
    unsigned int*              h_marginViolations,
    unsigned int*              h_periodicCellSmall,
    const bool                 rescalePairlistTolerance,
    const bool                 isPeriodic, 
    cudaStream_t               stream);
     
   void apply_Efield(
     const int         numAtoms, 
     const bool        normalized, 
     const bool        doEnergy, 
     const double3     eField,
     const double      eFieldOmega, 
     const double      eFieldPhi, 
     const double      t, 
     const Lattice     lat, 
     const char3*      transform, 
     const float*      charges,
     const double*     pos_x, 
     const double*     pos_y, 
     const double*     pos_z, 
     double*           f_normal_x, 
     double*           f_normal_y,
     double*           f_normal_z,
     double3*          d_extForce, 
     cudaTensor*       d_extVirial, 
     double*           d_extEnergy,
     double3*          h_extForce, 
     cudaTensor*       h_extVirial, 
     double*           h_extEnergy, 
     unsigned int*     tbcatomic, 
     cudaStream_t      stream
   );

   void mergeForcesFromPeers(
     const int              devID, 
     const int              maxForceNumber, 
     const Lattice          lat, 
     const int              numPatchesHomeAndProxy,
     const int              numPatchesHome,
     // ------- Force buffers to be merged - ----- //  
     double**               f_normal_x, 
     double**               f_normal_y, 
     double**               f_normal_z, 
     double**               f_nbond_x, 
     double**               f_nbond_y, 
     double**               f_nbond_z, 
     double**               f_slow_x, 
     double**               f_slow_y, 
     double**               f_slow_z, 
     const CudaForce*       pmeForces, 
     CudaLocalRecord*       localRecords,
     CudaPeerRecord*        peerRecords,
     std::vector<int>&      atomCounts,
     cudaStream_t           stream
   );

   void copyForcesToHostSOA(
     const int               numPatches,
     CudaLocalRecord*        localRecords,
     const int               maxForceNumber, 
     const double*           d_f_normal_x,
     const double*           d_f_normal_y,
     const double*           d_f_normal_z,
     const double*           d_f_nbond_x,
     const double*           d_f_nbond_y,
     const double*           d_f_nbond_z,
     const double*           d_f_slow_x,
     const double*           d_f_slow_y,
     const double*           d_f_slow_z,
     PatchDataSOA*           d_HostPatchDataSOA,
     cudaStream_t            stream
   );

   void copyPositionsToHostSOA(
     const int               numPatches,
     CudaLocalRecord*        localRecords,
     const double*           pos_x, 
     const double*           pos_y, 
     const double*           pos_z, 
     PatchDataSOA*           d_HostPatchDataSOA,
     cudaStream_t            stream
   );

  void redistributeTip4pForces(
    double*        d_f_normal_x,
    double*        d_f_normal_y,
    double*        d_f_normal_z,
    double*        d_f_nbond_x,
    double*        d_f_nbond_y,
    double*        d_f_nbond_z,
    double*        d_f_slow_x,
    double*        d_f_slow_y,
    double*        d_f_slow_z,
    cudaTensor*    d_virial_normal,
    cudaTensor*    d_virial_nbond,
    cudaTensor*    d_virial_slow,
    const double*  d_pos_x,
    const double*  d_pos_y,
    const double*  d_pos_z,
    const float*   d_mass,
    const int      numAtoms,
    const int      doVirial,
    const int      maxForceNumber,
    cudaStream_t   stream
  );
};

#endif // NODEGROUP_FORCE_REGISTER
#endif // SEQUENCERCUDAKERNEL_H
