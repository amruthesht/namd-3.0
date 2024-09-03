#ifndef SEQUENCERCUDA_H
#define SEQUENCERCUDA_H

#ifdef NAMD_CUDA
#include <curand.h>
//**< Filed to be used for external force, energy, and virial */
#endif

#ifdef NAMD_HIP
#include <hiprand/hiprand.h>
#endif

#include "NamdTypes.h"
#include "HomePatch.h"
#include "PatchTypes.h"
#include "ProcessorPrivate.h"
#include "ReductionMgr.h"
#include "SimParameters.h"
#include "SequencerCUDAKernel.h"
#include "MShakeKernel.h"
#include "ComputeRestraintsCUDA.h"
#include "ComputeSMDCUDA.h"
#include "ComputeGroupRestraintsCUDA.h"
#include "PatchData.h"
#include "Lattice.h"
#include "MigrationCUDAKernel.h"
#include "HipDefines.h"

#ifdef NODEGROUP_FORCE_REGISTER

enum ExtForceTag {
  EXT_CONSTRAINTS = 0,
  EXT_ELEC_FIELD,
  EXT_SMD,
  EXT_GROUP_RESTRAINTS,
  EXT_FORCE_TOTAL
};

class SequencerCUDA{
  friend class HomePatch;
  friend class Sequencer;
public:
  static SequencerCUDA *InstanceInit(const int, SimParameters*);
  inline static SequencerCUDA *Object() { return CkpvAccess(SequencerCUDA_instance); }

  int numPatchesCheckedIn;
  int numPatchesReady;
  std::vector<CthThread> waitingThreads;
  CthThread masterThread;  
  bool masterThreadSleeping = false;
  bool breakSuspends =  false;
  SequencerCUDA(const int, SimParameters*);
  ~SequencerCUDA();
  void initialize();
  void zeroScalars();
  bool reallocateArrays(int in_numAtomsHome, int in_numAtomsHomeAndProxy);
  void reallocateMigrationDestination();
  void deallocateArrays();
  void deallocateStaticArrays();

  void copyAoSDataToHost();
  void copyPatchDataToHost();
  void copyAtomDataToDeviceAoS();

  void copyAtomDataToDevice(bool copyForces, int maxForceNumber);

  bool copyPatchData(const bool copyIn, const bool startup);
  void copyDataToPeers(const bool copyIn);
  void migrationLocalInit();
  void migrationPerform();
  void migrationLocalPost(int startup);
  void migrationUpdateAdvancedFeatures(const int startup);
  void migrationUpdateAtomCounts();
  void migrationUpdateAtomOffsets();
  void migrationUpdateRemoteOffsets();
  void migrationUpdateProxyDestination();
  void migrationUpdateDestination();
  void migrationSortAtomsNonbonded();
  void sync();
  
  void copyMigrationInfo(HomePatch *p, int patchIndex);
  
  void assembleOrderedPatchList();
  /*! backup force and position, scale fragments with respect to their geometric center,
    and set up position for recalculating system force and energy */ 
  void monteCarloPressure_part1(Tensor &factor, Vector &origin, Lattice &oldLattice);
  /*! accumulate the force to soa and calculate external forces and energies */
  void monteCarloPressure_part2(NodeReduction *reduction, int step, int maxForceNumber,
    const bool doEnergy, const bool doVirial);
 
  /*! restore force and position to their's original values */  
  void monteCarloPressure_reject(Lattice &lattice);
  /*! calculate the new com, add half step values that was set to zero */  
  void monteCarloPressure_accept(NodeReduction *reduction, const int doMigration);

  /// Use to restore symmetry to partially computed pressure tensor.
  /// Makes assumptions as to which tensor components are calculated.
  inline static void tensor_enforce_symmetry(Tensor& t) {
    t.xy = t.yx;
    t.xz = t.zx;
    t.yz = t.zy;
  }

  void launch_part1(
    int           step,		    
    double        dt_normal,
    double        dt_nbond,
    double        dt_slow,
    double        velrescaling,
    const double  maxvel2,
    NodeReduction *reduction,
    Tensor        &factor,
    Vector        &origin,
    Lattice       &lattice,
    int           reassignVelocitiesStep,
    int           langevinPistonStep,    
    int           berendsenPressureStep,
    int           maxForceNumber,
    const int     copyIn,
    const int     savePairlists,
    const int     usePairlists, 
    const bool    doEnergy);

  void launch_part11(
    double        dt_normal,
    double        dt_nbond,
    double        dt_slow,
    double        velrescaling,
    const double  maxvel2,
    NodeReduction *reduction,
    Tensor        &factor,
    Vector        &origin,
    Lattice       &lattice, 
    int           langevinPistonStep,    
    int           maxForceNumber,
    const int     copyIn,
    const int     savePairlists,
    const int     usePairlists, 
    const bool    doEnergy);

  void launch_set_compute_positions();

  void launch_part2(
    const int     doMCPressure,
    double        dt_normal,
    double        dt_nbond,
    double        dt_slow,
    NodeReduction *reduction,
    Vector        &origin,
    int           step,
    int           maxForceNumber,
    const int     langevinPistonStep, 
    const int     copyIn,
    const int     copyOut,
    const int     doGlobal,
    const bool    doEnergy);
  
  void launch_part3(
    const int     doMCPressure,
    double        dt_normal,
    double        dt_nbond,
    double        dt_slow,
    NodeReduction *reduction,
    Vector        &origin,
    int           step,
    int           maxForceNumber,
    const int     forceRequested, 
    const int     copyIn,
    const int     copyOut,
    const bool    doEnergy);

  /*! aggregate normal forces and copy to device buffer*/
  void copyGlobalForcesToDevice();

  void copySettleParameter();
  void finish_part1(const int copyIn,
                    const int savePairlists,
                    const int usePairlists, 
                    NodeReduction* reduction);

  void update_patch_flags();
  void finish_patch_flags(int isMigration);
  void updatePairlistFlags(const int doMigration);

  cudaStream_t stream, stream2;
  PatchData *patchData;
private:
  const int  deviceID;
  bool mGpuOn;    /*<! True if nDevices > 1*/
  int  nDevices;  /*<! Total number of GPUs in the simulation -- number of MasterPEs */

  
  int  deviceIndex;
  cudaEvent_t stream2CopyDone, stream2CopyAfter;
  SimParameters *const simParams;      
  
  std::vector<AtomMap*>  atomMapList;

  // Migration Structures
  FullAtom * d_atomdata_AoS_in;
  FullAtom * d_atomdata_AoS;

  int   *d_sortIndex;
  int   *d_sortSoluteIndex;
  int4  *d_migrationDestination;

  int *d_migrationGroupSize; 
  int *d_migrationGroupIndex; 

  int   *d_idMig;
  int   *d_vdwType;

  int   *idMig;
  int   *vdwType;

  // Device arrays
#if 1
  double *d_recipMass;
#else
  float *d_recipMass; // mass should be float
#endif
  double *d_f_raw; // raw buffer for all arrays for faster setting -> broadcasting
  double *d_f_normal_x, *d_f_normal_y, *d_f_normal_z;
  double *d_f_nbond_x, *d_f_nbond_y, *d_f_nbond_z;
  double *d_f_slow_x, *d_f_slow_y, *d_f_slow_z;
  double *d_vel_x, *d_vel_y, *d_vel_z;  
  double *d_pos_raw; 
  double *d_pos_x, *d_pos_y, *d_pos_z;


  double *d_posNew_raw;
  double *d_posNew_x, *d_posNew_y, *d_posNew_z;

  double *d_f_global_x, *d_f_global_y, *d_f_global_z;
  
  double *d_rcm_x, *d_rcm_y, *d_rcm_z;
  double *d_vcm_x, *d_vcm_y, *d_vcm_z;

  // backup forces for monte carlo barostat
  double *d_f_rawMC; // raw buffer for all backup force array in MC barostat
  double *d_pos_rawMC; // raw buffer for all backup position array in MC barostat
  double *d_f_normalMC_x, *d_f_normalMC_y, *d_f_normalMC_z;
  double *d_f_nbondMC_x, *d_f_nbondMC_y, *d_f_nbondMC_z;
  double *d_f_slowMC_x, *d_f_slowMC_y, *d_f_slowMC_z;
  // backup positions for monte carlo barostat
  double *d_posMC_x, *d_posMC_y, *d_posMC_z;
  
  int *d_id; /*<! global atom index */
  int *d_idOrder; /*<! Maps global atom index to local*/
  int *d_moleculeStartIndex; /*<! starting index of each molecule */
  int *d_moleculeAtom;/*<! Atom index sorted for all molecules */

  // XXX TODO: Maybe we can get away with not allocating these arrays
  double *d_velNew_x, *d_velNew_y, *d_velNew_z;
  double *d_posSave_x, *d_posSave_y, *d_posSave_z;
  char3  *d_transform;

  int *d_patchOffsetTemp;
 
  float *d_rigidBondLength;
  int   *d_atomFixed;
  float *d_mass;
  float *d_charge;
  int   *d_partition;
  float *d_langevinParam;
  float *d_langScalVelBBK2;
  float *d_langScalRandBBK2;
  float *d_gaussrand_x, *d_gaussrand_y, *d_gaussrand_z;
  int *d_hydrogenGroupSize; 
  int *d_consFailure;
  size_t d_consFailureSize;
  int *settleList; /*<! Indexes of HGs to be calculated through SETTLE*/
  size_t settleListSize;
  CudaRattleElem* rattleList; /*<! Indexes of HGs with rigid bonds to be calculated through RATTLE*/  
  size_t rattleListSize; 
  int*     d_globalToLocalID; /*<! maps PatchID to localID inside SOA structures */
  int*     d_patchToDeviceMap; /*<! Maps a patch to a device */
  double3* d_patchCenter; /*<! patch centers on SOA datastructure */
  double3* d_awayDists; /*<! 'away' field from marginCheck */
  double3* d_patchMin; /*<! 'min' field from HomePatch */
  double3* d_patchMax; /*<! 'max' field from HomePatch */
  double*  d_patchMaxAtomMovement; /*<! maxMovement of atoms in the patch*/
  double*  d_patchNewTolerance; /*<! maxMovement of atoms in the patch*/
  unsigned int*   d_tbcatomic;

  PatchDataSOA* d_HostPatchDataSOA;

  // Host arrays
  double *recipMass;
  double *f_global_x, *f_global_y, *f_global_z;
  double *f_normal_x, *f_normal_y, *f_normal_z;
  double *f_nbond_x, *f_nbond_y, *f_nbond_z;
  double *f_slow_x, *f_slow_y, *f_slow_z;
  double *vel_x, *vel_y, *vel_z;
  double *pos_x, *pos_y, *pos_z;
  char3  *transform;
  
  float *mass;
  float *charge;
  int   *partition;
  float *langevinParam;
  float *langScalVelBBK2;
  float *langScalRandBBK2;
  //float *gaussrand_x, *gaussrand_y, *gaussrand_z;
  int   *hydrogenGroupSize; /*<! hydrogen size for each heavy atom. Hydrogens have 0*/
  float *rigidBondLength;
  int*   atomFixed;
  int*   globalToLocalID; /*<! maps PatchID to localID inside SOA structure */
  int*   patchToDeviceMap;       /*<! maps PatchID to localID inside SOA structure */
  int*   id; /*<! Global atom index */
  double3* patchCenter;
  double3* awayDists;
  double3* patchMin;
  double3* patchMax;
  double* patchMaxAtomMovement;
  double* patchNewTolerance;
  int*  computeNbondPosition;
  int*  sortOrder;
  int*  unsortOrder;
  Lattice* pairlist_lattices;
  double pairlist_newTolerance;
  Lattice myLattice;
  Lattice myLatticeOld;

  CudaMInfo *mInfo;
  
  // Host-Mapped scalars
  int* killme;
  BigReal* kineticEnergy_half;
  BigReal* intKineticEnergy_half;
  BigReal* kineticEnergy;
  BigReal* intKineticEnergy;
  BigReal* momentum_x;
  BigReal* momentum_y;
  BigReal* momentum_z;
  BigReal* angularMomentum_x;
  BigReal* angularMomentum_y;
  BigReal* angularMomentum_z;
  cudaTensor *virial;
  cudaTensor *virial_half;
  cudaTensor *intVirialNormal;
  cudaTensor *intVirialNormal_half;
  cudaTensor *rigidVirial;
  cudaTensor *intVirialNbond;
  cudaTensor *intVirialSlow;
  cudaTensor *lpVirialNormal;
  cudaTensor *lpVirialNbond;
  cudaTensor *lpVirialSlow;

  double3 *extForce;
  double  *extEnergy;
  cudaTensor *extVirial;

  unsigned int *h_marginViolations;
  unsigned int *h_periodicCellSmall;

  unsigned int totalMarginViolations;
  
  // Device scalars
  bool buildRigidLists;
  int *d_killme;
  BigReal *d_kineticEnergy;
  BigReal *d_intKineticEnergy;
  cudaTensor *d_virial;
  cudaTensor *d_intVirialNormal;
  cudaTensor *d_intVirialNbond;
  cudaTensor *d_intVirialSlow;
  cudaTensor *d_rigidVirial;
  cudaTensor *d_lpVirialNormal;
  cudaTensor *d_lpVirialNbond;
  cudaTensor *d_lpVirialSlow;
  // cudaTensor *d_extTensor;
  BigReal *d_momentum_x;
  BigReal *d_momentum_y;
  BigReal *d_momentum_z;
  BigReal *d_angularMomentum_x;
  BigReal *d_angularMomentum_y;
  BigReal *d_angularMomentum_z;
  Lattice *d_lattices;
  Lattice *d_pairlist_lattices;

  double3 *d_extForce;
  double  *d_extEnergy;
  cudaTensor *d_extVirial;
  CudaMInfo *d_mInfo;

  int *d_sortOrder;
  int *d_unsortOrder;

  int numAtomsHome;
  int numAtomsHomePrev;
  int numAtomsHomeAllocated;
  int numAtomsHomeAndProxy;
  int numAtomsHomeAndProxyAllocated;
  
  int numPatchesGlobal;
  int numPatchesHomeAndProxy;
  int numPatchesHome;

  int marginViolations;
  bool rescalePairlistTolerance;
  int nSettle, nRattle;
  BigReal maxvel2;
  SettleParameters *sp;
  CudaAtom** cudaAtomLists;

  CmiNodeLock printlock;

  cudaEvent_t eventStart, eventStop; // events for kernel timing
  float t_total;

  // Launch_pt1 timers
  float t_vverlet;
  float t_pairlistCheck;
  float t_setComputePositions;

  // Launch_pt2 timers 
  float t_accumulateForceKick;
  float t_rattle;
  float t_submitHalf;
  float t_submitReductions1;
  float t_submitReductions2;

  // True if system is periodic in all directions
  bool isPeriodic;

  std::vector<HomePatch*> patchList;
#if 1
  // TODO remove once GPU migration is merged
  std::vector<HomePatch*> patchListHomeAndProxy;
#endif

  int* consFailure;
  unsigned long long int d_ullmaxtol;
  SequencerCUDAKernel *CUDASequencerKernel;
  MigrationCUDAKernel *CUDAMigrationKernel;
  ComputeRestraintsCUDA *restraintsKernel; 
  ComputeSMDCUDA  *SMDKernel;                        /**< SMD kernel*/
  ComputeGroupRestraintsCUDA *groupRestraintsKernel; /**< Group restriants kernel*/
  curandGenerator_t curandGen;

  // alchemical used PME grids
  size_t num_used_grids;
  std::vector<int> used_grids;

#if 1
  unsigned int* deviceQueue; // pointer to work queue this device holds
  
  // if we have more than one device running this, we keep a record of all devices' data here
  double** d_peer_pos_x;
  double** d_peer_pos_y;
  double** d_peer_pos_z;
  float** d_peer_charge;
  int** d_peer_partition;
  double** d_peer_vel_x;
  double** d_peer_vel_y;
  double** d_peer_vel_z;
  double** d_peer_fb_x;
  double** d_peer_fb_y;
  double** d_peer_fb_z;
  double** d_peer_fn_x;
  double** d_peer_fn_y;
  double** d_peer_fn_z;
  double** d_peer_fs_x;
  double** d_peer_fs_y;
  double** d_peer_fs_z;
  bool**   h_patchRecordHasForces;
  bool**   d_patchRecordHasForces;
  char*    d_barrierFlag;

  // GPU Migration peer buffers

  int4** d_peer_migrationDestination;
  int** d_peer_sortSoluteIndex;

  int** d_peer_id;
  int** d_peer_vdwType;
  int** d_peer_sortOrder;
  int** d_peer_unsortOrder;
  double3** d_peer_patchCenter;

  FullAtom** d_peer_atomdata;
  CudaLocalRecord** d_peer_record;

#endif

  void maximumMove(
    const double maxvel2,
    const int    numAtoms);
  void submitHalf(
    NodeReduction *reduction,
    int numAtoms, int part,
    const bool doEnergy);
  void submitReductions(
    NodeReduction *reduction,
    BigReal origin_x,
    BigReal origin_y,
    BigReal origin_z,
    int marginViolations,
    int doEnergy, // doEnergy
    int doMomentum, 
    int numAtoms,
    int maxForceNumber);
  
  void submitReductionValues();
  void copyPositionsAndVelocitiesToHost(bool copyOut, const int doGlobal);
  void copyPositionsToHost();
  void startRun1(int maxForceNumber, const Lattice& lat);
  void startRun2(
    double dt_normal, 
    double dt_nbond, 
    double dt_slow, 
    Vector origin, 
    NodeReduction *reduction,
    int doGlobal,
    int maxForceNumber);
   void startRun3(
    double dt_normal, 
    double dt_nbond, 
    double dt_slow, 
    Vector origin, 
    NodeReduction *reduction,
    int forceRequested,
    int maxForceNumber);

  // TIP4 water model
  void redistributeTip4pForces(
    NodeReduction *reduction,
    const int maxForceNumber,
    const int doVirial);
 
  void printSOAForces();
  void printSOAPositionsAndVelocities();
  void registerSOAPointersToHost();
  void copySOAHostRegisterToDevice();
  /*! calculate external force and energy, such as efield, restraints,
    grouprestraints, and SMD */
  void calculateExternalForces(
    const int step, 
    NodeReduction *reduction,
    const int maxForceNumber,
    const int doEnergy,
    const int doVirial);

  void atomUpdatePme();
  void saveForceCUDASOA(const int maxForceNumber);

  void updateHostPatchDataSOA();
  void saveForceCUDASOA_direct(const int maxForceNumber);
  void copyPositionsToHost_direct();
};

#endif // NAMD_CUDA
#endif // SEQUENCERCUDA_H
