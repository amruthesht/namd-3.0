#ifndef PATCHDATA_H
#define PATCHDATA_H
#if __cplusplus < 201103L
#undef USING_ATOMICS
#else
#define USING_ATOMICS
#endif

#ifdef USING_ATOMICS
#include <atomic>
#endif
#include "ComputeCUDAMgr.h"
#include "PatchData.decl.h"
#include "NamdTypes.h"
#include "HomePatch.h"
#include "Sequencer.h"
#include "ResizeArray.h"
#include "CudaUtils.h"
#include "CudaTileListKernel.h"    // CudaPatchRecord
#include "CudaTileListKernel.hip.h"
#include "CudaRecord.h"
#include "ReductionMgr.h"
#include "Controller.h"
#include "TupleTypesCUDA.h"
#include "ComputeMgr.h"

#ifdef NAMD_CUDA
#include <cuda.h>
#endif
#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif

#include "HipDefines.h"

#ifdef NODEGROUP_FORCE_REGISTER
#ifdef NAMD_NCCL_ALLREDUCE
#ifdef NAMD_CUDA
#include "nccl.h"
#endif
#ifdef NAMD_HIP
#include "rccl.h"
#endif
#endif


class CollectionMaster;
class Output;
class IMDOutput;

// DeviceData holds the CUDAMgr pointers to force data
// These are particular to each device
struct DeviceData{
  // NONBONDED STUFF
  float4              *f_nbond;
  float4              *f_nbond_slow;
  int                  f_nbond_size;
  CudaPatchRecord     *nbond_precord;
  CudaTileListKernel  *nbond_tkernel;
  cudaStream_t         nbond_stream;
  float4* nb_datoms; //The idea is to fill this after integration
  size_t size_nb_datoms;
  int nb_precord_size;


  //BONDED STUFF
  double      *f_bond;
  double      *f_bond_nbond;
  double      *f_bond_slow;
  int          f_bond_size;
  int          bond_pr_size;
  int          bond_pi_size;
  int          forceStride;
  PatchRecord *bond_pr;
  int         *bond_pi;
  float4      *b_datoms;

  // SLOW STUFF
  CudaForce *f_slow;
  int        f_slow_size;
  int*       slow_patchPositions;
  int        slow_patchPositionsSize;
  int*       slow_pencilPatchIndex;
  int        slow_pencilPatchIndexSize;
  int*       slow_patchID;
  int        slow_patchIDSize;
  CudaAtom*  s_datoms; // slow atoms that will be built for PME
  int*       s_datoms_partition;

  bool*      h_hasPatches;
  bool*      d_hasPatches;

  int*       d_globalToLocalID;
  int*       d_patchToDeviceMap;

  std::vector<HomePatch*> patches; // Pointers to HomePatches this device owns

  // Mapping data
  std::vector<CudaLocalRecord> h_localPatches;
  CudaLocalRecord* d_localPatches;

  std::vector<CudaPeerRecord> h_peerPatches;
  CudaPeerRecord* d_peerPatches;

  int numAtomsHome;
  int numPatchesHome;
  int numPatchesHomeAndProxy;

  DeviceData();
  ~DeviceData();
};

#endif

class PatchData : public CBase_PatchData {

  public:

    PatchData();
    ~PatchData();

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
    // Pressure control
    cudaTensor *d_langevinPiston_strainRate;
    cudaTensor *d_langevinPiston_origStrainRate;
    cudaTensor *d_strainRate_old;  // for langevinPistonBarrier no
    cudaTensor *d_positionRescaleFactor;  // for langevinPistonBarrier no
#endif

    // I need "reduction-like" object here to reduce stuff on a single-node
    // avoiding reduction objects will let me avoid calling  submit() on every ULT
    NodeReduction *reduction;
    // backup reduction values for MC barostat
    NodeReduction *nodeReductionSave; 
  
    // Single node flags are stored here and copied by patches
    Flags flags;

    // Controller pointer for outputting stuff
    Controller *c_out;
    // Lattice from masterPe
    Lattice *lat;
    Tensor  *factor;
    Vector  *origin;

    // Provide global (node level) access to the collection chare
    CollectionMaster *ptrCollectionMaster;
    Output *ptrOutput;
    IMDOutput *imd;
    PDB* pdb;

    // Pointer to script
    ScriptTcl *script;

    // Flag for marking migrations
    CmiNodeLock printlock;
#ifdef NODEGROUP_FORCE_REGISTER
    // SOA register of all device SOA force arrays
    // Bonded SOA pointers
    double** h_soa_fb_x;
    double** h_soa_fb_y;
    double** h_soa_fb_z;

    // Nonbonded SOA pointers
    double** h_soa_fn_x;
    double** h_soa_fn_y;
    double** h_soa_fn_z;

    // Slow SOA pointers
    double** h_soa_fs_x;
    double** h_soa_fs_y;
    double** h_soa_fs_z;

    double** h_soa_pos_x;
    double** h_soa_pos_y;
    double** h_soa_pos_z;

    double** h_soa_vel_x; 
    double** h_soa_vel_y; 
    double** h_soa_vel_z; 

    float**   h_soa_charge;

    // Device Migration 
    int**     h_soa_sortOrder;
    int**     h_soa_unsortOrder;
    int**     h_soa_id;
    int**     h_soa_vdwType;
    double3** h_soa_patchCenter;
    int4**    h_soa_migrationDestination;
    int**    h_soa_sortSoluteIndex;

    FullAtom** h_atomdata_AoS;
    CudaLocalRecord** h_peer_record;

    int**     h_soa_partition;

    // we do need a device queue for registering work done
    // work queue for each device
    bool**         h_devHasForces;
    unsigned int** d_queues;
    unsigned int*  d_queueCounters;
    std::vector<int> migrationFlagPerDevice; // Migration flag for each device
    std::vector<int> tupleReallocationFlagPerDevice; // Migration flag for each device
    std::vector<int> atomReallocationFlagPerDevice; // Migration flag for each device

    std::atomic<int> maxNumBonds;
    std::atomic<int> maxNumAngles;
    std::atomic<int> maxNumDihedrals;
    std::atomic<int> maxNumImpropers;
    std::atomic<int> maxNumModifiedExclusions;
    std::atomic<int> maxNumExclusions;
    std::atomic<int> maxNumCrossterms;
    std::vector<int> devicePatchMapFlag; // Device Patch Map Creation flag per PE
#ifdef NAMD_NCCL_ALLREDUCE
    ncclUniqueId  ncclId;
#endif

     TupleDataStagePeer h_tupleDataStage;
     TupleIntArraysPeer h_tupleCount;
     TupleIntArraysPeer h_tupleOffset;

        /** This is a device-wise pointer to each cudaMgr. Each masterPE holds a  
     *  singular copy of this pointer, which we need to have access to for a few
     *  operations.
     *  PEs that share a device also share the manager.
     */
    ComputeBondedCUDA     **cudaBondedList;
    CudaComputeNonbonded  **cudaNonbondedList;

    std::vector<DeviceData> devData;
    ComputeMgr *master_mgr; /**< Manager object that contains masterServerObject for global master interface */
    CmiNodeLock nodeLock;   /**< Auxiliary node lock for atomic operation for GlobalMaster interface */
    std::vector<int> cbStore;
  std::atomic<int> suspendCounter;
#endif
};

#endif // PATCHDATA_H
