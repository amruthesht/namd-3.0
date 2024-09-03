#ifndef MIGRATIONCUDA_H
#define MIGRATIONCUDA_H
#include "NamdTypes.h"
#include "CudaUtils.h"
#include "CudaRecord.h"
#include "Lattice.h"
#ifdef NODEGROUP_FORCE_REGISTER

class MigrationCUDAKernel {
private:
  int* d_patchOffset_temp;
  int* d_patchOffsetNB_temp;
  int patchDeviceScan_alloc;
  char* d_patchDeviceScan_scratch;

public:
  static constexpr int kSortNumThreads = 512;
  static constexpr int kValuesPerThread = 4;
  static constexpr int kMaxAtomsPerPatch = kSortNumThreads * kValuesPerThread;

  static constexpr int kAtomsPerBuffer = 32;
  static constexpr int kNumSoABuffers = 17;

  MigrationCUDAKernel();
  ~MigrationCUDAKernel();
  void allocateScratch(const int numPatchesHomeAndProxy);
  
  void sortAtoms( 
    const int               numPatches,
    const int               numAtoms,
    const CudaLocalRecord*  records,
    const double3*          patchMin,
    const double3*          patchMax,
    const double*           pos_x,
    const double*           pos_y,
    const double*           pos_z,
    int*                    sortOrder,
    int*                    unsortOrder,
    int*                    sortIndex,
    cudaStream_t            stream
  );

  void copy_AoS_to_SoA(
    const int               numPatches,
    const bool              alchOn,
    const bool              langevinOn,
    const double            dt,
    const double            kbT,
    const double            tempFactor,
    const CudaLocalRecord*  records,
    const FullAtom*         atomdata_AoS,
    double*                 recipMass,
    double*                 vel_x,
    double*                 vel_y,
    double*                 vel_z,
    double*                 pos_x,
    double*                 pos_y,
    double*                 pos_z,
    float*                  mass,
    float*                  charge,
    int*                    id,
    int*                    vdwType,
    int*                    hydrogenGroupSize,
    int*                    migrationGroupSize,
    int*                    atomFixed,
    float*                  rigidBondLength,
    char3*                  transform,
    int*                    partition,
    float*                  langevinParam,
    float*                  langScalVelBBK2,
    float*                  langScalRandBBK2,
    cudaStream_t            stream
  );

  void sortSolventAtoms( 
    const int               numPatches,
    const CudaLocalRecord*  records,
    const FullAtom*     atomdata_AoS_in,
    FullAtom*           atomdata_AoS_out,
    int*                    sortIndex,
    cudaStream_t            stream
  );

  void computeMigrationGroupIndex( 
    const int               numPatches,
    CudaLocalRecord*        records,
    const int*              migrationGroupSize,
    int*                    migrationGroupIndex,
    cudaStream_t            stream
  );

  void transformMigratedPositions( 
    const int               numPatches,
    const CudaLocalRecord*  localRecords,
    const double3*          patchCenter,
    FullAtom*           atomdata_AoS,
    const Lattice       lattice,
    cudaStream_t            stream
  );

  void transformPositions( 
    const int               numPatches,
    const CudaLocalRecord*  localRecords,
    const double3*          patchCenter,
    FullAtom*           atomdata_AoS,
    const Lattice       lattice,
    const int*              hydrogenGroupSize,
    const int*              migrationGroupSize,
    const int*              migrationGroupIndex,
    double*                 pos_x,
    double*                 pos_y,
    double*                 pos_z,
    cudaStream_t            stream
  );

  void computeMigrationDestination(
    const int               numPatches,
    CudaLocalRecord*        localRecords,
    const Lattice       lattice,
    const CudaMInfo*        mInfo,
    const int*              patchToDeviceMap,
    const int*              globalToLocalMap,
    const double3*          patchMin,
    const double3*          patchMax,
    const int*              hydrogenGroupSize,
    const int*              migrationGroupSize,
    const int*              migrationGroupIndex,
    const double*           pos_x,
    const double*           pos_y,
    const double*           pos_z,
    int4*                   migrationDestination,
    cudaStream_t            stream
  );

  void update_AoS(
    const int               numPatches,
    const CudaLocalRecord*  records,
    FullAtom*           atomdata_AoS,
    const double*           vel_x,
    const double*           vel_y,
    const double*           vel_z,
    const double*           pos_x,
    const double*           pos_y,
    const double*           pos_z,
    cudaStream_t            stream
  );

  void performLocalMigration(
    const int               numPatches,
    CudaLocalRecord*        records,
    const FullAtom*     atomdata_AoS_in,
    FullAtom*           atomdata_AoS_out,
    int4*                   migrationDestination,
    cudaStream_t            stream
  ); 
 
  void performMigration(
    const int               numPatches,
    CudaLocalRecord*        records,
    CudaLocalRecord**       peer_records,
    const FullAtom*     local_atomdata_AoS,
    FullAtom**          peer_atomdata_AoS,
    const int*              migrationGroupSize,
    const int*              migrationGroupIndex,
    int4*                   migrationDestination,
    cudaStream_t            stream
  );

  void updateMigrationDestination(
    const int               numAtomsHome,
    int4*                   migrationDestination,
    int**                   d_peer_sortSoluteIndex,
    cudaStream_t            stream
  );

  void copyDataToProxies(
    const int               deviceIndex,
    const int               numPatchesHome,
    const int               numPatchesHomeAndProxy,
    const CudaLocalRecord*  records,
    int**                   peer_id,
    int**                   peer_vdwType,
    int**                   peer_sortOrder,
    int**                   peer_unsortOrder,
    float**                 peer_charge,
    int**                   peer_partition,
    double3**               peer_patchCenter,
    bool                    doAlch,
    cudaStream_t            stream
  );

  void copyMigrationDestinationToProxies(
    const int               deviceIndex,
    const int               numPatchesHome,
    const int               numPatchesHomeAndProxy,
    const CudaLocalRecord*  records,
    const CudaPeerRecord* peerRecords,
    int4**                  peer_migrationDestination,
    cudaStream_t            stream
  );

  void updateLocalRecords(
    const int               numPatchesHome,
    CudaLocalRecord*        records,
    CudaLocalRecord**       peer_records,
    const CudaPeerRecord* peerRecords,
    cudaStream_t            stream
  );


  void updateLocalRecordsOffset(
    const int              numPatchesHomeAndProxy,
    CudaLocalRecord*       records,
    cudaStream_t           stream
  );

  void updatePeerRecords(
    const int               numPatchesHomeAndProxy,
    CudaLocalRecord*        records,
    CudaLocalRecord**       peer_records,
    CudaPeerRecord*       peerRecords,
    cudaStream_t            stream
  );

};

#endif // NAMD_CUDA
#endif // MIGRATIONCUDA_H
