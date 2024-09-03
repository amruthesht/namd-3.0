#ifndef MIGRATIONBONDEDCUDA_H
#define MIGRATIONBONDEDCUDA_H
#include "CudaUtils.h"
#include "CudaRecord.h"
#include "TupleTypesCUDA.h"
#include "Lattice.h"
#ifdef NODEGROUP_FORCE_REGISTER

/**
 * \brief Wrapper class for tuple migration kernels
 *
 */
class MigrationBondedCUDAKernel {
private:

  //
  // Tuple level structures
  // 
  size_t srcTotalAlloc;
  TupleSizes srcAlloc;
  TupleSizes dstAlloc;

  TupleDataStage dataSrc;
  TupleDataStage dataDst;
  TupleDataStagePeer d_peer_data;

  TupleIntArraysContiguous d_device;
  TupleIntArraysContiguous d_downstreamPatchIndex;
  TupleIntArraysContiguous d_dstIndex;

  //
  // Patch level data structures
  //
  int numPatchesHomePad;
  TupleIntArraysContiguous d_counts;
  TupleIntArraysContiguous d_offsets;
  TupleIntArraysPeer d_peer_counts;
  TupleIntArraysPeer d_peer_offsets;

  //
  // Used to copy tuple counts back to device with single copy
  //
  TupleCounts* d_totalCount;
  TupleCounts* h_totalCount;

  //
  // Scratch space used by CUB 
  //
  size_t patchDeviceScan_alloc;
  char* d_patchDeviceScan_scratch;

public:
  static constexpr int kNumThreads = 128;
  static constexpr int kPatchNumberPad = 64; // Pad to 256 bytes. Is this overkill?
  static constexpr int kNumTuplesTypes = 7;
  
  // TODO Tune these
  static constexpr int kPatchThreads = 1024;
  static constexpr int kPatchItemsPerThread = 8;
  static constexpr int kMaxPatchesForSingleThreadBlock = kPatchThreads * kPatchItemsPerThread;

  static constexpr int kScanThreads = 512;
  static constexpr int kScanItemsPerThread = 4;

  MigrationBondedCUDAKernel();
  ~MigrationBondedCUDAKernel();
  void setup(const int numDevices, const int numPatchesHome);

  TupleDataStage getDstBuffers() { return dataDst; }
  TupleIntArraysContiguous getDeviceTupleCounts() { return d_counts; }
  TupleIntArraysContiguous getDeviceTupleOffsets() { return d_offsets; }

  void computeTupleDestination(
    const int               myDeviceIndex,
    TupleCounts             count,
    const int               numPatchesHome,
    const int4*             migrationDestination,
    const int*              patchIDtoGPU,
    const int*              patchIDtoHomePatchIndex,
    const int               aDim,
    const int               bDim,
    const int               cMaxIndex, 
    const int               bMaxIndex, 
    const int               aMaxIndex,
    cudaStream_t            stream
  );

  void reserveTupleDestination(
    const int               myDeviceIndex,
    const int               numPatchesHome,
    cudaStream_t            stream
  );

  void computePatchOffsets(
    const int               numPatchesHome,
    cudaStream_t            stream
  );

  void performTupleMigration(
    TupleCounts             count,
    cudaStream_t            stream
  );

  void updateTuples(
    TupleCounts             count,
    TupleData               data,
    const int*              ids,
    const PatchRecord*      patches,
    const double3*          d_patchMapCenter,
    const float4*           xyzq,
    const Lattice       lattice,
    cudaStream_t            stream
  );

  void copyTupleToDevice(
    TupleCounts             count,
    const int               numPatchesHome,
    TupleDataStage          h_dataStage,
    TupleIntArrays          h_counts,
    TupleIntArrays          h_offsets,
    cudaStream_t            stream
  );

  bool reallocateBufferDst(TupleCounts counts);
  bool reallocateBufferSrc(TupleCounts counts);

  TupleCounts fetchTupleCounts(const int numPatchesHome, cudaStream_t stream);
  void clearTupleCounts(const int numPatchesHome, cudaStream_t stream);

  void copyPeerDataToDevice(
    TupleDataStagePeer h_peer_data,
    TupleIntArraysPeer h_peer_counts,
    TupleIntArraysPeer h_peer_offsets,
    const int numDevices,
    cudaStream_t stream
  );

};

#endif // NAMD_CUDA
#endif // MIGRATIONBONDEDCUDA_H

