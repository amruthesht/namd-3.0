#include "CudaUtils.h"
#include "ReductionMgr.h"
#include "SequencerCUDA.h"
#include "ComputeNonbondedUtil.h"
#include "DeviceCUDA.h"
#include "TestArray.h"
#include "ComputeRestraintsCUDA.h"
#include "NamdEventsProfiling.h"
#include "AtomMap.h"
#include "common.h"
#include <algorithm> // std::fill()
//#define DEBUGM
//#define MIN_DEBUG_LEVEL 3
#ifdef NODEGROUP_FORCE_REGISTER
extern __thread DeviceCUDA *deviceCUDA;

#if 1
#define AGGREGATE_HOME_ATOMS_TO_DEVICE(fieldName, type, stream) do {     \
    size_t offset = 0;                                          \
    for (int i = 0; i < numPatchesHome; i++) { \
      PatchDataSOA& current = patchData->devData[deviceIndex].patches[i]->patchDataSOA; \
      const int numPatchAtoms = current.numAtoms;                       \
      memcpy(fieldName + offset, current.fieldName, numPatchAtoms*sizeof(type)); \
      offset += numPatchAtoms;                                          \
    }                                                                   \
    copy_HtoD<type>(fieldName, d_ ## fieldName, numAtomsHome, stream);      \
  } while(0);

#define AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(fieldName, type, stream) do {     \
    size_t offset = 0;                                          \
    for (int i = 0; i < numPatchesHomeAndProxy; i++) { \
      PatchDataSOA& current = patchListHomeAndProxy[i]->patchDataSOA; \
      const int numPatchAtoms = current.numAtoms;                       \
      memcpy(fieldName + offset, current.fieldName, numPatchAtoms*sizeof(type)); \
      offset += numPatchAtoms;                                          \
    }                                                                   \
    copy_HtoD<type>(fieldName, d_ ## fieldName, numAtomsHomeAndProxy, stream);      \
  } while(0);

#else
#define AGGREGATE_HOME_ATOMS_TO_DEVICE(fieldName, type, stream) do {     \
    size_t offset = 0;                                          \
    for (HomePatchElem *elem = patchMap->homePatchList()->begin(); elem != patchMap->homePatchList()->end(); elem++) { \
      PatchDataSOA& current = elem->patch->patchDataSOA;                \
      const int numPatchAtoms = current.numAtoms;                       \
      memcpy(fieldName + offset, current.fieldName, numPatchAtoms*sizeof(type)); \
      offset += numPatchAtoms;                                          \
    }                                                                   \
    copy_HtoD<type>(fieldName, d_ ## fieldName, numAtoms, stream);      \
  } while(0);
#endif

// This function sets whatever SOA pointer I have an 
void SequencerCUDA::registerSOAPointersToHost(){
  // fprintf(stderr, "Dev[%d] setting memories\n", deviceID);
  patchData->h_soa_pos_x[deviceIndex] = this->d_pos_x;
  patchData->h_soa_pos_y[deviceIndex] = this->d_pos_y;
  patchData->h_soa_pos_z[deviceIndex] = this->d_pos_z;

  patchData->h_soa_vel_x[deviceIndex] = this->d_vel_x;
  patchData->h_soa_vel_y[deviceIndex] = this->d_vel_y;
  patchData->h_soa_vel_z[deviceIndex] = this->d_vel_z;

  patchData->h_soa_fb_x[deviceIndex] = this->d_f_normal_x;
  patchData->h_soa_fb_y[deviceIndex] = this->d_f_normal_y;
  patchData->h_soa_fb_z[deviceIndex] = this->d_f_normal_z;

  patchData->h_soa_fn_x[deviceIndex] = this->d_f_nbond_x;
  patchData->h_soa_fn_y[deviceIndex] = this->d_f_nbond_y;
  patchData->h_soa_fn_z[deviceIndex] = this->d_f_nbond_z;

  patchData->h_soa_fs_x[deviceIndex] = this->d_f_slow_x;
  patchData->h_soa_fs_y[deviceIndex] = this->d_f_slow_y;
  patchData->h_soa_fs_z[deviceIndex] = this->d_f_slow_z;

  patchData->h_soa_id[deviceIndex]                   = this->d_idMig;
  patchData->h_soa_vdwType[deviceIndex]              = this->d_vdwType;
  patchData->h_soa_sortOrder[deviceIndex]            = this->d_sortOrder;
  patchData->h_soa_unsortOrder[deviceIndex]          = this->d_unsortOrder;
  patchData->h_soa_charge[deviceIndex]               = this->d_charge;
  patchData->h_soa_patchCenter[deviceIndex]          = this->d_patchCenter;
  patchData->h_soa_migrationDestination[deviceIndex] = this->d_migrationDestination;
  patchData->h_soa_sortSoluteIndex[deviceIndex]      = this->d_sortSoluteIndex;

  patchData->h_soa_partition[deviceIndex]            = this->d_partition;

  patchData->h_atomdata_AoS[deviceIndex] = (FullAtom*) this->d_atomdata_AoS_in;
  patchData->h_peer_record[deviceIndex] = patchData->devData[deviceIndex].d_localPatches;
}

void SequencerCUDA::copySOAHostRegisterToDevice(){
  // This function gets the host-registered SOA device pointers and copies the register itself to the device
  // NOTE: This needs to be called only when ALL masterPEs have safely called ::registerSOAPointersToHost()
  cudaCheck(cudaSetDevice(deviceID));
  copy_HtoD<double*>(patchData->h_soa_pos_x, this->d_peer_pos_x, nDevices, stream);
  copy_HtoD<double*>(patchData->h_soa_pos_y, this->d_peer_pos_y, nDevices, stream);
  copy_HtoD<double*>(patchData->h_soa_pos_z, this->d_peer_pos_z, nDevices, stream);

  copy_HtoD<float*>(patchData->h_soa_charge, this->d_peer_charge, nDevices, stream);
  if (simParams->alchOn) {
    copy_HtoD<int*>(patchData->h_soa_partition, this->d_peer_partition, nDevices, stream);
  }

  copy_HtoD<double*>(patchData->h_soa_vel_x, this->d_peer_vel_x, nDevices, stream);
  copy_HtoD<double*>(patchData->h_soa_vel_y, this->d_peer_vel_y, nDevices, stream);
  copy_HtoD<double*>(patchData->h_soa_vel_z, this->d_peer_vel_z, nDevices, stream);

  copy_HtoD<double*>(patchData->h_soa_fb_x, this->d_peer_fb_x, nDevices, stream);
  copy_HtoD<double*>(patchData->h_soa_fb_y, this->d_peer_fb_y, nDevices, stream);
  copy_HtoD<double*>(patchData->h_soa_fb_z, this->d_peer_fb_z, nDevices, stream);

  copy_HtoD<double*>(patchData->h_soa_fn_x, this->d_peer_fn_x, nDevices, stream);
  copy_HtoD<double*>(patchData->h_soa_fn_y, this->d_peer_fn_y, nDevices, stream);
  copy_HtoD<double*>(patchData->h_soa_fn_z, this->d_peer_fn_z, nDevices, stream);

  copy_HtoD<double*>(patchData->h_soa_fs_x, this->d_peer_fs_x, nDevices, stream);
  copy_HtoD<double*>(patchData->h_soa_fs_y, this->d_peer_fs_y, nDevices, stream);
  copy_HtoD<double*>(patchData->h_soa_fs_z, this->d_peer_fs_z, nDevices, stream);

  copy_HtoD<int4*>(patchData->h_soa_migrationDestination, this->d_peer_migrationDestination, nDevices, stream);
  copy_HtoD<int*>(patchData->h_soa_sortSoluteIndex, this->d_peer_sortSoluteIndex, nDevices, stream);

  copy_HtoD<int*>(patchData->h_soa_id, this->d_peer_id, nDevices, stream);
  copy_HtoD<int*>(patchData->h_soa_vdwType, this->d_peer_vdwType, nDevices, stream);
  copy_HtoD<int*>(patchData->h_soa_sortOrder, this->d_peer_sortOrder, nDevices, stream);
  copy_HtoD<int*>(patchData->h_soa_unsortOrder, this->d_peer_unsortOrder, nDevices, stream);
  copy_HtoD<double3*>(patchData->h_soa_patchCenter, this->d_peer_patchCenter, nDevices, stream);

  copy_HtoD<FullAtom*>(patchData->h_atomdata_AoS, this->d_peer_atomdata, nDevices, stream);
  copy_HtoD<CudaLocalRecord*>(patchData->h_peer_record, this->d_peer_record, nDevices, stream);

  // aggregate device pointers
  for(int i = 0; i < this->nDevices; i++)
    h_patchRecordHasForces[i] = patchData->devData[i].d_hasPatches;
  copy_HtoD_sync<bool*>(h_patchRecordHasForces, d_patchRecordHasForces, this->nDevices);

}

void SequencerCUDA::printSOAPositionsAndVelocities() {
  BigReal* h_pos_x = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);
  BigReal* h_pos_y = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);
  BigReal* h_pos_z = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);

  BigReal* h_vel_x = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);
  BigReal* h_vel_y = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);
  BigReal* h_vel_z = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);

  // DMC think this condition was a holdover from the NCCL code path
  if(false && mGpuOn){
    copy_DtoH_sync<BigReal>(d_posNew_x, h_pos_x, numAtomsHome);
    copy_DtoH_sync<BigReal>(d_posNew_y, h_pos_y, numAtomsHome);
    copy_DtoH_sync<BigReal>(d_posNew_z, h_pos_z, numAtomsHome);
  }else{
    copy_DtoH_sync<BigReal>(d_pos_x, h_pos_x, numAtomsHome);
    copy_DtoH_sync<BigReal>(d_pos_y, h_pos_y, numAtomsHome);
    copy_DtoH_sync<BigReal>(d_pos_z, h_pos_z, numAtomsHome);
  }

  copy_DtoH_sync<BigReal>(d_vel_x, h_vel_x, numAtomsHome);
  copy_DtoH_sync<BigReal>(d_vel_y, h_vel_y, numAtomsHome);
  copy_DtoH_sync<BigReal>(d_vel_z, h_vel_z, numAtomsHome);

  CmiLock(this->patchData->printlock);
  std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
  //  fprintf(stderr,  "PE[%d] pos/vel printout, numPatchesHome = %d\n", CkMyPe(), numPatchesHome);
  std::vector<HomePatch*>& homePatches = patchData->devData[deviceIndex].patches;
  for(int i =0 ; i < numPatchesHome; i++){
    CudaLocalRecord record = localPatches[i];
    const int patchID = record.patchID;
    const int stride = record.bufferOffset;
    const int numPatchAtoms = record.numAtoms;
    PatchDataSOA& current = homePatches[i]->patchDataSOA;

    fprintf(stderr, "Patch [%d]:\n", patchID);
    for(int j = 0; j < numPatchAtoms; j++){
      fprintf(stderr, " [%d, %d, %d] = %lf %lf %lf %lf %lf %lf\n", j, stride + j, current.id[j],
        h_pos_x[stride + j], h_pos_y[stride + j], h_pos_z[stride + j],
        h_vel_x[stride + j], h_vel_y[stride + j], h_vel_z[stride + j]);
    }

  }
  CmiUnlock(this->patchData->printlock);

  free(h_pos_x);
  free(h_pos_y);
  free(h_pos_z);
  free(h_vel_x);
  free(h_vel_y);
  free(h_vel_z);
}

void SequencerCUDA::printSOAForces() {
  BigReal* h_f_normal_x = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);
  BigReal* h_f_normal_y = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);
  BigReal* h_f_normal_z = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);

  BigReal* h_f_nbond_x = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);
  BigReal* h_f_nbond_y = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);
  BigReal* h_f_nbond_z = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);

  BigReal* h_f_slow_x = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);
  BigReal* h_f_slow_y = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);
  BigReal* h_f_slow_z = (BigReal*)malloc(sizeof(BigReal)*numAtomsHome);

  copy_DtoH_sync<BigReal>(d_f_normal_x, h_f_normal_x, numAtomsHome);
  copy_DtoH_sync<BigReal>(d_f_normal_y, h_f_normal_y, numAtomsHome);
  copy_DtoH_sync<BigReal>(d_f_normal_z, h_f_normal_z, numAtomsHome);

  copy_DtoH_sync<BigReal>(d_f_nbond_x, h_f_nbond_x, numAtomsHome);
  copy_DtoH_sync<BigReal>(d_f_nbond_y, h_f_nbond_y, numAtomsHome);
  copy_DtoH_sync<BigReal>(d_f_nbond_z, h_f_nbond_z, numAtomsHome);

  copy_DtoH_sync<BigReal>(d_f_slow_x, h_f_slow_x, numAtomsHome);
  copy_DtoH_sync<BigReal>(d_f_slow_y, h_f_slow_y, numAtomsHome);
  copy_DtoH_sync<BigReal>(d_f_slow_z, h_f_slow_z, numAtomsHome);

  // Great, now let's
  CmiLock(this->patchData->printlock);
  std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;

  fprintf(stderr,  "PE[%d] force printout\n", CkMyPe());
  for(int i =0 ; i < numPatchesHome; i++){
    CudaLocalRecord record = localPatches[i];
    const int patchID = record.patchID;
    const int stride = record.bufferOffset;
    const int numPatchAtoms = record.numAtoms;
    fprintf(stderr, "Patch [%d]:\n", patchID);
    for(int j = 0; j < numPatchAtoms; j++){
      fprintf(stderr, " [%d] = %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", j,
        h_f_normal_x[stride+j], h_f_normal_y[stride+j], h_f_normal_z[stride+j],
        h_f_nbond_x[stride+j],  h_f_nbond_y[stride+j],  h_f_nbond_z[stride+j],
        h_f_slow_x[stride+j],   h_f_slow_y[stride+j],   h_f_slow_z[stride+j] );
    }
  }
  CmiUnlock(this->patchData->printlock);

  free(h_f_normal_x);
  free(h_f_normal_y);
  free(h_f_normal_z);

  free(h_f_nbond_x);
  free(h_f_nbond_y);
  free(h_f_nbond_z);

  free(h_f_slow_x);
  free(h_f_slow_y);
  free(h_f_slow_z);

}

SequencerCUDA* SequencerCUDA::InstanceInit(const int deviceID_ID,
                                       SimParameters *const sim_Params) {
  if (CkpvAccess(SequencerCUDA_instance) == 0) {
    CkpvAccess(SequencerCUDA_instance) = new SequencerCUDA(deviceID_ID, sim_Params);
  }
  return CkpvAccess(SequencerCUDA_instance);
}

SequencerCUDA::SequencerCUDA(const int deviceID_ID,
                             SimParameters *const sim_Params):
  deviceID(deviceID_ID), simParams(sim_Params)
{
  restraintsKernel = NULL;
  SMDKernel = NULL;
  groupRestraintsKernel = NULL;
#if 1
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();
#endif
  initialize();
  CUDASequencerKernel = new SequencerCUDAKernel();
  CUDAMigrationKernel = new MigrationCUDAKernel();
  num_used_grids = simParams->alchGetNumOfPMEGrids();
  used_grids.resize(num_used_grids, 0);
  if (simParams->alchFepOn) {
    // at least two grids are used
    used_grids[0] = 0;
    used_grids[1] = 1;
    // if alchDecouple then two more grids are used
    if (simParams->alchDecouple) {
      used_grids[2] = 2;
      used_grids[3] = 3;
      // an extra for soft-core potential
      if (simParams->alchElecLambdaStart > 0) {
        used_grids[4] = 4;
      }
    } else {
      // in this case alchDecouple is false
      // but if there is still soft-core potential
      // then a total of 3 grids are used
      // mark the last grid for soft-core potential
      if (simParams->alchElecLambdaStart > 0) {
        used_grids[2] = 4;
      }
    }
  }
  if (simParams->alchThermIntOn) {
    used_grids[0] = 0;
    used_grids[1] = 1;
    // in TI, no matter whether soft-core potential is used
    // it has at least three grids
    if (simParams->alchDecouple) {
      used_grids[2] = 2;
      used_grids[3] = 3;
      used_grids[4] = 4;
    } else {
      used_grids[2] = 4;
    }
  }
}

SequencerCUDA::~SequencerCUDA(){
    cudaCheck(cudaSetDevice(deviceID));
    deallocateArrays();
    deallocateStaticArrays();
    deallocate_device<SettleParameters>(&sp);
    deallocate_device<int>(&settleList);
    deallocate_device<CudaRattleElem>(&rattleList);
    deallocate_device<int>(&d_consFailure);
    if (CUDASequencerKernel != NULL) delete CUDASequencerKernel;
    if (CUDAMigrationKernel != NULL) delete CUDAMigrationKernel;
    if (restraintsKernel != NULL) delete restraintsKernel;
    if(SMDKernel != NULL) delete SMDKernel;
    if (groupRestraintsKernel != NULL) delete groupRestraintsKernel;
#if 0
    cudaCheck(cudaStreamDestroy(stream));
#endif
    cudaCheck(cudaStreamDestroy(stream2));
    curandCheck(curandDestroyGenerator(curandGen));
    CmiDestroyLock(printlock);

}

void SequencerCUDA::zeroScalars(){
  numAtomsHomeAndProxyAllocated = 0;
  numAtomsHomeAllocated = 0;
  buildRigidLists = true;
  numPatchesCheckedIn = 0;
  numPatchesReady= 0;
}

void SequencerCUDA::initialize(){
  cudaCheck(cudaSetDevice(deviceID));
  nDevices       = deviceCUDA->getNumDevice() + 1 *deviceCUDA->isGpuReservedPme();
  deviceIndex    = deviceCUDA->getDeviceIndex();
#if CUDA_VERSION >= 5050 || defined(NAMD_HIP)
  int leastPriority, greatestPriority;
  cudaCheck(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
#if 0
  cudaCheck(cudaStreamCreateWithPriority(&stream, cudaStreamDefault, greatestPriority));
#else
  stream = (CkMyPe() == deviceCUDA->getMasterPe()) ? patchData->devData[deviceIndex].nbond_stream : 0;
#endif
  cudaCheck(cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, greatestPriority));
#else
  cudaCheck(cudaStreamCreate(&stream));
  cudaCheck(cudaStreamCreate(&stream2));
#endif
  curandCheck(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
  // each PE's seed needs to be different
  unsigned long long seed = simParams->randomSeed + CkMyPe();
  curandCheck(curandSetPseudoRandomGeneratorSeed(curandGen, seed));

  numAtomsHomeAllocated = 0;
  numAtomsHomeAndProxyAllocated = 0;

  totalMarginViolations = 0;
  buildRigidLists = true;
  numPatchesCheckedIn = 0;
  numPatchesReady= 0;
  PatchMap* patchMap = PatchMap::Object();
#if 1
  numPatchesGlobal = patchMap->numPatches();
#else
  numPatchesGlobal = patchMap->homePatchList()->size();
#endif
  mGpuOn      = nDevices > 1;
  // Great, now allocate the queue
  // allocate_device<unsigned int>(&deviceQueue, nDevices);
  // cudaCheck(cudaMemset(deviceQueue, 999, sizeof(unsigned int)* nDevices));
  // Allocates and registers the local queue in patchData
  // patchData->d_queues[deviceIndex] = deviceQueue;

  printlock  = CmiCreateLock();

  int numPes = CkNumPes();
  atomMapList.resize(numPes);

  //Allocates SOA registers
  allocate_device<double*>(&d_peer_pos_x, nDevices);
  allocate_device<double*>(&d_peer_pos_y, nDevices);
  allocate_device<double*>(&d_peer_pos_z, nDevices);
  allocate_device<float*>(&d_peer_charge, nDevices);
  if (simParams->alchOn) {
    allocate_device<int*>(&d_peer_partition, nDevices);
  }
  allocate_device<double*>(&d_peer_vel_x, nDevices);
  allocate_device<double*>(&d_peer_vel_y, nDevices);
  allocate_device<double*>(&d_peer_vel_z, nDevices);

  allocate_device<double*>(&d_peer_fb_x, nDevices);
  allocate_device<double*>(&d_peer_fb_y, nDevices);
  allocate_device<double*>(&d_peer_fb_z, nDevices);
  allocate_device<double*>(&d_peer_fn_x, nDevices);
  allocate_device<double*>(&d_peer_fn_y, nDevices);
  allocate_device<double*>(&d_peer_fn_z, nDevices);
  allocate_device<double*>(&d_peer_fs_x, nDevices);
  allocate_device<double*>(&d_peer_fs_y, nDevices);
  allocate_device<double*>(&d_peer_fs_z, nDevices);

  allocate_device<int4*>(&d_peer_migrationDestination, nDevices);
  allocate_device<int*>(&d_peer_sortSoluteIndex, nDevices);
  allocate_device<int*>(&d_peer_id, nDevices);
  allocate_device<int*>(&d_peer_vdwType, nDevices);
  allocate_device<int*>(&d_peer_sortOrder, nDevices);
  allocate_device<int*>(&d_peer_unsortOrder, nDevices);
  allocate_device<double3*>(&d_peer_patchCenter, nDevices);

  allocate_device<FullAtom*>(&d_peer_atomdata, nDevices);
  allocate_device<CudaLocalRecord*>(&d_peer_record, nDevices);

  allocate_device<bool*>(&d_patchRecordHasForces, nDevices);

  allocate_host<bool*>(&h_patchRecordHasForces, nDevices);
  // Patch-related host datastructures
  allocate_host<CudaAtom*>(&cudaAtomLists, numPatchesGlobal);
  allocate_host<double3>(&patchCenter,  numPatchesGlobal);
  allocate_host<int>(&globalToLocalID, numPatchesGlobal);
  allocate_host<int>(&patchToDeviceMap,numPatchesGlobal);
  allocate_host<double3>(&awayDists, numPatchesGlobal);
  allocate_host<double3>(&patchMin, numPatchesGlobal);
  allocate_host<double3>(&patchMax, numPatchesGlobal);
  //allocate_host<Lattice>(&lattices, numPatchesGlobal);
  allocate_host<Lattice>(&pairlist_lattices, numPatchesGlobal); // only needed for langevin
  allocate_host<double>(&patchMaxAtomMovement, numPatchesGlobal);
  allocate_host<double>(&patchNewTolerance, numPatchesGlobal);
  allocate_host<CudaMInfo>(&mInfo, numPatchesGlobal);

  //Patch-related device datastructures
  allocate_device<double3>(&d_awayDists, numPatchesGlobal);
  allocate_device<double3>(&d_patchMin, numPatchesGlobal);
  allocate_device<double3>(&d_patchMax, numPatchesGlobal);
  allocate_device<int>(&d_globalToLocalID, numPatchesGlobal);
  allocate_device<int>(&d_patchToDeviceMap, numPatchesGlobal);
  allocate_device<double3>(&d_patchCenter,  numPatchesGlobal);
  allocate_device<Lattice>(&d_lattices, numPatchesGlobal);
  allocate_device<Lattice>(&d_pairlist_lattices, numPatchesGlobal);
  allocate_device<double>(&d_patchMaxAtomMovement, numPatchesGlobal);
  allocate_device<double>(&d_patchNewTolerance, numPatchesGlobal);
  allocate_device<CudaMInfo>(&d_mInfo, numPatchesGlobal);

  // Allocate host memory for scalar variables
  allocate_device<int>(&d_killme, 1);
  allocate_device<char>(&d_barrierFlag, 1);
  allocate_device<unsigned int>(&d_tbcatomic, 5);
  allocate_device<BigReal>(&d_kineticEnergy, ATOMIC_BINS);
  allocate_device<BigReal>(&d_intKineticEnergy, ATOMIC_BINS);
  allocate_device<BigReal>(&d_momentum_x, ATOMIC_BINS);
  allocate_device<BigReal>(&d_momentum_y, ATOMIC_BINS);
  allocate_device<BigReal>(&d_momentum_z, ATOMIC_BINS);
  allocate_device<BigReal>(&d_angularMomentum_x, ATOMIC_BINS);
  allocate_device<BigReal>(&d_angularMomentum_y, ATOMIC_BINS);
  allocate_device<BigReal>(&d_angularMomentum_z, ATOMIC_BINS);
  allocate_device<cudaTensor>(&d_virial, ATOMIC_BINS);
  allocate_device<cudaTensor>(&d_intVirialNormal, ATOMIC_BINS);
  allocate_device<cudaTensor>(&d_intVirialNbond, ATOMIC_BINS);
  allocate_device<cudaTensor>(&d_intVirialSlow, ATOMIC_BINS);
  allocate_device<cudaTensor>(&d_rigidVirial, ATOMIC_BINS);
  // for lone pairs
  allocate_device<cudaTensor>(&d_lpVirialNormal, 1);
  allocate_device<cudaTensor>(&d_lpVirialNbond, 1);
  allocate_device<cudaTensor>(&d_lpVirialSlow, 1);
  //space for globalmaster forces on device
  allocate_device<cudaTensor>(&d_extVirial, ATOMIC_BINS * EXT_FORCE_TOTAL);
  allocate_device<double3>(&d_extForce, ATOMIC_BINS * EXT_FORCE_TOTAL);
  allocate_device<double>(&d_extEnergy, ATOMIC_BINS * EXT_FORCE_TOTAL);

  allocate_device<SettleParameters>(&sp, 1);

  //allocates host scalars in host-mapped, pinned memory
  allocate_host<int>(&killme, 1);
  allocate_host<BigReal>(&kineticEnergy, 1);
  allocate_host<BigReal>(&intKineticEnergy, 1);
  allocate_host<BigReal>(&kineticEnergy_half, 1);
  allocate_host<BigReal>(&intKineticEnergy_half, 1);
  allocate_host<BigReal>(&momentum_x, 1);
  allocate_host<BigReal>(&momentum_y, 1);
  allocate_host<BigReal>(&momentum_z, 1);
  allocate_host<BigReal>(&angularMomentum_x, 1);
  allocate_host<BigReal>(&angularMomentum_y, 1);
  allocate_host<BigReal>(&angularMomentum_z, 1);
  allocate_host<int>(&consFailure, 1);
  allocate_host<double>(&extEnergy, EXT_FORCE_TOTAL);
  allocate_host<double3>(&extForce, EXT_FORCE_TOTAL);
  allocate_host<unsigned int>(&h_marginViolations, 1);
  allocate_host<unsigned int>(&h_periodicCellSmall, 1);

  // allocates host cudaTensors in host-mapped, pinned memory
  allocate_host<cudaTensor>(&virial, 1);
  allocate_host<cudaTensor>(&virial_half, 1);
  allocate_host<cudaTensor>(&intVirialNormal, 1);
  allocate_host<cudaTensor>(&intVirialNormal_half, 1);
  allocate_host<cudaTensor>(&intVirialNbond, 1);
  allocate_host<cudaTensor>(&intVirialSlow, 1);
  allocate_host<cudaTensor>(&rigidVirial, 1);
  allocate_host<cudaTensor>(&extVirial, EXT_FORCE_TOTAL);
  allocate_host<cudaTensor>(&lpVirialNormal, 1);
  allocate_host<cudaTensor>(&lpVirialNbond, 1);
  allocate_host<cudaTensor>(&lpVirialSlow, 1);


  //Sets values for scalars
  *kineticEnergy = 0.0;
  *intKineticEnergy = 0.0;
  *kineticEnergy_half = 0.0;
  *intKineticEnergy_half = 0.0;
  *momentum_x = 0.0;
  *momentum_y = 0.0;
  *momentum_z = 0.0;
  *angularMomentum_x = 0.0;
  *angularMomentum_y = 0.0;
  *angularMomentum_z = 0.0;
  *consFailure = 0;
  *killme = 0;

  // JM: Basic infrastructure to time kernels
  //  XXX TODO: Add timing functions to be switched on/off for each of these
  t_total               = 0;
  t_vverlet             = 0;
  t_pairlistCheck       = 0;
  t_setComputePositions = 0;
  t_accumulateForceKick = 0;
  t_rattle              = 0;
  t_submitHalf          = 0;
  t_submitReductions1   = 0;
  t_submitReductions2   = 0;

  cudaEventCreate(&eventStart);
  cudaEventCreate(&eventStop);
  cudaCheck(cudaMemset(d_patchNewTolerance, 0, sizeof(BigReal)*numPatchesGlobal));
  cudaCheck(cudaMemset(d_kineticEnergy, 0, sizeof(BigReal)));
  cudaCheck(cudaMemset(d_tbcatomic, 0, sizeof(unsigned int) * 5));
  cudaCheck(cudaMemset(d_momentum_x, 0, ATOMIC_BINS * sizeof(BigReal)));
  cudaCheck(cudaMemset(d_momentum_y, 0, ATOMIC_BINS * sizeof(BigReal)));
  cudaCheck(cudaMemset(d_momentum_z, 0, ATOMIC_BINS * sizeof(BigReal)));
  cudaCheck(cudaMemset(d_angularMomentum_x, 0, ATOMIC_BINS * sizeof(BigReal)));
  cudaCheck(cudaMemset(d_angularMomentum_y, 0, ATOMIC_BINS * sizeof(BigReal)));
  cudaCheck(cudaMemset(d_angularMomentum_z, 0, ATOMIC_BINS * sizeof(BigReal)));
  cudaCheck(cudaMemset(d_intKineticEnergy, 0, ATOMIC_BINS * sizeof(BigReal)));
  cudaCheck(cudaMemset(d_virial, 0, ATOMIC_BINS * sizeof(cudaTensor)));
  cudaCheck(cudaMemset(d_rigidVirial, 0, ATOMIC_BINS * sizeof(cudaTensor)));
  cudaCheck(cudaMemset(d_intVirialNormal, 0, ATOMIC_BINS * sizeof(cudaTensor)));
  cudaCheck(cudaMemset(d_intVirialNbond, 0, ATOMIC_BINS * sizeof(cudaTensor)));
  cudaCheck(cudaMemset(d_intVirialSlow, 0, ATOMIC_BINS * sizeof(cudaTensor)));
  cudaCheck(cudaMemset(d_lpVirialNormal, 0, 1 * sizeof(cudaTensor)));
  cudaCheck(cudaMemset(d_lpVirialNbond, 0, 1 * sizeof(cudaTensor)));
  cudaCheck(cudaMemset(d_lpVirialSlow, 0, 1 * sizeof(cudaTensor)));
  cudaCheck(cudaMemset(d_extVirial, 0, ATOMIC_BINS * EXT_FORCE_TOTAL * sizeof(cudaTensor)));
  cudaCheck(cudaMemset(d_extForce, 0, ATOMIC_BINS * EXT_FORCE_TOTAL * sizeof(double3)));
  cudaCheck(cudaMemset(d_extEnergy, 0, ATOMIC_BINS * EXT_FORCE_TOTAL * sizeof(double)));

  memset(h_marginViolations,  0, sizeof(unsigned int));
  memset(h_periodicCellSmall, 0, sizeof(unsigned int));
  memset(virial, 0, sizeof(cudaTensor));
  memset(rigidVirial, 0, sizeof(cudaTensor));
  memset(intVirialNormal, 0, sizeof(cudaTensor));
  memset(intVirialNbond, 0, sizeof(cudaTensor));
  memset(intVirialSlow, 0, sizeof(cudaTensor));
  memset(lpVirialNormal, 0, sizeof(cudaTensor));
  memset(lpVirialNbond, 0, sizeof(cudaTensor));
  memset(lpVirialSlow, 0, sizeof(cudaTensor));
  memset(globalToLocalID, -1, sizeof(int)*numPatchesGlobal);

  settleList = NULL;
  settleListSize = 0;
  rattleList = NULL;
  rattleListSize = 0;
  d_consFailure = NULL;
  d_consFailureSize = 0;

  // JM: I can bundle the CompAtom and CudaAtomList pointers from the
  //     patches here and fill the Host-arrays after integration. :]
  // if single-gpu simulation, numPatchesHome is the same as numPatchesGlobal
  numPatchesHome = numPatchesGlobal; 


  int count = 0;
  // Single-GPU case
  if(!mGpuOn){
    // JM NOTE: This works for a single device only, but it's fast if we want to have multiple-PE's sharing it
    if(deviceCUDA->getMasterPe() == CkMyPe()){
      for(int i = 0; i < numPes; i++) {
        atomMapList[i] = AtomMap::ObjectOnPe(i);
        PatchMap* patchMap = PatchMap::ObjectOnPe(i);
        int npatch = patchMap->numPatches();
        for (int j = 0; j < npatch; j++) {
          HomePatch *patch = patchMap->homePatch(j);
          // JM NOTE: This data structure can be preserved through migration steps
          if(patch != NULL) {
            patchList.push_back(patch);
            patchNewTolerance[count++] =
              0.5 * ( simParams->pairlistDist - simParams->cutoff );
            numAtomsHomeAndProxy += patchMap->patch(j)->getNumAtoms();

            patchData->devData[deviceIndex].patches.push_back(patch);
            patchListHomeAndProxy.push_back(patch);
          }
        }
      }
    }
  }else{
    // Multi-device case
    /* The logic here is a bit trickier than the one on the single-GPU case.
       Each GPU will only allocate space for its home patches and any required proxy patches, 
       This is going to eliminate the possibility of using NCCL-based all reduce for communication
       but DMC thinks this is okay...
    */

    if(deviceCUDA->getMasterPe() == CkMyPe()){
      for(int i = 0; i < deviceCUDA->getNumPesSharingDevice(); i++){
        PatchMap* pm = PatchMap::ObjectOnPe(deviceCUDA->getPesSharingDevice(i));
        for (int j = 0; j < pm->numPatches(); j++) {
          // Aggregates the patches in a separate data structure for now
          HomePatch *patch = pm->homePatch(j);
          if(patch != NULL) {
            patchData->devData[deviceIndex].patches.push_back(patch);
          }
        }
      }

      // if MGPU is On, we also try to set up peer access
      deviceCUDA->setupDevicePeerAccess();
#ifdef NAMD_NCCL_ALLREDUCE
      deviceCUDA->setNcclUniqueId(patchData->ncclId);
      deviceCUDA->setupNcclComm();
#endif

    }
  }
  PatchMap *pm = PatchMap::Object();
  isPeriodic = (pm->periodic_a() && pm->periodic_b() && pm->periodic_c());

  // XXX TODO decide how the biasing methods will work in the future -
  //    for multiple devices this will be a problem, how to solve it?
  if (simParams->constraintsOn) {
    restraintsKernel = new ComputeRestraintsCUDA(patchList, atomMapList,
        stream);
  }
  if (simParams->SMDOn) {
    SMDKernel = new ComputeSMDCUDA(patchList, simParams->SMDk, simParams->SMDk2,
      simParams->SMDVel, make_double3(simParams->SMDDir.x, simParams->SMDDir.y, simParams->SMDDir.z),
      simParams->SMDOutputFreq, simParams->firstTimestep, simParams->SMDFile, 
      Node::Object()->molecule->numAtoms);
  }
  if (simParams->groupRestraintsOn) {
    groupRestraintsKernel = new ComputeGroupRestraintsCUDA(simParams->outputEnergies,
      simParams->groupRestraints);
  }
}

bool SequencerCUDA::reallocateArrays(int in_numAtomsHome, int in_numAtomsHomeAndProxy)
{
  cudaCheck(cudaSetDevice(deviceID));
  const float OVERALLOC = 1.5f;

  if (in_numAtomsHomeAndProxy <= numAtomsHomeAndProxyAllocated && in_numAtomsHome <= numAtomsHomeAllocated ) {
    return false;
  }

  deallocateArrays();

  numAtomsHomeAndProxyAllocated = (int) ((float) in_numAtomsHomeAndProxy * OVERALLOC);
  numAtomsHomeAllocated = (int) ((float) in_numAtomsHome * OVERALLOC);
 
  allocate_host<double>(&f_normal_x, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&f_normal_y, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&f_normal_z, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&f_nbond_x, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&f_nbond_y, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&f_nbond_z, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&f_slow_x, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&f_slow_y, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&f_slow_z, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&pos_x, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&pos_y, numAtomsHomeAndProxyAllocated);
  allocate_host<double>(&pos_z, numAtomsHomeAndProxyAllocated);
  if (simParams->colvarsOn || simParams->tclForcesOn){
    allocate_host<double>(&f_global_x, numAtomsHomeAndProxyAllocated);
    allocate_host<double>(&f_global_y, numAtomsHomeAndProxyAllocated);
    allocate_host<double>(&f_global_z, numAtomsHomeAndProxyAllocated);
  }
  allocate_host<float>(&charge, numAtomsHomeAndProxyAllocated);
  allocate_host<int>(&sortOrder, numAtomsHomeAndProxyAllocated);
  allocate_host<int>(&unsortOrder, numAtomsHomeAndProxyAllocated);
 
  allocate_host<double>(&recipMass, numAtomsHomeAllocated);
  allocate_host<double>(&vel_x, numAtomsHomeAllocated);
  allocate_host<double>(&vel_y, numAtomsHomeAllocated);
  allocate_host<double>(&vel_z, numAtomsHomeAllocated);
  allocate_host<char3>(&transform, numAtomsHomeAllocated);
  allocate_host<float>(&mass,   numAtomsHomeAllocated);
  if (simParams->alchOn) {
    allocate_host<int>(&partition, numAtomsHomeAndProxyAllocated);
  }
  allocate_host<float>(&langevinParam, numAtomsHomeAllocated);
  allocate_host<float>(&langScalVelBBK2, numAtomsHomeAllocated);
  allocate_host<float>(&langScalRandBBK2, numAtomsHomeAllocated);

  // array buffers for pseudorandom normal distribution must be even length
  // choose n to be the smallest even number >= numIntegrationAtoms
  // guarantees that n is always > numIntegrationAtoms
  int n = (numAtomsHomeAllocated + 1) & (~1);
  allocate_host<int>(&hydrogenGroupSize, numAtomsHomeAllocated);
  allocate_host<int>(&atomFixed,       numAtomsHomeAllocated);
  allocate_host<float>(&rigidBondLength, numAtomsHomeAllocated);


  if (simParams->useDeviceMigration) {
    allocate_host<int>(&idMig, numAtomsHomeAllocated);
    allocate_host<int>(&vdwType, numAtomsHomeAllocated);
  }

  allocate_device<double>(&d_f_raw, 9 * numAtomsHomeAndProxyAllocated); // Total number of force buffers 
  d_f_normal_x = &d_f_raw[numAtomsHomeAndProxyAllocated*0];
  d_f_normal_y = &d_f_raw[numAtomsHomeAndProxyAllocated*1];
  d_f_normal_z = &d_f_raw[numAtomsHomeAndProxyAllocated*2];
  d_f_nbond_x =  &d_f_raw[numAtomsHomeAndProxyAllocated*3];
  d_f_nbond_y =  &d_f_raw[numAtomsHomeAndProxyAllocated*4];
  d_f_nbond_z =  &d_f_raw[numAtomsHomeAndProxyAllocated*5];
  d_f_slow_x =   &d_f_raw[numAtomsHomeAndProxyAllocated*6];
  d_f_slow_y =   &d_f_raw[numAtomsHomeAndProxyAllocated*7];
  d_f_slow_z =   &d_f_raw[numAtomsHomeAndProxyAllocated*8];
  allocate_device<double>(&d_pos_raw, 3 * numAtomsHomeAndProxyAllocated);
  d_pos_x    = &d_pos_raw[numAtomsHomeAndProxyAllocated*0];
  d_pos_y    = &d_pos_raw[numAtomsHomeAndProxyAllocated*1];
  d_pos_z    = &d_pos_raw[numAtomsHomeAndProxyAllocated*2];
  allocate_device<float>(&d_charge, numAtomsHomeAndProxyAllocated);
  if (simParams->colvarsOn || simParams->tclForcesOn){  
    allocate_device<double>(&d_f_global_x, numAtomsHomeAndProxyAllocated);
    allocate_device<double>(&d_f_global_y, numAtomsHomeAndProxyAllocated);
    allocate_device<double>(&d_f_global_z, numAtomsHomeAndProxyAllocated);
  }
  allocate_device<int>(&d_sortOrder, numAtomsHomeAndProxyAllocated);
  allocate_device<int>(&d_unsortOrder, numAtomsHomeAndProxyAllocated);

  // allocate memory for backup forces in MC barostat
  if (simParams->monteCarloPressureOn) {
    // Total number of backup force and positions buffers 
    allocate_device<double>(&d_f_rawMC, numAtomsHomeAndProxyAllocated*9); 
    allocate_device<double>(&d_pos_rawMC, numAtomsHomeAndProxyAllocated*3); 
    d_f_normalMC_x = &d_f_rawMC[numAtomsHomeAndProxyAllocated*0];
    d_f_normalMC_y = &d_f_rawMC[numAtomsHomeAndProxyAllocated*1];
    d_f_normalMC_z = &d_f_rawMC[numAtomsHomeAndProxyAllocated*2];
    d_f_nbondMC_x =  &d_f_rawMC[numAtomsHomeAndProxyAllocated*3];
    d_f_nbondMC_y =  &d_f_rawMC[numAtomsHomeAndProxyAllocated*4];
    d_f_nbondMC_z =  &d_f_rawMC[numAtomsHomeAndProxyAllocated*5];
    d_f_slowMC_x =   &d_f_rawMC[numAtomsHomeAndProxyAllocated*6];
    d_f_slowMC_y =   &d_f_rawMC[numAtomsHomeAndProxyAllocated*7];
    d_f_slowMC_z =   &d_f_rawMC[numAtomsHomeAndProxyAllocated*8];
    d_posMC_x = &d_pos_rawMC[numAtomsHomeAndProxyAllocated*0];
    d_posMC_y = &d_pos_rawMC[numAtomsHomeAndProxyAllocated*1];
    d_posMC_z = &d_pos_rawMC[numAtomsHomeAndProxyAllocated*2];

    allocate_host<int>(&id, numAtomsHomeAndProxyAllocated);
    allocate_device<int>(&d_id, numAtomsHomeAndProxyAllocated);
    allocate_device<int>(&d_idOrder, numAtomsHomeAndProxyAllocated);
    allocate_device<int>(&d_moleculeAtom, numAtomsHomeAndProxyAllocated);
    // we can use molecule_size + 1, rather than atom size
    allocate_device<int>(&d_moleculeStartIndex, numAtomsHomeAndProxyAllocated);
  }
  
  if (simParams->alchOn) {
    allocate_device<int>(&d_partition, numAtomsHomeAndProxyAllocated);
  }

  allocate_device<double>(&d_posNew_raw, 3 * numAtomsHomeAllocated);
  d_posNew_x = &d_posNew_raw[numAtomsHomeAllocated*0];
  d_posNew_y = &d_posNew_raw[numAtomsHomeAllocated*1];
  d_posNew_z = &d_posNew_raw[numAtomsHomeAllocated*2];
  allocate_device<double>(&d_vel_x, numAtomsHomeAllocated);
  allocate_device<double>(&d_vel_y, numAtomsHomeAllocated);
  allocate_device<double>(&d_vel_z, numAtomsHomeAllocated);
  allocate_device<double>(&d_recipMass, numAtomsHomeAllocated);
  allocate_device<char3>(&d_transform, numAtomsHomeAllocated);
  allocate_device<double>(&d_velNew_x, numAtomsHomeAllocated);
  allocate_device<double>(&d_velNew_y, numAtomsHomeAllocated);
  allocate_device<double>(&d_velNew_z, numAtomsHomeAllocated);
  allocate_device<double>(&d_posSave_x, numAtomsHomeAllocated);
  allocate_device<double>(&d_posSave_y, numAtomsHomeAllocated);
  allocate_device<double>(&d_posSave_z, numAtomsHomeAllocated);
  allocate_device<double>(&d_rcm_x, numAtomsHomeAllocated);
  allocate_device<double>(&d_rcm_y, numAtomsHomeAllocated);
  allocate_device<double>(&d_rcm_z, numAtomsHomeAllocated);
  allocate_device<double>(&d_vcm_x, numAtomsHomeAllocated);
  allocate_device<double>(&d_vcm_y, numAtomsHomeAllocated);
  allocate_device<double>(&d_vcm_z, numAtomsHomeAllocated);

  allocate_device<float>(&d_mass,   numAtomsHomeAllocated);
  allocate_device<float>(&d_langevinParam, numAtomsHomeAllocated);
  allocate_device<float>(&d_langScalVelBBK2, numAtomsHomeAllocated);
  allocate_device<float>(&d_langScalRandBBK2, numAtomsHomeAllocated);
  allocate_device<float>(&d_gaussrand_x, numAtomsHomeAllocated);
  allocate_device<float>(&d_gaussrand_y, numAtomsHomeAllocated);
  allocate_device<float>(&d_gaussrand_z, numAtomsHomeAllocated);
  allocate_device<int>(&d_hydrogenGroupSize, numAtomsHomeAllocated);
  allocate_device<float>(&d_rigidBondLength, numAtomsHomeAllocated);
  allocate_device<int>(&d_atomFixed, numAtomsHomeAllocated);

  if (simParams->useDeviceMigration) {
    allocate_device<int>(&d_idMig, numAtomsHomeAndProxyAllocated);
    allocate_device<int>(&d_vdwType, numAtomsHomeAndProxyAllocated);
    allocate_device<FullAtom>(&d_atomdata_AoS,   numAtomsHomeAllocated);
    allocate_device<int>(&d_migrationGroupSize, numAtomsHomeAllocated);
    allocate_device<int>(&d_migrationGroupIndex, numAtomsHomeAllocated);
    allocate_device<int>(&d_sortIndex, numAtomsHomeAllocated);
  }

  // Memsets the d_pos arrays in order to reduce them afterwards;
  memset(pos_x, 0, sizeof(double)*numAtomsHomeAndProxyAllocated);
  memset(pos_y, 0, sizeof(double)*numAtomsHomeAndProxyAllocated);
  memset(pos_z, 0, sizeof(double)*numAtomsHomeAndProxyAllocated);
  cudaCheck(cudaMemset(d_pos_x, 0 , sizeof(double)*numAtomsHomeAndProxyAllocated)); 
  cudaCheck(cudaMemset(d_pos_y, 0 , sizeof(double)*numAtomsHomeAndProxyAllocated));
  cudaCheck(cudaMemset(d_pos_z, 0 , sizeof(double)*numAtomsHomeAndProxyAllocated));
  cudaCheck(cudaMemset(d_vel_x, 0 , sizeof(double)*numAtomsHomeAllocated)); 
  cudaCheck(cudaMemset(d_vel_y, 0 , sizeof(double)*numAtomsHomeAllocated));
  cudaCheck(cudaMemset(d_vel_z, 0 , sizeof(double)*numAtomsHomeAllocated));

  cudaCheck(cudaMemset(d_posNew_x, 0 , sizeof(double)*numAtomsHomeAllocated)); 
  cudaCheck(cudaMemset(d_posNew_y, 0 , sizeof(double)*numAtomsHomeAllocated));
  cudaCheck(cudaMemset(d_posNew_z, 0 , sizeof(double)*numAtomsHomeAllocated));
  cudaCheck(cudaMemset(d_velNew_x, 0 , sizeof(double)*numAtomsHomeAllocated)); 
  cudaCheck(cudaMemset(d_velNew_y, 0 , sizeof(double)*numAtomsHomeAllocated));
  cudaCheck(cudaMemset(d_velNew_z, 0 , sizeof(double)*numAtomsHomeAllocated));

  return true;
}

void SequencerCUDA::reallocateMigrationDestination() {
  if (d_migrationDestination != NULL) deallocate_device<int4>(&d_migrationDestination);
  allocate_device<int4>(&d_migrationDestination, numAtomsHomeAndProxyAllocated);
}

void SequencerCUDA::deallocateArrays() {
  if (numAtomsHomeAndProxyAllocated != 0) {
    cudaCheck(cudaSetDevice(deviceID));

    deallocate_host<double>(&f_normal_x);
    deallocate_host<double>(&f_normal_y);
    deallocate_host<double>(&f_normal_z);
    if (simParams->colvarsOn || simParams->tclForcesOn){    
      deallocate_host<double>(&f_global_x);
      deallocate_host<double>(&f_global_y);
      deallocate_host<double>(&f_global_z);
    }
    deallocate_host<double>(&f_nbond_x);
    deallocate_host<double>(&f_nbond_y);
    deallocate_host<double>(&f_nbond_z);
    deallocate_host<double>(&f_slow_x);
    deallocate_host<double>(&f_slow_y);
    deallocate_host<double>(&f_slow_z);
    deallocate_host<double>(&pos_x);
    deallocate_host<double>(&pos_y);
    deallocate_host<double>(&pos_z);
    deallocate_host<float>(&charge);
    deallocate_host<int>(&sortOrder);
    deallocate_host<int>(&unsortOrder);
    deallocate_host<double>(&recipMass);
    deallocate_host<double>(&vel_x);
    deallocate_host<double>(&vel_y);
    deallocate_host<double>(&vel_z);
    deallocate_host<char3>(&transform);
    deallocate_host<float>(&mass);
    if (simParams->alchOn) {
      deallocate_host<int>(&partition);
    }
    deallocate_host<float>(&langevinParam);
    deallocate_host<float>(&langScalVelBBK2);
    deallocate_host<float>(&langScalRandBBK2);

    deallocate_host<int>(&hydrogenGroupSize);
    deallocate_host<int>(&atomFixed);
    deallocate_host<float>(&rigidBondLength);

    deallocate_device<double>(&d_f_raw);
    deallocate_device<double>(&d_pos_raw);
    deallocate_device<float>(&d_charge);
    deallocate_device<int>(&d_sortOrder);
    deallocate_device<int>(&d_unsortOrder);
    if (simParams->alchOn) {
      deallocate_device<int>(&d_partition);
    }

    deallocate_device<double>(&d_posNew_raw);
    
    if (simParams->monteCarloPressureOn) {
      deallocate_device<double>(&d_f_rawMC);
      deallocate_device<double>(&d_pos_rawMC);

      deallocate_host<int>(&id);
      deallocate_device<int>(&d_id);
      deallocate_device<int>(&d_idOrder);
      deallocate_device<int>(&d_moleculeAtom);
      deallocate_device<int>(&d_moleculeStartIndex);
    }

    if (simParams->useDeviceMigration) {
      deallocate_host<int>(&idMig);
      deallocate_host<int>(&vdwType);
      deallocate_device<int>(&d_idMig);
      deallocate_device<int>(&d_vdwType);
      deallocate_device<FullAtom>(&d_atomdata_AoS);
      deallocate_device<int>(&d_migrationGroupSize);
      deallocate_device<int>(&d_migrationGroupIndex);
      deallocate_device<int>(&d_sortIndex);
    }
    if (simParams->colvarsOn || simParams->tclForcesOn){
      deallocate_device<double>(&d_f_global_x);
      deallocate_device<double>(&d_f_global_y);
      deallocate_device<double>(&d_f_global_z);
    }
    deallocate_device<double>(&d_vel_x);
    deallocate_device<double>(&d_vel_y);
    deallocate_device<double>(&d_vel_z);
    deallocate_device<double>(&d_recipMass);
    deallocate_device<char3>(&d_transform);
    deallocate_device<double>(&d_velNew_x);
    deallocate_device<double>(&d_velNew_y);
    deallocate_device<double>(&d_velNew_z);
    deallocate_device<double>(&d_posSave_x);
    deallocate_device<double>(&d_posSave_y);
    deallocate_device<double>(&d_posSave_z);
    deallocate_device<double>(&d_rcm_x);
    deallocate_device<double>(&d_rcm_y);
    deallocate_device<double>(&d_rcm_z);
    deallocate_device<double>(&d_vcm_x);
    deallocate_device<double>(&d_vcm_y);
    deallocate_device<double>(&d_vcm_z);
    deallocate_device<float>(&d_mass);
    deallocate_device<float>(&d_langevinParam);
    deallocate_device<float>(&d_langScalVelBBK2);
    deallocate_device<float>(&d_langScalRandBBK2);
    deallocate_device<float>(&d_gaussrand_x);
    deallocate_device<float>(&d_gaussrand_y);
    deallocate_device<float>(&d_gaussrand_z);
    deallocate_device<int>(&d_hydrogenGroupSize);
    deallocate_device<float>(&d_rigidBondLength);
    deallocate_device<int>(&d_atomFixed);
  }
}

void SequencerCUDA::deallocateStaticArrays() {
  cudaCheck(cudaSetDevice(deviceID));

  deallocate_host<cudaTensor>(&extVirial);
  deallocate_host<double3>(&extForce);
  deallocate_host<double>(&extEnergy);
    deallocate_host<unsigned int>(&h_marginViolations);
  deallocate_host<unsigned int>(&h_periodicCellSmall);

  // XXX TODO: Deallocate the additional arrays we added for shmem version
  deallocate_host<double3>(&awayDists);
  deallocate_host<double3>(&patchMin);
  deallocate_host<double3>(&patchMax);
  deallocate_host<CudaAtom*>(&cudaAtomLists);
  deallocate_host<double3>(&patchCenter);
  deallocate_host<int>(&globalToLocalID);
  deallocate_host<int>(&patchToDeviceMap);
  deallocate_host<Lattice>(&pairlist_lattices);
  deallocate_host<double>(&patchMaxAtomMovement);
  deallocate_host<double>(&patchNewTolerance);
  deallocate_host<CudaMInfo>(&mInfo);
  deallocate_host<bool*>(&h_patchRecordHasForces);

  deallocate_host<cudaTensor>(&lpVirialNormal);
  deallocate_host<cudaTensor>(&lpVirialNbond);
  deallocate_host<cudaTensor>(&lpVirialSlow);

  deallocate_device<double3>(&d_awayDists);
  deallocate_device<double3>(&d_patchMin);
  deallocate_device<double3>(&d_patchMax);
  deallocate_device<int>(&d_globalToLocalID);
  deallocate_device<int>(&d_patchToDeviceMap);
  deallocate_device<int>(&d_sortOrder);
  deallocate_device<int>(&d_unsortOrder);
  deallocate_device<double3>(&d_patchCenter);
  deallocate_device<Lattice>(&d_lattices);
  deallocate_device<Lattice>(&d_pairlist_lattices);
  deallocate_device<double>(&d_patchMaxAtomMovement);
  deallocate_device<double>(&d_patchNewTolerance);
  deallocate_device<CudaMInfo>(&d_mInfo);

  deallocate_device<int>(&d_killme);
  deallocate_device<char>(&d_barrierFlag);
  deallocate_device<unsigned int>(&d_tbcatomic);
  deallocate_device<BigReal>(&d_kineticEnergy);
  deallocate_device<BigReal>(&d_intKineticEnergy);
  deallocate_device<BigReal>(&d_momentum_x);
  deallocate_device<BigReal>(&d_momentum_y);
  deallocate_device<BigReal>(&d_momentum_z);
  deallocate_device<BigReal>(&d_angularMomentum_x);
  deallocate_device<BigReal>(&d_angularMomentum_y);
  deallocate_device<BigReal>(&d_angularMomentum_z);
  deallocate_device<cudaTensor>(&d_virial);
  deallocate_device<cudaTensor>(&d_intVirialNormal);
  deallocate_device<cudaTensor>(&d_intVirialNbond);
  deallocate_device<cudaTensor>(&d_intVirialSlow);
  deallocate_device<cudaTensor>(&d_lpVirialNormal);
  deallocate_device<cudaTensor>(&d_lpVirialNbond);
  deallocate_device<cudaTensor>(&d_lpVirialSlow);
  deallocate_device<cudaTensor>(&d_rigidVirial);
  deallocate_device<cudaTensor>(&d_extVirial);
  deallocate_device<double3>(&d_extForce);
  deallocate_device<double>(&d_extEnergy);
  deallocate_device<SettleParameters>(&sp);
  deallocate_device<unsigned int>(&deviceQueue);
  deallocate_device<double*>(&d_peer_pos_x);
  deallocate_device<double*>(&d_peer_pos_y);
  deallocate_device<double*>(&d_peer_pos_z);
  deallocate_device<float*>(&d_peer_charge);
  deallocate_device<int*>(&d_peer_partition);
  deallocate_device<double*>(&d_peer_vel_x);
  deallocate_device<double*>(&d_peer_vel_y);
  deallocate_device<double*>(&d_peer_vel_z);
  deallocate_device<double*>(&d_peer_fb_x);
  deallocate_device<double*>(&d_peer_fb_y);
  deallocate_device<double*>(&d_peer_fb_z);
  deallocate_device<double*>(&d_peer_fn_x);
  deallocate_device<double*>(&d_peer_fn_y);
  deallocate_device<double*>(&d_peer_fn_z);
  deallocate_device<double*>(&d_peer_fs_x);
  deallocate_device<double*>(&d_peer_fs_y);
  deallocate_device<double*>(&d_peer_fs_z);
  
  deallocate_device<bool*>(&d_patchRecordHasForces);

  if (simParams->useDeviceMigration) {
    deallocate_device<FullAtom>(&d_atomdata_AoS_in);
    deallocate_device<int>(&d_sortSoluteIndex);
    if (d_migrationDestination != NULL) deallocate_device<int4>(&d_migrationDestination);
  }

  deallocate_device<PatchDataSOA>(&d_HostPatchDataSOA);
}

void SequencerCUDA::copyMigrationInfo(HomePatch *p, int patchIndex){
  CudaMInfo &m = mInfo[patchIndex];
  if (!p->patchMapRead) p->readPatchMap();
  for(int x = 0; x < 3; x++){
    for(int y = 0; y < 3; y++){
      for(int z = 0; z < 3; z++){
        // copies migration info over
        MigrationInfo *hm = p->mInfo[x][y][z];
        if(hm != NULL) m.destPatchID[x][y][z] = hm->destPatchID;
        else m.destPatchID[x][y][z] = -1; // let's flag this as -1 for now
      }
    }
  }
  if (simParams->useDeviceMigration) {
    m.destPatchID[1][1][1] = p->getPatchID();
  }
}

void SequencerCUDA::assembleOrderedPatchList(){
  // Assembles the patches on each Pe sharing a device into a device-ordered list
  patchList.clear();
  
  // Handle our own patches
  for (int i = 0; i < patchData->devData[deviceIndex].patches.size(); i++) {
    HomePatch *p = patchData->devData[deviceIndex].patches[i];
    patchList.push_back(p);
  }


  // Do we really need this?
#if 1
  patchListHomeAndProxy.clear();
  // Set up list of device patches. We need the home patches for everyone
  std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
  for (int i = 0; i < numPatchesHomeAndProxy; i++) {
    const int patchID = localPatches[i].patchID;


    for(int d = 0; d < CkNumPes(); d++){
      PatchMap* pm = PatchMap::ObjectOnPe(d);
      HomePatch *patch = pm->homePatch(patchID);
      if(patch != NULL) {
        patchListHomeAndProxy.push_back(patch);
      }
    }
  }

#endif
}

/**
 * \brief Copies atom data from device to host
 *
 * This will first update the position and velocities in the AoS structure
 * the AoS structs are then copied to the individual home patches
 *
 */
void SequencerCUDA::copyAoSDataToHost() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();

  std::vector<HomePatch*>& integrationPatches = patchData->devData[deviceIndex].patches;
  std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;

  CUDAMigrationKernel->update_AoS(
    numPatchesHome,
    patchData->devData[deviceIndex].d_localPatches,
    (FullAtom*) d_atomdata_AoS,
    d_vel_x, d_vel_y, d_vel_z,
    d_pos_x, d_pos_y, d_pos_z,
    stream
  );

  for (int i = 0; i < integrationPatches.size(); i++) {
    const int numAtoms = localPatches[i].numAtoms;
    const int offset = localPatches[i].bufferOffset;
    HomePatch *patch = integrationPatches[i];
    patch->updateAtomCount(numAtoms, false);
    patch->updateAtomBuffers();
    FullAtomList& h_atomdata = patch->getAtomList();
    copy_DtoH<FullAtom>(d_atomdata_AoS + offset, (FullAtom*)h_atomdata.begin(), numAtoms, stream);
  }
  cudaCheck(cudaStreamSynchronize(stream));
}


/**
 * \brief This function lauches the initial, local work for migration
 *
 * TODO learn how to link to the descriptions of these functions with doxygen... 
 *
 */
void SequencerCUDA::migrationLocalInit() {
  CUDAMigrationKernel->computeMigrationGroupIndex(
    numPatchesHome,
    patchData->devData[deviceIndex].d_localPatches,
    d_migrationGroupSize,
    d_migrationGroupIndex,
    stream
  );
  
  CUDAMigrationKernel->update_AoS(
    numPatchesHome,
    patchData->devData[deviceIndex].d_localPatches,
    (FullAtom*) d_atomdata_AoS,
    d_vel_x, d_vel_y, d_vel_z,
    d_pos_x, d_pos_y, d_pos_z,
    stream
  );

  CUDAMigrationKernel->computeMigrationDestination(
    numPatchesHome,
    patchData->devData[deviceIndex].d_localPatches,
    myLattice,
    d_mInfo,
    d_patchToDeviceMap,
    d_globalToLocalID,
    d_patchMin,
    d_patchMax,
    d_hydrogenGroupSize,
    d_migrationGroupSize,
    d_migrationGroupIndex,
    d_pos_x, d_pos_y, d_pos_z,
    d_migrationDestination,
    stream
  );

  CUDAMigrationKernel->performLocalMigration(
    numPatchesHome,
    patchData->devData[deviceIndex].d_localPatches,
    (FullAtom*) d_atomdata_AoS,
    (FullAtom*) d_atomdata_AoS_in,
    d_migrationDestination,
    stream
  );

  cudaCheck(cudaStreamSynchronize(stream));
}

/**
 * \brief Moves the migrating atoms
 *
 */
void SequencerCUDA::migrationPerform() {
  CUDAMigrationKernel->performMigration(
    numPatchesHome,
    patchData->devData[deviceIndex].d_localPatches,
    d_peer_record,
    (FullAtom*) d_atomdata_AoS,
    d_peer_atomdata,
    d_migrationGroupSize,
    d_migrationGroupIndex,
    d_migrationDestination,
    stream
  );
  cudaCheck(cudaStreamSynchronize(stream));
}


void SequencerCUDA::migrationUpdateAtomCounts() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();

  CUDAMigrationKernel->updateLocalRecords(
    numPatchesHome,
    patchData->devData[deviceIndex].d_localPatches,
    d_peer_record,
    patchData->devData[deviceIndex].d_peerPatches,
    stream
  );

  cudaCheck(cudaStreamSynchronize(stream));
}

void SequencerCUDA::migrationUpdateAtomOffsets() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();
  
  CUDAMigrationKernel->updateLocalRecordsOffset(
    numPatchesHomeAndProxy,
    patchData->devData[deviceIndex].d_localPatches,
    stream
  );

  cudaCheck(cudaStreamSynchronize(stream));
}

void SequencerCUDA::migrationUpdateRemoteOffsets() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();

  CUDAMigrationKernel->updatePeerRecords(
    numPatchesHomeAndProxy,
    patchData->devData[deviceIndex].d_localPatches,
    d_peer_record,
    patchData->devData[deviceIndex].d_peerPatches,
    stream
  );

  cudaCheck(cudaStreamSynchronize(stream));
}

void SequencerCUDA::migrationUpdateProxyDestination() {
  if (mGpuOn) {
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    patchData = cpdata.ckLocalBranch();

    // This is implemented as a put instead of a get to avoid the need
    // for a node synchronization.
    CUDAMigrationKernel->copyMigrationDestinationToProxies(
      deviceIndex, 
      numPatchesHome,
      numPatchesHomeAndProxy,
      patchData->devData[deviceIndex].d_localPatches,
      patchData->devData[deviceIndex].d_peerPatches,
      d_peer_migrationDestination,
      stream 
    );
  }
}

void SequencerCUDA::copyPatchDataToHost() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();
  std::vector<HomePatch*>& integrationPatches = patchData->devData[deviceIndex].patches;

  std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
  const int numPatchesHomeAndProxy = patchData->devData[deviceIndex].numPatchesHomeAndProxy;

  copy_DtoH<CudaLocalRecord>(patchData->devData[deviceIndex].d_localPatches, localPatches.data(), numPatchesHomeAndProxy, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  

  // Update Atom Counts
  for (int i = 0; i < numPatchesHome; i++) {
    HomePatch* hp = integrationPatches[i];
    hp->updateAtomCount(localPatches[i].numAtoms, false);
  }
  cudaCheck(cudaStreamSynchronize(stream));
}

// This code path is used only by device migration
void SequencerCUDA::copyAtomDataToDeviceAoS() {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();

  std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
  const int numPatchesHomeAndProxy = patchData->devData[deviceIndex].numPatchesHomeAndProxy;
  std::vector<HomePatch*>& integrationPatches = patchData->devData[deviceIndex].patches;


  for (int i = 0; i < integrationPatches.size(); i++) {
    const int numAtoms = localPatches[i].numAtoms;
    if (numAtoms > MigrationCUDAKernel::kMaxAtomsPerPatch) {
      iout << iERROR << "The number of atoms in patch " << i << " is "
           << numAtoms << ", greater than the limit for GPU atom migration ("
           << MigrationCUDAKernel::kMaxAtomsPerPatch << ").\n" << endi;
      NAMD_bug("NAMD has stopped simulating due to the error above, "
               "but you could disable GPUAtomMigration and try again.\n");
    }
    const int offset = localPatches[i].bufferOffset;
    HomePatch *patch = integrationPatches[i];
    FullAtomList& h_atomdata = patch->getAtomList();
    copy_HtoD<FullAtom>((FullAtom*)h_atomdata.begin(), d_atomdata_AoS_in + ((int64_t) i) * MigrationCUDAKernel::kMaxAtomsPerPatch, numAtoms, stream);
  }
  cudaCheck(cudaStreamSynchronize(stream));
}

/*
 * Aggregates and copys various data from the host to device
 *
 * Some data is only copied for home patches, while other data is copied for
 * home and proxy patches
 *
 */
void SequencerCUDA::copyAtomDataToDevice(bool copyForces, int maxForceNumber) {

 AGGREGATE_HOME_ATOMS_TO_DEVICE(recipMass, double, stream);
  if(copyForces){
    switch (maxForceNumber) {
      case 2:
        AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(f_slow_x, double, stream);
        AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(f_slow_y, double, stream);
        AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(f_slow_z, double, stream);
      case 1:
        AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(f_nbond_x, double, stream);
        AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(f_nbond_y, double, stream);
        AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(f_nbond_z, double, stream);
      case 0:
        AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(f_normal_x, double, stream);
        AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(f_normal_y, double, stream);
        AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(f_normal_z, double, stream);
    }
  }

  AGGREGATE_HOME_ATOMS_TO_DEVICE(vel_x, double, stream);
  AGGREGATE_HOME_ATOMS_TO_DEVICE(vel_y, double, stream);
  AGGREGATE_HOME_ATOMS_TO_DEVICE(vel_z, double, stream);
  AGGREGATE_HOME_ATOMS_TO_DEVICE(pos_x, double, stream);
  AGGREGATE_HOME_ATOMS_TO_DEVICE(pos_y, double, stream);
  AGGREGATE_HOME_ATOMS_TO_DEVICE(pos_z, double, stream);
  AGGREGATE_HOME_ATOMS_TO_DEVICE(mass, float, stream);
  AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(charge, float, stream);
  if (simParams->alchOn) {
    AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(partition, int, stream);
  }
  
  if (simParams->langevinOn) {
    AGGREGATE_HOME_ATOMS_TO_DEVICE(langevinParam, float, stream);
    AGGREGATE_HOME_ATOMS_TO_DEVICE(langScalVelBBK2, float, stream);
    AGGREGATE_HOME_ATOMS_TO_DEVICE(langScalRandBBK2, float, stream);
  }

  AGGREGATE_HOME_ATOMS_TO_DEVICE(hydrogenGroupSize, int, stream);
  AGGREGATE_HOME_ATOMS_TO_DEVICE(atomFixed, int, stream);
  AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(sortOrder, int, stream);
  AGGREGATE_HOME_AND_PROXY_ATOMS_TO_DEVICE(unsortOrder, int, stream);
  AGGREGATE_HOME_ATOMS_TO_DEVICE(rigidBondLength, float, stream);

  if (simParams->monteCarloPressureOn) {
    AGGREGATE_HOME_ATOMS_TO_DEVICE(id, int, stream);
    //set up initial mapping for global index to local array index
    CUDASequencerKernel->SetAtomIndexOrder(d_id, d_idOrder, numAtomsHome, stream);
  }
}


void SequencerCUDA::migrationLocalPost(int startup) {
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();

  if (simParams->useDeviceMigration) {
    if (!startup) {
      CUDAMigrationKernel->transformMigratedPositions(
        numPatchesHome,
        patchData->devData[deviceIndex].d_localPatches,
        d_patchCenter,
        (FullAtom*) d_atomdata_AoS_in,
        myLattice,
        stream
      );
    }

    // sort solvent/solute data
    CUDAMigrationKernel->sortSolventAtoms(
      numPatchesHome,
      patchData->devData[deviceIndex].d_localPatches,
      (FullAtom*) d_atomdata_AoS_in,
      (FullAtom*) d_atomdata_AoS,
      d_sortSoluteIndex,
      stream
    );

    double dt = 1.0;
    double kbT = 1.0;
    double tempFactor = 1.0;
    if (simParams->langevinOn) {
      dt = simParams->dt *  0.001;  // convert timestep to ps
      kbT = BOLTZMANN * simParams->langevinTemp;
      int lesReduceTemp = (simParams->lesOn && simParams->lesReduceTemp);
      tempFactor = (lesReduceTemp ? 1. / simParams->lesFactor : 1);
    }
    CUDAMigrationKernel->copy_AoS_to_SoA(
      numPatchesHome, simParams->alchOn, 
      simParams->langevinOn, dt, kbT, tempFactor,
      patchData->devData[deviceIndex].d_localPatches,
      (FullAtom*) d_atomdata_AoS,
      d_recipMass,
      d_vel_x, d_vel_y, d_vel_z,
      d_pos_x, d_pos_y, d_pos_z,
      d_mass, d_charge,
      d_idMig, d_vdwType,
      d_hydrogenGroupSize, d_migrationGroupSize,
      d_atomFixed,
      d_rigidBondLength,
      d_transform,
      d_partition,
      d_langevinParam,
      d_langScalVelBBK2,
      d_langScalRandBBK2,
      stream
    );
  }

  // Other migration post processing steps
  if (simParams->useDeviceMigration) {
    if (simParams->monteCarloPressureOn) {
      CUDASequencerKernel->SetAtomIndexOrder(d_idMig, d_idOrder, numAtomsHome, stream);
    }
  }


  // JM: Saving position to these doubles for pairListCheck
  copy_DtoD<double>(d_pos_x, d_posSave_x, numAtomsHome, stream);
  copy_DtoD<double>(d_pos_y, d_posSave_y, numAtomsHome, stream);
  copy_DtoD<double>(d_pos_z, d_posSave_z, numAtomsHome, stream);

  // JM NOTE: We need to save the lattice at the beggining of the cycle
  //          in order to use it in SequencerCUDAKernel::pairlistCheck();
  myLatticeOld = myLattice;
  
  // JM NOTE: Recalculates the centers of mass since we have new positions
  CUDASequencerKernel->centerOfMass(
      d_pos_x, d_pos_y, d_pos_z, 
      d_rcm_x, d_rcm_y, d_rcm_z,
      d_mass, d_hydrogenGroupSize, numAtomsHome, stream);
  CUDASequencerKernel->centerOfMass(
      d_vel_x, d_vel_y, d_vel_z, 
      d_vcm_x, d_vcm_y, d_vcm_z,
      d_mass, d_hydrogenGroupSize, numAtomsHome, stream);

  cudaCheck(cudaStreamSynchronize(stream));
}

void SequencerCUDA::migrationUpdateAdvancedFeatures(const int startup) {
  if(simParams->eFieldOn || simParams->SMDOn || simParams->groupRestraintsOn ||
    simParams->monteCarloPressureOn){

    // Handcopies the transform field in a char* SOA
    // JM NOTE: We're copying ints into char because "transform" is an int
    //          field in PatchDataSOA but a signed char in FullAtom
    size_t offset = 0;
    for (int i = 0; i < numPatchesHome; i++) {
      PatchDataSOA& current = patchList[i]->patchDataSOA;
      const int numPatchAtoms = current.numAtoms;
      // memcpy(fieldName + offset, current.fieldName, numPatchAtoms*sizeof(type));
      for(int j = 0; j < numPatchAtoms; j++){
        transform[offset + j].x = current.transform_i[j];
        transform[offset + j].y = current.transform_j[j];
        transform[offset + j].z = current.transform_k[j];
      }
      offset += numPatchAtoms;
    }
    copy_HtoD<char3>(transform, d_transform, numAtomsHome, stream);
  }

  if (!startup) {
    if(simParams->constraintsOn) {
      restraintsKernel->updateRestrainedAtoms(atomMapList, patchData->devData[deviceIndex].h_localPatches, globalToLocalID);
    }
    if(simParams->SMDOn) {
      SMDKernel->updateAtoms(atomMapList, patchData->devData[deviceIndex].h_localPatches, globalToLocalID);
    }
    if(simParams->groupRestraintsOn) {
      groupRestraintsKernel->updateAtoms(atomMapList, patchData->devData[deviceIndex].h_localPatches, globalToLocalID);
    }
  }
}

void SequencerCUDA::migrationUpdateDestination() {
  CUDAMigrationKernel->updateMigrationDestination(
    numAtomsHomePrev,
    d_migrationDestination,
    d_peer_sortSoluteIndex,
    stream
  );
}

bool SequencerCUDA::copyPatchData(
  const bool copyIn, 
  const bool startup
) {
  bool reallocated = false;
  if (copyIn) {
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    patchData = cpdata.ckLocalBranch();

    std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;

    std::vector<CudaPeerRecord>& peerPatches = patchData->devData[deviceIndex].h_peerPatches;
    std::vector<HomePatch*>& homePatches = patchData->devData[deviceIndex].patches;

    if (startup) {
      // numPatchesHomeAndProxy is set when the patch data is constructed
      numPatchesHomeAndProxy = patchData->devData[deviceIndex].numPatchesHomeAndProxy;
      numPatchesHome = homePatches.size();
      patchData->devData[deviceIndex].numPatchesHome = numPatchesHome;

      if (simParams->useDeviceMigration) {
        // Padded data structures which will not be reallocated
        allocate_device<FullAtom>(&d_atomdata_AoS_in, ((int64_t) numPatchesHome) * MigrationCUDAKernel::kMaxAtomsPerPatch);
        allocate_device<int>(&d_sortSoluteIndex, numPatchesHome * MigrationCUDAKernel::kMaxAtomsPerPatch);
        d_migrationDestination = NULL;
      }

      allocate_device<CudaLocalRecord>(&patchData->devData[deviceIndex].d_localPatches, numPatchesHomeAndProxy);
      allocate_device<CudaPeerRecord>(&patchData->devData[deviceIndex].d_peerPatches, peerPatches.size());
      if (simParams->useDeviceMigration) {
        CUDAMigrationKernel->allocateScratch(numPatchesHomeAndProxy);
      }

      copy_HtoD<CudaLocalRecord>(localPatches.data(), patchData->devData[deviceIndex].d_localPatches, 
                                 numPatchesHomeAndProxy, stream);
      copy_HtoD<CudaPeerRecord>(peerPatches.data(), patchData->devData[deviceIndex].d_peerPatches, 
                                  peerPatches.size(), stream);
      if(true || mGpuOn) {
        this->assembleOrderedPatchList();
      }
      this->copySettleParameter();

      for (int i = 0; i < numPatchesHome; i++) {
        HomePatch *patch = homePatches[i];
        this->copyMigrationInfo(patch, i);
        patchNewTolerance[i] = 0.5 * ( simParams->pairlistDist - simParams->cutoff);

        globalToLocalID[patch->getPatchID()] = i;
        patchToDeviceMap[patch->getPatchID()] = deviceIndex;
      }
      copy_HtoD<double>(patchNewTolerance, d_patchNewTolerance, numPatchesHome, stream);
      copy_HtoD<CudaMInfo>(mInfo, d_mInfo, numPatchesHome, stream);

      // Need the globalToLocalID and patchToDeviceMap data structures to be system wide for migration
      // They are also used in tuple migration, so we add them to patchData, so they can be easily
      // accessed elsewhere
      for (int i = 0; i < deviceCUDA->getNumDevice(); i++) {
        if (i == deviceIndex) continue;
        std::vector<HomePatch*>& otherPatches = patchData->devData[i].patches;
        for (int j = 0; j < otherPatches.size(); j++) {
          HomePatch *patch = otherPatches[j];
          globalToLocalID[patch->getPatchID()] = j;
          patchToDeviceMap[patch->getPatchID()] = i;
        }
      }
      copy_HtoD<int>(globalToLocalID, d_globalToLocalID, numPatchesGlobal, stream);
      copy_HtoD<int>(patchToDeviceMap, d_patchToDeviceMap, numPatchesGlobal, stream);
      patchData->devData[deviceIndex].d_globalToLocalID = d_globalToLocalID;
      patchData->devData[deviceIndex].d_patchToDeviceMap = d_patchToDeviceMap;

      // Allocate more data
      allocate_device<PatchDataSOA>(&d_HostPatchDataSOA, numPatchesHome);
    }

    for (int i = 0; i < numPatchesHomeAndProxy; i++) {
      HomePatch *patch = patchListHomeAndProxy[i];
      awayDists[i].x = patch->aAwayDist;
      awayDists[i].y = patch->bAwayDist;
      awayDists[i].z = patch->cAwayDist;
      COPY_CUDAVECTOR(patch->center,   patchCenter[i]);
      COPY_CUDAVECTOR(patch->getMin(), patchMin[i]);
      COPY_CUDAVECTOR(patch->getMax(), patchMax[i]);
    }

    copy_HtoD<double3>(awayDists, d_awayDists, numPatchesHomeAndProxy, stream);
    copy_HtoD<double3>(patchMin, d_patchMin, numPatchesHomeAndProxy, stream);
    copy_HtoD<double3>(patchMax, d_patchMax, numPatchesHomeAndProxy, stream);
    copy_HtoD<double3>(patchCenter, d_patchCenter, numPatchesHomeAndProxy, stream);

    const int totalAtomCount = localPatches[numPatchesHomeAndProxy-1].bufferOffset +
                               localPatches[numPatchesHomeAndProxy-1].numAtoms;

    const int homeAtomCount = localPatches[numPatchesHome-1].bufferOffset +
                              localPatches[numPatchesHome-1].numAtoms;

    reallocated = reallocateArrays(homeAtomCount, totalAtomCount);
    

    numAtomsHomePrev = numAtomsHome;
    numAtomsHomeAndProxy = totalAtomCount;
    numAtomsHome = homeAtomCount;

    patchData->devData[deviceIndex].numAtomsHome = numAtomsHome;
    
    if (!startup) {
      copy_HtoD<CudaLocalRecord>(localPatches.data(), patchData->devData[deviceIndex].d_localPatches, 
                                 numPatchesHomeAndProxy, stream);
      copy_HtoD<CudaPeerRecord>(peerPatches.data(), patchData->devData[deviceIndex].d_peerPatches, 
                                  peerPatches.size(), stream);
    }
    if (startup) {
      if (simParams->monteCarloPressureOn) {
        //Only at startup, copy the molecule's index info to GPU
        Molecule *molecule = Node::Object()->molecule;
        copy_HtoD<int>(molecule->moleculeAtom, d_moleculeAtom, numAtomsHome, stream);
        copy_HtoD<int>(molecule->moleculeStartIndex, d_moleculeStartIndex, molecule->numMolecules + 1, stream);
      }
    }
  }
  return reallocated;
}

void SequencerCUDA::copyDataToPeers(
  const bool copyIn
) {
  if (!copyIn) return;
  // Positions will be copied by the kernel
  // Forces don't need to be copied
  // Atom data needed to be copied: sortOrder, unsortOrder

  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  patchData = cpdata.ckLocalBranch();
  if (mGpuOn) {
    CUDAMigrationKernel->copyDataToProxies(
      deviceIndex, 
      numPatchesHome,
      numPatchesHomeAndProxy,
      patchData->devData[deviceIndex].d_localPatches,
      d_peer_id,
      d_peer_vdwType,
      d_peer_sortOrder,
      d_peer_unsortOrder,
      d_peer_charge,
      d_peer_partition,
      d_peer_patchCenter,
      simParams->alchOn,
      stream 
    );
  }
  cudaCheck(cudaStreamSynchronize(stream));
}

void SequencerCUDA::migrationSortAtomsNonbonded() {
  CUDAMigrationKernel->sortAtoms(
    numPatchesHome, numAtomsHome,
    patchData->devData[deviceIndex].d_localPatches,
    patchMin, patchMax,
    d_pos_x, d_pos_y, d_pos_z,
    d_sortOrder,
    d_unsortOrder,
    d_sortIndex,
    stream
  );
}

void SequencerCUDA::maximumMove(
  const double maxvel2,
  const int    numAtoms)
{
  CUDASequencerKernel->maximumMove(
    maxvel2, d_vel_x, d_vel_y, d_vel_z,
    killme, numAtoms, stream);
}

void SequencerCUDA::submitHalf(
  NodeReduction *reduction,
  int numAtoms, int part, const bool doEnergy)
{
  //BigReal kineticEnergy;
  Tensor reduction_virial;
  //cudaTensor h_virial;
  //BigReal intKineticEnergy;
  Tensor reduction_intVirialNormal;
  //cudaTensor h_intVirialNormal;
  int hgs;

  if(doEnergy){
#if 0
    cudaCheck(cudaEventRecord(eventStart,stream));
#endif
    CUDASequencerKernel->submitHalf(
      d_vel_x, d_vel_y, d_vel_z,
      d_vcm_x, d_vcm_y, d_vcm_z, d_mass,
      d_kineticEnergy, d_intKineticEnergy,
      d_virial, d_intVirialNormal, kineticEnergy_half, intKineticEnergy_half,
      virial_half, intVirialNormal_half,
      d_hydrogenGroupSize, numAtoms, d_tbcatomic, stream);
#if 0
    cudaCheck(cudaEventRecord(eventStop, stream));
    cudaCheck(cudaEventSynchronize(eventStop));
    cudaCheck(cudaEventElapsedTime(&t_submitHalf, eventStart, eventStop));
    fprintf(stderr, "submitHalf total elapsed time: %f\n", t_submitHalf);
    t_submitReductions2 = 0;
#endif
  }
}

void SequencerCUDA::submitReductions(
  NodeReduction *reduction,
  BigReal origin_x,
  BigReal origin_y,
  BigReal origin_z,
  int marginViolations,
  int doEnergy,
  int doMomentum, 
  int numAtomsReduction,
  int maxForceNumber)
{
  // reduction->item(REDUCTION_ATOM_CHECKSUM) += numAtomsReduction; //moved to launch_part2, startRun2
  // where do I get the margin violations?
  // reduction->item(REDUCTION_MARGIN_VIOLATIONS) += marginViolations;
  if(doEnergy){
    if(doMomentum){
      // JM NOTE: Calculates momenta if copyOut
      CUDASequencerKernel->submitReduction1(
        d_pos_x, d_pos_y, d_pos_z,
        d_vel_x, d_vel_y, d_vel_z, d_mass,
        d_kineticEnergy,
        d_momentum_x, d_momentum_y, d_momentum_z,
        d_angularMomentum_x, d_angularMomentum_y, d_angularMomentum_z,
        origin_x, origin_y, origin_z, kineticEnergy, momentum_x, momentum_y,
        momentum_z, angularMomentum_x, angularMomentum_y, angularMomentum_z, d_tbcatomic,
        numAtomsReduction, stream);
    }
    Tensor regintVirialNormal;
    Tensor regintVirialNbond;
    Tensor regintVirialSlow;

#if 0
    cudaCheck(cudaEventRecord(eventStart,stream));
#endif
    CUDASequencerKernel->submitReduction2(
        d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z,
        d_rcm_x, d_rcm_y, d_rcm_z, d_vcm_x, d_vcm_y, d_vcm_z,
        d_f_normal_x, d_f_normal_y, d_f_normal_z,
        d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
        d_f_slow_x, d_f_slow_y, d_f_slow_z,
        d_mass, d_hydrogenGroupSize,
        d_kineticEnergy, kineticEnergy,
        d_intKineticEnergy, intKineticEnergy,
        d_intVirialNormal, d_intVirialNbond, d_intVirialSlow,
        intVirialNormal, intVirialNbond, intVirialSlow, d_rigidVirial, rigidVirial,
        d_tbcatomic, numAtomsReduction, maxForceNumber, stream);
  }
#if 0
    cudaCheck(cudaEventRecord(eventStop, stream));
    cudaCheck(cudaEventSynchronize(eventStop));
    cudaCheck(cudaEventElapsedTime(&t_submitReductions2, eventStart, eventStop));
    fprintf(stderr, "submitReductions2 total elapsed time: %f\n", t_submitReductions2);
    t_submitReductions2 = 0;
#endif
}

void SequencerCUDA::copySettleParameter(){
  // Searching for a patch that contains initialized settle parameters
  cudaCheck(cudaSetDevice(deviceID));
  if(simParams->rigidBonds != RIGID_NONE){
    HomePatch *patch = NULL;
    // PatchList contains all patches in the node, so if there's a single water in the system,
    // this is guaranteed to catch it
    for(int i = 0; i < patchList.size(); i++){
      if(patchList[i]->settle_initialized) {
        patch = patchList[i];
        break;
      }
    }
    if ( patch ) {
      SettleParameters h_sp;
      h_sp.mO = patch->settle_mO;
      h_sp.mH = patch->settle_mH;
      h_sp.mOrmT = patch->settle_mOrmT;
      h_sp.mHrmT = patch->settle_mHrmT;
      h_sp.rra = patch->settle_rra;
      h_sp.ra = patch->settle_ra;
      h_sp.rb = patch->settle_rb;
      h_sp.rc = patch->settle_rc;
      h_sp.r_om = patch->r_om;
      h_sp.r_ohc = patch->r_ohc;

      // fprintf(stderr, "SETTLEPARAMETER Found: Values %lf %lf %lf %lf %lf %lf %lf %lf\n",
      //    h_sp.mO, h_sp.mH, h_sp.mOrmT, h_sp.mHrmT, h_sp.rra, h_sp.ra, h_sp.rb, h_sp.rc );
      copy_HtoD<SettleParameters>(&h_sp, this->sp, 1, stream);
    }
  }
}

// Does rattle1_SOA
void SequencerCUDA::startRun1(
  int maxForceNumber,
  const Lattice& lattice
) {
  // cudaCheck(cudaSetDevice(deviceID));
  myLattice = lattice;

  // JM: Enforcing rigid bonds on first iteration
  CUDASequencerKernel->rattle1(1, 0, 
      numAtomsHome, 0.f, 0.f,
      2.0 * simParams->rigidTol,
      d_vel_x, d_vel_y, d_vel_z,
      d_pos_x, d_pos_y, d_pos_z,
      d_velNew_x, d_velNew_y, d_velNew_z,
      d_posNew_x, d_posNew_y, d_posNew_z,
      d_f_normal_x, d_f_normal_y, d_f_normal_z,
      d_hydrogenGroupSize, d_rigidBondLength, d_mass, d_atomFixed,
      &settleList, settleListSize, &d_consFailure, 
      d_consFailureSize, &rattleList, rattleListSize,
      &nSettle, &nRattle,
      d_rigidVirial, rigidVirial, d_tbcatomic, 1, sp,
      buildRigidLists, consFailure, simParams->watmodel, stream);

  this->copyPositionsAndVelocitiesToHost(1,0);
  cudaCheck(cudaDeviceSynchronize());
#if 0
  printSOAPositionsAndVelocities();
#endif
}

void SequencerCUDA::startRun2(
  double dt_normal,
  double dt_nbond,
  double dt_slow,
  Vector origin,
  NodeReduction *reduction,
  int doGlobal,
  int maxForceNumber
){
  reduction->item(REDUCTION_ATOM_CHECKSUM) += numAtomsHome;

  // This is a patch-based kernel -> which means each threadblock deals with an entire patch
  // So, in the multigpu case, we want to keep non-offset pointers but we want
  //     to deal only with a handful of patches
  
  // We dont need to and we should not set normal force to zero
  // It stores global forces. Additionally, we set normall forces, 
  // every step, it's not an addition
  //  cudaCheck(cudaMemset(d_f_normal_x, 0, sizeof(double)*numAtomsHomeAndProxy));
  // cudaCheck(cudaMemset(d_f_normal_y, 0, sizeof(double)*numAtomsHomeAndProxy));
  // cudaCheck(cudaMemset(d_f_normal_z, 0, sizeof(double)*numAtomsHomeAndProxy));

  cudaCheck(cudaMemset(d_f_nbond_x, 0, sizeof(double)*numAtomsHomeAndProxy));
  cudaCheck(cudaMemset(d_f_nbond_y, 0, sizeof(double)*numAtomsHomeAndProxy));
  cudaCheck(cudaMemset(d_f_nbond_z, 0, sizeof(double)*numAtomsHomeAndProxy));

  cudaCheck(cudaMemset(d_f_slow_x, 0, sizeof(double)*numAtomsHomeAndProxy));
  cudaCheck(cudaMemset(d_f_slow_y, 0, sizeof(double)*numAtomsHomeAndProxy));
  cudaCheck(cudaMemset(d_f_slow_z, 0, sizeof(double)*numAtomsHomeAndProxy));
  
  CUDASequencerKernel->accumulateForceToSOA(
      doGlobal,					    
      maxForceNumber,
      numPatchesHomeAndProxy,
      nDevices,
      patchData->devData[deviceIndex].d_localPatches,
      patchData->devData[deviceIndex].f_bond,
      patchData->devData[deviceIndex].f_bond_nbond,
      patchData->devData[deviceIndex].f_bond_slow,
      patchData->devData[deviceIndex].forceStride,
      patchData->devData[deviceIndex].f_nbond,
      patchData->devData[deviceIndex].f_nbond_slow,
      patchData->devData[deviceIndex].f_slow,
      d_f_global_x,
      d_f_global_y,
      d_f_global_z,
      d_f_normal_x,
      d_f_normal_y,
      d_f_normal_z,
      d_f_nbond_x,
      d_f_nbond_y,
      d_f_nbond_z,
      d_f_slow_x,
      d_f_slow_y,
      d_f_slow_z,
      d_unsortOrder,
      myLattice,
      patchData->d_queues, 
      patchData->d_queueCounters, 
      d_tbcatomic, 
      stream
  );
  if(mGpuOn){
    // Synchonize device before node barrier
    cudaCheck(cudaDeviceSynchronize());
  }

#if 0
  printSOAPositionsAndVelocities();
#endif
}

void SequencerCUDA::startRun3(
  double dt_normal,
  double dt_nbond, 
  double dt_slow, 
  Vector origin, 
  NodeReduction *reduction,
  int forceRequested,
  int maxForceNumber
){
  if(mGpuOn){
    // XXX TODO we need to call the force merging kernel here
#if 1
    // JM - Awful: We need to busy wait inside accumulateForceToSOA instead
    std::vector<int> atom_counts;
    for (int i = 0; i < deviceCUDA->getDeviceCount(); i++) {
      atom_counts.push_back(patchData->devData[i].numAtomsHome);
    }
    CUDASequencerKernel->mergeForcesFromPeers(
      deviceIndex, 
      maxForceNumber, 
      myLattice, 
      numPatchesHomeAndProxy,
      numPatchesHome, 
      this->d_peer_fb_x, 
      this->d_peer_fb_y, 
      this->d_peer_fb_z,
      this->d_peer_fn_x,
      this->d_peer_fn_y,
      this->d_peer_fn_z,
      this->d_peer_fs_x,
      this->d_peer_fs_y,
      this->d_peer_fs_z,
      // patchData->devData[deviceCUDA->getPmeDevice()].f_slow,
      patchData->devData[deviceCUDA->getPmeDeviceIndex()].f_slow,
      patchData->devData[deviceIndex].d_localPatches,
      patchData->devData[deviceIndex].d_peerPatches,
      atom_counts,
      stream
    );
#else

  // Before I call nccl, let's see the forces here
  // ncclAllReduce(d_f_normal_x, d_f_normal_x, numAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream);

  // ncclAllReduce(d_f_normal_y, d_f_normal_y, numAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream);
  // ncclAllReduce(d_f_normal_z, d_f_normal_z, numAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream);
  // ncclAllReduce(d_f_nbond_x,   d_f_nbond_x, numAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream);
  // ncclAllReduce(d_f_nbond_y,   d_f_nbond_y, numAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream);
  // ncclAllReduce(d_f_nbond_z,   d_f_nbond_z, numAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream);
  // ncclAllReduce(d_f_slow_x,     d_f_slow_x, numAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream);
  // ncclAllReduce(d_f_slow_y,     d_f_slow_y, numAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream);
  // ncclAllReduce(d_f_slow_z,     d_f_slow_z, numAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream);
  int numReducedAtoms = (3 * (maxForceNumber+1)) * numAtoms;
  ncclAllReduce(d_f_raw, d_f_raw, numReducedAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream );
#endif
  }

#if 0
  cudaCheck(cudaDeviceSynchronize());
  if(true || deviceID == 0){ 
    this->printSOAForces();
  }
#endif

  // do external forces calculation and store energy and virial
  calculateExternalForces(simParams->firstTimestep, reduction, maxForceNumber, 1, 1);

  CUDASequencerKernel->addForceToMomentum(
      -0.5, dt_normal, dt_nbond, dt_slow, 1.0,
      d_recipMass,
      d_f_normal_x, d_f_normal_y, d_f_normal_z,
      d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
      d_f_slow_x, d_f_slow_y, d_f_slow_z,
      d_vel_x, d_vel_y, d_vel_z,
      numAtomsHome, maxForceNumber, stream);

  CUDASequencerKernel->rattle1(1, 0, 
      numAtomsHome, -dt_normal, -1.0/(dt_normal), 
      2.0 * simParams->rigidTol,
      d_vel_x, d_vel_y, d_vel_z,
      d_pos_x, d_pos_y, d_pos_z,
      d_velNew_x, d_velNew_y, d_velNew_z,
      d_posNew_x, d_posNew_y, d_posNew_z,
      d_f_normal_x, d_f_normal_y, d_f_normal_z,
      d_hydrogenGroupSize, d_rigidBondLength, d_mass, d_atomFixed,
      &settleList, settleListSize, &d_consFailure, 
      d_consFailureSize, &rattleList, rattleListSize,
      &nSettle, &nRattle,
      d_rigidVirial, rigidVirial, d_tbcatomic, true, sp,
      true, consFailure, simParams->watmodel, stream);

  CUDASequencerKernel->centerOfMass(
      d_vel_x, d_vel_y, d_vel_z,
      d_vcm_x, d_vcm_y, d_vcm_z, d_mass, 
      d_hydrogenGroupSize, numAtomsHome, stream);


  {
    // SubmitHalf and its corresponding reductions
    submitHalf(reduction, numAtomsHome, 1, 1);
    // submitHalf reductions
    cudaCheck(cudaStreamSynchronize(stream));
    Tensor reduction_virial;
    Tensor reduction_intVirialNormal;
    COPY_CUDATENSOR(virial_half[0], reduction_virial);
    COPY_CUDATENSOR(intVirialNormal_half[0], reduction_intVirialNormal);
    reduction->item(REDUCTION_HALFSTEP_KINETIC_ENERGY) += (kineticEnergy_half[0] * 0.25);
    tensor_enforce_symmetry(reduction_virial);
    reduction_virial *= 0.5;
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,reduction_virial);
    // fprintf(stderr, "GPU calculated internal kinetic energy = %lf\n", intKineticEnergy_half);
    reduction->item(REDUCTION_INT_HALFSTEP_KINETIC_ENERGY)
      += (intKineticEnergy_half[0] * 0.25);
    reduction_intVirialNormal *= 0.5;
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NORMAL,
                        reduction_intVirialNormal);
  }

  CUDASequencerKernel->addForceToMomentum(
      1.0, dt_normal, dt_nbond, dt_slow, 1.0,
      d_recipMass,
      d_f_normal_x, d_f_normal_y, d_f_normal_z,
      d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
      d_f_slow_x, d_f_slow_y, d_f_slow_z,
      d_vel_x, d_vel_y, d_vel_z,
      numAtomsHome, maxForceNumber, stream);

   CUDASequencerKernel->rattle1(1, 1, 
      numAtomsHome, dt_normal, 1.0/dt_normal, 
      2.0 * simParams->rigidTol,
      d_vel_x, d_vel_y, d_vel_z,
      d_pos_x, d_pos_y, d_pos_z,
      d_velNew_x, d_velNew_y, d_velNew_z,
      d_posNew_x, d_posNew_y, d_posNew_z,
      d_f_normal_x, d_f_normal_y, d_f_normal_z,
      d_hydrogenGroupSize, d_rigidBondLength, d_mass, d_atomFixed,
      &settleList, settleListSize, &d_consFailure, 
      d_consFailureSize, &rattleList, rattleListSize,
      &nSettle, &nRattle,
      d_rigidVirial, rigidVirial, d_tbcatomic, 1, sp,
      buildRigidLists, consFailure, simParams->watmodel, stream);

  CUDASequencerKernel->centerOfMass(
      d_vel_x, d_vel_y, d_vel_z,
      d_vcm_x, d_vcm_y, d_vcm_z, d_mass, 
      d_hydrogenGroupSize, numAtomsHome, stream);

  {
    // JM: SubmitHalf and its corresponding reductions
    submitHalf(reduction, numAtomsHome, 1, 1);
    // submitHalf reductions
    cudaCheck(cudaStreamSynchronize(stream));
    Tensor reduction_virial;
    Tensor reduction_intVirialNormal;
    COPY_CUDATENSOR(virial_half[0], reduction_virial);
    COPY_CUDATENSOR(intVirialNormal_half[0], reduction_intVirialNormal);
    reduction->item(REDUCTION_HALFSTEP_KINETIC_ENERGY) += (kineticEnergy_half[0] * 0.25);
    tensor_enforce_symmetry(reduction_virial);
    reduction_virial *= 0.5;
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,reduction_virial);
    // fprintf(stderr, "GPU calculated internal kinetic energy = %lf\n", intKineticEnergy_half);
    reduction->item(REDUCTION_INT_HALFSTEP_KINETIC_ENERGY)
      += (intKineticEnergy_half[0] * 0.25);
    reduction_intVirialNormal *= 0.5;
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NORMAL,
                        reduction_intVirialNormal);
  }

  CUDASequencerKernel->addForceToMomentum(
      -0.5, dt_normal, dt_nbond, dt_slow, 1.0,
      d_recipMass,
      d_f_normal_x, d_f_normal_y, d_f_normal_z,
      d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
      d_f_slow_x, d_f_slow_y, d_f_slow_z,
      d_vel_x, d_vel_y, d_vel_z,
      numAtomsHome, maxForceNumber, stream);

  if(forceRequested) {
    // store the forces for next step, 
    // when we need it for colvars and Tcl scripting
    saveForceCUDASOA_direct(maxForceNumber);
  }

  CUDASequencerKernel->centerOfMass(
      d_vel_x, d_vel_y, d_vel_z,
      d_vcm_x, d_vcm_y, d_vcm_z, d_mass, 
      d_hydrogenGroupSize, numAtomsHome, stream);
  
  submitReductions(reduction, origin.x, origin.y, origin.z,
                   marginViolations, 1, 
                   1, 
                   numAtomsHome, maxForceNumber);

  copyPositionsAndVelocitiesToHost(1,0);

  if(consFailure[0]){
      // Constraint failure. Abort.
    int dieOnError = simParams->rigidDie;
    if(dieOnError){
      // Bails out
      //iout << iWARN << "constraint failure during GPU integration \n" << endi;
      NAMD_die("constraint failure during CUDA rattle!\n");
    }else{
      iout << iWARN << "constraint failure during CUDA rattle!\n" << endi;
    }
  }else if(1){
    cudaCheck(cudaStreamSynchronize(stream));
    if(simParams->rigidBonds != RIGID_NONE){
      Tensor reduction_rigidVirial;
      COPY_CUDATENSOR(rigidVirial[0], reduction_rigidVirial);
      tensor_enforce_symmetry(reduction_rigidVirial);
      ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL, reduction_rigidVirial);
    }

    //submitReductions1
    reduction->item(REDUCTION_CENTERED_KINETIC_ENERGY) += (kineticEnergy[0] * 0.5);
    Vector momentum(*momentum_x, *momentum_y, *momentum_z);
    ADD_VECTOR_OBJECT(reduction,REDUCTION_MOMENTUM,momentum);
    Vector angularMomentum(*angularMomentum_x,
                          *angularMomentum_y,
                          *angularMomentum_z);
    ADD_VECTOR_OBJECT(reduction,REDUCTION_ANGULAR_MOMENTUM,angularMomentum);
    //submitReductions2
    Tensor regintVirialNormal;
    Tensor regintVirialNbond;
    Tensor regintVirialSlow;
    COPY_CUDATENSOR(intVirialNormal[0], regintVirialNormal);
    if (maxForceNumber >= 1) {
      COPY_CUDATENSOR(intVirialNbond[0],  regintVirialNbond);
    }
    if (maxForceNumber >= 2) {
      COPY_CUDATENSOR(intVirialSlow[0],   regintVirialSlow);
    }

    reduction->item(REDUCTION_INT_CENTERED_KINETIC_ENERGY) += (intKineticEnergy[0] * 0.5);
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NORMAL, regintVirialNormal);
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NBOND,  regintVirialNbond);
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_SLOW,   regintVirialSlow);
  }

#if 0
  if(deviceID == 0){
    this->printSOAForces();
  }
#endif

#if 0
  printSOAPositionsAndVelocities();
#endif
}

void SequencerCUDA::monteCarloPressure_reject(Lattice &lattice)
{
  // Restore the myLattice
  myLattice = lattice;
  double *temp;

  // Restore positions and forces
  temp = d_f_normal_x; d_f_normal_x = d_f_normalMC_x; d_f_normalMC_x = temp;
  temp = d_f_normal_y; d_f_normal_y = d_f_normalMC_y; d_f_normalMC_y = temp;
  temp = d_f_normal_z; d_f_normal_z = d_f_normalMC_z; d_f_normalMC_z = temp;
  temp = d_f_nbond_x; d_f_nbond_x = d_f_nbondMC_x; d_f_nbondMC_x = temp;
  temp = d_f_nbond_y; d_f_nbond_y = d_f_nbondMC_y; d_f_nbondMC_y = temp;
  temp = d_f_nbond_z; d_f_nbond_z = d_f_nbondMC_z; d_f_nbondMC_z = temp;
  temp = d_f_slow_x; d_f_slow_x = d_f_slowMC_x; d_f_slowMC_x = temp;
  temp = d_f_slow_y; d_f_slow_y = d_f_slowMC_y; d_f_slowMC_y = temp;
  temp = d_f_slow_z; d_f_slow_z = d_f_slowMC_z; d_f_slowMC_z = temp;
  #ifdef NAMD_NCCL_ALLREDUCE
  if(mGpuOn) {
    temp = d_posNew_x; d_posNew_x = d_posMC_x; d_posMC_x = temp;
    temp = d_posNew_y; d_posNew_y = d_posMC_y; d_posMC_y = temp;
    temp = d_posNew_z; d_posNew_z = d_posMC_z; d_posMC_z = temp;
  } else {
    temp = d_pos_x; d_pos_x = d_posMC_x; d_posMC_x = temp;
    temp = d_pos_y; d_pos_y = d_posMC_y; d_posMC_y = temp;
    temp = d_pos_z; d_pos_z = d_posMC_z; d_posMC_z = temp;
  }
  #else
  temp = d_pos_x; d_pos_x = d_posMC_x; d_posMC_x = temp;
  temp = d_pos_y; d_pos_y = d_posMC_y; d_posMC_y = temp;
  temp = d_pos_z; d_pos_z = d_posMC_z; d_posMC_z = temp;
  #endif
}

void SequencerCUDA::monteCarloPressure_accept(
  NodeReduction *reduction,
  const int doMigration)
{
  // do we need to update center of masses?
  CUDASequencerKernel->centerOfMass(
    d_pos_x, d_pos_y, d_pos_z,
    d_rcm_x, d_rcm_y, d_rcm_z, d_mass, 
    d_hydrogenGroupSize, numAtomsHome, stream);

  // Add half step kinetic contribution to energy, intEnergy, virial, intVirial, 
  // calculated by submitHalf in launch_part11
  Tensor reduction_virial;
  Tensor reduction_intVirialNormal;
  COPY_CUDATENSOR(virial_half[0], reduction_virial);
  COPY_CUDATENSOR(intVirialNormal_half[0], reduction_intVirialNormal);
  tensor_enforce_symmetry(reduction_virial);
  reduction_virial *= 0.5;
  reduction_intVirialNormal *= 0.5;
  ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,reduction_virial);
  ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NORMAL,
                    reduction_intVirialNormal);
  reduction->item(REDUCTION_HALFSTEP_KINETIC_ENERGY) += (kineticEnergy_half[0] * 0.25);
  reduction->item(REDUCTION_INT_HALFSTEP_KINETIC_ENERGY) += (intKineticEnergy_half[0] * 0.25);

  // If we were on migration steps and move was accepted, we need to update
  // the myLatticeOld in order to use it in SequencerCUDAKernel::pairlistCheck();
  if(doMigration) {
    myLatticeOld = myLattice;
  }
}

void SequencerCUDA::monteCarloPressure_part1(
  Tensor         &factor,
  Vector         &origin,
  Lattice        &oldLattice)
{ 
  // Backup positions and forces
  //copy_DtoD<double>(d_f_raw, d_f_rawMC, numAtoms*9, stream);
  copy_DtoD<double>(d_f_normal_x, d_f_normalMC_x, numAtomsHome, stream);
  copy_DtoD<double>(d_f_normal_y, d_f_normalMC_y, numAtomsHome, stream);
  copy_DtoD<double>(d_f_normal_z, d_f_normalMC_z, numAtomsHome, stream);
  copy_DtoD<double>(d_f_nbond_x, d_f_nbondMC_x, numAtomsHome, stream);
  copy_DtoD<double>(d_f_nbond_y, d_f_nbondMC_y, numAtomsHome, stream);
  copy_DtoD<double>(d_f_nbond_z, d_f_nbondMC_z, numAtomsHome, stream);
  copy_DtoD<double>(d_f_slow_x, d_f_slowMC_x, numAtomsHome, stream);
  copy_DtoD<double>(d_f_slow_y, d_f_slowMC_y, numAtomsHome, stream);
  copy_DtoD<double>(d_f_slow_z, d_f_slowMC_z, numAtomsHome, stream);
#ifdef NAMD_NCCL_ALLREDUCE
  if(mGpuOn) {
    //copy_DtoD<double>(d_posNew_raw, d_pos_rawMC, numAtomsHome*3, stream);
    copy_DtoD<double>(d_posNew_x, d_posMC_x, numAtomsHome, stream);
    copy_DtoD<double>(d_posNew_y, d_posMC_y, numAtomsHome, stream);
    copy_DtoD<double>(d_posNew_z, d_posMC_z, numAtomsHome, stream);
  } else {
    //copy_DtoD<double>(d_pos_raw, d_pos_rawMC, numAtomsHome*3, stream);
    copy_DtoD<double>(d_pos_x, d_posMC_x, numAtomsHome, stream);
    copy_DtoD<double>(d_pos_y, d_posMC_y, numAtomsHome, stream);
    copy_DtoD<double>(d_pos_z, d_posMC_z, numAtomsHome, stream);
  }
#else
  //copy_DtoD<double>(d_pos_raw, d_pos_rawMC, numAtomsHome*3, stream);
  copy_DtoD<double>(d_pos_x, d_posMC_x, numAtomsHome, stream);
  copy_DtoD<double>(d_pos_y, d_posMC_y, numAtomsHome, stream);
  copy_DtoD<double>(d_pos_z, d_posMC_z, numAtomsHome, stream);
#endif

  // Scale the old lattice with factor. We need both lattice and newLattice
  // to properly unwrap and wrap the atom's coordinate
  Lattice newLattice = oldLattice;
  newLattice.rescale(factor);
  cudaTensor cuFactor;
  cudaVector cuOrigin;
  COPY_CUDATENSOR(factor, cuFactor);
  COPY_CUDAVECTOR(origin, cuOrigin);

  // Scale the coordinate using Molecule's geometric center
  Molecule *molecule = Node::Object()->molecule;
  CUDASequencerKernel->scaleCoordinateUsingGC(
            d_pos_x, d_pos_y, d_pos_z, d_idOrder, d_moleculeStartIndex,
            d_moleculeAtom, cuFactor, cuOrigin, myLattice, newLattice, 
            d_transform, molecule->numMolecules, molecule->numLargeMolecules,
            stream);
  
  // Update the cuda lattice with newLattice for force calculation
  myLattice = newLattice;

  // Set up compute position before calling bonded and nonbonded kernel
  const double charge_scaling = sqrt(COULOMB * ComputeNonbondedUtil::scaling *
     ComputeNonbondedUtil::dielectric_1);
  bool doNbond = patchData->flags.doNonbonded;
  bool doSlow = patchData->flags.doFullElectrostatics;
  bool doFEP = false;
  bool doTI = false;
  bool doAlchDecouple = false;
  bool doAlchSoftCore = false;
  if (simParams->alchOn) {
    if (simParams->alchFepOn) doFEP = true;
    if (simParams->alchThermIntOn) doTI = true;
    if (simParams->alchDecouple) doAlchDecouple = true;
    if (simParams->alchElecLambdaStart > 0) doAlchSoftCore = true;
  }

  std::vector<int> atom_counts;
  for (int i = 0; i < deviceCUDA->getDeviceCount(); i++) {
    atom_counts.push_back(patchData->devData[i].numAtomsHome);
  }
  CUDASequencerKernel->set_compute_positions(
                deviceIndex, 
                deviceCUDA->getIsPmeDevice(), 
                nDevices, 
                numPatchesHomeAndProxy, numPatchesHome, doNbond, doSlow, 
                doFEP, doTI, doAlchDecouple, doAlchSoftCore,
#ifdef NAMD_NCCL_ALLREDUCE
                (mGpuOn) ? d_posNew_x: d_pos_x, 
                (mGpuOn) ? d_posNew_y: d_pos_y, 
                (mGpuOn) ? d_posNew_z: d_pos_z, 
#else
                d_pos_x, 
                d_pos_y, 
                d_pos_z, 
                d_peer_pos_x, // passes double-pointer if mgpuOn
                d_peer_pos_y, 
                d_peer_pos_z,
                d_peer_charge,
                d_peer_partition,
#endif
                d_charge, d_partition, charge_scaling,
                d_patchCenter,
                patchData->devData[deviceIndex].slow_patchPositions,
                patchData->devData[deviceIndex].slow_pencilPatchIndex, patchData->devData[deviceIndex].slow_patchID, 
                d_sortOrder, newLattice, 
                (float4*) patchData->devData[deviceIndex].nb_datoms, patchData->devData[deviceIndex].b_datoms,
                (float4*)patchData->devData[deviceIndex].s_datoms, patchData->devData[deviceIndex].s_datoms_partition, 
                Node::Object()->molecule->numAtoms,
                patchData->devData[deviceIndex].d_localPatches,
                patchData->devData[deviceIndex].d_peerPatches,
                atom_counts,
                stream);

  cudaCheck(cudaStreamSynchronize(stream));
}

void SequencerCUDA::monteCarloPressure_part2(
  NodeReduction *reduction,
  int           step,
  int           maxForceNumber,
  const bool    doEnergy,
  const bool    doVirial)
{
  // we zero all reduction value in part1. Need to add this
  reduction->item(REDUCTION_ATOM_CHECKSUM) += numAtomsHome;

  if(mGpuOn){
#ifdef NAMD_NCCL_ALLREDUCE
    cudaCheck(cudaMemset(d_f_raw, 0, sizeof(double)*numAtoms*3*(maxForceNumber+1)));
#endif
  }
  int doTcl = simParams->tclForcesOn;
  int doColvars = simParams->colvarsOn;
  const int doGlobal = (doTcl || doColvars);
  //Update SOA buffer
  CUDASequencerKernel->accumulateForceToSOA(
      doGlobal,
      maxForceNumber,
      numPatchesHomeAndProxy,
      nDevices,
      patchData->devData[deviceIndex].d_localPatches,
      patchData->devData[deviceIndex].f_bond,
      patchData->devData[deviceIndex].f_bond_nbond,
      patchData->devData[deviceIndex].f_bond_slow,
      patchData->devData[deviceIndex].forceStride,
      patchData->devData[deviceIndex].f_nbond,
      patchData->devData[deviceIndex].f_nbond_slow,
      patchData->devData[deviceIndex].f_slow,
      d_f_global_x,
      d_f_global_y,
      d_f_global_z,
      d_f_normal_x,
      d_f_normal_y,
      d_f_normal_z,
      d_f_nbond_x,
      d_f_nbond_y,
      d_f_nbond_z,
      d_f_slow_x,
      d_f_slow_y,
      d_f_slow_z,
      d_unsortOrder,
      myLattice,
      patchData->d_queues, 
      patchData->d_queueCounters, 
      d_tbcatomic, 
      stream
  );
  if(mGpuOn){
#ifndef NAMD_NCCL_ALLREDUCE
    // JM - Awful: We need to busy wait inside accumulateForceToSOA instead
    //ncclBroadcast(d_barrierFlag, d_barrierFlag, 1, ncclChar, 
    //  0, deviceCUDA->getNcclComm(), stream);
    std::vector<int> atom_counts;
    for (int i = 0; i < deviceCUDA->getDeviceCount(); i++) {
      atom_counts.push_back(patchData->devData[i].numAtomsHome);
    }
    CUDASequencerKernel->mergeForcesFromPeers(
      deviceIndex, 
      maxForceNumber, 
      myLattice, 
      numPatchesHomeAndProxy,
      numPatchesHome, 
      this->d_peer_fb_x, 
      this->d_peer_fb_y, 
      this->d_peer_fb_z,
      this->d_peer_fn_x, 
      this->d_peer_fn_y, 
      this->d_peer_fn_z, 
      this->d_peer_fs_x, 
      this->d_peer_fs_y, 
      this->d_peer_fs_z,
      // patchData->devData[deviceCUDA->getPmeDevice()].f_slow,
      patchData->devData[deviceCUDA->getPmeDeviceIndex()].f_slow,
      patchData->devData[deviceIndex].d_localPatches,
      patchData->devData[deviceIndex].d_peerPatches,
      atom_counts,
      stream
    );
#else
    int numReducedAtoms = (3 * (maxForceNumber+1)) * numAtoms;
    ncclAllReduce(d_f_raw, d_f_raw, numReducedAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream );
#endif
  }
  // do external forces calculation
  calculateExternalForces(step, reduction, maxForceNumber, doEnergy, doVirial);
}

void SequencerCUDA::launch_part1(
  int step,
  double         dt_normal,
  double         dt_nbond,
  double         dt_slow,
  double         velrescaling,
  const double   maxvel2,
  NodeReduction *reduction,
  Tensor         &factor,
  Vector         &origin,
  Lattice        &lattice,
  int            reassignVelocitiesStep,
  int            langevinPistonStep,
  int            berendsenPressureStep,
  int            maxForceNumber,
  const int      copyIn,
  const int      savePairlists,
  const int      usePairlists,
  const bool     doEnergy)
{
  PatchMap* patchMap = PatchMap::Object();
  // Aggregate data from all patches
  cudaCheck(cudaSetDevice(deviceID));
  this->maxvel2 = maxvel2;
  const bool doVirial = simParams->langevinPistonOn || simParams->berendsenPressureOn;
  // JM: for launch_part1: 
  //     copyIn:  first call
  myLattice = lattice;
  if(reassignVelocitiesStep)
    {
      const int reassignFreq = simParams->reassignFreq;
      BigReal newTemp = simParams->reassignTemp;
      newTemp += ( step / reassignFreq ) * simParams->reassignIncr;
      if ( simParams->reassignIncr > 0.0 ) {
	if ( newTemp > simParams->reassignHold && simParams->reassignHold > 0.0 )
	  newTemp = simParams->reassignHold;
      } else {
	if ( newTemp < simParams->reassignHold )
	  newTemp = simParams->reassignHold;
      }
      const BigReal kbT = BOLTZMANN * newTemp;

      CUDASequencerKernel->reassignVelocities(
      dt_normal,
      d_gaussrand_x, d_gaussrand_y, d_gaussrand_z,
      d_vel_x, d_vel_y, d_vel_z,
      d_recipMass, kbT,
      numAtomsHome, numAtomsHome, 0,
      curandGen, stream);
    }

  // scale the position for berendsen Pressure Controller
  if(berendsenPressureStep) {
    cudaTensor cuFactor;
    cudaVector cuOrigin;
    COPY_CUDATENSOR(factor, cuFactor);
    COPY_CUDAVECTOR(origin, cuOrigin);
    CUDASequencerKernel->scaleCoordinateWithFactor(
            d_pos_x, d_pos_y, d_pos_z, d_mass, d_hydrogenGroupSize,
            cuFactor, cuOrigin, simParams->useGroupPressure, numAtomsHome, stream);
  }

  if(!langevinPistonStep){
    // kernel fusion here
    // JM TODO: Fuse kernels for the langevin thermostat
    CUDASequencerKernel->velocityVerlet1(patchData->flags.step, 0.5, dt_normal, dt_nbond,
      dt_slow, velrescaling, d_recipMass,
      d_vel_x, d_vel_y, d_vel_z, maxvel2, killme, d_pos_x, d_pos_y, d_pos_z,
      pos_x, pos_y, pos_z, d_f_normal_x, d_f_normal_y, d_f_normal_z,
      d_f_nbond_x, d_f_nbond_y, d_f_nbond_z, d_f_slow_x, d_f_slow_y, d_f_slow_z, 
      numAtomsHome, maxForceNumber, stream);
  }else{
    // Zero-out force buffers here
    CUDASequencerKernel->addForceToMomentum(
      0.5, dt_normal, dt_nbond, dt_slow, velrescaling,
      d_recipMass,
      d_f_normal_x, d_f_normal_y, d_f_normal_z,
      d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
      d_f_slow_x, d_f_slow_y, d_f_slow_z,
      d_vel_x, d_vel_y, d_vel_z,
      numAtomsHome, maxForceNumber, stream);

    maximumMove(maxvel2, numAtomsHome);
    cudaTensor cuFactor;
    cudaVector cuOrigin;
    COPY_CUDATENSOR(factor, cuFactor);
    COPY_CUDAVECTOR(origin, cuOrigin);
    double velFactor_x = namd_reciprocal(factor.xx);
    double velFactor_y = namd_reciprocal(factor.yy);
    double velFactor_z = namd_reciprocal(factor.zz);

    CUDASequencerKernel->addVelocityToPosition(
        0.5*dt_normal, d_vel_x, d_vel_y, d_vel_z,
        d_pos_x, d_pos_y, d_pos_z,
        pos_x, pos_y, pos_z, numAtomsHome, false, stream);
    CUDASequencerKernel->langevinPiston(
        d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z,
        d_mass, d_hydrogenGroupSize,
        cuFactor, cuOrigin, velFactor_x, velFactor_y, velFactor_z,
        simParams->useGroupPressure, numAtomsHome, stream);
    CUDASequencerKernel->addVelocityToPosition(
        0.5*dt_normal, d_vel_x, d_vel_y, d_vel_z,
        d_pos_x, d_pos_y, d_pos_z,
        pos_x, pos_y, pos_z, numAtomsHome, false, stream);
  }

  // JM: Recalculate centers of mass if energy calculation or langevinPistonOn
  if( (doEnergy || doVirial) ) {
    CUDASequencerKernel->centerOfMass(
      d_pos_x, d_pos_y, d_pos_z,
      d_rcm_x, d_rcm_y, d_rcm_z, d_mass, 
      d_hydrogenGroupSize, numAtomsHome, stream);
    CUDASequencerKernel->centerOfMass(
      d_vel_x, d_vel_y, d_vel_z,
      d_vcm_x, d_vcm_y, d_vcm_z, d_mass, 
      d_hydrogenGroupSize, numAtomsHome, stream);
  }

  const double charge_scaling = sqrt(COULOMB * ComputeNonbondedUtil::scaling *
     ComputeNonbondedUtil::dielectric_1);
  // We need to find doNbond and doSlow for upcoming step
  bool doNbond = patchData->flags.doNonbonded;
  bool doSlow = patchData->flags.doFullElectrostatics;

  bool doFEP = false;
  bool doTI = false;
  bool doAlchDecouple = false;
  bool doAlchSoftCore = false;
  if (simParams->alchOn) {
    if (simParams->alchFepOn) doFEP = true;
    if (simParams->alchThermIntOn) doTI = true;
    if (simParams->alchDecouple) doAlchDecouple = true;
    if (simParams->alchElecLambdaStart > 0) doAlchSoftCore = true;
  }
  if ( ! savePairlists ) {
    double minSize = simParams->patchDimension - simParams->margin;
    double sysdima = lattice.a_r().unit() * lattice.a();
    double sysdimb = lattice.b_r().unit() * lattice.b();
    double sysdimc = lattice.c_r().unit() * lattice.c();
    // Let's pass migrationInfo here
    CUDASequencerKernel->PairListMarginCheck(numPatchesHome,
        patchData->devData[deviceIndex].d_localPatches,
        d_pos_x, d_pos_y, d_pos_z, d_posSave_x, d_posSave_y, d_posSave_z,
        d_awayDists, 
        myLattice, myLatticeOld, 
        d_patchMin, d_patchMax, d_patchCenter, 
        d_mInfo, 
        d_tbcatomic, simParams->pairlistTrigger,
        simParams->pairlistGrow, simParams->pairlistShrink,
        d_patchMaxAtomMovement, patchMaxAtomMovement,
        d_patchNewTolerance, patchNewTolerance,
        minSize, simParams->cutoff, sysdima, sysdimb, sysdimc,
        h_marginViolations,
        h_periodicCellSmall,
        rescalePairlistTolerance,
        isPeriodic, stream);
    rescalePairlistTolerance = false;
  }
  else {
    rescalePairlistTolerance = true;
  }

  if(mGpuOn){
    // Synchonize device before node barrier
    cudaCheck(cudaStreamSynchronize(stream));
  }
}

void SequencerCUDA::launch_part11(
  double         dt_normal,
  double         dt_nbond,
  double         dt_slow,
  double         velrescaling,
  const double   maxvel2,
  NodeReduction *reduction,
  Tensor         &factor,
  Vector         &origin,
  Lattice        &lattice, 
  int            langevinPistonStep,
  int            maxForceNumber,
  const int      copyIn,
  const int      savePairlists,
  const int      usePairlists, 
  const bool     doEnergy)
{
  const bool doVirial = simParams->langevinPistonOn;
  const double charge_scaling = sqrt(COULOMB * ComputeNonbondedUtil::scaling *
     ComputeNonbondedUtil::dielectric_1);
  // We need to find doNbond and doSlow for upcoming step
  bool doNbond = patchData->flags.doNonbonded;
  bool doSlow = patchData->flags.doFullElectrostatics;

  bool doFEP = false;
  bool doTI = false;
  bool doAlchDecouple = false;
  bool doAlchSoftCore = false;
  if (simParams->alchOn) {
    if (simParams->alchFepOn) doFEP = true;
    if (simParams->alchThermIntOn) doTI = true;
    if (simParams->alchDecouple) doAlchDecouple = true;
    if (simParams->alchElecLambdaStart > 0) doAlchSoftCore = true;
  }

  submitHalf(reduction, numAtomsHome, 1, doEnergy || doVirial);

  // Updating numerical flags 
  NAMD_EVENT_START(1, NamdProfileEvent::CPY_PATCHFLAGS);
  this->update_patch_flags();
  NAMD_EVENT_STOP(1, NamdProfileEvent::CPY_PATCHFLAGS);

  finish_part1(copyIn, patchList[0]->flags.savePairlists, 
    patchList[0]->flags.usePairlists, reduction);
}


void SequencerCUDA::launch_set_compute_positions() {

  const double charge_scaling = sqrt(COULOMB * ComputeNonbondedUtil::scaling *
     ComputeNonbondedUtil::dielectric_1);
  // We need to find doNbond and doSlow for upcoming step
  bool doNbond = patchData->flags.doNonbonded;
  bool doSlow = patchData->flags.doFullElectrostatics;

  bool doFEP = false;
  bool doTI = false;
  bool doAlchDecouple = false;
  bool doAlchSoftCore = false;
  if (simParams->alchOn) {
    if (simParams->alchFepOn) doFEP = true;
    if (simParams->alchThermIntOn) doTI = true;
    if (simParams->alchDecouple) doAlchDecouple = true;
    if (simParams->alchElecLambdaStart > 0) doAlchSoftCore = true;
  }
  bool doGlobal = simParams->tclForcesOn || simParams->colvarsOn;
  if (1) {
    //fprintf(stderr, "calling set_compute_positions() ****************************************\n");
    //fprintf(stderr, "calling set_compute_positions\n");
    //fprintf(stderr, "doNbond=%d  doSlow=%d\n", doNbond, doSlow);
    std::vector<int> atom_counts;
    for (int i = 0; i < deviceCUDA->getDeviceCount(); i++) {
      atom_counts.push_back(patchData->devData[i].numAtomsHome);
    }
    CUDASequencerKernel->set_compute_positions(
                  deviceIndex, 
                  deviceCUDA->getIsPmeDevice(), 
                  nDevices, 
                  numPatchesHomeAndProxy, numPatchesHome, doNbond, doSlow, 
                  doFEP, doTI, doAlchDecouple, doAlchSoftCore,
#ifdef NAMD_NCCL_ALLREDUCE
                  (mGpuOn) ? d_posNew_x: d_pos_x,
                  (mGpuOn) ? d_posNew_y: d_pos_y,
                  (mGpuOn) ? d_posNew_z: d_pos_z,
#else
                  d_pos_x,
                  d_pos_y,
                  d_pos_z,
                  d_peer_pos_x, // passes double-pointer if mgpuOn
                  d_peer_pos_y,
                  d_peer_pos_z,
                  d_peer_charge,
                  d_peer_partition,
#endif
                  d_charge, d_partition, charge_scaling,
                  d_patchCenter,
                  patchData->devData[deviceIndex].slow_patchPositions,
                  patchData->devData[deviceIndex].slow_pencilPatchIndex, patchData->devData[deviceIndex].slow_patchID, 
                  d_sortOrder, myLattice,
                  (float4*) patchData->devData[deviceIndex].nb_datoms, patchData->devData[deviceIndex].b_datoms,
                  (float4*)patchData->devData[deviceIndex].s_datoms, patchData->devData[deviceIndex].s_datoms_partition, 
                  Node::Object()->molecule->numAtoms,
                  patchData->devData[deviceIndex].d_localPatches,
                  patchData->devData[deviceIndex].d_peerPatches,
                  atom_counts,
                  stream);
    // For global forces, copy the coordinate to host with kernel overlap
    if (doGlobal) {
      NAMD_EVENT_START(1, NamdProfileEvent::GM_CPY_POSITION);
      //      CkPrintf("WARNING this probably needs to be changed for multihost\n");
      copyPositionsToHost_direct();
      //copyPositionsAndVelocitiesToHost(1,0);
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_CPY_POSITION);
    }
  }
}

void SequencerCUDA:: finish_part1( const int copyIn,
                                   const int savePairlists,
                                   const int usePairlists,
                                   NodeReduction* reduction)
{
  // JM: If we're not in a migration step, let's overlap the  flagging the
  //     positions before we synchronize the stream to lessen the compute
  //     launch overhead
  // Hopefully we will see some overlap in this region
  //
  // TODO: We can just call this a different function and start calling positionsReady
  // if we're not on a migration step, so that we can overlap some of the work with the kernels
  // before we synchronize, we can clear the device memory
  // sets the tileListStat for the nbondKernel
  cudaCheck(cudaStreamSynchronize(stream));

  // Checks if periodic cell became too small
  if(*h_periodicCellSmall){
    NAMD_die("Periodic cell has become too small for original patch grid!\n"
      "Possible solutions are to restart from a recent checkpoint,\n"
      "increase margin, or disable useFlexibleCell for liquid simulation.");
  }

  if (killme[0]) {
    // Found at least one atom that is moving too fast.
    // Terminating, so loop performance below doesn't matter.
    // Loop does not vectorize
    double *vel_x, *vel_y, *vel_z;
    allocate_host<double>(&vel_x, numAtomsHome);
    allocate_host<double>(&vel_y, numAtomsHome);
    allocate_host<double>(&vel_z, numAtomsHome);
    copy_DtoH<double>(d_vel_x, vel_x, numAtomsHome);
    copy_DtoH<double>(d_vel_y, vel_y, numAtomsHome);
    copy_DtoH<double>(d_vel_z, vel_z, numAtomsHome);
    int cnt = 0;
    for (int i=0;  i < numAtomsHome;  i++) {
      BigReal vel2 =
        vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] + vel_z[i] * vel_z[i];
      if (vel2 > maxvel2) {
        ++cnt;
        iout << iERROR << " velocity is "
             << PDBVELFACTOR * vel_x[i] << " "
             << PDBVELFACTOR * vel_y[i] << " "
             << PDBVELFACTOR * vel_z[i]
             << " (limit is "
             << ( PDBVELFACTOR * sqrt(maxvel2) ) << ", atom "
             << i << " of " << numAtomsHome
             << " pe " << CkMyPe() << ")\n" << endi;
      }
    }
    iout << iERROR << "Atoms moving too fast at timestep " << patchList[0]->flags.step <<
      "; simulation has become unstable ("
      << cnt << " atoms on pe " << CkMyPe() << ").\n" << endi;
    deallocate_host<double>(&vel_x);
    deallocate_host<double>(&vel_y);
    deallocate_host<double>(&vel_z);
    NAMD_die("SequencerCUDA: Atoms moving too fast");
  }
  else{
    // submitHalf reductions
    Tensor reduction_virial;
    Tensor reduction_intVirialNormal;
    COPY_CUDATENSOR(virial_half[0], reduction_virial);
    COPY_CUDATENSOR(intVirialNormal_half[0], reduction_intVirialNormal);
    reduction->item(REDUCTION_HALFSTEP_KINETIC_ENERGY) += (kineticEnergy_half[0] * 0.25);
    tensor_enforce_symmetry(reduction_virial);
    reduction_virial *= 0.5;
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,reduction_virial);
    // fprintf(stderr, "GPU calculated internal kinetic energy = %lf\n", intKineticEnergy_half);
    reduction->item(REDUCTION_INT_HALFSTEP_KINETIC_ENERGY)
      += (intKineticEnergy_half[0] * 0.25);
    reduction_intVirialNormal *= 0.5;
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NORMAL,
                      reduction_intVirialNormal);

    int migration = (h_marginViolations[0] != 0) ? 1 :0; // flags migration as TRUE if margin violation occured
    // if(migration != 0 ) fprintf(stderr, "DEV[%d] = MIGRATION[%d]\n", deviceID, migration);
    patchData->migrationFlagPerDevice[deviceIndex] = migration; // Saves the updated migration flag
    h_marginViolations[0] = 0;
  }
}

void SequencerCUDA::copyPositionsAndVelocitiesToHost(bool copyOut, const int doGlobal){
  //    CkPrintf("copy positions and velocities to host copyout %d doGlobal %d\n", copyOut, doGlobal);
  if(copyOut){
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    patchData = cpdata.ckLocalBranch();
    std::vector<CudaPeerRecord>& myPeerPatches = patchData->devData[deviceIndex].h_peerPatches;
    std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
    std::vector<HomePatch*>& homePatches = patchData->devData[deviceIndex].patches;
    
    const int numAtomsToCopy = numAtomsHome;
    copy_DtoH<double>(d_vel_x, vel_x, numAtomsToCopy, stream);
    copy_DtoH<double>(d_vel_y, vel_y, numAtomsToCopy, stream);
    copy_DtoH<double>(d_vel_z, vel_z, numAtomsToCopy, stream);
    if (!doGlobal) {
      // We already copied coordinate if we have global forces
      copy_DtoH<double>(d_pos_x, pos_x, numAtomsToCopy, stream);
      copy_DtoH<double>(d_pos_y, pos_y, numAtomsToCopy, stream);
      copy_DtoH<double>(d_pos_z, pos_z, numAtomsToCopy, stream);
    }
    cudaCheck(cudaDeviceSynchronize());

    for(int i = 0; i < homePatches.size(); i++){

      // TODO do we need to copy proxy patches as well
      PatchDataSOA& current = homePatches[i]->patchDataSOA;
      const int numPatchAtoms = localPatches[i].numAtoms;
      const int offset = localPatches[i].bufferOffset;
      memcpy(current.vel_x, vel_x + offset, numPatchAtoms*sizeof(double));
      memcpy(current.vel_y, vel_y + offset, numPatchAtoms*sizeof(double));
      memcpy(current.vel_z, vel_z + offset, numPatchAtoms*sizeof(double));
      if (!doGlobal) {
	// We already copied coordinate if we have global forces
	memcpy(current.pos_x, pos_x + offset, numPatchAtoms*sizeof(double));
	memcpy(current.pos_y, pos_y + offset, numPatchAtoms*sizeof(double));
	memcpy(current.pos_z, pos_z + offset, numPatchAtoms*sizeof(double));
      }
    }
  }
}

void SequencerCUDA::copyPositionsToHost(){
  //    CkPrintf("copy positions to host \n");
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    patchData = cpdata.ckLocalBranch();
    std::vector<CudaPeerRecord>& myPeerPatches = patchData->devData[deviceIndex].h_peerPatches;
    std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
    std::vector<HomePatch*>& homePatches = patchData->devData[deviceIndex].patches;
    
    const int numAtomsToCopy = numAtomsHome;
    // We already copied coordinate if we have global forces
    copy_DtoH<double>(d_pos_x, pos_x, numAtomsToCopy, stream);
    copy_DtoH<double>(d_pos_y, pos_y, numAtomsToCopy, stream);
    copy_DtoH<double>(d_pos_z, pos_z, numAtomsToCopy, stream);
    cudaCheck(cudaDeviceSynchronize());

    for(int i = 0; i < homePatches.size(); i++){

      // TODO do we need to copy proxy patches as well
      PatchDataSOA& current = homePatches[i]->patchDataSOA;
      const int numPatchAtoms = localPatches[i].numAtoms;
      const int offset = localPatches[i].bufferOffset;
      memcpy(current.pos_x, pos_x + offset, numPatchAtoms*sizeof(double));
      memcpy(current.pos_y, pos_y + offset, numPatchAtoms*sizeof(double));
      memcpy(current.pos_z, pos_z + offset, numPatchAtoms*sizeof(double));
    }
}

void SequencerCUDA::update_patch_flags()
{
  // int pairlists = 1;
  int pairlists = (patchData->flags.step < simParams->N);
  for (int i=0;  i < numPatchesHome;  i++) {
    HomePatch *patch = patchList[i];
    patch->flags.copyIntFlags(patchData->flags); // copy global flags
  }
}

void SequencerCUDA::updatePairlistFlags(const int doMigration){
  int pairlists = patchList[0]->flags.step < simParams->N; 
  for(int i = 0; i < numPatchesHome; i++){
  //for(int i = 0; i < numPatches; i++){
    HomePatch *patch = patchList[i];
    Sequencer *seq = patch->sequencer;
    // the following logic is duplicated from Sequencer::runComputeObjects
    // Migrations always invalidates pairlists
    if (doMigration) {
      seq->pairlistsAreValid = 0;
    }
    if (seq->pairlistsAreValid &&
      ( patch->flags.doFullElectrostatics || ! simParams->fullElectFrequency )
        && (seq->pairlistsAge > seq->pairlistsAgeLimit) ) {
        seq->pairlistsAreValid = 0;
    }
    patch->flags.usePairlists  = pairlists ||  seq->pairlistsAreValid;
    patch->flags.savePairlists = pairlists && !seq->pairlistsAreValid;
    if(patch->flags.savePairlists){
      // We need to rebuild pairlists -> reset tolerance values
      patch->flags.pairlistTolerance = patchList[i]->doPairlistCheck_newTolerance; // update pairListTolerance
      patch->flags.maxAtomMovement = 0;
      patch->doPairlistCheck_newTolerance *= (1 - simParams->pairlistShrink);
    }else if(patch->flags.usePairlists){
      // We can keep going with the existing pairlists -> update tolerances
      patch->flags.maxAtomMovement = patchMaxAtomMovement[i];
      patch->doPairlistCheck_newTolerance = patchNewTolerance[i];
    }else{
      // End of simulation
      patch->flags.maxAtomMovement=99999.;
      patch->flags.pairlistTolerance = 0.;
    }
  }
  if(patchList[0]->flags.savePairlists){
    // Backs up d_posSave_* for pairlistCheck
    copy_DtoD<double>(d_pos_x, d_posSave_x, numAtomsHome, stream);
    copy_DtoD<double>(d_pos_y, d_posSave_y, numAtomsHome, stream);
    copy_DtoD<double>(d_pos_z, d_posSave_z, numAtomsHome, stream);
    myLatticeOld = myLattice;
  }
}

void SequencerCUDA::finish_patch_flags(int migration)
{
  for (int i=0;  i < numPatchesHome;  i++) {
    HomePatch *patch = patchList[i];
    Sequencer *seq = patch->sequencer;
    if (patch->flags.savePairlists && patch->flags.doNonbonded) {
      seq->pairlistsAreValid = 1;
      seq->pairlistsAge = 0;
    }
    if (seq->pairlistsAreValid /* && ! pressureStep */) {
      ++(seq->pairlistsAge);
    }
  }
}


void SequencerCUDA::launch_part2(
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
  const bool    doEnergy)
{
  PatchMap* patchMap = PatchMap::Object();
  Tensor localVirial;
  //cudaTensor h_rigidVirial;
  bool doNbond = false;
  bool doSlow  = false;
  int doTcl = simParams->tclForcesOn;
  int doColvars = simParams->colvarsOn;
  cudaCheck(cudaSetDevice(deviceID));
  const int doVirial = simParams->langevinPistonOn || simParams->berendsenPressureOn;
  // const int doVirial = langevinPistonStep;
  // JM: For launch_part2:
  //   copyIn   = migration steps

  reduction->item(REDUCTION_ATOM_CHECKSUM) += numAtomsHome;

  if(mGpuOn){
#ifdef NAMD_NCCL_ALLREDUCE
    cudaCheck(cudaMemset(d_f_raw, 0, sizeof(double)*numAtomsHomeAndProxy*3*(maxForceNumber+1)));
#endif
  }

  if(!simParams->langevinOn && !simParams->eFieldOn && !simParams->constraintsOn &&
    !simParams->SMDOn && !simParams->groupRestraintsOn && !doMCPressure && !mGpuOn &&
    (simParams->watmodel == WaterModel::TIP3)){
    CUDASequencerKernel->accumulate_force_kick(
      doGlobal,					       
      maxForceNumber,
      numPatchesHomeAndProxy,
      patchData->devData[deviceIndex].d_localPatches,
      patchData->devData[deviceIndex].f_bond,
      patchData->devData[deviceIndex].f_bond_nbond,
      patchData->devData[deviceIndex].f_bond_slow,
      patchData->devData[deviceIndex].forceStride,
      patchData->devData[deviceIndex].f_nbond,
      patchData->devData[deviceIndex].f_nbond_slow,
      patchData->devData[deviceIndex].f_slow,
      d_f_global_x,
      d_f_global_y,
      d_f_global_z,
      d_f_normal_x,
      d_f_normal_y,
      d_f_normal_z,
      d_f_nbond_x,
      d_f_nbond_y,
      d_f_nbond_z,
      d_f_slow_x,
      d_f_slow_y,
      d_f_slow_z,
      d_vel_x,
      d_vel_y,
      d_vel_z,
      d_recipMass, 
      dt_normal, 
      dt_nbond, 
      dt_slow, 
      1.0, 
      d_unsortOrder,
      myLattice,
      stream
      );
  }else{
    CUDASequencerKernel->accumulateForceToSOA(
      doGlobal,					       
      maxForceNumber,
      numPatchesHomeAndProxy,
      nDevices,
      patchData->devData[deviceIndex].d_localPatches,
      patchData->devData[deviceIndex].f_bond,
      patchData->devData[deviceIndex].f_bond_nbond,
      patchData->devData[deviceIndex].f_bond_slow,
      patchData->devData[deviceIndex].forceStride,
      patchData->devData[deviceIndex].f_nbond,
      patchData->devData[deviceIndex].f_nbond_slow,
      patchData->devData[deviceIndex].f_slow,
      d_f_global_x,
      d_f_global_y,
      d_f_global_z,
      d_f_normal_x,
      d_f_normal_y,
      d_f_normal_z,
      d_f_nbond_x,
      d_f_nbond_y,
      d_f_nbond_z,
      d_f_slow_x,
      d_f_slow_y,
      d_f_slow_z,
      d_unsortOrder,
      myLattice,
      patchData->d_queues, 
      patchData->d_queueCounters, 
      d_tbcatomic, 
      stream
    );

  }

  if (mGpuOn) {
    // Synchonize device before node barrier
    cudaCheck(cudaDeviceSynchronize());
  }
}

// launch_part2 is broken into 2 part to support MC barostat
void SequencerCUDA::launch_part3(
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
  const bool    doEnergy)
{
  const int doVirial = simParams->langevinPistonOn || simParams->berendsenPressureOn;
  const double velrescaling = 1;  // no rescaling

  if(simParams->langevinOn || simParams->eFieldOn || simParams->constraintsOn ||
    simParams->SMDOn || simParams->groupRestraintsOn || forceRequested || mGpuOn ||
    (simParams->watmodel != WaterModel::TIP3)){
    if(mGpuOn){
#ifndef NAMD_NCCL_ALLREDUCE
    // JM - Awful: We need to busy wait inside accumulateForceToSOA instead
    //ncclBroadcast(d_barrierFlag, d_barrierFlag, 1, ncclChar, 
    //  0, deviceCUDA->getNcclComm(), stream);

    std::vector<int> atom_counts;
    for (int i = 0; i < deviceCUDA->getDeviceCount(); i++) {
      atom_counts.push_back(patchData->devData[i].numAtomsHome);
    }
    CUDASequencerKernel->mergeForcesFromPeers(
      deviceIndex, 
      maxForceNumber, 
      myLattice, 
      numPatchesHomeAndProxy,
      numPatchesHome, 
      this->d_peer_fb_x, 
      this->d_peer_fb_y, 
      this->d_peer_fb_z,
      this->d_peer_fn_x, 
      this->d_peer_fn_y, 
      this->d_peer_fn_z, 
      this->d_peer_fs_x, 
      this->d_peer_fs_y, 
      this->d_peer_fs_z,
      // patchData->devData[deviceCUDA->getPmeDevice()].f_slow,
      patchData->devData[deviceCUDA->getPmeDeviceIndex()].f_slow,
      patchData->devData[deviceIndex].d_localPatches,
      patchData->devData[deviceIndex].d_peerPatches,
      atom_counts,
      stream
    );
#else
    int numReducedAtoms = (3 * (maxForceNumber+1)) * numAtoms;
    ncclAllReduce(d_f_raw, d_f_raw, numReducedAtoms, ncclDouble, ncclSum, deviceCUDA->getNcclComm(), stream );
#endif
    }
    // do external forces calculation
    calculateExternalForces(step, reduction, maxForceNumber, doEnergy, doVirial);
  }

  if (simParams->langevinOn) {
    CUDASequencerKernel->langevinVelocitiesBBK1(
      dt_normal, d_langevinParam, d_vel_x, d_vel_y, d_vel_z, numAtomsHome, stream);
  }
  
  if(simParams->langevinOn || simParams->eFieldOn || simParams->constraintsOn || 
    simParams->SMDOn || simParams->groupRestraintsOn || doMCPressure || mGpuOn ||
    (simParams->watmodel != WaterModel::TIP3)){
    CUDASequencerKernel->addForceToMomentum(
      1.0, dt_normal, dt_nbond, dt_slow, velrescaling,
      d_recipMass,
      d_f_normal_x, d_f_normal_y, d_f_normal_z,
      d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
      d_f_slow_x, d_f_slow_y, d_f_slow_z,
      d_vel_x, d_vel_y, d_vel_z,
      numAtomsHome, maxForceNumber, stream);
  }

  if (simParams->langevinOn) {

    // must enforce rigid bond constraints if langevin gammas differ
    if (simParams->rigidBonds != RIGID_NONE &&
        simParams->langevinGammasDiffer) {
      CUDASequencerKernel->rattle1(doEnergy || doVirial, 1, numAtomsHome, dt_normal, 1.0/dt_normal,
        2.0 * simParams->rigidTol,
        d_vel_x, d_vel_y, d_vel_z,
        d_pos_x, d_pos_y, d_pos_z,
        d_velNew_x, d_velNew_y, d_velNew_z,
        d_posNew_x, d_posNew_y, d_posNew_z,
        d_f_normal_x, d_f_normal_y, d_f_normal_z,
        d_hydrogenGroupSize, d_rigidBondLength, d_mass, d_atomFixed,
        &settleList, settleListSize, &d_consFailure, 
        d_consFailureSize, &rattleList, rattleListSize,
        &nSettle, &nRattle,
        d_rigidVirial, rigidVirial, d_tbcatomic, copyIn, sp,
        buildRigidLists, consFailure, simParams->watmodel, stream);
      buildRigidLists = false;
    }
    CUDASequencerKernel->langevinVelocitiesBBK2(
      dt_normal, d_langScalVelBBK2, d_langScalRandBBK2,
      d_gaussrand_x, d_gaussrand_y, d_gaussrand_z,
      d_vel_x, d_vel_y, d_vel_z,
      numAtomsHome, numAtomsHome, 0, 
      curandGen, stream);
  }
  if(simParams->rigidBonds != RIGID_NONE){
    CUDASequencerKernel->rattle1(doEnergy || doVirial, 1,  numAtomsHome, dt_normal, 1.0/dt_normal,
      2.0 * simParams->rigidTol,
      d_vel_x, d_vel_y, d_vel_z,
      d_pos_x, d_pos_y, d_pos_z,
      d_velNew_x, d_velNew_y, d_velNew_z,
      d_posNew_x, d_posNew_y, d_posNew_z,
      d_f_normal_x, d_f_normal_y, d_f_normal_z,
      d_hydrogenGroupSize, d_rigidBondLength, d_mass, d_atomFixed,
      &settleList, settleListSize, &d_consFailure, 
      d_consFailureSize, &rattleList, rattleListSize,
      &nSettle, &nRattle,
      d_rigidVirial, rigidVirial, d_tbcatomic, copyIn, sp,
      buildRigidLists, consFailure, simParams->watmodel, stream);
    buildRigidLists = false;
  }

  // Update velocity center of mass here
  if(doEnergy || doVirial){
    CUDASequencerKernel->centerOfMass(
      d_vel_x, d_vel_y, d_vel_z,
      d_vcm_x, d_vcm_y, d_vcm_z,
      d_mass, d_hydrogenGroupSize, numAtomsHome, stream);
  }

  submitHalf(reduction, numAtomsHome, 2, doEnergy || doVirial);
  
  CUDASequencerKernel->addForceToMomentum(
    -0.5, dt_normal, dt_nbond, dt_slow, velrescaling,
    d_recipMass,
    d_f_normal_x, d_f_normal_y, d_f_normal_z,
    d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
    d_f_slow_x, d_f_slow_y, d_f_slow_z,
    d_vel_x, d_vel_y, d_vel_z,
    numAtomsHome, maxForceNumber, stream);

  if(forceRequested) {
    // store the forces for next step, 
    // when we need it for colvars and Tcl scripting
    saveForceCUDASOA_direct(maxForceNumber);
  }

  //cudaCheck(cudaStreamSynchronize(stream));
  submitReductions(reduction, origin.x, origin.y, origin.z,
                   marginViolations, doEnergy || doVirial, 
                   copyOut && simParams->outputMomenta != 0, 
                   numAtomsHome, maxForceNumber);

  // This is for collecting coordinate and velocity to print
  copyPositionsAndVelocitiesToHost(copyOut, 0);

  if(consFailure[0]){
    // Constraint failure. Abort.
    int dieOnError = simParams->rigidDie;
    if(dieOnError){
      // Bails out
      //iout << iWARN << "constraint failure during GPU integration \n" << endi;
      NAMD_die("constraint failure during CUDA rattle!\n");
    }else{
      iout << iWARN << "constraint failure during CUDA rattle!\n" << endi;
    }
  }else if(doEnergy || doVirial){
    cudaCheck(cudaStreamSynchronize(stream));
    if(simParams->rigidBonds != RIGID_NONE){
      Tensor reduction_rigidVirial;
      COPY_CUDATENSOR(rigidVirial[0], reduction_rigidVirial);
      tensor_enforce_symmetry(reduction_rigidVirial);
      ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL, reduction_rigidVirial);
    }

    // SUBMITHALF reductions
    Tensor reduction_virial;
    Tensor reduction_intVirialNormal;
    COPY_CUDATENSOR(virial_half[0], reduction_virial);
    COPY_CUDATENSOR(intVirialNormal_half[0], reduction_intVirialNormal);
    reduction->item(REDUCTION_HALFSTEP_KINETIC_ENERGY) += (kineticEnergy_half[0] * 0.25);
    tensor_enforce_symmetry(reduction_virial);
    reduction_virial *= 0.5;
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,reduction_virial);

    reduction->item(REDUCTION_INT_HALFSTEP_KINETIC_ENERGY)
      += (intKineticEnergy_half[0] * 0.25);
    reduction_intVirialNormal *= 0.5;
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NORMAL,
                      reduction_intVirialNormal);

    //submitReductions1
    reduction->item(REDUCTION_CENTERED_KINETIC_ENERGY) += (kineticEnergy[0] * 0.5);
    Vector momentum(*momentum_x, *momentum_y, *momentum_z);
    ADD_VECTOR_OBJECT(reduction,REDUCTION_MOMENTUM,momentum);
    Vector angularMomentum(*angularMomentum_x,
                           *angularMomentum_y,
                           *angularMomentum_z);
    ADD_VECTOR_OBJECT(reduction,REDUCTION_ANGULAR_MOMENTUM,angularMomentum);
    //submitReductions2
    Tensor regintVirialNormal;
    Tensor regintVirialNbond;
    Tensor regintVirialSlow;
    COPY_CUDATENSOR(intVirialNormal[0], regintVirialNormal);
    if (maxForceNumber >= 1) {
    COPY_CUDATENSOR(intVirialNbond[0],  regintVirialNbond);
    }
    if (maxForceNumber >= 2) {
    COPY_CUDATENSOR(intVirialSlow[0],   regintVirialSlow);
    }

    reduction->item(REDUCTION_INT_CENTERED_KINETIC_ENERGY) += (intKineticEnergy[0] * 0.5);
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NORMAL, regintVirialNormal);
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_NBOND,  regintVirialNbond);
    ADD_TENSOR_OBJECT(reduction, REDUCTION_INT_VIRIAL_SLOW,   regintVirialSlow);
  }
}

// Adding this function back temporarily until GPU migration is merged
#if 1
// This function will aggregate data within a single GPU, so we need to have it copied over pmePositions
void SequencerCUDA::atomUpdatePme()
{
  const double charge_scaling = sqrt(COULOMB * ComputeNonbondedUtil::scaling *
     ComputeNonbondedUtil::dielectric_1);
  // We need to find doNbond and doSlow for upcoming step
  bool doNbond = false;
  bool doSlow = true;

  bool doFEP = false;
  bool doTI = false;
  bool doAlchDecouple = false;
  bool doAlchSoftCore = false;
  if (simParams->alchOn) {
    if (simParams->alchFepOn) doFEP = true;
    if (simParams->alchThermIntOn) doTI = true;
    if (simParams->alchDecouple) doAlchDecouple = true;
    if (simParams->alchElecLambdaStart > 0) doAlchSoftCore = true;
  }


  std::vector<int> atom_counts;
  for (int i = 0; i < deviceCUDA->getDeviceCount(); i++) {
    atom_counts.push_back(patchData->devData[i].numAtomsHome);
  }
  CUDASequencerKernel->set_pme_positions(
                  deviceIndex, 
                  deviceCUDA->getIsPmeDevice(), 
                  nDevices, 
                  numPatchesHomeAndProxy, numPatchesHome, doNbond, doSlow, 
                  doFEP, doTI, doAlchDecouple, doAlchSoftCore,
#ifdef NAMD_NCCL_ALLREDUCE
                  (mGpuOn) ? d_posNew_x: d_pos_x, 
                  (mGpuOn) ? d_posNew_y: d_pos_y, 
                  (mGpuOn) ? d_posNew_z: d_pos_z, 
#else
                  d_pos_x, 
                  d_pos_y, 
                  d_pos_z, 
                  d_peer_pos_x, // passes double-pointer if mgpuOn
                  d_peer_pos_y, 
                  d_peer_pos_z,
                  d_peer_charge,
                  d_peer_partition,
#endif
                  d_charge, d_partition, charge_scaling,
                  d_patchCenter,
                  patchData->devData[deviceIndex].slow_patchPositions,
                  patchData->devData[deviceIndex].slow_pencilPatchIndex, patchData->devData[deviceIndex].slow_patchID, 
                  d_sortOrder, myLattice,
                  (float4*) patchData->devData[deviceIndex].nb_datoms, patchData->devData[deviceIndex].b_datoms,
                  (float4*)patchData->devData[deviceIndex].s_datoms, patchData->devData[deviceIndex].s_datoms_partition, 
                  Node::Object()->molecule->numAtoms,
                  patchData->devData[deviceIndex].d_localPatches,
                  patchData->devData[deviceIndex].d_peerPatches,
                  atom_counts,
                  stream);

  cudaCheck(cudaStreamSynchronize(stream));
}
#endif


void SequencerCUDA::sync() {
  cudaCheck(cudaStreamSynchronize(stream));
}

void SequencerCUDA::calculateExternalForces(
  const int step,
  NodeReduction *reduction,
  const int maxForceNumber,
  const int doEnergy,
  const int doVirial) {

  if (simParams->watmodel == WaterModel::TIP4) {
    redistributeTip4pForces(reduction, maxForceNumber, doEnergy || doVirial);
  }

  if(simParams->eFieldOn){
      double3 efield;
      efield.x = simParams->eField.x;
      efield.y = simParams->eField.y;
      efield.z = simParams->eField.z;
      
      double efield_omega = TWOPI * simParams->eFieldFreq / 1000.;
      double efield_phi = PI/180. * simParams->eFieldPhase;
      double t = step * simParams->dt;
      
      CUDASequencerKernel->apply_Efield(numAtomsHome, simParams->eFieldNormalized, 
        doEnergy || doVirial, efield, efield_omega, efield_phi, t , myLattice, d_transform, 
        d_charge, d_pos_x, d_pos_y, d_pos_z, 
        d_f_normal_x, d_f_normal_y, d_f_normal_z, 
        &d_extForce[EXT_ELEC_FIELD], &d_extVirial[EXT_ELEC_FIELD],
        &d_extEnergy[EXT_ELEC_FIELD], &extForce[EXT_ELEC_FIELD], 
        &extVirial[EXT_ELEC_FIELD], &extEnergy[EXT_ELEC_FIELD], 
        d_tbcatomic, stream);
  }

  if(simParams->constraintsOn){
    restraintsKernel->doForce(&myLattice, doEnergy, doVirial, step, 
      d_pos_x, d_pos_y, d_pos_z, 
      d_f_normal_x, d_f_normal_y, d_f_normal_z, 
      &d_extEnergy[EXT_CONSTRAINTS], &extEnergy[EXT_CONSTRAINTS],
      &d_extForce[EXT_CONSTRAINTS], &extForce[EXT_CONSTRAINTS],
      &d_extVirial[EXT_CONSTRAINTS], &extVirial[EXT_CONSTRAINTS]);
  }

  if(simParams->SMDOn){
    SMDKernel->doForce(step, myLattice, doEnergy || doVirial, 
      d_mass, d_pos_x, d_pos_y, d_pos_z, d_transform,
      d_f_normal_x, d_f_normal_y, d_f_normal_z,
      &d_extVirial[EXT_SMD], &extEnergy[EXT_SMD],
      &extForce[EXT_SMD], &extVirial[EXT_SMD], stream);
  }

  if(simParams->groupRestraintsOn){
    groupRestraintsKernel->doForce(step, doEnergy, doVirial, 
      myLattice, d_transform,
      d_mass, d_pos_x, d_pos_y, d_pos_z, 
      d_f_normal_x, d_f_normal_y, d_f_normal_z,  
      &d_extVirial[EXT_GROUP_RESTRAINTS], &extEnergy[EXT_GROUP_RESTRAINTS],
      &extForce[EXT_GROUP_RESTRAINTS], &extVirial[EXT_GROUP_RESTRAINTS], stream);
  }

  if(doEnergy || doVirial) {
    // Store the external forces and energy data
    cudaCheck(cudaStreamSynchronize(stream));
    if(simParams->eFieldOn){
      reduction->item(REDUCTION_MISC_ENERGY) += extEnergy[EXT_ELEC_FIELD];
      if (!simParams->eFieldNormalized){
        ADD_VECTOR_OBJECT(reduction, REDUCTION_EXT_FORCE_NORMAL, extForce[EXT_ELEC_FIELD]);
        ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NORMAL, extVirial[EXT_ELEC_FIELD]);
      }
    }

    if(simParams->constraintsOn){
      reduction->item(REDUCTION_BC_ENERGY) += extEnergy[EXT_CONSTRAINTS];
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NORMAL, extVirial[EXT_CONSTRAINTS]);
      ADD_VECTOR_OBJECT(reduction, REDUCTION_EXT_FORCE_NORMAL, extForce[EXT_CONSTRAINTS]);
    }

    if(simParams->SMDOn){
      reduction->item(REDUCTION_MISC_ENERGY) += extEnergy[EXT_SMD];
      ADD_VECTOR_OBJECT(reduction, REDUCTION_EXT_FORCE_NORMAL, extForce[EXT_SMD]);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NORMAL, extVirial[EXT_SMD]);
    }    
        
    if(simParams->groupRestraintsOn){
      reduction->item(REDUCTION_MISC_ENERGY) += extEnergy[EXT_GROUP_RESTRAINTS];
      ADD_VECTOR_OBJECT(reduction, REDUCTION_EXT_FORCE_NORMAL, extForce[EXT_GROUP_RESTRAINTS]);
      ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NORMAL, extVirial[EXT_GROUP_RESTRAINTS]); 
    }
  }
}

void SequencerCUDA::copyGlobalForcesToDevice(){
  // copy the globalMaster forces  from host to device.
  // Use normal force on host to aggregate it
  std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
  //  fprintf(stderr,  "PE[%d] pos/vel printout, numPatchesHome = %d\n", CkMyPe(), numPatchesHome);
  std::vector<HomePatch*>& homePatches = patchData->devData[deviceIndex].patches;
  // TODO: determine if this aggregation needs peers and to be home and proxy
  for(int i =0 ; i < numPatchesHome; i++){
    CudaLocalRecord record = localPatches[i];
    const int patchID = record.patchID;
    const int stride = record.bufferOffset;
    const int numPatchAtoms = record.numAtoms;
    PatchDataSOA& current = homePatches[i]->patchDataSOA;
    memcpy(f_global_x + stride, current.f_global_x, numPatchAtoms*sizeof(double));
    memcpy(f_global_y + stride, current.f_global_y, numPatchAtoms*sizeof(double));
    memcpy(f_global_z + stride, current.f_global_z, numPatchAtoms*sizeof(double));
  }
  // copy aggregated force to device buffer
  copy_HtoD<double>(f_global_x, d_f_global_x, numAtomsHome, stream);
  copy_HtoD<double>(f_global_y, d_f_global_y, numAtomsHome, stream);
  copy_HtoD<double>(f_global_z, d_f_global_z, numAtomsHome, stream);

}

void SequencerCUDA::saveForceCUDASOA(const int maxForceNumber){
  switch (maxForceNumber) {
    case 2:
      copy_DtoH<double>(d_f_slow_x, f_slow_x, numAtomsHome, stream);
      copy_DtoH<double>(d_f_slow_y, f_slow_y, numAtomsHome, stream);
      copy_DtoH<double>(d_f_slow_z, f_slow_z, numAtomsHome, stream);
    case 1:
      copy_DtoH<double>(d_f_nbond_x, f_nbond_x, numAtomsHome, stream);
      copy_DtoH<double>(d_f_nbond_y, f_nbond_y, numAtomsHome, stream);
      copy_DtoH<double>(d_f_nbond_z, f_nbond_z, numAtomsHome, stream);
    case 0:
      copy_DtoH<double>(d_f_normal_x, f_normal_x, numAtomsHome, stream);
      copy_DtoH<double>(d_f_normal_y, f_normal_y, numAtomsHome, stream);
      copy_DtoH<double>(d_f_normal_z, f_normal_z, numAtomsHome, stream);
  }

  cudaCheck(cudaStreamSynchronize(stream));
  std::vector<CudaLocalRecord>& localPatches = patchData->devData[deviceIndex].h_localPatches;
  //  fprintf(stderr,  "PE[%d] pos/vel printout, numPatchesHome = %d\n", CkMyPe(), numPatchesHome);
  std::vector<HomePatch*>& homePatches = patchData->devData[deviceIndex].patches;
  switch (maxForceNumber) {
    case 2:
      for(int i =0 ; i < numPatchesHome; i++){
        CudaLocalRecord record = localPatches[i];
        const int stride = record.bufferOffset;
        const int numPatchAtoms = record.numAtoms;
        PatchDataSOA& current = homePatches[i]->patchDataSOA;
        memcpy(current.f_saved_slow_x, f_slow_x + stride, numPatchAtoms*sizeof(double));
        memcpy(current.f_saved_slow_y, f_slow_y + stride, numPatchAtoms*sizeof(double));
        memcpy(current.f_saved_slow_z, f_slow_z + stride, numPatchAtoms*sizeof(double));
      }
    case 1:
      for(int i =0 ; i < numPatchesHome; i++){
        CudaLocalRecord record = localPatches[i];
        const int stride = record.bufferOffset;
        const int numPatchAtoms = record.numAtoms;
        PatchDataSOA& current = homePatches[i]->patchDataSOA;
        memcpy(current.f_saved_nbond_x, f_nbond_x + stride, numPatchAtoms*sizeof(double));
        memcpy(current.f_saved_nbond_y, f_nbond_y + stride, numPatchAtoms*sizeof(double));
        memcpy(current.f_saved_nbond_z, f_nbond_z + stride, numPatchAtoms*sizeof(double));
      }
    case 0:
      for(int i =0 ; i < numPatchesHome; i++){
        CudaLocalRecord record = localPatches[i];
        const int stride = record.bufferOffset;
        const int numPatchAtoms = record.numAtoms;
        PatchDataSOA& current = homePatches[i]->patchDataSOA;
        memcpy(current.f_normal_x, f_normal_x + stride, numPatchAtoms*sizeof(double));
        memcpy(current.f_normal_y, f_normal_y + stride, numPatchAtoms*sizeof(double));
        memcpy(current.f_normal_z, f_normal_z + stride, numPatchAtoms*sizeof(double));
      }
  }
}

void SequencerCUDA::updateHostPatchDataSOA() {
    std::vector<PatchDataSOA> host_copy(numPatchesHome);
    std::vector<HomePatch*>& homePatches = patchData->devData[deviceIndex].patches;

    for(int i =0 ; i < numPatchesHome; i++) {
        host_copy[i] = homePatches[i]->patchDataSOA;
    }
    copy_HtoD<PatchDataSOA>(host_copy.data(), d_HostPatchDataSOA, numPatchesHome);
    cudaCheck(cudaDeviceSynchronize());
}

void SequencerCUDA::saveForceCUDASOA_direct(const int maxForceNumber) {
  CUDASequencerKernel->copyForcesToHostSOA(
      numPatchesHome,
      patchData->devData[deviceIndex].d_localPatches,
      maxForceNumber,
      d_f_normal_x,
      d_f_normal_y,
      d_f_normal_z,
      d_f_nbond_x,
      d_f_nbond_y,
      d_f_nbond_z,
      d_f_slow_x,
      d_f_slow_y,
      d_f_slow_z,
      d_HostPatchDataSOA,
      stream
  );
  cudaCheck(cudaStreamSynchronize(stream));
}

void SequencerCUDA::copyPositionsToHost_direct() {
  CUDASequencerKernel->copyPositionsToHostSOA(
      numPatchesHome,
      patchData->devData[deviceIndex].d_localPatches,
      d_pos_x,
      d_pos_y,
      d_pos_z,
      d_HostPatchDataSOA,
      stream
  );
  cudaCheck(cudaStreamSynchronize(stream));
}

void SequencerCUDA::redistributeTip4pForces(
  NodeReduction *reduction,
  const int maxForceNumber,
  const int doVirial) {
  CUDASequencerKernel->redistributeTip4pForces(
    d_f_normal_x, d_f_normal_y, d_f_normal_z,
    d_f_nbond_x, d_f_nbond_y, d_f_nbond_z,
    d_f_slow_x, d_f_slow_y, d_f_slow_z,
    d_lpVirialNormal, d_lpVirialNbond, d_lpVirialSlow,
    d_pos_x, d_pos_y, d_pos_z, d_mass,
    numAtomsHome, doVirial, maxForceNumber, stream
  );
  cudaCheck(cudaStreamSynchronize(stream));
  if (reduction && doVirial) {
    switch (maxForceNumber) {
      case 2:
        copy_DtoH_sync<cudaTensor>(d_lpVirialSlow, lpVirialSlow, 1);
        ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_SLOW, lpVirialSlow[0]);
        cudaCheck(cudaMemset(d_lpVirialSlow, 0, 1 * sizeof(cudaTensor)));
      case 1:
        copy_DtoH_sync<cudaTensor>(d_lpVirialNbond, lpVirialNbond, 1);
        ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NBOND, lpVirialNbond[0]);
        cudaCheck(cudaMemset(d_lpVirialNbond, 0, 1 * sizeof(cudaTensor)));
      case 0:
        copy_DtoH_sync<cudaTensor>(d_lpVirialNormal, lpVirialNormal, 1);
        ADD_TENSOR_OBJECT(reduction, REDUCTION_VIRIAL_NORMAL, lpVirialNormal[0]);
        cudaCheck(cudaMemset(d_lpVirialNormal, 0, 1 * sizeof(cudaTensor)));
    }
  }
}

#endif // NAMD_CUDA
