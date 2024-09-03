#include "NamdTypes.h"
#include "common.h"
#include "Node.h"
#include "ComputeCUDAMgr.h"
#include "PatchData.h"
#include "DeviceCUDA.h"
#include "CudaUtils.h"
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#ifdef WIN32
#define __thread __declspec(thread)
#endif
extern __thread DeviceCUDA *deviceCUDA;

//
// Class constructor
//
ComputeCUDAMgr::ComputeCUDAMgr() {
	// __sdag_init();
  numDevices = 0;
  // numNodesContributed = 0;
  // numDevicesMax = 0;
  cudaPmeOneDevice = NULL;  // XXX is this needed?
}

//
// Class constructor
//
ComputeCUDAMgr::ComputeCUDAMgr(CkMigrateMessage *) {
	// __sdag_init();
  NAMD_bug("ComputeCUDAMgr cannot be migrated");
  numDevices = 0;
  // numNodesContributed = 0;
  // numDevicesMax = 0;
  cudaPmeOneDevice = NULL;  // XXX is this needed?
}

//
// Class destructor
//
ComputeCUDAMgr::~ComputeCUDAMgr() {
  for (int i=0;i < numDevices;i++) {
    if (cudaNonbondedTablesList[i] != NULL) delete cudaNonbondedTablesList[i];
    if (cudaComputeNonbondedList[i] != NULL) delete cudaComputeNonbondedList[i];
#ifdef BONDED_CUDA
    if (computeBondedCUDAList[i] != NULL) delete computeBondedCUDAList[i];
#endif
  }
  delete cudaPmeOneDevice;
}

//
// Initialize manager
// This gets called on rank 0 of each node
//
void ComputeCUDAMgr::initialize(CkQdMsg *msg) {
	if (msg != NULL) delete msg;

	numDevices = deviceCUDA->getDeviceCount();
#ifdef NODEGROUP_FORCE_REGISTER
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    PatchData *pdata = cpdata.ckLocalBranch();
    int ndevs = deviceCUDA->getNumDevice() + 1*deviceCUDA->isGpuReservedPme();
    pdata->devData.resize(numDevices);

    {
      // Pointers to SOA integration data
      allocate_host<bool*>(&(pdata->h_devHasForces),ndevs);
      allocate_host<double*>(&(pdata->h_soa_fb_x),  ndevs);
      allocate_host<double*>(&(pdata->h_soa_fb_y),  ndevs);
      allocate_host<double*>(&(pdata->h_soa_fb_z),  ndevs);

      allocate_host<double*>(&(pdata->h_soa_fn_x),  ndevs);
      allocate_host<double*>(&(pdata->h_soa_fn_y),  ndevs);
      allocate_host<double*>(&(pdata->h_soa_fn_z),  ndevs);

      allocate_host<double*>(&(pdata->h_soa_fs_x),  ndevs);
      allocate_host<double*>(&(pdata->h_soa_fs_y),  ndevs);
      allocate_host<double*>(&(pdata->h_soa_fs_z),  ndevs);

      allocate_host<double*>(&(pdata->h_soa_pos_x), ndevs);
      allocate_host<double*>(&(pdata->h_soa_pos_y), ndevs);
      allocate_host<double*>(&(pdata->h_soa_pos_z), ndevs);

      allocate_host<double*>(&(pdata->h_soa_vel_x), deviceCUDA->getNumDevice());
      allocate_host<double*>(&(pdata->h_soa_vel_y), deviceCUDA->getNumDevice());
      allocate_host<double*>(&(pdata->h_soa_vel_z), deviceCUDA->getNumDevice());

      allocate_host<float*>  (&(pdata->h_soa_charge), deviceCUDA->getNumDevice());

      allocate_host<int*>    (&(pdata->h_soa_id),                   deviceCUDA->getNumDevice());
      allocate_host<int*>    (&(pdata->h_soa_vdwType),              deviceCUDA->getNumDevice());
      allocate_host<int*>    (&(pdata->h_soa_sortOrder),            deviceCUDA->getNumDevice());
      allocate_host<int*>    (&(pdata->h_soa_unsortOrder),          deviceCUDA->getNumDevice());
      allocate_host<double3*>(&(pdata->h_soa_patchCenter),          deviceCUDA->getNumDevice());
      allocate_host<int4*>   (&(pdata->h_soa_migrationDestination), deviceCUDA->getNumDevice());
      allocate_host<int*>    (&(pdata->h_soa_sortSoluteIndex),      deviceCUDA->getNumDevice());

      allocate_host<int*>    (&(pdata->h_soa_partition),            deviceCUDA->getNumDevice());

      allocate_host<FullAtom*>(&(pdata->h_atomdata_AoS), deviceCUDA->getNumDevice());
      allocate_host<CudaLocalRecord*>(&(pdata->h_peer_record), deviceCUDA->getNumDevice());

      allocate_host<int*>(&(pdata->h_tupleCount.bond), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleCount.angle), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleCount.dihedral), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleCount.improper), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleCount.modifiedExclusion), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleCount.exclusion), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleCount.crossterm), deviceCUDA->getNumDevice());

      allocate_host<int*>(&(pdata->h_tupleOffset.bond), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleOffset.angle), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleOffset.dihedral), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleOffset.improper), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleOffset.modifiedExclusion), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleOffset.exclusion), deviceCUDA->getNumDevice());
      allocate_host<int*>(&(pdata->h_tupleOffset.crossterm), deviceCUDA->getNumDevice());

      allocate_host<CudaBondStage*>(&(pdata->h_tupleDataStage.bond), deviceCUDA->getNumDevice());
      allocate_host<CudaAngleStage*>(&(pdata->h_tupleDataStage.angle), deviceCUDA->getNumDevice());
      allocate_host<CudaDihedralStage*>(&(pdata->h_tupleDataStage.dihedral), deviceCUDA->getNumDevice());
      allocate_host<CudaDihedralStage*>(&(pdata->h_tupleDataStage.improper), deviceCUDA->getNumDevice());
      allocate_host<CudaExclusionStage*>(&(pdata->h_tupleDataStage.modifiedExclusion), deviceCUDA->getNumDevice());
      allocate_host<CudaExclusionStage*>(&(pdata->h_tupleDataStage.exclusion), deviceCUDA->getNumDevice());
      allocate_host<CudaCrosstermStage*>(&(pdata->h_tupleDataStage.crossterm), deviceCUDA->getNumDevice());
    }

    // Allocate the work queues
    allocate_host<unsigned int*>(&(pdata->d_queues), ndevs);
    allocate_host<unsigned int>(&(pdata->d_queueCounters), ndevs);

    cudaCheck(cudaMemset(pdata->d_queueCounters, 0, sizeof(unsigned int)*ndevs));

    pdata->migrationFlagPerDevice.resize(deviceCUDA->getNumDevice());

    pdata->tupleReallocationFlagPerDevice.resize(deviceCUDA->getNumDevice());
    pdata->atomReallocationFlagPerDevice.resize(deviceCUDA->getNumDevice());
    pdata->maxNumBonds.store(0);
    pdata->maxNumAngles.store(0);
    pdata->maxNumDihedrals.store(0);
    pdata->maxNumImpropers.store(0);
    pdata->maxNumModifiedExclusions.store(0);
    pdata->maxNumExclusions.store(0);
    pdata->maxNumCrossterms.store(0);
    pdata->devicePatchMapFlag.resize(CkNumPes(), 0);
#ifdef NAMD_NCCL_ALLREDUCE
    // Allocate NCCL-related stuff
    deviceCUDA->setupNcclUniqueId();
    // After I do this, I can go ahead and register it in patchData
    pdata->ncclId = deviceCUDA->getNcclUniqueId(); // registered in ngroup
#endif
#endif

  // Create pointers to devices
  cudaNonbondedTablesList.resize(numDevices, NULL);
  cudaComputeNonbondedList.resize(numDevices, NULL);
#ifdef BONDED_CUDA
  computeBondedCUDAList.resize(numDevices, NULL);
#endif
  if (cudaPmeOneDevice != NULL) delete cudaPmeOneDevice;
  cudaPmeOneDevice = NULL;

  // Create CUDA non-bonded tables for all devices that are used for computation
  for (int i=0;i < deviceCUDA->getNumDevice();i++) {
    int deviceID = deviceCUDA->getDeviceIDbyRank(i);
    cudaNonbondedTablesList[deviceID] = new CudaNonbondedTables(deviceID);
  }

  

}

//
// Update nonbonded tables
// Should be called only on rank 0 of each node
//
void ComputeCUDAMgr::update() {
  if ( CkMyRank() ) NAMD_bug("ComputeCUDAMgr::update() should be called only by rank 0");
  for (int i=0;  i < deviceCUDA->getNumDevice();  i++) {
    int deviceID = deviceCUDA->getDeviceIDbyRank(i);
    // calls update function from CudaNonbondedTables
    cudaNonbondedTablesList[deviceID]->updateTables();
  }
}

ComputeCUDAMgr* ComputeCUDAMgr::getComputeCUDAMgr() {
  // Get pointer to ComputeCUDAMgr on this node
  CProxy_ComputeCUDAMgr computeCUDAMgrProxy = CkpvAccess(BOCclass_group).computeCUDAMgr;
  ComputeCUDAMgr* computeCUDAMgr = computeCUDAMgrProxy.ckLocalBranch();
  if (computeCUDAMgr == NULL)
    NAMD_bug("getComputeCUDAMgr, unable to locate local branch of BOC entry ComputeCUDAMgr");
  return computeCUDAMgr;
}

CudaPmeOneDevice* ComputeCUDAMgr::createCudaPmeOneDevice() {
  // initialize pmeGrid from simParams
  SimParameters *simParams = Node::Object()->simParameters;
  PmeGrid pmeGrid;
  pmeGrid.K1 = simParams->PMEGridSizeX;
  pmeGrid.K2 = simParams->PMEGridSizeY;
  pmeGrid.K3 = simParams->PMEGridSizeZ;
  pmeGrid.order = simParams->PMEInterpOrder;
  pmeGrid.dim2 = pmeGrid.K2;
  pmeGrid.dim3 = 2 * (pmeGrid.K3/2 + 1);
  // override settings for PME pencils
  pmeGrid.xBlocks = 1;
  pmeGrid.yBlocks = 1;
  pmeGrid.zBlocks = 1;
  pmeGrid.block1 = pmeGrid.K1;
  pmeGrid.block2 = pmeGrid.K2;
  pmeGrid.block3 = pmeGrid.K3;
  // use shared deviceID class
  int deviceID = 0;
  int deviceIndex = 0;
#ifdef NODEGROUP_FORCE_REGISTER
  deviceID    = deviceCUDA->getPmeDevice();
  deviceIndex = deviceCUDA->getPmeDeviceIndex();
#endif
  if (cudaPmeOneDevice != NULL) delete cudaPmeOneDevice;
  cudaPmeOneDevice = new CudaPmeOneDevice(pmeGrid, deviceID, deviceIndex);
  return cudaPmeOneDevice;
}

CudaPmeOneDevice* ComputeCUDAMgr::getCudaPmeOneDevice() {
  return cudaPmeOneDevice;
}

//
// Creates CudaComputeNonbonded object
//
CudaComputeNonbonded* ComputeCUDAMgr::createCudaComputeNonbonded(ComputeID c) {
  int deviceID = deviceCUDA->getDeviceID();
  if (cudaComputeNonbondedList.at(deviceID) != NULL)
    NAMD_bug("ComputeCUDAMgr::createCudaComputeNonbonded called twice");
  if (cudaNonbondedTablesList.at(deviceID) == NULL)
    NAMD_bug("ComputeCUDAMgr::createCudaComputeNonbonded, non-bonded CUDA tables not created");
  //bool doStreaming = !deviceCUDA->getNoStreaming() && !Node::Object()->simParameters->GBISOn && !Node::Object()->simParameters->CUDASOAintegrate;
  bool doStreaming = !deviceCUDA->getNoStreaming() && !Node::Object()->simParameters->GBISOn && !Node::Object()->simParameters->CUDASOAintegrate;
  cudaComputeNonbondedList[deviceID] = new CudaComputeNonbonded(c, deviceID, *cudaNonbondedTablesList[deviceID], doStreaming);
  return cudaComputeNonbondedList[deviceID];
}

//
// Returns CudaComputeNonbonded for this Pe
//
CudaComputeNonbonded* ComputeCUDAMgr::getCudaComputeNonbonded() {
  // Get device ID for this Pe
  int deviceID = deviceCUDA->getDeviceID();
  CudaComputeNonbonded* p = cudaComputeNonbondedList[deviceID];
  if (p == NULL)
    NAMD_bug("ComputeCUDAMgr::getCudaComputeNonbonded(), device not created yet");
  return p;
}

#ifdef BONDED_CUDA
//
// Creates ComputeBondedCUDA object
//
ComputeBondedCUDA* ComputeCUDAMgr::createComputeBondedCUDA(ComputeID c, ComputeMgr* computeMgr) {
  int deviceID = deviceCUDA->getDeviceID();
  if (computeBondedCUDAList.at(deviceID) != NULL)
    NAMD_bug("ComputeCUDAMgr::createComputeBondedCUDA called twice");
  if (cudaNonbondedTablesList.at(deviceID) == NULL)
    NAMD_bug("ComputeCUDAMgr::createCudaComputeNonbonded, non-bonded CUDA tables not created");
  computeBondedCUDAList[deviceID] = new ComputeBondedCUDA(c, computeMgr, deviceID, *cudaNonbondedTablesList[deviceID]);
  return computeBondedCUDAList[deviceID];
}

//
// Returns ComputeBondedCUDA for this Pe
//
ComputeBondedCUDA* ComputeCUDAMgr::getComputeBondedCUDA() {
  // Get device ID for this Pe
  int deviceID = deviceCUDA->getDeviceID();
  ComputeBondedCUDA* p = computeBondedCUDAList[deviceID];
  if (p == NULL)
    NAMD_bug("ComputeCUDAMgr::getComputeBondedCUDA(), device not created yet");
  return p;
}
#endif // BONDED_CUDA

#endif // NAMD_CUDA

#include "ComputeCUDAMgr.def.h"
