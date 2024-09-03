#include <iomanip>
#include "Node.h"
#include "Priorities.h"
#include "ComputeNonbondedUtil.h"
#include "CudaPmeSolverUtil.h"
#include "ComputePmeCUDAMgr.h"
#include "ComputePmeCUDAMgr.decl.h"
#include "CudaPmeSolver.h"
#include "DeviceCUDA.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#ifdef WIN32
#define __thread __declspec(thread)
#endif
extern __thread DeviceCUDA *deviceCUDA;
//#define DISABLE_P2P

void CudaPmePencilXYZ::initialize(CudaPmeXYZInitMsg *msg) {
  pmeGrid = msg->pmeGrid;
  delete msg;
}

//
// CUDA specific initialization
//
void CudaPmePencilXYZ::initializeDevice(InitDeviceMsg *msg) {
  // Store device proxy
  deviceProxy = msg->deviceProxy;
  delete msg;
  int deviceID = deviceProxy.ckLocalBranch()->getDeviceID();
  cudaStream_t stream = deviceProxy.ckLocalBranch()->getStream();
  CProxy_ComputePmeCUDAMgr mgrProxy = deviceProxy.ckLocalBranch()->getMgrProxy();
  // Setup fftCompute and pmeKSpaceCompute
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (deviceProxy.ckLocalBranch()->isGridEnabled(iGrid) == true) {
      fftComputes[iGrid] = new CudaFFTCompute(deviceID, stream);
      pmeKSpaceComputes[iGrid] = new CudaPmeKSpaceCompute(pmeGrid, Perm_cX_Y_Z, 0, 0, ComputeNonbondedUtil::ewaldcof, deviceID, stream, iGrid);
      energyReady[iGrid] = 0;
    } else {
      fftComputes[iGrid] = NULL;
      pmeKSpaceComputes[iGrid] = NULL;
      energyReady[iGrid] = -1;
    }
  }
}

void CudaPmePencilXYZ::backwardDone() {
  deviceProxy[CkMyNode()].gatherForce();
//   ((CudaPmeKSpaceCompute *)pmeKSpaceComputes[0])->energyAndVirialSetCallback(this);
//   ((CudaPmeKSpaceCompute *)pmeKSpaceCompute2)->energyAndVirialSetCallback(this);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeKSpaceComputes[iGrid] != NULL)
      ((CudaPmeKSpaceCompute *)pmeKSpaceComputes[iGrid])->energyAndVirialSetCallback(this);
  }

  // ((CudaPmeKSpaceCompute *)pmeKSpaceCompute)->waitEnergyAndVirial();
  // submitReductions();
  // deviceProxy[CkMyNode()].gatherForce();
}

void CudaPmePencilXYZ::energyAndVirialDone(unsigned int iGrid) {
  submitReductions(iGrid);
  // deviceProxy[CkMyNode()].gatherForce();
}

//###########################################################################
//###########################################################################
//###########################################################################

void CudaPmePencilXY::initialize(CudaPmeXYInitMsg *msg) {
  pmeGrid = msg->pmeGrid;
  pmePencilZ = msg->pmePencilZ;
  zMap = msg->zMap;

  delete msg;

  initBlockSizes();
}

CudaPmePencilXY::~CudaPmePencilXY() {
  if (eventCreated) cudaCheck(cudaEventDestroy(event));
}

//
// CUDA specific initialization
//
void CudaPmePencilXY::initializeDevice(InitDeviceMsg *msg) {
  // Store device proxy
  deviceProxy = msg->deviceProxy;
  delete msg;
  deviceID = deviceProxy.ckLocalBranch()->getDeviceID();
  stream = deviceProxy.ckLocalBranch()->getStream();
  CProxy_ComputePmeCUDAMgr mgrProxy = deviceProxy.ckLocalBranch()->getMgrProxy();
  // Setup fftCompute and pmeKSpaceCompute
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (deviceProxy.ckLocalBranch()->isGridEnabled(iGrid) == true) {
      fftComputes[iGrid] = new CudaFFTCompute(deviceID, stream);
      pmeTransposes[iGrid] = new CudaPmeTranspose(pmeGrid, Perm_cX_Y_Z, 0, thisIndex.z, deviceID, stream);
    } else {
      fftComputes[iGrid] = NULL;
      pmeTransposes[iGrid] = NULL;
    }
  }

  deviceBuffers.resize(pmeGrid.xBlocks, DeviceBuffer(-1, false));
  numDeviceBuffers = 0;

  // Create event. NOTE: Events are tied to devices, hence the cudaSetDevice() here
  cudaCheck(cudaSetDevice(deviceID));
  cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  eventCreated = true;

/*
  bool useMultiGPUfft = true;
  bool allDeviceOnSameNode = true;
  for (int x=0;x < pmeGrid.xBlocks;x++) {
    int pe = zMap.ckLocalBranch()->procNum(0, CkArrayIndex3D(x,0,0));
    allDeviceOnSameNode &= (CkNodeOf(pe) == CkMyNode());
  }

  if (useMultiGPUfft && allDeviceOnSameNode && pmeGrid.xBlocks > 1) {
  // WARNING: code may be incomplete here!
  // CHC: Assuming there are two GPUs on the same node and we use:
  //        PMEGridSpacing 2.0
  //        PMEPencilsX 2  
  //        PMEPencilsY 1
  //        PMEPencilsZ 1
  //      and running NAMD with all GPUs and two CPU threads,
  //      this "if" statement is satisfied


  } else {
*/

  for (int x=0;x < pmeGrid.xBlocks;x++) {
    int pe = zMap.ckLocalBranch()->procNum(0, CkArrayIndex3D(x,0,0));
    if (CkNodeOf(pe) == CkMyNode()) {
      // Get device ID on a device on this node
      int deviceID0 = mgrProxy.ckLocalBranch()->getDeviceIDPencilZ(x, 0);
      // Check for Peer-to-Peer access
      int canAccessPeer = 0;
      if (deviceID != deviceID0) {
        cudaCheck(cudaSetDevice(deviceID));
        cudaCheck(cudaDeviceCanAccessPeer(&canAccessPeer, deviceID, deviceID0));
#ifdef DISABLE_P2P
        canAccessPeer = 0;
#endif
        if (canAccessPeer) {
          unsigned int flags = 0;
          cudaCheck(cudaDeviceEnablePeerAccess(deviceID0, flags));
          // fprintf(stderr, "device %d can access device %d\n", deviceID, deviceID0);
        }
      }
      numDeviceBuffers++;
      // CHC: I have tried to use deviceID instead of deviceID0, but NAMD still crashes.
      deviceBuffers[x] = DeviceBuffer(deviceID0, canAccessPeer);
      pmePencilZ(x,0,0).getDeviceBuffer(thisIndex.z, (deviceID0 == deviceID) || canAccessPeer, thisProxy);
    }
  }

  // }

}

//
// CUDA specific start
//
void CudaPmePencilXY::start(const CkCallback &cb) {
  thisProxy[thisIndex].recvDeviceBuffers(cb);
}

void CudaPmePencilXY::setDeviceBuffers() {
  std::array<std::vector<float2*>, NUM_GRID_MAX> dataPtrsGrid;
//   std::vector<float2*> data2Ptrs(pmeGrid.xBlocks, (float2*)0);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    dataPtrsGrid[iGrid] = std::vector<float2*>(pmeGrid.xBlocks, (float2*)0);
    for (int x=0;x < pmeGrid.xBlocks;x++) {
      if (deviceBuffers[x].dataGrid[iGrid] != NULL) {
        if (deviceBuffers[x].deviceID == deviceID || deviceBuffers[x].isPeerDevice) {
          // Device buffer on same device => directly transpose into destination pencil
          // dataPtrs[x] = deviceBuffers[x].dataGrid[0];
          dataPtrsGrid[iGrid][x] = deviceBuffers[x].dataGrid[iGrid];
          // Otherwise, when device buffer on different device on same node => transpose locally and then 
          // use cudaMemcpy3DPeerAsync to perform the copying
          // WARNING: code may be incomplete here!
        }
      }
      if (pmeTransposes[iGrid] != NULL) {
        ((CudaPmeTranspose *)pmeTransposes[iGrid])->setDataPtrsZXY(dataPtrsGrid[iGrid], (float2 *)fftComputes[iGrid]->getDataDst());
      }
    }
  }
}

std::array<float2*, NUM_GRID_MAX> CudaPmePencilXY::getData(const int i, const bool sameDevice) {
  std::array<float2*, NUM_GRID_MAX> data_grid;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) {
#ifndef P2P_ENABLE_3D
      if (sameDevice) {
        int i0, i1, j0, j1, k0, k1;
        getBlockDim(pmeGrid, Perm_cX_Y_Z, i, 0, 0, i0, i1, j0, j1, k0, k1);
        data_grid[iGrid] = (float2 *)fftComputes[iGrid]->getDataDst() + i0;
      } else {
        data_grid[iGrid] = ((CudaPmeTranspose *)pmeTransposes[iGrid])->getBuffer(i);
      }
#else
      int i0, i1, j0, j1, k0, k1;
      getBlockDim(pmeGrid, Perm_cX_Y_Z, i, 0, 0, i0, i1, j0, j1, k0, k1);
      data_grid[iGrid] = (float2 *)fftComputes[iGrid]->getDataDst() + i0;
#endif
    } else {
      data_grid[iGrid] = NULL;
    }
  }
  return data_grid;
}

void CudaPmePencilXY::backwardDone() {
  deviceProxy[CkMyNode()].gatherForce();
}

void CudaPmePencilXY::forwardDone() {
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeTransposes[iGrid] != NULL) {
      // Transpose locally
      pmeTransposes[iGrid]->transposeXYZtoZXY((float2 *)fftComputes[iGrid]->getDataDst());
      // Direct Device-To-Device communication within node
      if (numDeviceBuffers > 0) {
        // Copy data
        for (int x=0;x < pmeGrid.xBlocks;x++) {
          if (deviceBuffers[x].dataGrid[iGrid] != NULL) {
            if (deviceBuffers[x].deviceID != deviceID && !deviceBuffers[x].isPeerDevice) {
              ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataToPeerDeviceZXY(x, deviceBuffers[x].deviceID,
                Perm_Z_cX_Y, deviceBuffers[x].dataGrid[iGrid]);
            }
          }
        }

        // Record event for this pencil
        cudaCheck(cudaEventRecord(event, stream));
        // Send empty message
        for (int x=0;x < pmeGrid.xBlocks;x++) {
          if (deviceBuffers[x].dataGrid[iGrid] != NULL) {
            PmeBlockMsg* msg = new (0, PRIORITY_SIZE) PmeBlockMsg();
            msg->dataSize = 0;
            msg->x = x;
            msg->y = thisIndex.y;
            msg->z = thisIndex.z;
            msg->doEnergy = doEnergy;
            msg->doVirial = doVirial;
            msg->lattice  = lattice;
            msg->numStrayAtoms = numStrayAtoms;
            msg->grid = iGrid;
            msg->simulationStep = simulationStep;
            pmePencilZ(x,0,0).recvBlock(msg);
          }
        }
      }

      // Copy-Via-Host communication
      for (int x=0;x < pmeGrid.xBlocks;x++) {
        if (deviceBuffers[x].dataGrid[iGrid] == NULL) {
          PmeBlockMsg* msg = new (blockSizes[x], PRIORITY_SIZE) PmeBlockMsg();
          msg->dataSize = blockSizes[x];
          msg->x = x;
          msg->y = thisIndex.y;
          msg->z = thisIndex.z;
          msg->doEnergy = doEnergy;
          msg->doVirial = doVirial;
          msg->lattice  = lattice;
          msg->numStrayAtoms = numStrayAtoms;
          msg->simulationStep = simulationStep;
          msg->grid = iGrid;
          ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataDeviceToHost(x, msg->data, msg->dataSize);
          ((CudaPmeTranspose *)pmeTransposes[iGrid])->waitStreamSynchronize();
          pmePencilZ(x,0,0).recvBlock(msg);
        }
      }
    }
  }
}

void CudaPmePencilXY::recvDataFromZ(PmeBlockMsg *msg) {
  if (msg->dataSize != 0) {
    // CHC: is the checking of null pointer redundant?
    // Buffer is coming from a different node
    ((CudaPmeTranspose *)(pmeTransposes[msg->grid]))->copyDataHostToDevice(msg->x, msg->data, (float2 *)fftComputes[msg->grid]->getDataDst());
  } else {
    // Buffer is coming from the same node
    // Wait for event that was recorded on the sending pencil
    // device ID = deviceBuffers[msg->x].deviceID
    // event     = deviceBuffers[msg->x].event
    cudaCheck(cudaStreamWaitEvent(stream, deviceBuffers[msg->x].event, 0));
#ifndef P2P_ENABLE_3D
    if (deviceBuffers[msg->x].dataGrid[msg->grid] != NULL && deviceBuffers[msg->x].deviceID != deviceID && !deviceBuffers[msg->x].isPeerDevice) {
      // Data is in temporary device buffer, copy it into final fft-buffer
      ((CudaPmeTranspose *)(pmeTransposes[msg->grid]))->copyDataDeviceToDevice(msg->x, (float2 *)fftComputes[msg->grid]->getDataDst());
    }
#endif
  }
  delete msg;
}

//###########################################################################
//###########################################################################
//###########################################################################

void CudaPmePencilX::initialize(CudaPmeXInitMsg *msg) {
  pmeGrid = msg->pmeGrid;
  pmePencilY = msg->pmePencilY;
  yMap = msg->yMap;

  delete msg;

  initBlockSizes();

}

CudaPmePencilX::~CudaPmePencilX() {
  if (eventCreated) cudaCheck(cudaEventDestroy(event));
}

//
// CUDA specific initialization
//
void CudaPmePencilX::initializeDevice(InitDeviceMsg *msg) {
  // Store device proxy
  deviceProxy = msg->deviceProxy;
  delete msg;
  deviceID = deviceProxy.ckLocalBranch()->getDeviceID();
  stream = deviceProxy.ckLocalBranch()->getStream();
  CProxy_ComputePmeCUDAMgr mgrProxy = deviceProxy.ckLocalBranch()->getMgrProxy();
  // Setup fftCompute and pmeKSpaceCompute
//   fftCompute = new CudaFFTCompute(deviceID, stream);
//   fftCompute2 = new CudaFFTCompute(deviceID, stream);
//   pmeTranspose = new CudaPmeTranspose(pmeGrid, Perm_cX_Y_Z, thisIndex.y, thisIndex.z, deviceID, stream);
//   pmeTranspose2 = new CudaPmeTranspose(pmeGrid, Perm_cX_Y_Z, thisIndex.y, thisIndex.z, deviceID, stream);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (deviceProxy.ckLocalBranch()->isGridEnabled(iGrid) == true) {
      fftComputes[iGrid] = new CudaFFTCompute(deviceID, stream);
      pmeTransposes[iGrid] = new CudaPmeTranspose(pmeGrid, Perm_cX_Y_Z, thisIndex.y, thisIndex.z, deviceID, stream);
    } else {
      fftComputes[iGrid] = NULL;
      pmeTransposes[iGrid] = NULL;
    }
  }

  // Create event. NOTE: Events are tied to devices, hence the cudaSetDevice() here
  cudaCheck(cudaSetDevice(deviceID));
  cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  eventCreated = true;

  deviceBuffers.resize(pmeGrid.xBlocks, DeviceBuffer(-1, false));
  numDeviceBuffers = 0;

  for (int x=0;x < pmeGrid.xBlocks;x++) {
    int pe = yMap.ckLocalBranch()->procNum(0, CkArrayIndex3D(x,0,thisIndex.z));
    if (CkNodeOf(pe) == CkMyNode()) {
      int deviceID0 = mgrProxy.ckLocalBranch()->getDeviceIDPencilY(x, thisIndex.z);
      numDeviceBuffers++;
      deviceBuffers[x] = DeviceBuffer(deviceID0, false);
      pmePencilY(x,0,thisIndex.z).getDeviceBuffer(thisIndex.y, (deviceID0 == deviceID), thisProxy);
    }
  }

}

//
// CUDA specific start
//
void CudaPmePencilX::start(const CkCallback &cb) {
  thisProxy[thisIndex].recvDeviceBuffers(cb);
}

//
// Setup direct device buffers
//
void CudaPmePencilX::setDeviceBuffers() {
  std::array<std::vector<float2*>, NUM_GRID_MAX> dataPtrsGrid;
//   std::vector<float2*> dataPtrs(pmeGrid.xBlocks, (float2*)0);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    dataPtrsGrid[iGrid] = std::vector<float2*>(pmeGrid.xBlocks, (float2*)0);
    for (int x=0;x < pmeGrid.xBlocks;x++) {
      if (deviceBuffers[x].dataGrid[iGrid] != NULL) {
        if (deviceBuffers[x].deviceID == deviceID) {
          // Device buffer on same device => directly transpose into destination pencil
          dataPtrsGrid[iGrid][x] = deviceBuffers[x].dataGrid[iGrid];
          // Otherwise, when device buffer on different device on same node => transpose locally and then 
          // use cudaMemcpy3DPeerAsync to perform the copying
        }
      }
    }
    if (pmeTransposes[iGrid] != NULL) {
      ((CudaPmeTranspose *)pmeTransposes[iGrid])->setDataPtrsYZX(dataPtrsGrid[iGrid], (float2 *)fftComputes[iGrid]->getDataDst());
    }
  }
}

std::array<float2*, NUM_GRID_MAX> CudaPmePencilX::getData(const int i, const bool sameDevice) {
  std::array<float2*, NUM_GRID_MAX> data_grid;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) {
#ifndef P2P_ENABLE_3D
      if (sameDevice) {
        int i0, i1, j0, j1, k0, k1;
        getBlockDim(pmeGrid, Perm_cX_Y_Z, i, 0, 0, i0, i1, j0, j1, k0, k1);
        data_grid[iGrid] = (float2 *)fftComputes[iGrid]->getDataDst() + i0;
      } else {
        data_grid[iGrid] = ((CudaPmeTranspose *)pmeTransposes[iGrid])->getBuffer(i);
      }
#else
      int i0, i1, j0, j1, k0, k1;
      getBlockDim(pmeGrid, Perm_cX_Y_Z, i, 0, 0, i0, i1, j0, j1, k0, k1);
      data_grid[iGrid] = (float2 *)fftComputes[iGrid]->getDataDst() + i0;
#endif
    } else {
      data_grid[iGrid] = NULL;
    }
  }
  return data_grid;
}

void CudaPmePencilX::backwardDone() {
  deviceProxy[CkMyNode()].gatherForce();
}

void CudaPmePencilX::forwardDone() {
  if (pmeTransposes[0] == NULL)
    NAMD_bug("CudaPmePencilX::forwardDone, pmeTranspose not initialized");
  if (blockSizes.size() == 0)
    NAMD_bug("CudaPmePencilX::forwardDone, blockSizes not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeTransposes[iGrid] != NULL) {
      // Transpose locally
      pmeTransposes[iGrid]->transposeXYZtoYZX((float2 *)fftComputes[iGrid]->getDataDst());

      // Send data to y-pencils that share the same z-coordinate. There are pmeGrid.xBlocks of them
      // Direct-Device-To-Device communication
      if (numDeviceBuffers > 0) {
        // Copy data
        for (int x=0;x < pmeGrid.xBlocks;x++) {
          if (deviceBuffers[x].dataGrid[iGrid] != NULL) {
            if (deviceBuffers[x].deviceID != deviceID) {
              ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataToPeerDeviceYZX(x, deviceBuffers[x].deviceID,
                Perm_Y_Z_cX, deviceBuffers[x].dataGrid[iGrid]);
            }
          }
        }

        // Record event for this pencil
        cudaCheck(cudaEventRecord(event, stream));
        // Send empty messages
        for (int x=0;x < pmeGrid.xBlocks;x++) {
          if (deviceBuffers[x].dataGrid[iGrid] != NULL) {
            PmeBlockMsg* msg = new (0, PRIORITY_SIZE) PmeBlockMsg();
            msg->dataSize = 0;
            msg->x = x;
            msg->y = thisIndex.y;
            msg->z = thisIndex.z;
            msg->doEnergy = doEnergy;
            msg->doVirial = doVirial;
            msg->lattice  = lattice;
            msg->numStrayAtoms = numStrayAtoms;
            msg->simulationStep = simulationStep;
            msg->grid = iGrid;
            pmePencilY(x,0,thisIndex.z).recvBlock(msg);     
          }
        }
      }

      // Copy-To-Host communication
      for (int x=0;x < pmeGrid.xBlocks;x++) {
        if (deviceBuffers[x].dataGrid[iGrid] == NULL) {
          PmeBlockMsg* msg = new (blockSizes[x], PRIORITY_SIZE) PmeBlockMsg();
          msg->dataSize = blockSizes[x];
          msg->x = x;
          msg->y = thisIndex.y;
          msg->z = thisIndex.z;
          msg->doEnergy = doEnergy;
          msg->doVirial = doVirial;
          msg->lattice  = lattice;
          msg->numStrayAtoms = numStrayAtoms;
          msg->simulationStep = simulationStep;
          msg->grid = iGrid;
          ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataDeviceToHost(x, msg->data, msg->dataSize);
          ((CudaPmeTranspose *)pmeTransposes[iGrid])->waitStreamSynchronize();
          pmePencilY(x,0,thisIndex.z).recvBlock(msg);
        }
      } 
    }
  }
}

void CudaPmePencilX::recvDataFromY(PmeBlockMsg *msg) {
  if (msg->dataSize != 0) {
    // Buffer is coming from a different node
    ((CudaPmeTranspose *)pmeTransposes[msg->grid])->copyDataHostToDevice(msg->x, msg->data, (float2 *)fftComputes[msg->grid]->getDataDst());
  } else {
    // Buffer is coming from the same node
    // Wait for event that was recorded on the sending pencil
    // device ID = deviceBuffers[msg->x].deviceID
    // event     = deviceBuffers[msg->x].event
    cudaCheck(cudaStreamWaitEvent(stream, deviceBuffers[msg->x].event, 0));
#ifndef P2P_ENABLE_3D
    if (deviceBuffers[msg->x].dataGrid[msg->grid] != NULL && deviceBuffers[msg->x].deviceID != deviceID) {
      // Data is in temporary device buffer, copy it into final fft-buffer
      ((CudaPmeTranspose *)pmeTransposes[msg->grid])->copyDataDeviceToDevice(msg->x, (float2 *)fftComputes[msg->grid]->getDataDst());
    }
#endif
  }
  delete msg;
}

//###########################################################################
//###########################################################################
//###########################################################################

void CudaPmePencilY::initialize(CudaPmeXInitMsg *msg) {
  pmeGrid = msg->pmeGrid;
  pmePencilX = msg->pmePencilX;
  pmePencilZ = msg->pmePencilZ;
  xMap = msg->xMap;
  zMap = msg->zMap;

  delete msg;

  initBlockSizes();
}

CudaPmePencilY::~CudaPmePencilY() {
  if (eventCreated) cudaCheck(cudaEventDestroy(event));
}

//
// CUDA specific initialization
//
void CudaPmePencilY::initializeDevice(InitDeviceMsg2 *msg) {
  // Get device proxy
  CProxy_ComputePmeCUDADevice deviceProxy = msg->deviceProxy;
  deviceID = msg->deviceID;
  stream = msg->stream;
  CProxy_ComputePmeCUDAMgr mgrProxy = msg->mgrProxy;
  delete msg;
  // deviceID = deviceProxy.ckLocalBranch()->getDeviceID();
  // cudaStream_t stream = deviceProxy.ckLocalBranch()->getStream();
  // CProxy_ComputePmeCUDAMgr mgrProxy = deviceProxy.ckLocalBranch()->getMgrProxy();
  // Setup fftCompute and pmeKSpaceCompute
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (deviceProxy.ckLocalBranch()->isGridEnabled(iGrid) == true) {
      fftComputes[iGrid] = new CudaFFTCompute(deviceID, stream);
      pmeTransposes[iGrid] = new CudaPmeTranspose(pmeGrid, Perm_Y_Z_cX, thisIndex.z, thisIndex.x, deviceID, stream);
    } else {
      fftComputes[iGrid] = NULL;
      pmeTransposes[iGrid] = NULL;
    }
  }
  
  // Create event. NOTE: Events are tied to devices, hence the cudaSetDevice() here
  cudaCheck(cudaSetDevice(deviceID));
  cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  eventCreated = true;

  deviceBuffersZ.resize(pmeGrid.yBlocks, DeviceBuffer(-1, false));
  deviceBuffersX.resize(pmeGrid.yBlocks, DeviceBuffer(-1, false));
  numDeviceBuffersZ = 0;
  numDeviceBuffersX = 0;

  for (int y=0;y < pmeGrid.yBlocks;y++) {
    int pe;
    pe = zMap.ckLocalBranch()->procNum(0, CkArrayIndex3D(thisIndex.x, y, 0));
    if (CkNodeOf(pe) == CkMyNode()) {
      int deviceID0 = mgrProxy.ckLocalBranch()->getDeviceIDPencilZ(thisIndex.x, y);
      numDeviceBuffersZ++;
      deviceBuffersZ[y] = DeviceBuffer(deviceID0, false);
      pmePencilZ(thisIndex.x, y, 0).getDeviceBuffer(thisIndex.z, (deviceID0 == deviceID), thisProxy);
    }
    pe = xMap.ckLocalBranch()->procNum(0, CkArrayIndex3D(0, y, thisIndex.z));
    if (CkNodeOf(pe) == CkMyNode()) {
      int deviceID0 = mgrProxy.ckLocalBranch()->getDeviceIDPencilX(y, thisIndex.z);
      numDeviceBuffersX++;
      deviceBuffersX[y] = DeviceBuffer(deviceID0, false);
      pmePencilX(0, y, thisIndex.z).getDeviceBuffer(thisIndex.x, (deviceID0 == deviceID), thisProxy);
    }
  }

}

//
// CUDA specific start
//
void CudaPmePencilY::start(const CkCallback &cb) {
  thisProxy[thisIndex].recvDeviceBuffers(cb);
}

//
// Setup direct device buffers
//
void CudaPmePencilY::setDeviceBuffers() {
  std::array<std::vector<float2*>, NUM_GRID_MAX> dataPtrsYZXGrid;
  std::array<std::vector<float2*>, NUM_GRID_MAX> dataPtrsZXYGrid;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    dataPtrsYZXGrid[iGrid] = std::vector<float2*>(pmeGrid.yBlocks, (float2*)0);
    dataPtrsZXYGrid[iGrid] = std::vector<float2*>(pmeGrid.yBlocks, (float2*)0);
    for (int y=0;y < pmeGrid.yBlocks;y++) {
      if (deviceBuffersZ[y].dataGrid[iGrid] != NULL) {
        if (deviceBuffersZ[y].deviceID == deviceID) {
          dataPtrsYZXGrid[iGrid][y] = deviceBuffersZ[y].dataGrid[iGrid];
        }
      }
      if (deviceBuffersX[y].dataGrid[iGrid] != NULL) {
        if (deviceBuffersX[y].deviceID == deviceID) {
          dataPtrsZXYGrid[iGrid][y] = deviceBuffersX[y].dataGrid[iGrid];
        }
      }
    }
    if (pmeTransposes[iGrid] != NULL) {
      ((CudaPmeTranspose *)pmeTransposes[iGrid])->setDataPtrsYZX(dataPtrsYZXGrid[iGrid], (float2 *)fftComputes[iGrid]->getDataDst());
      ((CudaPmeTranspose *)pmeTransposes[iGrid])->setDataPtrsZXY(dataPtrsZXYGrid[iGrid], (float2 *)fftComputes[iGrid]->getDataSrc());
    }
  }
}

std::array<float2*, NUM_GRID_MAX> CudaPmePencilY::getDataForX(const int i, const bool sameDevice) {
  std::array<float2*, NUM_GRID_MAX> data_grid;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) {
#ifndef P2P_ENABLE_3D
      if (sameDevice) {
        int i0, i1, j0, j1, k0, k1;
        getBlockDim(pmeGrid, Perm_Y_Z_cX, i, 0, 0, i0, i1, j0, j1, k0, k1);
        data_grid[iGrid] = (float2 *)fftComputes[iGrid]->getDataSrc() + i0;
      } else {
        data_grid[iGrid] = ((CudaPmeTranspose *)pmeTransposes[iGrid])->getBuffer(i);
      }
#else
      int i0, i1, j0, j1, k0, k1;
      getBlockDim(pmeGrid, Perm_Y_Z_cX, i, 0, 0, i0, i1, j0, j1, k0, k1);
      data_grid[iGrid] = (float2 *)fftComputes[iGrid]->getDataSrc() + i0;
#endif
    } else {
      data_grid[iGrid] = NULL;
    }
  }
  return data_grid;
}

std::array<float2*, NUM_GRID_MAX> CudaPmePencilY::getDataForZ(const int i, const bool sameDevice) {
  std::array<float2*, NUM_GRID_MAX> data_grid;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) {
#ifndef P2P_ENABLE_3D
      if (sameDevice) {
        int i0, i1, j0, j1, k0, k1;
        getBlockDim(pmeGrid, Perm_Y_Z_cX, i, 0, 0, i0, i1, j0, j1, k0, k1);
        data_grid[iGrid] = (float2 *)fftComputes[iGrid]->getDataDst() + i0;
      } else {
        data_grid[iGrid] = ((CudaPmeTranspose *)pmeTransposes[iGrid])->getBuffer(i);
      }
#else
      int i0, i1, j0, j1, k0, k1;
      getBlockDim(pmeGrid, Perm_Y_Z_cX, i, 0, 0, i0, i1, j0, j1, k0, k1);
      data_grid[iGrid] = (float2 *)fftComputes[iGrid]->getDataDst() + i0;
#endif
    } else {
      data_grid[iGrid] = NULL;
    }
  }
  return data_grid;
}

void CudaPmePencilY::backwardDone() {
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeTransposes[iGrid] != NULL) {
      // Transpose locally
      pmeTransposes[iGrid]->transposeXYZtoZXY((float2 *)fftComputes[iGrid]->getDataSrc());

      // Send data to x-pencils that share the same x-coordinate. There are pmeGrid.yBlocks of them
      // Direct-Device-To-Device communication
      if (numDeviceBuffersX > 0) {
        for (int y=0;y < pmeGrid.yBlocks;y++) {
          if (deviceBuffersX[y].dataGrid[iGrid] != NULL) {
            if (deviceBuffersX[y].deviceID != deviceID) {
              ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataToPeerDeviceZXY(y, deviceBuffersX[y].deviceID,
                Perm_cX_Y_Z, deviceBuffersX[y].dataGrid[iGrid]);
            }
          }
        }
        // Record event for this pencil
        cudaCheck(cudaEventRecord(event, stream));
        // Send empty message
        for (int y=0;y < pmeGrid.yBlocks;y++) {
          if (deviceBuffersX[y].dataGrid[iGrid] != NULL) {
            PmeBlockMsg* msg = new (0, PRIORITY_SIZE) PmeBlockMsg();
            msg->dataSize = 0;
            msg->x = thisIndex.x;
            msg->y = y;
            msg->z = thisIndex.z;
            msg->grid = iGrid;
            msg->simulationStep = simulationStep;
            pmePencilX(0,y,thisIndex.z).recvBlock(msg);
          }
        }
      }

      // Copy via host
      for (int y=0;y < pmeGrid.yBlocks;y++) {
        if (deviceBuffersX[y].dataGrid[iGrid] == NULL) {
          PmeBlockMsg* msg = new (blockSizes[y], PRIORITY_SIZE) PmeBlockMsg();
          msg->dataSize = blockSizes[y];
          msg->x = thisIndex.x;
          msg->y = y;
          msg->z = thisIndex.z;
          msg->grid = iGrid;
          msg->simulationStep = simulationStep;
          ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataDeviceToHost(y, msg->data, msg->dataSize);
          ((CudaPmeTranspose *)pmeTransposes[iGrid])->waitStreamSynchronize();
          pmePencilX(0,y,thisIndex.z).recvBlock(msg);
        }
      }
    }
  }
}

void CudaPmePencilY::forwardDone() {
  if (pmeTransposes[0] == NULL)
    NAMD_bug("CudaPmePencilY::forwardDone, pmeTranspose not initialized");
  if (blockSizes.size() == 0)
    NAMD_bug("CudaPmePencilY::forwardDone, blockSizes not initialized");

  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeTransposes[iGrid] != NULL) {
      // Transpose locally
      pmeTransposes[iGrid]->transposeXYZtoYZX((float2 *)fftComputes[iGrid]->getDataDst());

      // Send data to z-pencils that share the same x-coordinate. There are pmeGrid.yBlocks of them
      // Direct-Device-To-Device communication
      if (numDeviceBuffersZ > 0) {
        for (int y=0;y < pmeGrid.yBlocks;y++) {
          if (deviceBuffersZ[y].dataGrid[iGrid] != NULL) {
            if (deviceBuffersZ[y].deviceID != deviceID) {
              ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataToPeerDeviceYZX(y, deviceBuffersZ[y].deviceID,
                Perm_Z_cX_Y, deviceBuffersZ[y].dataGrid[iGrid]);
            }
          }
        }

        // Record event for this pencil
        cudaCheck(cudaEventRecord(event, stream));
        // Send empty message
        for (int y=0;y < pmeGrid.yBlocks;y++) {
          if (deviceBuffersZ[y].dataGrid[iGrid] != NULL) {
            PmeBlockMsg* msg = new (0, PRIORITY_SIZE) PmeBlockMsg();
            msg->dataSize = 0;
            msg->x = thisIndex.x;
            msg->y = y;
            msg->z = thisIndex.z;
            msg->doEnergy = doEnergy;
            msg->doVirial = doVirial;
            msg->lattice  = lattice;
            msg->numStrayAtoms = numStrayAtoms;
            msg->grid = iGrid;
            msg->simulationStep = simulationStep;
            pmePencilZ(thisIndex.x,y,0).recvBlock(msg);
          }
        }
      }

      // Copy-To-Host communication
      for (int y=0;y < pmeGrid.yBlocks;y++) {
        if (deviceBuffersZ[y].dataGrid[iGrid] == NULL) {
          PmeBlockMsg* msg = new (blockSizes[y], PRIORITY_SIZE) PmeBlockMsg();
          msg->dataSize = blockSizes[y];
          msg->x = thisIndex.x;
          msg->y = y;
          msg->z = thisIndex.z;
          msg->doEnergy = doEnergy;
          msg->doVirial = doVirial;
          msg->lattice  = lattice;
          msg->numStrayAtoms = numStrayAtoms;
          msg->grid = iGrid;
          msg->simulationStep = simulationStep;
          ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataDeviceToHost(y, msg->data, msg->dataSize);
          ((CudaPmeTranspose *)pmeTransposes[iGrid])->waitStreamSynchronize();
          pmePencilZ(thisIndex.x,y,0).recvBlock(msg);
        }
      }
    }
  }
}

void CudaPmePencilY::recvDataFromX(PmeBlockMsg *msg) {
  if (msg->dataSize != 0) {
    // Buffer is coming from a different node
    ((CudaPmeTranspose *)pmeTransposes[msg->grid])->copyDataHostToDevice(msg->y, msg->data, (float2 *)fftComputes[msg->grid]->getDataSrc());
  } else {
    // Buffer is coming from the same node
    // Wait for event that was recorded on the sending pencil
    // device ID = deviceBuffersX[msg->y].deviceID
    // event     = deviceBuffersX[msg->y].event
    cudaCheck(cudaStreamWaitEvent(stream, deviceBuffersX[msg->y].event, 0));
#ifndef P2P_ENABLE_3D
    if (deviceBuffersX[msg->y].dataGrid[msg->grid] != NULL && deviceBuffersX[msg->y].deviceID != deviceID) {
      // Data is in temporary device buffer, copy it into final fft-buffer
      ((CudaPmeTranspose *)pmeTransposes[msg->grid])->copyDataDeviceToDevice(msg->y, (float2 *)fftComputes[msg->grid]->getDataSrc());
    }
#endif
  }
  delete msg;
}

void CudaPmePencilY::recvDataFromZ(PmeBlockMsg *msg) {
  if (msg->dataSize != 0) {
    // Buffer is coming from a different node
    ((CudaPmeTranspose *)pmeTransposes[msg->grid])->copyDataHostToDevice(msg->y, msg->data, (float2 *)fftComputes[msg->grid]->getDataDst());
  } else {
    // Buffer is coming from the same node
    // Wait for event that was recorded on the sending pencil
    // device ID = deviceBuffersZ[msg->y].deviceID
    // event     = deviceBuffersZ[msg->y].event
    cudaCheck(cudaStreamWaitEvent(stream, deviceBuffersZ[msg->y].event, 0));
#ifndef P2P_ENABLE_3D
    if (deviceBuffersZ[msg->y].dataGrid[msg->grid] != NULL && deviceBuffersZ[msg->y].deviceID != deviceID) {
      // Data is in temporary device buffer, copy it into final fft-buffer
      ((CudaPmeTranspose *)pmeTransposes[msg->grid])->copyDataDeviceToDevice(msg->y, (float2 *)fftComputes[msg->grid]->getDataDst());
    }
#endif
  }
  delete msg;
}

//###########################################################################
//###########################################################################
//###########################################################################

void CudaPmePencilZ::initialize(CudaPmeXInitMsg *msg) {
  useXYslab = false;
  pmeGrid = msg->pmeGrid;
  pmePencilY = msg->pmePencilY;
  yMap = msg->yMap;

  delete msg;

  initBlockSizes();
}

void CudaPmePencilZ::initialize(CudaPmeXYInitMsg *msg) {
  useXYslab = true;
  pmeGrid = msg->pmeGrid;
  pmePencilXY = msg->pmePencilXY;
  xyMap = msg->xyMap;

  delete msg;

  initBlockSizes();
}

CudaPmePencilZ::~CudaPmePencilZ() {
  if (eventCreated) cudaCheck(cudaEventDestroy(event));
}

//
// CUDA specific initialization
//
void CudaPmePencilZ::initializeDevice(InitDeviceMsg2 *msg) {
  // Get device proxy
  CProxy_ComputePmeCUDADevice deviceProxy = msg->deviceProxy;
  deviceID = msg->deviceID;
  stream = msg->stream;
  CProxy_ComputePmeCUDAMgr mgrProxy = msg->mgrProxy;
  delete msg;
  // deviceID = deviceProxy.ckLocalBranch()->getDeviceID();
  // cudaStream_t stream = deviceProxy.ckLocalBranch()->getStream();
  // CProxy_ComputePmeCUDAMgr mgrProxy = deviceProxy.ckLocalBranch()->getMgrProxy();
  // Setup fftCompute and pmeKSpaceCompute
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (deviceProxy.ckLocalBranch()->isGridEnabled(iGrid) == true) {
      fftComputes[iGrid] = new CudaFFTCompute(deviceID, stream);
      pmeTransposes[iGrid] = new CudaPmeTranspose(pmeGrid, Perm_Z_cX_Y, thisIndex.x, thisIndex.y, deviceID, stream);
      pmeKSpaceComputes[iGrid] = new CudaPmeKSpaceCompute(pmeGrid, Perm_Z_cX_Y, thisIndex.x, thisIndex.y, ComputeNonbondedUtil::ewaldcof, deviceID, stream, iGrid);
      energyReady[iGrid] = 0;
    } else {
      fftComputes[iGrid] = NULL;
      pmeTransposes[iGrid] = NULL;
      pmeKSpaceComputes[iGrid] = NULL;
      energyReady[iGrid] = -1;
    }
  }

  // Create event. NOTE: Events are tied to devices, hence the cudaSetDevice() here
  cudaCheck(cudaSetDevice(deviceID));
  cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  eventCreated = true;

  deviceBuffers.resize(pmeGrid.zBlocks, DeviceBuffer(-1, false));
  numDeviceBuffers = 0;

  if (useXYslab) {
    for (int z=0;z < pmeGrid.zBlocks;z++) {
      int pe = xyMap.ckLocalBranch()->procNum(0, CkArrayIndex3D(0,0,z));
      if (CkNodeOf(pe) == CkMyNode()) {
        int deviceID0 = mgrProxy.ckLocalBranch()->getDeviceIDPencilX(0, z);
        // Check for Peer-to-Peer access
        int canAccessPeer = 0;
        if (deviceID != deviceID0) {
          cudaCheck(cudaSetDevice(deviceID));
          cudaCheck(cudaDeviceCanAccessPeer(&canAccessPeer, deviceID, deviceID0));
        }
#ifdef DISABLE_P2P
        canAccessPeer = 0;
#endif
        numDeviceBuffers++;
        deviceBuffers[z] = DeviceBuffer(deviceID0, canAccessPeer);
        pmePencilXY(0,0,z).getDeviceBuffer(thisIndex.x, (deviceID0 == deviceID) || canAccessPeer, thisProxy);
      }
    }
  } else {
    for (int z=0;z < pmeGrid.zBlocks;z++) {
      int pe = yMap.ckLocalBranch()->procNum(0, CkArrayIndex3D(thisIndex.x,0,z));
      if (CkNodeOf(pe) == CkMyNode()) {
        int deviceID0 = mgrProxy.ckLocalBranch()->getDeviceIDPencilY(thisIndex.x, z);
        numDeviceBuffers++;
        deviceBuffers[z] = DeviceBuffer(deviceID0, false);
        pmePencilY(thisIndex.x,0,z).getDeviceBuffer(thisIndex.y, (deviceID0 == deviceID), thisProxy);
      }
    }
  }

}

//
// CUDA specific start
//
void CudaPmePencilZ::start(const CkCallback &cb) {
  thisProxy[thisIndex].recvDeviceBuffers(cb);
}

void CudaPmePencilZ::setDeviceBuffers() {
  std::array<std::vector<float2*>, NUM_GRID_MAX> dataPtrsGrid;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    dataPtrsGrid[iGrid] = std::vector<float2*>(pmeGrid.zBlocks, (float2*)0);
    for (int z=0;z < pmeGrid.zBlocks;z++) {
      if (deviceBuffers[z].dataGrid[iGrid] != NULL) {
        if (deviceBuffers[z].deviceID == deviceID || deviceBuffers[z].isPeerDevice) {
          dataPtrsGrid[iGrid][z] = deviceBuffers[z].dataGrid[iGrid];
        }
      }
    }
    if (useXYslab) {
      if (pmeTransposes[iGrid] != NULL) {
        ((CudaPmeTranspose *)pmeTransposes[iGrid])->setDataPtrsYZX(dataPtrsGrid[iGrid], (float2 *)fftComputes[iGrid]->getDataSrc());
      }
    } else {
      if (pmeTransposes[iGrid] != NULL) {
        ((CudaPmeTranspose *)pmeTransposes[iGrid])->setDataPtrsZXY(dataPtrsGrid[iGrid], (float2 *)fftComputes[iGrid]->getDataSrc());
      }
    }
  }
}

std::array<float2*, NUM_GRID_MAX> CudaPmePencilZ::getData(const int i, const bool sameDevice) {
  std::array<float2*, NUM_GRID_MAX> data_grid;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) {
#ifndef P2P_ENABLE_3D
      if (sameDevice) {
        int i0, i1, j0, j1, k0, k1;
        getBlockDim(pmeGrid, Perm_Z_cX_Y, i, 0, 0, i0, i1, j0, j1, k0, k1);
        data_grid[iGrid] = (float2 *)fftComputes[iGrid]->getDataSrc() + i0;
      } else {
        data_grid[iGrid] = ((CudaPmeTranspose *)pmeTransposes[iGrid])->getBuffer(i);
      }
#else
      int i0, i1, j0, j1, k0, k1;
      getBlockDim(pmeGrid, Perm_Z_cX_Y, i, 0, 0, i0, i1, j0, j1, k0, k1);
      data_grid[iGrid] = (float2 *)fftComputes[iGrid]->getDataSrc() + i0;
#endif
    } else {
      data_grid[iGrid] = NULL;
    }
  }
  return data_grid;
}

void CudaPmePencilZ::backwardDone() {
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeTransposes[iGrid] != NULL) {
      // Transpose locally
      if (useXYslab) {
        pmeTransposes[iGrid]->transposeXYZtoYZX((float2 *)fftComputes[iGrid]->getDataSrc());
      } else {
        pmeTransposes[iGrid]->transposeXYZtoZXY((float2 *)fftComputes[iGrid]->getDataSrc());
      }

      if (useXYslab) {
        // Direct-Device-To-Device communication
        if (numDeviceBuffers > 0) {
          for (int z=0;z < pmeGrid.zBlocks;z++) {
            if (deviceBuffers[z].dataGrid[iGrid] != NULL) {
              if (deviceBuffers[z].deviceID != deviceID && !deviceBuffers[z].isPeerDevice) {
                ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataToPeerDeviceYZX(z, deviceBuffers[z].deviceID,
                  Perm_cX_Y_Z, deviceBuffers[z].dataGrid[iGrid]);
              }
            }
          }
          // Record event for this pencil
          cudaCheck(cudaEventRecord(event, stream));
          // Send empty message
          for (int z=0;z < pmeGrid.zBlocks;z++) {
            if (deviceBuffers[z].dataGrid[iGrid] != NULL) {
              PmeBlockMsg* msg = new (0, PRIORITY_SIZE) PmeBlockMsg();
              msg->dataSize = 0;
              msg->x = thisIndex.x;
              msg->y = thisIndex.y;
              msg->z = z;
              msg->grid = iGrid;
              msg->simulationStep = simulationStep;
              pmePencilXY(0,0,z).recvBlock(msg);
            }
          }
        }

        // Copy-To-Host communication
        for (int z=0;z < pmeGrid.zBlocks;z++) {
          if (deviceBuffers[z].dataGrid[iGrid] == NULL) {
            PmeBlockMsg* msg = new (blockSizes[z], PRIORITY_SIZE) PmeBlockMsg();
            msg->dataSize = blockSizes[z];
            msg->x = thisIndex.x;
            msg->y = thisIndex.y;
            msg->z = z;
            msg->grid = iGrid;
            msg->simulationStep = simulationStep;
            ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataDeviceToHost(z, msg->data, msg->dataSize);
            ((CudaPmeTranspose *)pmeTransposes[iGrid])->waitStreamSynchronize();
            pmePencilXY(0,0,z).recvBlock(msg);
          }
        }
      } else {
        // Send data to y-pencils that share the same x-coordinate. There are pmeGrid.zBlocks of them
        // Direct-Device-To-Device communication
        if (numDeviceBuffers > 0) {
          for (int z=0;z < pmeGrid.zBlocks;z++) {
            if (deviceBuffers[z].dataGrid[iGrid] != NULL) {
              if (deviceBuffers[z].deviceID != deviceID) {
                ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataToPeerDeviceZXY(z, deviceBuffers[z].deviceID,
                  Perm_Y_Z_cX, deviceBuffers[z].dataGrid[iGrid]);
              }
            }
          }
          // Record event for this pencil
          cudaCheck(cudaEventRecord(event, stream));
          // Send empty message
          for (int z=0;z < pmeGrid.zBlocks;z++) {
            if (deviceBuffers[z].dataGrid[iGrid] != NULL) {
              PmeBlockMsg* msg = new (0, PRIORITY_SIZE) PmeBlockMsg();
              msg->dataSize = 0;
              msg->x = thisIndex.x;
              msg->y = thisIndex.y;
              msg->z = z;
              msg->grid = iGrid;
              msg->simulationStep = simulationStep;
              pmePencilY(thisIndex.x,0,z).recvBlock(msg);
            }
          }
        }

        // Copy-To-Host communication
        for (int z=0;z < pmeGrid.zBlocks;z++) {
          if (deviceBuffers[z].dataGrid[iGrid] == NULL) {
            PmeBlockMsg* msg = new (blockSizes[z], PRIORITY_SIZE) PmeBlockMsg();
            msg->dataSize = blockSizes[z];
            msg->x = thisIndex.x;
            msg->y = thisIndex.y;
            msg->z = z;
            msg->grid = iGrid;
            msg->simulationStep = simulationStep;
            ((CudaPmeTranspose *)pmeTransposes[iGrid])->copyDataDeviceToHost(z, msg->data, msg->dataSize);
            ((CudaPmeTranspose *)pmeTransposes[iGrid])->waitStreamSynchronize();
            pmePencilY(thisIndex.x,0,z).recvBlock(msg);
          }
        }
      }

      // Submit reductions
      ((CudaPmeKSpaceCompute *)pmeKSpaceComputes[iGrid])->energyAndVirialSetCallback(this);
      // ((CudaPmeKSpaceCompute *)pmeKSpaceCompute)->waitEnergyAndVirial();
      // submitReductions();
    }
  }
}

void CudaPmePencilZ::energyAndVirialDone(unsigned int iGrid) {
  submitReductions(iGrid);
}

void CudaPmePencilZ::recvDataFromY(PmeBlockMsg *msg) {
  // NOTE: No need to synchronize stream here since memory copies are in the stream
  if (msg->dataSize != 0) {
    // Buffer is coming from a different node
    ((CudaPmeTranspose *)pmeTransposes[msg->grid])->copyDataHostToDevice(msg->z, msg->data, (float2 *)fftComputes[msg->grid]->getDataSrc());
  } else {
    // Buffer is coming from the same node
    // Wait for event that was recorded on the sending pencil
    // device ID = deviceBuffers[msg->z].deviceID
    // event     = deviceBuffers[msg->z].event
    cudaCheck(cudaStreamWaitEvent(stream, deviceBuffers[msg->z].event, 0));
#ifndef P2P_ENABLE_3D
    if (deviceBuffers[msg->z].dataGrid[0] != NULL && deviceBuffers[msg->z].deviceID != deviceID && !deviceBuffers[msg->z].isPeerDevice) {
      // Data is in temporary device buffer, copy it into final fft-buffer
      ((CudaPmeTranspose *)pmeTransposes[msg->grid])->copyDataDeviceToDevice(msg->z, (float2 *)fftComputes[msg->grid]->getDataSrc());
    }
#endif
  }
  delete msg;
}
#endif // NAMD_CUDA

#include "CudaPmeSolver.def.h"
