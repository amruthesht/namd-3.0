#include <vector>
#include <numeric>
#include <algorithm>
#include "Node.h"
#include "PatchMap.h"
#include "HomePatch.h"
#include "WorkDistrib.h"
#include "Priorities.h"
#include "PatchData.h"
#include "CudaUtils.h"

#include "SimParameters.h"
#include "CudaPmeSolverUtil.h"

#include "ComputePmeCUDAMgr.h"

#include "CudaPmeSolver.h"
#include "ComputePmeCUDA.h"

#include "DeviceCUDA.h"
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#ifdef WIN32
#define __thread __declspec(thread)
#endif
extern __thread DeviceCUDA *deviceCUDA;

void createStream(cudaStream_t& stream) {
#if CUDA_VERSION >= 5050 || defined(NAMD_HIP)
  int leastPriority, greatestPriority;
  cudaCheck(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  cudaCheck(cudaStreamCreateWithPriority(&stream,cudaStreamDefault,greatestPriority));
  // cudaCheck(cudaStreamCreateWithPriority(&stream,cudaStreamDefault,leastPriority));
#else
  cudaCheck(cudaStreamCreate(&stream));
#endif
}

//
// CUDA implementation of atom storage
//
class CudaPmeAtomStorage : public PmeAtomStorage {
public:
  CudaPmeAtomStorage(const bool useIndex) : PmeAtomStorage(useIndex) {}
  ~CudaPmeAtomStorage() {
    if (atom != NULL) dealloc_((void *)atom);
    if (atomIndex != NULL) dealloc_((void *)atomIndex);
    if (overflowAtom != NULL) dealloc_((void *)overflowAtom);
    if (overflowAtomIndex != NULL) dealloc_((void *)overflowAtomIndex);
    if (alchOn) {
      for (unsigned int i = 0; i < totalFactorArrays; ++i) {
        if (atomElecFactorArrays[i] != NULL) dealloc_((void *)(atomElecFactorArrays[i]));
        if (overflowAtomElecFactorArrays[i] != NULL) dealloc_((void *)(overflowAtomElecFactorArrays[i]));
      }
    }
  }
private:
  
  // Allocate array of size bytes
  void* alloc_(const size_t size) {
    void* p;
    cudaCheck(cudaMallocHost(&p, size));
    return p;
  }

  // Deallocate array
  void dealloc_(void *p) {
    cudaCheck(cudaFreeHost(p));
  }

};

//
// CPU implementation of atom storage
//
class CpuPmeAtomStorage : public PmeAtomStorage {
public:
  CpuPmeAtomStorage(const bool useIndex) : PmeAtomStorage(useIndex) {}
  ~CpuPmeAtomStorage() {
    if (atom != NULL) dealloc_((void *)atom);
    if (atomIndex != NULL) dealloc_((void *)atomIndex);
    if (overflowAtom != NULL) dealloc_((void *)overflowAtom);
    if (overflowAtomIndex != NULL) dealloc_((void *)overflowAtomIndex);
    if (alchOn) {
      for (unsigned int i = 0; i < totalFactorArrays; ++i) {
        if (atomElecFactorArrays[i] != NULL) dealloc_((void *)(atomElecFactorArrays[i]));
        if (overflowAtomElecFactorArrays[i] != NULL) dealloc_((void *)(overflowAtomElecFactorArrays[i]));
      }
    }
  }
private:
  
  // Allocate array of size bytes
  void* alloc_(const size_t size) {
    return (void *)(new char[size]);
  }

  // Deallocate array
  void dealloc_(void *p) {
    delete [] (char *)p;
  }

};

PmeAtomFiler::PmeAtomFiler() {
  for (int p=0;p < 10;p++) {
    pencil[p] = NULL;
    pencilCapacity[p] = 0;
  }  
}
PmeAtomFiler::PmeAtomFiler(CkMigrateMessage *m) {
  for (int p=0;p < 10;p++) {
    pencil[p] = NULL;
    pencilCapacity[p] = 0;
  }
}
PmeAtomFiler::~PmeAtomFiler() {
  for (int p=0;p < 10;p++) {
    if (pencil[p] != NULL) delete [] pencil[p];
  }
}

//
// File atoms into PME pencils. Each atom can belong to maximum 9 pencils
// (oy, oz) = origin of the atoms
// (miny, minz) = grid minimum corner for this patch
// NOTE: This method can only be called locally from the same Pe
//
void PmeAtomFiler::fileAtoms(const int numAtoms, const CudaAtom* atoms, Lattice &lattice, const PmeGrid &pmeGrid,
  const int pencilIndexY, const int pencilIndexZ, const int ylo, const int yhi, const int zlo, const int zhi) {

  // Make sure there's enough room in the pencil arrays
  for (int p=0;p < 10;p++) {
    if (pencil[p] != NULL && pencilCapacity[p] < numAtoms) {
      delete [] pencil[p];
      pencil[p] = NULL;
      pencilCapacity[p] = 0;
    }
    if (pencil[p] == NULL) {
      int newSize = (int)(numAtoms*1.5);
      pencil[p] = new int[newSize];
      pencilCapacity[p] = newSize;
    }
    pencilSize[p] = 0;
  }

  const float recip11 = lattice.a_r().x;
  const float recip22 = lattice.b_r().y;
  const float recip33 = lattice.c_r().z;
  const int order1 = pmeGrid.order - 1;
  const int K1 = pmeGrid.K1;
  const int K2 = pmeGrid.K2;
  const int K3 = pmeGrid.K3;
  const int yBlocks = pmeGrid.yBlocks;
  const int zBlocks = pmeGrid.zBlocks;

  for (int i=0;i < numAtoms;i++) {
    float frx, fry, frz;
    // PmeRealSpaceCompute::calcGridCoord(atoms[i].uix, atoms[i].uiy, atoms[i].uiz,
    //   K1, K2, K3, frx, fry, frz);
    PmeRealSpaceCompute::calcGridCoord(atoms[i].x, atoms[i].y, atoms[i].z, K1, K2, K3, frx, fry, frz);
    // Charge is spread in the region [y0 ... y0+order-1] x [z0 ... z0+order-1]
    int y0 = (int)fry;
    int z0 = (int)frz;
    if (y0 < 0 || y0 >= K2 || z0 < 0 || z0 >= K3) {
      // Add to "Stray pencil" and skip to next atom
      pencil[9][pencilSize[9]++] = i;
      continue;
      // fprintf(stderr, "%lf %lf %lf\n", atoms[i].x, atoms[i].y, atoms[i].z);
      // NAMD_bug("PmeAtomFiler::fileAtoms, charge out of bounds");
    }
    // Calculate pencil index for the four corners of the order X order area
    // The corners determine the pencil indices for this atom.
    int occupied = 0;
    int plist[4];
#pragma unroll
    for (int j=0;j < 4;j++) {

      int py = getPencilIndexY(pmeGrid, (y0 + (j%2)*order1) % K2) - pencilIndexY;
      if (py < ylo) py += yBlocks;
      if (py > yhi) py -= yBlocks;

      int pz = getPencilIndexZ(pmeGrid, (z0 + (j/2)*order1) % K3) - pencilIndexZ;
      if (pz < zlo) pz += zBlocks;
      if (pz > zhi) pz -= zBlocks;

      if (py < ylo || py > yhi || pz < zlo || pz > zhi) {
        // Add to "Stray pencil" and skip to next atom
        pencil[9][pencilSize[9]++] = i;
        goto breakjloop;
        // fprintf(stderr, "py %d [%d ... %d] pz %d [%d ... %d]\n", pz, zlo, zhi);
        // NAMD_bug("PmeAtomFiler::fileAtoms, py,pz outside bounds");
      }
      // p = 0,1,2,3,4,5,6,7,8 (maximum range)
      plist[j] = (py-ylo) + (pz-zlo)*3;
    }

#pragma unroll
    for (int j=0;j < 4;j++) {
      int p = plist[j];
      // pbit = 0, 2, 4, 8, 16, 32, 64, 128, 256
      int pbit = (1 << p);
      if (!(occupied & pbit)) {
        pencil[p][pencilSize[p]++] = i;
        occupied |= pbit;
      }
    }

breakjloop:
    continue;
  }

}

//
// Class constructor
//
ComputePmeCUDAMgr::ComputePmeCUDAMgr() {
  __sdag_init();
  numDevices = 0;
  numTotalPatches = 0;
  numNodesContributed = 0;
  numDevicesMax = 0;
}

//
// Class constructor
//
ComputePmeCUDAMgr::ComputePmeCUDAMgr(CkMigrateMessage *) {
  __sdag_init();
  NAMD_bug("ComputePmeCUDAMgr cannot be migrated");
  numDevices = 0;
  numTotalPatches = 0;
  numNodesContributed = 0;
  numDevicesMax = 0;
}

//
// Class destructor
//
ComputePmeCUDAMgr::~ComputePmeCUDAMgr() {
  for (int i=0;i < extraDevices.size();i++) {
    cudaCheck(cudaSetDevice(extraDevices[i].deviceID));
    cudaCheck(cudaStreamDestroy(extraDevices[i].stream));
  }
}

//
// Returns home pencil (homey, homez)
// Home pencil = pencil with most overlap with this patch
//
void ComputePmeCUDAMgr::getHomePencil(PatchID patchID, int& homey, int& homez) {
  PatchMap *patchMap = PatchMap::Object();

  BigReal miny = patchMap->min_b(patchID);
  BigReal maxy = patchMap->max_b(patchID);

  BigReal minz = patchMap->min_c(patchID);
  BigReal maxz = patchMap->max_c(patchID);

  // Determine home pencil = pencil with most overlap
  
  // Calculate patch grid coordinates
  int patch_y0 = floor((miny+0.5)*pmeGrid.K2);
  int patch_y1 = floor((maxy+0.5)*pmeGrid.K2)-1;
  int patch_z0 = floor((minz+0.5)*pmeGrid.K3);
  int patch_z1 = floor((maxz+0.5)*pmeGrid.K3)-1;

  if (patch_y0 < 0 || patch_y1 >= pmeGrid.K2 || patch_z0 < 0 || patch_z1 >= pmeGrid.K3) {
    NAMD_bug("ComputePmeCUDAMgr::getHomePencil, patch bounds are outside grid bounds");
  }

  int maxOverlap = 0;
  homey = -1;
  homez = -1;
  for (int iz=0;iz < pmeGrid.zBlocks;iz++) {
    for (int iy=0;iy < pmeGrid.yBlocks;iy++) {
      int pencil_x0, pencil_x1, pencil_y0, pencil_y1, pencil_z0, pencil_z1;
      getPencilDim(pmeGrid, Perm_X_Y_Z, iy, iz,
        pencil_x0, pencil_x1, pencil_y0, pencil_y1, pencil_z0, pencil_z1);

      if (pencil_y1 - pencil_y0 < pmeGrid.order || pencil_z1 - pencil_z0 < pmeGrid.order)
        NAMD_bug("ComputePmeCUDAMgr::getHomePencil, pencil size must be >= PMEInterpOrder");

      int y0 = std::max(patch_y0, pencil_y0);
      int y1 = std::min(patch_y1, pencil_y1);
      int z0 = std::max(patch_z0, pencil_z0);
      int z1 = std::min(patch_z1, pencil_z1);

      int overlap = (y1-y0 > 0 && z1-z0 > 0) ? (y1-y0)*(z1-z0) : -1;

      if (overlap > maxOverlap) {
        maxOverlap = overlap;
        homey = iy;
        homez = iz;
      }
    }
  }

  if (homey == -1 || homez == -1)
    NAMD_bug("ComputePmeCUDAMgr::getHomePencil, home pencil not found");
}

//
// Calculates maximum number of PME pencils
//
void ComputePmeCUDAMgr::restrictToMaxPMEPencils() {
  PatchMap *patchMap = PatchMap::Object();
  SimParameters *simParams = Node::Object()->simParameters;
  // need to initialize with Lattice from SimParameters
  Lattice lattice = simParams->lattice;
  BigReal sysdimb = lattice.b_r().unit() * lattice.b();
  BigReal sysdimc = lattice.c_r().unit() * lattice.c();
  BigReal cutoff = simParams->cutoff;
  BigReal patchdim = simParams->patchDimension;
  BigReal marginb = 0.5 * ( patchdim - cutoff ) / sysdimb;
  BigReal marginc = 0.5 * ( patchdim - cutoff ) / sysdimc;
  int numPatches = patchMap->numPatches();

  pmeGrid.xBlocks = std::min(pmeGrid.xBlocks, pmeGrid.K1);

  int pid = 0;
  while (pid < numPatches) {
    // Get home pencil
    int homey, homez;
    getHomePencil(pid, homey, homez);
    // Y
    {
      BigReal miny = patchMap->min_b(pid);
      BigReal maxy = patchMap->max_b(pid);
      // min2 (max2) is smallest (largest) grid line for this patch
      int min2 = ((int) floor(pmeGrid.K2 * (miny+0.5 - marginb)));
      int max2 = ((int) floor(pmeGrid.K2 * (maxy+0.5 + marginb))) + (pmeGrid.order - 1);
      // Restrict grid lines to [0 ... pmeGrid.K2-1]
      if (min2 < 0) min2 += pmeGrid.K2;
      if (max2 >= pmeGrid.K2) max2 -= pmeGrid.K2;
      // Get pencil indices for the grid lines
      int min2pi = getPencilIndexY(pmeGrid, min2);
      int max2pi = getPencilIndexY(pmeGrid, max2);
      // Distance from home pencil
      int dmin2pi = homey - min2pi;
      if (dmin2pi < 0) dmin2pi += pmeGrid.yBlocks;
      if (dmin2pi < 0)
        NAMD_bug("ComputePmeCUDAMgr::restrictToMaxPMEPencils, Error in dmin2pi");
      // If distance is > 1, must decrease the number of y-pencils and try again
      if (dmin2pi > 1) {
        pmeGrid.yBlocks--;
        if (pmeGrid.yBlocks <= 0) break;
        continue;
      }
      int dmax2pi = max2pi - homey;
      if (dmax2pi < 0) dmax2pi += pmeGrid.yBlocks;
      if (dmax2pi < 0)
        NAMD_bug("ComputePmeCUDAMgr::restrictToMaxPMEPencils, Error in dmax2pi");
      // If distance is > 1, must decrease the number of y-pencils and try again
      if (dmax2pi > 1) {
        pmeGrid.yBlocks--;
        if (pmeGrid.yBlocks <= 0) break;
        continue;
      }
    }

    // Z
    {
      BigReal minz = patchMap->min_c(pid);
      BigReal maxz = patchMap->max_c(pid);
      // min3 (max3) is smallest (largest) grid line for this patch
      int min3 = ((int) floor(pmeGrid.K3 * (minz+0.5 - marginc)));
      int max3 = ((int) floor(pmeGrid.K3 * (maxz+0.5 + marginc))) + (pmeGrid.order - 1);
      // Restrict grid lines to [0 ... pmeGrid.K3-1]
      if (min3 < 0) min3 += pmeGrid.K3;
      if (max3 >= pmeGrid.K3) max3 -= pmeGrid.K3;
      // Get pencil indices for the grid lines
      int min3pi = getPencilIndexZ(pmeGrid, min3);
      int max3pi = getPencilIndexZ(pmeGrid, max3);
      // Distance from home pencil
      int dmin3pi = homez - min3pi;
      if (dmin3pi < 0) dmin3pi += pmeGrid.zBlocks;
      if (dmin3pi < 0)
        NAMD_bug("ComputePmeCUDAMgr::restrictToMaxPMEPencils, Error in dmin3pi");
      // If distance is > 1, must decrease the number of z-pencils and try again
      if (dmin3pi > 1) {
        pmeGrid.zBlocks--;
        if (pmeGrid.zBlocks <= 0) break;
        continue;
      }
      int dmax3pi = max3pi - homez;
      if (dmax3pi < 0) dmax3pi += pmeGrid.zBlocks;
      if (dmax3pi < 0)
        NAMD_bug("ComputePmeCUDAMgr::restrictToMaxPMEPencils, Error in dmax3pi");
      // If distance is > 1, must decrease the number of z-pencils and try again
      if (dmax3pi > 1) {
        pmeGrid.zBlocks--;
        if (pmeGrid.zBlocks <= 0) break;
        continue;
      }
    }

    pid++;
  }

  // if (CkMyNode() == 0)
  //   fprintf(stderr, "pmeGrid.yBlocks %d pmeGrid.zBlocks %d\n", pmeGrid.yBlocks, pmeGrid.zBlocks);

  if (pmeGrid.xBlocks <= 0 || pmeGrid.yBlocks <= 0 || pmeGrid.zBlocks <= 0)
    NAMD_bug("ComputePmeCUDAMgr::restrictToMaxPMEPencils, zero PME pencils found");

  if (pmeGrid.xBlocks > pmeGrid.K1 || pmeGrid.yBlocks > pmeGrid.K2|| pmeGrid.zBlocks > pmeGrid.K3)
    NAMD_bug("ComputePmeCUDAMgr::restrictToMaxPMEPencils, unable to restrict number of PME pencils");
}

//
// Sets up pencils. May be called multiple times. 
//
void ComputePmeCUDAMgr::setupPencils() {
  SimParameters *simParams = Node::Object()->simParameters;

  pmeGrid.K1 = simParams->PMEGridSizeX;
  pmeGrid.K2 = simParams->PMEGridSizeY;
  pmeGrid.K3 = simParams->PMEGridSizeZ;
  pmeGrid.order = simParams->PMEInterpOrder;
  pmeGrid.dim2 = pmeGrid.K2;
  pmeGrid.dim3 = 2 * (pmeGrid.K3/2 + 1);

  // Count the total number of devices assuming all nodes have the same number as this node
  // NOTE: This should be changed in the future to support heterogeneous nodes!!!
  int numDevicesTmp = deviceCUDA->getNumDevice();

  int numDeviceTot = CkNumNodes() * numDevicesTmp;
  // Use approximately 1/4th of the devices for PME
  int numDeviceToUse = std::max(1, numDeviceTot/4);

  if (numDeviceToUse < 4) {
    // 2D Slab
    pmeGrid.yBlocks = 1;
    pmeGrid.xBlocks = pmeGrid.zBlocks = numDeviceToUse;
  } else {
    // 1D Pencil
    pmeGrid.yBlocks = (int)sqrt((double)numDeviceToUse);
    pmeGrid.zBlocks = numDeviceToUse/pmeGrid.yBlocks;
    pmeGrid.xBlocks = pmeGrid.zBlocks;
  }

  if ( simParams->PMEPencilsX > 0 ) pmeGrid.xBlocks = simParams->PMEPencilsX;
  if ( simParams->PMEPencilsY > 0 ) pmeGrid.yBlocks = simParams->PMEPencilsY;
  if ( simParams->PMEPencilsZ > 0 ) pmeGrid.zBlocks = simParams->PMEPencilsZ;

  // Restrict number of pencils to the maximum number
  restrictToMaxPMEPencils();

  // Fix pencil numbers if they don't make sense w.r.t. number of devices
  if (pmeGrid.yBlocks == 1) {
    // 2D Slab
    if (pmeGrid.xBlocks > numDeviceTot) pmeGrid.xBlocks = numDeviceTot;
    if (pmeGrid.zBlocks > numDeviceTot) pmeGrid.zBlocks = numDeviceTot;
  } else {
    // 1D Pencil
    if (pmeGrid.yBlocks*pmeGrid.zBlocks > numDeviceTot ||
      pmeGrid.xBlocks*pmeGrid.zBlocks > numDeviceTot ||
      pmeGrid.xBlocks*pmeGrid.yBlocks > numDeviceTot) {
      pmeGrid.yBlocks = std::min(pmeGrid.yBlocks, (int)sqrt((double)numDeviceTot));
      pmeGrid.zBlocks = std::min(pmeGrid.zBlocks, numDeviceTot/pmeGrid.yBlocks);
    }
    pmeGrid.xBlocks = std::min(pmeGrid.yBlocks, pmeGrid.zBlocks);
  }

  // Here (block1, block2, block3) define the size of charge grid pencil in each direction
  pmeGrid.block1 = ( pmeGrid.K1 + pmeGrid.xBlocks - 1 ) / pmeGrid.xBlocks;
  pmeGrid.block2 = ( pmeGrid.K2 + pmeGrid.yBlocks - 1 ) / pmeGrid.yBlocks;
  pmeGrid.block3 = ( pmeGrid.K3 + pmeGrid.zBlocks - 1 ) / pmeGrid.zBlocks;

  // Determine type of FFT
  if (pmeGrid.xBlocks == 1 && pmeGrid.yBlocks == 1 && pmeGrid.zBlocks == 1) {
    // Single block => 3D FFT
    pmePencilType = 3;
    iout << iINFO << "Use 3D box decompostion in PME FFT.\n" << endi;
  } else if (pmeGrid.yBlocks == 1) {
    // Blocks in all but y-dimension => 2D FFT
    pmePencilType = 2;
    iout << iINFO << "Use 2D slab decompostion in PME FFT.\n" << endi;
  } else {
    // Blocks in all dimensions => 1D FFT
    pmePencilType = 1;
    iout << iINFO << "Use 1D pencil decompostion in PME FFT.\n" << endi;
  }

  //--------------------------------------------------------------------------
  // Map pencils into Pes
  xPes.resize(pmeGrid.yBlocks*pmeGrid.zBlocks);

  if (pmePencilType == 1 || pmePencilType == 2) {
    zPes.resize(pmeGrid.xBlocks*pmeGrid.yBlocks);
  }
  if (pmePencilType == 1) {
    yPes.resize(pmeGrid.xBlocks*pmeGrid.zBlocks);
  }  

    // i % numDeviceTot                              = device index
    // (i % numDeviceTot)/deviceCUDA->getNumDevice() = node index
    // (i % CkNodeSize(node))                        = pe displacement
  for (int i=0;i < xPes.size();i++) {
    int node = (i % numDeviceTot)/numDevicesTmp;
    xPes[i] = CkNodeFirst(node) + (i + 0) % CkNodeSize(node);
  }
  for (int i=0;i < yPes.size();i++) {
    int node = (i % numDeviceTot)/numDevicesTmp;
    yPes[i] = CkNodeFirst(node) + (i + 0) % CkNodeSize(node);
  }
  for (int i=0;i < zPes.size();i++) {
    int node = (i % numDeviceTot)/numDevicesTmp;
    zPes[i] = CkNodeFirst(node) + (i + 0) % CkNodeSize(node);
  }

  // char peStr[256];
  // char *p = peStr;
  // p += sprintf(p, "%2d | xPes", CkMyPe());
  // for (int i=0;i < xPes.size();i++)
  //   p += sprintf(p, " %d", xPes[i]);
  // p += sprintf(p, " yPes");
  // for (int i=0;i < yPes.size();i++)
  //   p += sprintf(p, " %d", yPes[i]);
  // p += sprintf(p, " zPes");
  // for (int i=0;i < zPes.size();i++)
  //   p += sprintf(p, " %d", zPes[i]);
  // fprintf(stderr, "%s | %d %d\n",peStr, CkNodeFirst(CkMyNode()), CkNodeSize(CkMyNode()));

  //--------------------------------------------------------------------------
  // Build global node list for x-pencils
  nodeDeviceList.resize(xPes.size());
  numDevices = 0;
  for (int k=0;k < xPes.size();k++) {
    nodeDeviceList[k].node = CkNodeOf(xPes[k]);
    nodeDeviceList[k].device = -1;
    if (nodeDeviceList[k].node == CkMyNode()) {
      nodeDeviceList[k].device = numDevices++;
    }
  }

  ijPencilX.clear();
  ijPencilY.clear();
  ijPencilZ.clear();

  // Construct list of pencil coordinates (i,j) for each device held by this node
  for (int k=0;k < xPes.size();k++) {
    if (CkMyNode() == CkNodeOf(xPes[k])) {
      IJ ij;
      ij.i = k % pmeGrid.yBlocks;
      ij.j = k / pmeGrid.yBlocks;
      ijPencilX.push_back(ij);
    }
  }
  if (ijPencilX.size() != numDevices)
    NAMD_bug("ComputePmeCUDAMgr::setupPencils, error setting up x-pencils and devices");

  int numDevicesY = 0;
  for (int k=0;k < yPes.size();k++) {
    if (CkMyNode() == CkNodeOf(yPes[k])) {
      IJ ij;
      ij.i = k % pmeGrid.xBlocks;
      ij.j = k / pmeGrid.xBlocks;
      ijPencilY.push_back(ij);
      numDevicesY++;
    }
  }

  int numDevicesZ = 0;
  for (int k=0;k < zPes.size();k++) {
    if (CkMyNode() == CkNodeOf(zPes[k])) {
      IJ ij;
      ij.i = k % pmeGrid.xBlocks;
      ij.j = k / pmeGrid.xBlocks;
      ijPencilZ.push_back(ij);
      numDevicesZ++;
    }
  }
}

//
// Returns true if PE "pe" is used in PME
//
bool ComputePmeCUDAMgr::isPmePe(int pe) {
  for (int i=0;i < xPes.size();i++) {
    if (pe == xPes[i]) return true;
  }
  return false;
}

//
// Returns true if node "node" is used for PME
//
bool ComputePmeCUDAMgr::isPmeNode(int node) {
  for (int i=0;i < nodeDeviceList.size();i++) {
    if (nodeDeviceList[i].node == node) {
      return true;
    }
  }
  return false;
}

//
// Returns true if device "deviceID" is used for PME
//
bool ComputePmeCUDAMgr::isPmeDevice(int deviceID) {
  for (int i=0;i < nodeDeviceList.size();i++) {
    if (deviceCUDA->getDeviceIDbyRank(nodeDeviceList[i].device % deviceCUDA->getNumDevice()) == deviceID) {
      return true;
    }
  }
  return false;
}

//
// Initialize compute manager
// This gets called on one Pe on each node from Node::startup()
//
void ComputePmeCUDAMgr::initialize(CkQdMsg *msg) {
	if (msg != NULL) delete msg;

  setupPencils();

  if ( ! CkMyNode() ) {
    iout << iINFO << "PME using " << pmeGrid.xBlocks << " x " <<
      pmeGrid.yBlocks << " x " << pmeGrid.zBlocks <<
      " pencil grid for FFT and reciprocal sum.\n" << endi;
  }

  // Initialize list that contains the number of home patches for each device on this manager
  numHomePatchesList.resize(numDevices, 0);

  //--------------------------------------------------------------------------
  // Create devices and atom filer
  // numDevices = number of devices we'll be using, possibly different on each node
  // Use root node to compute the maximum number of devices in use over all nodes
  thisProxy[0].initializeDevicesAndAtomFiler(new NumDevicesMsg(numDevices));
  //--------------------------------------------------------------------------

  if (CkMyNode() == 0) {

    if (pmePencilType == 3) {
  		// Single block => 3D FFT
      CProxy_PmePencilXYZMap xyzMap = CProxy_PmePencilXYZMap::ckNew(xPes[0]);
      CkArrayOptions xyzOpts(1);
      xyzOpts.setMap(xyzMap);
      xyzOpts.setAnytimeMigration(false);
      xyzOpts.setStaticInsertion(true);
      pmePencilXYZ = CProxy_CudaPmePencilXYZ::ckNew(xyzOpts);
      pmePencilXYZ[0].initialize(new CudaPmeXYZInitMsg(pmeGrid));
      thisProxy.recvPencils(pmePencilXYZ);
    } else if (pmePencilType == 2) {
      // Blocks in all but y-dimension => 2D FFT
      CProxy_PmePencilXYMap xyMap = CProxy_PmePencilXYMap::ckNew(xPes);
      CProxy_PmePencilXMap zMap = CProxy_PmePencilXMap::ckNew(0, 1, pmeGrid.xBlocks, zPes);
      CkArrayOptions xyOpts(1, 1, pmeGrid.zBlocks);
      CkArrayOptions zOpts(pmeGrid.xBlocks, 1, 1);
      xyOpts.setMap(xyMap);
      zOpts.setMap(zMap);
      xyOpts.setAnytimeMigration(false);
      zOpts.setAnytimeMigration(false);
      xyOpts.setStaticInsertion(true);
      zOpts.setStaticInsertion(true);
      pmePencilXY = CProxy_CudaPmePencilXY::ckNew(xyOpts);
      pmePencilZ = CProxy_CudaPmePencilZ::ckNew(zOpts);
      // Send pencil proxies to other nodes
      thisProxy.recvPencils(pmePencilXY, pmePencilZ);
      pmePencilXY.initialize(new CudaPmeXYInitMsg(pmeGrid, pmePencilXY, pmePencilZ, xyMap, zMap));
      pmePencilZ.initialize(new CudaPmeXYInitMsg(pmeGrid, pmePencilXY, pmePencilZ, xyMap, zMap));
    } else {
  		// Blocks in all dimensions => 1D FFT
      CProxy_PmePencilXMap xMap = CProxy_PmePencilXMap::ckNew(1, 2, pmeGrid.yBlocks, xPes);
      CProxy_PmePencilXMap yMap = CProxy_PmePencilXMap::ckNew(0, 2, pmeGrid.xBlocks, yPes);
      CProxy_PmePencilXMap zMap = CProxy_PmePencilXMap::ckNew(0, 1, pmeGrid.xBlocks, zPes);
      CkArrayOptions xOpts(1, pmeGrid.yBlocks, pmeGrid.zBlocks);
      CkArrayOptions yOpts(pmeGrid.xBlocks, 1, pmeGrid.zBlocks);
  		CkArrayOptions zOpts(pmeGrid.xBlocks, pmeGrid.yBlocks, 1);
      xOpts.setMap(xMap);
      yOpts.setMap(yMap);
      zOpts.setMap(zMap);
      xOpts.setAnytimeMigration(false);
      yOpts.setAnytimeMigration(false);
      zOpts.setAnytimeMigration(false);
      xOpts.setStaticInsertion(true);
      yOpts.setStaticInsertion(true);
      zOpts.setStaticInsertion(true);
      pmePencilX = CProxy_CudaPmePencilX::ckNew(xOpts);
      pmePencilY = CProxy_CudaPmePencilY::ckNew(yOpts);
      pmePencilZ = CProxy_CudaPmePencilZ::ckNew(zOpts);
      // Send pencil proxies to other nodes
      thisProxy.recvPencils(pmePencilX, pmePencilY, pmePencilZ);
      pmePencilX.initialize(new CudaPmeXInitMsg(pmeGrid, pmePencilX, pmePencilY, pmePencilZ, xMap, yMap, zMap));
      pmePencilY.initialize(new CudaPmeXInitMsg(pmeGrid, pmePencilX, pmePencilY, pmePencilZ, xMap, yMap, zMap));
      pmePencilZ.initialize(new CudaPmeXInitMsg(pmeGrid, pmePencilX, pmePencilY, pmePencilZ, xMap, yMap, zMap));
    }
  }

}

void ComputePmeCUDAMgr::createDevicesAndAtomFiler() {
  if (CkMyNode() != 0)
    NAMD_bug("ComputePmeCUDAMgr::createDevicesAndAtomFiler can only be called on root node");

  // Root node creates all device proxies
  // NOTE: Only root node has numDevicesMax
  RecvDeviceMsg* msg = new (numDevicesMax, PRIORITY_SIZE) RecvDeviceMsg();
  msg->numDevicesMax = numDevicesMax;
  for (int i=0;i < numDevicesMax;i++) {
    CProxy_ComputePmeCUDADevice dev = CProxy_ComputePmeCUDADevice::ckNew();
    memcpy(&msg->dev[i], &dev, sizeof(CProxy_ComputePmeCUDADevice));
  }
  thisProxy.recvDevices(msg);

  CProxy_PmeAtomFiler filer = CProxy_PmeAtomFiler::ckNew();
  thisProxy.recvAtomFiler(filer);

}

void ComputePmeCUDAMgr::recvAtomFiler(CProxy_PmeAtomFiler filer) {
  pmeAtomFiler = filer;
}

void ComputePmeCUDAMgr::recvDevices(RecvDeviceMsg* msg) {
  numDevicesMax = msg->numDevicesMax;
  if (numDevices > numDevicesMax)
    NAMD_bug("ComputePmeCUDAMgr::recvDevices, numDevices > numDevicesMax");
  deviceProxy.resize(numDevices);
  for (int i=0;i < numDevices;i++) {
    deviceProxy[i] = msg->dev[i];
  }
  delete msg;
}

void ComputePmeCUDAMgr::recvPencils(CProxy_CudaPmePencilXYZ xyz) {
  pmePencilXYZ = xyz;
}

void ComputePmeCUDAMgr::recvPencils(CProxy_CudaPmePencilXY xy, CProxy_CudaPmePencilZ z) {
  pmePencilXY = xy;
  pmePencilZ = z;
}

void ComputePmeCUDAMgr::recvPencils(CProxy_CudaPmePencilX x, CProxy_CudaPmePencilY y, CProxy_CudaPmePencilZ z) {
  pmePencilX = x;
  pmePencilY = y;
  pmePencilZ = z;
}

//
// Initialize pencils on this node
// This gets called on one rank on each node
//
void ComputePmeCUDAMgr::initialize_pencils(CkQdMsg *msg) {
  if (msg != NULL) delete msg;

  int numDevicesTmp = deviceCUDA->getNumDevice();

  // Initialize device proxies for real-space interfacing
  for (int i=0;i < ijPencilX.size();i++) {
    // NOTE: i is here the device ID
    int deviceID = deviceCUDA->getDeviceIDbyRank(i % numDevicesTmp);
    deviceProxy[i].ckLocalBranch()->initialize(pmeGrid, ijPencilX[i].i, ijPencilX[i].j,
      deviceID, pmePencilType, thisProxy, pmeAtomFiler);
    if (pmePencilType == 1) {
      deviceProxy[i].ckLocalBranch()->setPencilProxy(pmePencilX);
    } else if (pmePencilType == 2) {
      deviceProxy[i].ckLocalBranch()->setPencilProxy(pmePencilXY);
    } else {
      deviceProxy[i].ckLocalBranch()->setPencilProxy(pmePencilXYZ);
    }
  }

  // Use above initialized device proxies for the PME pencils that interface with real-space
  for (int i=0;i < ijPencilX.size();i++) {
    if (pmePencilType == 1) {
      pmePencilX(0, ijPencilX[i].i, ijPencilX[i].j).initializeDevice(new InitDeviceMsg(deviceProxy[i]));
    } else if (pmePencilType == 2) {
      pmePencilXY(0, 0, ijPencilX[i].j).initializeDevice(new InitDeviceMsg(deviceProxy[i]));
    } else {
      pmePencilXYZ[0].initializeDevice(new InitDeviceMsg(deviceProxy[i]));
    }
  }

  // Create extra devices for Y and Z pencils if necessary
  int n = std::max(ijPencilY.size(), ijPencilZ.size());
  if (n > ijPencilX.size()) {
    int nextra = n - ijPencilX.size();
    extraDevices.resize(nextra);
    for (int i=0;i < nextra;i++) {
      extraDevices[i].deviceID = deviceCUDA->getDeviceIDbyRank((i + ijPencilX.size()) % numDevicesTmp);
      cudaCheck(cudaSetDevice(extraDevices[i].deviceID));
      createStream(extraDevices[i].stream);
    }
  }

  // Initialize Y pencils
  for (int i=0;i < ijPencilY.size();i++) {
    int deviceID;
    cudaStream_t stream;
    if (i < ijPencilX.size()) {
      deviceID = deviceProxy[i].ckLocalBranch()->getDeviceID();
      stream   = deviceProxy[i].ckLocalBranch()->getStream();
    } else {
      deviceID = extraDevices[i-ijPencilX.size()].deviceID;
      stream   = extraDevices[i-ijPencilX.size()].stream;
    }
    pmePencilY(ijPencilY[i].i, 0, ijPencilY[i].j).initializeDevice(new InitDeviceMsg2(deviceID, stream, thisProxy, deviceProxy[i]));
  }

  // Initialize Z pencils
  for (int i=0;i < ijPencilZ.size();i++) {
    int deviceID;
    cudaStream_t stream;
    if (i < ijPencilX.size()) {
      deviceID = deviceProxy[i].ckLocalBranch()->getDeviceID();
      stream   = deviceProxy[i].ckLocalBranch()->getStream();
    } else {
      deviceID = extraDevices[i-ijPencilX.size()].deviceID;
      stream   = extraDevices[i-ijPencilX.size()].stream;
    }
    pmePencilZ(ijPencilZ[i].i, ijPencilZ[i].j, 0).initializeDevice(new InitDeviceMsg2(deviceID, stream, thisProxy, deviceProxy[i]));
  }

}

//
// Activate (start) pencils
// This gets called on rank 0 Pe on each node
//
void ComputePmeCUDAMgr::activate_pencils(CkQdMsg *msg) {
  if (msg != NULL) delete msg;

  for (int device=0;device < numDevices;device++) {
    deviceProxy[device].ckLocalBranch()->activate_pencils();
  }

  for (int i=0;i < ijPencilY.size();i++) {
    PmeStartMsg* pmeStartYMsg = new PmeStartMsg();
    for (unsigned iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
      pmeStartYMsg->dataGrid[iGrid] = NULL;
      pmeStartYMsg->dataSizes[iGrid] = 0;
      pmeStartYMsg->enabledGrid[iGrid] = deviceProxy[0].ckLocalBranch()->isGridEnabled(iGrid);
    }
    pmePencilY(ijPencilY[i].i, 0, ijPencilY[i].j).start(pmeStartYMsg);
  }

  for (int i=0;i < ijPencilZ.size();i++) {
    PmeStartMsg* pmeStartZMsg = new PmeStartMsg();
    for (unsigned iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
      pmeStartZMsg->dataGrid[iGrid] = NULL;
      pmeStartZMsg->dataSizes[iGrid] = 0;
      pmeStartZMsg->enabledGrid[iGrid] = deviceProxy[0].ckLocalBranch()->isGridEnabled(iGrid);
    }
    pmePencilZ(ijPencilZ[i].i, ijPencilZ[i].j, 0).start(pmeStartZMsg);
  }

}

//
// Returns node that contains x-pencil i,j
//
int ComputePmeCUDAMgr::getNode(int i, int j) {
  if (i < 0 || i >= pmeGrid.yBlocks || j < 0 || j >= pmeGrid.zBlocks)
    NAMD_bug("ComputePmeCUDAMgr::getNode, pencil index out of bounds");
  int ind = i + j*pmeGrid.yBlocks;
  return nodeDeviceList[ind].node;
}

//
// Returns home node for a patch
//
int ComputePmeCUDAMgr::getHomeNode(PatchID patchID) {
  int homey, homez;
  getHomePencil(patchID, homey, homez);
  return getNode(homey, homez);
}

//
// Returns device index on this node that contains x-pencil i,j
//
int ComputePmeCUDAMgr::getDevice(int i, int j) {
  if (i < 0 || i >= pmeGrid.yBlocks || j < 0 || j >= pmeGrid.zBlocks)
    NAMD_bug("ComputePmeCUDAMgr::getDevice, pencil index out of bounds");
  int ind = i + j*pmeGrid.yBlocks;
  int device = nodeDeviceList[ind].device;
  if (device == -1)
    NAMD_bug("ComputePmeCUDAMgr::getDevice, no device found");
  return device;
}

//
// Returns device index on this node that contains y-pencil i,j
//
int ComputePmeCUDAMgr::getDevicePencilY(int i, int j) {
  if (i < 0 || i >= pmeGrid.xBlocks || j < 0 || j >= pmeGrid.zBlocks)
    NAMD_bug("ComputePmeCUDAMgr::getDevicePencilY, pencil index out of bounds");
  for (int device=0;device < ijPencilY.size();device++) {
    if (ijPencilY[device].i == i && ijPencilY[device].j == j) return device;
  }
  char str[256];
  sprintf(str, "ComputePmeCUDAMgr::getDevicePencilY, no device found at i %d j %d",i,j);
  NAMD_bug(str);
  return -1;
}

//
// Returns device index on this node that contains z-pencil i,j
//
int ComputePmeCUDAMgr::getDevicePencilZ(int i, int j) {
  if (i < 0 || i >= pmeGrid.xBlocks || j < 0 || j >= pmeGrid.yBlocks)
    NAMD_bug("ComputePmeCUDAMgr::getDevicePencilZ, pencil index out of bounds");
  for (int device=0;device < ijPencilZ.size();device++) {
    if (ijPencilZ[device].i == i && ijPencilZ[device].j == j) return device;
  }
  NAMD_bug("ComputePmeCUDAMgr::getDevicePencilZ, no device found");
  return -1;
}

//
// Returns device ID on this node that contains x-pencil i,j
//
int ComputePmeCUDAMgr::getDeviceIDPencilX(int i, int j) {
  int device = getDevice(i, j);
  return deviceProxy[device].ckLocalBranch()->getDeviceID();
}

//
// Returns device ID on this node that contains y-pencil i,j
//
int ComputePmeCUDAMgr::getDeviceIDPencilY(int i, int j) {
  int device = getDevicePencilY(i, j);
  return deviceProxy[device].ckLocalBranch()->getDeviceID();
}

//
// Returns device ID on this node that contains z-pencil i,j
//
int ComputePmeCUDAMgr::getDeviceIDPencilZ(int i, int j) {
  int device = getDevicePencilZ(i, j);
  return deviceProxy[device].ckLocalBranch()->getDeviceID();
}

//
// Skip this round of PME, call skip on all Z-pencils (this is needed to get the reductions submitted)
//
void ComputePmeCUDAMgr::skip() {
  switch(pmePencilType) {
    case 1:
    pmePencilZ.skip();
    break;
    case 2:
    pmePencilZ.skip();
    break;
    case 3:
    pmePencilXYZ[0].skip();
    break;
  }
}

void ComputePmeCUDAMgr::recvAtoms(PmeAtomMsg *msg) {
  int device = getDevice(msg->i, msg->j);
  deviceProxy[device].ckLocalBranch()->recvAtoms(msg);
}

ComputePmeCUDADevice::ComputePmeCUDADevice() {
  // __sdag_init();
  numHomePatches = 0;
//   forceCapacity = 0;
//   force = NULL;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    pmeRealSpaceComputes[iGrid] = NULL;
    enabledGrid[iGrid] = false;
    forces[iGrid] = NULL;
    forceCapacities[iGrid] = 0;
  }
//   pmeRealSpaceCompute = NULL;
  streamCreated = false;
  lock_numHomePatchesMerged = CmiCreateLock();
  lock_numPencils = CmiCreateLock();
  lock_numNeighborsRecv = CmiCreateLock();
  lock_recvAtoms = CmiCreateLock();
  numNeighborsExpected = 0;
  numStrayAtoms = 0;
  // Reset counters
  numNeighborsRecv = 0;
  numHomePatchesRecv = 0;
  numHomePatchesMerged = 0;
  atomI = 0;
  forceI = 1;
}

ComputePmeCUDADevice::ComputePmeCUDADevice(CkMigrateMessage *m) {
  // __sdag_init();
  numHomePatches = 0;
//   forceCapacity = 0;
//   force = NULL;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    pmeRealSpaceComputes[iGrid] = NULL;
    enabledGrid[iGrid] = false;
    forces[iGrid] = NULL;
    forceCapacities[iGrid] = 0;
  }
  streamCreated = false;
  lock_numHomePatchesMerged = CmiCreateLock();
  lock_numPencils = CmiCreateLock();
  lock_numNeighborsRecv = CmiCreateLock();
  lock_recvAtoms = CmiCreateLock();
  numNeighborsExpected = 0;
  numStrayAtoms = 0;
  // Reset counters
  numNeighborsRecv = 0;
  numHomePatchesRecv = 0;
  numHomePatchesMerged = 0;
  atomI = 0;
  forceI = 1;
}

ComputePmeCUDADevice::~ComputePmeCUDADevice() {
  if (streamCreated) {
    cudaCheck(cudaSetDevice(deviceID));
    cudaCheck(cudaStreamDestroy(stream));
  }
  for (int j=0;j < 2;j++)
    for (int i=0;i < pmeAtomStorage[j].size();i++) {
      if (pmeAtomStorageAllocatedHere[i]) delete pmeAtomStorage[j][i];
    }
//   if (force != NULL) deallocate_host<CudaForce>(&force);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeRealSpaceComputes[iGrid] != NULL) delete pmeRealSpaceComputes[iGrid];
    if (forces[iGrid] != NULL) deallocate_host<CudaForce>(&forces[iGrid]);
    enabledGrid[iGrid] = false;
  }
  CmiDestroyLock(lock_numHomePatchesMerged);
  CmiDestroyLock(lock_numPencils);
  CmiDestroyLock(lock_numNeighborsRecv);
  CmiDestroyLock(lock_recvAtoms);
}

void ComputePmeCUDADevice::initialize(PmeGrid& pmeGrid_in, int pencilIndexY_in, int pencilIndexZ_in,
  int deviceID_in, int pmePencilType_in, CProxy_ComputePmeCUDAMgr mgrProxy_in,
  CProxy_PmeAtomFiler pmeAtomFiler_in) {

  deviceID = deviceID_in;
  cudaCheck(cudaSetDevice(deviceID));
  pmePencilType = pmePencilType_in;
  pmeGrid = pmeGrid_in;
#ifdef DEBUGM
  pmeGrid.print();
#endif
  pencilIndexY = pencilIndexY_in;
  pencilIndexZ = pencilIndexZ_in;
  mgrProxy = mgrProxy_in;
  pmeAtomFiler = pmeAtomFiler_in;
  // Size of the neighboring pencil grid, max 3x3
  yNBlocks = std::min(pmeGrid.yBlocks, 3);
  zNBlocks = std::min(pmeGrid.zBlocks, 3);
  // Local pencil is at y=0,z=0
  if (yNBlocks == 1) {
    ylo = 0;
    yhi = 0;
  } else if (yNBlocks == 2) {
    ylo = -1;
    yhi = 0;
  } else {
    ylo = -1;
    yhi = 1;
  }
  if (zNBlocks == 1) {
    zlo = 0;
    zhi = 0;
  } else if (zNBlocks == 2) {
    zlo = -1;
    zhi = 0;
  } else {
    zlo = -1;
    zhi = 1;
  }
  
  neighborForcePencilMsgs.resize(yNBlocks*zNBlocks, NULL);
  // neighborForcePencils.resize(yNBlocks*zNBlocks);
  for (int j=0;j < 2;j++)
    homePatchIndexList[j].resize(yNBlocks*zNBlocks);
  neighborPatchIndex.resize(yNBlocks*zNBlocks);

  pmeAtomStorageAllocatedHere.resize(yNBlocks*zNBlocks, false);
  SimParameters *simParams = Node::Object()->simParameters;
  for (int j=0;j < 2;j++) {
    pmeAtomStorage[j].resize(yNBlocks*zNBlocks, NULL);
    for (int z=zlo;z <= zhi;z++) {
      for (int y=ylo;y <= yhi;y++) {
        int pp = y-ylo + (z-zlo)*yNBlocks;
        int yt = (pencilIndexY + y + pmeGrid.yBlocks) % pmeGrid.yBlocks;
        int zt = (pencilIndexZ + z + pmeGrid.zBlocks) % pmeGrid.zBlocks;
        if (y == 0 && z == 0) {
          // Primary pencil
          pmeAtomStorage[j][pp] = new CudaPmeAtomStorage(pmePencilType != 3);
          pmeAtomStorage[j][pp]->setupAlch(*simParams);
        } else {
          pmeAtomStorage[j][pp] = new CpuPmeAtomStorage(pmePencilType != 3);
          pmeAtomStorage[j][pp]->setupAlch(*simParams);
        }
        pmeAtomStorageAllocatedHere[pp] = true;
      }
    }
  }

  // Create stream for this device
  createStream(stream);
  streamCreated = true;
  // CHC: enable at least 1 grid
  // CHC: do we need a different stream?
  pmeRealSpaceComputes[0] = new CudaPmeRealSpaceCompute(pmeGrid, pencilIndexY, pencilIndexZ, deviceID, stream);
  pmeRealSpaceComputes[0]->setGrid(0);
  enabledGrid[0] = true;
  if (simParams->alchOn) {
    pmeRealSpaceComputes[1] = new CudaPmeRealSpaceCompute(pmeGrid, pencilIndexY, pencilIndexZ, deviceID, stream);
    pmeRealSpaceComputes[1]->setGrid(1);
    // at least two grids are required for alchemical transformation
    enabledGrid[1] = true;
    if (simParams->alchDecouple) {
      pmeRealSpaceComputes[2] = new CudaPmeRealSpaceCompute(pmeGrid, pencilIndexY, pencilIndexZ, deviceID, stream);
      pmeRealSpaceComputes[2]->setGrid(2);
      enabledGrid[2] = true;
      pmeRealSpaceComputes[3] = new CudaPmeRealSpaceCompute(pmeGrid, pencilIndexY, pencilIndexZ, deviceID, stream);
      pmeRealSpaceComputes[3]->setGrid(3);
      enabledGrid[3] = true;
    }
    if (simParams->alchElecLambdaStart || simParams->alchThermIntOn) {
      pmeRealSpaceComputes[4] = new CudaPmeRealSpaceCompute(pmeGrid, pencilIndexY, pencilIndexZ, deviceID, stream);
      pmeRealSpaceComputes[4]->setGrid(4);
      enabledGrid[4] = true;
    }
  }
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (enabledGrid[iGrid]) {
      forceReady[iGrid] = 0;
    } else {
      forceReady[iGrid] = -1;
    }
  }
}

cudaStream_t ComputePmeCUDADevice::getStream() {
  return stream;
}

int ComputePmeCUDADevice::getDeviceID() {
  return deviceID;
}

CProxy_ComputePmeCUDAMgr ComputePmeCUDADevice::getMgrProxy() {
  return mgrProxy;
}

bool ComputePmeCUDADevice::isGridEnabled(unsigned int i) const {
  return enabledGrid[i];
}

void ComputePmeCUDADevice::setPencilProxy(CProxy_CudaPmePencilXYZ pmePencilXYZ_in) {
  if (pmePencilType != 3)
    NAMD_bug("ComputePmeCUDADevice::setPencilProxy(1), invalid pmePencilType");
  pmePencilXYZ = pmePencilXYZ_in;
}

void ComputePmeCUDADevice::setPencilProxy(CProxy_CudaPmePencilXY pmePencilXY_in) {
  if (pmePencilType != 2)
    NAMD_bug("ComputePmeCUDADevice::setPencilProxy(2), invalid pmePencilType");
  pmePencilXY = pmePencilXY_in;
}

void ComputePmeCUDADevice::setPencilProxy(CProxy_CudaPmePencilX pmePencilX_in) {
  if (pmePencilType != 1)
    NAMD_bug("ComputePmeCUDADevice::setPencilProxy(3), invalid pmePencilType");
  pmePencilX = pmePencilX_in;
}

void ComputePmeCUDADevice::activate_pencils() {
  if (pmePencilType == 1) {
    PmeStartMsg* pmeStartXMsg = new PmeStartMsg();
    for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
      if (enabledGrid[iGrid] == true) {
        pmeStartXMsg->dataGrid[iGrid] = pmeRealSpaceComputes[iGrid]->getData();
        pmeStartXMsg->dataSizes[iGrid] = pmeRealSpaceComputes[iGrid]->getDataSize();
        pmeStartXMsg->enabledGrid[iGrid] = true;
      } else {
        pmeStartXMsg->dataGrid[iGrid] = NULL;
        pmeStartXMsg->dataSizes[iGrid] = 0;
        pmeStartXMsg->enabledGrid[iGrid] = false;
      }
    }
    pmePencilX(0, pencilIndexY, pencilIndexZ).start(pmeStartXMsg);
  } else if (pmePencilType == 2) {
    PmeStartMsg* pmeStartXMsg = new PmeStartMsg();
    for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
      if (enabledGrid[iGrid] == true) {
        pmeStartXMsg->dataGrid[iGrid] = pmeRealSpaceComputes[iGrid]->getData();
        pmeStartXMsg->dataSizes[iGrid] = pmeRealSpaceComputes[iGrid]->getDataSize();
        pmeStartXMsg->enabledGrid[iGrid] = true;
      } else {
        pmeStartXMsg->dataGrid[iGrid] = NULL;
        pmeStartXMsg->dataSizes[iGrid] = 0;
        pmeStartXMsg->enabledGrid[iGrid] = false;
      }
    }
    pmePencilXY(0, 0, pencilIndexZ).start(pmeStartXMsg);
  } else if (pmePencilType == 3) {
    PmeStartMsg* pmeStartMsg = new PmeStartMsg();
    for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
      if (enabledGrid[iGrid] == true) {
        pmeStartMsg->dataGrid[iGrid] = pmeRealSpaceComputes[iGrid]->getData();
        pmeStartMsg->dataSizes[iGrid] = pmeRealSpaceComputes[iGrid]->getDataSize();
        pmeStartMsg->enabledGrid[iGrid] = true;
      } else {
        pmeStartMsg->dataGrid[iGrid] = NULL;
        pmeStartMsg->dataSizes[iGrid] = 0;
        pmeStartMsg->enabledGrid[iGrid] = false;
      }
    }
    pmePencilXYZ[0].start(pmeStartMsg);
  }
}

void ComputePmeCUDADevice::initializePatches(int numHomePatches_in) {
  numHomePatches = numHomePatches_in;
  for (int j=0;j < 2;j++)
    numPencils[j].resize(numHomePatches);
  for (int j=0;j < 2;j++)
    plList[j].resize(numHomePatches);
  for (int j=0;j < 2;j++)
    homePatchForceMsgs[j].resize(numHomePatches);
  // for (int j=0;j < 2;j++)
  //   numHomeAtoms[j].resize(numHomePatches);
  // If we have home patches, register this pencil with the neighbors and with self
  if (numHomePatches > 0) {
    for (int z=zlo;z <= zhi;z++) {
      for (int y=ylo;y <= yhi;y++) {
        int yt = (pencilIndexY + y + pmeGrid.yBlocks) % pmeGrid.yBlocks;
        int zt = (pencilIndexZ + z + pmeGrid.zBlocks) % pmeGrid.zBlocks;
        int node = mgrProxy.ckLocalBranch()->getNode(yt, zt);
        mgrProxy[node].registerNeighbor(yt, zt);
      }
    }
  }
}

void ComputePmeCUDADevice::registerNeighbor() {
  CmiLock(lock_numHomePatchesMerged);
  numNeighborsExpected++;
  CmiUnlock(lock_numHomePatchesMerged);
}

//
// Recevice atoms from patch and file them into pencils
//
void ComputePmeCUDADevice::recvAtoms(PmeAtomMsg *msg) {

  PmeAtomFiler *pmeAtomFilerPtr = pmeAtomFiler[CkMyPe()].ckLocalBranch();
  // Store "virial" and "energy" flags
  doVirial = msg->doVirial;
  doEnergy = msg->doEnergy;
  simulationStep = msg->simulationStep;
  // Store lattice
  SimParameters *simParams = Node::Object()->simParameters;
  lattice = msg->lattice;

  // Primary pencil index
  int pp0 = 0-ylo + (0-zlo)*yNBlocks;
  int p0 = 0;
  int pencilPatchIndex[9];
  int numStrayAtomsPatch = 0;
  if (pmePencilType == 3) {
    // 3D box => store atoms directly without index
    // NOTE: We don't check for stray atoms here!
    if (simParams->alchOn) {
      if (simParams->alchFepOn && !simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == false)) {
        // only FEP, no alchDecouple and alchElecLambdaStart == 0, use 2 grids
        pencilPatchIndex[p0] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2});
      }
      if (simParams->alchFepOn && simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == false)) {
        // FEP with alchDecouple, use 4 grids
        pencilPatchIndex[p0] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, msg->chargeFactors3, msg->chargeFactors4});
      }
      if (simParams->alchFepOn && simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == true)) {
        // FEP with alchDecouple and alchElecLambdaStart > 0, use 5 grids
        pencilPatchIndex[p0] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, msg->chargeFactors3, msg->chargeFactors4, msg->chargeFactors5});
      }
      if (simParams->alchFepOn && !simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == true)) {
        // FEP without alchDecouple and alchElecLambdaStart > 0, use 3 grids (1,2,5)
        pencilPatchIndex[p0] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, NULL, NULL, msg->chargeFactors5});
      }
      if (simParams->alchThermIntOn && !simParams->alchDecouple) {
        // TI estimator without alchDecouple, use 3 grids (1,2,5)
        pencilPatchIndex[p0] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, NULL, NULL, msg->chargeFactors5});
      }
      if (simParams->alchThermIntOn && simParams->alchDecouple) {
        // TI estimator with alchDecouple, use all grids
        pencilPatchIndex[p0] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, msg->chargeFactors3, msg->chargeFactors4, msg->chargeFactors5});
      }
    } else {
      // no alchemistry
      pencilPatchIndex[p0] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{});
    }
  } else {

    // File atoms
    pmeAtomFilerPtr->fileAtoms(msg->numAtoms, msg->atoms, lattice, pmeGrid,
      pencilIndexY, pencilIndexZ, ylo, yhi, zlo, zhi);

    // Loop through pencils and add atoms to pencil atom lists
    // NOTE: we only store to neighboring pencil if there are atoms to store
    int numAtomsCheck = 0;
    for (int p=0;p < 9;p++) {

      int y = (p % 3);
      int z = (p / 3);

      int pp = y + z*yNBlocks;
      int numAtoms = pmeAtomFilerPtr->getNumAtoms(p);
      if (pp == pp0) p0 = p;
      if (pp == pp0 || numAtoms > 0) {
        if (pmeGrid.yBlocks == 1 && pmeGrid.zBlocks == 1 && (y != 0 || z != 0))
          NAMD_bug("ComputePmeCUDADevice::recvAtoms, problem with atom filing");
        int* index = pmeAtomFilerPtr->getAtomIndex(p);
        if (simParams->alchOn) {
          if (simParams->alchFepOn && !simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == false)) {
            // only FEP, no alchDecouple and alchElecLambdaStart == 0, use 2 grids
            pencilPatchIndex[p] = pmeAtomStorage[atomI][pp]->addAtomsWithIndex(numAtoms, msg->atoms, index, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2});
          }
          if (simParams->alchFepOn && simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == false)) {
            // FEP with alchDecouple, use 4 grids
            pencilPatchIndex[p] = pmeAtomStorage[atomI][pp]->addAtomsWithIndex(numAtoms, msg->atoms, index, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, msg->chargeFactors3, msg->chargeFactors4});
          }
          if (simParams->alchFepOn && !simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == true)) {
            // FEP without alchDecouple and alchElecLambdaStart > 0, use 3 grids (1,2,5)
            pencilPatchIndex[p] = pmeAtomStorage[atomI][pp]->addAtomsWithIndex(numAtoms, msg->atoms, index, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, NULL, NULL, msg->chargeFactors5});
          }
          if (simParams->alchFepOn && simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == true)) {
            // FEP with alchDecouple and alchElecLambdaStart > 0, use 5 grids
            pencilPatchIndex[p] = pmeAtomStorage[atomI][pp]->addAtomsWithIndex(numAtoms, msg->atoms, index, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, msg->chargeFactors3, msg->chargeFactors4, msg->chargeFactors5});
          }
          if (simParams->alchThermIntOn && !simParams->alchDecouple) {
            // TI estimator without alchDecouple, use 3 grids (1,2,5)
            pencilPatchIndex[p] = pmeAtomStorage[atomI][pp]->addAtomsWithIndex(numAtoms, msg->atoms, index, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, NULL, NULL, msg->chargeFactors5});
          }
          if (simParams->alchThermIntOn && simParams->alchDecouple) {
            // TI estimator with alchDecouple, use all grids
            pencilPatchIndex[p] = pmeAtomStorage[atomI][pp]->addAtomsWithIndex(numAtoms, msg->atoms, index, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, msg->chargeFactors3, msg->chargeFactors4, msg->chargeFactors5});
          }
        } else {
          // no alchemistry
          pencilPatchIndex[p] = pmeAtomStorage[atomI][pp]->addAtomsWithIndex(numAtoms, msg->atoms, index, std::vector<float*>{});
        }
        // Number of patches in this storage tells you how many home patches contributed and
        // homePatchIndex (pe) tells you which patch contributed
        numAtomsCheck += numAtoms;
      }
    }

    // Deal with stray atoms
    numStrayAtomsPatch = pmeAtomFilerPtr->getNumAtoms(9);
    if (numStrayAtomsPatch > 0) {
      int* index = pmeAtomFilerPtr->getAtomIndex(9);
      CkPrintf("%d stray charges detected. Up to 10 listed below (index in patch, x, y, z):\n", numStrayAtomsPatch);
      for (int i=0;i < std::min(numStrayAtomsPatch, 10);i++) {
        int j = index[i];
        CkPrintf("%d %f %f %f\n", j, msg->atoms[j].x, msg->atoms[j].y, msg->atoms[j].z);
      }
    }

    if (numAtomsCheck + numStrayAtomsPatch < msg->numAtoms)
      NAMD_bug("ComputePmeCUDADevice::recvAtoms, missing atoms");
  }

  // Create storage for home patch forces
  PmeForceMsg *forceMsg;
  if (pmePencilType == 3 && CkNodeOf(msg->pe) == CkMyNode()) {
    // 3D FFT and compute resides on the same node => use zero-copy forces
    // CHC: forces are zero-copy so do we need this for alchDecouple?
    forceMsg = new (0, 0, 0, 0, 0, PRIORITY_SIZE) PmeForceMsg();
    forceMsg->zeroCopy = true;
  } else {
    const int alchGrid = simParams->alchOn ? 1 : 0;
    const int alchDecoupleGrid = simParams->alchDecouple ? 1: 0;
    const int alchSoftCoreOrTI = (simParams->alchElecLambdaStart > 0 || simParams->alchThermIntOn) ? 1 : 0;
    forceMsg = new (msg->numAtoms, alchGrid * msg->numAtoms,
                    alchDecoupleGrid * msg->numAtoms, alchDecoupleGrid * msg->numAtoms,
                    alchSoftCoreOrTI * msg->numAtoms, PRIORITY_SIZE) PmeForceMsg();
    forceMsg->zeroCopy = false;
  }
  forceMsg->numAtoms = msg->numAtoms;
  forceMsg->pe = msg->pe;
  forceMsg->compute = msg->compute;
  forceMsg->numStrayAtoms = numStrayAtomsPatch;

  bool done = false;
  // ----------------------------- lock start ---------------------------
  // Only after writing has finished, we get homePatchIndex
  // This quarantees that for whatever thread that receives "done=true", writing has finished on
  // ALL threads.
  CmiLock(lock_recvAtoms);
  numStrayAtoms += numStrayAtomsPatch;
  // Secure homePatchIndex. All writes after this must be inside lock-region
  int homePatchIndex = numHomePatchesRecv;
  // Store primary pencil first
  plList[atomI][homePatchIndex].push_back(PencilLocation(pp0, pencilPatchIndex[p0]));
  if (pmePencilType != 3) {
    // Go back to through neighboring pencils and store "homePatchIndex"
    for (int p=0;p < 9;p++) {

      int y = (p % 3);
      int z = (p / 3);

      int pp = y + z*yNBlocks;
      int numAtoms = pmeAtomFilerPtr->getNumAtoms(p);
      if (pp != pp0 && numAtoms > 0) {
        homePatchIndexList[atomI][pp].push_back(homePatchIndex);
        // plList[0...numHomePatches-1] = for each home patch stores the location of pencils that are
        //                                sharing it
        // plList[homePatchIndex].size() tells the number of pencils that the home patch is shared with
        plList[atomI][homePatchIndex].push_back(PencilLocation(pp, pencilPatchIndex[p]));
      }
    }
  }
  homePatchForceMsgs[atomI][homePatchIndex] = forceMsg;
  // numHomeAtoms[atomI][homePatchIndex] = msg->numAtoms;
  // Set the number of pencils contributing to this home patch
  numPencils[atomI][homePatchIndex] = plList[atomI][homePatchIndex].size();
  //
  numHomePatchesRecv++;
  if (numHomePatchesRecv == numHomePatches) {
    // Reset counter
    numHomePatchesRecv = 0;
    done = true;
  }
  CmiUnlock(lock_recvAtoms);
  // ----------------------------- lock end  ---------------------------

  // plList[atomI][homePatchIndex] array tells you the location of pencils that are sharing this home patch

  delete msg;

  if (done) {
    // Pencil has received all home patches and writing to memory is done => send atoms to neighbors
    sendAtomsToNeighbors();
  }
}

//
// Loop through pencils and send atoms to neighboring nodes
//
void ComputePmeCUDADevice::sendAtomsToNeighbors() {
  for (int z=zlo;z <= zhi;z++) {
    for (int y=ylo;y <= yhi;y++) {
      // Only send to neighbors, not self
      if (y != 0 || z != 0) {
        // NOTE: Must send atomI -value since this will change in spreadCharge(), which might occur
        // before these sends have been performed
        thisProxy[CkMyNode()].sendAtomsToNeighbor(y, z, atomI);
      }
    }
  }
  // Register primary pencil
  registerRecvAtomsFromNeighbor();
}

void ComputePmeCUDADevice::sendAtomsToNeighbor(int y, int z, int atomIval) {
  // Pencil index  
  int pp = y-ylo + (z-zlo)*yNBlocks;
  // This neighbor pencil is done, finish it up before accessing it
  pmeAtomStorage[atomIval][pp]->finish();
  // Compute destination neighbor pencil index (yt,zt)
  int yt = (pencilIndexY + y + pmeGrid.yBlocks) % pmeGrid.yBlocks;
  int zt = (pencilIndexZ + z + pmeGrid.zBlocks) % pmeGrid.zBlocks;
  int numAtoms = pmeAtomStorage[atomIval][pp]->getNumAtoms();
  CudaAtom* atoms = pmeAtomStorage[atomIval][pp]->getAtoms();
  SimParameters *simParams = Node::Object()->simParameters;
  PmeAtomPencilMsg* msgPencil;
  if (simParams->alchOn) {
    if (simParams->alchFepOn && !simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == false)) {
      msgPencil = new (numAtoms, numAtoms, numAtoms, 0, 0, 0, PRIORITY_SIZE) PmeAtomPencilMsg;
      float* chargeFactors1 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(0);
      float* chargeFactors2 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(1);
      memcpy(msgPencil->chargeFactors1, chargeFactors1, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors2, chargeFactors2, numAtoms*sizeof(float));
    }
    if (simParams->alchFepOn && simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == false)) {
      msgPencil = new (numAtoms, numAtoms, numAtoms, numAtoms, numAtoms, 0, PRIORITY_SIZE) PmeAtomPencilMsg;
      float* chargeFactors1 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(0);
      float* chargeFactors2 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(1);
      float* chargeFactors3 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(2);
      float* chargeFactors4 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(3);
      memcpy(msgPencil->chargeFactors1, chargeFactors1, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors2, chargeFactors2, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors3, chargeFactors3, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors4, chargeFactors4, numAtoms*sizeof(float));
    }
    if (simParams->alchFepOn && simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == true)) {
      msgPencil = new (numAtoms, numAtoms, numAtoms, numAtoms, numAtoms, numAtoms, PRIORITY_SIZE) PmeAtomPencilMsg;
      float* chargeFactors1 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(0);
      float* chargeFactors2 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(1);
      float* chargeFactors3 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(2);
      float* chargeFactors4 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(3);
      float* chargeFactors5 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(4);
      memcpy(msgPencil->chargeFactors1, chargeFactors1, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors2, chargeFactors2, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors3, chargeFactors3, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors4, chargeFactors4, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors5, chargeFactors5, numAtoms*sizeof(float));
    }
    if (simParams->alchFepOn && !simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == true)) {
      msgPencil = new (numAtoms, numAtoms, numAtoms, 0, 0, numAtoms, PRIORITY_SIZE) PmeAtomPencilMsg;
      float* chargeFactors1 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(0);
      float* chargeFactors2 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(1);
      float* chargeFactors5 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(4);
      memcpy(msgPencil->chargeFactors1, chargeFactors1, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors2, chargeFactors2, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors5, chargeFactors5, numAtoms*sizeof(float));
    }
    if (simParams->alchThermIntOn && !simParams->alchDecouple) {
      msgPencil = new (numAtoms, numAtoms, numAtoms, 0, 0, numAtoms, PRIORITY_SIZE) PmeAtomPencilMsg;
      float* chargeFactors1 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(0);
      float* chargeFactors2 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(1);
      float* chargeFactors5 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(4);
      memcpy(msgPencil->chargeFactors1, chargeFactors1, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors2, chargeFactors2, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors5, chargeFactors5, numAtoms*sizeof(float));
    }
    if (simParams->alchThermIntOn && simParams->alchDecouple) {
      msgPencil = new (numAtoms, numAtoms, numAtoms, numAtoms, numAtoms, numAtoms, PRIORITY_SIZE) PmeAtomPencilMsg;
      float* chargeFactors1 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(0);
      float* chargeFactors2 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(1);
      float* chargeFactors3 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(2);
      float* chargeFactors4 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(3);
      float* chargeFactors5 = pmeAtomStorage[atomIval][pp]->getAtomElecFactors(4);
      memcpy(msgPencil->chargeFactors1, chargeFactors1, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors2, chargeFactors2, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors3, chargeFactors3, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors4, chargeFactors4, numAtoms*sizeof(float));
      memcpy(msgPencil->chargeFactors5, chargeFactors5, numAtoms*sizeof(float));
    }
  } else {
    msgPencil = new (numAtoms, 0, 0, 0, 0, 0, PRIORITY_SIZE) PmeAtomPencilMsg;
  }
  memcpy(msgPencil->atoms, atoms, numAtoms*sizeof(CudaAtom));
  msgPencil->numAtoms = numAtoms;
  // Store destination pencil index
  msgPencil->y = yt;
  msgPencil->z = zt;
  // Store source pencil index
  msgPencil->srcY = pencilIndexY;
  msgPencil->srcZ = pencilIndexZ;
  // Store energy and virial flags
  msgPencil->doEnergy = doEnergy;
  msgPencil->doVirial = doVirial;
  msgPencil->simulationStep = simulationStep;
  // Store lattice
  msgPencil->lattice = lattice;
  int node = mgrProxy.ckLocalBranch()->getNode(yt, zt);
  mgrProxy[node].recvAtomsFromNeighbor(msgPencil);
}

void ComputePmeCUDADevice::recvAtomsFromNeighbor(PmeAtomPencilMsg *msg) {
  // Store into primary pencil
  int pp0 = 0-ylo + (0-zlo)*yNBlocks;
  // Compute pencil index relative to primary pencil
  int y = msg->srcY - pencilIndexY;
  if (y < ylo) y += pmeGrid.yBlocks;
  if (y > yhi) y -= pmeGrid.yBlocks;
  int z = msg->srcZ - pencilIndexZ;
  if (z < zlo) z += pmeGrid.zBlocks;
  if (z > zhi) z -= pmeGrid.zBlocks;
  if (y < ylo || y > yhi || z < zlo || z > zhi || (y == 0 && z == 0)) {
    NAMD_bug("ComputePmeCUDADevice::recvAtomsFromNeighbor, pencil index outside bounds");
  }
  // Read energy and virial flags
  doEnergy = msg->doEnergy;
  doVirial = msg->doVirial;
  simulationStep = msg->simulationStep;
  // Read lattice
  lattice = msg->lattice;
  // Pencil index where atoms came from
  int pp = y-ylo + (z-zlo)*yNBlocks;
  // Store atoms and mark down the patch index where these atoms were added
  SimParameters *simParams = Node::Object()->simParameters;
  if (simParams->alchOn) {
    if (simParams->alchFepOn && !simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == false)) {
      neighborPatchIndex[pp] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2});
    }
    if (simParams->alchFepOn && simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == false)) {
      neighborPatchIndex[pp] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, msg->chargeFactors3, msg->chargeFactors4});
    }
    if (simParams->alchFepOn && simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == true)) {
      neighborPatchIndex[pp] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, msg->chargeFactors3, msg->chargeFactors4, msg->chargeFactors5});
    }
    if (simParams->alchFepOn && !simParams->alchDecouple && (bool(simParams->alchElecLambdaStart) == true)) {
      neighborPatchIndex[pp] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, 0, 0, msg->chargeFactors5});
    }
    if (simParams->alchThermIntOn && !simParams->alchDecouple) {
      neighborPatchIndex[pp] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, 0, 0, msg->chargeFactors5});
    }
    if (simParams->alchThermIntOn && simParams->alchDecouple) {
      neighborPatchIndex[pp] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{msg->chargeFactors1, msg->chargeFactors2, msg->chargeFactors3, msg->chargeFactors4, msg->chargeFactors5});
    }
  } else {
    neighborPatchIndex[pp] = pmeAtomStorage[atomI][pp0]->addAtoms(msg->numAtoms, msg->atoms, std::vector<float*>{});
  }

  delete msg;

  registerRecvAtomsFromNeighbor();
}

void ComputePmeCUDADevice::registerRecvAtomsFromNeighbor() {
  // Primary pencil
  int pp0 = 0-ylo + (0-zlo)*yNBlocks;

  bool done = false;
  // ----------------------------- lock start ---------------------------
  CmiLock(lock_numNeighborsRecv);
  numNeighborsRecv++;
  if (numNeighborsRecv == numNeighborsExpected) {
    // Reset counter
    numNeighborsRecv = 0;
    done = true;
  }
  CmiUnlock(lock_numNeighborsRecv);
  // ----------------------------- lock end  ---------------------------

  if (done) {
    // Primary pencil has received all atoms and writing has finished => spread charge
    spreadCharge();
  }  
}

void ComputePmeCUDADevice::spreadCharge() {
  // Spread charges in primary pencil
  int pp0 = 0-ylo + (0-zlo)*yNBlocks;
  // Primary pencil is done, finish it up before accessing it
  // (clearing is done in mergeForcesOnPatch)
  pmeAtomStorage[atomI][pp0]->finish();
  // Get the number of atoms and pointer to atoms
  int numAtoms = pmeAtomStorage[atomI][pp0]->getNumAtoms();
  CudaAtom* atoms = pmeAtomStorage[atomI][pp0]->getAtoms();
  SimParameters *simParams = Node::Object()->simParameters;
  CudaAtom* atoms2 = NULL;
  CudaAtom* atoms3 = NULL;
  CudaAtom* atoms4 = NULL;
  CudaAtom* atoms5 = NULL;
  float* chargeFactors1 = NULL;
  float* chargeFactors2 = NULL;
  float* chargeFactors3 = NULL;
  float* chargeFactors4 = NULL;
  float* chargeFactors5 = NULL;
  if (simParams->alchOn) {
    chargeFactors1 = pmeAtomStorage[atomI][pp0]->getAtomElecFactors(0);
    chargeFactors2 = pmeAtomStorage[atomI][pp0]->getAtomElecFactors(1);
    allocate_host<CudaAtom>(&atoms2, numAtoms);
    if (simParams->alchDecouple) {
      chargeFactors3 = pmeAtomStorage[atomI][pp0]->getAtomElecFactors(2);
      chargeFactors4 = pmeAtomStorage[atomI][pp0]->getAtomElecFactors(3);
      allocate_host<CudaAtom>(&atoms3, numAtoms);
      allocate_host<CudaAtom>(&atoms4, numAtoms);
    }
    if (bool(simParams->alchElecLambdaStart) == true || simParams->alchThermIntOn) {
      chargeFactors5 = pmeAtomStorage[atomI][pp0]->getAtomElecFactors(4);
      allocate_host<CudaAtom>(&atoms5, numAtoms);
    }
  }
  // Flip atomI <-> forceI
  std::swap(atomI, forceI);
  // Re-allocate force buffer if needed
  reallocate_host<CudaForce>(&forces[0], &forceCapacities[0], numAtoms, 1.5f);
  if (simParams->alchOn) {
    reallocate_host<CudaForce>(&forces[1], &forceCapacities[1], numAtoms, 1.5f);
    if (simParams->alchDecouple) {
      reallocate_host<CudaForce>(&forces[2], &forceCapacities[2], numAtoms, 1.5f);
      reallocate_host<CudaForce>(&forces[3], &forceCapacities[3], numAtoms, 1.5f);
    }
    if (bool(simParams->alchElecLambdaStart) == true || simParams->alchThermIntOn) {
      reallocate_host<CudaForce>(&forces[4], &forceCapacities[4], numAtoms, 1.5f);
    }
  }
  // Setup patches and atoms
  // Lattice lattice = simParams->lattice;
  if (simParams->alchOn) {
    for (int i = 0; i < numAtoms; ++i) {
      // copy atoms and scale the charges with factors
      atoms2[i].x = atoms[i].x;
      atoms2[i].y = atoms[i].y;
      atoms2[i].z = atoms[i].z;
      atoms2[i].q = atoms[i].q * chargeFactors2[i];
      if (simParams->alchDecouple) {
        atoms3[i].x = atoms[i].x;
        atoms3[i].y = atoms[i].y;
        atoms3[i].z = atoms[i].z;
        atoms3[i].q = atoms[i].q * chargeFactors3[i];
        atoms4[i].x = atoms[i].x;
        atoms4[i].y = atoms[i].y;
        atoms4[i].z = atoms[i].z;
        atoms4[i].q = atoms[i].q * chargeFactors4[i];
      }
      if (bool(simParams->alchElecLambdaStart) == true || simParams->alchThermIntOn) {
        atoms5[i].x = atoms[i].x;
        atoms5[i].y = atoms[i].y;
        atoms5[i].z = atoms[i].z;
        atoms5[i].q = atoms[i].q * chargeFactors5[i];
      }
      atoms[i].q *= chargeFactors1[i];
    }
    pmeRealSpaceComputes[0]->copyAtoms(numAtoms, atoms);
    pmeRealSpaceComputes[1]->copyAtoms(numAtoms, atoms2);
    if (simParams->alchDecouple) {
      pmeRealSpaceComputes[2]->copyAtoms(numAtoms, atoms3);
      pmeRealSpaceComputes[3]->copyAtoms(numAtoms, atoms4);
      deallocate_host<CudaAtom>(&atoms4);
      deallocate_host<CudaAtom>(&atoms3);
    }
    if (bool(simParams->alchElecLambdaStart) == true || simParams->alchThermIntOn) {
      pmeRealSpaceComputes[4]->copyAtoms(numAtoms, atoms5);
      deallocate_host<CudaAtom>(&atoms5);
    }
    deallocate_host<CudaAtom>(&atoms2);
  } else {
    pmeRealSpaceComputes[0]->copyAtoms(numAtoms, atoms);
  }
  // Spread charge
  beforeWalltime = CmiWallTimer();
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (enabledGrid[iGrid] == true) {
      pmeRealSpaceComputes[iGrid]->spreadCharge(lattice);
    }
  }
  // Send "charge grid ready to PME solver"
  PmeRunMsg *pmeRunMsg = new PmeRunMsg();
  pmeRunMsg->doVirial = doVirial;
  pmeRunMsg->doEnergy = doEnergy;
  pmeRunMsg->simulationStep = simulationStep;
  pmeRunMsg->lattice = lattice;
  pmeRunMsg->numStrayAtoms = numStrayAtoms;
  // Reset stray atom counter
  numStrayAtoms = 0;
  switch(pmePencilType) {
    case 1:
    pmePencilX(0, pencilIndexY, pencilIndexZ).chargeGridReady(pmeRunMsg);
    break;
    case 2:
    pmePencilXY(0, 0, pencilIndexZ).chargeGridReady(pmeRunMsg);
    break;
    case 3:
    pmePencilXYZ[0].chargeGridReady(pmeRunMsg);
    break;
  }
}

//
// After PME solver is done, we return here
//
void ComputePmeCUDADevice::gatherForce() {
  traceUserBracketEvent(CUDA_PME_SPREADCHARGE_EVENT, beforeWalltime, CmiWallTimer());
  beforeWalltime = CmiWallTimer();
  // gather (i.e. un-grid) forces
  SimParameters *simParams = Node::Object()->simParameters;
//   if (simParameters->alchOn) {
//     pmeRealSpaceComputes[1]->gatherForce(lattice, force);
//     ((CudaPmeRealSpaceCompute*)(pmeRealSpaceComputes[1]))->gatherForceSetCallback(this);
//   } else {
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (enabledGrid[iGrid]) {
//       fprintf(stdout, "gatherForce at grid %u\n", iGrid);
      pmeRealSpaceComputes[iGrid]->gatherForce(lattice, forces[iGrid]);
      // Set callback that will call gatherForceDone() once gatherForce is done
      ((CudaPmeRealSpaceCompute*)(pmeRealSpaceComputes[iGrid]))->gatherForceSetCallback(this);
    }
  }
  // ((CudaPmeRealSpaceCompute*)pmeRealSpaceCompute)->waitGatherForceDone();
    // gatherForceDone();
//   }
}

static inline void gatherForceDoneLoop(int first, int last, void *result, int paraNum, void *param) {
  ComputePmeCUDADevice* c = (ComputePmeCUDADevice *)param;
  c->gatherForceDoneSubset(first, last);
}

void ComputePmeCUDADevice::gatherForceDoneSubset(int first, int last) {
  for (int homePatchIndex=first;homePatchIndex <= last;homePatchIndex++) {
    bool done = false;
    // ----------------------------- lock start ---------------------------
    // NOTE: We use node-wide lock here for the entire numPencils[] array, while
    //       we really would only need to each element but this would required
    //       numHomePatches number of locks.
    if (pmePencilType != 3) CmiLock(lock_numPencils);
    numPencils[forceI][homePatchIndex]--;
    if (numPencils[forceI][homePatchIndex] == 0) done = true;
    if (pmePencilType != 3) CmiUnlock(lock_numPencils);
    // ----------------------------- lock end  ---------------------------
    if (done) {
      // This home patch is done, launch force merging
      mergeForcesOnPatch(homePatchIndex);
    }
  }
}

void ComputePmeCUDADevice::gatherForceDone(unsigned int iGrid) {
  // CHC: prevent race condition when there are multiple pmeRealSpaceCompute objects
  forceReady[iGrid] = 1;
  bool all_force_ready = true;
//   fprintf(stdout, "gatherForceDone at grid %u\n", iGrid);
  // loop over forceReady to check if all forces are gathered
  for (unsigned int i = 0; i < NUM_GRID_MAX; ++i) {
    if (forceReady[i] == -1) continue;
    if (forceReady[i] == 0) all_force_ready = false;
  }
  if (all_force_ready) {
    for (unsigned int i = 0; i < NUM_GRID_MAX; ++i) {
      if (forceReady[i] == -1) continue;
      if (forceReady[i] == 1) forceReady[i] = 0;
    }
//     fprintf(stdout, "all force ready\n");
    // Primary pencil has the forces

    traceUserBracketEvent(CUDA_PME_GATHERFORCE_EVENT, beforeWalltime, CmiWallTimer());

    // Send forces to neighbors
    sendForcesToNeighbors();

#if CMK_SMP && USE_CKLOOP
    int useCkLoop = Node::Object()->simParameters->useCkLoop;
    if (useCkLoop >= 1) {
      CkLoop_Parallelize(gatherForceDoneLoop, 1, (void *)this, CkMyNodeSize(), 0, numHomePatches-1);
    } else
#endif

    {
      // Loop through home patches and mark the primary pencil as "done"
      for (int homePatchIndex=0;homePatchIndex < numHomePatches;homePatchIndex++) {
        bool done = false;
        // ----------------------------- lock start ---------------------------
        // NOTE: We use node-wide lock here for the entire numPencils[] array, while
        //       we really would only need to each element but this would required
        //       numHomePatches number of locks.
        if (pmePencilType != 3) CmiLock(lock_numPencils);
        numPencils[forceI][homePatchIndex]--;
        if (numPencils[forceI][homePatchIndex] == 0) done = true;
        if (pmePencilType != 3) CmiUnlock(lock_numPencils);
        // ----------------------------- lock end  ---------------------------
        if (done) {
          // This home patch is done, launch force merging
          thisProxy[CkMyNode()].mergeForcesOnPatch(homePatchIndex);
        }
      }
    }

    // In case we have no home patches, clear the primary pencil storage here
    if (numHomePatches == 0) {
      int pp0 = 0-ylo + (0-zlo)*yNBlocks;
      pmeAtomStorage[forceI][pp0]->clear();
    }
  }
}

//
// After gatherForce is done, we end up here
//
void ComputePmeCUDADevice::sendForcesToNeighbors() {
  // Primary pencil has the forces
  int pp0 = 0-ylo + (0-zlo)*yNBlocks;
  int* patchPos = pmeAtomStorage[forceI][pp0]->getPatchPos();
  SimParameters *simParams = Node::Object()->simParameters;
  const int alchGrid = simParams->alchOn ? 1 : 0;
  const int alchDecoupleGrid = simParams->alchDecouple ? 1: 0;
  const int alchSoftCoreOrTI = (simParams->alchElecLambdaStart > 0 || simParams->alchThermIntOn) ? 1 : 0;
  // Loop through neighboring pencils
  for (int z=zlo;z <= zhi;z++) {
    for (int y=ylo;y <= yhi;y++) {
      // Only send to neighbors, not self
      if (y != 0 || z != 0) {
        int pp = y-ylo + (z-zlo)*yNBlocks;
        int patchIndex = neighborPatchIndex[pp];
        int atomStart = (patchIndex == 0) ? 0 : patchPos[patchIndex-1];
        int atomEnd   = patchPos[patchIndex];
        int natom = atomEnd-atomStart;
        // copy forces
        PmeForcePencilMsg *msg;
        msg = new (natom, alchGrid * natom, alchDecoupleGrid * natom, 
                   alchDecoupleGrid * natom, alchSoftCoreOrTI * natom,
                   PRIORITY_SIZE) PmeForcePencilMsg;
        msg->numAtoms = natom;
        memcpy(msg->force, forces[0]+atomStart, natom*sizeof(CudaForce));
        if (simParams->alchOn) {
          memcpy(msg->force2, forces[1]+atomStart, natom*sizeof(CudaForce));
          if (simParams->alchDecouple) {
            memcpy(msg->force3, forces[2]+atomStart, natom*sizeof(CudaForce));
            memcpy(msg->force4, forces[3]+atomStart, natom*sizeof(CudaForce));
          }
          if (bool(simParams->alchElecLambdaStart) == true || simParams->alchThermIntOn) {
            memcpy(msg->force5, forces[4]+atomStart, natom*sizeof(CudaForce));
          }
        }
        // Calculate destination pencil index (dstY, dstZ) for this neighbor
        int dstY = (pencilIndexY + y + pmeGrid.yBlocks) % pmeGrid.yBlocks;
        int dstZ = (pencilIndexZ + z + pmeGrid.zBlocks) % pmeGrid.zBlocks;
        int node = mgrProxy.ckLocalBranch()->getNode(dstY, dstZ);
        msg->y = dstY;
        msg->z = dstZ;
        // Store source pencil index
        msg->srcY = pencilIndexY;
        msg->srcZ = pencilIndexZ;
        mgrProxy[node].recvForcesFromNeighbor(msg);
      }
    }
  }
}

void ComputePmeCUDADevice::recvForcesFromNeighbor(PmeForcePencilMsg *msg) {

  // Source pencil index
  int y = msg->srcY - pencilIndexY;
  if (y < ylo) y += pmeGrid.yBlocks;
  if (y > yhi) y -= pmeGrid.yBlocks;
  int z = msg->srcZ - pencilIndexZ;
  if (z < zlo) z += pmeGrid.zBlocks;
  if (z > zhi) z -= pmeGrid.zBlocks;

  if (y < ylo || y > yhi || z < zlo || z > zhi || (y == 0 && z == 0)) {
    NAMD_bug("ComputePmeCUDADevice::recvForcesFromNeighbor, pencil index outside bounds");
  }

  // Source pencil
  int pp = y-ylo + (z-zlo)*yNBlocks;

  // Store message (deleted in mergeForcesOnPatch)
  neighborForcePencilMsgs[pp] = msg;

  // neighborForcePencils[pp].force = new CudaForce[msg->numAtoms];
  // memcpy(neighborForcePencils[pp].force, msg->force, sizeof(CudaForce)*msg->numAtoms);
  // neighborForcePencils[pp].numAtoms = msg->numAtoms;
  // neighborForcePencils[pp].y = msg->y;
  // neighborForcePencils[pp].z = msg->z;
  // neighborForcePencils[pp].srcY = msg->srcY;
  // neighborForcePencils[pp].srcZ = msg->srcZ;
  // delete msg;

  // numPatches = number of home patches this pencil has
  int numPatches = pmeAtomStorage[forceI][pp]->getNumPatches();
  if (numPatches != homePatchIndexList[forceI][pp].size()) {
    NAMD_bug("ComputePmeCUDADevice::recvForcesFromNeighbor, numPatches incorrect");
  }
  for (int i=0;i < numPatches;i++) {
    // this pencil contributed to home patch with index "homePatchIndex"
    int homePatchIndex = homePatchIndexList[forceI][pp][i];
    // ----------------------------- lock start ---------------------------
    // NOTE: We use node-wide lock here for the entire numPencils[] array, while
    //       we really would only need to each element but this would required
    //       numHomePatches number of locks.
    bool done = false;
    CmiLock(lock_numPencils);
    numPencils[forceI][homePatchIndex]--;
    if (numPencils[forceI][homePatchIndex] == 0) done = true;
    CmiUnlock(lock_numPencils);
    // ----------------------------- lock end  ---------------------------
    if (done) {
      // This home patch is done, launch force merging
      thisProxy[CkMyNode()].mergeForcesOnPatch(homePatchIndex);
    }
  }

}

void ComputePmeCUDADevice::mergeForcesOnPatch(int homePatchIndex) {
  // We have all the forces for this patch => merge on a single Pe

  int pp0 = 0-ylo + (0-zlo)*yNBlocks;

  // Message that goes out to the compute
  PmeForceMsg *forceMsg = homePatchForceMsgs[forceI][homePatchIndex];

  SimParameters* simParams = Node::Object()->simParameters;
  if (pmePencilType == 3) {
    // 3D box => simple memory copy will do
    // Location of forces in the force[] array
    int* patchPos = pmeAtomStorage[forceI][pp0]->getPatchPos();
    // plList[homePatchIndex] array tells you the location of pencils that are sharing this home patch
    int pencilPatchIndex = plList[forceI][homePatchIndex][0].pencilPatchIndex;
    int atomStart = (pencilPatchIndex == 0) ? 0 : patchPos[pencilPatchIndex-1];
    int atomEnd   = patchPos[pencilPatchIndex];
    int numAtoms = atomEnd-atomStart;
    if (forceMsg->zeroCopy) {
      // Zero-copy, just pass the pointer
      forceMsg->force = forces[0]+atomStart;
      if (simParams->alchOn) {
        forceMsg->force2 = forces[1]+atomStart;
        if (simParams->alchDecouple) {
          forceMsg->force3 = forces[2]+atomStart;
          forceMsg->force4 = forces[3]+atomStart;
        }
        if (bool(simParams->alchElecLambdaStart) == true || simParams->alchThermIntOn) {
          forceMsg->force5 = forces[4]+atomStart;
        }
      }
    } else {
      memcpy(forceMsg->force, forces[0]+atomStart, numAtoms*sizeof(CudaForce));
      if (simParams->alchOn) {
        memcpy(forceMsg->force2, forces[1]+atomStart, numAtoms*sizeof(CudaForce));
        if (simParams->alchDecouple) {
          memcpy(forceMsg->force3, forces[2]+atomStart, numAtoms*sizeof(CudaForce));
          memcpy(forceMsg->force4, forces[3]+atomStart, numAtoms*sizeof(CudaForce));
        }
        if (bool(simParams->alchElecLambdaStart) == true || simParams->alchThermIntOn) {
          memcpy(forceMsg->force5, forces[4]+atomStart, numAtoms*sizeof(CudaForce));
        }
      }
    }
  } else {

    // Zero force array
    // memset(forceMsg->force, 0, numHomeAtoms[forceI][homePatchIndex]*sizeof(CudaForce));
    memset(forceMsg->force, 0, forceMsg->numAtoms*sizeof(CudaForce));
    if (simParams->alchOn) {
      memset(forceMsg->force2, 0, forceMsg->numAtoms*sizeof(CudaForce));
      if (simParams->alchDecouple) {
        memset(forceMsg->force3, 0, forceMsg->numAtoms*sizeof(CudaForce));
        memset(forceMsg->force4, 0, forceMsg->numAtoms*sizeof(CudaForce));
      }
      if (bool(simParams->alchElecLambdaStart) == true || simParams->alchThermIntOn) {
        memset(forceMsg->force5, 0, forceMsg->numAtoms*sizeof(CudaForce));
      }
    }

    // Store forces from primary pencil
    {
      int* patchPos = pmeAtomStorage[forceI][pp0]->getPatchPos();
      int* index = pmeAtomStorage[forceI][pp0]->getAtomIndex();
      int pencilPatchIndex = plList[forceI][homePatchIndex][0].pencilPatchIndex;
      int atomStart = (pencilPatchIndex == 0) ? 0 : patchPos[pencilPatchIndex-1];
      int atomEnd   = patchPos[pencilPatchIndex];
      int numAtoms = atomEnd-atomStart;

      // Copy in local forces that are stored in the force[] array
      for (int i=0;i < numAtoms;i++) {
        forceMsg->force[index[atomStart + i]] = forces[0][atomStart + i];
        if (simParams->alchOn) {
          forceMsg->force2[index[atomStart + i]] = forces[1][atomStart + i];
          if (simParams->alchDecouple) {
            forceMsg->force3[index[atomStart + i]] = forces[2][atomStart + i];
            forceMsg->force4[index[atomStart + i]] = forces[3][atomStart + i];
          }
          if (bool(simParams->alchElecLambdaStart) == true || simParams->alchThermIntOn) {
            forceMsg->force5[index[atomStart + i]] = forces[4][atomStart + i];
          }
        }
      }

    }

    // Add forces from neighboring pencils
    for (int j=1;j < plList[forceI][homePatchIndex].size();j++) {
      int pp               = plList[forceI][homePatchIndex][j].pp;
      int pencilPatchIndex = plList[forceI][homePatchIndex][j].pencilPatchIndex;

      int* patchPos = pmeAtomStorage[forceI][pp]->getPatchPos();
      int* index = pmeAtomStorage[forceI][pp]->getAtomIndex();
      int atomStart = (pencilPatchIndex == 0) ? 0 : patchPos[pencilPatchIndex-1];
      int atomEnd   = patchPos[pencilPatchIndex];
      int numAtoms = atomEnd-atomStart;
      CudaForce *dstForce = forceMsg->force;
      // CudaForce *srcForce = neighborForcePencils[pp].force;
      CudaForce *dstForce2 = forceMsg->force2;
      CudaForce *dstForce3 = forceMsg->force3;
      CudaForce *dstForce4 = forceMsg->force4;
      CudaForce *dstForce5 = forceMsg->force5;
      CudaForce *srcForce = neighborForcePencilMsgs[pp]->force;
      CudaForce *srcForce2 = neighborForcePencilMsgs[pp]->force2;
      CudaForce *srcForce3 = neighborForcePencilMsgs[pp]->force3;
      CudaForce *srcForce4 = neighborForcePencilMsgs[pp]->force4;
      CudaForce *srcForce5 = neighborForcePencilMsgs[pp]->force5;

      for (int i=0;i < numAtoms;i++) {
        dstForce[index[atomStart + i]].x += srcForce[atomStart + i].x;
        dstForce[index[atomStart + i]].y += srcForce[atomStart + i].y;
        dstForce[index[atomStart + i]].z += srcForce[atomStart + i].z;
        if (simParams->alchOn) {
          dstForce2[index[atomStart + i]].x += srcForce2[atomStart + i].x;
          dstForce2[index[atomStart + i]].y += srcForce2[atomStart + i].y;
          dstForce2[index[atomStart + i]].z += srcForce2[atomStart + i].z;
          if (simParams->alchDecouple) {
            dstForce3[index[atomStart + i]].x += srcForce3[atomStart + i].x;
            dstForce3[index[atomStart + i]].y += srcForce3[atomStart + i].y;
            dstForce3[index[atomStart + i]].z += srcForce3[atomStart + i].z;
            dstForce4[index[atomStart + i]].x += srcForce4[atomStart + i].x;
            dstForce4[index[atomStart + i]].y += srcForce4[atomStart + i].y;
            dstForce4[index[atomStart + i]].z += srcForce4[atomStart + i].z;
          }
          if (bool(simParams->alchElecLambdaStart) == true || simParams->alchThermIntOn) {
            dstForce5[index[atomStart + i]].x += srcForce5[atomStart + i].x;
            dstForce5[index[atomStart + i]].y += srcForce5[atomStart + i].y;
            dstForce5[index[atomStart + i]].z += srcForce5[atomStart + i].z;
          }
        }
      }

    }
  }

  // Clear storage
  plList[forceI][homePatchIndex].clear();

  // ----------------------------- lock start ---------------------------
  // bool done = false;
  CmiLock(lock_numHomePatchesMerged);
  numHomePatchesMerged++;
  if (numHomePatchesMerged == numHomePatches) {
    // Reset counter
    numHomePatchesMerged = 0;

    // Delete messages
    for (int i=0;i < neighborForcePencilMsgs.size();i++) {
      if (neighborForcePencilMsgs[i] != NULL) {
        delete neighborForcePencilMsgs[i];
        neighborForcePencilMsgs[i] = NULL;
      }
    }

    // Done merging and sending forces => clear storage
    for (int pp=0;pp < homePatchIndexList[forceI].size();pp++)
      homePatchIndexList[forceI][pp].clear();
    for (int pp=0;pp < pmeAtomStorage[forceI].size();pp++)
      pmeAtomStorage[forceI][pp]->clear();

  }
  CmiUnlock(lock_numHomePatchesMerged);
  // ----------------------------- lock end  ---------------------------

  // Patch is done => send over to the node that contains the ComputePmeCUDA compute,
  // this node will then rely the message to the Pe that originally sent the atoms
  int pe = forceMsg->pe;
  if (CkNodeOf(pe) != CkMyNode())
    thisProxy[CkNodeOf(pe)].sendForcesToPatch(forceMsg);
  else
    sendForcesToPatch(forceMsg);

}

void ComputePmeCUDADevice::sendForcesToPatch(PmeForceMsg *forceMsg) {
  // Now we're on the node that has Pe, hence "compute" -pointer is valid
  int pe                  = forceMsg->pe;
  ComputePmeCUDA *compute = forceMsg->compute;

  // Store message for use in ComputePmeCUDA, where it'll also be deleted.
  if (compute->storePmeForceMsg(forceMsg)) {
    // Enqueue on the pe that sent the atoms in the first place
    LocalWorkMsg *lmsg = compute->localWorkMsg;
    CProxy_WorkDistrib wdProxy(CkpvAccess(BOCclass_group).workDistrib);
    wdProxy[pe].enqueuePme(lmsg);
  }
}
#endif // NAMD_CUDA

#include "ComputePmeCUDAMgr.def.h"
