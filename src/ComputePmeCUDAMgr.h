#ifndef COMPUTEPMECUDAMGR_H
#define COMPUTEPMECUDAMGR_H

#ifdef NAMD_CUDA
#include <cuda_runtime.h>  // Needed for cudaStream_t that is used in ComputePmeCUDAMgr -class
#endif
#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif

#include <vector>

#include "PmeBase.h"
#include "PmeSolver.h"
#include "PmeSolverUtil.h"
#include "ComputePmeCUDAMgr.decl.h"
#include "HipDefines.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
class ComputePmeCUDA;

//
// Base class for thread safe atom storage
//
class PmeAtomStorage {
public:
  PmeAtomStorage(const bool useIndex) : useIndex(useIndex) {
    numAtoms = 0;
    atomCapacity = 0;
    atom = NULL;
    atomIndexCapacity = 0;
    atomIndex = NULL;
    overflowStart = 0;
    overflowEnd = 0;
    overflowAtomCapacity = 0;
    overflowAtom = NULL;
    overflowAtomIndexCapacity = 0;
    overflowAtomIndex = NULL;
    totalFactorArrays = 0;
    atomElecFactorArrays = std::vector<float*>(0);
    overflowAtomElecFactorArrays = std::vector<float*>(0);
    overflowAtomElecFactorCapacities = std::vector<int>(0);
    alchOn = false;
    alchFepOn = false;
    alchDecouple = false;
    lock_ = CmiCreateLock();
  }
  virtual ~PmeAtomStorage() {
    CmiDestroyLock(lock_);
  }

  int addAtoms(const int natom, const CudaAtom* src, const std::vector<float*>& lambdaArrays) {
    return addAtoms_(natom, src, NULL, lambdaArrays);
  }

  int addAtomsWithIndex(const int natom, const CudaAtom* src, const int* index, const std::vector<float*>& lambdaArrays) {
    return addAtoms_(natom, src, index, lambdaArrays);
  }

  // Finish up, must be called after "done" is returned by addAtoms.
  // Only the last thread that gets the "done" signal from addAtoms can enter here.
  void finish() {
    if (overflowEnd-overflowStart > 0) {
      resize_((void **)&atom, numAtoms, atomCapacity, sizeof(CudaAtom));
      if (useIndex) resize_((void **)&atomIndex, numAtoms, atomIndexCapacity, sizeof(int));
      if (alchOn) {
        for (unsigned int i = 0; i < totalFactorArrays; ++i) {
          if (enabledGrid[i] == true)
            resize_((void **)&(atomElecFactorArrays[i]), numAtoms, overflowAtomCapacity, sizeof(float));
        }
      }
      memcpy_(atom+overflowStart, overflowAtom, (overflowEnd - overflowStart)*sizeof(CudaAtom));
      if (useIndex) memcpy_(atomIndex+overflowStart, overflowAtomIndex, (overflowEnd - overflowStart)*sizeof(int));
      if (alchOn) {
        for (unsigned int i = 0; i < totalFactorArrays; ++i) {
          if (enabledGrid[i] == true)
            memcpy_(atomElecFactorArrays[i]+overflowStart, overflowAtomElecFactorArrays[i], (overflowEnd - overflowStart)*sizeof(float));
        }
      }
      overflowStart = 0;
      overflowEnd = 0;
    }
  }

  // Clear and reset storage to initial stage.
  // Only the last thread that gets the "done" signal from addAtoms can enter here.
  void clear() {
    patchPos.clear();
    numAtoms = 0;    
  }

  // Return pointer to atom data
  CudaAtom* getAtoms() {
    return atom;
  }

  // CHC: return pointer to alchemical lambdas
  float* getAtomElecFactors(unsigned int iGrid) {
    return atomElecFactorArrays[iGrid];
  }
  void setupAlch(const SimParameters& simParams) {
    alchOn = simParams.alchOn;
    alchFepOn = simParams.alchFepOn;
    alchDecouple = simParams.alchDecouple;
    if (alchOn) {
      enabledGrid.resize(NUM_GRID_MAX, false);
      totalFactorArrays = 2;
      enabledGrid[0] = true;
      enabledGrid[1] = true;
      if (alchDecouple) {
        totalFactorArrays = 4;
        enabledGrid[2] = true;
        enabledGrid[3] = true;
      }
      if (bool(simParams.alchElecLambdaStart)) {
        totalFactorArrays = 5;
        enabledGrid[4] = true;
      }
      if (simParams.alchThermIntOn) {
        totalFactorArrays = 5;
        enabledGrid[4] = true;
      }
      atomElecFactorArrays.assign(totalFactorArrays, NULL);
      overflowAtomElecFactorArrays.assign(totalFactorArrays, NULL);
      overflowAtomElecFactorCapacities.assign(totalFactorArrays, 0);
    }
  }

  // Return pointer to patch positions
  int* getPatchPos() {
    return patchPos.data();
  }

  int getNumPatches() {
    return patchPos.size();
  }

  int getNumAtoms() {
    return numAtoms;
  }

  int* getAtomIndex() {
    if (!useIndex)
      NAMD_bug("PmeAtomStorage::getAtomIndex, no indexing enabled");
    return atomIndex;
  }

protected:
  // Atom array
  CudaAtom* atom;
  // Atom index array
  int* atomIndex;
  // Overflow atom array
  CudaAtom* overflowAtom;
  // Overflow atom index array
  int* overflowAtomIndex;
  // CHC: factor arrays for alchemical transformation
  std::vector<float*> atomElecFactorArrays;
  std::vector<float*> overflowAtomElecFactorArrays;
  std::vector<int> overflowAtomElecFactorCapacities;
  std::vector<bool> enabledGrid;
  unsigned int totalFactorArrays;
  bool alchOn;
  bool alchFepOn;
  bool alchDecouple;

private:
  // If true, uses indexed atom arrays
  const bool useIndex;
  // Node lock
  CmiNodeLock lock_;
  // Data overflow
  int overflowAtomCapacity;
  int overflowAtomIndexCapacity;
  int overflowStart;
  int overflowEnd;
  // Number of atoms currently in storage
  int numAtoms;
  // Atom patch position
  std::vector<int> patchPos;
  // Atom array capacity
  int atomCapacity;
  // Atom index array capacity
  int atomIndexCapacity;

  // Resize array with 1.5x extra storage
  void resize_(void **array, int sizeRequested, int& arrayCapacity, const size_t sizeofType) {
    // If array is not NULL and has enough capacity => we have nothing to do
    if (*array != NULL && arrayCapacity >= sizeRequested) return;

    // Otherwise, allocate new array
    int newArrayCapacity = (int)(sizeRequested*1.5);
    void* newArray = alloc_(sizeofType*newArrayCapacity);

    if (*array != NULL) {
      // We have old array => copy contents to new array
      memcpy_(newArray, *array, arrayCapacity*sizeofType);
      // De-allocate old array
      dealloc_(*array);
    }

    // Set new capacity and array pointer
    arrayCapacity = newArrayCapacity;
    *array = newArray;
  }

  virtual void memcpy_(void *dst, const void* src, const int size) {
    memcpy(dst, src, size);
  }

  // CHC: use template to make the function can copy any array with indices
  // also remove the virtual keyword since no derived class overwrite it
  // I don't know why Antti-Pekka use virtual for all memory related member functions
  // did he plan to support some alternative memory managements/allocators?
  template <typename array_type>
  void copyWithIndex_(array_type* dst, const array_type* src, const int natom, const int* indexSrc) {
    for (int i=0;i < natom;i++) dst[i] = src[indexSrc[i]];
  }

  // Allocate array of size bytes
  virtual void* alloc_(const size_t size)=0;

  // Deallocate array
  virtual void dealloc_(void *p)=0;

  // Add atoms in thread-safe manner.
  // Returns the patch index where the atoms were added
  int addAtoms_(const int natom, const CudaAtom* src, const int* index, const std::vector<float*>& lambdaArrays) {
    CmiLock(lock_);
    // Accumulate position for patches:
    // atoms for patch i are in the range [ patchPos[i-1], patchPos[i]-1 ]
    int patchInd = patchPos.size();
    int ppos = (patchInd == 0) ? natom : patchPos[patchInd-1] + natom;
    patchPos.push_back(ppos);
    int pos = numAtoms;
    bool overflow = false;
    numAtoms += natom;
    // Check for overflow
    if (numAtoms > atomCapacity || (useIndex && numAtoms > atomIndexCapacity)) {
      // number of atoms exceeds capacity, store into overflow buffer
      // Note: storing to overflow should be very infrequent, most likely only
      // in the initial call
      if (overflowEnd-overflowStart == 0) {
        overflowStart = pos;
        overflowEnd = pos;
      }
      overflowEnd += natom;
      if (overflowEnd-overflowStart > overflowAtomCapacity) {
        resize_((void **)&overflowAtom, overflowEnd-overflowStart, overflowAtomCapacity, sizeof(CudaAtom));
      }
      if (useIndex && overflowEnd-overflowStart > overflowAtomIndexCapacity) {
        resize_((void **)&overflowAtomIndex, overflowEnd-overflowStart, overflowAtomIndexCapacity, sizeof(int));
      }
      // CHC: copy alchemical lambda value
      if (alchOn) {
        for (unsigned int i = 0; i < totalFactorArrays; ++i) {
          if (lambdaArrays[i] != NULL && overflowEnd-overflowStart > overflowAtomElecFactorCapacities[i]) {
            resize_((void **)&(overflowAtomElecFactorArrays[i]), overflowEnd-overflowStart, overflowAtomElecFactorCapacities[i], sizeof(float));
          }
        }
      }
      if (index != NULL) {
        if (useIndex) memcpy_(overflowAtomIndex+overflowEnd-overflowStart-natom, index, natom*sizeof(int));
        copyWithIndex_(overflowAtom+overflowEnd-overflowStart-natom, src, natom, index);
        if (alchOn) {
          for (unsigned int i = 0; i < totalFactorArrays; ++i) {
            if (lambdaArrays[i] != NULL) {
              copyWithIndex_(overflowAtomElecFactorArrays[i]+overflowEnd-overflowStart-natom, lambdaArrays[i], natom, index);
            }
          }
        }
      } else {
        memcpy_(overflowAtom+overflowEnd-overflowStart-natom, src, natom*sizeof(CudaAtom));
        if (alchOn) {
          for (unsigned int i = 0; i < totalFactorArrays; ++i) {
            if (lambdaArrays[i] != NULL) {
              memcpy_(overflowAtomElecFactorArrays[i]+overflowEnd-overflowStart-natom, lambdaArrays[i], natom*sizeof(float));
            }
          }
        }
      }
      overflow = true;
    }
    CmiUnlock(lock_);
    // If no overflow, copy to final position
    if (!overflow) {
      if (index != NULL) {
        if (useIndex) memcpy_(atomIndex+pos, index, natom*sizeof(int));
        copyWithIndex_(atom+pos, src, natom, index);
        if (alchOn) {
          for (unsigned int i = 0; i < totalFactorArrays; ++i) {
            if (lambdaArrays[i] != NULL) {
              copyWithIndex_(atomElecFactorArrays[i]+pos, lambdaArrays[i], natom, index);
            }
          }
        }
      } else {
        memcpy_(atom+pos, src, natom*sizeof(CudaAtom));
        if (alchOn) {
          for (unsigned int i = 0; i < totalFactorArrays; ++i) {
            if (lambdaArrays[i] != NULL) {
              memcpy_(atomElecFactorArrays[i]+pos, lambdaArrays[i], natom*sizeof(float));
            }
          }
        }
      }
    }
    return patchInd;
  }

};

class PmeAtomMsg : public CMessage_PmeAtomMsg {
public:
  CudaAtom *atoms;
  float *chargeFactors1;
  float *chargeFactors2;
  float *chargeFactors3;
  float *chargeFactors4;
  float *chargeFactors5;
  int numAtoms;
  int i, j;
  ComputePmeCUDA* compute;
  int pe;
  bool doEnergy, doVirial;
  Lattice lattice;
  int simulationStep;
  // int miny, minz;
};

class PmeForceMsg : public CMessage_PmeForceMsg {
public:
  CudaForce *force;
  CudaForce *force2;
  CudaForce *force3; // force from the third grid of alchDecouple
  CudaForce *force4;
  CudaForce *force5;
  int alchPartition;
  int pe;
  int numAtoms;
  int numStrayAtoms;
  bool zeroCopy;
  ComputePmeCUDA* compute;
};

class PmeLaunchMsg : public CMessage_PmeLaunchMsg {
public:
  CudaForce* force;
  int natom;
  int pe;
  ComputePmeCUDA* compute;
};

class RegisterPatchMsg : public CMessage_RegisterPatchMsg {
public:
  int i, j;
};

class NumDevicesMsg : public CMessage_NumDevicesMsg {
public:
  NumDevicesMsg(int numDevices) : numDevices(numDevices) {}
  int numDevices;
};

class PmeAtomPencilMsg : public CMessage_PmeAtomPencilMsg {
public:
  CudaAtom* atoms;
  float *chargeFactors1;
  float *chargeFactors2;
  float *chargeFactors3;
  float *chargeFactors4;
  float *chargeFactors5;
  int numAtoms;
  int y, z;
  int srcY, srcZ;
  bool doEnergy, doVirial;
  Lattice lattice;
  int simulationStep;
};

class PmeForcePencilMsg : public CMessage_PmeForcePencilMsg {
public:
  CudaForce* force;
  CudaForce *force2;
  CudaForce *force3; // force from the third grid of alchDecouple
  CudaForce *force4;
  CudaForce *force5;
  int numAtoms;
  int y, z;
  int srcY, srcZ;
};

class CProxy_ComputePmeCUDADevice;
class RecvDeviceMsg : public CMessage_RecvDeviceMsg {
public:
  CProxy_ComputePmeCUDADevice* dev;
  int numDevicesMax;
};

class PmeAtomFiler : public CBase_PmeAtomFiler {
public:
  PmeAtomFiler();
  PmeAtomFiler(CkMigrateMessage *);
  ~PmeAtomFiler();
  void fileAtoms(const int numAtoms, const CudaAtom* atoms, Lattice &lattice, const PmeGrid &pmeGrid,
    const int pencilIndexY, const int pencilIndexZ, const int ylo, const int yhi, const int zlo, const int zhi);
  // static inline int yBlock(int p) {return p % 3;}
  // static inline int zBlock(int p) {return p / 3;}
  int getNumAtoms(int p) {return pencilSize[p];}
  int* getAtomIndex(int p) {return pencil[p];}
private:
  // 9 Pencils + 1 Stay atom pencil
  int pencilSize[9+1];
  int pencilCapacity[9+1];
  int* pencil[9+1];
};


class CProxy_ComputePmeCUDAMgr;
class ComputePmeCUDADevice : public CBase_ComputePmeCUDADevice {
public:
  // ComputePmeCUDADevice_SDAG_CODE;
  ComputePmeCUDADevice();
  ComputePmeCUDADevice(CkMigrateMessage *m);
  ~ComputePmeCUDADevice();
  void initialize(PmeGrid& pmeGrid_in, int pencilIndexY_in, int pencilIndexZ_in,
    int deviceID_in, int pmePencilType_in, CProxy_ComputePmeCUDAMgr mgrProxy_in,
    CProxy_PmeAtomFiler pmeAtomFiler_in);
  int getDeviceID();
  cudaStream_t getStream();
  CProxy_ComputePmeCUDAMgr getMgrProxy();
  void setPencilProxy(CProxy_CudaPmePencilXYZ pmePencilXYZ_in);
  void setPencilProxy(CProxy_CudaPmePencilXY pmePencilXY_in);
  void setPencilProxy(CProxy_CudaPmePencilX pmePencilX_in);
  void activate_pencils();
  void initializePatches(int numHomePatches_in);
  void registerNeighbor();
  void recvAtoms(PmeAtomMsg *msg);
  void sendAtomsToNeighbors();
  void sendAtomsToNeighbor(int y, int z, int atomIval);
  void recvAtomsFromNeighbor(PmeAtomPencilMsg *msg);
  void registerRecvAtomsFromNeighbor();
  void spreadCharge();
  void gatherForce();
  void gatherForceDone(unsigned int iGrid);
  void sendForcesToNeighbors();
  void recvForcesFromNeighbor(PmeForcePencilMsg *msg);
  void mergeForcesOnPatch(int homePatchIndex);
  void sendForcesToPatch(PmeForceMsg *forceMsg);

  void gatherForceDoneSubset(int first, int last);

  bool isGridEnabled(unsigned int i) const;

private:
  // Store updated lattice
  Lattice lattice;
  // Store virial and energy flags
  bool doVirial, doEnergy;
  // PME grid definiton
  PmeGrid pmeGrid;
  // PME pencil type
  int pmePencilType;
  // Neighboring pencil bounds, [-1,1]
  int ylo, yhi, zlo, zhi;
  // Size of the neighboring pencil grid, maximum value 3. yNBlocks = yhi - ylo + 1
  int yNBlocks, zNBlocks;
  // Number of home patches for this device
  int numHomePatches;
  // Pencil location for this device
  int pencilIndexY, pencilIndexZ;

  // Number of neighbors expected to provide atoms including self
  int numNeighborsExpected;

  // Number of stray atoms
  int numStrayAtoms;

  // Node locks
  CmiNodeLock lock_numHomePatchesMerged;
  CmiNodeLock lock_numPencils;
  CmiNodeLock lock_numNeighborsRecv;
  CmiNodeLock lock_recvAtoms;

  int atomI, forceI;

  //----------------------------------------------------------------------------------
  // Book keeping
  // NOTE: We keep two copies of pmeAtomStorage and homePatchIndexList so that forces can be
  //       merged while next patch of atoms is already being received
  //----------------------------------------------------------------------------------
  // Storage for each pencil on the yNBlocks x zNBlocks grid
  std::vector< PmeAtomStorage* > pmeAtomStorage[2];
  std::vector<bool> pmeAtomStorageAllocatedHere;

  // Size numHomePatches:
  // Tells how many pencils have contributed to home patch
  std::vector<int> numPencils[2];

  // Pencil location
  struct PencilLocation {
    // Pencil index
    int pp;
    // Patch location in the pencil
    int pencilPatchIndex;
    PencilLocation(int pp, int pencilPatchIndex) : pp(pp), pencilPatchIndex(pencilPatchIndex) {}
  };

  // Size numHomePatches
  std::vector< std::vector<PencilLocation> > plList[2];

  // Size numHomePatches
  std::vector< PmeForceMsg* > homePatchForceMsgs[2];

  // // Size numHomePatches
  // std::vector<int> numHomeAtoms[2];

  std::vector< std::vector<int> > homePatchIndexList[2];

  // Number of neighbors from which we have received atoms
  int numNeighborsRecv;

  // Number of home patches we have received atom from
  int numHomePatchesRecv;

  // Number of home patches we have merged forces for
  int numHomePatchesMerged;

  // Size yNBlocks*zNBlocks
  std::vector< PmeForcePencilMsg* > neighborForcePencilMsgs;
  // std::vector< PmeForcePencil > neighborForcePencils;

  // Size yNBlocks*zNBlocks
  std::vector<int> neighborPatchIndex;
  //----------------------------------------------------------------------------------

  // CUDA stream
  cudaStream_t stream;
  bool streamCreated;
  // Device ID
  int deviceID;
  // Charge spreading and force gathering

  // Used for alchemical transformation
  int simulationStep;

  // multiple PmeRealSpaceCompute for multiple grids
  std::array<PmeRealSpaceCompute*, NUM_GRID_MAX> pmeRealSpaceComputes;
  std::array<bool, NUM_GRID_MAX> enabledGrid;
  // Host memory force array
  std::array<size_t, NUM_GRID_MAX> forceCapacities;
  std::array<CudaForce*, NUM_GRID_MAX> forces;

  // Array for avoiding race conditions in force collection
  // forceReady[i] == -1 => grid i is not enabled
  // forceReady[i] == 0  => force is not ready
  // forceReady[i] == 1  => force is ready
  // For example, if only grid 1 and 2 are used , then forceReady is initialized to {0, 0, -1, -1, -1}
  // When pmeRealSpaceComputes[i] calls gatherForceDone by the callback gatherForceSetCallback(this), forceReady[i] is set to 1
  // The actual stuff or force merging that gatherForceDone does will launch when forceReady becomes {1, 1, -1, -1, -1}
  // After that forceReady will be reset to {0, 0, -1, -1, -1} again.
  std::array<int, NUM_GRID_MAX> forceReady;

  // Proxy for the manager
  CProxy_ComputePmeCUDAMgr mgrProxy;

  // Atom filer proxy
  CProxy_PmeAtomFiler pmeAtomFiler;

  // Pencil proxy
  CProxy_CudaPmePencilXYZ pmePencilXYZ;
  CProxy_CudaPmePencilXY pmePencilXY;
  CProxy_CudaPmePencilX pmePencilX;

  // For event tracing
  double beforeWalltime;
};

class ComputePmeCUDAMgr : public CBase_ComputePmeCUDAMgr {
public:
  ComputePmeCUDAMgr_SDAG_CODE;
	ComputePmeCUDAMgr();
  ComputePmeCUDAMgr(CkMigrateMessage *);
	~ComputePmeCUDAMgr();
  void setupPencils();
  void initialize(CkQdMsg *msg);
  void initialize_pencils(CkQdMsg *msg);
  void activate_pencils(CkQdMsg *msg);
  PmeGrid getPmeGrid() {return pmeGrid;}
  int getNode(int i, int j);
  int getDevice(int i, int j);
  int getDevicePencilY(int i, int j);
  int getDevicePencilZ(int i, int j);
  int getDeviceIDPencilX(int i, int j);
  int getDeviceIDPencilY(int i, int j);
  int getDeviceIDPencilZ(int i, int j);
  void recvPencils(CProxy_CudaPmePencilXYZ xyz);
  void recvPencils(CProxy_CudaPmePencilXY xy, CProxy_CudaPmePencilZ z);
  void recvPencils(CProxy_CudaPmePencilX x, CProxy_CudaPmePencilY y, CProxy_CudaPmePencilZ z);

  void createDevicesAndAtomFiler();
  void recvDevices(RecvDeviceMsg* msg);
  void recvAtomFiler(CProxy_PmeAtomFiler filer);
  void skip();
  void recvAtoms(PmeAtomMsg *msg);
  void getHomePencil(PatchID patchID, int& homey, int& homez);
  int getHomeNode(PatchID patchID);

  bool isPmePe(int pe);
  bool isPmeNode(int node);
  bool isPmeDevice(int deviceID);

  static ComputePmeCUDAMgr* Object() {
    CProxy_ComputePmeCUDAMgr mgrProxy(CkpvAccess(BOCclass_group).computePmeCUDAMgr);
    return mgrProxy.ckLocalBranch();    
  }
protected:

private:
  void restrictToMaxPMEPencils();

  // ---------------------------------------------
  // For .ci file
  // Counter for determining numDevicesMax
  int numNodesContributed;
  int numDevicesMax;

  // Number of home patches for each device on this manager
  std::vector<int> numHomePatchesList;

  // Counter for "registerPatchDone"
  int numTotalPatches;
  // ---------------------------------------------

  // PME pencil type: 1=column, 2=slab, 3=box
  int pmePencilType;

	PmeGrid pmeGrid;

  // Number of CUDA devices on this node that are used for PME computation
  int numDevices;

  std::vector<int> xPes;
  std::vector<int> yPes;
  std::vector<int> zPes;

  // List of pencil coordinates (i,j) for each device held by this node
  struct IJ {
    int i, j;
  };
  std::vector<IJ> ijPencilX;
  std::vector<IJ> ijPencilY;
  std::vector<IJ> ijPencilZ;

  struct NodeDevice {
    int node;
    int device;
  };
  std::vector<NodeDevice> nodeDeviceList;

  // Atom filer proxy
  CProxy_PmeAtomFiler pmeAtomFiler;

  // Device proxies
  std::vector<CProxy_ComputePmeCUDADevice> deviceProxy;

  // Extra devices
  struct ExtraDevice {
    int deviceID;
    cudaStream_t stream;
  };
  std::vector<ExtraDevice> extraDevices;

  // Pencil proxies
  CProxy_CudaPmePencilXYZ pmePencilXYZ;
  CProxy_CudaPmePencilXY pmePencilXY;
  CProxy_CudaPmePencilX pmePencilX;
  CProxy_CudaPmePencilY pmePencilY;
  CProxy_CudaPmePencilZ pmePencilZ;

};
#else // NAMD_CUDA
class ComputePmeCUDAMgr : public CBase_ComputePmeCUDAMgr {
};
#endif // NAMD_CUDA

#endif // COMPUTEPMECUDAMGR_H
