#ifndef DEVICECUDA_H
#define DEVICECUDA_H

#ifdef NAMD_CUDA
#include <cuda_runtime.h>
#ifdef NODEGROUP_FORCE_REGISTER
#ifdef NAMD_NCCL_ALLREDUCE
#include "nccl.h"
#endif
#endif

#endif  // NAMD_CUDA

#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#ifdef NODEGROUP_FORCE_REGISTER
#ifdef NAMD_NCCL_ALLREDUCE
#include "rccl.h"
#endif
#endif
#endif  // NAMD_HIP

#include "HipDefines.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)

#define CUDA_PME_SPREADCHARGE_EVENT 90
#define CUDA_PME_GATHERFORCE_EVENT 91
#define CUDA_BONDED_KERNEL_EVENT 92
#define CUDA_DEBUG_EVENT 93
#define CUDA_NONBONDED_KERNEL_EVENT 94
#define CUDA_GBIS1_KERNEL_EVENT 95
#define CUDA_GBIS2_KERNEL_EVENT 96
#define CUDA_GBIS3_KERNEL_EVENT 97

#define CUDA_EVENT_ID_POLL_REMOTE 98
#define CUDA_TRACE_POLL_REMOTE \
  traceUserEvent(CUDA_EVENT_ID_POLL_REMOTE)
#define CUDA_EVENT_ID_POLL_LOCAL 99
#define CUDA_TRACE_POLL_LOCAL \
  traceUserEvent(CUDA_EVENT_ID_POLL_LOCAL)
#define CUDA_EVENT_ID_BASE 100
#define CUDA_TRACE_REMOTE(START,END) \
  do { int dev; cudaGetDevice(&dev); traceUserBracketEvent( \
       CUDA_EVENT_ID_BASE + 2 * dev, START, END); } while (0)
#define CUDA_TRACE_LOCAL(START,END) \
  do { int dev; cudaGetDevice(&dev); traceUserBracketEvent( \
       CUDA_EVENT_ID_BASE + 2 * dev + 1, START, END); } while (0)


//
// Class that handles PE <=> CUDA device mapping
//
class DeviceCUDA {

private:
	// Command line argument settings
	char *devicelist;
	int usedevicelist;
	int devicesperreplica;
	int ignoresharing;
	int mergegrids;
	int nomergegrids;
	int nostreaming;

	// Number of devices on this physical node
	int deviceCount;

	// Number of devices on this physical node that are used for computation
	int ndevices;

	// List of device IDs on this physical node that are used for computation
	int *devices;

	// Number of devices that are used for computation by this node
	int nnodedevices;

	// List of device IDs that are used for computation by this node
	int *nodedevices;

	// True when GPU is shared between PEs
	bool sharedGpu;
	// Index of next GPU sharing this GPU
	int nextPeSharingGpu;
	// Index of the master PE for this GPU
	int masterPe;
	// Number of PEs that share this GPU
	int numPesSharingDevice;
	// List of PEs that share this GPU
	int *pesSharingDevice;
	// True when what???
	int gpuIsMine;

	// Device ID for this Pe
	int deviceID;

	// Device properties for all devices on this node
	cudaDeviceProp* deviceProps;

	int deviceIndex; // Position of device in devices[] list
#ifdef NODEGROUP_FORCE_REGISTER
#ifdef NAMD_NCCL_ALLREDUCE
 	ncclUniqueId ncclId;
  	ncclComm_t   ncclComm;
#endif
#endif
	bool  reservePme;     // true if a device is doing only PME
  	int	  pmeDevice;      // holds deviceID of device doing PME
	int   pmeDeviceIndex; // holds positions of the device doing PME on the devices[] list
  	bool  isPmeDevice;    // true if pmeDevice == deviceID
  	int   masterDevice;   // holds index of device that should do I/O on CUDASOA
  	bool  isMasterDevice; // true if this device should do IO

	void register_user_events();

public:
	DeviceCUDA();
	~DeviceCUDA();

	void initialize();

	int getDeviceCount() {return deviceCount;}
	int getNumDevice() {return nnodedevices;}

	bool device_shared_with_pe(int pe);
	bool one_device_per_node();

	int getNoStreaming() {return nostreaming;}
	int getNoMergeGrids() {return nomergegrids;}
	int getMergeGrids() {return mergegrids;}
	void setMergeGrids(const int val) {mergegrids = val;}

	bool getSharedGpu() {return sharedGpu;}
	int getNextPeSharingGpu() {return nextPeSharingGpu;}
	int getMasterPe() {return masterPe;}
	int getNumPesSharingDevice() {return numPesSharingDevice;}
	int getPesSharingDevice(const int i) {return pesSharingDevice[i];}

	int getGpuIsMine() {return gpuIsMine;}
	void setGpuIsMine(const int val) {gpuIsMine = val;}

	int getDeviceID() {return deviceID;}
	int getDeviceIDbyRank(int rank) {return nodedevices[rank];}
	int getDeviceIDforPe(int pe);
	int getMasterPeForDeviceID(int deviceID);

	int getMaxNumThreads();
	int getMaxNumBlocks();

	void setupDevicePeerAccess();

#ifdef NODEGROUP_FORCE_REGISTER
#ifdef NAMD_NCCL_ALLREDUCE
  ncclUniqueId getNcclUniqueId(){ return ncclId; }
  ncclComm_t   getNcclComm(){return ncclComm; }
  void setNcclUniqueId(ncclUniqueId &other) { ncclId = other;}
  void setupNcclUniqueId(); // this one gets called by rank 0
  void setupNcclComm();
#endif
#endif

  bool isGpuReservedPme() { return reservePme; }
  int  getPmeDevice()   { return pmeDevice;   }
  int  getDeviceIndex() { return deviceIndex; }
  int  getPmeDeviceIndex() { return pmeDeviceIndex; }
  bool getIsPmeDevice() { return isPmeDevice; }
  bool getIsMasterDevice();
};
#endif //NAMD_CUDA

#endif // DEVICECUDA_H
