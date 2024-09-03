#ifndef CUDAPMESOLVER_H
#define CUDAPMESOLVER_H
#include "PmeSolver.h"
#include "CudaPmeSolver.decl.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
class CudaPmeXYZInitMsg : public CMessage_CudaPmeXYZInitMsg {
public:
	CudaPmeXYZInitMsg(PmeGrid& pmeGrid) : pmeGrid(pmeGrid) {}
	PmeGrid pmeGrid;
};

class CudaPmeXYInitMsg : public CMessage_CudaPmeXYInitMsg {
public:
	CudaPmeXYInitMsg(PmeGrid& pmeGrid, CProxy_CudaPmePencilXY& pmePencilXY, CProxy_CudaPmePencilZ& pmePencilZ,
		CProxy_PmePencilXYMap& xyMap, CProxy_PmePencilXMap& zMap) : 
		pmeGrid(pmeGrid), pmePencilXY(pmePencilXY), pmePencilZ(pmePencilZ), xyMap(xyMap), zMap(zMap) {}
	PmeGrid pmeGrid;
  CProxy_CudaPmePencilXY pmePencilXY;
  CProxy_CudaPmePencilZ pmePencilZ;
  CProxy_PmePencilXMap zMap;
  CProxy_PmePencilXYMap xyMap;
};

class CudaPmeXInitMsg : public CMessage_CudaPmeXInitMsg {
public:
	CudaPmeXInitMsg(PmeGrid& pmeGrid,
		CProxy_CudaPmePencilX& pmePencilX, CProxy_CudaPmePencilY& pmePencilY, CProxy_CudaPmePencilZ& pmePencilZ,
		CProxy_PmePencilXMap& xMap, CProxy_PmePencilXMap& yMap, CProxy_PmePencilXMap& zMap) : 
		pmeGrid(pmeGrid), pmePencilX(pmePencilX), pmePencilY(pmePencilY), pmePencilZ(pmePencilZ),
		xMap(xMap), yMap(yMap), zMap(zMap) {}
	PmeGrid pmeGrid;
  CProxy_CudaPmePencilX pmePencilX;
  CProxy_CudaPmePencilY pmePencilY;
  CProxy_CudaPmePencilZ pmePencilZ;
  CProxy_PmePencilXMap xMap;
  CProxy_PmePencilXMap yMap;
  CProxy_PmePencilXMap zMap;
};

class InitDeviceMsg : public CMessage_InitDeviceMsg {
public:
	InitDeviceMsg(CProxy_ComputePmeCUDADevice deviceProxy) : deviceProxy(deviceProxy) {}
	CProxy_ComputePmeCUDADevice deviceProxy;
};

class InitDeviceMsg2 : public CMessage_InitDeviceMsg2 {
public:
	InitDeviceMsg2(int deviceID, cudaStream_t stream, CProxy_ComputePmeCUDAMgr mgrProxy, CProxy_ComputePmeCUDADevice deviceProxy) : 
	deviceID(deviceID), stream(stream), mgrProxy(mgrProxy), deviceProxy(deviceProxy) {}
	int deviceID;
	cudaStream_t stream;
	CProxy_ComputePmeCUDAMgr mgrProxy;
        CProxy_ComputePmeCUDADevice deviceProxy;
};

class CudaPmePencilXYZ : public CBase_CudaPmePencilXYZ {
public:
	CudaPmePencilXYZ() {}
	CudaPmePencilXYZ(CkMigrateMessage *m) {}
	void initialize(CudaPmeXYZInitMsg *msg);
	void initializeDevice(InitDeviceMsg *msg);
	void energyAndVirialDone(unsigned int iGrid);
private:
	void backwardDone();
  CProxy_ComputePmeCUDADevice deviceProxy;
};

// CHC: struct DeviceBuffer is not trivial but only standard-layout (I don't know whether this struct needs to be compatible with memcpy)
struct DeviceBuffer {
        DeviceBuffer(int deviceID, bool isPeerDevice, std::array<float2*, NUM_GRID_MAX> dataGrid) : deviceID(deviceID), isPeerDevice(isPeerDevice), dataGrid(dataGrid) {}
        DeviceBuffer(int deviceID, bool isPeerDevice) : deviceID(deviceID), isPeerDevice(isPeerDevice) {
            for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
                dataGrid[iGrid] = NULL;
            }
        }
	bool isPeerDevice;
	int deviceID;
	cudaEvent_t event;
        // adding std::array here should not affect its standard-layout
	std::array<float2*, NUM_GRID_MAX> dataGrid;
};

// The shallow copy constructor is intentionally used
class DeviceDataMsg : public CMessage_DeviceDataMsg {
public:
        DeviceDataMsg(int i, cudaEvent_t event, std::array<float2*, NUM_GRID_MAX> dataGrid) : i(i), event(event), dataGrid(dataGrid) {}
        DeviceDataMsg(int i, cudaEvent_t event) : i(i), event(event) {
            for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
                dataGrid[iGrid] = NULL;
            }
        }
	int i;
	cudaEvent_t event;
        // When copying, we only copy the pointer address itself, not the data
	std::array<float2*, NUM_GRID_MAX> dataGrid;
};

class CudaPmePencilXY : public CBase_CudaPmePencilXY {
public:
	CudaPmePencilXY_SDAG_CODE
	CudaPmePencilXY() : numGetDeviceBuffer(0), eventCreated(false) {}
	CudaPmePencilXY(CkMigrateMessage *m) : numGetDeviceBuffer(0), eventCreated(false) {}
	~CudaPmePencilXY();
	void initialize(CudaPmeXYInitMsg *msg);
	void initializeDevice(InitDeviceMsg *msg);
private:
	void forwardDone();
	void backwardDone();
	void recvDataFromZ(PmeBlockMsg *msg);
	void start(const CkCallback &);
	void setDeviceBuffers();
	std::array<float2*, NUM_GRID_MAX> getData(const int i, const bool sameDevice);
	int deviceID;
	cudaStream_t stream;
	cudaEvent_t event;
	bool eventCreated;
	int imsgZ;
	int numDeviceBuffers;
	int numGetDeviceBuffer;
	std::vector<DeviceBuffer> deviceBuffers;
  CProxy_ComputePmeCUDADevice deviceProxy;
  CProxy_CudaPmePencilZ pmePencilZ;
  CProxy_PmePencilXMap zMap;
};

class CudaPmePencilX : public CBase_CudaPmePencilX {
public:
	CudaPmePencilX_SDAG_CODE
	CudaPmePencilX() : numGetDeviceBuffer(0), eventCreated(false) {}
	CudaPmePencilX(CkMigrateMessage *m) : numGetDeviceBuffer(0), eventCreated(false) {}
	~CudaPmePencilX();
	void initialize(CudaPmeXInitMsg *msg);
	void initializeDevice(InitDeviceMsg *msg);
private:
	void forwardDone();
	void backwardDone();
	void recvDataFromY(PmeBlockMsg *msg);
	void start(const CkCallback &);
	void setDeviceBuffers();
	std::array<float2*, NUM_GRID_MAX> getData(const int i, const bool sameDevice);
	int deviceID;
	cudaStream_t stream;
	cudaEvent_t event;
	bool eventCreated;
	int imsgY;
	int numDeviceBuffers;
	int numGetDeviceBuffer;
	std::vector<DeviceBuffer> deviceBuffers;
  CProxy_ComputePmeCUDADevice deviceProxy;
  CProxy_CudaPmePencilY pmePencilY;
  CProxy_PmePencilXMap yMap;
};

class CudaPmePencilY : public CBase_CudaPmePencilY {
public:
	CudaPmePencilY_SDAG_CODE
	CudaPmePencilY() : numGetDeviceBufferZ(0), numGetDeviceBufferX(0), eventCreated(false) {}
	CudaPmePencilY(CkMigrateMessage *m) : numGetDeviceBufferZ(0), numGetDeviceBufferX(0), eventCreated(false) {}
	~CudaPmePencilY();
	void initialize(CudaPmeXInitMsg *msg);
	void initializeDevice(InitDeviceMsg2 *msg);
private:
	void forwardDone();
	void backwardDone();
	void recvDataFromX(PmeBlockMsg *msg);
	void recvDataFromZ(PmeBlockMsg *msg);
	void start(const CkCallback &);
	void setDeviceBuffers();
	std::array<float2*, NUM_GRID_MAX> getDataForX(const int i, const bool sameDevice);
	std::array<float2*, NUM_GRID_MAX> getDataForZ(const int i, const bool sameDevice);
	int deviceID;
	cudaStream_t stream;
	cudaEvent_t event;
	bool eventCreated;
	int imsgZ, imsgX;
	int imsgZZ, imsgXX;
	int numGetDeviceBufferZ;
	int numGetDeviceBufferX;
	int numDeviceBuffersZ;
	int numDeviceBuffersX;
	std::vector<DeviceBuffer> deviceBuffersZ;
	std::vector<DeviceBuffer> deviceBuffersX;
  CProxy_CudaPmePencilX pmePencilX;
  CProxy_CudaPmePencilZ pmePencilZ;
  CProxy_PmePencilXMap xMap;
  CProxy_PmePencilXMap zMap;
};

class CudaPmePencilZ : public CBase_CudaPmePencilZ {
public:
	CudaPmePencilZ_SDAG_CODE
	CudaPmePencilZ() : numGetDeviceBufferY(0), numGetDeviceBufferXY(0), eventCreated(false) {}
	CudaPmePencilZ(CkMigrateMessage *m) : numGetDeviceBufferY(0), numGetDeviceBufferXY(0), eventCreated(false) {}
	~CudaPmePencilZ();
	void initialize(CudaPmeXInitMsg *msg);
	void initialize(CudaPmeXYInitMsg *msg);
	void initializeDevice(InitDeviceMsg2 *msg);
	void energyAndVirialDone(unsigned int iGrid);
private:
	void backwardDone();
	void recvDataFromY(PmeBlockMsg *msg);
	void start(const CkCallback &);
	void setDeviceBuffers();
	std::array<float2*, NUM_GRID_MAX> getData(const int i, const bool sameDevice);
	int deviceID;
	cudaStream_t stream;
	cudaEvent_t event;
	bool eventCreated;
	int imsgY;
	int numDeviceBuffers;
	int numGetDeviceBufferY;
	std::vector<DeviceBuffer> deviceBuffers;
  CProxy_CudaPmePencilY pmePencilY;
  CProxy_PmePencilXMap yMap;

	bool useXYslab;
	int numGetDeviceBufferXY;
  CProxy_CudaPmePencilXY pmePencilXY;
  CProxy_PmePencilXYMap xyMap;
};

#endif // NAMD_CUDA
#endif //CUDAPMESOLVER_H
