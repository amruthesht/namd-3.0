#ifndef CUDAPMESOLVERUTIL_H
#define CUDAPMESOLVERUTIL_H

#ifdef NAMD_CUDA
#include <cuda.h>
#include <cufft.h>
#endif // NAMD_CUDA

#if defined(NAMD_HIP)
#ifndef NAMD_CUDA
#include <hipfft/hipfft.h>
#endif
#endif

#include <stdio.h>

#include "PmeSolverUtil.h"
#include "CudaUtils.h"
#include "CudaPmeSolverUtilKernel.h"
#include "ReductionMgr.h"
#include "HipDefines.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
void writeComplexToDisk(const float2 *d_data, const int size, const char* filename, cudaStream_t stream);
void writeHostComplexToDisk(const float2 *h_data, const int size, const char* filename);
void writeRealToDisk(const float *d_data, const int size, const char* filename, cudaStream_t stream);

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#define cufftCheck(stmt) do {						\
  cufftResult err = stmt;						\
  if (err != CUFFT_SUCCESS) {						\
  	char msg[128];	\
	  sprintf(msg, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
	  cudaDie(msg); \
  }									\
} while(0)
#endif
//
// CUDA implementation of FFTCompute
//
class CudaFFTCompute : public FFTCompute {
private:
#if defined(NAMD_CUDA) || defined(NAMD_HIP) 
  cufftHandle forwardPlan, backwardPlan;
  cufftType_t forwardType, backwardType;
#endif
  int deviceID;
	cudaStream_t stream;
	void setStream();

private:
	float* allocateData(const int dataSizeRequired);
	void plan3D(int *n, int flags);
	void plan2D(int *n, int howmany, int flags);
	void plan1DX(int *n, int howmany, int flags);
	void plan1DY(int *n, int howmany, int flags);
	void plan1DZ(int *n, int howmany, int flags);
	// int ncall, plantype;

public:
	CudaFFTCompute(int deviceID, cudaStream_t stream);
	~CudaFFTCompute();
	void forward();
	void backward();
};

//
// Cuda implementation of PmeKSpaceCompute class
//
class CudaPmePencilXYZ;
class CudaPmePencilZ;

class CudaPmeKSpaceCompute : public PmeKSpaceCompute {
private:
	int deviceID;
	cudaStream_t stream;
	// Device memory versions of (bm1, bm2, bm3)
	float *d_bm1, *d_bm2, *d_bm3;
	//float *prefac_x, *prefac_y, *prefac_z;
	struct EnergyVirial {
		double energy;
		double virial[9];
	};
	EnergyVirial* d_energyVirial;
	EnergyVirial* h_energyVirial;
	cudaEvent_t copyEnergyVirialEvent;
	bool ortho;
  // Check counter for event polling in energyAndVirialCheck()
  int checkCount;
	static void energyAndVirialCheck(void *arg, double walltime);
	CudaPmePencilXYZ* pencilXYZPtr;
	CudaPmePencilZ* pencilZPtr;
public:
	CudaPmeKSpaceCompute(PmeGrid pmeGrid, const int permutation,
		const int jblock, const int kblock, double kappa,
		int deviceID, cudaStream_t stream, unsigned int iGrid = 0);
	~CudaPmeKSpaceCompute();
	void solve(Lattice &lattice, const bool doEnergy, const bool doVirial, float* data);
	void waitEnergyAndVirial();
	double getEnergy();
	void getVirial(double *virial);
	void energyAndVirialSetCallback(CudaPmePencilXYZ* pencilPtr);
	void energyAndVirialSetCallback(CudaPmePencilZ* pencilPtr);
};

//
// Cuda implementation of PmeRealSpaceCompute class
//

class ComputePmeCUDADevice;

class CudaPmeRealSpaceCompute : public PmeRealSpaceCompute {
private:
#ifdef NAMD_CUDA
	bool gridTexObjActive;
	cudaTextureObject_t gridTexObj;
	int tex_data_len;
	float* tex_data;
#else
	int grid_data_len;
	float* grid_data;
#endif
	int deviceID;
	cudaStream_t stream;
	void setupGridData(float* data, int data_len);
	// Device memory for atoms
	size_t d_atomsCapacity;
	CudaAtom* d_atoms;
	// Device memory for patches
	// int d_patchesCapacity;
	// PatchInfo* d_patches;
	// Device memory for forces
	size_t d_forceCapacity;
	CudaForce* d_force;
	// // Device memory for self energy
	// double* d_selfEnergy;
  // Events
  cudaEvent_t gatherForceEvent;
  // Check counter for event polling
  int checkCount;
  // Store device pointer for event polling
  ComputePmeCUDADevice* devicePtr;
  static void cuda_gatherforce_check(void *arg, double walltime);
public:
	CudaPmeRealSpaceCompute(PmeGrid pmeGrid, const int jblock, const int kblock,
		int deviceID, cudaStream_t stream);
	~CudaPmeRealSpaceCompute();
	void copyAtoms(const int numAtoms, const CudaAtom* atoms);
	void spreadCharge(Lattice &lattice);
	void gatherForce(Lattice &lattice, CudaForce* force);
	void gatherForceSetCallback(ComputePmeCUDADevice* devicePtr_in);
	void waitGatherForceDone();
};

//
// Cuda implementation of PmeTranspose class
//
class CudaPmeTranspose : public PmeTranspose {
private:
	int deviceID;
	cudaStream_t stream;
	float2* d_data;
#ifndef P2P_ENABLE_3D
	float2* d_buffer;
#endif
	// List of device data pointers for transpose destinations on:
	// (a) this device on a different pencil (e.g. in XYZ->YZX transpose, on Y -pencil)
	// (b) different device on a different pencil
	// If NULL, use the local d_data -buffer
	std::vector<float2*> dataPtrsYZX;
	std::vector<float2*> dataPtrsZXY;

	// Batch data
	int max_nx_YZX[3];
	TransposeBatch<float2> *batchesYZX;
	int max_nx_ZXY[3];
	TransposeBatch<float2> *batchesZXY;

	void copyDataToPeerDevice(const int iblock,
		const int iblock_out, const int jblock_out, const int kblock_out,
		int deviceID_out, int permutation_out, float2* data_out);
public:
	CudaPmeTranspose(PmeGrid pmeGrid, const int permutation,
		const int jblock, const int kblock, int deviceID, cudaStream_t stream);
	~CudaPmeTranspose();
	void setDataPtrsYZX(std::vector<float2*>& dataPtrsNew, float2* data);
	void setDataPtrsZXY(std::vector<float2*>& dataPtrsNew, float2* data);
	void transposeXYZtoYZX(const float2* data);
	void transposeXYZtoZXY(const float2* data);
	// void waitTransposeDone();
	void waitStreamSynchronize();
	void copyDataDeviceToHost(const int iblock, float2* h_data, const int h_dataSize);
	void copyDataHostToDevice(const int iblock, float2* data_in, float2* data_out);
#ifndef P2P_ENABLE_3D
	void copyDataDeviceToDevice(const int iblock, float2* data_out);
	float2* getBuffer(const int iblock);
#endif
	void copyDataToPeerDeviceYZX(const int iblock, int deviceID_out, int permutation_out, float2* data_out);
	void copyDataToPeerDeviceZXY(const int iblock, int deviceID_out, int permutation_out, float2* data_out);
};


/** PME for single GPU case, where data persists on GPU
 * calls real space, FFT, and K space parts
 * receives atom and charge data as float4 * allocated on device
 * returns force data as float3 * allocated on device
 * returns energy and virial allocated on device
 */
class CudaPmeOneDevice {
  public:
    PmeGrid pmeGrid;
    int deviceID;
	int deviceIndex;
    cudaStream_t stream;

    int natoms;
    size_t num_used_grids;
    
    float4* d_atoms;
    int* d_partition;
    float3* d_forces;
    float* d_scaling_factors; // alchemical scaling factors
#ifndef USE_TABLE_ARRAYS
    cudaTextureObject_t* gridTexObjArrays;
#endif

    float* d_grids; /**< on device grid of charge before forward FFT R->C,
                     then grid of potential after backward FFT C->R */
    

    float2* d_trans;  /**< on device FFT transformation to complex */

    size_t gridsize;
    size_t transize;

#if defined(NAMD_CUDA) || defined(NAMD_HIP) //to enable when hipfft full support is ready
    cufftHandle* forwardPlans;
    cufftHandle* backwardPlans;
#endif

    float *d_bm1;
    float *d_bm2;
    float *d_bm3;

    double kappa;

    struct EnergyVirial {
      double energy;
      double virial[6];
    };
    EnergyVirial* d_energyVirials;
    EnergyVirial* h_energyVirials;

    bool self_energy_alch_first_time; // check if this is the first time to compute self energy for alch
    bool force_scaling_alch_first_time; // check if this is the first time to compute the force scaling factors
    double *d_selfEnergy;  // on device
    double *d_selfEnergy_FEP;
    double *d_selfEnergy_TI_1;
    double *d_selfEnergy_TI_2;
    double selfEnergy;     // remains constant for a given set of charges
    double selfEnergy_FEP;
    double selfEnergy_TI_1;
    double selfEnergy_TI_2;
    int m_step;

    CudaPmeOneDevice(PmeGrid pmeGrid_, int deviceID_, int deviceIndex_);
    ~CudaPmeOneDevice();

    void compute(
        const Lattice &lattice,
#if 0
        const CudaAtom *d_atoms, /**< device float4 buffer x/y/z/q */
        CudaForce *d_force,      /**< device float3 buffer fx/fy/fz */
        int natoms,              /**< length of buffers */
#endif
        int doEnergyVirial,
        int step
        );

    void finishReduction( bool doEnergyVirial);
private:
  void calcSelfEnergyAlch(int step);
  void scaleAndComputeFEPEnergyVirials(const EnergyVirial* energyVirials, int step, double& energy, double& energy_F, double (&virial)[9]);
  void scaleAndComputeTIEnergyVirials(const EnergyVirial* energyVirials, int step, double& energy, double& energy_TI_1, double& energy_TI_2, double (&virial)[9]);
  void scaleAndMergeForce(int step);
};


#endif // NAMD_CUDA
#endif // CUDAPMESOLVERUTIL_H

