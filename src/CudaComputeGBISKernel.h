#ifndef CUDACOMPUTEGBISKERNEL_H
#define CUDACOMPUTEGBISKERNEL_H

#include "CudaTileListKernel.h"
#include "CudaTileListKernel.hip.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)

class CudaComputeGBISKernel {
private:

  int deviceID;

  float* intRad0;
  size_t intRad0Size;

  float* intRadS;
  size_t intRadSSize;

  float* psiSum;
  size_t psiSumSize;

  float* bornRad;
  size_t bornRadSize;

  float* dEdaSum;
  size_t dEdaSumSize;

  float* dHdrPrefix;
  size_t dHdrPrefixSize;

public:
	CudaComputeGBISKernel(int deviceID);
	~CudaComputeGBISKernel();

	void updateIntRad(const int atomStorageSize, float* intRad0H, float* intRadSH,
		cudaStream_t stream);

  void updateBornRad(const int atomStorageSize, float* bornRadH, cudaStream_t stream);

  void update_dHdrPrefix(const int atomStorageSize, float* dHdrPrefixH, cudaStream_t stream);

	void GBISphase1(CudaTileListKernel& tlKernel, const int atomStorageSize,
		const float3 lata, const float3 latb, const float3 latc, const float a_cut, float* h_psiSum,
  	cudaStream_t stream);

  void GBISphase2(CudaTileListKernel& tlKernel, const int atomStorageSize,
    const bool doEnergy, const bool doSlow,
    const float3 lata, const float3 latb, const float3 latc,
    const float r_cut, const float scaling, const float kappa, const float smoothDist,
    const float epsilon_p, const float epsilon_s,
    float4* d_forces,
    float* h_dEdaSum, cudaStream_t stream);

  void GBISphase3(CudaTileListKernel& tlKernel, const int atomStorageSize,
    const float3 lata, const float3 latb, const float3 latc, const float a_cut,
    float4* d_forces,
    cudaStream_t stream);

};

#endif // NAMD_CUDA
#endif //CUDACOMPUTEGBISKERNEL_H
