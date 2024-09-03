#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#ifdef NAMD_CUDA
#include <cuda_runtime.h>
#endif
#ifdef NAMD_HIP 
#include <hip/hip_runtime.h>
#endif
#include "HipDefines.h"
#include "ComputeNonbondedUtil.h"
#include "ComputePmeCUDAMgr.h"
#include "CudaPmeSolver.h"
#include "CudaPmeSolverUtil.h"
#include "Node.h"
#include "PatchData.h"

#include "NamdEventsProfiling.h"
#include "TestArray.h"
#include "DeviceCUDA.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
extern __thread DeviceCUDA *deviceCUDA;

extern "C" void CcdCallBacksReset(void *ignored, double curWallTime);  // fix Charm++

void writeComplexToDisk(const float2 *d_data, const int size, const char* filename, cudaStream_t stream) {
  fprintf(stderr, "writeComplexToDisk %d %s\n", size, filename);
  float2* h_data = new float2[size];
  copy_DtoH<float2>(d_data, h_data, size, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  FILE *handle = fopen(filename, "w");
  for (int i=0;i < size;i++)
    fprintf(handle, "%f %f\n", h_data[i].x, h_data[i].y);
  fclose(handle);
  delete [] h_data;
}

void writeHostComplexToDisk(const float2 *h_data, const int size, const char* filename) {
  FILE *handle = fopen(filename, "w");
  for (int i=0;i < size;i++)
    fprintf(handle, "%f %f\n", h_data[i].x, h_data[i].y);
  fclose(handle);
}

void writeRealToDisk(const float *d_data, const int size, const char* filename, cudaStream_t stream) {
  fprintf(stderr, "writeRealToDisk %d %s\n", size, filename);
  float* h_data = new float[size];
  copy_DtoH<float>(d_data, h_data, size, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  FILE *handle = fopen(filename, "w");
  for (int i=0;i < size;i++)
    fprintf(handle, "%f\n", h_data[i]);
  fclose(handle);
  delete [] h_data;
}

	CudaFFTCompute::CudaFFTCompute(int deviceID, cudaStream_t stream) 
    : deviceID(deviceID), stream(stream) {
    }

void CudaFFTCompute::plan3D(int *n, int flags) {
  cudaCheck(cudaSetDevice(deviceID));
  forwardType = CUFFT_R2C;
  backwardType = CUFFT_C2R;
  cufftCheck(cufftPlan3d(&forwardPlan, n[2], n[1], n[0], CUFFT_R2C));
  cufftCheck(cufftPlan3d(&backwardPlan, n[2], n[1], n[0], CUFFT_C2R));
  setStream();
  // plantype = 3;
}

void CudaFFTCompute::plan2D(int *n, int howmany, int flags) {
  cudaCheck(cudaSetDevice(deviceID));
  forwardType = CUFFT_R2C;
  backwardType = CUFFT_C2R;
  int nt[2] = {n[1], n[0]};
  cufftCheck(cufftPlanMany(&forwardPlan, 2, nt, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, howmany));
  cufftCheck(cufftPlanMany(&backwardPlan, 2, nt, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, howmany));
  setStream();
  // plantype = 2;
}

void CudaFFTCompute::plan1DX(int *n, int howmany, int flags) {
  cudaCheck(cudaSetDevice(deviceID));
  forwardType = CUFFT_R2C;
  backwardType = CUFFT_C2R;
  cufftCheck(cufftPlanMany(&forwardPlan, 1, n, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, howmany));
  cufftCheck(cufftPlanMany(&backwardPlan, 1, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, howmany));
  setStream();
  // plantype = 1;
}

void CudaFFTCompute::plan1DY(int *n, int howmany, int flags) {
  cudaCheck(cudaSetDevice(deviceID));
  forwardType = CUFFT_C2C;
  backwardType = CUFFT_C2C;
  cufftCheck(cufftPlanMany(&forwardPlan, 1, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, howmany));
  cufftCheck(cufftPlanMany(&backwardPlan, 1, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, howmany));
  setStream();
  // plantype = 1;
}

void CudaFFTCompute::plan1DZ(int *n, int howmany, int flags) {
  cudaCheck(cudaSetDevice(deviceID));
  forwardType = CUFFT_C2C;
  backwardType = CUFFT_C2C;
  cufftCheck(cufftPlanMany(&forwardPlan, 1, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, howmany));
  cufftCheck(cufftPlanMany(&backwardPlan, 1, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, howmany));
  setStream();
  // plantype = 1;
}

CudaFFTCompute::~CudaFFTCompute() {
  cudaCheck(cudaSetDevice(deviceID));
	cufftCheck(cufftDestroy(forwardPlan));
	cufftCheck(cufftDestroy(backwardPlan));
  if (dataSrcAllocated) deallocate_device<float>(&dataSrc);
  if (dataDstAllocated) deallocate_device<float>(&dataDst);
}

float* CudaFFTCompute::allocateData(const int dataSizeRequired) {
  cudaCheck(cudaSetDevice(deviceID));
  float* tmp = NULL;
  allocate_device<float>(&tmp, dataSizeRequired);
  return tmp;
}

// int ncall = 0;

void CudaFFTCompute::forward() {
  cudaCheck(cudaSetDevice(deviceID));
  // ncall++;
  if (forwardType == CUFFT_R2C) {
    cufftCheck(cufftExecR2C(forwardPlan, (cufftReal *)dataSrc, (cufftComplex *)dataDst));
#ifdef TESTPID
    if (1) {
      cudaCheck(cudaStreamSynchronize(stream));
      fprintf(stderr, "AP FORWARD FFT\n");
      fprintf(stderr, "COPY DEVICE ARRAYS BACK TO HOST\n");
      int m = dataDstSize;
      float *tran = 0;
      allocate_host<float>(&tran, m);
      copy_DtoH<float>(dataDst, tran, m, stream);
      cudaCheck(cudaStreamSynchronize(stream));
      TestArray_write<float>("tran_charge_grid_good.bin",
          "transformed charge grid good", tran, m);
      deallocate_host<float>(&tran);
    }
#endif

    // if (ncall == 1) {
    //   writeComplexToDisk((float2 *)dataSrc, (isize/2+1)*jsize*ksize, "dataSrc.txt", stream);
    // }

    // if (ncall == 1 && plantype == 2) {
    //   writeComplexToDisk((float2 *)data, (isize/2+1)*jsize*ksize, "data_fx_fy_z.txt", stream);
    // }

  } else if (forwardType == CUFFT_C2C) {
    // nc2cf++;
    // if (ncall == 1 && nc2cf == 1)
    //   writeComplexToDisk((float2 *)data, 33*64*64, "data_y_z_fx.txt");
    // else if (ncall == 1 && nc2cf == 2)
    //   writeComplexToDisk((float2 *)data, 33*64*64, "data_z_fx_fy.txt");
    cufftCheck(cufftExecC2C(forwardPlan, (cufftComplex *)dataSrc, (cufftComplex *)dataDst, CUFFT_FORWARD));
    // fprintf(stderr, "ncall %d plantype %d\n", ncall, plantype);
    // if (ncall == 1 && plantype == 1 && isize == 62) {
    //   writeComplexToDisk((float2 *)data, isize*jsize*(ksize/2+1), "data_fy_z_fx.txt", stream);
    // }
    // if (ncall == 1 && nc2cf == 1)
    //   writeComplexToDisk((float2 *)data, 33*64*64, "data_fy_z_fx.txt");
    // else if (ncall == 1 && nc2cf == 2)
    //   writeComplexToDisk((float2 *)data, 33*64*64, "data_fz_fx_fy.txt");
  } else {
    cudaNAMD_bug("CudaFFTCompute::forward(), unsupported FFT type");
  }
}

void CudaFFTCompute::backward() {
  cudaCheck(cudaSetDevice(deviceID));
  if (backwardType == CUFFT_C2R) {
    // if (ncall == 1) {
    //   if (plantype == 1)
    //     writeComplexToDisk((float2 *)data, 33*64*64, "data_fx_by_bz.txt");
    //   else
    //     writeComplexToDisk((float2 *)data, 33*64*64, "data_fx_fy_fz_2.txt");
    // }
    cufftCheck(cufftExecC2R(backwardPlan, (cufftComplex *)dataDst, (cufftReal *)dataSrc));
#ifdef TESTPID
  if (1) {
    cudaCheck(cudaStreamSynchronize(stream));
    fprintf(stderr, "AP BACKWARD FFT\n");
    fprintf(stderr, "COPY DEVICE ARRAYS BACK TO HOST\n");
    float *grid;
    int gridsize = dataSrcSize;
    allocate_host<float>(&grid, gridsize);
    copy_DtoH<float>((float*)dataSrc, grid, gridsize, stream);
    cudaCheck(cudaStreamSynchronize(stream));
    TestArray_write<float>("potential_grid_good.bin",
        "potential grid good", grid, gridsize);
    deallocate_host<float>(&grid);
  }
#endif

    // if (ncall == 1)
    //   if (plantype == 1)
    //     writeRealToDisk(data, 64*64*64, "data_bx_by_bz_1D.txt");
    //   else
    //     writeRealToDisk(data, 64*64*64, "data_bx_by_bz_3D.txt");
  } else if (backwardType == CUFFT_C2C) {
    // nc2cb++;
    // if (ncall == 1 && nc2cb == 1)
    //   writeComplexToDisk((float2 *)data, 33*64*64, "data_fz_fx_fy_2.txt");
    // else if (ncall == 1 && nc2cb == 2)
    //   writeComplexToDisk((float2 *)data, 33*64*64, "data_fy_bz_fx.txt");
    cufftCheck(cufftExecC2C(backwardPlan, (cufftComplex *)dataDst, (cufftComplex *)dataSrc, CUFFT_INVERSE));
    // if (ncall == 1 && nc2cb == 1)
    //   writeComplexToDisk((float2 *)data, 33*64*64, "data_bz_fx_fy.txt");
    // else if (ncall == 1 && nc2cb == 2)
    //   writeComplexToDisk((float2 *)data, 33*64*64, "data_by_bz_fx.txt");
  } else {
    cudaNAMD_bug("CudaFFTCompute::backward(), unsupported FFT type");
  }
}

void CudaFFTCompute::setStream() {
  cudaCheck(cudaSetDevice(deviceID));
  cufftCheck(cufftSetStream(forwardPlan, stream));
  cufftCheck(cufftSetStream(backwardPlan, stream));
}


CudaPmeKSpaceCompute::CudaPmeKSpaceCompute(PmeGrid pmeGrid, const int permutation,
  const int jblock, const int kblock, double kappa, int deviceID, cudaStream_t stream, unsigned int iGrid) : 
  PmeKSpaceCompute(pmeGrid, permutation, jblock, kblock, kappa, iGrid),
  deviceID(deviceID), stream(stream) {

  cudaCheck(cudaSetDevice(deviceID));

  // Copy bm1 -> prefac_x on GPU memory
  float *bm1f = new float[pmeGrid.K1];
  float *bm2f = new float[pmeGrid.K2];
  float *bm3f = new float[pmeGrid.K3];
  for (int i=0;i < pmeGrid.K1;i++) bm1f[i] = (float)bm1[i];
  for (int i=0;i < pmeGrid.K2;i++) bm2f[i] = (float)bm2[i];
  for (int i=0;i < pmeGrid.K3;i++) bm3f[i] = (float)bm3[i];
  allocate_device<float>(&d_bm1, pmeGrid.K1);
  allocate_device<float>(&d_bm2, pmeGrid.K2);
  allocate_device<float>(&d_bm3, pmeGrid.K3);
  copy_HtoD_sync<float>(bm1f, d_bm1, pmeGrid.K1);
  copy_HtoD_sync<float>(bm2f, d_bm2, pmeGrid.K2);
  copy_HtoD_sync<float>(bm3f, d_bm3, pmeGrid.K3);
  delete [] bm1f;
  delete [] bm2f;
  delete [] bm3f;
  allocate_device<EnergyVirial>(&d_energyVirial, 1);
  allocate_host<EnergyVirial>(&h_energyVirial, 1);
  // cudaCheck(cudaEventCreateWithFlags(&copyEnergyVirialEvent, cudaEventDisableTiming));
  cudaCheck(cudaEventCreate(&copyEnergyVirialEvent));
  // ncall = 0;
}

CudaPmeKSpaceCompute::~CudaPmeKSpaceCompute() {
  cudaCheck(cudaSetDevice(deviceID));
  deallocate_device<float>(&d_bm1);
  deallocate_device<float>(&d_bm2);
  deallocate_device<float>(&d_bm3);
  deallocate_device<EnergyVirial>(&d_energyVirial);
  deallocate_host<EnergyVirial>(&h_energyVirial);
  cudaCheck(cudaEventDestroy(copyEnergyVirialEvent));
}

void CudaPmeKSpaceCompute::solve(Lattice &lattice, const bool doEnergy, const bool doVirial, float* data) {
#if 0
  // Check lattice to make sure it is updating for constant pressure
  fprintf(stderr, "K-SPACE LATTICE  %g %g %g  %g %g %g  %g %g %g\n",
      lattice.a().x, lattice.a().y, lattice.a().z,
      lattice.b().x, lattice.b().y, lattice.b().z,
      lattice.c().x, lattice.c().y, lattice.c().z);
#endif
  cudaCheck(cudaSetDevice(deviceID));

  const bool doEnergyVirial = (doEnergy || doVirial);

  int nfft1, nfft2, nfft3;
  float *prefac1, *prefac2, *prefac3;

  BigReal volume = lattice.volume();
  Vector a_r = lattice.a_r();
  Vector b_r = lattice.b_r();
  Vector c_r = lattice.c_r();
  float recip1x, recip1y, recip1z;
  float recip2x, recip2y, recip2z;
  float recip3x, recip3y, recip3z;

  if (permutation == Perm_Z_cX_Y) {
    // Z, X, Y
    nfft1 = pmeGrid.K3;
    nfft2 = pmeGrid.K1;
    nfft3 = pmeGrid.K2;
    prefac1 = d_bm3;
    prefac2 = d_bm1;
    prefac3 = d_bm2;
    recip1x = c_r.z;
    recip1y = c_r.x;
    recip1z = c_r.y;
    recip2x = a_r.z;
    recip2y = a_r.x;
    recip2z = a_r.y;
    recip3x = b_r.z;
    recip3y = b_r.x;
    recip3z = b_r.y;
  } else if (permutation == Perm_cX_Y_Z) {
    // X, Y, Z
    nfft1 = pmeGrid.K1;
    nfft2 = pmeGrid.K2;
    nfft3 = pmeGrid.K3;
    prefac1 = d_bm1;
    prefac2 = d_bm2;
    prefac3 = d_bm3;
    recip1x = a_r.x;
    recip1y = a_r.y;
    recip1z = a_r.z;
    recip2x = b_r.x;
    recip2y = b_r.y;
    recip2z = b_r.z;
    recip3x = c_r.x;
    recip3y = c_r.y;
    recip3z = c_r.z;
  } else {
    NAMD_bug("CudaPmeKSpaceCompute::solve, invalid permutation");
  }

  // ncall++;
  // if (ncall == 1) {
  //   char filename[256];
  //   sprintf(filename,"dataf_%d_%d.txt",jblock,kblock);
  //   writeComplexToDisk((float2*)data, size1*size2*size3, filename, stream);
  // }

  // if (ncall == 1) {
  //   float2* h_data = new float2[size1*size2*size3];
  //   float2* d_data = (float2*)data;
  //   copy_DtoH<float2>(d_data, h_data, size1*size2*size3, stream);
  //   cudaCheck(cudaStreamSynchronize(stream));
  //   FILE *handle = fopen("dataf.txt", "w");
  //   for (int z=0;z < pmeGrid.K3;z++) {
  //     for (int y=0;y < pmeGrid.K2;y++) {
  //       for (int x=0;x < pmeGrid.K1/2+1;x++) {
  //         int i;
  //         if (permutation == Perm_cX_Y_Z) {
  //           i = x + y*size1 + z*size1*size2;
  //         } else {
  //           i = z + x*size1 + y*size1*size2;
  //         }
  //         fprintf(handle, "%f %f\n", h_data[i].x, h_data[i].y);
  //       }
  //     }
  //   }
  //   fclose(handle);
  //   delete [] h_data;
  // }

  // Clear energy and virial array if needed
  if (doEnergyVirial) clear_device_array<EnergyVirial>(d_energyVirial, 1, stream);

#ifdef TESTPID
  if (1) {
    cudaCheck(cudaStreamSynchronize(stream));
    fprintf(stderr, "AP calling scalar sum\n");
    fprintf(stderr, "(permutation == Perm_cX_Y_Z) = %s\n",
        (permutation == Perm_cX_Y_Z ? "true" : "false"));
    fprintf(stderr, "nfft1=%d  nfft2=%d  nfft3=%d\n", nfft1, nfft2, nfft3);
    fprintf(stderr, "size1=%d  size2=%d  size3=%d\n", size1, size2, size3);
    fprintf(stderr, "kappa=%g\n", kappa);
    fprintf(stderr, "recip1x=%g  recip1y=%g  recip1z=%g\n",
        (double)recip1x, (double)recip1y, (double)recip1z);
    fprintf(stderr, "recip2x=%g  recip2y=%g  recip2z=%g\n",
        (double)recip2x, (double)recip2y, (double)recip2z);
    fprintf(stderr, "recip3x=%g  recip3y=%g  recip3z=%g\n",
        (double)recip3x, (double)recip3y, (double)recip3z);
    fprintf(stderr, "volume=%g\n", volume);
    fprintf(stderr, "j0=%d  k0=%d\n", j0, k0);
    float *bm1, *bm2, *bm3;
    allocate_host<float>(&bm1, nfft1);
    allocate_host<float>(&bm2, nfft2);
    allocate_host<float>(&bm3, nfft3);
    copy_DtoH<float>(prefac1, bm1, nfft1, stream);
    copy_DtoH<float>(prefac2, bm2, nfft2, stream);
    copy_DtoH<float>(prefac3, bm3, nfft3, stream);
    TestArray_write<float>("bm1_good.bin", "structure factor bm1 good",
        bm1, nfft1);
    TestArray_write<float>("bm2_good.bin", "structure factor bm2 good",
        bm2, nfft2);
    TestArray_write<float>("bm3_good.bin", "structure factor bm3 good",
        bm3, nfft3);
    deallocate_host<float>(&bm1);
    deallocate_host<float>(&bm2);
    deallocate_host<float>(&bm3);
  }
#endif

  scalar_sum(permutation == Perm_cX_Y_Z, nfft1, nfft2, nfft3, size1, size2, size3, kappa,
    recip1x, recip1y, recip1z, recip2x, recip2y, recip2z, recip3x, recip3y, recip3z,
    volume, prefac1, prefac2, prefac3, j0, k0, doEnergyVirial,
    &d_energyVirial->energy, d_energyVirial->virial, (float2*)data, 
    stream);
#ifdef TESTPID
  if (1) {
    cudaCheck(cudaStreamSynchronize(stream));
    fprintf(stderr, "AP SCALAR SUM\n");
    fprintf(stderr, "COPY DEVICE ARRAYS BACK TO HOST\n");
    int m = 2 * (nfft1/2 + 1) * nfft2 * nfft3;
    float *tran = 0;
    allocate_host<float>(&tran, m);
    copy_DtoH<float>((float*)data, tran, m, stream);
    cudaCheck(cudaStreamSynchronize(stream));
    TestArray_write<float>("tran_potential_grid_good.bin",
          "transformed potential grid good", tran, m);
    deallocate_host<float>(&tran);
  }
#endif

  // Copy energy and virial to host if needed
  if (doEnergyVirial) {
    copy_DtoH<EnergyVirial>(d_energyVirial, h_energyVirial, 1, stream);
    cudaCheck(cudaEventRecord(copyEnergyVirialEvent, stream));
    // cudaCheck(cudaStreamSynchronize(stream));
  }

}

// void CudaPmeKSpaceCompute::waitEnergyAndVirial() {
//   cudaCheck(cudaSetDevice(deviceID));
//   cudaCheck(cudaEventSynchronize(copyEnergyVirialEvent));
// }

void CudaPmeKSpaceCompute::energyAndVirialCheck(void *arg, double walltime) {
  CudaPmeKSpaceCompute* c = (CudaPmeKSpaceCompute *)arg;

  cudaError_t err = cudaEventQuery(c->copyEnergyVirialEvent);
  if (err == cudaSuccess) {
    // Event has occurred
    c->checkCount = 0;
    if (c->pencilXYZPtr != NULL)
      c->pencilXYZPtr->energyAndVirialDone(c->multipleGridIndex);
    else if (c->pencilZPtr != NULL)
      c->pencilZPtr->energyAndVirialDone(c->multipleGridIndex);
    else
      NAMD_bug("CudaPmeKSpaceCompute::energyAndVirialCheck, pencilXYZPtr and pencilZPtr not set");
    return;
  } else if (err == cudaErrorNotReady) {
    // Event has not occurred
    c->checkCount++;
    if (c->checkCount >= 1000000) {
      char errmsg[256];
      sprintf(errmsg,"CudaPmeKSpaceCompute::energyAndVirialCheck polled %d times",
              c->checkCount);
      cudaDie(errmsg,err);
    }
  } else {
    // Anything else is an error
    char errmsg[256];
    sprintf(errmsg,"in CudaPmeKSpaceCompute::energyAndVirialCheck after polling %d times",
            c->checkCount);
    cudaDie(errmsg,err);
  }

  // Call again 
  CcdCallBacksReset(0, walltime);
  CcdCallFnAfter(energyAndVirialCheck, arg, 0.1);
}

void CudaPmeKSpaceCompute::energyAndVirialSetCallback(CudaPmePencilXYZ* pencilPtr) {
  cudaCheck(cudaSetDevice(deviceID));
  pencilXYZPtr = pencilPtr;
  pencilZPtr = NULL;
  checkCount = 0;
  CcdCallBacksReset(0, CmiWallTimer());
  // Set the call back at 0.1ms
  CcdCallFnAfter(energyAndVirialCheck, this, 0.1);
}

void CudaPmeKSpaceCompute::energyAndVirialSetCallback(CudaPmePencilZ* pencilPtr) {
  cudaCheck(cudaSetDevice(deviceID));
  pencilXYZPtr = NULL;
  pencilZPtr = pencilPtr;
  checkCount = 0;
  CcdCallBacksReset(0, CmiWallTimer());
  // Set the call back at 0.1ms
  CcdCallFnAfter(energyAndVirialCheck, this, 0.1);
}

double CudaPmeKSpaceCompute::getEnergy() {
  return h_energyVirial->energy;
}

void CudaPmeKSpaceCompute::getVirial(double *virial) {
  if (permutation == Perm_Z_cX_Y) {
    // h_energyVirial->virial is storing ZZ, ZX, ZY, XX, XY, YY
    virial[0] = h_energyVirial->virial[3];
    virial[1] = h_energyVirial->virial[4];
    virial[2] = h_energyVirial->virial[1];

    virial[3] = h_energyVirial->virial[4];
    virial[4] = h_energyVirial->virial[5];
    virial[5] = h_energyVirial->virial[2];

    virial[6] = h_energyVirial->virial[1];
    virial[7] = h_energyVirial->virial[7];
    virial[8] = h_energyVirial->virial[0];
  } else if (permutation == Perm_cX_Y_Z) {
    // h_energyVirial->virial is storing XX, XY, XZ, YY, YZ, ZZ
    virial[0] = h_energyVirial->virial[0];
    virial[1] = h_energyVirial->virial[1];
    virial[2] = h_energyVirial->virial[2];

    virial[3] = h_energyVirial->virial[1];
    virial[4] = h_energyVirial->virial[3];
    virial[5] = h_energyVirial->virial[4];

    virial[6] = h_energyVirial->virial[2];
    virial[7] = h_energyVirial->virial[4];
    virial[8] = h_energyVirial->virial[5];
  }
#if 0
  fprintf(stderr, "AP PME VIRIAL =\n"
      "  %g  %g  %g\n  %g  %g  %g\n  %g %g %g\n",
      virial[0], virial[1], virial[2], virial[3], virial[4],
      virial[5], virial[6], virial[7], virial[8]);
#endif
}


//###########################################################################
//###########################################################################
//###########################################################################

//
// Class constructor
//
CudaPmeRealSpaceCompute::CudaPmeRealSpaceCompute(PmeGrid pmeGrid,
  const int jblock, const int kblock, int deviceID, cudaStream_t stream) : 
  PmeRealSpaceCompute(pmeGrid, jblock, kblock), deviceID(deviceID), stream(stream) {
  if (dataSize < xsize*ysize*zsize)
    NAMD_bug("CudaPmeRealSpaceCompute::CudaPmeRealSpaceCompute, insufficient dataSize");
  cudaCheck(cudaSetDevice(deviceID));
  d_atomsCapacity = 0;
  d_atoms = NULL;
  d_forceCapacity = 0;
  d_force = NULL;
  #ifdef NAMD_CUDA
  tex_data = NULL;
  tex_data_len = 0;
  #else
  grid_data = NULL;
  grid_data_len = 0;
  #endif
  allocate_device<float>(&data, dataSize);
  setupGridData(data, xsize*ysize*zsize);
  cudaCheck(cudaEventCreate(&gatherForceEvent));
}

//
// Class desctructor
//
CudaPmeRealSpaceCompute::~CudaPmeRealSpaceCompute() {
  cudaCheck(cudaSetDevice(deviceID));
  if (d_atoms != NULL) deallocate_device<CudaAtom>(&d_atoms);
  if (d_force != NULL) deallocate_device<CudaForce>(&d_force);
  // if (d_patches != NULL) deallocate_device<PatchInfo>(&d_patches);
  // deallocate_device<double>(&d_selfEnergy);
  deallocate_device<float>(&data);
  cudaCheck(cudaEventDestroy(gatherForceEvent));
}

// //
// // Copy patches and atoms to device memory
// //
// void CudaPmeRealSpaceCompute::setPatchesAtoms(const int numPatches, const PatchInfo* patches,
//   const int numAtoms, const CudaAtom* atoms) {

//   this->numPatches = numPatches;
//   this->numAtoms = numAtoms;

//   // Reallocate device arrays as neccessary
//   reallocate_device<CudaAtom>(&d_atoms, &d_atomsCapacity, numAtoms, 1.5f);
//   reallocate_device<PatchInfo>(&d_patches, &d_patchesCapacity, numPatches, 1.5f);

//   // Copy atom and patch data to device
//   copy_HtoD<CudaAtom>(atoms, d_atoms, numAtoms, stream);
//   copy_HtoD<PatchInfo>(patches, d_patches, numPatches, stream);
// }

//
// Copy atoms to device memory
//
void CudaPmeRealSpaceCompute::copyAtoms(const int numAtoms, const CudaAtom* atoms) {
  cudaCheck(cudaSetDevice(deviceID));
  this->numAtoms = numAtoms;

  // Reallocate device arrays as neccessary
  reallocate_device<CudaAtom>(&d_atoms, &d_atomsCapacity, numAtoms, 1.5f);

  // Copy atom data to device
  copy_HtoD<CudaAtom>(atoms, d_atoms, numAtoms, stream);
}

//
// Spread charges on grid
//
void CudaPmeRealSpaceCompute::spreadCharge(Lattice &lattice) {
  cudaCheck(cudaSetDevice(deviceID));
#if 0
  if (1) {
    static int step = 0;
    float *xyzq;
    int natoms = numAtoms;
    allocate_host<float>(&xyzq, 4*natoms);
    copy_DtoH<float>((float *)d_atoms, xyzq, 4*natoms, stream);
    cudaCheck(cudaStreamSynchronize(stream));
    char fname[64], remark[64];
    sprintf(fname, "pme_atoms_xyzq_soa_%d.bin", step);
    sprintf(remark, "SOA PME atoms xyzq, step %d\n", step);
    TestArray_write<float>(fname, remark, xyzq, 4*natoms);
    deallocate_host<float>(&xyzq);
    step += 2;
  }
#endif

  NAMD_EVENT_START(1, NamdProfileEvent::SPREAD_CHARGE);

  // Clear grid
  clear_device_array<float>(data, xsize*ysize*zsize, stream);

#if defined(TESTPID)
  fprintf(stderr, "Calling spread_charge with parameters:\n");
  fprintf(stderr, "numAtoms = %d\n", numAtoms);
  fprintf(stderr, "pmeGrid.K1 = %d\n", pmeGrid.K1);
  fprintf(stderr, "pmeGrid.K2 = %d\n", pmeGrid.K2);
  fprintf(stderr, "pmeGrid.K3 = %d\n", pmeGrid.K3);
  fprintf(stderr, "xsize = %d\n", xsize);
  fprintf(stderr, "ysize = %d\n", ysize);
  fprintf(stderr, "zsize = %d\n", zsize);
  fprintf(stderr, "y0 = %d\n", y0);
  fprintf(stderr, "z0 = %d\n", z0);
  fprintf(stderr, "(pmeGrid.yBlocks == 1) = %d\n", (pmeGrid.yBlocks == 1));
  fprintf(stderr, "(pmeGrid.zBlocks == 1) = %d\n", (pmeGrid.zBlocks == 1));
  fprintf(stderr, "pmeGrid.order = %d\n", pmeGrid.order);
#endif
  spread_charge((const float4*)d_atoms, numAtoms,
    pmeGrid.K1, pmeGrid.K2, pmeGrid.K3, xsize, ysize, zsize,
    xsize, y0, z0, (pmeGrid.yBlocks == 1), (pmeGrid.zBlocks == 1),
    data, pmeGrid.order, stream);
#ifdef TESTPID
  if (1) {
    cudaCheck(cudaStreamSynchronize(stream));
    fprintf(stderr, "AP SPREAD CHARGES\n");
    fprintf(stderr, "COPY DEVICE ARRAYS BACK TO HOST\n");
    float *xyzq;
    allocate_host<float>(&xyzq, 4*numAtoms);
    copy_DtoH<float>((float *)d_atoms, xyzq, 4*numAtoms, stream);
    int gridlen = pmeGrid.K1 * pmeGrid.K2 * pmeGrid.K3;
    float *grid;
    allocate_host<float>(&grid, gridlen);
    copy_DtoH<float>(data, grid, gridlen, stream);
    cudaCheck(cudaStreamSynchronize(stream));
    TestArray_write<float>("xyzq_good.bin", "xyzq good", xyzq, 4*numAtoms);
    TestArray_write<float>("charge_grid_good.bin", "charge grid good",
        grid, gridlen);
    deallocate_host<float>(&xyzq);
    deallocate_host<float>(&grid);
  }
#endif

  // ncall++;

  // if (ncall == 1) writeRealToDisk(data, xsize*ysize*zsize, "data.txt");
  NAMD_EVENT_STOP(1, NamdProfileEvent::SPREAD_CHARGE);
}

void CudaPmeRealSpaceCompute::cuda_gatherforce_check(void *arg, double walltime) {
  CudaPmeRealSpaceCompute* c = (CudaPmeRealSpaceCompute *)arg;
  cudaCheck(cudaSetDevice(c->deviceID));

  cudaError_t err = cudaEventQuery(c->gatherForceEvent);
  if (err == cudaSuccess) {
    // Event has occurred
    c->checkCount = 0;
//    c->deviceProxy[CkMyNode()].gatherForceDone();
    c->devicePtr->gatherForceDone(c->multipleGridIndex);
    return;
  } else if (err == cudaErrorNotReady) {
    // Event has not occurred
    c->checkCount++;
    if (c->checkCount >= 1000000) {
      char errmsg[256];
      sprintf(errmsg,"CudaPmeRealSpaceCompute::cuda_gatherforce_check polled %d times",
              c->checkCount);
      cudaDie(errmsg,err);
    }
  } else {
    // Anything else is an error
    char errmsg[256];
    sprintf(errmsg,"in CudaPmeRealSpaceCompute::cuda_gatherforce_check after polling %d times",
            c->checkCount);
    cudaDie(errmsg,err);
  }

  // Call again 
  CcdCallBacksReset(0, walltime);
  CcdCallFnAfter(cuda_gatherforce_check, arg, 0.1);
}

void CudaPmeRealSpaceCompute::gatherForceSetCallback(ComputePmeCUDADevice* devicePtr_in) {
  cudaCheck(cudaSetDevice(deviceID));
  devicePtr = devicePtr_in;
  checkCount = 0;
  CcdCallBacksReset(0, CmiWallTimer());
  // Set the call back at 0.1ms
  CcdCallFnAfter(cuda_gatherforce_check, this, 0.1);
}

void CudaPmeRealSpaceCompute::waitGatherForceDone() {
  cudaCheck(cudaSetDevice(deviceID));
  cudaCheck(cudaEventSynchronize(gatherForceEvent));
}

void CudaPmeRealSpaceCompute::setupGridData(float* data, int data_len) {
  #ifdef NAMD_CUDA
    //HIP runtime error when using hipCreateTextureObject. No longer needed anyway, so we are moving in that direction now.  
  /*
  
  FATAL ERROR: CUDA error hipCreateTextureObject(&gridTexObj, &desc, &tdesc, NULL) in file src/CudaPmeSolverUtil.C, function setupGridTexture, line 744
 on Pe 11 (jparada-MS-7B09 device 0 pci 0:43:0): hipErrorRuntimeOther
------------- Processor 11 Exiting: Called CmiAbort ------------
Reason: FATAL ERROR: CUDA error hipCreateTextureObject(&gridTexObj, &desc, &tdesc, NULL) in file src/CudaPmeSolverUtil.C, function setupGridTexture, line 744
 on Pe 11 (jparada-MS-7B09 device 0 pci 0:43:0): hipErrorRuntimeOther

  */
  if (tex_data == data && tex_data_len == data_len) return;
  tex_data = data;
  tex_data_len = data_len;
  // Use texture objects
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = data;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = sizeof(float)*8;
  resDesc.res.linear.sizeInBytes = data_len*sizeof(float);
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  cudaCheck(cudaCreateTextureObject(&gridTexObj, &resDesc, &texDesc, NULL));
  #else
  if (grid_data == data && grid_data_len == data_len) return;
  grid_data = data;
  grid_data_len = data_len;
  #endif
}

void CudaPmeRealSpaceCompute::gatherForce(Lattice &lattice, CudaForce* force) {
  cudaCheck(cudaSetDevice(deviceID));

  NAMD_EVENT_START(1, NamdProfileEvent::GATHER_FORCE);

  // Re-allocate force array if needed
  reallocate_device<CudaForce>(&d_force, &d_forceCapacity, numAtoms, 1.5f);

#ifdef TESTPID
  if (1) {
    fprintf(stderr, "AP gather force arguments\n");
    fprintf(stderr, "numAtoms = %d\n", numAtoms);
    fprintf(stderr, "pmeGrid.K1 = %d\n", pmeGrid.K1);
    fprintf(stderr, "pmeGrid.K2 = %d\n", pmeGrid.K2);
    fprintf(stderr, "pmeGrid.K3 = %d\n", pmeGrid.K3);
    fprintf(stderr, "xsize = %d\n", xsize);
    fprintf(stderr, "ysize = %d\n", ysize);
    fprintf(stderr, "zsize = %d\n", zsize);
    fprintf(stderr, "y0 = %d\n", y0);
    fprintf(stderr, "z0 = %d\n", z0);
    fprintf(stderr, "(pmeGrid.yBlocks == 1) = %d\n", (pmeGrid.yBlocks == 1));
    fprintf(stderr, "(pmeGrid.zBlocks == 1) = %d\n", (pmeGrid.zBlocks == 1));
    fprintf(stderr, "pmeGrid.order = %d\n", pmeGrid.order);
    fprintf(stderr, "gridTexObj = %p\n", gridTexObj);
  }
#endif
  gather_force((const float4*)d_atoms, numAtoms,
    pmeGrid.K1, pmeGrid.K2, pmeGrid.K3,
    xsize, ysize, zsize, xsize, y0, z0, (pmeGrid.yBlocks == 1), (pmeGrid.zBlocks == 1),
    data, pmeGrid.order, (float3*)d_force, 
#ifdef NAMD_CUDA
    gridTexObj,
#endif
    stream);
#ifdef TESTPID
  if (1) {
    cudaCheck(cudaStreamSynchronize(stream));
    fprintf(stderr, "AP GATHER FORCE\n");
    fprintf(stderr, "COPY DEVICE ARRAYS BACK TO HOST\n");
    float *xyz;
    int natoms = numAtoms;
    allocate_host<float>(&xyz, 3*natoms);
    copy_DtoH<float>((float*)d_force, xyz, 3*natoms, stream);
    cudaCheck(cudaStreamSynchronize(stream));
    TestArray_write<float>("gather_force_good.bin",
        "gather force good", xyz, 3*natoms);
    deallocate_host<float>(&xyz);
  }
#endif

  copy_DtoH<CudaForce>(d_force, force, numAtoms, stream);

  cudaCheck(cudaEventRecord(gatherForceEvent, stream));

  NAMD_EVENT_STOP(1, NamdProfileEvent::GATHER_FORCE);
}

/*
double CudaPmeRealSpaceCompute::calcSelfEnergy() {
  double h_selfEnergy;
  clear_device_array<double>(d_selfEnergy, 1);
  calc_sum_charge_squared((const float4*)d_atoms, numAtoms, d_selfEnergy, stream);
  copy_DtoH<double>(d_selfEnergy, &h_selfEnergy, 1, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  // 1.7724538509055160273 = sqrt(pi)
  h_selfEnergy *= -ComputeNonbondedUtil::ewaldcof/1.7724538509055160273;
  return h_selfEnergy;
}
*/

//###########################################################################
//###########################################################################
//###########################################################################

CudaPmeTranspose::CudaPmeTranspose(PmeGrid pmeGrid, const int permutation,
    const int jblock, const int kblock, int deviceID, cudaStream_t stream) : 
  PmeTranspose(pmeGrid, permutation, jblock, kblock), deviceID(deviceID), stream(stream) {
  cudaCheck(cudaSetDevice(deviceID));

  allocate_device<float2>(&d_data, dataSize);
#ifndef P2P_ENABLE_3D
  allocate_device<float2>(&d_buffer, dataSize);
#endif

  // Setup data pointers to NULL, these can be overridden later on by using setDataPtrs()
  dataPtrsYZX.resize(nblock, NULL);
  dataPtrsZXY.resize(nblock, NULL);

  allocate_device< TransposeBatch<float2> >(&batchesYZX, 3*nblock);
  allocate_device< TransposeBatch<float2> >(&batchesZXY, 3*nblock);
}

CudaPmeTranspose::~CudaPmeTranspose() {
  cudaCheck(cudaSetDevice(deviceID));
  deallocate_device<float2>(&d_data);
#ifndef P2P_ENABLE_3D
  deallocate_device<float2>(&d_buffer);
#endif
  deallocate_device< TransposeBatch<float2> >(&batchesZXY);
  deallocate_device< TransposeBatch<float2> >(&batchesYZX);
}

//
// Set dataPtrsYZX
//
void CudaPmeTranspose::setDataPtrsYZX(std::vector<float2*>& dataPtrsNew, float2* data) {
  if (dataPtrsYZX.size() != dataPtrsNew.size())
    NAMD_bug("CudaPmeTranspose::setDataPtrsYZX, invalid dataPtrsNew size");
  for (int iblock=0;iblock < nblock;iblock++) {
    dataPtrsYZX[iblock] = dataPtrsNew[iblock];
  }
  // Build batched data structures
  TransposeBatch<float2> *h_batchesYZX = new TransposeBatch<float2>[3*nblock];

  for (int iperm=0;iperm < 3;iperm++) {
    int isize_out;
    if (iperm == 0) {
      // Perm_Z_cX_Y:
      // ZXY -> XYZ
      isize_out = pmeGrid.K1/2+1;
    } else if (iperm == 1) {
      // Perm_cX_Y_Z:
      // XYZ -> YZX
      isize_out = pmeGrid.K2;
    } else {
      // Perm_Y_Z_cX:
      // YZX -> ZXY
      isize_out = pmeGrid.K3;
    }

    int max_nx = 0;
    for (int iblock=0;iblock < nblock;iblock++) {

      int x0 = pos[iblock];
      int nx = pos[iblock+1] - x0;
      max_nx = std::max(max_nx, nx);

      int width_out;
      float2* data_out;
      if (dataPtrsYZX[iblock] == NULL) {
        // Local transpose, use internal buffer
        data_out = d_data + jsize*ksize*x0;
        width_out = jsize;
      } else {
        // Non-local tranpose, use buffer in dataPtr[] and the size of that buffer
        data_out = dataPtrsYZX[iblock];
        width_out = isize_out;
      }

      TransposeBatch<float2> batch;
      batch.nx        = nx;
      batch.ysize_out = width_out;
      batch.zsize_out = ksize;
      batch.data_in   = data+x0;
      batch.data_out  = data_out;

      h_batchesYZX[iperm*nblock + iblock] = batch;

    // transpose_xyz_yzx(
    //   nx, jsize, ksize,
    //   isize, jsize,
    //   width_out, ksize,
    //   data+x0, data_out, stream);
    }

    max_nx_YZX[iperm] = max_nx;
  }

  copy_HtoD< TransposeBatch<float2> >(h_batchesYZX, batchesYZX, 3*nblock, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  delete [] h_batchesYZX;
}

//
// Set dataPtrsZXY
//
void CudaPmeTranspose::setDataPtrsZXY(std::vector<float2*>& dataPtrsNew, float2* data) {
  if (dataPtrsZXY.size() != dataPtrsNew.size())
    NAMD_bug("CudaPmeTranspose::setDataPtrsZXY, invalid dataPtrsNew size");
  for (int iblock=0;iblock < nblock;iblock++) {
    dataPtrsZXY[iblock] = dataPtrsNew[iblock];
  }

  // Build batched data structures
  TransposeBatch<float2> *h_batchesZXY = new TransposeBatch<float2>[3*nblock];

  for (int iperm=0;iperm < 3;iperm++) {
    int isize_out;
    if (iperm == 0) {
      // Perm_cX_Y_Z:
      // XYZ -> ZXY
      isize_out = pmeGrid.K3;
    } else if (iperm == 1) {
      // Perm_Z_cX_Y:
      // ZXY -> YZX
      isize_out = pmeGrid.K2;
    } else {
      // Perm_Y_Z_cX:
      // YZX -> XYZ
      isize_out = pmeGrid.K1/2+1;
    }

    int max_nx = 0;
    for (int iblock=0;iblock < nblock;iblock++) {

      int x0 = pos[iblock];
      int nx = pos[iblock+1] - x0;
      max_nx = std::max(max_nx, nx);

      int width_out;
      float2* data_out;
      if (dataPtrsZXY[iblock] == NULL) {
        // Local transpose, use internal buffer
        data_out = d_data + jsize*ksize*x0;
        width_out = ksize;
      } else {
        // Non-local tranpose, use buffer in dataPtr[] and the size of that buffer
        data_out = dataPtrsZXY[iblock];
        width_out = isize_out;
      }

      TransposeBatch<float2> batch;
      batch.nx        = nx;
      batch.zsize_out = width_out;
      batch.xsize_out = nx;
      batch.data_in   = data+x0;
      batch.data_out  = data_out;
      h_batchesZXY[iperm*nblock + iblock] = batch;
    }

    max_nx_ZXY[iperm] = max_nx;
  }

  copy_HtoD< TransposeBatch<float2> >(h_batchesZXY, batchesZXY, 3*nblock, stream);
  cudaCheck(cudaStreamSynchronize(stream));
  delete [] h_batchesZXY;
}

void CudaPmeTranspose::transposeXYZtoYZX(const float2* data) {
  cudaCheck(cudaSetDevice(deviceID));

  int iperm;
  switch(permutation) {
    case Perm_Z_cX_Y:
    // ZXY -> XYZ
    iperm = 0;
    break;
    case Perm_cX_Y_Z:
    // XYZ -> YZX
    iperm = 1;
    break;
    case Perm_Y_Z_cX:
    // YZX -> ZXY
    iperm = 2;
    break;
    default:
    NAMD_bug("PmeTranspose::transposeXYZtoYZX, invalid permutation");
    break;
  }

  batchTranspose_xyz_yzx(
    nblock, batchesYZX + iperm*nblock,
    max_nx_YZX[iperm], jsize, ksize,
    isize, jsize, stream);


/*
  int isize_out;
  switch(permutation) {
    case Perm_Z_cX_Y:
    // ZXY -> XYZ
    isize_out = pmeGrid.K1/2+1;
    break;
    case Perm_cX_Y_Z:
    // XYZ -> YZX
    isize_out = pmeGrid.K2;
    break;
    case Perm_Y_Z_cX:
    // YZX -> ZXY
    isize_out = pmeGrid.K3;
    break;
    default:
    NAMD_bug("PmeTranspose::transposeXYZtoYZX, invalid permutation");
    break;
  }

  for (int iblock=0;iblock < nblock;iblock++) {

    int x0 = pos[iblock];
    int nx = pos[iblock+1] - x0;

    int width_out;
    float2* data_out;
    if (dataPtrsYZX[iblock] == NULL) {
      // Local transpose, use internal buffer
      data_out = d_data + jsize*ksize*x0;
      width_out = jsize;
    } else {
      // Non-local tranpose, use buffer in dataPtr[] and the size of that buffer
      data_out = dataPtrsYZX[iblock];
      width_out = isize_out;
    }

    transpose_xyz_yzx(
      nx, jsize, ksize,
      isize, jsize,
      width_out, ksize,
      data+x0, data_out, stream);
  }
*/
}

void CudaPmeTranspose::transposeXYZtoZXY(const float2* data) {
  cudaCheck(cudaSetDevice(deviceID));

  int iperm;
  switch(permutation) {
    case Perm_cX_Y_Z:
    // XYZ -> ZXY
    iperm = 0;
    break;
    case Perm_Z_cX_Y:
    // ZXY -> YZX
    iperm = 1;
    break;
    case Perm_Y_Z_cX:
    // YZX -> XYZ
    iperm = 2;
    break;
    default:
    NAMD_bug("PmeTranspose::transposeXYZtoZXY, invalid permutation");
    break;
  }

  batchTranspose_xyz_zxy(
    nblock, batchesZXY + iperm*nblock,
    max_nx_ZXY[iperm], jsize, ksize,
    isize, jsize, stream);

/*
  int isize_out;
  switch(permutation) {
    case Perm_cX_Y_Z:
    // XYZ -> ZXY
    isize_out = pmeGrid.K3;
    break;
    case Perm_Z_cX_Y:
    // ZXY -> YZX
    isize_out = pmeGrid.K2;
    break;
    case Perm_Y_Z_cX:
    // YZX -> XYZ
    isize_out = pmeGrid.K1/2+1;
    break;
    default:
    NAMD_bug("PmeTranspose::transposeXYZtoZXY, invalid permutation");
    break;
  }

  for (int iblock=0;iblock < nblock;iblock++) {

    int x0 = pos[iblock];
    int nx = pos[iblock+1] - x0;

    int width_out;
    float2* data_out;
    if (dataPtrsZXY[iblock] == NULL) {
      // Local transpose, use internal buffer
      data_out = d_data + jsize*ksize*x0;
      width_out = ksize;
    } else {
      // Non-local tranpose, use buffer in dataPtr[] and the size of that buffer
      data_out = dataPtrsZXY[iblock];
      width_out = isize_out;
    }

    transpose_xyz_zxy(
      nx, jsize, ksize,
      isize, jsize,
      width_out, nx,
      data+x0, data_out, stream);
  }
*/
}

void CudaPmeTranspose::waitStreamSynchronize() {
  cudaCheck(cudaSetDevice(deviceID));
  cudaCheck(cudaStreamSynchronize(stream));
}

void CudaPmeTranspose::copyDataDeviceToHost(const int iblock, float2* h_data, const int h_dataSize) {
  cudaCheck(cudaSetDevice(deviceID));

  if (iblock >= nblock)
    NAMD_bug("CudaPmeTranspose::copyDataDeviceToHost, block index exceeds number of blocks");

  int x0 = pos[iblock];
  int nx = pos[iblock+1] - x0;

  int copySize  = jsize*ksize*nx;
  int copyStart = jsize*ksize*x0;

  if (copyStart + copySize > dataSize)
    NAMD_bug("CudaPmeTranspose::copyDataDeviceToHost, dataSize exceeded");

  if (copySize > h_dataSize) 
    NAMD_bug("CudaPmeTranspose::copyDataDeviceToHost, h_dataSize exceeded");

  copy_DtoH<float2>(d_data+copyStart, h_data, copySize, stream);
}

void CudaPmeTranspose::copyDataHostToDevice(const int iblock, float2* data_in, float2* data_out) {
  cudaCheck(cudaSetDevice(deviceID));

  if (iblock >= nblock)
    NAMD_bug("CudaPmeTranspose::copyDataHostToDevice, block index exceeds number of blocks");

  // Determine block size = how much we're copying
  int i0, i1, j0, j1, k0, k1;
  getBlockDim(pmeGrid, permutation, iblock, jblock, kblock, i0, i1, j0, j1, k0, k1);
  int ni = i1-i0+1;
  int nj = j1-j0+1;
  int nk = k1-k0+1;

  copy3D_HtoD<float2>(data_in, data_out,
    0, 0, 0,
    ni, nj,
    i0, 0, 0,
    isize, jsize,
    ni, nj, nk, stream);
}

#ifndef P2P_ENABLE_3D
//
// Copy from temporary buffer to final buffer
//
void CudaPmeTranspose::copyDataDeviceToDevice(const int iblock, float2* data_out) {
  cudaCheck(cudaSetDevice(deviceID));

  if (iblock >= nblock)
    NAMD_bug("CudaPmeTranspose::copyDataDeviceToDevice, block index exceeds number of blocks");

  // Determine block size = how much we're copying
  int i0, i1, j0, j1, k0, k1;
  getBlockDim(pmeGrid, permutation, iblock, jblock, kblock, i0, i1, j0, j1, k0, k1);
  int ni = i1-i0+1;
  int nj = j1-j0+1;
  int nk = k1-k0+1;

  float2* data_in = d_buffer + i0*nj*nk;

  copy3D_DtoD<float2>(data_in, data_out,
    0, 0, 0,
    ni, nj,
    i0, 0, 0,
    isize, jsize,
    ni, nj, nk, stream);
}

//
// Return temporary buffer for block "iblock"
//
float2* CudaPmeTranspose::getBuffer(const int iblock) {
  if (iblock >= nblock)
    NAMD_bug("CudaPmeTranspose::getBuffer, block index exceeds number of blocks");

  // Determine block size = how much we're copying
  int i0, i1, j0, j1, k0, k1;
  getBlockDim(pmeGrid, permutation, iblock, jblock, kblock, i0, i1, j0, j1, k0, k1);
  int ni = i1-i0+1;
  int nj = j1-j0+1;
  int nk = k1-k0+1;

  return d_buffer + i0*nj*nk;
}
#endif

void CudaPmeTranspose::copyDataToPeerDeviceYZX(const int iblock, int deviceID_out, int permutation_out,
  float2* data_out) {

  int iblock_out = jblock;
  int jblock_out = kblock;
  int kblock_out = iblock;

  copyDataToPeerDevice(iblock, iblock_out, jblock_out, kblock_out, deviceID_out, permutation_out, data_out);
}

void CudaPmeTranspose::copyDataToPeerDeviceZXY(const int iblock, int deviceID_out, int permutation_out,
  float2* data_out) {

  int iblock_out = kblock;
  int jblock_out = iblock;
  int kblock_out = jblock;

  copyDataToPeerDevice(iblock, iblock_out, jblock_out, kblock_out, deviceID_out, permutation_out, data_out);
}

void CudaPmeTranspose::copyDataToPeerDevice(const int iblock,
  const int iblock_out, const int jblock_out, const int kblock_out,
  int deviceID_out, int permutation_out, float2* data_out) {

  cudaCheck(cudaSetDevice(deviceID));

  // Determine block size = how much we're copying
  int i0, i1, j0, j1, k0, k1;
  getBlockDim(pmeGrid, permutation_out, iblock_out, jblock_out, kblock_out, i0, i1, j0, j1, k0, k1);
  int ni = i1-i0+1;
  int nj = j1-j0+1;
  int nk = k1-k0+1;

  getPencilDim(pmeGrid, permutation_out, jblock_out, kblock_out, i0, i1, j0, j1, k0, k1);
  int isize_out = i1-i0+1;
  int jsize_out = j1-j0+1;

  int x0 = pos[iblock];
  float2* data_in = d_data + jsize*ksize*x0;

#ifndef P2P_ENABLE_3D
  // Copy into temporary peer device buffer
  copy_PeerDtoD<float2>(deviceID, deviceID_out, data_in, data_out, ni*nj*nk, stream);
#else
  copy3D_PeerDtoD<float2>(deviceID, deviceID_out, data_in, data_out,
    0, 0, 0,
    ni, nj,
    0, 0, 0,
    isize_out, jsize_out,
    ni, nj, nk, stream);
#endif

}


CudaPmeOneDevice::CudaPmeOneDevice(
    PmeGrid pmeGrid_,
    int deviceID_, 
    int deviceIndex_
    ) :
  pmeGrid(pmeGrid_), deviceID(deviceID_), deviceIndex(deviceIndex_),
  natoms(0), d_atoms(0), d_forces(0),
  d_grids(0), gridsize(0),
  d_trans(0), transize(0),
  d_bm1(0), d_bm2(0), d_bm3(0),
  kappa(ComputeNonbondedUtil::ewaldcof),
  self_energy_alch_first_time(true),
  force_scaling_alch_first_time(true),
  selfEnergy(0), selfEnergy_FEP(0), selfEnergy_TI_1(0), selfEnergy_TI_2(0), m_step(0)
{
//   fprintf(stderr, "CudaPmeOneDevice constructor START ******************************************\n");
  const SimParameters& sim_params = *(Node::Object()->simParameters);
  natoms = Node::Object()->molecule->numAtoms;
  // Determine how many grids we need for the alchemical route
  if (sim_params.alchOn) {
    num_used_grids = sim_params.alchGetNumOfPMEGrids();
  } else {
    num_used_grids = 1;
  }
  cudaCheck(cudaSetDevice(deviceID));

  // create our own CUDA stream
#if CUDA_VERSION >= 5050 || defined(NAMD_HIP) 
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  int leastPriority, greatestPriority;
  cudaCheck(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  cudaCheck(cudaStreamCreateWithPriority(&stream, cudaStreamDefault, greatestPriority));
#else
  cudaCheck(cudaStreamCreate(&stream));
#endif

  allocate_host<EnergyVirial>(&h_energyVirials, num_used_grids);
  allocate_device<EnergyVirial>(&d_energyVirials, num_used_grids);
  allocate_device<float>(&d_scaling_factors, num_used_grids);
  allocate_device<double>(&d_selfEnergy, 1);
  if (sim_params.alchFepOn) {
    allocate_device<double>(&d_selfEnergy_FEP, 1);
  } else {
    d_selfEnergy_FEP = NULL;
  }
  if (sim_params.alchThermIntOn) {
    allocate_device<double>(&d_selfEnergy_TI_1, 1);
    allocate_device<double>(&d_selfEnergy_TI_2, 1);
  } else {
    d_selfEnergy_TI_1 = NULL;
    d_selfEnergy_TI_2 = NULL;
  }

  // create device buffer space for atom positions and forces
  // to be accessed externally through PatchData
  allocate_device<float4>(&d_atoms, num_used_grids * natoms);
  allocate_device<float3>(&d_forces, num_used_grids * natoms);
  if (sim_params.alchOn) {
    allocate_device<int>(&d_partition, natoms);
  } else {
    d_partition = NULL;
  }
#ifdef NODEGROUP_FORCE_REGISTER
  DeviceData& devData = cpdata.ckLocalBranch()->devData[deviceIndex];
  devData.s_datoms = (CudaAtom *) (d_atoms);
  devData.f_slow = (CudaForce *) (d_forces);
  devData.f_slow_size = natoms;
  devData.s_datoms_partition = d_partition;
#endif
  int k1 = pmeGrid.K1;
  int k2 = pmeGrid.K2;
  int k3 = pmeGrid.K3;
  int order = pmeGrid.order;
  gridsize = k1 * k2 * k3;
  transize = (k1/2 + 1) * k2 * k3;

#if defined(NAMD_CUDA) || defined(NAMD_HIP)

  // set up cufft
  forwardPlans = new cufftHandle[num_used_grids];
  backwardPlans = new cufftHandle[num_used_grids];
  for (size_t iGrid = 0; iGrid < num_used_grids; ++iGrid) {
    cufftCheck(cufftPlan3d(&(forwardPlans[iGrid]), k3, k2, k1, CUFFT_R2C));
    cufftCheck(cufftPlan3d(&(backwardPlans[iGrid]), k3, k2, k1, CUFFT_C2R));
    cufftCheck(cufftSetStream(forwardPlans[iGrid], stream));
    cufftCheck(cufftSetStream(backwardPlans[iGrid], stream));
  }
#endif

#ifdef NAMD_CUDA 
  cudaDeviceProp deviceProp;
  cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceID));
  const int texture_alignment = int(deviceProp.textureAlignment);
  // d_grids and d_grids + N * gridsize will be used as device pointers for ::cudaResourceDesc::res::linear::devPtr
  // check if (d_grids + N * gridsize) is an address aligned to ::cudaDeviceProp::textureAlignment
  // which is required by cudaCreateTextureObject()
  // or maybe I should use cudaMallocPitch()?
  if ((gridsize % texture_alignment) != 0) {
    // if it is not aligned, padding is required
    gridsize = (int(gridsize / texture_alignment) + 1) * texture_alignment;
  }
  // Is it necesary to align transize too?
//   if ((transize % texture_alignment) != 0) {
//     // if it is not aligned, padding is required
//     transize = (int(transize / texture_alignment) + 1) * texture_alignment;
//   }
  allocate_device<float>(&d_grids, num_used_grids * gridsize);
  allocate_device<float2>(&d_trans, num_used_grids * transize);
  gridTexObjArrays = new cudaTextureObject_t[num_used_grids];
  for (size_t iGrid = 0; iGrid < num_used_grids; ++iGrid) {
    // set up texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = (void*)(d_grids + iGrid * (size_t)gridsize);
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = sizeof(float)*8;
    resDesc.res.linear.sizeInBytes = gridsize*sizeof(float);
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    cudaCheck(cudaCreateTextureObject(&(gridTexObjArrays[iGrid]), &resDesc, &texDesc, NULL));
  }
#else
  allocate_device<float>(&d_grids, num_used_grids * gridsize);
  allocate_device<float2>(&d_trans, num_used_grids * transize);
#endif
  // calculate prefactors
  double *bm1 = new double[k1];
  double *bm2 = new double[k2];
  double *bm3 = new double[k3];
  // Use compute_b_moduli from PmeKSpace.C
  extern void compute_b_moduli(double *bm, int k, int order);
  compute_b_moduli(bm1, k1, order);
  compute_b_moduli(bm2, k2, order);
  compute_b_moduli(bm3, k3, order);

  // allocate space for and copy prefactors onto GPU
  float *bm1f = new float[k1];
  float *bm2f = new float[k2];
  float *bm3f = new float[k3];
  for (int i=0;  i < k1;  i++)  bm1f[i] = (float) bm1[i];
  for (int i=0;  i < k2;  i++)  bm2f[i] = (float) bm2[i];
  for (int i=0;  i < k3;  i++)  bm3f[i] = (float) bm3[i];
  allocate_device<float>(&d_bm1, k1);
  allocate_device<float>(&d_bm2, k2);
  allocate_device<float>(&d_bm3, k3);
  copy_HtoD_sync<float>(bm1f, d_bm1, k1);
  copy_HtoD_sync<float>(bm2f, d_bm2, k2);
  copy_HtoD_sync<float>(bm3f, d_bm3, k3);
  delete [] bm1f;
  delete [] bm2f;
  delete [] bm3f;
  delete [] bm1;
  delete [] bm2;
  delete [] bm3;

  cudaCheck(cudaStreamSynchronize(stream));

//   fprintf(stderr, "CudaPmeOneDevice constructor END ********************************************\n");
}

CudaPmeOneDevice::~CudaPmeOneDevice() {
  deallocate_device<float4>(&d_atoms);
  deallocate_device<float3>(&d_forces);
  deallocate_device<float2>(&d_trans);
  deallocate_device<float>(&d_grids);
  deallocate_host<EnergyVirial>(&h_energyVirials);
  deallocate_device<EnergyVirial>(&d_energyVirials);
  deallocate_device<float>(&d_scaling_factors);
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
  for (size_t iGrid = 0; iGrid < num_used_grids; ++iGrid) {
    cufftCheck(cufftDestroy(forwardPlans[iGrid]));
    cufftCheck(cufftDestroy(backwardPlans[iGrid]));
#if defined(NAMD_CUDA) // only CUDA uses texture objects
    cudaCheck(cudaDestroyTextureObject(gridTexObjArrays[iGrid]));
#endif
  }

  delete[] forwardPlans;
  delete[] backwardPlans;
#if defined(NAMD_CUDA) // only CUDA uses texture objects
  delete[] gridTexObjArrays;
#endif


#endif
  deallocate_device<double>(&d_selfEnergy);
  if (d_partition != NULL) deallocate_device<int>(&d_partition);
  if (d_selfEnergy_FEP != NULL) deallocate_device<double>(&d_selfEnergy_FEP);
  if (d_selfEnergy_TI_1 != NULL) deallocate_device<double>(&d_selfEnergy_TI_1);
  if (d_selfEnergy_TI_2 != NULL) deallocate_device<double>(&d_selfEnergy_TI_2);
  deallocate_device<float>(&d_bm1);
  deallocate_device<float>(&d_bm2);
  deallocate_device<float>(&d_bm3);
  cudaCheck(cudaStreamDestroy(stream));
}

void CudaPmeOneDevice::compute(
    const Lattice &lattice,
//    const CudaAtom *d_atoms,
//    CudaForce *d_force,
//    int natoms,
    int doEnergyVirial,
    int step
#if 0
    double d_energy[1],
    double d_virial[6]
    NodeReduction& reduction
#endif
    ) {
//   fprintf(stderr, "CudaPmeOneDevice compute ****************************************************\n");
  int k1 = pmeGrid.K1;
  int k2 = pmeGrid.K2;
  int k3 = pmeGrid.K3;
  int order = pmeGrid.order;
  double volume = lattice.volume();
  Vector a_r = lattice.a_r();
  Vector b_r = lattice.b_r();
  Vector c_r = lattice.c_r();
  float arx = a_r.x;
  float ary = a_r.y;
  float arz = a_r.z;
  float brx = b_r.x;
  float bry = b_r.y;
  float brz = b_r.z;
  float crx = c_r.x;
  float cry = c_r.y;
  float crz = c_r.z;
  m_step = step;

  //JM:  actually necessary if you reserve a PME device!
  cudaCheck(cudaSetDevice(deviceID));
  const SimParameters& sim_params = *(Node::Object()->simParameters); 

  // clear force array
  //fprintf(stderr, "Calling clear_device_array on d_force\n");
  clear_device_array<float3>(d_forces, num_used_grids * natoms, stream);
  // clear grid
  //fprintf(stderr, "Calling clear_device_array on d_grid\n");
  clear_device_array<float>(d_grids, num_used_grids * gridsize, stream);
  clear_device_array<float2>(d_trans, num_used_grids * transize, stream);

  // Clear energy and virial array if needed
  if (doEnergyVirial) {
    // clear_device_array<EnergyVirial>(d_energyVirial, 1, stream);
    clear_device_array<EnergyVirial>(d_energyVirials, num_used_grids * 1, stream);
    const bool updateSelfEnergy = (step == sim_params.firstTimestep) || (selfEnergy == 0);
    if (updateSelfEnergy && (sim_params.alchOn == false)) {
      clear_device_array<double>(d_selfEnergy, 1, stream);
      // calculate self energy term if not yet done
      selfEnergy = compute_selfEnergy(d_selfEnergy, d_atoms, natoms,
          kappa, stream);
      //fprintf(stderr, "selfEnergy = %12.8f\n", selfEnergy);
    }
    /* the self energy depends on the scaling factor, or lambda
     * the cases when self energy will be changed:
     * 1. If alchLambdaFreq > 0, we will have a linear scaling of lambda. Lambda is changed EVERY STEP!
     * 2. In most cases, users will not use alchLambdaFreq > 0, but simulations may enter another lambda-window by using TCL scripts.
     * in summary, the self energy will be not changed unless lambda is changed.
     * so calcSelfEnergyAlch() would compare lambda of current step with the one from last step.
     * only if lambda is changed, the calcSelfEnergyFEPKernel or calcSelfEnergyTIKernel will be executed again.
     */
    if (sim_params.alchOn) calcSelfEnergyAlch(m_step);
  }

#if 0

  spread_charge(d_atoms, natoms, k1, k2, k3, k1, k2, k3,
      k1 /* xsize */, 0 /* jBlock */, 0 /* kBlock */,
      true /* pmeGrid.yBlocks == 1 */, true /* pmeGrid.zBlocks == 1 */,
      d_grid, order, stream);
#else
  const int order3 = ((order*order*order-1)/WARPSIZE + 1)*WARPSIZE;
  for (size_t iGrid = 0; iGrid < num_used_grids; ++iGrid) {
    spread_charge_v2(d_atoms + iGrid * natoms, natoms, k1, k2, k3, 
        float(k1), (float)k2, (float)k3, order3,
        k1, k2, k3,
        k1 /* xsize */, 0 /* jBlock */, 0 /* kBlock */,
        true /* pmeGrid.yBlocks == 1 */, true /* pmeGrid.zBlocks == 1 */,
        d_grids + iGrid * gridsize, order, stream);
  }

#endif
  //cudaCheck(cudaStreamSynchronize(stream));

  // forward FFT
  //fprintf(stderr, "Calling cufftExecR2C\n");
  //cufftCheck(cufftExecR2C(forwardPlan, (cufftReal *)d_grid,
  //      (cufftComplex *)d_tran));

  for (size_t iGrid = 0; iGrid < num_used_grids; ++iGrid) {
    cufftCheck(cufftExecR2C(forwardPlans[iGrid],
          (cufftReal *)(d_grids + iGrid * gridsize),
          (cufftComplex *)(d_trans + iGrid * transize)));
  }

  //cudaCheck(cudaStreamSynchronize(stream));

  // reciprocal space calculation
  //fprintf(stderr, "Calling scalar_sum\n");
  for (size_t iGrid = 0; iGrid < num_used_grids; ++iGrid) {
    scalar_sum(true /* Perm_cX_Y_Z */, k1, k2, k3, (k1/2 + 1), k2, k3,
        kappa, arx, ary, arz, brx, bry, brz, crx, cry, crz, volume,
        d_bm1, d_bm2, d_bm3, 0 /* jBlock */, 0 /* kBlock */,
        (bool) doEnergyVirial, &(d_energyVirials[iGrid].energy),
        d_energyVirials[iGrid].virial, d_trans + iGrid * transize, stream);
  }
  //scalar_sum(true /* Perm_cX_Y_Z */, k1, k2, k3, (k1/2 + 1), k2, k3,
  //    kappa, arx, ary, arz, brx, bry, brz, crx, cry, crz, volume,
  //    d_bm1, d_bm2, d_bm3, 0 /* jBlock */, 0 /* kBlock */,
  //    (bool) doEnergyVirial, &(d_energyVirial->energy),
  //    d_energyVirial->virial, d_tran, stream);
  //cudaCheck(cudaStreamSynchronize(stream));

  // backward FFT
  //fprintf(stderr, "Calling cufftExecC2R\n");
  for (size_t iGrid = 0; iGrid < num_used_grids; ++iGrid) {
    cufftCheck(cufftExecC2R(backwardPlans[iGrid],
          (cufftComplex *)(d_trans + iGrid * transize),
          (cufftReal *)(d_grids + iGrid * gridsize)));
  }

  //cufftCheck(cufftExecC2R(backwardPlan, (cufftComplex *)d_tran,
  //      (cufftReal *)d_grid));
  //cudaCheck(cudaStreamSynchronize(stream));

  // gather force from grid to atoms
  // missing cudaTextureObject_t below works for __CUDA_ARCH__ >= 350
  //fprintf(stderr, "Calling gather_force\n");
  for (unsigned int iGrid = 0; iGrid < num_used_grids; ++iGrid) {
      gather_force(&(d_atoms[iGrid * natoms]), natoms, k1, k2, k3, k1, k2, k3,
        k1 /* xsize */, 0 /* jBlock */, 0 /* kBlock */,
        true /* pmeGrid.yBlocks == 1 */, true /* pmeGrid.zBlocks == 1 */,
        d_grids + iGrid * gridsize, order, d_forces + iGrid * natoms,
#ifdef NAMD_CUDA
        gridTexObjArrays[iGrid] /* cudaTextureObject_t */,
#endif
        stream);
  }

  //gather_force(d_atoms, natoms, k1, k2, k3, k1, k2, k3,
  //    k1 /* xsize */, 0 /* jBlock */, 0 /* kBlock */,
  //    true /* pmeGrid.yBlocks == 1 */, true /* pmeGrid.zBlocks == 1 */,
  //    d_grid, order, d_force, gridTexObj /* cudaTextureObject_t */,
  //    stream);
  //cudaCheck(cudaStreamSynchronize(stream));

  // Copy energy and virial to host if needed
  if (doEnergyVirial) {
    //fprintf(stderr, "Calling copy_DtoH on d_energyVirial\n");
    copy_DtoH<EnergyVirial>(d_energyVirials, h_energyVirials,
        num_used_grids, stream);
    //cudaCheck(cudaEventRecord(copyEnergyVirialEvent, stream));
    //cudaCheck(cudaStreamSynchronize(stream));
  }

  // XXX debugging, quick test for borked forces
  //clear_device_array<float3>(d_force, natoms, stream);
  if (sim_params.alchOn) {
    scaleAndMergeForce(m_step);
  }
}


// call this after device-host memory transfer has completed
void CudaPmeOneDevice::finishReduction(
    bool doEnergyVirial
    ) {
  cudaCheck(cudaStreamSynchronize(stream));
  if(doEnergyVirial){
    CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
    PatchData* patchData = cpdata.ckLocalBranch();
    NodeReduction *reduction = patchData->reduction;
    cudaCheck(cudaSetDevice(deviceID));
    double virial[9];
    double energy, energy_F, energy_TI_1, energy_TI_2;
    const SimParameters& sim_params = *(Node::Object()->simParameters);
    if (sim_params.alchOn) {
      if (sim_params.alchFepOn) {
        scaleAndComputeFEPEnergyVirials(h_energyVirials, m_step, energy, energy_F, virial);
        energy += selfEnergy;
        energy_F += selfEnergy_FEP;
      }
      if (sim_params.alchThermIntOn) {
        scaleAndComputeTIEnergyVirials(h_energyVirials, m_step, energy, energy_TI_1, energy_TI_2, virial);
        energy += selfEnergy;
        energy_TI_1 += selfEnergy_TI_1;
        energy_TI_2 += selfEnergy_TI_2;
      }
    } else {
      virial[0] = h_energyVirials[0].virial[0];
      virial[1] = h_energyVirials[0].virial[1];
      virial[2] = h_energyVirials[0].virial[2];
      virial[3] = h_energyVirials[0].virial[1];
      virial[4] = h_energyVirials[0].virial[3];
      virial[5] = h_energyVirials[0].virial[4];
      virial[6] = h_energyVirials[0].virial[2];
      virial[7] = h_energyVirials[0].virial[4];
      virial[8] = h_energyVirials[0].virial[5];
      energy = h_energyVirials[0].energy + selfEnergy;
    }
  #if 0
    fprintf(stderr, "PME ENERGY = %g %g\n", h_energyVirials[0].energy, selfEnergy );
    fprintf(stderr, "PME VIRIAL =\n"
        "  %g  %g  %g\n  %g  %g  %g\n  %g %g %g\n",
        virial[0], virial[1], virial[2], virial[3], virial[4],
        virial[5], virial[6], virial[7], virial[8]);
  #endif
    (*reduction)[REDUCTION_VIRIAL_SLOW_XX] += virial[0];
    (*reduction)[REDUCTION_VIRIAL_SLOW_XY] += virial[1];
    (*reduction)[REDUCTION_VIRIAL_SLOW_XZ] += virial[2];
    (*reduction)[REDUCTION_VIRIAL_SLOW_YX] += virial[3];
    (*reduction)[REDUCTION_VIRIAL_SLOW_YY] += virial[4];
    (*reduction)[REDUCTION_VIRIAL_SLOW_YZ] += virial[5];
    (*reduction)[REDUCTION_VIRIAL_SLOW_ZX] += virial[6];
    (*reduction)[REDUCTION_VIRIAL_SLOW_ZY] += virial[7];
    (*reduction)[REDUCTION_VIRIAL_SLOW_ZZ] += virial[8];
    (*reduction)[REDUCTION_ELECT_ENERGY_SLOW] += energy;
    if (sim_params.alchFepOn) {
      (*reduction)[REDUCTION_ELECT_ENERGY_SLOW_F] += energy_F;
    }
    if (sim_params.alchThermIntOn) {
      (*reduction)[REDUCTION_ELECT_ENERGY_SLOW_TI_1] += energy_TI_1;
      (*reduction)[REDUCTION_ELECT_ENERGY_SLOW_TI_2] += energy_TI_2;
    }
  }
}

void CudaPmeOneDevice::calcSelfEnergyAlch(int step) {
  SimParameters& sim_params = *(Node::Object()->simParameters);
  if (sim_params.alchFepOn) {
    const BigReal alchLambda1 = sim_params.getCurrentLambda(step);
    const BigReal alchLambda2 = sim_params.getCurrentLambda2(step);
    static BigReal lambda1Up   = sim_params.getElecLambda(alchLambda1);
    static BigReal lambda2Up   = sim_params.getElecLambda(alchLambda2);
    static BigReal lambda1Down = sim_params.getElecLambda(1.0 - alchLambda1);
    static BigReal lambda2Down = sim_params.getElecLambda(1.0 - alchLambda2);
    // compute self energy at the first call
    // only compute self energy if factors are changed
    if ((lambda1Up != sim_params.getElecLambda(alchLambda1)) ||
        (lambda2Up != sim_params.getElecLambda(alchLambda2)) ||
        (lambda1Down != sim_params.getElecLambda(1.0 - alchLambda1)) ||
        (lambda2Down != sim_params.getElecLambda(1.0 - alchLambda2)) ||
         self_energy_alch_first_time) {
      lambda1Up   = sim_params.getElecLambda(alchLambda1);
      lambda2Up   = sim_params.getElecLambda(alchLambda2);
      lambda1Down = sim_params.getElecLambda(1.0 - alchLambda1);
      lambda2Down = sim_params.getElecLambda(1.0 - alchLambda2);
      selfEnergy = 0.0; // self energy for ÃƒÅ½Ã‚Â»_1
      selfEnergy_FEP = 0.0; // self energy for ÃƒÅ½Ã‚Â»_2
      cudaCheck(cudaMemsetAsync(d_selfEnergy, 0, sizeof(double), stream));
      cudaCheck(cudaMemsetAsync(d_selfEnergy_FEP, 0, sizeof(double), stream));
      calcSelfEnergyFEPWrapper(d_selfEnergy, d_selfEnergy_FEP, selfEnergy, selfEnergy_FEP, d_atoms, d_partition, natoms, kappa, sim_params.alchDecouple, lambda1Up, lambda2Up, lambda1Down, lambda2Down, stream);
      if (self_energy_alch_first_time) self_energy_alch_first_time = false;
    }
  }
  if (sim_params.alchThermIntOn) {
    const BigReal alchLambda1 = sim_params.getCurrentLambda(step);
    static BigReal lambda1Up   = sim_params.getElecLambda(alchLambda1);
    static BigReal lambda1Down = sim_params.getElecLambda(1.0 - alchLambda1);
    if ((lambda1Up != sim_params.getElecLambda(alchLambda1)) ||
        (lambda1Down != sim_params.getElecLambda(1.0 - alchLambda1)) ||
        self_energy_alch_first_time) {
      lambda1Up   = sim_params.getElecLambda(alchLambda1);
      lambda1Down = sim_params.getElecLambda(1.0 - alchLambda1);
      selfEnergy = 0.0;
      selfEnergy_TI_1 = 0.0;
      selfEnergy_TI_2 = 0.0;
      cudaCheck(cudaMemsetAsync(d_selfEnergy, 0, sizeof(double), stream));
      cudaCheck(cudaMemsetAsync(d_selfEnergy_TI_1, 0, sizeof(double), stream));
      cudaCheck(cudaMemsetAsync(d_selfEnergy_TI_2, 0, sizeof(double), stream));
      calcSelfEnergyTIWrapper(d_selfEnergy, d_selfEnergy_TI_1, d_selfEnergy_TI_2, selfEnergy, selfEnergy_TI_1, selfEnergy_TI_2, d_atoms, d_partition, natoms, kappa, sim_params.alchDecouple, lambda1Up, lambda1Down, stream);
      if (self_energy_alch_first_time) self_energy_alch_first_time = false;
    }
  }
}

void CudaPmeOneDevice::scaleAndMergeForce(int step) {
  SimParameters& sim_params = *(Node::Object()->simParameters);
  const double alchLambda1   = sim_params.getCurrentLambda(step);
  static BigReal lambda1Up   = sim_params.getElecLambda(alchLambda1);
  static BigReal lambda1Down = sim_params.getElecLambda(1.0 - alchLambda1);
  if ((lambda1Up != sim_params.getElecLambda(alchLambda1)) ||
      (lambda1Down != sim_params.getElecLambda(1.0 - alchLambda1)) ||
      force_scaling_alch_first_time) {
    std::vector<float> scale_factors(num_used_grids);
    lambda1Up   = sim_params.getElecLambda(alchLambda1);
    lambda1Down = sim_params.getElecLambda(1.0 - alchLambda1);
    scale_factors[0] = lambda1Up;
    scale_factors[1] = lambda1Down;
    if (sim_params.alchDecouple) {
      scale_factors[2] = 1.0 - lambda1Up;
      scale_factors[3] = 1.0 - lambda1Down;
    }
    if (bool(sim_params.alchElecLambdaStart) || sim_params.alchThermIntOn) {
      scale_factors[num_used_grids-1] = (lambda1Up + lambda1Down - 1.0) * (-1.0);
    }
    copy_HtoD<float>(scale_factors.data(), d_scaling_factors, num_used_grids);
    if (force_scaling_alch_first_time) force_scaling_alch_first_time = false;
  }
  scaleAndMergeForceWrapper(d_forces, d_scaling_factors, num_used_grids, natoms, stream);
}

void CudaPmeOneDevice::scaleAndComputeFEPEnergyVirials(const EnergyVirial* energyVirials, int step, double& energy, double& energy_F, double (&virial)[9]) {
  double scale1 = 1.0;
  double scale2 = 1.0;
  energy = 0;
  energy_F = 0;
  for (unsigned int i = 0; i < 9; ++i) {
    virial[i] = 0;
  }
  SimParameters& sim_params = *(Node::Object()->simParameters);
  const BigReal alchLambda  = sim_params.getCurrentLambda(step);
  const BigReal alchLambda2 = sim_params.getCurrentLambda2(step);
  const BigReal elecLambdaUp  = sim_params.getElecLambda(alchLambda);
  const BigReal elecLambda2Up = sim_params.getElecLambda(alchLambda2);
  const BigReal elecLambdaDown  = sim_params.getElecLambda(1 - alchLambda);
  const BigReal elecLambda2Down = sim_params.getElecLambda(1 - alchLambda2);
  energy   += energyVirials[0].energy * elecLambdaUp;
  energy_F += energyVirials[0].energy * elecLambda2Up;
  energy   += energyVirials[1].energy * elecLambdaDown;
  energy_F += energyVirials[1].energy * elecLambda2Down;
  virial[0] += energyVirials[0].virial[0] * elecLambdaUp;
  virial[1] += energyVirials[0].virial[1] * elecLambdaUp;
  virial[2] += energyVirials[0].virial[2] * elecLambdaUp;
  virial[3] += energyVirials[0].virial[1] * elecLambdaUp;
  virial[4] += energyVirials[0].virial[3] * elecLambdaUp;
  virial[5] += energyVirials[0].virial[4] * elecLambdaUp;
  virial[6] += energyVirials[0].virial[2] * elecLambdaUp;
  virial[7] += energyVirials[0].virial[4] * elecLambdaUp;
  virial[8] += energyVirials[0].virial[5] * elecLambdaUp;
  virial[0] += energyVirials[1].virial[0] * elecLambdaDown;
  virial[1] += energyVirials[1].virial[1] * elecLambdaDown;
  virial[2] += energyVirials[1].virial[2] * elecLambdaDown;
  virial[3] += energyVirials[1].virial[1] * elecLambdaDown;
  virial[4] += energyVirials[1].virial[3] * elecLambdaDown;
  virial[5] += energyVirials[1].virial[4] * elecLambdaDown;
  virial[6] += energyVirials[1].virial[2] * elecLambdaDown;
  virial[7] += energyVirials[1].virial[4] * elecLambdaDown;
  virial[8] += energyVirials[1].virial[5] * elecLambdaDown;
  if (sim_params.alchDecouple) {
    energy   += energyVirials[2].energy * (1.0 - elecLambdaUp);
    energy_F += energyVirials[2].energy * (1.0 - elecLambda2Up);
    energy   += energyVirials[3].energy * (1.0 - elecLambdaDown);
    energy_F += energyVirials[3].energy * (1.0 - elecLambda2Down);
    virial[0] += energyVirials[2].virial[0] * (1.0 - elecLambdaUp);
    virial[1] += energyVirials[2].virial[1] * (1.0 - elecLambdaUp);
    virial[2] += energyVirials[2].virial[2] * (1.0 - elecLambdaUp);
    virial[3] += energyVirials[2].virial[1] * (1.0 - elecLambdaUp);
    virial[4] += energyVirials[2].virial[3] * (1.0 - elecLambdaUp);
    virial[5] += energyVirials[2].virial[4] * (1.0 - elecLambdaUp);
    virial[6] += energyVirials[2].virial[2] * (1.0 - elecLambdaUp);
    virial[7] += energyVirials[2].virial[4] * (1.0 - elecLambdaUp);
    virial[8] += energyVirials[2].virial[5] * (1.0 - elecLambdaUp);
    virial[0] += energyVirials[3].virial[0] * (1.0 - elecLambdaDown);
    virial[1] += energyVirials[3].virial[1] * (1.0 - elecLambdaDown);
    virial[2] += energyVirials[3].virial[2] * (1.0 - elecLambdaDown);
    virial[3] += energyVirials[3].virial[1] * (1.0 - elecLambdaDown);
    virial[4] += energyVirials[3].virial[3] * (1.0 - elecLambdaDown);
    virial[5] += energyVirials[3].virial[4] * (1.0 - elecLambdaDown);
    virial[6] += energyVirials[3].virial[2] * (1.0 - elecLambdaDown);
    virial[7] += energyVirials[3].virial[4] * (1.0 - elecLambdaDown);
    virial[8] += energyVirials[3].virial[5] * (1.0 - elecLambdaDown);
    if (sim_params.alchElecLambdaStart > 0) {
      energy   += energyVirials[4].energy * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      energy_F += energyVirials[4].energy * (-1.0 * (elecLambda2Up + elecLambda2Down - 1.0));
      virial[0] += energyVirials[4].virial[0] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[1] += energyVirials[4].virial[1] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[2] += energyVirials[4].virial[2] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[3] += energyVirials[4].virial[1] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[4] += energyVirials[4].virial[3] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[5] += energyVirials[4].virial[4] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[6] += energyVirials[4].virial[2] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[7] += energyVirials[4].virial[4] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[8] += energyVirials[4].virial[5] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    }
  } else {
    if (sim_params.alchElecLambdaStart > 0) {
      energy   += energyVirials[2].energy * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      energy_F += energyVirials[2].energy * (-1.0 * (elecLambda2Up + elecLambda2Down - 1.0));
      virial[0] += energyVirials[2].virial[0] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[1] += energyVirials[2].virial[1] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[2] += energyVirials[2].virial[2] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[3] += energyVirials[2].virial[1] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[4] += energyVirials[2].virial[3] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[5] += energyVirials[2].virial[4] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[6] += energyVirials[2].virial[2] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[7] += energyVirials[2].virial[4] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
      virial[8] += energyVirials[2].virial[5] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    }
  }
}

void CudaPmeOneDevice::scaleAndComputeTIEnergyVirials(const EnergyVirial* energyVirials, int step, double& energy, double& energy_TI_1, double& energy_TI_2, double (&virial)[9]) {
  double scale1 = 1.0;
  energy =0;
  energy_TI_1 = 0;
  energy_TI_2 = 0;
  for (unsigned int i = 0; i < 9; ++i) {
    virial[i] = 0;
  }
  SimParameters& sim_params = *(Node::Object()->simParameters);
  const BigReal alchLambda   = sim_params.getCurrentLambda(step);
  const BigReal elecLambdaUp = sim_params.getElecLambda(alchLambda);
  const BigReal elecLambdaDown = sim_params.getElecLambda(1 - alchLambda);
  energy      += energyVirials[0].energy * elecLambdaUp;
  energy      += energyVirials[1].energy * elecLambdaDown;
  energy_TI_1 += energyVirials[0].energy;
  energy_TI_2 += energyVirials[1].energy;
  virial[0] += energyVirials[0].virial[0] * elecLambdaUp;
  virial[1] += energyVirials[0].virial[1] * elecLambdaUp;
  virial[2] += energyVirials[0].virial[2] * elecLambdaUp;
  virial[3] += energyVirials[0].virial[1] * elecLambdaUp;
  virial[4] += energyVirials[0].virial[3] * elecLambdaUp;
  virial[5] += energyVirials[0].virial[4] * elecLambdaUp;
  virial[6] += energyVirials[0].virial[2] * elecLambdaUp;
  virial[7] += energyVirials[0].virial[4] * elecLambdaUp;
  virial[8] += energyVirials[0].virial[5] * elecLambdaUp;
  virial[0] += energyVirials[1].virial[0] * elecLambdaDown;
  virial[1] += energyVirials[1].virial[1] * elecLambdaDown;
  virial[2] += energyVirials[1].virial[2] * elecLambdaDown;
  virial[3] += energyVirials[1].virial[1] * elecLambdaDown;
  virial[4] += energyVirials[1].virial[3] * elecLambdaDown;
  virial[5] += energyVirials[1].virial[4] * elecLambdaDown;
  virial[6] += energyVirials[1].virial[2] * elecLambdaDown;
  virial[7] += energyVirials[1].virial[4] * elecLambdaDown;
  virial[8] += energyVirials[1].virial[5] * elecLambdaDown;
  if (sim_params.alchDecouple) {
    energy      += energyVirials[2].energy * (1.0 - elecLambdaUp);
    energy      += energyVirials[3].energy * (1.0 - elecLambdaDown);
    energy_TI_1 += -1.0 * energyVirials[2].energy;
    energy_TI_2 += -1.0 * energyVirials[3].energy;
    virial[0] += energyVirials[2].virial[0] * (1.0 - elecLambdaUp);
    virial[1] += energyVirials[2].virial[1] * (1.0 - elecLambdaUp);
    virial[2] += energyVirials[2].virial[2] * (1.0 - elecLambdaUp);
    virial[3] += energyVirials[2].virial[1] * (1.0 - elecLambdaUp);
    virial[4] += energyVirials[2].virial[3] * (1.0 - elecLambdaUp);
    virial[5] += energyVirials[2].virial[4] * (1.0 - elecLambdaUp);
    virial[6] += energyVirials[2].virial[2] * (1.0 - elecLambdaUp);
    virial[7] += energyVirials[2].virial[4] * (1.0 - elecLambdaUp);
    virial[8] += energyVirials[2].virial[5] * (1.0 - elecLambdaUp);
    virial[0] += energyVirials[3].virial[0] * (1.0 - elecLambdaDown);
    virial[1] += energyVirials[3].virial[1] * (1.0 - elecLambdaDown);
    virial[2] += energyVirials[3].virial[2] * (1.0 - elecLambdaDown);
    virial[3] += energyVirials[3].virial[1] * (1.0 - elecLambdaDown);
    virial[4] += energyVirials[3].virial[3] * (1.0 - elecLambdaDown);
    virial[5] += energyVirials[3].virial[4] * (1.0 - elecLambdaDown);
    virial[6] += energyVirials[3].virial[2] * (1.0 - elecLambdaDown);
    virial[7] += energyVirials[3].virial[4] * (1.0 - elecLambdaDown);
    virial[8] += energyVirials[3].virial[5] * (1.0 - elecLambdaDown);
    energy      += energyVirials[4].energy * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    energy_TI_1 += -1.0 * energyVirials[4].energy;
    energy_TI_2 += -1.0 * energyVirials[4].energy;
    virial[0] += energyVirials[4].virial[0] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[1] += energyVirials[4].virial[1] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[2] += energyVirials[4].virial[2] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[3] += energyVirials[4].virial[1] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[4] += energyVirials[4].virial[3] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[5] += energyVirials[4].virial[4] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[6] += energyVirials[4].virial[2] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[7] += energyVirials[4].virial[4] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[8] += energyVirials[4].virial[5] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
  } else {
    energy      += energyVirials[2].energy * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    energy_TI_1 += -1.0 * energyVirials[2].energy;
    energy_TI_2 += -1.0 * energyVirials[2].energy;
    virial[0] += energyVirials[2].virial[0] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[1] += energyVirials[2].virial[1] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[2] += energyVirials[2].virial[2] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[3] += energyVirials[2].virial[1] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[4] += energyVirials[2].virial[3] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[5] += energyVirials[2].virial[4] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[6] += energyVirials[2].virial[2] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[7] += energyVirials[2].virial[4] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
    virial[8] += energyVirials[2].virial[5] * (-1.0 * (elecLambdaUp + elecLambdaDown - 1.0));
  }
}

#endif // NAMD_CUDA

