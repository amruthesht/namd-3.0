#ifndef COMPUTECUDAMGR_H
#define COMPUTECUDAMGR_H
#include <vector>
#include "CudaUtils.h"
#include "ComputeCUDAMgr.decl.h"
#include "CudaNonbondedTables.h"
#include "CudaComputeNonbonded.h"
#include "CudaPmeSolverUtil.h"
#include "ComputeBondedCUDA.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)

class ComputeCUDAMgr : public CBase_ComputeCUDAMgr {
public:
  // ComputeCUDAMgr_SDAG_CODE;
  ComputeCUDAMgr();
  ComputeCUDAMgr(CkMigrateMessage *);
  ~ComputeCUDAMgr();
  void initialize(CkQdMsg *msg);
  void initialize_devices(CkQdMsg *msg);
  void update();
  static ComputeCUDAMgr* getComputeCUDAMgr();
  CudaComputeNonbonded* createCudaComputeNonbonded(ComputeID c);
  CudaComputeNonbonded* getCudaComputeNonbonded();
#ifdef BONDED_CUDA
  ComputeBondedCUDA* createComputeBondedCUDA(ComputeID c, ComputeMgr* computeMgr);
  ComputeBondedCUDA* getComputeBondedCUDA();
#endif
  CudaPmeOneDevice* createCudaPmeOneDevice();
  CudaPmeOneDevice* getCudaPmeOneDevice();
private:

  // Number of CUDA devices on this node that are used in computation
  int numDevices;
  std::vector<CudaNonbondedTables*> cudaNonbondedTablesList;
  std::vector<CudaComputeNonbonded*> cudaComputeNonbondedList;
#ifdef BONDED_CUDA
  std::vector<ComputeBondedCUDA*> computeBondedCUDAList;
#endif
  CudaPmeOneDevice* cudaPmeOneDevice;
};

#else // NAMD_CUDA

class ComputeCUDAMgr : public CBase_ComputeCUDAMgr {
};

#endif // NAMD_CUDA
#endif // COMPUTECUDAMGR_H

