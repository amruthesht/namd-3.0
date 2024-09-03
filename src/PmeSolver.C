#include <stdio.h>
#include <iomanip>
#include "Priorities.h"
#if !(defined(NAMD_HIP) || defined(NAMD_CUDA))
struct float2 {float x,y;};
#endif
#include "PmeSolver.h"
#include "PatchData.h"

//
// Data flow for PmePencilXYZ
//
// dataSrc [xyz]   dataDst
//
// dataDst [solve] dataDst
//
// dataDst [xyz]   dataSrc
//
// dataSrc [force]
//

PmePencilXYZ::PmePencilXYZ() {
  __sdag_init();
  setMigratable(false);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    pmeKSpaceComputes[iGrid] = NULL;
    fftComputes[iGrid] = NULL;
  }
  reduction = ReductionMgr::Object()->willSubmit(REDUCTIONS_BASIC);
#ifdef NODEGROUP_FORCE_REGISTER
// #if false
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();
  nodeReduction = patchData->reduction;
#endif
//   multipleGridLock = CmiCreateLock();
  doEnergy = false;
  doVirial = false;
  simulationStep = 0;
}

PmePencilXYZ::PmePencilXYZ(CkMigrateMessage *m) {
  NAMD_bug("PmePencilXYZ cannot be migrated");
  //__sdag_init();
  // setMigratable(false);
  // fftCompute = NULL;
  // pmeKSpaceCompute = NULL;
  // reduction = ReductionMgr::Object()->willSubmit(REDUCTIONS_BASIC);
}

PmePencilXYZ::~PmePencilXYZ() {
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeKSpaceComputes[iGrid] != NULL) delete pmeKSpaceComputes[iGrid];
    if (fftComputes[iGrid] != NULL) delete fftComputes[iGrid];
  }
  delete reduction;
//   CmiDestroyLock(multipleGridLock);
}

void PmePencilXYZ::initFFT(PmeStartMsg *msg) {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilXYZ::initFFT, fftCompute not initialized");
//   fftCompute->init(msg->dataGrid[0], msg->dataSizes[0], NULL, 0, Perm_X_Y_Z, pmeGrid, 3, 0, 0, 0);
//   fftCompute2->init(msg->dataGrid[1], msg->dataSizes[1], NULL, 0, Perm_X_Y_Z, pmeGrid, 3, 0, 0, 0);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (msg->enabledGrid[iGrid] == true) {
      fftComputes[iGrid]->init(msg->dataGrid[iGrid], msg->dataSizes[iGrid], NULL, 0, Perm_X_Y_Z, pmeGrid, 3, 0, 0, 0);
    }
  }
}

void PmePencilXYZ::forwardFFT() {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilXYZ::forwardFFT, fftCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) fftComputes[iGrid]->forward();
  }
}

void PmePencilXYZ::backwardFFT() {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilXYZ::backwardFFT, fftCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) fftComputes[iGrid]->backward();
  }
}

void PmePencilXYZ::forwardDone() {
  if (pmeKSpaceComputes[0] == NULL)
    NAMD_bug("PmePencilXYZ::forwardDone, pmeKSpaceCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeKSpaceComputes[iGrid] != NULL && fftComputes[iGrid] != NULL)
      pmeKSpaceComputes[iGrid]->solve(lattice, doEnergy, doVirial, fftComputes[iGrid]->getDataDst());
  }
}

void PmePencilXYZ::backwardDone() {
  NAMD_bug("PmePencilXYZ::backwardDone(), base class method called");
}

void PmePencilXYZ::submitReductions(unsigned int iGrid) {
//  fprintf(stderr, "PmePencilXYZ::submitReductions\n");
  if (pmeKSpaceComputes[iGrid] == NULL)
    NAMD_bug("PmePencilXYZ::submitReductions, pmeKSpaceCompute not initialized");
  SimParameters* simParams = Node::Object()->simParameters;
  double virial[9];
  double energy = pmeKSpaceComputes[iGrid]->getEnergy();
  pmeKSpaceComputes[iGrid]->getVirial(virial);
  if (simParams->alchOn) {
    if (simParams->alchFepOn) {
      double energy_F = energy;
      double scale1 = 1.0; // energy scaling factor for λ_1
      double scale2 = 1.0; // energy scaling factor for λ_2
      switch (iGrid) {
        case 0: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal alchLambda2 = simParams->getCurrentLambda2(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          const BigReal elecLambda2Up = simParams->getElecLambda(alchLambda2);
          scale1 = elecLambdaUp;
          scale2 = elecLambda2Up;
          break;
        }
        case 1: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal alchLambda2 = simParams->getCurrentLambda2(simulationStep);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          const BigReal elecLambda2Down = simParams->getElecLambda(1 - alchLambda2);
          scale1 = elecLambdaDown;
          scale2 = elecLambda2Down;
          break;
        }
        case 2: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal alchLambda2 = simParams->getCurrentLambda2(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          const BigReal elecLambda2Up = simParams->getElecLambda(alchLambda2);
          scale1 = 1.0 - elecLambdaUp;
          scale2 = 1.0 - elecLambda2Up;
          break;
        }
        case 3: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal alchLambda2 = simParams->getCurrentLambda2(simulationStep);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          const BigReal elecLambda2Down = simParams->getElecLambda(1 - alchLambda2);
          scale1 = 1.0 - elecLambdaDown;
          scale2 = 1.0 - elecLambda2Down;
          break;
        }
        case 4: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal alchLambda2 = simParams->getCurrentLambda2(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          const BigReal elecLambda2Up = simParams->getElecLambda(alchLambda2);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          const BigReal elecLambda2Down = simParams->getElecLambda(1 - alchLambda2);
          scale1 = -1.0 * (elecLambdaUp + elecLambdaDown - 1.0);
          scale2 = -1.0 * (elecLambda2Up + elecLambda2Down - 1.0);
          break;
        }
      }
      energy *= scale1;
      energy_F *= scale2;
//       fprintf(stdout, "KSpace Grid %u ; E1 = %lf ; E2 = %lf ; scale1 = %lf ; scale2 = %lf\n", iGrid, energy, energy_F, scale1, scale2);
      for (size_t i = 0; i < 9; ++i) {
        virial[i] *= scale1;
      }
#if NODEGROUP_FORCE_REGISTER
      nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_F) += energy_F;
#endif
      reduction->item(REDUCTION_ELECT_ENERGY_SLOW_F) += energy_F;
    }
    if (simParams->alchThermIntOn) {
      double energy_TI_1 = 0.0;
      double energy_TI_2 = 0.0;
      double scale1 = 1.0;
      switch (iGrid) {
        case 0: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          scale1 = elecLambdaUp;
          energy_TI_1 = energy;
          break;
        }
        case 1: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          scale1 = elecLambdaDown;
          energy_TI_2 = energy;
          break;
        }
        case 2: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          scale1 = 1.0 - elecLambdaUp;
          energy_TI_1 = -1.0 * energy;
          break;
        }
        case 3: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          scale1 = 1.0 - elecLambdaDown;
          energy_TI_2 = -1.0 * energy;
          break;
        }
        case 4: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          scale1 = -1.0 * (elecLambdaUp + elecLambdaDown - 1.0);
          energy_TI_1 = -1.0 * energy;
          energy_TI_2 = -1.0 * energy;
          break;
        }
      }
      for (size_t i = 0; i < 9; ++i) {
        virial[i] *= scale1;
      }
      energy *= scale1;
//       fprintf(stdout, "Grid %u : energy_TI_1 = %lf ; energy_TI_2 = %lf\n", iGrid, energy_TI_1, energy_TI_2);
#if NODEGROUP_FORCE_REGISTER
      nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1) += energy_TI_1;
      nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2) += energy_TI_2;
#endif
      reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1) += energy_TI_1;
      reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2) += energy_TI_2;
    }
  }
#ifdef NODEGROUP_FORCE_REGISTER
  // #if false
  // XXX Expect a race condition, need atomic access to nodeReduction
  nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW) += energy;
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_XX) += virial[0];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_XY) += virial[1];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_XZ) += virial[2];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_YX) += virial[3];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_YY) += virial[4];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_YZ) += virial[5];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_ZX) += virial[6];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_ZY) += virial[7];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_ZZ) += virial[8];
#endif
  reduction->item(REDUCTION_ELECT_ENERGY_SLOW) += energy;
  reduction->item(REDUCTION_VIRIAL_SLOW_XX) += virial[0];
  reduction->item(REDUCTION_VIRIAL_SLOW_XY) += virial[1];
  reduction->item(REDUCTION_VIRIAL_SLOW_XZ) += virial[2];
  reduction->item(REDUCTION_VIRIAL_SLOW_YX) += virial[3];
  reduction->item(REDUCTION_VIRIAL_SLOW_YY) += virial[4];
  reduction->item(REDUCTION_VIRIAL_SLOW_YZ) += virial[5];
  reduction->item(REDUCTION_VIRIAL_SLOW_ZX) += virial[6];
  reduction->item(REDUCTION_VIRIAL_SLOW_ZY) += virial[7];
  reduction->item(REDUCTION_VIRIAL_SLOW_ZZ) += virial[8];
  reduction->item(REDUCTION_STRAY_CHARGE_ERRORS) += numStrayAtoms;
  energyReady[iGrid] = 1;
//   CmiLock(multipleGridLock);
  bool ready_to_submit = true;
  for (size_t i = 0; i < NUM_GRID_MAX; ++i) {
    if (energyReady[i] == -1) continue;
    if (energyReady[i] == 0) ready_to_submit = false;
  }
  if (ready_to_submit) {
//     fprintf(stdout, "all energy ready\n");
    reduction->submit();
    for (size_t i = 0; i < NUM_GRID_MAX; ++i) {
      if (energyReady[i] == -1) continue;
      if (energyReady[i] == 1) energyReady[i] = 0;
    }
  }
//   CmiUnlock(multipleGridLock);
}

void PmePencilXYZ::skip() {
  reduction->submit();  
}

//###########################################################################
//###########################################################################
//###########################################################################

//
// Data flow for PmePencilXY & PmePencilZ
//
// dataSrc(XY) [xy]     dataDst(XY)
//
// dataDst(XY) [transp] dataSrc(Z)
//---------------------------------
//
// dataSrc(Z)  [z]      dataDst(Z)
//
// dataDst(Z)  [solve]  dataDst(Z)
//
// dataDst(Z)  [z]      dataSrc(Z)
//
// dataSrc(Z)  [transp] dataDst(XY)
//---------------------------------
//
// dataDst(XY) [xy]     dataSrc(XY)
//
// dataSrc(XY) [force]
//

PmePencilXY::PmePencilXY() {
  __sdag_init();
  setMigratable(false);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    pmeTransposes[iGrid] = NULL;
    fftComputes[iGrid] = NULL;
  }
}

PmePencilXY::PmePencilXY(CkMigrateMessage *m) {
  NAMD_bug("PmePencilXY cannot be migrated");
//__sdag_init();
  // setMigratable(false);
  // fftCompute = NULL;
  // pmeTranspose = NULL;
}

PmePencilXY::~PmePencilXY() {
//   if (fftCompute != NULL) delete fftCompute;
//   if (pmeTranspose != NULL) delete pmeTranspose;
//   if (fftCompute2 != NULL) delete fftCompute2;
//   if (pmeTranspose2 != NULL) delete pmeTranspose2;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeTransposes[iGrid] != NULL) delete pmeTransposes[iGrid];
    if (fftComputes[iGrid] != NULL) delete fftComputes[iGrid];
  }
}

void PmePencilXY::initFFT(PmeStartMsg *msg) {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilXY::initFFT, fftCompute not initialized");
//   fftCompute->init(msg->dataGrid[0], msg->dataSizes[0],  NULL, 0, Perm_X_Y_Z, pmeGrid, 2, 0, thisIndex.z, 0);
//   fftCompute2->init(msg->dataGrid[1], msg->dataSizes[1],  NULL, 0, Perm_X_Y_Z, pmeGrid, 2, 0, thisIndex.z, 0);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (msg->enabledGrid[iGrid] == true) {
      fftComputes[iGrid]->init(msg->dataGrid[iGrid], msg->dataSizes[iGrid],  NULL, 0, Perm_X_Y_Z, pmeGrid, 2, 0, thisIndex.z, 0);
    }
  }
}

void PmePencilXY::forwardFFT() {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilXY::forwardFFT, fftCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) fftComputes[iGrid]->forward();
  }
}

void PmePencilXY::backwardFFT() {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilXY::backwardFFT, fftCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) fftComputes[iGrid]->backward();
  }
}

void PmePencilXY::initBlockSizes() {
  blockSizes.resize(pmeGrid.xBlocks);
  for (int x=0;x < pmeGrid.xBlocks;x++) {
    int i0, i1, j0, j1, k0, k1;
    getBlockDim(pmeGrid, Perm_cX_Y_Z, x, 0, thisIndex.z,
      i0, i1, j0, j1, k0, k1);
    int size = (i1-i0+1)*(j1-j0+1)*(k1-k0+1);
    blockSizes[x] = size;
  }
}

void PmePencilXY::forwardDone() {
  NAMD_bug("PmePencilXY::forwardDone(), base class method called");
}

void PmePencilXY::backwardDone() {
  NAMD_bug("PmePencilXY::backwardDone(), base class method called");
}

void PmePencilXY::recvDataFromZ(PmeBlockMsg *msg) {
  NAMD_bug("PmePencilXY::recvDataFromZ(), base class method called");
}

void PmePencilXY::start(const CkCallback &) {
  NAMD_bug("PmePencilXY::start(), base class method called");
}

//###########################################################################
//###########################################################################
//###########################################################################

//
// Data flow for PmePencilX & PmePencilX & PmePencilZ
//
// dataSrc(X) [x]     dataDst(X)
//
// dataDst(X) [transp] dataSrc(Y)
//---------------------------------
//
// dataSrc(Y) [y]      dataDst(Y)
//
// dataDst(Y) [transp] dataSrc(Z)
//---------------------------------
//
// dataSrc(Z) [z]      dataDst(Z)
//
// dataDst(Z) [solve]  dataDst(Z)
//
// dataDst(Z) [z]      dataSrc(Z)
//
// dataSrc(Z) [transp] dataDst(Y)
//---------------------------------
//
// dataDst(Y) [y]      dataSrc(Y)
//
// dataSrc(Y) [transp] dataDst(X)
//---------------------------------
//
// dataDst(X) [x]      dataSrc(X)
//
// dataSrc(X) [force]
//

PmePencilX::PmePencilX() {
  __sdag_init();
  setMigratable(false);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    pmeTransposes[iGrid] = NULL;
    fftComputes[iGrid] = NULL;
  }
  numStrayAtoms = 0;
}

PmePencilX::PmePencilX(CkMigrateMessage *m) {
  NAMD_bug("PmePencilX cannot be migrated");
//__sdag_init();
  // setMigratable(false);
  // fftCompute = NULL;
  // pmeTranspose = NULL;
}

PmePencilX::~PmePencilX() {
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeTransposes[iGrid] != NULL) delete pmeTransposes[iGrid];
    if (fftComputes[iGrid] != NULL) delete fftComputes[iGrid];
  }
}

void PmePencilX::initFFT(PmeStartMsg *msg) {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilX::initFFT, fftCompute not initialized");
//   fftCompute->init(msg->dataGrid[0], msg->dataSizes[0],  NULL, 0, Perm_X_Y_Z, pmeGrid, 1, thisIndex.y, thisIndex.z, 0);
//   fftCompute2->init(msg->dataGrid[1], msg->dataSizes[1],  NULL, 0, Perm_X_Y_Z, pmeGrid, 1, thisIndex.y, thisIndex.z, 0);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (msg->enabledGrid[iGrid] == true) {
      fftComputes[iGrid]->init(msg->dataGrid[iGrid], msg->dataSizes[iGrid],  NULL, 0, Perm_X_Y_Z, pmeGrid, 1, thisIndex.y, thisIndex.z, 0);
    }
  }
}

void PmePencilX::forwardFFT() {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilX::forwardFFT, fftCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) fftComputes[iGrid]->forward();
  }
}

void PmePencilX::backwardFFT() {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilX::backwardFFT, fftCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) fftComputes[iGrid]->backward();
  }
}

void PmePencilX::initBlockSizes() {
  blockSizes.resize(pmeGrid.xBlocks);
  for (int x=0;x < pmeGrid.xBlocks;x++) {
    int i0, i1, j0, j1, k0, k1;
    getBlockDim(pmeGrid, Perm_cX_Y_Z, x, thisIndex.y, thisIndex.z,
      i0, i1, j0, j1, k0, k1);
    int size = (i1-i0+1)*(j1-j0+1)*(k1-k0+1);
    blockSizes[x] = size;
  }
}

void PmePencilX::forwardDone() {
  NAMD_bug("PmePencilX::forwardDone(), base class method called");
}

void PmePencilX::backwardDone() {
  NAMD_bug("PmePencilX::backwardDone(), base class method called");
}

void PmePencilX::recvDataFromY(PmeBlockMsg *msg) {
  NAMD_bug("PmePencilX::recvDataFromY(), base class method called");
}

void PmePencilX::start(const CkCallback &) {
  NAMD_bug("PmePencilX::start(), base class method called");
}

//###########################################################################
//###########################################################################
//###########################################################################

PmePencilY::PmePencilY() {
  __sdag_init();
  setMigratable(false);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    pmeTransposes[iGrid] = NULL;
    fftComputes[iGrid] = NULL;
  }
  numStrayAtoms = 0;
}

PmePencilY::PmePencilY(CkMigrateMessage *m) {
  NAMD_bug("PmePencilY cannot be migrated");
  // __sdag_init();
  // setMigratable(false);
  // fftCompute = NULL;
  // pmeTranspose = NULL;
}

PmePencilY::~PmePencilY() {
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeTransposes[iGrid] != NULL) delete pmeTransposes[iGrid];
    if (fftComputes[iGrid] != NULL) delete fftComputes[iGrid];
  }
}

void PmePencilY::initFFT(PmeStartMsg *msg) {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilY::initFFT, fftCompute not initialized");
//   fftCompute->init(msg->dataGrid[0], msg->dataSizes[0],  NULL, 0, Perm_Y_Z_cX, pmeGrid, 1, thisIndex.z, thisIndex.x, 0);
//   fftCompute2->init(msg->dataGrid[1], msg->dataSizes[1],  NULL, 0, Perm_Y_Z_cX, pmeGrid, 1, thisIndex.z, thisIndex.x, 0);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (msg->enabledGrid[iGrid] == true) {
      fftComputes[iGrid]->init(msg->dataGrid[iGrid], msg->dataSizes[iGrid],  NULL, 0, Perm_Y_Z_cX, pmeGrid, 1, thisIndex.z, thisIndex.x, 0);
    }
  }
}

void PmePencilY::forwardFFT() {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilY::forwardFFT, fftCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) fftComputes[iGrid]->forward();
  }
}

void PmePencilY::backwardFFT() {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilY::backwardFFT, fftCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) fftComputes[iGrid]->backward();
  }
}

void PmePencilY::initBlockSizes() {
  blockSizes.resize(pmeGrid.yBlocks);
  for (int y=0;y < pmeGrid.yBlocks;y++) {
    int i0, i1, j0, j1, k0, k1;
    getBlockDim(pmeGrid, Perm_Y_Z_cX, y, thisIndex.z, thisIndex.x,
      i0, i1, j0, j1, k0, k1);
    int size = (i1-i0+1)*(j1-j0+1)*(k1-k0+1);
    blockSizes[y] = size;
  }
}

void PmePencilY::forwardDone() {
  NAMD_bug("PmePencilY::forwardDone(), base class method called");
}

void PmePencilY::backwardDone() {
  NAMD_bug("PmePencilY::backwardDone(), base class method called");
}

void PmePencilY::recvDataFromX(PmeBlockMsg *msg) {
  NAMD_bug("PmePencilY::recvDataFromX(), base class method called");
}

void PmePencilY::recvDataFromZ(PmeBlockMsg *msg) {
  NAMD_bug("PmePencilY::recvDataFromZ(), base class method called");
}

void PmePencilY::start(const CkCallback &) {
  NAMD_bug("PmePencilY::start(), base class method called");
}

//###########################################################################
//###########################################################################
//###########################################################################

PmePencilZ::PmePencilZ() {
  __sdag_init();
  setMigratable(false);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    pmeTransposes[iGrid] = NULL;
    fftComputes[iGrid] = NULL;
    pmeKSpaceComputes[iGrid] = NULL;
  }
  reduction = ReductionMgr::Object()->willSubmit(REDUCTIONS_BASIC);
#ifdef NODEGROUP_FORCE_REGISTER
// #if false
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();
  nodeReduction = patchData->reduction;
#endif
  doEnergy = false;
  doVirial = false;
  numStrayAtoms = 0;
}

PmePencilZ::PmePencilZ(CkMigrateMessage *m) {
  NAMD_bug("PmePencilZ cannot be migrated");
  //__sdag_init();
  // setMigratable(false);
  // fftCompute = NULL;
  // pmeTranspose = NULL;
}

PmePencilZ::~PmePencilZ() {
//   if (fftCompute != NULL) delete fftCompute;
//   if (pmeTranspose != NULL) delete pmeTranspose;
//   if (pmeKSpaceCompute != NULL) delete pmeKSpaceCompute;
//   if (fftCompute2 != NULL) delete fftCompute2;
//   if (pmeTranspose2 != NULL) delete pmeTranspose2;
//   if (pmeKSpaceCompute2 != NULL) delete pmeKSpaceCompute2;
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeTransposes[iGrid] != NULL) delete pmeTransposes[iGrid];
    if (fftComputes[iGrid] != NULL) delete fftComputes[iGrid];
    if (pmeKSpaceComputes[iGrid] != NULL) delete pmeKSpaceComputes[iGrid];
  }
  delete reduction;
}

void PmePencilZ::initFFT(PmeStartMsg *msg) {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilZ::initFFT, fftCompute not initialized");
//   fftCompute->init(msg->dataGrid[0], msg->dataSizes[0],  NULL, 0, Perm_Z_cX_Y, pmeGrid, 1, thisIndex.x, thisIndex.y, 0);
//   fftCompute2->init(msg->dataGrid[1], msg->dataSizes[1],  NULL, 0, Perm_Z_cX_Y, pmeGrid, 1, thisIndex.x, thisIndex.y, 0);
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (msg->enabledGrid[iGrid] == true) {
      fftComputes[iGrid]->init(msg->dataGrid[iGrid], msg->dataSizes[iGrid],  NULL, 0, Perm_Z_cX_Y, pmeGrid, 1, thisIndex.x, thisIndex.y, 0);
    }
  }
}

void PmePencilZ::forwardFFT() {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilZ::forwardFFT, fftCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) fftComputes[iGrid]->forward();
  }
}

void PmePencilZ::backwardFFT() {
  if (fftComputes[0] == NULL)
    NAMD_bug("PmePencilZ::backwardFFT, fftCompute not initialized");
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (fftComputes[iGrid] != NULL) fftComputes[iGrid]->backward();
  }
}

void PmePencilZ::initBlockSizes() {
  blockSizes.resize(pmeGrid.zBlocks);
  for (int z=0;z < pmeGrid.zBlocks;z++) {
    int i0, i1, j0, j1, k0, k1;
    getBlockDim(pmeGrid, Perm_Z_cX_Y, z, thisIndex.x, thisIndex.y,
      i0, i1, j0, j1, k0, k1);
    int size = (i1-i0+1)*(j1-j0+1)*(k1-k0+1);
    blockSizes[z] = size;
  }
}

void PmePencilZ::forwardDone() {
  if (pmeKSpaceComputes[0] == NULL)
    NAMD_bug("PmePencilZ::forwardDone, pmeKSpaceCompute not initialized");
//   pmeKSpaceCompute->solve(lattice, doEnergy, doVirial, fftCompute->getDataDst());
//   pmeKSpaceCompute2->solve(lattice, doEnergy, doVirial, fftCompute2->getDataDst());
  for (unsigned int iGrid = 0; iGrid < NUM_GRID_MAX; ++iGrid) {
    if (pmeKSpaceComputes[iGrid] != NULL) {
      pmeKSpaceComputes[iGrid]->solve(lattice, doEnergy, doVirial, fftComputes[iGrid]->getDataDst());
    }
  }
}

void PmePencilZ::submitReductions(unsigned int iGrid) {
  if (pmeKSpaceComputes[iGrid] == NULL)
    NAMD_bug("PmePencilZ::submitReductions, pmeKSpaceCompute not initialized");
//  fprintf(stderr, "PmePencilZ::submitReductions\n");
  SimParameters* simParams = Node::Object()->simParameters;
  double virial[9];
  double energy = pmeKSpaceComputes[iGrid]->getEnergy();
  pmeKSpaceComputes[iGrid]->getVirial(virial);
  if (simParams->alchOn) {
    double energy_F = energy;
    double scale1 = 1.0;
    double scale2 = 1.0;
    if (simParams->alchFepOn) {
      switch (iGrid) {
        case 0: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal alchLambda2 = simParams->getCurrentLambda2(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          const BigReal elecLambda2Up = simParams->getElecLambda(alchLambda2);
          scale1 = elecLambdaUp;
          scale2 = elecLambda2Up;
          break;
        }
        case 1: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal alchLambda2 = simParams->getCurrentLambda2(simulationStep);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          const BigReal elecLambda2Down = simParams->getElecLambda(1 - alchLambda2);
          scale1 = elecLambdaDown;
          scale2 = elecLambda2Down;
          break;
        }
        case 2: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal alchLambda2 = simParams->getCurrentLambda2(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          const BigReal elecLambda2Up = simParams->getElecLambda(alchLambda2);
          scale1 = 1.0 - elecLambdaUp;
          scale2 = 1.0 - elecLambda2Up;
          break;
        }
        case 3: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal alchLambda2 = simParams->getCurrentLambda2(simulationStep);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          const BigReal elecLambda2Down = simParams->getElecLambda(1 - alchLambda2);
          scale1 = 1.0 - elecLambdaDown;
          scale2 = 1.0 - elecLambda2Down;
          break;
        }
        case 4: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal alchLambda2 = simParams->getCurrentLambda2(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          const BigReal elecLambda2Up = simParams->getElecLambda(alchLambda2);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          const BigReal elecLambda2Down = simParams->getElecLambda(1 - alchLambda2);
          scale1 = -1.0 * (elecLambdaUp + elecLambdaDown - 1.0);
          scale2 = -1.0 * (elecLambda2Up + elecLambda2Down - 1.0);
          break;
        }
      }
      energy *= scale1;
      energy_F *= scale2;
      for (size_t i = 0; i < 9; ++i) {
        virial[i] *= scale1;
      }
#if NODEGROUP_FORCE_REGISTER
      nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_F) += energy_F;
#endif
      reduction->item(REDUCTION_ELECT_ENERGY_SLOW_F) += energy_F;
    }
    if (simParams->alchThermIntOn) {
      double energy_TI_1 = 0.0;
      double energy_TI_2 = 0.0;
      double scale1 = 1.0;
      switch (iGrid) {
        case 0: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          scale1 = elecLambdaUp;
          energy_TI_1 = energy;
          break;
        }
        case 1: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          scale1 = elecLambdaDown;
          energy_TI_2 = energy;
          break;
        }
        case 2: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          scale1 = 1.0 - elecLambdaUp;
          energy_TI_1 = -1.0 * energy;
          break;
        }
        case 3: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          scale1 = 1.0 - elecLambdaDown;
          energy_TI_2 = -1.0 * energy;
          break;
        }
        case 4: {
          const BigReal alchLambda = simParams->getCurrentLambda(simulationStep);
          const BigReal elecLambdaUp = simParams->getElecLambda(alchLambda);
          const BigReal elecLambdaDown = simParams->getElecLambda(1 - alchLambda);
          scale1 = -1.0 * (elecLambdaUp + elecLambdaDown - 1.0);
          energy_TI_1 = -1.0 * energy;
          energy_TI_2 = -1.0 * energy;
          break;
        }
      }
//       fprintf(stdout, "Grid %u : energy_TI_1 = %lf ; energy_TI_2 = %lf\n", iGrid, energy_TI_1, energy_TI_2);
#if NODEGROUP_FORCE_REGISTER
      nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1) += energy_TI_1;
      nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2) += energy_TI_2;
#endif
      reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1) += energy_TI_1;
      reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2) += energy_TI_2;
    }
  }
#ifdef NODEGROUP_FORCE_REGISTER
// #if false
  // XXX Expect a race condition, need atomic access to nodeReduction
  nodeReduction->item(REDUCTION_ELECT_ENERGY_SLOW) += energy;
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_XX) += virial[0];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_XY) += virial[1];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_XZ) += virial[2];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_YX) += virial[3];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_YY) += virial[4];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_YZ) += virial[5];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_ZX) += virial[6];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_ZY) += virial[7];
  nodeReduction->item(REDUCTION_VIRIAL_SLOW_ZZ) += virial[8];
#endif
  reduction->item(REDUCTION_ELECT_ENERGY_SLOW) += energy;
  reduction->item(REDUCTION_VIRIAL_SLOW_XX) += virial[0];
  reduction->item(REDUCTION_VIRIAL_SLOW_XY) += virial[1];
  reduction->item(REDUCTION_VIRIAL_SLOW_XZ) += virial[2];
  reduction->item(REDUCTION_VIRIAL_SLOW_YX) += virial[3];
  reduction->item(REDUCTION_VIRIAL_SLOW_YY) += virial[4];
  reduction->item(REDUCTION_VIRIAL_SLOW_YZ) += virial[5];
  reduction->item(REDUCTION_VIRIAL_SLOW_ZX) += virial[6];
  reduction->item(REDUCTION_VIRIAL_SLOW_ZY) += virial[7];
  reduction->item(REDUCTION_VIRIAL_SLOW_ZZ) += virial[8];
  reduction->item(REDUCTION_STRAY_CHARGE_ERRORS) += numStrayAtoms;
  numStrayAtoms = 0;
  energyReady[iGrid] = 1;
  bool ready_to_submit = true;
  for (size_t i = 0; i < NUM_GRID_MAX; ++i) {
    if (energyReady[i] == -1) continue;
    if (energyReady[i] == 0) ready_to_submit = false;
  }
  if (ready_to_submit) {
    reduction->submit();
    for (size_t i = 0; i < NUM_GRID_MAX; ++i) {
      if (energyReady[i] == -1) continue;
      if (energyReady[i] == 1) energyReady[i] = 0;
    }
  }
}

void PmePencilZ::backwardDone() {
  NAMD_bug("PmePencilZ::backwardDone(), base class method called");
}

void PmePencilZ::recvDataFromY(PmeBlockMsg *msg) {
  NAMD_bug("PmePencilY::recvDataFromY(), base class method called");
}

void PmePencilZ::start(const CkCallback &) {
  NAMD_bug("PmePencilZ::start(), base class method called");
}

void PmePencilZ::skip() {
  reduction->submit();  
}

#include "PmeSolver.def.h"
