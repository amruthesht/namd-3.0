#ifndef COMPUTEPMECUDA_H
#define COMPUTEPMECUDA_H

#include <vector>
#include <list>
#include "PmeBase.h"

#include "PatchTypes.h"       // Results
#include "Compute.h"
#include "Box.h"
#include "OwnerBox.h"
#include "ComputePmeCUDAMgr.decl.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
class HomePatch;

class ComputePmeCUDA : public Compute {
public:
  ComputePmeCUDA(ComputeID c, PatchIDList& pids);
  ComputePmeCUDA(ComputeID c, PatchID pid);
  virtual ~ComputePmeCUDA();
  void initialize();
  void atomUpdate();
  int noWork();
  void doWork();
  bool storePmeForceMsg(PmeForceMsg *msg);
private:
  struct PatchRecord {
    PatchRecord() {
      pmeForceMsg = NULL;
      patch = NULL;
      positionBox = NULL;
      avgPositionBox = NULL;
      forceBox = NULL;
    }
    // Message that contains the pointers to forces
    PmeForceMsg* pmeForceMsg;
    // Home pencil
    int homePencilY;
    int homePencilZ;
    int homePencilNode;
    // Pointer to patch
    Patch *patch;
    // Patch ID
    PatchID patchID;
    // Boxes
    Box<Patch,CompAtom> *positionBox;
    Box<Patch,CompAtom> *avgPositionBox;
    Box<Patch,Results> *forceBox;
  };

  void calcSelfEnergy(int numAtoms, CompAtom *x, bool isFirstStep);
  void calcSelfEnergyFEP(int numAtoms, CompAtom *atoms, bool isFirstStep);
  void calcSelfEnergyTI(int numAtoms, CompAtom *atoms, bool isFirstStep);
  void sendAtoms();
  void recvForces();
  void setupActivePencils();

  CmiNodeLock lock;
  int patchCounter;

  std::vector<PatchRecord> patches;

  SubmitReduction *reduction;
#ifdef NODEGROUP_FORCE_REGISTER
  NodeReduction *nodeReduction;
#endif

  bool sendAtomsDone;

  PmeGrid pmeGrid;

  ComputePmeCUDAMgr *mgr;

  CProxy_ComputePmeCUDAMgr computePmeCUDAMgrProxy;

  bool atomsChanged;

  // saved self energies
  double selfEnergy;
  double selfEnergyFEP;
  double selfEnergyTI1;
  double selfEnergyTI2;
};
#endif // NAMD_CUDA

#endif // COMPUTEPMECUDA_H
