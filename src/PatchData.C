#include "PatchData.h"
#include "PatchMap.h"

#ifdef NODEGROUP_FORCE_REGISTER

// initialize all pointers to NULL and all scalars to zero
DeviceData::DeviceData(){
  // bonded scalars
  bond_pi_size = 0;
  bond_pr_size = 0;
  forceStride  = 0;
  f_bond_size  = 0;

  // bonded pointers
  bond_pi = NULL;
  bond_pr = NULL;
  f_bond  = NULL;
  f_bond_nbond = NULL;

  // slow scalars
  f_slow_size = 0;
  slow_patchPositionsSize = 0;
  slow_pencilPatchIndexSize = 0;
  slow_patchIDSize = 0;

  // slow vectors
  f_slow = NULL;
  slow_patchPositions = NULL;
  slow_pencilPatchIndex = NULL;
  slow_patchID = NULL;

  // nonbonded scalars
  f_nbond_size = 0;

  // nonbonded vectors
  f_nbond = NULL;
  f_nbond_slow = NULL;

  numAtomsHome = 0;
}

DeviceData::~DeviceData(){
  free(h_hasPatches);
  cudaCheck(cudaFree(d_hasPatches));
}
#endif


PatchData::PatchData() {
  ptrCollectionMaster = NULL;
  ptrOutput = NULL;
  imd = NULL;
  pdb = NULL;

  reduction = new NodeReduction();
  nodeReductionSave = new NodeReduction();
#ifdef NODEGROUP_FORCE_REGISTER
  cudaBondedList = new ComputeBondedCUDA*[CkNumPes()];
  cudaNonbondedList = new CudaComputeNonbonded*[CkNumPes()];
  printlock = CmiCreateLock();
  nodeLock = CmiCreateLock();
  suspendCounter.store(CmiMyNodeSize());
#endif
}

PatchData::~PatchData() {
  delete reduction;
  delete nodeReductionSave;
#ifdef NODEGROUP_FORCE_REGISTER
  free(h_soa_fb_x);
  free(h_soa_fb_y);
  free(h_soa_fb_z);
  free(h_soa_fn_x);
  free(h_soa_fn_y);
  free(h_soa_fn_z);
  free(h_soa_fs_x);
  free(h_soa_fs_y);
  free(h_soa_fs_z);
  free(h_soa_pos_x);
  free(h_soa_pos_y);
  free(h_soa_pos_z);
  free(h_soa_vel_x);
  free(h_soa_vel_y);
  free(h_soa_vel_z);

  free(h_soa_charge);

  // Device Migration
  free(h_soa_id);
  free(h_soa_vdwType);
  free(h_soa_sortOrder);
  free(h_soa_unsortOrder);
  free(h_soa_patchCenter);
  free(h_soa_migrationDestination);
  free(h_soa_sortSoluteIndex);

  free(h_atomdata_AoS);
  free(h_peer_record);

  free(h_soa_partition);

  free(h_tupleCount.bond);
  free(h_tupleCount.angle);
  free(h_tupleCount.dihedral);
  free(h_tupleCount.improper);
  free(h_tupleCount.modifiedExclusion);
  free(h_tupleCount.exclusion);
  free(h_tupleCount.crossterm);
  free(h_tupleOffset.bond);
  free(h_tupleOffset.angle);
  free(h_tupleOffset.dihedral);
  free(h_tupleOffset.improper);
  free(h_tupleOffset.modifiedExclusion);
  free(h_tupleOffset.exclusion);
  free(h_tupleOffset.crossterm);
  free(h_tupleDataStage.bond);
  free(h_tupleDataStage.angle);
  free(h_tupleDataStage.dihedral);
  free(h_tupleDataStage.improper);
  free(h_tupleDataStage.modifiedExclusion);
  free(h_tupleDataStage.exclusion);
  free(h_tupleDataStage.crossterm);
  CmiDestroyLock(nodeLock);

#endif
}

#include "PatchData.def.h"
