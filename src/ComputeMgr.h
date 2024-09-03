/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef COMPUTEMGR_H
#define COMPUTEMGR_H

#include "charm++.h"
#include "main.h"
#include <new>

#include "NamdTypes.h"
#include "BOCgroup.h"

#include "ResizeArray.h"

#include "GlobalMaster.h"
#include "GlobalMasterServer.h"
#include "ComputeMgr.decl.h"

class Compute;
class ComputeMap;
class CkQdMsg;

class ComputeGlobal;
class ComputeGlobalConfigMsg;
class ComputeGlobalDataMsg;
class ComputeGlobalResultsMsg;

class ComputeDPME;
class ComputeDPMEDataMsg;
class ComputeDPMEResultsMsg;
class ComputeConsForceMsg;

class ComputeEwald;
class ComputeEwaldMsg;

class FinishWorkMsg;

class ComputeNonbondedMIC;
class NonbondedMICSlaveMsg;
class NonbondedMICSkipMsg;

class ComputeNonbondedWorkArrays;
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
class CudaComputeNonbonded;
#ifdef BONDED_CUDA
class ComputeBondedCUDA;
#endif
#endif
class GMWakeMsg : public CMessage_GMWakeMsg{
public:
  int senderPe;
  GMWakeMsg(int pe): senderPe(pe)
  {}
};

class ComputeMgr : public CBase_ComputeMgr
{
public:

  ComputeMgr();
  ~ComputeMgr();
  void createComputes(ComputeMap *map);
  void updateComputes(int,CkGroupID);
  void updateComputes2(CkQdMsg *);
  void updateComputes3();
  void splitComputes();
  void splitComputes2(CkQdMsg *);
  void updateLocalComputes();
  void updateLocalComputes2(CkQdMsg *);
  void updateLocalComputes3();
  void updateLocalComputes4(CkQdMsg *);
  void updateLocalComputes5();
  void doneUpdateLocalComputes();

  void sendComputeGlobalConfig(ComputeGlobalConfigMsg *);
  void recvComputeGlobalConfig(ComputeGlobalConfigMsg *);
  void sendComputeGlobalData(ComputeGlobalDataMsg *);
  void recvComputeGlobalData(ComputeGlobalDataMsg *);
  void sendComputeGlobalResults(ComputeGlobalResultsMsg *);
  void recvComputeGlobalResults(ComputeGlobalResultsMsg *);
  void enableComputeGlobalResults();

  void sendComputeDPMEData(ComputeDPMEDataMsg *);
  void recvComputeDPMEData(ComputeDPMEDataMsg *);
  void sendComputeDPMEResults(ComputeDPMEResultsMsg *, int);
  void recvComputeDPMEResults(ComputeDPMEResultsMsg *);

  void sendComputeEwaldData(ComputeEwaldMsg *);
  void recvComputeEwaldData(ComputeEwaldMsg *);
  void sendComputeEwaldResults(ComputeEwaldMsg *);
  void recvComputeEwaldResults(ComputeEwaldMsg *);

  void recvComputeConsForceMsg(ComputeConsForceMsg *);
#ifdef NODEGROUP_FORCE_REGISTER
  
  CthThread stowedThread;
  std::atomic<int> *suspendCounter;

  void wakeStowedULTs(GMWakeMsg *m)
  {
    //    CkPrintf("[%d] wakey thread %d\n",CkMyPe(),CthGetToken(stowedThread)->serialNo);
    if(CkMyPe()!=m->senderPe)
      CthAwaken(stowedThread);
  }

  void wakeStowedULT()
  {
    //    CkPrintf("[%d] wakey thread %d\n",CkMyPe(),CthGetToken(stowedThread)->serialNo);
    CthAwaken(stowedThread);
  }

  //! stow this ULT in this PE's slot for it in the nodegroup
  void stowSuspendULT()
  {
    //    CkPrintf("[%d] stowSuspend thread %d\n",CkMyPe(),CthGetToken(CthSelf())->serialNo);
    stowedThread=CthSelf();
    // last one here wakes everyone up
    // the fetch_sub gives us the result before the decrement
    if(suspendCounter->fetch_sub(1)==1)
      {
	awakenStowedULTs();
      }
    else
      {
	//	CkPrintf("[%d] size suspendCounter %d:%d\n",CkMyPe(), suspendCounter->load(), CmiMyNodeSize());
	CthSuspend();
      }
  }

  //! send a message to all other PEs on the node to wake up the stowed thread
  void awakenStowedULTs()
  {
    //    CkPrintf("[%d] awakenStowedULTs at counter %d\n",CkMyPe(), suspendCounter->load());
    int size=CmiMyNodeSize();
    GMWakeMsg* wakeUp= new GMWakeMsg(CkMyPe());
    suspendCounter->store(size);
    thisProxy.wakeStowedULTs(wakeUp);
  }

#endif  
  // Made public in order to access the ComputeGlobal on the node
  ComputeGlobal *computeGlobalObject; /* node part of global computes */
  ResizeArray<ComputeGlobalResultsMsg*> computeGlobalResultsMsgs;
  int computeGlobalResultsMsgSeq;
  int computeGlobalResultsMsgMasterSeq;
  CkCallback callMeBackCB;

  void sendYieldDevice(int pe);
  void recvYieldDevice(int pe);

  // DMK
  void sendBuildMICForceTable();
  void recvBuildMICForceTable();

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
  void sendAssignPatchesOnPe(std::vector<int>& pes, CudaComputeNonbonded* c);
  void recvAssignPatchesOnPe(CudaComputeNonbondedMsg *msg);
  void sendSkipPatchesOnPe(std::vector<int>& pes, CudaComputeNonbonded* c);
  void recvSkipPatchesOnPe(CudaComputeNonbondedMsg *msg);
  void sendFinishPatchesOnPe(std::vector<int>& pes, CudaComputeNonbonded* c);
  void recvFinishPatchesOnPe(CudaComputeNonbondedMsg *msg);
  void sendFinishPatchOnPe(int pe, CudaComputeNonbonded* c, int i, PatchID patchID);
  void recvFinishPatchOnPe(CudaComputeNonbondedMsg *msg);
  void sendOpenBoxesOnPe(std::vector<int>& pes, CudaComputeNonbonded* c);
  void recvOpenBoxesOnPe(CudaComputeNonbondedMsg *msg);
  void sendFinishReductions(int pe, CudaComputeNonbonded* c);
  void recvFinishReductions(CudaComputeNonbondedMsg *msg);
  void sendMessageEnqueueWork(int pe, CudaComputeNonbonded* c);
  void recvMessageEnqueueWork(CudaComputeNonbondedMsg *msg);
  void sendLaunchWork(int pe, CudaComputeNonbonded* c);
  void recvLaunchWork(CudaComputeNonbondedMsg *msg);
  void sendUnregisterBoxesOnPe(std::vector<int>& pes, CudaComputeNonbonded* c);
  void recvUnregisterBoxesOnPe(CudaComputeNonbondedMsg *msg);
#ifdef BONDED_CUDA
  void sendAssignPatchesOnPe(std::vector<int>& pes, ComputeBondedCUDA* c);
  void recvAssignPatchesOnPe(ComputeBondedCUDAMsg *msg);
  void sendMessageEnqueueWork(int pe, ComputeBondedCUDA* c);
  void recvMessageEnqueueWork(ComputeBondedCUDAMsg *msg);
  void sendOpenBoxesOnPe(std::vector<int>& pes, ComputeBondedCUDA* c);
  void recvOpenBoxesOnPe(ComputeBondedCUDAMsg *msg);
  void sendLoadTuplesOnPe(std::vector<int>& pes, ComputeBondedCUDA* c);
  void recvLoadTuplesOnPe(ComputeBondedCUDAMsg *msg);
  void sendLaunchWork(int pe, ComputeBondedCUDA* c);
  void recvLaunchWork(ComputeBondedCUDAMsg *msg);
  void sendFinishPatchesOnPe(std::vector<int>& pes, ComputeBondedCUDA* c);
  void recvFinishPatchesOnPe(ComputeBondedCUDAMsg *msg);
  void sendFinishReductions(int pe, ComputeBondedCUDA* c);
  void recvFinishReductions(ComputeBondedCUDAMsg *msg);
  void sendUnregisterBoxesOnPe(std::vector<int>& pes, ComputeBondedCUDA* c);
  void recvUnregisterBoxesOnPe(ComputeBondedCUDAMsg *msg);
#endif
#endif
  void sendCreateNonbondedMICSlave(int,int);
  void recvCreateNonbondedMICSlave(NonbondedMICSlaveMsg *);
  void sendNonbondedMICSlaveReady(int,int,int,int);
  void recvNonbondedMICSlaveReady(int,int,int);
  void sendNonbondedMICSlaveSkip(ComputeNonbondedMIC *c, int);
  void recvNonbondedMICSlaveSkip(NonbondedMICSkipMsg *);
  void sendNonbondedMICSlaveEnqueue(ComputeNonbondedMIC *c, int,int,int,int);
  void sendMICPEData(int,int);
  void recvMICPEData(int,int);
  int isMICProcessor(int);
  
private:
  void createCompute(ComputeID, ComputeMap *);

  GlobalMasterServer *masterServerObject; /* master part of global computes */
  ComputeDPME *computeDPMEObject;

  ComputeEwald *computeEwaldObject;

  ComputeNonbondedMIC *computeNonbondedMICObject;

  ComputeNonbondedWorkArrays *computeNonbondedWorkArrays;

  int skipSplitting;
  int updateComputesCount;
  int updateComputesReturnEP;
  CkGroupID updateComputesReturnChareID;

  ResizeArray<int> computeFlag;

  int* micPEData;  // DMK : An array (1 bit per PE) which will hold a flag indicating if a given PE is driving a MIC card or not
};


#endif /* COMPUTEMGR_H */

