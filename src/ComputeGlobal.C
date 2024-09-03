/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/*
   Forwards atoms to master node for force evaluation.
*/

#include "InfoStream.h"
#include "Node.h"
#include "PatchMap.h"
#include "PatchMap.inl"
#include "AtomMap.h"
#include "ComputeGlobal.h"
#include "ComputeGlobalMsgs.h"
#include "GridForceGrid.h"
#include "PatchMgr.h"
#include "Molecule.h"
#include "ReductionMgr.h"
#include "ComputeMgr.h"
#include "ComputeMgr.decl.h"
#include "SimParameters.h"
#include "PatchData.h"
#include <stdio.h>
#include <algorithm>
#include "NamdEventsProfiling.h"
#define MIN_DEBUG_LEVEL 3
//#define DEBUGM
#include "Debug.h"

#include "GridForceGrid.inl"
#include "MGridforceParams.h"

// CLIENTS

ComputeGlobal::ComputeGlobal(ComputeID c, ComputeMgr *m)
  : ComputeHomePatches(c)
{
  DebugM(3,"Constructing client\n");
  aid.resize(0);
  gdef.resize(0);
  comm = m;
  firsttime = 1;
  isRequested = 0;
  isRequestedAllocSize = 0;
  endRequested = 0;
  numGroupsRequested = 0;
  SimParameters *sp = Node::Object()->simParameters;
  dofull = (sp->GBISserOn || sp->GBISOn || sp->fullDirectOn || sp->FMAOn || sp->PMEOn);
  forceSendEnabled = 0;
  if ( sp->tclForcesOn ) forceSendEnabled = 1;
  if ( sp->colvarsOn ) forceSendEnabled = 1;
  forceSendActive = 0;
  fid.resize(0);
  totalForce.resize(0);
  gfcount = 0;
  groupTotalForce.resize(0);
  reduction = ReductionMgr::Object()->willSubmit(REDUCTIONS_BASIC);
  int numPatches = PatchMap::Object()->numPatches();
  forcePtrs = new Force*[numPatches];
  atomPtrs = new FullAtom*[numPatches];
  for ( int i = 0; i < numPatches; ++i ) { forcePtrs[i] = 0; atomPtrs[i] = 0; }

  #ifdef NODEGROUP_FORCE_REGISTER
  CProxy_PatchData cpdata(CkpvAccess(BOCclass_group).patchData);
  PatchData *patchData = cpdata.ckLocalBranch();
  nodeReduction = patchData->reduction;
  #endif
  if (sp->SOAintegrateOn || sp->CUDASOAintegrateMode) {
    // Allocate memory for numPatches to access SOA data
    mass_soa = new float*[numPatches];
    pos_soa_x = new double*[numPatches];
    pos_soa_y = new double*[numPatches];
    pos_soa_z = new double*[numPatches];
    force_soa_x = new double*[numPatches];
    force_soa_y = new double*[numPatches];
    force_soa_z = new double*[numPatches];
    transform_soa_i = new int*[numPatches];
    transform_soa_j = new int*[numPatches];
    transform_soa_k = new int*[numPatches];
    for ( int i = 0; i < numPatches; ++i ) { 
      mass_soa[i] = NULL;
      pos_soa_x[i] = NULL; 
      pos_soa_y[i] = NULL; 
      pos_soa_z[i] = NULL;
      force_soa_x[i] = NULL;
      force_soa_y[i] = NULL;
      force_soa_z[i] = NULL;
      transform_soa_i[i] = NULL; 
      transform_soa_j[i] = NULL; 
      transform_soa_k[i] = NULL;
    }
  } else {
    mass_soa = NULL;
    pos_soa_x = NULL;
    pos_soa_y = NULL;
    pos_soa_z = NULL;
    force_soa_x = NULL;
    force_soa_y = NULL;
    force_soa_z = NULL;
    transform_soa_i = NULL;
    transform_soa_j = NULL;
    transform_soa_k = NULL;
  }
  gridForcesPtrs = new ForceList **[numPatches];
  numGridObjects = numActiveGridObjects = 0;
  for ( int i = 0; i < numPatches; ++i ) {
    forcePtrs[i] = NULL; atomPtrs[i] = NULL;
    gridForcesPtrs[i] = NULL;
  }
}

ComputeGlobal::~ComputeGlobal()
{
  delete[] isRequested;
  delete[] forcePtrs;
  deleteGridObjects();
  delete[] gridForcesPtrs;
  delete[] atomPtrs;
  delete reduction;

  if(mass_soa) delete [] mass_soa;
  if(pos_soa_x) delete [] pos_soa_x;
  if(pos_soa_y) delete [] pos_soa_y;
  if(pos_soa_z) delete [] pos_soa_z;
  if(force_soa_x) delete [] force_soa_x;
  if(force_soa_y) delete [] force_soa_y;
  if(force_soa_z) delete [] force_soa_z;
  if(transform_soa_i) delete [] transform_soa_i;
  if(transform_soa_j) delete [] transform_soa_j;
  if(transform_soa_k) delete [] transform_soa_k;
}

void ComputeGlobal::configure(AtomIDList &newaid, AtomIDList &newgdef, IntList &newgridobjid) {
  DebugM(4,"Receiving configuration (" << newaid.size() <<
         " atoms, " << newgdef.size() << " atoms/groups and " <<
         newgridobjid.size() << " grid objects) on client\n" << endi);

  AtomIDList::iterator a, a_e;

  if ( forceSendEnabled ) {
    // clear previous data
    int max = -1;
    for (a=newaid.begin(),a_e=newaid.end(); a!=a_e; ++a) {
      if ( *a > max ) max = *a;
    }
    for (a=newgdef.begin(),a_e=newgdef.end(); a!=a_e; ++a) {
      if ( *a > max ) max = *a;
    }
    endRequested = max+1;
    if ( endRequested > isRequestedAllocSize ) {
      delete [] isRequested;
      isRequestedAllocSize = endRequested+10;
      isRequested = new char[isRequestedAllocSize];
      memset(isRequested, 0, isRequestedAllocSize);
    } else {
      for (a=aid.begin(),a_e=aid.end(); a!=a_e; ++a) {
        isRequested[*a] = 0;
      }
      for (a=gdef.begin(),a_e=gdef.end(); a!=a_e; ++a) {
        if ( *a != -1 ) isRequested[*a] = 0;
      }
    }
    // reserve space
    gpair.resize(0);
    gpair.resize(newgdef.size());
    gpair.resize(0);
  }

  // store data
  aid.swap(newaid);
  gdef.swap(newgdef);

  if (newgridobjid.size()) configureGridObjects(newgridobjid);

  if ( forceSendEnabled ) {
    int newgcount = 0;
    for (a=aid.begin(),a_e=aid.end(); a!=a_e; ++a) {
      isRequested[*a] = 1;
    }
    for (a=gdef.begin(),a_e=gdef.end(); a!=a_e; ++a) {
      if ( *a == -1 ) ++newgcount;
      else {
        isRequested[*a] |= 2;
        gpair.add(intpair(*a,newgcount));
      }
    }
    std::sort(gpair.begin(),gpair.end());
    numGroupsRequested = newgcount;
  }
  DebugM(3,"Done configure on client\n");
}

void ComputeGlobal::deleteGridObjects()
{
  if (numGridObjects == 0) return;
  ResizeArrayIter<PatchElem> ap(patchList);
  for (ap = ap.begin(); ap != ap.end(); ap++) {
    ForceList **gridForces = gridForcesPtrs[ap->p->getPatchID()];
    if (gridForces != NULL) {
      for (size_t ig = 0; ig < numGridObjects; ig++) {
        if (gridForces[ig] != NULL) {
          delete gridForces[ig];
          gridForces[ig] = NULL;
        }
      }
      delete [] gridForces;
      gridForces = NULL;
    }
  }
  numGridObjects = numActiveGridObjects = 0;
}

void ComputeGlobal::configureGridObjects(IntList &newgridobjid)
{
  Molecule *mol = Node::Object()->molecule;

  deleteGridObjects();

  numGridObjects = mol->numGridforceGrids;
  numActiveGridObjects = 0;

  gridObjActive.resize(numGridObjects);
  gridObjActive.setall(0);

  IntList::const_iterator goid_i = newgridobjid.begin();
  IntList::const_iterator goid_e = newgridobjid.end();
  for ( ; goid_i != goid_e; goid_i++) {
    if ((*goid_i < 0) || (*goid_i >= numGridObjects)) {
      NAMD_bug("Requested illegal gridForceGrid index.");
    } else {
      DebugM(3,"Adding grid with index " << *goid_i << " to ComputeGlobal\n");
      gridObjActive[*goid_i] = 1;
      numActiveGridObjects++;
    }
  }

  for (size_t ig = 0; ig < numGridObjects; ig++) {
    DebugM(3,"Grid index " << ig << " is active or inactive? "
           << gridObjActive[ig] << "\n" << endi);
  }

  ResizeArrayIter<PatchElem> ap(patchList);
  for (ap = ap.begin(); ap != ap.end(); ap++) {
    gridForcesPtrs[ap->p->getPatchID()] = new ForceList *[numGridObjects];
    ForceList **gridForces = gridForcesPtrs[ap->p->getPatchID()];
    for (size_t ig = 0; ig < numGridObjects; ig++) {
      if (gridObjActive[ig]) {
        gridForces[ig] = new ForceList;
      } else {
        gridForces[ig] = NULL;
      }
    }
  }
}

#if 0
void ComputeGlobal::recvConfig(ComputeGlobalConfigMsg *msg) {
  DebugM(3,"Receiving configure on client\n");
  configure(msg->aid,msg->gdef);
  delete msg;
  sendData();
}
#endif

void ComputeGlobal::recvResults(ComputeGlobalResultsMsg *msg) {
  DebugM(3,"Receiving results (" << msg->aid.size() << " forces, "
	 << msg->newgdef.size() << " new group atoms) on client thread " << CthGetToken(CthSelf())->serialNo <<" msg->resendCoordinates " << msg->resendCoordinates << " msg->totalforces " << msg->totalforces<< "\n");

  forceSendActive = msg->totalforces;
  if ( forceSendActive && ! forceSendEnabled ) NAMD_bug("ComputeGlobal::recvResults forceSendActive without forceSendEnabled");

  // set the forces only if we aren't going to resend the data
  int setForces = !msg->resendCoordinates;
  SimParameters *sp = Node::Object()->simParameters;
  
  if(setForces) { // we are requested to 
    // Store forces to patches
    AtomMap *atomMap = AtomMap::Object();
    const Lattice & lattice = patchList[0].p->lattice;
    ResizeArrayIter<PatchElem> ap(patchList);
    Force **f = forcePtrs;
    FullAtom **t = atomPtrs;
    Force extForce = 0.;
    Tensor extVirial;

    for (ap = ap.begin(); ap != ap.end(); ap++) {
      (*ap).r = (*ap).forceBox->open();
      f[(*ap).patchID] = (*ap).r->f[Results::normal];
      t[(*ap).patchID] = (*ap).p->getAtomList().begin();

      if (sp->SOAintegrateOn || sp->CUDASOAintegrateMode) {
        // Assigne the pointer to SOA data structure
        PatchID pId = (*ap).patchID;
        mass_soa[pId] = (*ap).p->patchDataSOA.mass;
        force_soa_x[pId] = (*ap).p->patchDataSOA.f_global_x;
        force_soa_y[pId] = (*ap).p->patchDataSOA.f_global_y;
        force_soa_z[pId] = (*ap).p->patchDataSOA.f_global_z;
        transform_soa_i[pId] = (*ap).p->patchDataSOA.transform_i;
        transform_soa_j[pId] = (*ap).p->patchDataSOA.transform_j;
        transform_soa_k[pId] = (*ap).p->patchDataSOA.transform_k;
      }
    }


    AtomIDList::iterator a = msg->aid.begin();
    AtomIDList::iterator a_e = msg->aid.end();
    ForceList::iterator f2 = msg->f.begin();
    if (sp->SOAintegrateOn || sp->CUDASOAintegrateMode) {
      LocalID localID;
      PatchID lpid;
      int lidx;
      Position x_orig, x_atom;
      Transform trans;
      Force f_atom;
      for ( ; a != a_e; ++a, ++f2 ) {
        DebugM(1,"processing atom "<<(*a)<<", F="<<(*f2)<<"...\n");
        /* XXX if (*a) is out of bounds here we get a segfault */
        localID = atomMap->localID(*a);
        lpid = localID.pid;
        lidx = localID.index;
        if ( lpid == notUsed || ! f[lpid] ) continue;
        f_atom = (*f2);
        // printf("NAMD3-recv: atom %d, Before Force (%8.6f, %8.6f, %8.6f) \n", 
        //   *a, force_soa_x[lpid][lidx], force_soa_y[lpid][lidx], force_soa_z[lpid][lidx]);
	//	printf("NAMD3-recv: atom %d, Added Force (%8.6f, %8.6f, %8.6f) \n", *a, f_atom.x, f_atom.y, f_atom.z);
        force_soa_x[lpid][lidx] += f_atom.x;
        force_soa_y[lpid][lidx] += f_atom.y;
        force_soa_z[lpid][lidx] += f_atom.z;
        x_orig.x = pos_soa_x[lpid][lidx];
        x_orig.y = pos_soa_y[lpid][lidx];
        x_orig.z = pos_soa_z[lpid][lidx];
        trans.i = transform_soa_i[lpid][lidx];
        trans.j = transform_soa_j[lpid][lidx];
        trans.k = transform_soa_k[lpid][lidx];
        x_atom = lattice.reverse_transform(x_orig,trans);
        extForce += f_atom;
        extVirial += outer(f_atom,x_atom);
      }
    } else {
      for ( ; a != a_e; ++a, ++f2 ) {
        DebugM(1,"processing atom "<<(*a)<<", F="<<(*f2)<<"...\n");
        /* XXX if (*a) is out of bounds here we get a segfault */
        LocalID localID = atomMap->localID(*a);
        if ( localID.pid == notUsed || ! f[localID.pid] ) continue;
        Force f_atom = (*f2);
        // printf("NAMD3-recv: atom %d, Before Force (%8.6f, %8.6f, %8.6f) \n", 
        //   *a, f[localID.pid][localID.index].x, f[localID.pid][localID.index].y, f[localID.pid][localID.index].z);
	// printf("NAMD3-recv: atom %d, Added Force (%8.6f, %8.6f, %8.6f) \n", *a, f_atom.x, f_atom.y, f_atom.z);
        f[localID.pid][localID.index] += f_atom;
        FullAtom &atom = t[localID.pid][localID.index];
        Position x_orig = atom.position;
        Transform trans = atom.transform;
        Position x_atom = lattice.reverse_transform(x_orig,trans);
        extForce += f_atom;
        extVirial += outer(f_atom,x_atom);
      }
    }
    DebugM(1,"done with the loop\n");

    // calculate forces for atoms in groups
    AtomIDList::iterator g_i, g_e;
    g_i = gdef.begin(); g_e = gdef.end();
    ForceList::iterator gf_i = msg->gforce.begin();
    //iout << iDEBUG << "recvResults\n" << endi;
    if (sp->SOAintegrateOn || sp->CUDASOAintegrateMode) {
      LocalID localID;
      PatchID lpid;
      int lidx;
      Position x_orig, x_atom;
      Transform trans;
      Force f_atom;
      for ( ; g_i != g_e; ++g_i, ++gf_i ) {
        //iout << iDEBUG << *gf_i << '\n' << endi;
        Vector accel = (*gf_i);
        for ( ; *g_i != -1; ++g_i ) {
          //iout << iDEBUG << *g_i << '\n' << endi;
          localID = atomMap->localID(*g_i);
          lpid = localID.pid;
          lidx = localID.index;
          if ( lpid == notUsed || ! f[lpid] ) continue;
          f_atom = accel * mass_soa[lpid][lidx];
#if 0	  
          if (*g_i < 20) {
            CkPrintf("NAMD3-recv: group %d, Before Force (%8.6f, %8.6f, %8.6f) \n", 
              *g_i, force_soa_x[lpid][lidx], force_soa_y[lpid][lidx], force_soa_z[lpid][lidx]);
            CkPrintf("NAMD3-recv: group %d, Added Force (%8.6f, %8.6f, %8.6f) \n", *g_i, f_atom.x, f_atom.y, f_atom.z);
          }
#endif	  
          force_soa_x[lpid][lidx] += f_atom.x;
          force_soa_y[lpid][lidx] += f_atom.y;
          force_soa_z[lpid][lidx] += f_atom.z;
          x_orig.x = pos_soa_x[lpid][lidx];
          x_orig.y = pos_soa_y[lpid][lidx];
          x_orig.z = pos_soa_z[lpid][lidx];
          trans.i = transform_soa_i[lpid][lidx];
          trans.j = transform_soa_j[lpid][lidx];
          trans.k = transform_soa_k[lpid][lidx];
          x_atom = lattice.reverse_transform(x_orig,trans);
          extForce += f_atom;
          extVirial += outer(f_atom,x_atom);
        }
      }
    } else {
      for ( ; g_i != g_e; ++g_i, ++gf_i ) {
        //iout << iDEBUG << *gf_i << '\n' << endi;
        Vector accel = (*gf_i);
        for ( ; *g_i != -1; ++g_i ) {
          //iout << iDEBUG << *g_i << '\n' << endi;
          LocalID localID = atomMap->localID(*g_i);
          if ( localID.pid == notUsed || ! f[localID.pid] ) continue;
          FullAtom &atom = t[localID.pid][localID.index];
          Force f_atom = accel * atom.mass;
#if 0
          if (*g_i < 20) {
            CkPrintf("NAMD2-recv: group %d, Before Force (%8.6f, %8.6f, %8.6f) \n", 
              *g_i, f[localID.pid][localID.index].x, f[localID.pid][localID.index].y, f[localID.pid][localID.index].z);
            CkPrintf("NAMD2-recv: group %d, Added Force (%8.6f, %8.6f, %8.6f) \n", *g_i, f_atom.x, f_atom.y, f_atom.z);
          }
#endif	  
          f[localID.pid][localID.index] += f_atom;
          Position x_orig = atom.position;
          Transform trans = atom.transform;
          Position x_atom = lattice.reverse_transform(x_orig,trans);
          extForce += f_atom;
          extVirial += outer(f_atom,x_atom);
        }
      }
    }
    DebugM(1,"done with the groups\n");

    if (numActiveGridObjects > 0) {
      applyGridObjectForces(msg, &extForce, &extVirial);
    }
    //    printf("Finish receiving at step: %d ####################################################\n", 
    //      patchList[0].p->flags.step);

    #ifdef NODEGROUP_FORCE_REGISTER
    if (sp->CUDASOAintegrate) {
      ADD_VECTOR_OBJECT(nodeReduction,REDUCTION_EXT_FORCE_NORMAL,extForce);
      ADD_TENSOR_OBJECT(nodeReduction,REDUCTION_VIRIAL_NORMAL,extVirial);
    } else {
      ADD_VECTOR_OBJECT(reduction,REDUCTION_EXT_FORCE_NORMAL,extForce);
      ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,extVirial);
      reduction->submit();
    }
    #else
    ADD_VECTOR_OBJECT(reduction,REDUCTION_EXT_FORCE_NORMAL,extForce);
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,extVirial);
    reduction->submit();
    #endif
  }
  // done setting the forces, close boxes below

  // Get reconfiguration if present
  if ( msg->reconfig ) {
    DebugM(3,"Reconfiguring\n");
    configure(msg->newaid, msg->newgdef, msg->newgridobjid);
  }

  // send another round of data if requested

  if(msg->resendCoordinates) {
    DebugM(3,"Sending requested data right away\n");
    //    CkPrintf("*** Resending data on PE %d \n", CkMyPe());
    sendData();
  }

  groupTotalForce.resize(numGroupsRequested);
  for ( int i=0; i<numGroupsRequested; ++i ) groupTotalForce[i] = 0;
  DebugM(3,"resized\n");
  if(setForces) {
    DebugM(3,"setting forces\n");    
    ResizeArrayIter<PatchElem> ap(patchList);
    Force **f = forcePtrs;
    FullAtom **t = atomPtrs;
    for (ap = ap.begin(); ap != ap.end(); ap++) {
      CompAtom *x;
      PatchID pId = (*ap).patchID;
      if (!sp->CUDASOAintegrate) {
        (*ap).positionBox->close(&x);
        (*ap).forceBox->close(&((*ap).r));
	 DebugM(3,"closing boxes\n");    
      }
      f[pId] = 0;
      t[pId] = 0;
      if (sp->CUDASOAintegrate || sp->SOAintegrateOn) {
        // XXX Possibly code below is needed by SOAintegrate mode
        mass_soa[pId] = NULL;
        pos_soa_x[pId] = NULL;
        pos_soa_y[pId] = NULL;
        pos_soa_z[pId] = NULL;
        force_soa_x[pId] = NULL;
        force_soa_y[pId] = NULL;
        force_soa_z[pId] = NULL;
        transform_soa_i[pId] = NULL;
        transform_soa_j[pId] = NULL;
        transform_soa_k[pId] = NULL;
	DebugM(3,"nulling ptrs\n");    
      }
    }
    DebugM(3,"done setting forces\n");    
  }

  #ifdef NODEGROUP_FORCE_REGISTER
  if (!sp->CUDASOAintegrate) {
    // CUDASOAintegrate handles this on PE 0 in sendComputeGlobalResults
    delete msg;
  }
  #else
  delete msg;
  #endif
  DebugM(3,"Done processing results\n");
}

void ComputeGlobal::doWork()
{
  DebugM(2,"doWork thread " << CthGetToken(CthSelf())->serialNo << "\n");

  SimParameters *sp = Node::Object()->simParameters;
  ResizeArrayIter<PatchElem> ap(patchList);
  FullAtom **t = atomPtrs;

  // if(sp->CUDASOAintegrateOn) {
  //   hasPatchZero = 0;
  // }

  for (ap = ap.begin(); ap != ap.end(); ap++) {
    CompAtom *x = (*ap).positionBox->open();
    t[(*ap).patchID] = (*ap).p->getAtomList().begin();

    if (sp->SOAintegrateOn || sp->CUDASOAintegrateMode) {
      // Assigne the pointer to SOA data structure
      PatchID pId = (*ap).patchID;
      mass_soa[pId] = (*ap).p->patchDataSOA.mass;
      pos_soa_x[pId] = (*ap).p->patchDataSOA.pos_x;
      pos_soa_y[pId] = (*ap).p->patchDataSOA.pos_y;
      pos_soa_z[pId] = (*ap).p->patchDataSOA.pos_z;
      transform_soa_i[pId] = (*ap).p->patchDataSOA.transform_i;
      transform_soa_j[pId] = (*ap).p->patchDataSOA.transform_j;
      transform_soa_k[pId] = (*ap).p->patchDataSOA.transform_k;
      // if(sp->CUDASOAintegrateOn && (pId == 0)) {
      //   hasPatchZero = 1;
      // }
    }
  }

  if(!firsttime) {
    //    CkPrintf("*** Start NoFirstTime on PE %d \n", CkMyPe());
    sendData();
    //    CkPrintf("*** End NoFirstTime on PE %d \n", CkMyPe());
  } else {
    //    CkPrintf("*** Start FirstTime on PE %d \n", CkMyPe());
    if ( hasPatchZero ) {
      ComputeGlobalDataMsg *msg = new ComputeGlobalDataMsg;
      msg->lat.add(patchList[0].p->lattice);
      msg->step = -1;
      msg->count = 1;
      msg->patchcount = 0;
      //      CkPrintf("***DoWork calling sendComputeGlobalData PE %d \n", CkMyPe());
      comm->sendComputeGlobalData(msg);
    }
    #ifdef NODEGROUP_FORCE_REGISTER
    else if (sp->CUDASOAintegrate) {

      //      CkPrintf("***DoWork FirstTime barrier 1 on PE %d \n", CkMyPe());
      comm->stowSuspendULT();
      //      CmiNodeBarrier();
      //      CkPrintf("***DoWork FirstTime barrier 2 on PE %d \n", CkMyPe());
      comm->stowSuspendULT();
      //      CkPrintf("***DoWork out of barrier 2 on PE %d \n", CkMyPe());
      //      CmiNodeBarrier();
      ComputeGlobalResultsMsg* resultsMsg = CkpvAccess(ComputeGlobalResultsMsg_instance);
      //      CkPrintf("*** ComputeGlobal::doWork PE (%d) calling recvComputeGlobalResults in doWork at step: %d \n",CkMyPe(), patchList[0].p->flags.step);
      comm->recvComputeGlobalResults(resultsMsg);
    }
    #endif // NODEGROUP_FORCE_REGISTER
    firsttime = 0;
    //    CkPrintf("*** ComputeGlobal::doWork PE (%d) calling enableComputeGlobalResults in doWork at step: %d \n",CkMyPe(), patchList[0].p->flags.step);
    comm->enableComputeGlobalResults();

    //    CkPrintf("*** End FirstTime on PE %d \n", CkMyPe());
  }
  DebugM(2,"done with doWork\n");
}

void ComputeGlobal::sendData()
{
  DebugM(2,"sendData\n");
  // Get positions from patches
  AtomMap *atomMap = AtomMap::Object();
  const Lattice & lattice = patchList[0].p->lattice;
  ResizeArrayIter<PatchElem> ap(patchList);
  FullAtom **t = atomPtrs;

  ComputeGlobalDataMsg *msg = new  ComputeGlobalDataMsg;
  SimParameters *sp = Node::Object()->simParameters;

  msg->step = patchList[0].p->flags.step;
  msg->count = 0;
  msg->patchcount = 0;

  // CkPrintf("*** PE (%d) Start sending at step: %d \n", 
  //	     CkMyPe(), patchList[0].p->flags.step);
  AtomIDList::iterator a = aid.begin();
  AtomIDList::iterator a_e = aid.end();
  NAMD_EVENT_START(1, NamdProfileEvent::GM_MSGPADD);  
  if (sp->SOAintegrateOn || sp->CUDASOAintegrateMode) {
    LocalID localID;
    PatchID lpid;
    int lidx;
    Position x_orig;
    Transform trans;

    for ( ; a != a_e; ++a ) {
      localID = atomMap->localID(*a);
      lpid = localID.pid;
      lidx = localID.index;
      if ( lpid == notUsed || ! t[lpid] ) continue;
      msg->aid.add(*a);
      msg->count++;
      x_orig.x = pos_soa_x[lpid][lidx];
      x_orig.y = pos_soa_y[lpid][lidx];
      x_orig.z = pos_soa_z[lpid][lidx];
      trans.i = transform_soa_i[lpid][lidx];
      trans.j = transform_soa_j[lpid][lidx];
      trans.k = transform_soa_k[lpid][lidx];
      msg->p.add(lattice.reverse_transform(x_orig,trans));
      //      printf("NAMD3-send: step %d atom %d, POS (%8.6f, %8.6f, %8.6f) \n", patchList[0].p->flags.step, *a, x_orig.x, x_orig.y, x_orig.z);
    }
  } else {
    for ( ; a != a_e; ++a ) {
      LocalID localID = atomMap->localID(*a);
      if ( localID.pid == notUsed || ! t[localID.pid] ) continue;
      msg->aid.add(*a);
      msg->count++;
      FullAtom &atom = t[localID.pid][localID.index];
      Position x_orig = atom.position;
      Transform trans = atom.transform;
      msg->p.add(lattice.reverse_transform(x_orig,trans));
      //      printf("NAMD2-send: step %d atom %d, POS (%8.6f, %8.6f, %8.6f) \n", patchList[0].p->flags.step, *a, x_orig.x, x_orig.y, x_orig.z);
    }
  }
  NAMD_EVENT_STOP(1, NamdProfileEvent::GM_MSGPADD);  
  NAMD_EVENT_START(1, NamdProfileEvent::GM_GCOM);
  // calculate group centers of mass
  AtomIDList::iterator g_i, g_e;
  g_i = gdef.begin(); g_e = gdef.end();
  if (sp->SOAintegrateOn || sp->CUDASOAintegrateMode) {
    LocalID localID;
    PatchID lpid;
    int lidx;
    Position x_orig;
    Transform trans;
    for ( ; g_i != g_e; ++g_i ) {
      Vector com(0,0,0);
      BigReal mass = 0.;
      for ( ; *g_i != -1; ++g_i ) {
        localID = atomMap->localID(*g_i);
        lpid = localID.pid;
        lidx = localID.index;
        if ( lpid == notUsed || ! t[lpid] ) continue;
        msg->count++;
        x_orig.x = pos_soa_x[lpid][lidx];
        x_orig.y = pos_soa_y[lpid][lidx];
        x_orig.z = pos_soa_z[lpid][lidx];
        trans.i = transform_soa_i[lpid][lidx];
        trans.j = transform_soa_j[lpid][lidx];
        trans.k = transform_soa_k[lpid][lidx];
        com += lattice.reverse_transform(x_orig,trans) * mass_soa[lpid][lidx];
        mass += mass_soa[lpid][lidx];
#if 0
        if (*g_i < 20) {
	  printf("NAMD3-send: step %d atom %d, POS (%8.6f, %8.6f, %8.6f) \n", patchList[0].p->flags.step, *g_i, x_orig.x, x_orig.y, x_orig.z);
        }
#endif

      }
      //      CkPrintf("*** NAMD3-send (%d): step %d group %d, COM (%8.6f, %8.6f, %8.6f) \n", 
      //	       CkMyPe(), patchList[0].p->flags.step, *g_i, com.x, com.y, com.z);
      DebugM(1,"Adding center of mass "<<com<<"\n");
      NAMD_EVENT_START(1, NamdProfileEvent::GM_GCOMADD);      
      msg->gcom.add(com);
      msg->gmass.add(mass);
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_GCOMADD);      
    }
  } else {
    for ( ; g_i != g_e; ++g_i ) {
      Vector com(0,0,0);
      BigReal mass = 0.;
      for ( ; *g_i != -1; ++g_i ) {
        LocalID localID = atomMap->localID(*g_i);
        if ( localID.pid == notUsed || ! t[localID.pid] ) continue;
        msg->count++;
        FullAtom &atom = t[localID.pid][localID.index];
        Position x_orig = atom.position;
        Transform trans = atom.transform;
        com += lattice.reverse_transform(x_orig,trans) * atom.mass;
        mass += atom.mass;
#if 0	
        if (*g_i < 20) {
          printf("NAMD2-send: step %d atom %d, POS (%8.6f, %8.6f, %8.6f) \n", patchList[0].p->flags.step, *g_i, x_orig.x, x_orig.y, x_orig.z);
        }
#endif	
      }
      //      CkPrintf("*** NAMD2-send (%d): step %d group %d, COM (%8.6f, %8.6f, %8.6f) \n", 
      //	       CkMyPe(), patchList[0].p->flags.step, *g_i, com.x, com.y, com.z);

	    
      DebugM(1,"Adding center of mass "<<com<<"\n");
      NAMD_EVENT_START(1, NamdProfileEvent::GM_GCOMADD);      
      msg->gcom.add(com);
      msg->gmass.add(mass);
      NAMD_EVENT_STOP(1, NamdProfileEvent::GM_GCOMADD);      
    }
  }
  NAMD_EVENT_STOP(1, NamdProfileEvent::GM_GCOM);	    
//  printf("Finish sending at step: %d ####################################################\n", 
//      patchList[0].p->flags.step);
  if (numActiveGridObjects > 0) {
    computeGridObjects(msg);
  }

  msg->fid.swap(fid);
  msg->tf.swap(totalForce);
  fid.resize(0);
  totalForce.resize(0);

  if ( gfcount ) msg->gtf.swap(groupTotalForce);
  msg->count += ( msg->fid.size() + gfcount );
  gfcount = 0;

  DebugM(3,"Sending data (" << msg->p.size() << " positions, "
         << msg->gcom.size() << " groups, " << msg->gridobjvalue.size()
         << " grid objects) on client\n");
  if ( hasPatchZero ) { msg->count++;  msg->lat.add(lattice); }
  if ( msg->count || msg->patchcount )
    {
      //      CkPrintf("*** ComputeGlobal::sendData PE (%d) calling sendComputeGlobalData step: %d msg->count %d msg->patchcount %d\n", CkMyPe(), patchList[0].p->flags.step,msg->count, msg->patchcount);
      comm->sendComputeGlobalData(msg);
    }
  else
    {
      //      CkPrintf("*** ComputeGlobal::sendData PE (%d) skipping sendComputeGlobalData step: %d msg->count %d msg->patchcount %d\n", CkMyPe(), patchList[0].p->flags.step,msg->count, msg->patchcount);
      //      comm->sendComputeGlobalData(msg);
#ifdef NODEGROUP_FORCE_REGISTER
      // this PE doesn't have message work to do
      SimParameters *sp = Node::Object()->simParameters;
      if (sp->CUDASOAintegrate) {
	// we need to enter the barriers normally hit in sendComputeGlobalData
	//	CkPrintf("*** ComputeGlobal::sendData PE (%d) about to double stow\n");
	comm->stowSuspendULT();
	comm->stowSuspendULT();
	// and the one in sendComputeGlobalResults
	//	comm->stowSuspendULT();
      }
#endif      
      delete msg;
    }
  NAMD_EVENT_START(1, NamdProfileEvent::GM_GRESULTS);
  //  CkPrintf("*** ComputeGlobal::sendData PE (%d) calling enableComputeGlobalResults in sendData at step: %d \n",  CkMyPe(), patchList[0].p->flags.step);
  comm->enableComputeGlobalResults();
  NAMD_EVENT_STOP(1, NamdProfileEvent::GM_GRESULTS);	
}

template<class T> void ComputeGlobal::computeGridForceGrid(FullAtomList::iterator aii,
                                                           FullAtomList::iterator aei,
                                                           ForceList::iterator gfii,
                                                           Lattice const &lattice,
                                                           int gridIndex,
                                                           T *grid,
                                                           BigReal &gridObjValue)
{
  ForceList::iterator gfi = gfii;
  FullAtomList::iterator ai = aii;
  FullAtomList::iterator ae = aei;
  Molecule *mol = Node::Object()->molecule;
  for ( ; ai != ae; ai++, gfi++) {
    *gfi = Vector(0.0, 0.0, 0.0);
    if (! mol->is_atom_gridforced(ai->id, gridIndex)) {
      continue;
    }
    Real scale;
    Charge charge;
    Vector dV;
    float V;
    mol->get_gridfrc_params(scale, charge, ai->id, gridIndex);
    Position pos = grid->wrap_position(ai->position, lattice);
    DebugM(1, "id = " << ai->id << ", scale = " << scale
           << ", charge = " << charge << ", position = " << pos << "\n");
    if (grid->compute_VdV(pos, V, dV)) {
      // out-of-bounds atom
      continue;
    }
    // ignore global gfScale
    *gfi = -charge * scale * dV;
    gridObjValue += charge * scale * V;
    DebugM(1, "id = " << ai->id << ", force = " << *gfi << "\n");
  }
  DebugM(3, "gridObjValue = " << gridObjValue << "\n" << endi);
}

void ComputeGlobal::computeGridObjects(ComputeGlobalDataMsg *msg)
{
  DebugM(3,"computeGridObjects\n" << endi);
  Molecule *mol = Node::Object()->molecule;
  const Lattice &lattice = patchList[0].p->lattice;

  if (mol->numGridforceGrids < 1) {
    NAMD_bug("No grids loaded in memory but ComputeGlobal has been requested to use them.");
  }

  msg->gridobjindex.resize(numActiveGridObjects);
  msg->gridobjindex.setall(-1);
  msg->gridobjvalue.resize(numActiveGridObjects);
  msg->gridobjvalue.setall(0.0);

  size_t ig = 0, gridobjcount = 0;

  // loop over home patches
  ResizeArrayIter<PatchElem> ap(patchList);
  for (ap = ap.begin(); ap != ap.end(); ap++) {

    msg->patchcount++;

    int const numAtoms = ap->p->getNumAtoms();
    ForceList **gridForces = gridForcesPtrs[ap->p->getPatchID()];

    gridobjcount = 0;
    for (ig = 0; ig < numGridObjects; ig++) {

      DebugM(2,"Processing grid index " << ig << "\n" << endi);

      // Only process here objects requested by the GlobalMasters
      if (!gridObjActive[ig]) {
        DebugM(2,"Skipping grid index " << ig << "; it is handled by "
               "ComputeGridForce\n" << endi);
        continue;
      }

      ForceList *gridForcesGrid = gridForces[ig];
      gridForcesGrid->resize(numAtoms);

      ForceList::iterator gfi = gridForcesGrid->begin();
      FullAtomList::iterator ai = ap->p->getAtomList().begin();
      FullAtomList::iterator ae = ap->p->getAtomList().end();

      DebugM(2, "computeGridObjects(): patch = " << ap->p->getPatchID()
             << ", grid index = " << ig << "\n" << endi);
      GridforceGrid *grid = mol->get_gridfrc_grid(ig);

      msg->gridobjindex[gridobjcount] = ig;
      BigReal &gridobjvalue = msg->gridobjvalue[gridobjcount];

      if (grid->get_grid_type() == GridforceGrid::GridforceGridTypeFull) {

        GridforceFullMainGrid *g = dynamic_cast<GridforceFullMainGrid *>(grid);
        computeGridForceGrid(ai, ae, gfi, ap->p->lattice, ig, g, gridobjvalue);

      } else if (grid->get_grid_type() == GridforceGrid::GridforceGridTypeLite) {

        GridforceLiteGrid *g = dynamic_cast<GridforceLiteGrid *>(grid);
        computeGridForceGrid(ai, ae, gfi, ap->p->lattice, ig, g, gridobjvalue);
      }

      gridobjcount++;
    }
  }

  for (gridobjcount = 0; gridobjcount < numActiveGridObjects; gridobjcount++) {
    DebugM(3, "Total gridObjValue[" << msg->gridobjindex[gridobjcount]
           << "] = " << msg->gridobjvalue[gridobjcount] << "\n");
  }

  DebugM(2,"computeGridObjects done\n");
}

void ComputeGlobal::applyGridObjectForces(ComputeGlobalResultsMsg *msg,
                                          Force *extForce_in,
                                          Tensor *extVirial_in)
{
  if (msg->gridobjforce.size() == 0) return;

  if (msg->gridobjforce.size() != numActiveGridObjects) {
    NAMD_bug("ComputeGlobal received a different number of grid forces than active grids.");
  }

  Molecule *mol = Node::Object()->molecule;
  const Lattice &lattice = patchList[0].p->lattice;
  AtomMap *atomMap = AtomMap::Object();
  Force &extForce = *extForce_in;
  Tensor &extVirial = *extVirial_in;

  // map applied forces from the message
  BigRealList gridObjForces;
  gridObjForces.resize(numGridObjects);
  gridObjForces.setall(0.0);
  BigRealList::iterator gridobjforce_i = msg->gridobjforce.begin();
  BigRealList::iterator gridobjforce_e = msg->gridobjforce.end();
  int ig;
  for (ig = 0; gridobjforce_i != gridobjforce_e ;
       gridobjforce_i++, ig++) {
    if (!gridObjActive[ig]) continue;
    gridObjForces[ig] = *gridobjforce_i;
  }

  // loop over home patches
  ResizeArrayIter<PatchElem> ap(patchList);
  for (ap = ap.begin(); ap != ap.end(); ap++) {

    ForceList **gridForces = gridForcesPtrs[ap->p->getPatchID()];

    for (ig = 0; ig < numGridObjects; ig++) {

      if (!gridObjActive[ig]) continue;

      DebugM(2, "gof  = " << gridObjForces[ig] << "\n" << endi);

      ForceList *gridForcesGrid = gridForces[ig];

      FullAtomList::iterator ai = ap->p->getAtomList().begin();
      FullAtomList::iterator ae = ap->p->getAtomList().end();
      Force *f = ap->r->f[Results::normal];
      ForceList::iterator gfi = gridForcesGrid->begin();

      for ( ; ai != ae; ai++, gfi++) {
        if (! mol->is_atom_gridforced(ai->id, ig)) {
          *gfi = Vector(0.0, 0.0, 0.0);
          continue;
        }
	LocalID localID = atomMap->localID(ai->id);
        // forces were stored; flipping sign to get gradients
        Vector const gridforceatom(-1.0 * (*gfi) * gridObjForces[ig]);
        DebugM(2, "id = " << ai->id
               << ", pid = " << localID.pid
               << ", index = " << localID.index
               << ", force = " << gridforceatom << "\n" << endi);
        f[localID.index] += gridforceatom;
        extForce += gridforceatom;
        Position x_orig = ai->position;
        Transform transform = ai->transform;
        Position x_virial = lattice.reverse_transform(x_orig, transform);
        extVirial += outer(gridforceatom, x_virial);
      }
    }
  }
  // extForce and extVirial are being communicated by calling function
}

// This function is called by each HomePatch after force
// evaluation. It stores the indices and forces of the requested
// atoms here, to be sent to GlobalMasterServer during the next
// time step. The total force is the sum of three components:
// "normal", "nbond" and "slow", the latter two may be calculated
// less frequently, so their most recent values are stored in
// "f_saved" and used here. If we don't do full electrostatics,
// there's no "slow" part.
void ComputeGlobal::saveTotalForces(HomePatch *homePatch)
{
  if ( ! forceSendEnabled ) NAMD_bug("ComputeGlobal::saveTotalForces called unexpectedly");
  //if ( ! forceSendActive ) return;

  SimParameters *simParms = Node::Object()->simParameters;
  if ( simParms->accelMDOn && simParms->accelMDDebugOn && simParms->accelMDdihe ) {
    int num=homePatch->numAtoms;
    FullAtomList &atoms = homePatch->atom;
    ForceList &af=homePatch->f[Results::amdf];

    for (int i=0; i<num; ++i) {
      int index = atoms[i].id;
      if (index < endRequested && isRequested[index] & 1) {
        fid.add(index);
        totalForce.add(af[i]);
      }
    }
    return;
  }

  //  printf("Start saving force at step: %d ####################################################\n", 
  //        patchList[0].p->flags.step);
  int fixedAtomsOn = simParms->fixedAtomsOn;
  int num=homePatch->numAtoms;
  FullAtomList &atoms = homePatch->atom;
  ForceList &f1=homePatch->f[Results::normal], &f2=homePatch->f_saved[Results::nbond],
            &f3=homePatch->f_saved[Results::slow];
  
  double *f1_soa_x = homePatch->patchDataSOA.f_normal_x;
  double *f1_soa_y = homePatch->patchDataSOA.f_normal_y;
  double *f1_soa_z = homePatch->patchDataSOA.f_normal_z;
  double *f2_soa_x = homePatch->patchDataSOA.f_saved_nbond_x;
  double *f2_soa_y = homePatch->patchDataSOA.f_saved_nbond_y;
  double *f2_soa_z = homePatch->patchDataSOA.f_saved_nbond_z;
  double *f3_soa_x = homePatch->patchDataSOA.f_saved_slow_x;
  double *f3_soa_y = homePatch->patchDataSOA.f_saved_slow_y;
  double *f3_soa_z = homePatch->patchDataSOA.f_saved_slow_z;
  int hasSOA = (simParms->SOAintegrateOn || simParms->CUDASOAintegrateMode);
  Force f_sum;
  double f_sum_x, f_sum_y, f_sum_z;
  
  #if 0
  for (int i=0; i<num; ++i) {
    int index = atoms[i].id;
    if (index < 20) {
      if (hasSOA) {
        CkPrintf("ForceSaved: atom %d, ForceN  (%8.6f, %8.6f, %8.6f) \n", index, f1_soa_x[i], f1_soa_y[i], f1_soa_z[i]);
        CkPrintf("            atom %d, ForceNB (%8.6f, %8.6f, %8.6f) \n", index, f2_soa_x[i], f2_soa_y[i], f2_soa_z[i]);
        CkPrintf("            atom %d, ForceSL (%8.6f, %8.6f, %8.6f) \n", index, f3_soa_x[i], f3_soa_y[i], f3_soa_z[i]);
      } else {
        CkPrintf("ForceSaved: atom %d, ForceN  (%8.6f, %8.6f, %8.6f) \n", index, f1[i].x, f1[i].y, f1[i].z);
        CkPrintf("            atom %d, ForceNB (%8.6f, %8.6f, %8.6f) \n", index, f2[i].x, f2[i].y, f2[i].z);
	// not memory safe to access slow forces all the time like this
	//        CkPrintf("            atom %d, ForceSL (%8.6f, %8.6f, %8.6f) \n", index, f3[i].x, f3[i].y, f3[i].z);
      }
    }
  }

  printf("PE, PId (%d, %d) Stop saving at step: %d ####################################################\n", 
    CkMyPe(), homePatch->patchID, patchList[0].p->flags.step);
  #endif
  if ( ! forceSendActive ) return;
  for (int i=0; i<num; ++i) {
    int index = atoms[i].id;
    char reqflag;
    if (index < endRequested && (reqflag = isRequested[index])) {
      if (hasSOA) {
        f_sum_x = f1_soa_x[i] + f2_soa_x[i];
        f_sum_y = f1_soa_y[i] + f2_soa_y[i];
        f_sum_z = f1_soa_z[i] + f2_soa_z[i];
        if (dofull) {
          f_sum_x += f3_soa_x[i];
          f_sum_y += f3_soa_y[i];
          f_sum_z += f3_soa_z[i];
        }
        f_sum.x = f_sum_x;
        f_sum.y = f_sum_y;
        f_sum.z = f_sum_z;
      } else {
        f_sum = f1[i]+f2[i];
        if (dofull)
          f_sum += f3[i];
      }

      if ( fixedAtomsOn && atoms[i].atomFixed )
        f_sum = 0.;
      
      if ( reqflag  & 1 ) {  // individual atom
        fid.add(index);
        totalForce.add(f_sum);
      }
      if ( reqflag  & 2 ) {  // part of group
        intpair *gpend = gpair.end();
        intpair *gpi = std::lower_bound(gpair.begin(),gpend,intpair(index,0));
        if ( gpi == gpend || gpi->first != index )
          NAMD_bug("ComputeGlobal::saveTotalForces gpair corrupted.");
        do {
          ++gfcount;
          groupTotalForce[gpi->second] += f_sum;
        } while ( ++gpi != gpend && gpi->first == index );
      }
    }
  }
}
