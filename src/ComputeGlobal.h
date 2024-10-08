/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/*
   Forwards atoms to master node for force evaluation.
*/

#ifndef COMPUTEGLOBAL_H
#define COMPUTEGLOBAL_H

#include "ComputeHomePatches.h"
#include "NamdTypes.h"

class ComputeGlobalConfigMsg;
class ComputeGlobalDataMsg;
class ComputeGlobalResultsMsg;
class ComputeMgr;
class SubmitReduction;
class NodeReduction;

struct intpair {
  int first, second;
  intpair() {;}
  intpair(int f, int s) : first(f), second(s) {;}
};

inline bool operator<(const intpair &lhs, const intpair &rhs) {
  return lhs.first < rhs.first ? true :
         lhs.first != rhs.first ? false :
         lhs.second < rhs.second;
}

class ComputeGlobal : public ComputeHomePatches {
public:
  ComputeGlobal(ComputeID, ComputeMgr*);
  virtual ~ComputeGlobal();
  void doWork();
  // void recvConfig(ComputeGlobalConfigMsg *);
  void recvResults(ComputeGlobalResultsMsg *);
  // For "loadtotalforces" TCL command
  void saveTotalForces(HomePatch *);

private:
  ComputeMgr *comm;

  void sendData();
  void configure(AtomIDList &newaid, AtomIDList &newgdef, IntList &newgridobjid);

  AtomIDList aid;
  AtomIDList gdef;  // definitions of groups
  ResizeArray<intpair> gpair;
  
  // (For "loadtotalforces" TCL command)
  // The atom IDs and forces of the requested atoms on the node
  // after force evaluation. "fid" could be slightly different
  // from "aid", since the latter is after atom migration.
  AtomIDList fid;
  ForceList totalForce;
  ForceList groupTotalForce;
  int numGroupsRequested;

  Force **forcePtrs;
  FullAtom **atomPtrs;
  int **transform_soa_i; /**< SOA data for x element of transformation (wrap and unwrap) */
  int **transform_soa_j; /**< SOA data for y element of transformation (wrap and unwrap) */
  int **transform_soa_k; /**< SOA data for z element of transformation (wrap and unwrap) */
  float **mass_soa;      /**< SOA data for mass */
  double **pos_soa_x;    /**< SOA data for x coordinate */
  double **pos_soa_y;    /**< SOA data for y coordinate */
  double **pos_soa_z;    /**< SOA data for z coordinate */
  double **force_soa_x;  /**< SOA data for x element of force */
  double **force_soa_y;  /**< SOA data for y element of force */
  double **force_soa_z;  /**< SOA data for z element of force */

  /// Number of GridForceGrid object defined for this simulation
  size_t numGridObjects;

  /// Flags: 1 = processed by ComputeGlobal, 0 = processed by ComputeGridForce
  /// The length of this list is numGridObjects
  IntList gridObjActive;

  /// Sum of the elements of gridObjActive
  size_t numActiveGridObjects;

  /// Atomic gradients (and then forces) of the global grid objects
  ForceList ***gridForcesPtrs;

  void configureGridObjects(IntList &newgridobjid);
  void deleteGridObjects();
  void computeGridObjects(ComputeGlobalDataMsg *msg);
  /// Reimplementation of equivalent function from ComputeGridForce
  template<class T> void computeGridForceGrid(FullAtomList::iterator aii,
                                              FullAtomList::iterator aei,
                                              ForceList::iterator fii,
                                              Lattice const &lattice,
                                              int gridIndex,
                                              T *grid,
                                              BigReal &gridObjValue);
  void applyGridObjectForces(ComputeGlobalResultsMsg *msg,
                             Force *extForce, Tensor *extVirial);
  
  #ifdef NODEGROUP_FORCE_REGISTER
  NodeReduction *nodeReduction; /**< To add thermodynamic properties */
  #endif

  int forceSendEnabled; // are total forces received?
  int forceSendActive; // are total forces received this step?
  int gfcount;  // count of atoms contributing to group forces
  char *isRequested;  // whether this atom is requested by the TCL script
  int isRequestedAllocSize;  // size of array
  int endRequested;  // starting at this point assume not requested
  int dofull;  // whether "Results::slow" force will exist

  int firsttime;
  SubmitReduction *reduction;
};

#endif

