/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#include "Node.h"
#include "Molecule.h"
#include "NamdTypes.h"
#include "GlobalMaster.h"
#include "NamdEventsProfiling.h"

//#define DEBUGM
#define MIN_DEBUG_LEVEL 1
#include "Debug.h"

void GlobalMaster::processData(AtomIDList::iterator a_i,
                               AtomIDList::iterator a_e,
                               PositionList::iterator p_i,
                               PositionList::iterator g_i,
                               PositionList::iterator g_e,
                               BigRealList::iterator gm_i,
                               BigRealList::iterator gm_e,
                               ForceList::iterator gtf_i,
                               ForceList::iterator gtf_e,
                               IntList::iterator goi_i,
                               IntList::iterator goi_e,
                               BigRealList::iterator gov_i,
                               BigRealList::iterator gov_e,
                               AtomIDList::iterator last_atoms_forced_i,
                               AtomIDList::iterator last_atoms_forced_e,
                               ForceList::iterator last_forces_i,
                               AtomIDList::iterator forceid_i,
                               AtomIDList::iterator forceid_e,
                               ForceList::iterator totalforce_i) {
  atomIdBegin = a_i;
  atomIdEnd = a_e;
  atomPositionBegin = p_i;
  groupPositionBegin = g_i;
  groupPositionEnd = g_e;
  groupMassBegin = gm_i;
  groupMassEnd = gm_e;
  groupTotalForceBegin = gtf_i;
  groupTotalForceEnd = gtf_e;
  gridObjIndexBegin = goi_i;
  gridObjIndexEnd = goi_e;
  gridObjValueBegin = gov_i;
  gridObjValueEnd = gov_e;
  lastAtomsForcedBegin = last_atoms_forced_i;
  lastAtomsForcedEnd = last_atoms_forced_e;
  lastForcesBegin = last_forces_i;
  forceIdBegin = forceid_i;
  forceIdEnd = forceid_e;
  totalForceBegin = totalforce_i;

  calculate();

  /* check to make sure the force arrays still match */
  if(appForcesChanged) {
    check();
  }
}

void GlobalMaster::check() const {
  /* check to make sure the force arrays still match */
  if(fAtoms.size() != appForces.size())
    NAMD_die("# of atoms forced != # of forces given");
  if(grpForces.size() != groupMassEnd - groupMassBegin)
    NAMD_die("# of groups forced != # of groups requested");
  if(gridobjForces.size() != reqGridObjs.size())
    NAMD_die("# of grid objects forced != # of grid objects requested");
}

void GlobalMaster::clearChanged() {
  reqAtomsChanged = false;
  appForcesChanged = false;
  reqGroupsChanged = false;
  reqGridObjsChanged = false;
}

void GlobalMaster::calculate() {
  NAMD_die("Internal error: pure virtual function called");
}

GlobalMaster::GlobalMaster() {
  step = -1;
  old_num_groups_requested = 0;
  clearChanged();
  atomIdBegin = 0;
  atomIdEnd = 0;
  atomPositionBegin = 0;
  groupPositionBegin = 0;
  groupPositionEnd = 0;
  groupMassBegin = 0;
  groupMassEnd = 0;
  gridObjValueBegin = 0;
  gridObjValueEnd = 0;
  lastAtomsForcedBegin = 0;
  lastAtomsForcedEnd = 0;
  lastForcesBegin = 0;
  forceIdBegin = 0;
  forceIdEnd = 0;
  totalForceBegin = 0;
  lattice = 0;
  totalForceRequested = false;
}

bool GlobalMaster::changedAtoms() {
  return reqAtomsChanged;
}

bool GlobalMaster::changedForces() {
  return appForcesChanged;
}

bool GlobalMaster::changedGroups() {
  return reqGroupsChanged;
}

bool GlobalMaster::changedGridObjs() {
  return reqGridObjsChanged;
}

const AtomIDList &GlobalMaster::requestedAtoms() {
  return reqAtoms;
}

AtomIDList &GlobalMaster::modifyRequestedAtoms() {
  reqAtomsChanged = true;
  DebugM(3,"modifyRequestedAtoms()\n" << endi);
  return reqAtoms;
}

const AtomIDList &GlobalMaster::forcedAtoms() {
  return fAtoms;
}

const ForceList &GlobalMaster::appliedForces() {
  return appForces;
}

const ForceList &GlobalMaster::groupForces() {
  return grpForces;
}

const BigRealList &GlobalMaster::gridObjForces() {
  return gridobjForces;
}

const ResizeArray<AtomIDList> &GlobalMaster::requestedGroups() {
  return reqGroups;
}

const IntList &GlobalMaster::requestedGridObjs() {
  return reqGridObjs;
}

AtomIDList &GlobalMaster::modifyForcedAtoms() {
  appForcesChanged = true;
  return fAtoms;
}

ForceList &GlobalMaster::modifyAppliedForces() {
  appForcesChanged = true;
  return appForces;
}

ForceList &GlobalMaster::modifyGroupForces() {
  // XXX should we mark something else here?
  appForcesChanged = true;
  return grpForces;
}

IntList &GlobalMaster::modifyRequestedGridObjects() {
  reqGridObjsChanged = true;
  DebugM(3,"modifyRequestedGridObjects()\n" << endi);
  return reqGridObjs;
}

BigRealList &GlobalMaster::modifyGridObjForces() {
  appForcesChanged = true;
  return gridobjForces;
}

ResizeArray<AtomIDList> &GlobalMaster::modifyRequestedGroups() {
  reqGroupsChanged = true;
  DebugM(1,"Groups have changed.\n");
  return reqGroups;
}

AtomIDList::const_iterator GlobalMaster::getAtomIdBegin() {
  return atomIdBegin;
}

AtomIDList::const_iterator GlobalMaster::getAtomIdEnd() {
  return atomIdEnd;
}

PositionList::const_iterator GlobalMaster::getAtomPositionBegin() {
  return atomPositionBegin;
}

PositionList::const_iterator GlobalMaster::getGroupPositionBegin() {
  return groupPositionBegin;
}

PositionList::const_iterator GlobalMaster::getGroupPositionEnd() {
  return groupPositionEnd;
}

ForceList::const_iterator GlobalMaster::getGroupTotalForceBegin() {
  return groupTotalForceBegin;
}

ForceList::const_iterator GlobalMaster::getGroupTotalForceEnd() {
  return groupTotalForceEnd;
}

IntList::const_iterator GlobalMaster::getGridObjIndexBegin() {
  return gridObjIndexBegin;
}

IntList::const_iterator GlobalMaster::getGridObjIndexEnd() {
  return gridObjIndexEnd;
}

BigRealList::const_iterator GlobalMaster::getGridObjValueBegin() {
  return gridObjValueBegin;
}

BigRealList::const_iterator GlobalMaster::getGridObjValueEnd() {
  return gridObjValueEnd;
}

BigRealList::const_iterator GlobalMaster::getGroupMassBegin()
{
  return groupMassBegin;
}

BigRealList::const_iterator GlobalMaster::getGroupMassEnd() {
  return groupMassEnd;
}

AtomIDList::const_iterator GlobalMaster::getLastAtomsForcedBegin() {
  return lastAtomsForcedBegin;
}

AtomIDList::const_iterator GlobalMaster::getLastAtomsForcedEnd() {
  return lastAtomsForcedEnd;
}

ForceList::const_iterator GlobalMaster::getLastForcesBegin() {
  return lastForcesBegin;
}

AtomIDList::const_iterator GlobalMaster::getForceIdBegin()
{
  return forceIdBegin;
}

AtomIDList::const_iterator GlobalMaster::getForceIdEnd()
{
  return forceIdEnd;
}

ForceList::const_iterator GlobalMaster::getTotalForce()
{
  return totalForceBegin;
}
