/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef PATCHTYPES_H
#define PATCHTYPES_H

#include "NamdTypes.h"
#include "Lattice.h"

class Flags
{
public:
  int step;			// timestep number reported to user
  				// Same number may appear multiple times!
  int sequence;			// sequence number of compute call
				// changes by 1 every time!
  int doEnergy;
  int doVirial;
  int doNonbonded;
  int doFullElectrostatics;
  int doMolly;
  int doMinimize;
  // BEGIN LA
  int doLoweAndersen;
  // END LA
  int doGBIS;// gbis
  int doLCPO;//LCPO
  int submitLoadStats;
  int maxForceUsed;		// may ignore slower force classes
  int maxForceMerged;		// add this and faster to normal

#if defined(NAMD_NVTX_ENABLED) || defined(NAMD_CMK_TRACE_ENABLED) || defined(NAMD_ROCTX_ENABLED)
  int event_on;  // true or false to control NVTX profiling
#endif

  int usePairlists;
  int savePairlists;
  BigReal pairlistTolerance;
  BigReal maxAtomMovement;
  BigReal maxGroupRadius;

  Lattice lattice;		// rather than shipping around separately

  Flags() :
    step(0),
    sequence(0),
    doEnergy(0),
    doVirial(0),
    doNonbonded(0),
    doFullElectrostatics(0),
    doMolly(0),
    doMinimize(0),
    doLoweAndersen(0),
    doGBIS(0),
    doLCPO(0),
    submitLoadStats(0),
    maxForceUsed(0),
    maxForceMerged(0),
#if defined(NAMD_NVTX_ENABLED) || defined(NAMD_CMK_TRACE_ENABLED) || defined(NAMD_ROCTX_ENABLED)
    event_on(0),
#endif
    usePairlists(0),
    savePairlists(0),
    pairlistTolerance(0),
    maxAtomMovement(0),
    maxGroupRadius(0)
  { }

  void copyIntFlags(const Flags &flags) {
    step = flags.step;
    sequence = flags.sequence;
    doEnergy = flags.doEnergy;
    doVirial = flags.doVirial;
    doNonbonded = flags.doNonbonded;
    doFullElectrostatics = flags.doFullElectrostatics;
    doMolly = flags.doMolly;
    doMinimize = flags.doMinimize;
    doLoweAndersen = flags.doLoweAndersen;
    doGBIS = flags.doGBIS;
    doLCPO = flags.doLCPO;
    submitLoadStats = flags.submitLoadStats;
    maxForceUsed = flags.maxForceUsed;
    maxForceMerged = flags.maxForceMerged;
    lattice = flags.lattice;
#if defined(NAMD_NVTX_ENABLED) || defined(NAMD_CMK_TRACE_ENABLED) || defined(NAMD_ROCTX_ENABLED)
    event_on = flags.event_on;
#endif
  //  usePairlists = flags.usePairlists;
  //  savePairlists = flags.savePairlists;
  }

  inline Flags& operator= (const Flags &flags) 
    {
      CmiMemcpy (this, &flags, sizeof(Flags));
      return *this;
    }

#if 0
  void print() const {
    fprintf(stderr, "  step = %d\n", step);
    fprintf(stderr, "  sequence = %d\n", sequence);
    fprintf(stderr, "  doEnergy = %d\n", doEnergy);
    fprintf(stderr, "  doVirial = %d\n", doVirial);
    fprintf(stderr, "  doNonbonded = %d\n", doNonbonded);
    fprintf(stderr, "  doFullElectrostatics = %d\n", doFullElectrostatics);
    fprintf(stderr, "  doMolly = %d\n", doMolly);
    fprintf(stderr, "  doLoweAndersen = %d\n", doLoweAndersen);
    fprintf(stderr, "  doGBIS = %d\n", doGBIS);
    fprintf(stderr, "  doLCPO = %d\n", doLCPO);
    fprintf(stderr, "  submitLoadStats = %d\n", submitLoadStats);
    fprintf(stderr, "  maxForceUsed = %d\n", maxForceUsed);
    fprintf(stderr, "  maxForceMerged = %d\n", maxForceMerged);
#if defined(NAMD_NVTX_ENABLED) || defined(NAMD_CMK_TRACE_ENABLED)
    fprintf(stderr, "  event_on = %d\n", event_on);
#endif
    fprintf(stderr, "  usePairLists = %d\n", usePairlists);
    fprintf(stderr, "  savePairLists = %d\n", savePairlists);
    fprintf(stderr, "  pairlistTolerance = %g\n", pairlistTolerance);
    fprintf(stderr, "  maxAtomMovement = %g\n", maxAtomMovement);
    fprintf(stderr, "  maxGroupRadius = %g\n", maxGroupRadius);
    fprintf(stderr, "  lattice =\n");
    fprintf(stderr, "    a.x=%g  a.y=%g  a.z=%g\n",
        lattice.a().x, lattice.a().y, lattice.a().z);
    fprintf(stderr, "    b.x=%g  b.y=%g  b.z=%g\n",
        lattice.b().x, lattice.b().y, lattice.b().z);
    fprintf(stderr, "    c.x=%g  c.y=%g  c.z=%g\n",
        lattice.c().x, lattice.c().y, lattice.c().z);
    fprintf(stderr, "    o.x=%g  o.y=%g  o.z=%g\n",
        lattice.origin().x, lattice.origin().y, lattice.origin().z);
  }
#endif
};

class Results
{
public:
  enum { normal=0, nbond=1, slow=2, amdf=3,
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
    maxNumForces=4 };
#else
    nbond_virial=4, slow_virial=5, maxNumForces=6 };
#endif
  Force *f[maxNumForces];
};

#endif

