/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/*
   common definitions for namd.
*/

#ifndef COMMON_H
#define COMMON_H

#ifndef SEQUENCER_SOA
#define SEQUENCER_SOA
#endif

// Check all bonds that have atom IDs in the range given below.
#if defined(DEBUG_PROTOCELL)
// Use Noah's default index range unless others are explicitly provided.
#ifndef PRMIN
#define PRMIN  23749759
#endif
#ifndef PRMAX
#define PRMAX  23749892
#endif
#endif

#if !defined(WIN32) || defined(__CYGWIN__)
#include <unistd.h>
#endif
#include <stdio.h>
#include <climits>

#include <cstdint>
typedef int8_t   int8;
typedef int16_t  int16;
typedef int32_t  int32;
typedef int64_t  int64;
typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

#define NAMD_FILENAME_BUFFER_SIZE 4096

#if defined(PLACEMENT_NEW)
void * ::operator new (size_t, void *p) { return p; }
#elif defined(PLACEMENT_NEW_GLOBAL)
void * operator new (size_t, void *p) { return p; }
#endif

#define COULOMB 332.0636
#define BOLTZMANN 0.001987191
#define TIMEFACTOR 48.88821
#define PRESSUREFACTOR 6.95E4
#define PDBVELFACTOR 20.45482706
#define PDBVELINVFACTOR (1.0/PDBVELFACTOR)
#define PNPERKCALMOL 69.479

#define RECIP_TIMEFACTOR  (1.0 / TIMEFACTOR)

//
// Defining macro namdnearbyint(X).
//
// Some plagtforms don't have nearbyint or round, so we'll define one
// that works everywhere.
//
// Use namdnearbyint(X) instead of rint(X) because rint() is sensitive
// to the current rounding mode and floor() is not.  It's just safer.
//
#ifdef ARCH_POWERPC
#ifdef POWERPC_TANINT
// round for BlueGeneQ (and others that set POWERPC_TANINT)
extern "builtin" double __tanint(double); // IEEE round
#define namdnearbyint(x)  __tanint(x)
#else
// round for Linux POWER
#include <builtins.h>
#include <tgmath.h>
#define namdnearbyint(x)  (round(x))
#endif
#else
// fall back should work everywhere
#define namdnearbyint(x)  floor((x)+0.5)
#endif
//
// End defining macro namdnearbyint(X).
//

#ifndef PI
#define PI	3.141592653589793
#endif

#ifndef TWOPI
#define TWOPI	2.0 * PI
#endif

#ifndef ONE
#define ONE	1.000000000000000
#endif

#ifndef ZERO
#define ZERO	0.000000000000000
#endif

#ifndef SMALLRAD
#define SMALLRAD      0.0005
#endif

#ifndef SMALLRAD2
#define SMALLRAD2     SMALLRAD*SMALLRAD
#endif

/* Define the size for Real and BigReal.  Real is usually mapped to float */
/* and BigReal to double.  To get BigReal mapped to float, use the 	  */
/* -DSHORTREALS compile time option					  */
typedef float	Real;

#ifdef SHORTREALS
typedef float	BigReal;
#else
typedef double  BigReal;
#endif

#ifndef FALSE
#define FALSE 0
#define TRUE 1
#endif

#ifndef NO
#define NO 0
#define YES 1
#endif

#ifndef STRINGNULL
#define STRINGNULL '\0'
#endif

#define MAX_NEIGHBORS 27

typedef int Bool;

class Communicate;

// provide NAMD version in similar way as done for Tcl
enum NAMD_ReleaseType {
  NAMD_UNKNOWN_RELEASE = 0,
  NAMD_ALPHA_RELEASE   = 1,
  NAMD_BETA_RELEASE    = 2,
  NAMD_FINAL_RELEASE   = 3
};
// pass address of int variables or NULL to disregard
// releaseType gets set to one of the above enum constants
void NAMD_version(int *major, int *minor, int *patchLevel, int *releaseType);
// obtain a pointer to the raw version string
const char *NAMD_version_string(void);

// global functions
void NAMD_quit(const char *);
void NAMD_die(const char *);
void NAMD_err(const char *);  // also prints strerror(errno)
void NAMD_bug(const char *);
int NAMD_file_exists(const char *filename);
void NAMD_backup_file(const char *filename, const char *extension = 0);
int NAMD_open_text(const char *fname, int append=0);
void NAMD_write(int fd, const char *buf, size_t count, const char *fname = "in NAMD_write()"); // NAMD_die on error
void NAMD_close(int fd, const char *fname);
char *NAMD_stringdup(const char *);
FILE *Fopen(const char *filename, const char *mode);
int  Fclose(FILE *fout);

//Math functions 
int NAMD_gcd(int a, int b);

// message tags
#define SIMPARAMSTAG	100	//  Tag for SimParameters class
#define STATICPARAMSTAG 101	//  Tag for Parameters class
#define MOLECULETAG	102	//  Tag for Molecule class
#define FULLTAG	104
#define FULLFORCETAG 105
#define DPMTATAG 106
#define GRIDFORCEGRIDTAG 107
#define COMPUTEMAPTAG 108

#define CYCLE_BARRIER   0
#define PME_BARRIER     0
#define STEP_BARRIER    0

#define USE_BARRIER   (CYCLE_BARRIER || PME_BARRIER || STEP_BARRIER)


//! DMK - Atom Separation (water vs. non-water)
/*!
 *  Setting this define to a non-zero value will cause the
 *  HomePatches to separate the hydrogen groups in their
 *  HomePatch::atom lists (all water molecules first, in arbitrary
 *  order, followed by all non-waters, in arbitrary order).
 *
 *  Note from DH:  This macro appears to be broken.
 *  After fixing basic compilation issues (undefined reference to simParams),
 *  enabling it causes an infinite loop when trying to run STMV.
 */
#define NAMD_SeparateWaters    0

// DMK - Atom Sort
//   Setting this define to a non-zero value will cause the nonbonded compute
//   objects (pairs, not selfs) to sort the atoms along a line connecting the
//   center of masses of the two patches.  This is only done during timesteps
//   where the pairlists are being generated.  As the pairlist is being
//   generated, once an atom that is far enough away along the line is found,
//   the remaining atoms are automatically skipped (avoiding a distance
//   calculation/check for them).
// NOTE: The "less branches" flag toggles between two versions of merge sort.
//   When it is non-zero, a version that has fewer branches (but more integer
//   math) is used.  This version may or may not be faster or some architectures.
#define NAMD_ComputeNonbonded_SortAtoms                   1
  #define NAMD_ComputeNonbonded_SortAtoms_LessBranches    1

// plf -- alternate water models
enum class WaterModel {
  TIP3,
  TIP4,
  SWM4, /* Drude model (5 charge sites) */
};

// Haochuan: map water model enums to their corresponding water group sizes
inline constexpr int getWaterModelGroupSize(const WaterModel& watmodel) {
  return (watmodel == WaterModel::TIP3) ? 3:
         (watmodel == WaterModel::TIP4) ? 4:
         (watmodel == WaterModel::SWM4) ? 5: -1;
}

#if defined(__NVCC__) || defined(__HIPCC__)
#define NAMD_HOST_DEVICE __forceinline__ __device__ __host__
#else
#define NAMD_HOST_DEVICE inline
#endif

// Do not include converse when NVCC is being used. Make depends uses gcc so we
// define NAMD_NVCC to prevent inclusion there
#if !(defined(__NVCC__) || defined(NAMD_NVCC) || defined(__HIPCC__))
#include "converse.h"
#endif

#endif

