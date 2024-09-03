#ifndef AVXTILES_H
#define AVXTILES_H

#include "common.h"
#include "NamdTypes.h"

#ifdef NAMD_AVXTILES

#ifdef NAMD_CUDA
#error Cannot build with NAMD_CUDA and NAMD_AVXTILES defined
#endif

// Each patch is associtated with an AVXTiles object with
//   - pointer to CUDA single precision {x,y,z,q} data
//   - bounding box data for atoms as divided into tiles
//   - arrays for (slow) forces
//   - local storage of sort order, LJ types, and exclusion data
class AVXTiles {
 public:
  struct AVXTilesAtom { float x, y, z, q; };
  struct AVXTilesForce { float x, y, z, w; };

  AVXTiles();
  ~AVXTiles();

  inline int numAtoms() const { return _numAtoms; }
  inline int numFreeAtoms() const { return _numFreeAtoms; }
  inline int numTiles() const { return _numTiles; }
  
  // Signal that force data has been touched and should be reduced with Patch
  // data and zeroed on subsequent step
  inline void touch() { _touched = true; }
  
  inline void realloc(const int numAtoms, CudaAtom *ac) {
    atoms = (AVXTilesAtom *)ac;
    if (numAtoms != _numAtoms) {
      if (numAtoms > _numAtoms) _touched = true;
      _numAtoms = numAtoms;
      _numTiles = ((numAtoms - 1) >> 4) + 1;
      if (_numTiles > _numTilesAlloc) _realloc();
    } 
  }

  inline void zeroForces(const int doSlow) {
    if (_touched) {
      memset(forces, 0, _numAtoms * sizeof(AVXTilesForce));
      if (doSlow) memset(forcesSlow, 0, _numAtoms * sizeof(AVXTilesForce));
      _touched = false;
    }
  }

  void atomUpdate(const CompAtom *compAtom,
		  const CompAtomExt *compAtomExt);

  // Needed on steps where we will a rebuild of the tile list is required
  // - if multiple Computes use the patch, only calculate once per timestep
  inline void buildBoundingBoxes(const int step) {
    if (step != _lastBuild) _buildBoundingBoxes(step);
  }

  void nativeForceVirialUpdate(const int doSlow, const int doVirial, 
			       const CompAtom *p, const Vector &center,
			       Force * __restrict__ natForces,
			       Force * __restrict__ natForcesSlow,
			       const Force * __restrict__ natForcesVirial,
			       const Force * __restrict__ natForcesSlowVirial,
			       double virial[6], double virialSlow[6]);

  // ----------------------- Per Atom Data --------------------------
  AVXTilesAtom *atoms;
  AVXTilesForce *forces;
  AVXTilesForce *forcesSlow;

  int *vdwTypes;
  int *atomIndex;
#ifdef MEM_OPT_VERSION
  int *atomExclIndex;
#endif
  int *exclIndexStart, *exclIndexMaxDiff;
  int *reverseOrder;

  // ----------------------- Per Tile Data --------------------------

  float *bbox_x, *bbox_y, *bbox_z, *bbox_wx, *bbox_wy, *bbox_wz;

 private:
  int _numAtoms, _numFreeAtoms, _numTiles, _numTilesAlloc, _lastBuild;
  bool _touched;
  void _realloc();
  void _buildBoundingBoxes(const int step);
  template <int doSlow, int doVirial, int touched>
  void _nativeForceVirialUpdate(const CompAtom *p, const Vector &center,
				Force * __restrict__ natForces,
				Force * __restrict__ natForcesSlow,
				const Force * __restrict__ natForcesVirial,
				const Force * __restrict__ natForcesSlowVirial,
				double virial[6], double virialSlow[6]);
};

#endif // NAMD_AVXTILES
#endif // AVXTILES_H
