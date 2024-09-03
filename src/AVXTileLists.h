#ifndef AVXTILELISTS_H
#define AVXTILELISTS_H

#include "AVXTiles.h"
#include "Lattice.h"

#ifdef NAMD_AVXTILES
#include <immintrin.h>

//
// Replace missing macros and intrinsics when building with gcc/clang 
//
#if defined __INTEL_COMPILER || defined __INTEL_LLVM_COMPILER

#define FORCEINLINE __forceinline

#else  // fixes for non-Intel compilers

#define FORCEINLINE inline

#ifndef _MM_SCALE_8
#define _MM_SCALE_8 8
#endif

#ifndef _MM_SCALE_4
#define _MM_SCALE_4 4
#endif

#ifndef _popcnt32
#define _popcnt32 _mm_popcnt_u32
#endif

// _mm512_invsqrt_ps from SVML is unavailable, so provide our own version
// based on rsqrt14 14-bit approximation for reciprocal square root.
// Perform one iteration of Newtonr-Raphson to get full 23-bit precision.
static inline __m512 _mm512_invsqrt_ps(__m512 a){
  __m512 xn = _mm512_rsqrt14_ps(a);
  __m512 xn1 = _mm512_set1_ps(3.0f) - _mm512_mul_ps(a, _mm512_mul_ps(xn, xn));
  return _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(xn, xn1));
}

#ifndef _mm512_i32logather_pd
#define _mm512_i32logather_pd(vindex, base_addr, scale) _mm512_i32gather_pd(_mm512_extracti32x8_epi32((vindex), 0), (base_addr), (scale))
#endif

#ifndef _mm512_mask_i32logather_pd
#define _mm512_mask_i32logather_pd(src, mask, vindex, base_addr, scale) _mm512_mask_i32gather_pd((src), (mask), _mm512_extracti32x8_epi32((vindex), 0), (base_addr), (scale))
#endif

#ifndef _mm512_i32loscatter_pd
#define _mm512_i32loscatter_pd(base_addr, vindex, a, scale) _mm512_i32scatter_pd((base_addr), _mm512_extracti32x8_epi32((vindex), 0), (a), (scale))
#endif

#ifndef _mm512_mask_i32loscatter_pd
#define _mm512_mask_i32loscatter_pd(base_addr, k, vindex, a, scale) _mm512_mask_i32scatter_pd((base_addr), (k), _mm512_extracti32x8_epi32((vindex), 0), (a), (scale))
#endif

#endif // fixes for non-Intel compilers

// --------------------------------------------------------------------------
// Preprocessor defines for hybrid tiles/pair list
// - Disable with NAMD_AVXTILES_PAIR_THRESHOLD=0 and no ORDER_PATCHES
// --------------------------------------------------------------------------
// Threshold in number of atoms in i-tile to switch to pair list
#define NAMD_AVXTILES_PAIR_THRESHOLD 4
// Initial allocation size for number of neighbors per atom in pair list
#define NAMD_AVXTILES_IPAIRCOUNT 300
// Order patch pair so first patch is one with smallest # atoms in last tile
#define NAMD_AVXTILES_ORDER_PATCHES
// --------------------------------------------------------------------------

// Class for storing data for neighbor "j" tiles packed for all atoms in patch 
class AVXJTiles {
 public:
  AVXJTiles();
  ~AVXJTiles();

  inline int numTiles() const { return _numTiles; }
  inline int maxTiles() const { return _numTilesAlloc; }
  inline bool realloc(const int n) {
    _numTiles = n; 
    if (n>_numTilesAlloc) {
      _realloc(); 
      return true;
    } else
      return false;
  }

  // Bitwise exclusion data for neighbor tiles
  unsigned int *excl;
  // Starting index for atoms in neighbor tile
  int *atomStart;
  // Used for deleting empty tile lists on build steps
  int *status;

 private:
  int _numTiles, _numTilesAlloc;
  void _realloc();
};


// Data and routines for storing/building tile lists and computing forces,
// virials, and energies from this data for patch pairs (including self).
// - Modified and excluded pairs are also processed here with separate loops
class AVXTileLists {
 public:
  struct List {
    int atomStart_i;
    int jtileStart;
  };

  AVXTileLists();
  ~AVXTileLists();

  // Simulation parameters are passed to be explicit about data used for
  // tiles algorithm. Interpolation mode is set and documented in
  // ComputeNonbondedUtil.
  void setSimParams(const float scale, const float scale14, const float c1,
		    const float c3, const float switchOn2, float *fastTable,
		    float *fastEnergyTable, float *slowTable,
		    float *slowEnergyTable, float *eps4sigma,
		    float *eps4sigma14, float *ljTable, 
		    const float ljTableWidth, float *modifiedTable,
		    float *modifiedEnergyTable, float *excludedTable,
		    float *excludedEnergyTable, const int interpolationMode);

  inline void atomUpdate(AVXTiles *patch0tiles, AVXTiles *patch1tiles) {
    tiles_p0 = patch0tiles;
    tiles_p1 = patch1tiles;

    // Patch reordering currently doesn't help perf unless using hybrid pairs
    #ifdef NAMD_AVXTILES_ORDER_PATCHES
    _patchOrder0 = 0;
    _patchOrder1 = 1;
    bool reorder = false;
    const int rem0 = patch0tiles->numAtoms() & 15;
    const int rem1 = patch1tiles->numAtoms() & 15;
    if (rem1 && rem1 <= NAMD_AVXTILES_PAIR_THRESHOLD && rem1 < rem0)
      reorder = true;
    else if ((rem0 > NAMD_AVXTILES_PAIR_THRESHOLD || rem0 == 0) && 
	     patch1tiles->numAtoms() < patch0tiles->numAtoms())
      reorder = true;
    if (reorder) {
      tiles_p0 = patch1tiles;
      tiles_p1 = patch0tiles;
      _patchOrder0 = 1;
      _patchOrder1 = 0;
    } 
    #endif

    realloc(tiles_p0->numTiles());
  }

  inline void updateParams(const Lattice &lattice, const Vector &offset,
			   const double cutoff) {
    _cutoff2 = cutoff * cutoff;
    _paramMinvCut3 = -1.0 / (_cutoff2 * sqrt(_cutoff2));
    _paramCutUnder3 = 3.0 / sqrt(_cutoff2);
    _shx = offset.x*lattice.a().x + offset.y*lattice.b().x + 
      offset.z*lattice.c().x;
    _shy = offset.x*lattice.a().y + offset.y*lattice.b().y + 
      offset.z*lattice.c().y;
    _shz = offset.x*lattice.a().z + offset.y*lattice.b().z + 
      offset.z*lattice.c().z;
  }

  inline void updateBuildInfo(const int step, const int minPart,
			      const int maxPart, const int numParts,
			      const double plcutoff) {
    _lastBuild = step;
    _minPart = minPart;
    _maxPart = maxPart;
    _numParts = numParts;
    _plcutoff2 = plcutoff * plcutoff;
  }
  
  inline int numLists() const { return _numLists; }
  // Reallocate data for storing tile lists
  inline void realloc(const int numLists) { 
    _numLists = numLists; 
    if (numLists > _numListsAlloc) _realloc();
  }
  // Reallocate data for storing modified pairs
  inline void reallocModified(const int numModified) { 
    _numModified = numModified; 
    if (numModified > _numModifiedAlloc) _reallocModified();
  }
  // Reallocate data for storing excluded pairs
  inline void reallocExcluded(const int numExcluded) { 
    _numExcluded = numExcluded; 
    if (numExcluded > _numExcludedAlloc) _reallocExcluded();
  }
  // Reallocate data for pair lists when using hybrid tile / pairlists
  inline void reallocPairLists(const int numPairLists, const int maxPairs) {
    if (numPairLists > _maxPairLists || maxPairs > _maxPairs)
      _reallocPairLists(numPairLists, maxPairs);
  }

  inline int exclusionChecksum() const { return _exclusionChecksum; }
  inline float energyVdw() const { return _energyVdw; }
  inline float energyElec() const { return _energyElec; }
  inline float energySlow() const { return _energySlow; }
  inline float virialXX() const { return _fNet_x * _shx; }
  inline float virialXY() const { return _fNet_x * _shy; }
  inline float virialXZ() const { return _fNet_x * _shz; }
  inline float virialYY() const { return _fNet_y * _shy; }
  inline float virialYZ() const { return _fNet_y * _shz; }
  inline float virialZZ() const { return _fNet_z * _shz; }
  inline float virialSlowXX() const { return _fNetSlow_x * _shx; }
  inline float virialSlowXY() const { return _fNetSlow_x * _shy; }
  inline float virialSlowXZ() const { return _fNetSlow_x * _shz; }
  inline float virialSlowYY() const { return _fNetSlow_y * _shy; }
  inline float virialSlowYZ() const { return _fNetSlow_y * _shz; }
  inline float virialSlowZZ() const { return _fNetSlow_z * _shz; }

  #ifdef NAMD_AVXTILES_ORDER_PATCHES
  inline int patchOrder0() const { return _patchOrder0; }
  inline int patchOrder1() const { return _patchOrder1; }
  #else
  inline int patchOrder0() const { return 0; }
  inline int patchOrder1() const { return 1; }
  #endif

  // Build bounding boxes for tiles on both patches and build initial tile
  // lists based on bounding boxes
  // -- Paritioning for LB is based on number of neighbor tiles
  void build();
  // On build steps, delete any empty tile lists after refinement in force
  // calculation based on atom distances.
  void delEmptyLists();
  // Calculate forces, virials, energies for tile lists, pair lists, and
  // modified/excluded pairs
  void nbForceAVX512(const int doEnergy, const int doVirial, const int doList,
		     const int doSlow);

  List *lists;
  // Number of tile neighbors
  unsigned int *listDepth;

  // Tiles data for each patch in pair
  AVXTiles *tiles_p0, *tiles_p1;
  // Neighbor tile data
  AVXJTiles jTiles;

 private:
  template <bool count, bool partitionMode>
  int _buildBB();

  template <bool doEnergy, bool doVirial, bool doSlow, 
            bool doList, int interpMode>
  FORCEINLINE void nbAVX512Tiles(__m512 &energyVdw, __m512 &energyElec,
				   __m512 &energySlow, __m512 &fNet_x, 
				   __m512 &fNet_y, __m512 &fNet_z,
				   __m512 &fNetSlow_x, __m512 &fNetSlow_y,
				   __m512 &fNetSlow_z);
  template <bool doEnergy, bool doVirial, bool doSlow, int interpMode>
  FORCEINLINE void nbAVX512Pairs(__m512 &energyVdw, __m512 &energyElec,
				   __m512 &energySlow, __m512 &fNet_x,
				   __m512 &fNet_y, __m512 &fNet_z,
				   __m512 &fNetSlow_x, __m512 &fNetSlow_y,
				   __m512 &fNetSlow_z);
  template <bool doEnergy, bool doVirial, bool doSlow, int interpMode>
  inline void nbAVX512Modified(__m512 &energyVdw, __m512 &energyElec,
				      __m512 &energySlow, __m512 &fNet_x,
				      __m512 &fNet_y, __m512 &fNet_z,
				      __m512 &fNetSlow_x, __m512 &fNetSlow_y,
				      __m512 &fNetSlow_z);
  template <bool doEnergy, bool doVirial>
  inline void nbAVX512Excluded(__m512 &energySlow, __m512 &fNetSlow_x,
				      __m512 &fNetSlow_y, __m512 &fNetSlow_z);

  template <bool doEnergy, bool doVirial, bool doSlow, 
            bool doList, int interpMode>
  void doAll();

  void _realloc();
  void _reallocModified();
  void _reallocExcluded();
  void _reallocPairLists(const int numPairLists, const int maxPairs);

  float _cutoff2, _plcutoff2;

  float *_paramSlowTable, *_paramSlowEnergyTable;
  // -------------- NOT USED WITH INTERPOLATION MODES 2 and 3
  float *_paramEps4Sigma, *_paramEps4Sigma14;
  // -------------- NOT USED WITH INTERPOLATION MODE  3
  float _paramMinvCut3, _paramCutUnder3;
  // --------------
  float *_paramModifiedTable, *_paramModifiedEnergyTable;
  float *_paramExcludedTable, *_paramExcludedEnergyTable;

  // -------------- NOT USED WITH INTERPOLATION MODES 1 and 2
  float *_paramFastTable, *_paramFastEnergyTable;
  const float *_paramLjTable;
  int _paramLjWidth;
  // --------------

  float _shx, _shy, _shz;
  float _paramScale, _paramScale14, _paramC1, _paramC3, _paramSwitchOn2;
  int _numLists, _numListsAlloc;
  int _numModified, _numModifiedAlloc, _numExcluded, _numExcludedAlloc;
  int *_modified_i, *_modified_j, *_excluded_i, *_excluded_j;

  int _numPairLists, _maxPairLists, _maxPairs;
  int *_pair_i, *_numPairs, *_pairStart, *_pairLists;

  float _fNet_x, _fNet_y, _fNet_z, _fNetSlow_x, _fNetSlow_y, _fNetSlow_z;
  int _exclusionChecksum;
  float _energyVdw, _energyElec, _energySlow;

  int _interpolationMode, _minPart, _maxPart, _numParts, _lastBuild;

  #ifdef NAMD_AVXTILES_ORDER_PATCHES
  int _patchOrder0, _patchOrder1;
  #endif

  #ifndef MEM_OPT_VERSION
  char * _exclFlyListBuffer;
  char * _exclFlyLists[16];
  const int32 * _fullExcl[16], * _modExcl[16];
  int _lastFlyListTile;
  const char * buildExclFlyList(const int itileList, const int z,
				const __m512i &atomIndex_i, const int n,
				void *mol);
  #endif
};

#endif // NAMD_AVXTILES
#endif // AVXTILELISTS_H
