// -- WMB: For exclusions
#include "Node.h"
#include "Molecule.h"
// --

#include "AVXTilesKernel.h"

#ifdef NAMD_AVXTILES

#define MAX(A,B) ((A) > (B) ? (A) : (B))

#ifndef MEM_OPT_VERSION
const char * AVXTileLists::buildExclFlyList(const int itileList, const int z,
					    const __m512i &atomIndex_i,
					    const int n, void *molIn) {
  if (itileList == _lastFlyListTile) return _exclFlyLists[z];

  Molecule *mol = (Molecule *)molIn;

  if (!_exclFlyListBuffer) {
    _exclFlyListBuffer = new char[mol->numAtoms * 16];
    for (int i = 0; i < 16; i++)
      _exclFlyLists[i] = _exclFlyListBuffer + mol->numAtoms * i;
  }
  if (_lastFlyListTile == -1)
    memset( (void*) _exclFlyListBuffer, 0, mol->numAtoms * 16);

  for (int i = 0; i < 16; i++) {
    if (i >= n) break;
    char *exclFlyList = _exclFlyLists[i];
    const int32 *& fullExcl = _fullExcl[i];
    const int32 *& modExcl = _modExcl[i];
    
    if (_lastFlyListTile != -1) {
      int nl;
      nl = fullExcl[0] + 1;
      for (int l=1; l<nl; ++l ) exclFlyList[fullExcl[l]] = 0;
      nl = modExcl[0] + 1;
      for (int l=1; l<nl; ++l ) exclFlyList[modExcl[l]] = 0;
    }
  
    int nl;
    const int id = *((int*)&(atomIndex_i) + i);
    fullExcl = mol->get_full_exclusions_for_atom(id);
    nl = fullExcl[0] + 1;
    for (int l=1; l<nl; ++l )
      exclFlyList[fullExcl[l]] = EXCHCK_FULL;
    modExcl = mol->get_mod_exclusions_for_atom(id);
    nl = modExcl[0] + 1;
    for (int l=1; l<nl; ++l )
      exclFlyList[modExcl[l]] = EXCHCK_MOD;
  } // for i

  _lastFlyListTile = itileList;
  return _exclFlyLists[z];
}
#endif

//---------------------------------------------------------------------------
// Calculations for unmodified/unexcluded from tile lists
//---------------------------------------------------------------------------

template <bool doEnergy, bool doVirial, bool doSlow, bool doList, 
	  int interpMode>
void AVXTileLists::nbAVX512Tiles(__m512 &energyVdw, __m512 &energyElec,
				 __m512 &energySlow, __m512 &fNet_x, 
				 __m512 &fNet_y, __m512 &fNet_z,
				 __m512 &fNetSlow_x, __m512 &fNetSlow_y,
				 __m512 &fNetSlow_z) {

  #ifndef MEM_OPT_VERSION
  _lastFlyListTile = -1;
  #endif
  
  Molecule* mol = Node::Object()->molecule;

  const AVXTiles::AVXTilesAtom* __restrict__ xyzq_i = tiles_p0->atoms;
  const AVXTiles::AVXTilesAtom* __restrict__ xyzq_j = tiles_p1->atoms;
  AVXTiles::AVXTilesForce* __restrict__ forces_i = tiles_p0->forces;
  AVXTiles::AVXTilesForce* __restrict__ forces_j = tiles_p1->forces;
  AVXTiles::AVXTilesForce* __restrict__ forcesSlow_i;
  AVXTiles::AVXTilesForce* __restrict__ forcesSlow_j;
  if (doSlow) {
    forcesSlow_i = tiles_p0->forcesSlow;
    forcesSlow_j = tiles_p1->forcesSlow;
  }

  const float* __restrict__ fastTable;
  const float* __restrict__ energyTable;
  const float * __restrict__ ljTable;
  const float * __restrict__ eps4sigma;
  // Interpolation for long-range splitting function
  if (interpMode == 3) {
    fastTable = _paramFastTable;
    energyTable = _paramFastEnergyTable;
  }
  // LJ mixing not performed within kernel
  if (interpMode > 1) ljTable = _paramLjTable;
  // LJ mixing performed within kernel
  if (interpMode == 1) eps4sigma = _paramEps4Sigma; 
    
  const float* __restrict__ slowTable = _paramSlowTable;
  const float* __restrict__ slowEnergyTable = _paramSlowEnergyTable;

  #ifdef TILE_LIST_STAT_DEBUG
  int num_jtiles = 0;
  int num_jtiles_empty = 0;
  int num_rotates = 0;
  int num_rotates_empty = 0;
  int num_rotates_excl_empty = 0;
  #endif


  int numModified, numExcluded;
  if (doList) {
    numModified = 0;
    // WMB: Only need for slow electrostatics
    numExcluded = 0;
  }

  const bool zeroShift = ! (_shx*_shx + _shy*_shy + _shz*_shz > 0) &&
    (tiles_p0 == tiles_p1);

  const int numTileLists = numLists();
  for (int itileList = 0; itileList < numTileLists; itileList++) {

    bool jTileActive;
    unsigned int itileListLen;

    const int atomStart_i = lists[itileList].atomStart_i;
    const int jtileStart = lists[itileList].jtileStart;
    const int jtileEnd   = lists[itileList + 1].jtileStart;

    int atomSize_i, atomFreeSize_i, atomSize_j, atomFreeSize_j;
    int nFree_i;
    __mmask16 freeMask_i, sizeMask_i;
    bool doIpairList;
    float bbox_x, bbox_y, bbox_z, bbox_wx, bbox_wy, bbox_wz;
    __m512i atomIndex_i;
#ifdef MEM_OPT_VERSION
    __m512i atomExclIndex_i;
#endif
    __m512i exclIndex_i, exclMaxDiff_i;
    if (doList) {
      atomSize_i     = tiles_p0->numAtoms();
      atomFreeSize_i = tiles_p0->numFreeAtoms();
      atomSize_j     = tiles_p1->numAtoms();
      atomFreeSize_j = tiles_p1->numFreeAtoms();

      freeMask_i = 0xFFFF;
      nFree_i = MAX(atomFreeSize_i - atomStart_i, 0);
      if (nFree_i < 16)
	freeMask_i >>= 16 - nFree_i;
      sizeMask_i = 0xFFFF;
      const int check_i = atomSize_i - atomStart_i;
      if (check_i < 16) {
	sizeMask_i >>= 16 - check_i;
	if (NAMD_AVXTILES_PAIR_THRESHOLD > 0) {
	  if (check_i <= NAMD_AVXTILES_PAIR_THRESHOLD) doIpairList = true;
	  else doIpairList = false;
	}
      } else if (NAMD_AVXTILES_PAIR_THRESHOLD > 0)
	doIpairList = false;

      const int tileNum = atomStart_i >> 4;
      bbox_x = tiles_p0->bbox_x[tileNum] + _shx;
      bbox_y = tiles_p0->bbox_y[tileNum] + _shy;
      bbox_z = tiles_p0->bbox_z[tileNum] + _shz;
      bbox_wx = tiles_p0->bbox_wx[tileNum];
      bbox_wy = tiles_p0->bbox_wy[tileNum];
      bbox_wz = tiles_p0->bbox_wz[tileNum];

      itileListLen = 0;
      atomIndex_i = _mm512_loadu_epi32(tiles_p0->atomIndex + atomStart_i);
#ifdef MEM_OPT_VERSION
      atomExclIndex_i = _mm512_loadu_epi32(tiles_p0->atomExclIndex +
	atomStart_i);
#endif
      exclIndex_i = _mm512_loadu_epi32(tiles_p0->exclIndexStart + atomStart_i);
      exclMaxDiff_i = _mm512_loadu_epi32(tiles_p0->exclIndexMaxDiff +
					atomStart_i);
    }

    // Load i-atom data (and shift coordinates)
    const float *iptr = (float *)(xyzq_i + atomStart_i);
    const __m512 t0 = _mm512_loadu_ps(iptr);
    const __m512 t1 = _mm512_loadu_ps(iptr+16);
    const __m512 t2 = _mm512_loadu_ps(iptr+32);
    const __m512 t3 = _mm512_loadu_ps(iptr+48);
    const __m512i t4 = _mm512_set_epi32(29,25,21,17,13,9,5,1,
					28,24,20,16,12,8,4,0);
    const __m512 t5 = _mm512_permutex2var_ps(t0, t4, t1);
    const __m512i t6 = _mm512_set_epi32(28,24,20,16,12,8,4,0,
					29,25,21,17,13,9,5,1);
    const __m512 t7 = _mm512_permutex2var_ps(t2, t6, t3);
    const __mmask16 t9 = 0xFF00;
    const __m512i t5i = _mm512_castps_si512(t5);
    const __m512i t7i = _mm512_castps_si512(t7);
    const __m512 x_i = _mm512_add_ps(_mm512_castsi512_ps(
      _mm512_mask_blend_epi32(t9, t5i, t7i)),_mm512_set1_ps(_shx));
    const __m512 y_i = _mm512_add_ps(_mm512_shuffle_f32x4(t5, t7, 0x4E),
				     _mm512_set1_ps(_shy));
    const __m512i t12 = _mm512_set_epi32(31,27,23,19,15,11,7,3,
					 30,26,22,18,14,10,6,2);
    const __m512 t13 = _mm512_permutex2var_ps(t0, t12, t1);
    const __m512i t14 = _mm512_set_epi32(30,26,22,18,14,10,6,2,
					 31,27,23,19,15,11,7,3);
    const __m512 t15 = _mm512_permutex2var_ps(t2, t14, t3);
    const __m512i t13i = _mm512_castps_si512(t13);
    const __m512i t15i = _mm512_castps_si512(t15);
    const __m512 z_i = _mm512_add_ps(_mm512_castsi512_ps(
      _mm512_mask_blend_epi32(t9, t13i, t15i)), _mm512_set1_ps(_shz));
    const __m512 q_i = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t13i, t15i,
								0x4E));
    const __m512i type_i = _mm512_loadu_epi32(tiles_p0->vdwTypes+atomStart_i);

    // WMB: This can be masked by sizeMask_i; currently only get for doList
    __m512 eps4_i, sigma_i;
    if (interpMode == 1) {
      const __m512 t0 = (__m512)_mm512_i32logather_pd(type_i, eps4sigma,
						      _MM_SCALE_8);
      const __m512i type_i2 = _mm512_shuffle_i32x4(type_i, type_i, 238);
      const __m512 t1 = (__m512)_mm512_i32logather_pd(type_i2, eps4sigma,
						      _MM_SCALE_8);
      const __m512i t4 = _mm512_set_epi32(31,29,27,25,23,21,19,17,
					  15,13,11,9,7,5,3,1);
      sigma_i = _mm512_permutex2var_ps(t0, t4, t1);
      const __m512i t6 = _mm512_set_epi32(30,28,26,24,22,20,18,16,
					  14,12,10,8,6,4,2,0);
      eps4_i = _mm512_permutex2var_ps(t0, t6, t1);
    }

    // Zero i-tile force vectors
    __m512 forceSlow_i_x, forceSlow_i_y, forceSlow_i_z;
    __m512 force_i_x = _mm512_setzero_ps();
    __m512 force_i_y = _mm512_setzero_ps();
    __m512 force_i_z = _mm512_setzero_ps();
    if (doSlow) {
      forceSlow_i_x = _mm512_setzero_ps();
      forceSlow_i_y = _mm512_setzero_ps();
      forceSlow_i_z = _mm512_setzero_ps();
    }

    for (int jtile=jtileStart;jtile < jtileEnd;jtile++) {

      #ifdef TILE_LIST_STAT_DEBUG
      num_jtiles++;
      bool jtile_empty = true;
      #endif

      // Load j-atom starting index
      int atomStart_j = jTiles.atomStart[jtile];
      const bool self = zeroShift && (atomStart_i == atomStart_j);
      const int shiftVal = (self) ? 1 : 0;

      // Load j-atom positions / charges
      const float * jptr = (float *)(xyzq_j + atomStart_j);
      const __m512 t0 = _mm512_loadu_ps(jptr);
      const __m512 t1 = _mm512_loadu_ps(jptr+16);
      const __m512 t2 = _mm512_loadu_ps(jptr+32);
      const __m512 t3 = _mm512_loadu_ps(jptr+48);
      const __m512i t4 = _mm512_set_epi32(29,25,21,17,13,9,5,1,
					  28,24,20,16,12,8,4,0);
      const __m512 t5 = _mm512_permutex2var_ps(t0, t4, t1);
      const __m512i t6 = _mm512_set_epi32(28,24,20,16,12,8,4,0,
					  29,25,21,17,13,9,5,1);
      const __m512 t7 = _mm512_permutex2var_ps(t2, t6, t3);
      const __mmask16 t9 = 0xFF00;
      const __m512i t5i = _mm512_castps_si512(t5);
      const __m512i t7i = _mm512_castps_si512(t7);
      __m512 x_j = _mm512_castsi512_ps(_mm512_mask_blend_epi32(t9, t5i, t7i));
      __m512 y_j = _mm512_shuffle_f32x4(t5, t7, 0x4E);
      const __m512i t12 = _mm512_set_epi32(31,27,23,19,15,11,7,3,
					   30,26,22,18,14,10,6,2);
      const __m512 t13 = _mm512_permutex2var_ps(t0, t12, t1);
      const __m512i t14 = _mm512_set_epi32(30,26,22,18,14,10,6,2,
					   31,27,23,19,15,11,7,3);
      const __m512 t15 = _mm512_permutex2var_ps(t2, t14, t3);
      const __m512i t13i = _mm512_castps_si512(t13);
      const __m512i t15i = _mm512_castps_si512(t15);
      __m512 z_j = _mm512_castsi512_ps(_mm512_mask_blend_epi32(t9, t13i,
							       t15i));
      __m512 q_j = _mm512_castsi512_ps(_mm512_shuffle_i32x4(t13i, t15i, 0x4E));

      __m512i excl, atomIndex_j;
      __mmask16 freeMask_j, sizeMask_j;
      if (doList) {
	// Check for early bail from i-tile bounding box

	// dx = max(0.f, abs(bbox_x - x_j) - bbox_wx)
	const __m512 dx_one = _mm512_abs_ps(
	  _mm512_sub_ps(_mm512_set1_ps(bbox_x), x_j));
	const __m512 dx_two = _mm512_set1_ps(bbox_wx);
	const __mmask16 lxmask = _mm512_cmplt_ps_mask(dx_two, dx_one);
	const __m512 dx = _mm512_mask_sub_ps(_mm512_setzero_ps(), lxmask,
					     dx_one, dx_two);
	// dy = max(0.f, abs(bbox_y - y_j) - bbox_wy)
	const __m512 dy_one = _mm512_abs_ps(
          _mm512_sub_ps(_mm512_set1_ps(bbox_y), y_j));
	const __m512 dy_two = _mm512_set1_ps(bbox_wy);
	const __mmask16 lymask = _mm512_cmplt_ps_mask(dy_two, dy_one);
	const __m512 dy = _mm512_mask_sub_ps(_mm512_setzero_ps(), lymask,
					     dy_one, dy_two);
	// dz = max(0.f, abs(bbox_z - z_j) - bbox_wz)
	const __m512 dz_one = _mm512_abs_ps(
          _mm512_sub_ps(_mm512_set1_ps(bbox_z), z_j));
	const __m512 dz_two = _mm512_set1_ps(bbox_wz);
	const __mmask16 lzmask = _mm512_cmplt_ps_mask(dz_two, dz_one);
	const __m512 dz = _mm512_mask_sub_ps(_mm512_setzero_ps(), lzmask,
					     dz_one, dz_two);
	// r2bb = dx*dx + dy*dy + dz*dz
	const __m512 r2bb = _mm512_fmadd_ps(dx,dx,_mm512_fmadd_ps(dy,dy,
          _mm512_mul_ps(dz, dz)));

	// If no atoms within bounding box, skip this neighbor tile
	__mmask16 m = _mm512_cmple_ps_mask(r2bb, _mm512_set1_ps(_plcutoff2));
	if (!m) continue;

	// Load atom indices
	atomIndex_j = _mm512_loadu_epi32(tiles_p1->atomIndex+atomStart_j);
	// Zero exclusion data
	excl = _mm512_setzero_epi32();

	// Predication for number of j atoms and free atoms in tile
	freeMask_j = 0xFFFF;
	int nFree_j = MAX(atomFreeSize_j - atomStart_j, 0);
	if (nFree_j < 16) 
	  freeMask_j >>= 16 - nFree_j;
	const int check_j = atomSize_j - atomStart_j;
	sizeMask_j = 0xFFFF;
	if (check_j < 16) 
	  sizeMask_j >>= 16 - check_j;

	jTileActive = false;

	// If doing a pair list instead of tile list, make sure we have
	// room to store neighbor atom indices
	if (NAMD_AVXTILES_PAIR_THRESHOLD > 0 && doIpairList) {
	  int maxPairs = _numPairs[0];
	  for (int z = 1; z < NAMD_AVXTILES_PAIR_THRESHOLD; z++)
	    if (_numPairs[z] > maxPairs) maxPairs = _numPairs[z];
	  if (maxPairs + 16 >  _maxPairs) {
	    reallocPairLists(0, 1.4 * _maxPairs);
	  }
	}
      } else
	// Load tile exclusion data
	excl = _mm512_loadu_epi32(jTiles.excl + (jtile << 4));

      // Load j tile LJ types and epsilon/sigma for interp mode 1
      __m512i type_j = _mm512_loadu_epi32(tiles_p1->vdwTypes+atomStart_j);

      __m512 eps4_j, sigma_j;
      if (interpMode == 1) {
	const __m512 t0 = (__m512)_mm512_i32logather_pd(type_j, eps4sigma,
							_MM_SCALE_8);
	const __m512i type_j2 = _mm512_shuffle_i32x4(type_j, type_j, 238);
	const __m512 t1 = (__m512)_mm512_i32logather_pd(type_j2, eps4sigma,
							_MM_SCALE_8);
	const __m512i t4 = _mm512_set_epi32(31,29,27,25,23,21,19,17,
					    15,13,11,9,7,5,3,1);
	sigma_j = _mm512_permutex2var_ps(t0, t4, t1);
	const __m512i t6 = _mm512_set_epi32(30,28,26,24,22,20,18,16,
					    14,12,10,8,6,4,2,0);
	eps4_j = _mm512_permutex2var_ps(t0, t6, t1);
      }

      // Zero force vectors for j tile
      __m512 force_j_x = _mm512_setzero_ps();
      __m512 force_j_y = _mm512_setzero_ps();
      __m512 force_j_z = _mm512_setzero_ps();
 
      __m512 forceSlow_j_x, forceSlow_j_y, forceSlow_j_z;
      if (doSlow) {
	forceSlow_j_x = _mm512_setzero_ps();
	forceSlow_j_y = _mm512_setzero_ps();
	forceSlow_j_z = _mm512_setzero_ps();
      }

      int t = (self) ? 1 : 0;

      // WMB: If adding GBIS p 2, need to add check diagonal for excluded

      // Predication for self-interactions to prevent duplicate interactions
      __mmask16 mask_j;
      if (doList) mask_j = 0xFFFF;

      // If doList, excl is 0; if self, skipping first bit
      if (!doList && !self) excl = _mm512_slli_epi32(excl, 1);

      // Skip self interaction
      if (self) {
	x_j = (__m512)_mm512_alignr_epi32((__m512i)x_j, (__m512i)x_j, 1);
	y_j = (__m512)_mm512_alignr_epi32((__m512i)y_j, (__m512i)y_j, 1);
	z_j = (__m512)_mm512_alignr_epi32((__m512i)z_j, (__m512i)z_j, 1);
	q_j = (__m512)_mm512_alignr_epi32((__m512i)q_j, (__m512i)q_j, 1);
	if (doList) {
	  freeMask_j = (freeMask_j>>1) | (freeMask_j<<15);
	  sizeMask_j = (sizeMask_j>>1) | (sizeMask_j<<15);
	  if (doList) mask_j >>= shiftVal;
	  atomIndex_j = _mm512_alignr_epi32(atomIndex_j, atomIndex_j, 1);
	}
	if (interpMode == 1) {
	  eps4_j = (__m512)_mm512_alignr_epi32((__m512i)eps4_j, 
					       (__m512i)eps4_j, 1);
	  sigma_j = (__m512)_mm512_alignr_epi32((__m512i)sigma_j, 
						(__m512i)sigma_j, 1);
	} else 
	  type_j = _mm512_alignr_epi32(type_j, type_j, 1);
	// force is zero, no need to rotate
      } 

      for (; t < 16; t++) {
	// Exclusion bit for next pair
	excl = _mm512_srli_epi32(excl, 1);
	__mmask16 freeMask_ij;
	if (doList)
	  // Predication for partial tiles and fixed atoms at ends
	  freeMask_ij = (freeMask_j | freeMask_i) & (sizeMask_j & sizeMask_i);
	else {
          #ifdef TILE_LIST_STAT_DEBUG
	  __mmask16 r2mask = (excl & 1).notzero();
	  num_rotates++;
	  if (!r2mask) {num_rotates_empty++; num_rotates_excl_empty++;}
          #endif
	}

        // ------------- Distance squared calculation and cutoff check 
        __m512 dx, dy, dz, r2;
	__mmask16 r2mask;
	if (!doList || freeMask_ij) {
	  // dx = x_j - x_i;
	  dx = _mm512_sub_ps(x_j, x_i);
	  // dy = y_j - y_i;
	  dy = _mm512_sub_ps(y_j, y_i);
	  // dz = z_j - z_i;
	  dz = _mm512_sub_ps(z_j, z_i);
	  // r2 = dx*dx + dy*dy + dz*dz;
	  r2 = _mm512_fmadd_ps(dx,dx,_mm512_fmadd_ps(dy,dy,
            _mm512_mul_ps(dz, dz)));

          #ifdef TILE_LIST_STAT_DEBUG
	  if (!doList) {
	    __mmask16 t=r2 <= _plcutoff2; t &= r2mask;
	    if (!t) num_rotates_empty++; else jtile_empty = false;
          }
          #endif

	  if (doList) {
	    // Predication for atom pairs within build cutoff
	    r2mask = _mm512_cmple_ps_mask(r2, _mm512_set1_ps(_plcutoff2)) &
	      mask_j;
	    r2mask &= freeMask_ij;
	  } else {
	    // Predication for atom pairs within force cutoff
	    r2mask = _mm512_cmple_ps_mask(r2, _mm512_set1_ps(_cutoff2));
	    r2mask &= _mm512_cmpneq_epi32_mask(_mm512_and_epi32(excl,
	      _mm512_set1_epi32(1)), _mm512_setzero_epi32());
          }
	} else
	  r2mask = 0;

        // ------------- Exclusions
	if (doList && r2mask) {
	  if (_numModifiedAlloc - numModified < 16) {
	    int newNum = static_cast<double>(numModified) *
	      numTileLists / itileList;
	    if (newNum < _numModifiedAlloc + 16)
	      newNum = _numModifiedAlloc + 16;
	    reallocModified(newNum);
	  }
	  if (_numExcludedAlloc - numExcluded < 16) {
	    int newNum = static_cast<double>(numExcluded) *
	      numTileLists / itileList;
	    if (newNum < _numExcludedAlloc + 16)
	      newNum = _numExcludedAlloc + 16;
	    reallocExcluded(newNum);
	  }

	  // Predication for exclusions and cutoff
	  __mmask16 excludedMask = _mm512_cmpge_epi32_mask(atomIndex_j,
							   exclIndex_i);
	  excludedMask &= _mm512_cmple_epi32_mask(atomIndex_j, exclMaxDiff_i);
	  excludedMask &= r2mask;
	  __mmask16 scalarPos = 1;

	  // Check each pair for exclusions
	  for (int z = 0; z < 16; z++) {
	    if (scalarPos & excludedMask) {
              #ifdef MEM_OPT_VERSION
              const char *exclFlags = 
                mol->get_excl_check_for_idx(*((int*)&(atomExclIndex_i) + z))->
                  flags;
              #else
	      const char *exclFlags = 
		mol->get_excl_check_for_atom(*((int*)&(atomIndex_i) + z))->
		  flags;
              #endif
              #ifndef MEM_OPT_VERSION
	      if (exclFlags)
	      #endif
		exclFlags -= *((int*)&(exclIndex_i) + z);
              #ifndef MEM_OPT_VERSION
	      else {
		#pragma noinline
		exclFlags = buildExclFlyList(itileList, z, atomIndex_i,
					     atomSize_i - atomStart_i, mol);
	      }
	      #endif
	      const int exclType = exclFlags[*((int*)&atomIndex_j + z)];
	      if (exclType == 0)
		// Clear exclusion bit
	        excludedMask &= ~scalarPos;  
	      else if (exclType == 2) {
		// Exclude from tiles and add pair to modified list
		_modified_i[numModified] = atomStart_i + z;
		int jpos = (z + t) & 15;
		_modified_j[numModified] = atomStart_j + jpos;
		if (*((float*)&r2 + z) <= _cutoff2)
		  numModified++;
	      } else {
		// Excluded from tiles and add pair to fully excluded list
		_excluded_i[numExcluded] = atomStart_i + z;
		int jpos = (z + t) & 15;
		_excluded_j[numExcluded] = atomStart_j + jpos;
		if (*((float*)&r2 + z) <= _cutoff2)
		  numExcluded++;
	      }
	    }
	    // Next pair
	    scalarPos <<= 1;
	  }

	  // Note: For exclusions, use force cutoff and not list cutoff
	  r2mask &= ~excludedMask;

	  // If building a pair list, store each pair within cutoff
	  if (NAMD_AVXTILES_PAIR_THRESHOLD > 0 &&
	      doIpairList && r2mask) {
	    for (int z = 0; z < NAMD_AVXTILES_PAIR_THRESHOLD; z++)  {
	      if (r2mask & 1) {
		const int jPos = atomStart_j + ((z+t) & 15);
		_pairLists[_pairStart[z]+_numPairs[z]] = jPos;
		_numPairs[z]++;
	      }
	      r2mask >>= 1;
	    }
	    r2mask = 0;
	  } else {
	    // Mark tile list as active if neighbors within cutoff
	    excl = _mm512_mask_or_epi32(excl, r2mask, excl,
					_mm512_set1_epi32(0x8000));
	    if (r2mask) jTileActive = true;
	    // Redo predication for pairs w/in force cutoff rather build cutoff
	    r2mask = _mm512_cmple_ps_mask(r2, _mm512_set1_ps(_cutoff2)) &
	      mask_j & freeMask_ij & ~excludedMask;
	  }
        } // if doList

        // ------------- Force, Energy, Virial Calculations
	if (r2mask) {
	  // kqq = q_i * q_j
	  const __m512 kqq = _mm512_mul_ps(q_i, q_j);
	  __m512 force, forceSlow;

	  // Call force kernel
	  if (interpMode == 1)
	    forceEnergyInterp1<doEnergy, doSlow>(r2, kqq, force, forceSlow,
                  energyVdw, energyElec, energySlow, r2mask, _paramC1,
                  _paramC3, _paramSwitchOn2, _cutoff2, _paramMinvCut3,
		  _paramCutUnder3, slowTable, slowEnergyTable, eps4_i, eps4_j,
		  sigma_i, sigma_j);
	  else
	    forceEnergyInterp2<doEnergy, doSlow, interpMode>(r2, kqq, type_i,
                  type_j, force, forceSlow, energyVdw, energyElec, energySlow, 
                  r2mask, _paramScale, _paramC1, _paramC3, _paramSwitchOn2,
		  _cutoff2, _paramMinvCut3, _paramCutUnder3, fastTable,
		  energyTable, slowTable, slowEnergyTable, ljTable,
                  _paramLjWidth);

	  // force_i_. += d. * force
	  force_i_x = _mm512_mask_mov_ps(force_i_x, r2mask,
	    _mm512_fmadd_ps(dx, force, force_i_x));
	  force_i_y = _mm512_mask_mov_ps(force_i_y, r2mask,
	    _mm512_fmadd_ps(dy, force, force_i_y));
	  force_i_z = _mm512_mask_mov_ps(force_i_z, r2mask,
	    _mm512_fmadd_ps(dz, force, force_i_z));
	  // force_j_. += d. * force
	  force_j_x = _mm512_mask_mov_ps(force_j_x, r2mask,
	    _mm512_fnmadd_ps(dx, force, force_j_x));
	  force_j_y = _mm512_mask_mov_ps(force_j_y, r2mask,
	    _mm512_fnmadd_ps(dy, force, force_j_y));
	  force_j_z = _mm512_mask_mov_ps(force_j_z, r2mask,
	    _mm512_fnmadd_ps(dz, force, force_j_z));
	  if (doSlow) {
  	    // force_i_. += d. * forceSlow_i_.
	    forceSlow_i_x = _mm512_mask_mov_ps(forceSlow_i_x, r2mask,
	      _mm512_fmadd_ps(dx, forceSlow, forceSlow_i_x));
	    forceSlow_i_y = _mm512_mask_mov_ps(forceSlow_i_y, r2mask,
	      _mm512_fmadd_ps(dy, forceSlow, forceSlow_i_y));
	    forceSlow_i_z = _mm512_mask_mov_ps(forceSlow_i_z, r2mask,
	      _mm512_fmadd_ps(dz, forceSlow, forceSlow_i_z));
  	    // force_j_. += d. * forceSlow_j_.
	    forceSlow_j_x = _mm512_mask_mov_ps(forceSlow_j_x, r2mask,
	      _mm512_fnmadd_ps(dx, forceSlow, forceSlow_j_x));
	    forceSlow_j_y = _mm512_mask_mov_ps(forceSlow_j_y, r2mask,
	      _mm512_fnmadd_ps(dy, forceSlow, forceSlow_j_y));
	    forceSlow_j_z = _mm512_mask_mov_ps(forceSlow_j_z, r2mask,
	      _mm512_fnmadd_ps(dz, forceSlow, forceSlow_j_z));
	  }
	}

        // ------------- Next set of pairs for tile interactions
	x_j = (__m512)_mm512_alignr_epi32((__m512i)x_j, (__m512i)x_j, 1);
	y_j = (__m512)_mm512_alignr_epi32((__m512i)y_j, (__m512i)y_j, 1);
	z_j = (__m512)_mm512_alignr_epi32((__m512i)z_j, (__m512i)z_j, 1);
	q_j = (__m512)_mm512_alignr_epi32((__m512i)q_j, (__m512i)q_j, 1);
	if (doList) {
	  freeMask_j = (freeMask_j>>1) | (freeMask_j<<15);
	  sizeMask_j = (sizeMask_j>>1) | (sizeMask_j<<15);
	  mask_j >>= shiftVal;
	  atomIndex_j = _mm512_alignr_epi32(atomIndex_j, atomIndex_j, 1);
	}
	if (interpMode == 1) {
	  eps4_j = (__m512)_mm512_alignr_epi32((__m512i)eps4_j, 
					       (__m512i)eps4_j, 1);
	  sigma_j = (__m512)_mm512_alignr_epi32((__m512i)sigma_j, 
						(__m512i)sigma_j, 1);
	} else 
	  type_j = _mm512_alignr_epi32(type_j, type_j, 1);
	force_j_x = (__m512)_mm512_alignr_epi32((__m512i)force_j_x, 
						(__m512i)force_j_x, 1);
	force_j_y = (__m512)_mm512_alignr_epi32((__m512i)force_j_y, 
						(__m512i)force_j_y, 1);
	force_j_z = (__m512)_mm512_alignr_epi32((__m512i)force_j_z, 
						(__m512i)force_j_z, 1);
	if (doSlow) {
	  forceSlow_j_x = (__m512)_mm512_alignr_epi32((__m512i)forceSlow_j_x, 
						      (__m512i)forceSlow_j_x,
						      1);
	  forceSlow_j_y = (__m512)_mm512_alignr_epi32((__m512i)forceSlow_j_y, 
						      (__m512i)forceSlow_j_y,
						      1);
	  forceSlow_j_z = (__m512)_mm512_alignr_epi32((__m512i)forceSlow_j_z, 
						      (__m512i)forceSlow_j_z,
						      1);
	}	    
      } // for t

      // ------------- Accumulate j-forces in memory
      const __m512i tp0x = _mm512_set_epi32(0,19,11,3,0,18,10,2,
					    0,17,9,1,0,16,8,0);
      const __m512i tp1x = _mm512_set_epi32(0,23,15,7,0,22,14,6,0,21,
					    13,5,0,20,12,4);
      const __m512i tp2x = _mm512_set_epi32(0,27,11,3,0,26,10,2,
					    0,25,9,1,0,24,8,0);
      const __m512i tp3x = _mm512_set_epi32(0,31,15,7,0,30,14,6,
					    0,29,13,5,0,28,12,4);
      {
	float * jptr = (float *)(forces_j + atomStart_j);
	const __m512 v0 = _mm512_loadu_ps(jptr);
	const __m512 v1 = _mm512_loadu_ps(jptr + 16);
	const __m512 v2 = _mm512_loadu_ps(jptr + 32);
	const __m512 v3 = _mm512_loadu_ps(jptr + 48);
	const __m512 w1 = _mm512_shuffle_f32x4(force_j_x, force_j_y,
					       0b01000100);
	const __m512 w2 = _mm512_shuffle_f32x4(force_j_x, force_j_y,
					       0b11101110);
	__m512 tp0 = _mm512_permutex2var_ps(w1, tp0x, force_j_z);
	__m512 tp1 = _mm512_permutex2var_ps(w1, tp1x, force_j_z);
	__m512 tp2 = _mm512_permutex2var_ps(w2, tp2x, force_j_z);
	__m512 tp3 = _mm512_permutex2var_ps(w2, tp3x, force_j_z);
	tp0 = _mm512_add_ps(v0, tp0);
	tp1 = _mm512_add_ps(v1, tp1);
	tp2 = _mm512_add_ps(v2, tp2);
	tp3 = _mm512_add_ps(v3, tp3);
	_mm512_store_ps(jptr, tp0);
	_mm512_store_ps(jptr + 16, tp1);
	_mm512_store_ps(jptr + 32, tp2);
	_mm512_store_ps(jptr + 48, tp3);
      }

      if (doSlow) {
	float * jptr = (float *)(forcesSlow_j + atomStart_j);
	const __m512 v0 = _mm512_loadu_ps(jptr);
	const __m512 v1 = _mm512_loadu_ps(jptr + 16);
	const __m512 v2 = _mm512_loadu_ps(jptr + 32);
	const __m512 v3 = _mm512_loadu_ps(jptr + 48);
	const __m512 w1 = _mm512_shuffle_f32x4(forceSlow_j_x, forceSlow_j_y,
					       0b01000100);
	const __m512 w2 = _mm512_shuffle_f32x4(forceSlow_j_x, forceSlow_j_y,
					       0b11101110);
	__m512 tp0 = _mm512_permutex2var_ps(w1, tp0x, forceSlow_j_z);
	__m512 tp1 = _mm512_permutex2var_ps(w1, tp1x, forceSlow_j_z);
	__m512 tp2 = _mm512_permutex2var_ps(w2, tp2x, forceSlow_j_z);
	__m512 tp3 = _mm512_permutex2var_ps(w2, tp3x, forceSlow_j_z);
	tp0 = _mm512_add_ps(v0, tp0);
	tp1 = _mm512_add_ps(v1, tp1);
	tp2 = _mm512_add_ps(v2, tp2);
	tp3 = _mm512_add_ps(v3, tp3);
	_mm512_store_ps(jptr, tp0);
	_mm512_store_ps(jptr + 16, tp1);
	_mm512_store_ps(jptr + 32, tp2);
	_mm512_store_ps(jptr + 48, tp3);
      }

      // ------------- Write exclusions to memory
      if (doList) {
	if (jTileActive) {
	  int anyexcl = 65536;
	  if (_mm512_cmpneq_epi32_mask(excl, _mm512_setzero_epi32()))
	    anyexcl |= 1;
	  jTiles.status[jtile] = anyexcl;
          // Store exclusions
	  _mm512_store_epi32((void *)(jTiles.excl + (jtile << 4)),
			     excl);
          itileListLen += anyexcl;
	}
      } // if doList
      #ifdef TILE_LIST_STAT_DEBUG
      if (jtile_empty) num_jtiles_empty++;
      #endif
    } // jtile

    // ------------------------------------------------------

    // ------------- Accumulate i-forces in memory
    const __m512i tp0x = _mm512_set_epi32(0,19,11,3,0,18,10,2,
					  0,17,9,1,0,16,8,0);
    const __m512i tp1x = _mm512_set_epi32(0,23,15,7,0,22,14,6,0,21,
					  13,5,0,20,12,4);
    const __m512i tp2x = _mm512_set_epi32(0,27,11,3,0,26,10,2,
					  0,25,9,1,0,24,8,0);
    const __m512i tp3x = _mm512_set_epi32(0,31,15,7,0,30,14,6,
					  0,29,13,5,0,28,12,4);
    {
      float * iptr = (float *)(forces_i + atomStart_i);
      const __m512 v0 = _mm512_loadu_ps(iptr);
      const __m512 v1 = _mm512_loadu_ps(iptr + 16);
      const __m512 v2 = _mm512_loadu_ps(iptr + 32);
      const __m512 v3 = _mm512_loadu_ps(iptr + 48);
      const __m512 w1 = _mm512_shuffle_f32x4(force_i_x, force_i_y,
					     0b01000100);
      const __m512 w2 = _mm512_shuffle_f32x4(force_i_x, force_i_y,
					     0b11101110);
      __m512 tp0 = _mm512_permutex2var_ps(w1, tp0x, force_i_z);
      __m512 tp1 = _mm512_permutex2var_ps(w1, tp1x, force_i_z);
      __m512 tp2 = _mm512_permutex2var_ps(w2, tp2x, force_i_z);
      __m512 tp3 = _mm512_permutex2var_ps(w2, tp3x, force_i_z);
      tp0 = _mm512_add_ps(v0, tp0);
      tp1 = _mm512_add_ps(v1, tp1);
      tp2 = _mm512_add_ps(v2, tp2);
      tp3 = _mm512_add_ps(v3, tp3);
      _mm512_store_ps(iptr, tp0);
      _mm512_store_ps(iptr + 16, tp1);
      _mm512_store_ps(iptr + 32, tp2);
      _mm512_store_ps(iptr + 48, tp3);
    }

    if (doSlow) {
      float * iptr = (float *)(forcesSlow_i + atomStart_i);
      const __m512 v0 = _mm512_loadu_ps(iptr);
      const __m512 v1 = _mm512_loadu_ps(iptr + 16);
      const __m512 v2 = _mm512_loadu_ps(iptr + 32);
      const __m512 v3 = _mm512_loadu_ps(iptr + 48);
      const __m512 w1 = _mm512_shuffle_f32x4(forceSlow_i_x, forceSlow_i_y,
					     0b01000100);
      const __m512 w2 = _mm512_shuffle_f32x4(forceSlow_i_x, forceSlow_i_y,
					     0b11101110);
      __m512 tp0 = _mm512_permutex2var_ps(w1, tp0x, forceSlow_i_z);
      __m512 tp1 = _mm512_permutex2var_ps(w1, tp1x, forceSlow_i_z);
      __m512 tp2 = _mm512_permutex2var_ps(w2, tp2x, forceSlow_i_z);
      __m512 tp3 = _mm512_permutex2var_ps(w2, tp3x, forceSlow_i_z);
      tp0 = _mm512_add_ps(v0, tp0);
      tp1 = _mm512_add_ps(v1, tp1);
      tp2 = _mm512_add_ps(v2, tp2);
      tp3 = _mm512_add_ps(v3, tp3);
      _mm512_store_ps(iptr, tp0);
      _mm512_store_ps(iptr + 16, tp1);
      _mm512_store_ps(iptr + 32, tp2);
      _mm512_store_ps(iptr + 48, tp3);
    }

    if (doList)
      listDepth[itileList] = itileListLen;

    // Update net forces for virial
    if (doVirial) {
      // fNet_. += force_i_.
      fNet_x = _mm512_add_ps(fNet_x, force_i_x);
      fNet_y = _mm512_add_ps(fNet_y, force_i_y);
      fNet_z = _mm512_add_ps(fNet_z, force_i_z);
      if (doSlow) {
	// fNetSlow_. += forceSlow_i_.
	fNetSlow_x = _mm512_add_ps(fNetSlow_x, forceSlow_i_x);
	fNetSlow_y = _mm512_add_ps(fNetSlow_y, forceSlow_i_y);
	fNetSlow_z = _mm512_add_ps(fNetSlow_z, forceSlow_i_z);
      }
    } // if (doVirial)

    if (NAMD_AVXTILES_PAIR_THRESHOLD > 0 && doList) {
      _numPairLists = 0;
      for (int z = 0; z < NAMD_AVXTILES_PAIR_THRESHOLD; z++)
	if (_numPairs[z]) _numPairLists++;
    }
  } // for itileList

  if (doList) {
    _numModified = numModified;
    // WMB: Only needed for slow elec
    _numExcluded = numExcluded;
    _exclusionChecksum = numModified + numExcluded;
  }

  #ifdef TILE_LIST_STAT_DEBUG
  if (!doList)
    printf("TILE_DBG: JTILES %d EMPTY %.2f ROTATES %d EMPTY %.2f EXCL %.2f\n",
	   num_jtiles, double(num_jtiles_empty)/num_jtiles, num_rotates,
	   double(num_rotates_empty)/num_rotates, 
	   double(num_rotates_excl_empty)/num_rotates);
  #endif
} // nbAVX512Tiles() 

//---------------------------------------------------------------------------
// Calculations for unmodified/unexcluded from pair lists
//---------------------------------------------------------------------------

template <bool doEnergy, bool doVirial, bool doSlow, int interpMode>
void AVXTileLists::nbAVX512Pairs(__m512 &energyVdw, __m512 &energyElec,
				 __m512 &energySlow, __m512 &fNet_x,
				 __m512 &fNet_y, __m512 &fNet_z,
				 __m512 &fNetSlow_x, __m512 &fNetSlow_y,
				 __m512 &fNetSlow_z) {

  const AVXTiles::AVXTilesAtom* __restrict__ xyzq_i = tiles_p0->atoms;
  const AVXTiles::AVXTilesAtom* __restrict__ xyzq_j = tiles_p1->atoms;
  AVXTiles::AVXTilesForce* __restrict__ forces_i = tiles_p0->forces;
  AVXTiles::AVXTilesForce* __restrict__ forces_j = tiles_p1->forces;
  AVXTiles::AVXTilesForce* __restrict__ forcesSlow_i;
  AVXTiles::AVXTilesForce* __restrict__ forcesSlow_j;
  if (doSlow) {
    forcesSlow_i = tiles_p0->forcesSlow;
    forcesSlow_j = tiles_p1->forcesSlow;
  }

  const float* __restrict__ fastTable;
  const float* __restrict__ energyTable;
  const float * __restrict__ ljTable;
  const float * __restrict__ eps4sigma;
  // Interpolation for long-range splitting function
  if (interpMode == 3) {
    fastTable = _paramFastTable;
    energyTable = _paramFastEnergyTable;
  }
  // LJ mixing not performed within kernel
  if (interpMode > 1) ljTable = _paramLjTable;
  // LJ mixing performed within kernel
  if (interpMode == 1) eps4sigma = _paramEps4Sigma; 
    
  const float* __restrict__ slowTable = _paramSlowTable;
  const float* __restrict__ slowEnergyTable = _paramSlowEnergyTable;

  for (int z = 0; z < NAMD_AVXTILES_PAIR_THRESHOLD; z++) {
    if (!_numPairs[z]) continue;

    // Zero i-tile force vectors
    __m512 force_i_x = _mm512_setzero_ps();
    __m512 force_i_y = _mm512_setzero_ps();
    __m512 force_i_z = _mm512_setzero_ps();
    __m512 forceSlow_i_x, forceSlow_i_y, forceSlow_i_z;
    if (doSlow) {
      forceSlow_i_x = _mm512_setzero_ps();
      forceSlow_i_y = _mm512_setzero_ps();
      forceSlow_i_z = _mm512_setzero_ps();
    }

    const int pairI = _pair_i[z];
    const int nPairs = _numPairs[z];
    // Load i-atom data
    const __m512 x_i = _mm512_set1_ps(xyzq_i[pairI].x + _shx);
    const __m512 y_i = _mm512_set1_ps(xyzq_i[pairI].y + _shy);
    const __m512 z_i = _mm512_set1_ps(xyzq_i[pairI].z + _shz);
    const __m512 q_i = _mm512_set1_ps(xyzq_i[pairI].q);
    const int scalarType_i = tiles_p0->vdwTypes[pairI];
    const __m512i type_i = _mm512_set1_epi32(scalarType_i);
    __m512 eps4_i, sigma_i;
    if (interpMode == 1) {
      eps4_i = _mm512_set1_ps(eps4sigma[scalarType_i*2]);
      sigma_i = _mm512_set1_ps(eps4sigma[scalarType_i*2+1]);
    }

    __mmask16 loopMask = 0xFFFF;
    int listPos = _pairStart[z];
    for (int mv = 0; mv < nPairs; mv += 16) {
      // Remainder predication
      if (nPairs - mv < 16) 
	loopMask >>= (16 - (nPairs - mv));
      // Load j indices from pair list
      __m512i j = _mm512_loadu_epi32(_pairLists + listPos);
      listPos += 16;
	
      // Load j atom data
      const __m512i jt2 = _mm512_slli_epi32(j, 1);
      const __m512 t0 = (__m512)_mm512_mask_i32logather_pd(
        _mm512_undefined_pd(), loopMask, jt2, (float *)xyzq_j, _MM_SCALE_8);
      const __m512 t8 = (__m512)_mm512_mask_i32logather_pd(
	_mm512_undefined_pd(), loopMask, jt2, (float *)(xyzq_j) + 2,
        _MM_SCALE_8);
      const __m512i jt2_2 = _mm512_shuffle_i32x4(jt2, jt2, 238);
      const __mmask16 loopMask2 = loopMask >> 8;
      const __m512 t1 = (__m512)_mm512_mask_i32logather_pd(
        _mm512_undefined_pd(), loopMask2, jt2_2, (float *)xyzq_j, _MM_SCALE_8);
      const __m512 t9 = (__m512)_mm512_mask_i32logather_pd(
	_mm512_undefined_pd(), loopMask2, jt2_2, (float *)(xyzq_j) + 2,
        _MM_SCALE_8);
      const __m512i t4 = _mm512_set_epi32(31,29,27,25,23,21,19,17,
					  15,13,11,9,7,5,3,1);
      const __m512 y_j = _mm512_permutex2var_ps(t0, t4, t1);
      const __m512 q_j = _mm512_permutex2var_ps(t8, t4, t9);
      const __m512i t6 = _mm512_set_epi32(30,28,26,24,22,20,18,16,
					  14,12,10,8,6,4,2,0);
      const __m512 x_j = _mm512_permutex2var_ps(t0, t6, t1);
      const __m512 z_j = _mm512_permutex2var_ps(t8, t6, t9);
	
      // kqq = q_i * q_j
      const __m512 kqq = _mm512_mul_ps(q_i, q_j);

      // dx = x_i - x_j;
      const __m512 dx = _mm512_sub_ps(x_i, x_j);
      // dy = y_i - y_j;
      const __m512 dy = _mm512_sub_ps(y_i, y_j);
      // dz = z_i - z_j;
      const __m512 dz = _mm512_sub_ps(z_i, z_j);
      // r2 = dx*dx + dy*dy + dz*dz;
      const __m512 r2(_mm512_fmadd_ps(dx,dx,_mm512_fmadd_ps(dy,dy,
        _mm512_mul_ps(dz, dz))));
      // Atoms within cutoff
      const __mmask16 r2mask = _mm512_cmple_ps_mask(r2,
	_mm512_set1_ps(_cutoff2)) & loopMask;

      // Load LJ types
      const __m512i type_j = _mm512_mask_i32gather_epi32(type_j, r2mask, j,
	tiles_p1->vdwTypes, _MM_SCALE_4);

      // Load eps and sigma
      __m512 eps4_j, sigma_j;
      if (interpMode == 1) {
	const __m512 t0 = (__m512)_mm512_mask_i32logather_pd(
          _mm512_undefined_pd(), r2mask, type_j, eps4sigma, _MM_SCALE_8);
	const __m512i type_j2 = _mm512_shuffle_i32x4(type_j, type_j, 238);
	const __mmask16 r2mask2 = r2mask >> 8;
	const __m512 t1 = (__m512)_mm512_mask_i32logather_pd(
          _mm512_undefined_pd(), r2mask2, type_j2, eps4sigma, _MM_SCALE_8);
	sigma_j = _mm512_permutex2var_ps(t0, t4, t1);
	eps4_j = _mm512_permutex2var_ps(t0, t6, t1);
      }

      // Force, Energy, Virial calculation
      __m512 force, forceSlow;
      if (interpMode == 1)
	forceEnergyInterp1<doEnergy, doSlow>(r2, kqq, force, forceSlow,
    	    energyVdw, energyElec, energySlow, r2mask, _paramC1, _paramC3,
            _paramSwitchOn2, _cutoff2, _paramMinvCut3, _paramCutUnder3,
            slowTable, slowEnergyTable, eps4_i, eps4_j, sigma_i, sigma_j);
      else
	forceEnergyInterp2<doEnergy, doSlow, interpMode>(r2, kqq, type_i, 
            type_j, force, forceSlow, energyVdw, energyElec, energySlow,
            r2mask, _paramScale, _paramC1, _paramC3, _paramSwitchOn2,
            _cutoff2, _paramMinvCut3, _paramCutUnder3, fastTable, energyTable, 
            slowTable, slowEnergyTable, ljTable, _paramLjWidth);

      // force_. = d. * force
      const __m512 force_x = _mm512_mul_ps(dx, force);
      const __m512 force_y = _mm512_mul_ps(dy, force);
      const __m512 force_z = _mm512_mul_ps(dz, force);
      // Accumulate j forces in memory
      const __m512i j4 = _mm512_slli_epi32(j, 2);
      __m512 ft0 = (__m512)_mm512_mask_i32logather_pd(_mm512_undefined_pd(),
        r2mask, j4, (float*)forces_j, _MM_SCALE_4);
      const __m512i j4_2 = _mm512_shuffle_i32x4(j4, j4, 238);
      const __mmask16 r2mask2 = r2mask >> 8;
      __m512 ft1 = (__m512)_mm512_mask_i32logather_pd(_mm512_undefined_pd(),
	r2mask2, j4_2, (float *)forces_j, _MM_SCALE_4);
      const __m512i ft4 = _mm512_set_epi32(23,7,22,6,21,5,20,4,
					   19,3,18,2,17,1,16,0);
      const __m512 ft5 = _mm512_permutex2var_ps(force_x, ft4, force_y);
      ft0 = _mm512_add_ps(ft0, ft5);
      _mm512_mask_i32loscatter_pd((void *)forces_j, r2mask, j4, (__m512d)ft0,
				  _MM_SCALE_4);
      const __m512i ft2 = _mm512_set_epi32(31,15,30,14,29,13,28,12,
					   27,11,26,10,25,9,24,8);
      const __m512 ft3 = _mm512_permutex2var_ps(force_x, ft2, force_y);
      ft1 = _mm512_add_ps(ft1, ft3);
      _mm512_mask_i32loscatter_pd((void *)forces_j, r2mask2, j4_2,
				  (__m512d)ft1, _MM_SCALE_4);
      __m512 mem3 = _mm512_mask_i32gather_ps(_mm512_undefined_ps(), r2mask,
					     j4, (float *)(forces_j)+2,
					     _MM_SCALE_4);
      mem3 = _mm512_add_ps(mem3, force_z);
      _mm512_mask_i32scatter_ps((float *)(forces_j)+2, r2mask, j4, mem3,
				_MM_SCALE_4);

      // force_i_. -= force_.
      force_i_x = _mm512_mask_sub_ps(force_i_x, r2mask, force_i_x, force_x);
      force_i_y = _mm512_mask_sub_ps(force_i_y, r2mask, force_i_y, force_y);
      force_i_z = _mm512_mask_sub_ps(force_i_z, r2mask, force_i_z, force_z);

      __m512 forceSlow_x, forceSlow_y, forceSlow_z;
      if (doSlow) {
        // forceSlow_. = d. * forceSlow
	forceSlow_x = _mm512_mul_ps(dx, forceSlow);
	forceSlow_y = _mm512_mul_ps(dy, forceSlow);
	forceSlow_z = _mm512_mul_ps(dz, forceSlow);
	// Accumulate j slow forces in memory
	//	acc3(r2mask, (float*)forcesSlow_j, j, forceSlow_x,
	//	     forceSlow_y, forceSlow_z);
	__m512 ft10 = (__m512)_mm512_mask_i32logather_pd(_mm512_undefined_pd(),
          r2mask, j4, (float*)forcesSlow_j, _MM_SCALE_4);
	__m512 ft11 = (__m512)_mm512_mask_i32logather_pd(_mm512_undefined_pd(),
	  r2mask2, j4_2, (float *)forcesSlow_j, _MM_SCALE_4);
	const __m512 ft15 = _mm512_permutex2var_ps(forceSlow_x, ft4,
						   forceSlow_y);
	ft10 = _mm512_add_ps(ft10, ft15);
	_mm512_mask_i32loscatter_pd((void *)forcesSlow_j, r2mask, j4,
				    (__m512d)ft10, _MM_SCALE_4);
	const __m512 ft13 = _mm512_permutex2var_ps(forceSlow_x, ft2,
						  forceSlow_y);
	ft11 = _mm512_add_ps(ft11, ft13);
	_mm512_mask_i32loscatter_pd((void *)forcesSlow_j, r2mask2, j4_2,
				    (__m512d)ft11, _MM_SCALE_4);
	__m512 mem13 = _mm512_mask_i32gather_ps(_mm512_undefined_ps(), r2mask,
						j4, (float *)(forcesSlow_j)+2,
						_MM_SCALE_4);
	mem13 = _mm512_add_ps(mem13, forceSlow_z);
	_mm512_mask_i32scatter_ps((float *)(forcesSlow_j)+2, r2mask, j4, mem13,
				  _MM_SCALE_4);

	// forceSlow_i_. -= forceSlow_.
	forceSlow_i_x = _mm512_mask_sub_ps(forceSlow_i_x, r2mask, 
					   forceSlow_i_x, forceSlow_x);
	forceSlow_i_y = _mm512_mask_sub_ps(forceSlow_i_y, r2mask, 
					   forceSlow_i_y, forceSlow_y);
	forceSlow_i_z = _mm512_mask_sub_ps(forceSlow_i_z, r2mask, 
					   forceSlow_i_z, forceSlow_z);
      }
    } // for mv
    
    // Reduction on i vectors, accumulate in memory, update fNet_. for virial
    float fI = _mm512_reduce_add_ps(force_i_x);
    forces_i[pairI].x += fI;
    if (doVirial) *((float*)&fNet_x) += fI;
    fI = _mm512_reduce_add_ps(force_i_y);
    forces_i[pairI].y += fI;
    if (doVirial) *((float*)&fNet_y) += fI;
    fI = _mm512_reduce_add_ps(force_i_z);
    forces_i[pairI].z += fI;
    if (doVirial) *((float*)&fNet_z) += fI;
    if (doSlow) {
      fI = _mm512_reduce_add_ps(forceSlow_i_x);
      forcesSlow_i[pairI].x += fI;
      if (doVirial) *((float*)&fNetSlow_x) += fI;
      fI = _mm512_reduce_add_ps(forceSlow_i_y);
      forcesSlow_i[pairI].y += fI;
      if (doVirial) *((float*)&fNetSlow_y) += fI;
      fI = _mm512_reduce_add_ps(forceSlow_i_z);
      forcesSlow_i[pairI].z += fI;
      if (doVirial) *((float*)&fNetSlow_z) += fI;
    }
  } // for z
} // nbAVX512Pairs()

//---------------------------------------------------------------------------
// Calculations for modified pairs
//---------------------------------------------------------------------------

template <bool doEnergy, bool doVirial, bool doSlow, int interpMode>
void AVXTileLists::nbAVX512Modified(__m512 &energyVdw, __m512 &energyElec,
				    __m512 &energySlow, __m512 &fNet_x,
				    __m512 &fNet_y, __m512 &fNet_z,
				    __m512 &fNetSlow_x, __m512 &fNetSlow_y,
				    __m512 &fNetSlow_z) {

  const AVXTiles::AVXTilesAtom* __restrict__ xyzq_i = tiles_p0->atoms;
  const AVXTiles::AVXTilesAtom* __restrict__ xyzq_j = tiles_p1->atoms;
  AVXTiles::AVXTilesForce* __restrict__ forces_i = tiles_p0->forces;
  AVXTiles::AVXTilesForce* __restrict__ forces_j = tiles_p1->forces;
  AVXTiles::AVXTilesForce* __restrict__ forcesSlow_i;
  AVXTiles::AVXTilesForce* __restrict__ forcesSlow_j;
  if (doSlow) {
    forcesSlow_i = tiles_p0->forcesSlow;
    forcesSlow_j = tiles_p1->forcesSlow;
  }

  const float* __restrict__ fastTable;
  const float* __restrict__ energyTable;
  // Interpolation for long-range splitting function
  if (interpMode == 3) {
    fastTable = _paramFastTable;
    energyTable = _paramFastEnergyTable;
  }

  const float * __restrict__ mod_table = _paramModifiedTable;
  const float * __restrict__ mode_table = _paramModifiedEnergyTable;

  const float * __restrict__ ljTable14;
  const float * __restrict__ eps4sigma14; 
  if (interpMode > 1)
    ljTable14 = _paramLjTable + 2;
  else
    eps4sigma14 = _paramEps4Sigma14; 

  __mmask16 loopMask = 0xFFFF;
  #pragma novector
  for (int mv = 0; mv < _numModified; mv += 16) {
    // Remainder predication
    if (_numModified - mv < 16) 
      loopMask >>= (16 - (_numModified - mv));
    // Load i and j indices for pairs on modified list
    const __m512i i = _mm512_loadu_epi32(_modified_i + mv);
    const __m512i j = _mm512_loadu_epi32(_modified_j + mv);

    // Load i atom data and shift coordinates
    const __m512i it2 = _mm512_slli_epi32(i, 1);
    const __m512 t0 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask, it2, (float *)xyzq_i, _MM_SCALE_8);
    const __m512 t8 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask, it2, (float *)(xyzq_i) + 2,
      _MM_SCALE_8);
    const __m512i it2_2 = _mm512_shuffle_i32x4(it2, it2, 238);
    const __mmask16 loopMask2 = loopMask >> 8;
    const __m512 t1 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask2, it2_2, (float *)xyzq_i, _MM_SCALE_8);
    const __m512 t9 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask2, it2_2, (float *)(xyzq_i) + 2,
      _MM_SCALE_8);
    const __m512i t4 = _mm512_set_epi32(31,29,27,25,23,21,19,17,
					15,13,11,9,7,5,3,1);
    const __m512 y_i = _mm512_add_ps(_mm512_permutex2var_ps(t0, t4, t1),
				     _mm512_set1_ps(_shy));
    const __m512 q_i = _mm512_permutex2var_ps(t8, t4, t9);
    const __m512i t6 = _mm512_set_epi32(30,28,26,24,22,20,18,16,
					14,12,10,8,6,4,2,0);
    const __m512 x_i = _mm512_add_ps(_mm512_permutex2var_ps(t0, t6, t1),
				     _mm512_set1_ps(_shx));
    const __m512 z_i = _mm512_add_ps(_mm512_permutex2var_ps(t8, t6, t9),
				     _mm512_set1_ps(_shz));

    // Load j atom data
    const __m512i jt2 = _mm512_slli_epi32(j, 1);
    const __m512 t10 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask, jt2, (float *)xyzq_j, _MM_SCALE_8);
    const __m512 t18 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask, jt2, (float *)(xyzq_j) + 2,
      _MM_SCALE_8);
    const __m512i jt2_2 = _mm512_shuffle_i32x4(jt2, jt2, 238);
    const __m512 t11 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask2, jt2_2, (float *)xyzq_j, _MM_SCALE_8);
    const __m512 t19 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask2, jt2_2, (float *)(xyzq_j) + 2,
      _MM_SCALE_8);
    const __m512 y_j = _mm512_permutex2var_ps(t10, t4, t11);
    const __m512 q_j = _mm512_permutex2var_ps(t18, t4, t19);
    const __m512 x_j = _mm512_permutex2var_ps(t10, t6, t11);
    const __m512 z_j = _mm512_permutex2var_ps(t18, t6, t19);

    // kqq = q_i * q_j * _paramScale14
    const __m512 kqq = _mm512_mul_ps(q_i, _mm512_mul_ps(q_j,
      _mm512_set1_ps(_paramScale14)));

    // dx = x_i - x_j;
    const __m512 dx = _mm512_sub_ps(x_i, x_j);
    // dy = y_i - y_j;
    const __m512 dy = _mm512_sub_ps(y_i, y_j);
    // dz = z_i - z_j;
    const __m512 dz = _mm512_sub_ps(z_i, z_j);
    // r2 = dx*dx + dy*dy + dz*dz;
    const __m512 r2(_mm512_fmadd_ps(dx,dx,_mm512_fmadd_ps(dy,dy,
      _mm512_mul_ps(dz, dz))));
    // Atoms within cutoff
    __mmask16 r2mask = _mm512_cmple_ps_mask(r2, _mm512_set1_ps(_cutoff2)) &
      loopMask;

    // Load LJ types
    const __m512i type_i = _mm512_mask_i32gather_epi32(type_i, r2mask, i,
      tiles_p0->vdwTypes, _MM_SCALE_4);
    const __m512i type_j = _mm512_mask_i32gather_epi32(type_j, r2mask, j,
       tiles_p1->vdwTypes, _MM_SCALE_4);

    // Load eps and sigma
    __m512 eps4_i14, sigma_i14, eps4_j14, sigma_j14;
    if (interpMode == 1) {
      const __m512 t0 = (__m512)_mm512_mask_i32logather_pd(
        _mm512_undefined_pd(), r2mask, type_i, eps4sigma14, _MM_SCALE_8);
      const __m512i type_i2 = _mm512_shuffle_i32x4(type_i, type_i, 238);
      const __mmask16 r2mask2 = r2mask >> 8;
      const __m512 t1 = (__m512)_mm512_mask_i32logather_pd(
        _mm512_undefined_pd(), r2mask2, type_i2, eps4sigma14, _MM_SCALE_8);
      const __m512 t10 = (__m512)_mm512_mask_i32logather_pd(
        _mm512_undefined_pd(), r2mask, type_j, eps4sigma14, _MM_SCALE_8);
      const __m512i type_j2 = _mm512_shuffle_i32x4(type_j, type_j, 238);
      const __m512 t11 = (__m512)_mm512_mask_i32logather_pd(
        _mm512_undefined_pd(), r2mask2, type_j2, eps4sigma14, _MM_SCALE_8);

      sigma_i14 = _mm512_permutex2var_ps(t0, t4, t1);
      sigma_j14 = _mm512_permutex2var_ps(t10, t4, t11);
      eps4_i14 = _mm512_permutex2var_ps(t0, t6, t1);
      eps4_j14 = _mm512_permutex2var_ps(t10, t6, t11);
    }

    // Force, Energy, Virial calculation
    __m512 force, forceSlow;
    if (interpMode == 1)
      forceEnergyInterp1<doEnergy, doSlow>(r2, kqq, force, forceSlow,
  	   energyVdw, energyElec, energySlow, r2mask, _paramC1, _paramC3,
           _paramSwitchOn2, _cutoff2, _paramMinvCut3, _paramCutUnder3,
           mod_table, mode_table, eps4_i14, eps4_j14, sigma_i14, sigma_j14);
    else
      forceEnergyInterp2<doEnergy, doSlow, interpMode>(r2, kqq, type_i, type_j,
           force, forceSlow, energyVdw, energyElec, energySlow, r2mask,
           _paramScale, _paramC1, _paramC3, _paramSwitchOn2, _cutoff2,
           _paramMinvCut3, _paramCutUnder3, fastTable, energyTable, 
            mod_table, mode_table, ljTable14, _paramLjWidth);

    // force_i_. = d. * force
    const __m512 force_i_x = _mm512_mul_ps(dx, force);
    const __m512 force_i_y = _mm512_mul_ps(dy, force);
    const __m512 force_i_z = _mm512_mul_ps(dz, force);
    __m512 forceSlow_i_x, forceSlow_i_y, forceSlow_i_z;
    if (doSlow) {
      // forceSlow_i_. = d. * forceSlow
      forceSlow_i_x = _mm512_mul_ps(dx, forceSlow);
      forceSlow_i_y = _mm512_mul_ps(dy, forceSlow);
      forceSlow_i_z = _mm512_mul_ps(dz, forceSlow);
    }

    if (doVirial) {
      // fNet_. -= force_i_.
      fNet_x = _mm512_mask_sub_ps(fNet_x, r2mask, fNet_x, force_i_x);
      fNet_y = _mm512_mask_sub_ps(fNet_y, r2mask, fNet_y, force_i_y);
      fNet_z = _mm512_mask_sub_ps(fNet_z, r2mask, fNet_z, force_i_z);
      if (doSlow) {
	// fNetSlow_. -= forceSlow_i_.
	fNetSlow_x = _mm512_mask_sub_ps(fNetSlow_x, r2mask, 
					fNetSlow_x, forceSlow_i_x);
	fNetSlow_y = _mm512_mask_sub_ps(fNetSlow_y, r2mask, 
					fNetSlow_y, forceSlow_i_y);
	fNetSlow_z = _mm512_mask_sub_ps(fNetSlow_z, r2mask,
					fNetSlow_z, forceSlow_i_z);
      }
    }

    #pragma novector
    for (int z = 0; z < 16; z++) {
      // Skip if outside cutoff or remainder
      if (!(r2mask & 1)) {
	r2mask >>= 1;
	continue;
      }
      r2mask >>= 1;

      // WMB: Might be better to check if next i is same and update in mem once

      // Accumulate i and j forces in memory
      const int i_z = *((int*)&i + z);
      const int j_z = *((int*)&j + z);
      forces_i[i_z].x -= *((float*)&force_i_x + z);
      forces_i[i_z].y -= *((float*)&force_i_y + z);
      forces_i[i_z].z -= *((float*)&force_i_z + z);
      forces_j[j_z].x += *((float*)&force_i_x + z);
      forces_j[j_z].y += *((float*)&force_i_y + z);
      forces_j[j_z].z += *((float*)&force_i_z + z);
      if (doSlow) {
	forcesSlow_i[i_z].x -= *((float*)&forceSlow_i_x + z);
	forcesSlow_i[i_z].y -= *((float*)&forceSlow_i_y + z);
	forcesSlow_i[i_z].z -= *((float*)&forceSlow_i_z + z);
	forcesSlow_j[j_z].x += *((float*)&forceSlow_i_x + z);
	forcesSlow_j[j_z].y += *((float*)&forceSlow_i_y + z);
	forcesSlow_j[j_z].z += *((float*)&forceSlow_i_z + z);
      }
    }
  }
} // nbAVX512Modified()

//---------------------------------------------------------------------------
// Calculations for excluded pairs
//---------------------------------------------------------------------------

template <bool doEnergy, bool doVirial>
void AVXTileLists::nbAVX512Excluded(__m512 &energySlow, __m512 &fNetSlow_x,
				    __m512 &fNetSlow_y, __m512 &fNetSlow_z) {

  const AVXTiles::AVXTilesAtom* __restrict__ xyzq_i = tiles_p0->atoms;
  const AVXTiles::AVXTilesAtom* __restrict__ xyzq_j = tiles_p1->atoms;
  AVXTiles::AVXTilesForce* __restrict__ forcesSlow_i = tiles_p0->forcesSlow;
  AVXTiles::AVXTilesForce* __restrict__ forcesSlow_j = tiles_p1->forcesSlow;

  const float * __restrict__ exclTable = _paramExcludedTable;
  const float * __restrict__ exclEtable = _paramExcludedEnergyTable;

  __mmask16 loopMask = 0xFFFF;
  for (int mv = 0; mv < _numExcluded; mv += 16) {
    // Remainder predication
    if (_numExcluded - mv < 16) 
      loopMask >>= (16 - (_numExcluded - mv));
    // Load i and j indices for pairs on modified list
    const __m512i i = _mm512_loadu_epi32(_excluded_i + mv);
    const __m512i j = _mm512_loadu_epi32(_excluded_j + mv);

    // Load i atom data and shift coordinates
    const __m512i it2 = _mm512_slli_epi32(i, 1);
    const __m512 t0 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask, it2, (float *)xyzq_i, _MM_SCALE_8);
    const __m512 t8 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask, it2, (float *)(xyzq_i) + 2,
      _MM_SCALE_8);
    const __m512i it2_2 = _mm512_shuffle_i32x4(it2, it2, 238);
    const __mmask16 loopMask2 = loopMask >> 8;
    const __m512 t1 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask2, it2_2, (float *)xyzq_i, _MM_SCALE_8);
    const __m512 t9 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask2, it2_2, (float *)(xyzq_i) + 2,
      _MM_SCALE_8);
    const __m512i t4 = _mm512_set_epi32(31,29,27,25,23,21,19,17,
					15,13,11,9,7,5,3,1);
    const __m512 y_i = _mm512_add_ps(_mm512_permutex2var_ps(t0, t4, t1),
				     _mm512_set1_ps(_shy));
    const __m512 q_i = _mm512_permutex2var_ps(t8, t4, t9);
    const __m512i t6 = _mm512_set_epi32(30,28,26,24,22,20,18,16,
					14,12,10,8,6,4,2,0);
    const __m512 x_i = _mm512_add_ps(_mm512_permutex2var_ps(t0, t6, t1),
				     _mm512_set1_ps(_shx));
    const __m512 z_i = _mm512_add_ps(_mm512_permutex2var_ps(t8, t6, t9),
				     _mm512_set1_ps(_shz));

    // Load j atom data
    const __m512i jt2 = _mm512_slli_epi32(j, 1);
    const __m512 t10 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask, jt2, (float *)xyzq_j, _MM_SCALE_8);
    const __m512 t18 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask, jt2, (float *)(xyzq_j) + 2,
      _MM_SCALE_8);
    const __m512i jt2_2 = _mm512_shuffle_i32x4(jt2, jt2, 238);
    const __m512 t11 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask2, jt2_2, (float *)xyzq_j, _MM_SCALE_8);
    const __m512 t19 = (__m512)_mm512_mask_i32logather_pd(
      _mm512_undefined_pd(), loopMask2, jt2_2, (float *)(xyzq_j) + 2,
      _MM_SCALE_8);
    const __m512 y_j = _mm512_permutex2var_ps(t10, t4, t11);
    const __m512 q_j = _mm512_permutex2var_ps(t18, t4, t19);
    const __m512 x_j = _mm512_permutex2var_ps(t10, t6, t11);
    const __m512 z_j = _mm512_permutex2var_ps(t18, t6, t19);

    // kqq = q_i * q_j
    const __m512 kqq = _mm512_mul_ps(q_i, q_j);

    // dx = x_i - x_j;
    const __m512 dx = _mm512_sub_ps(x_i, x_j);
    // dy = y_i - y_j;
    const __m512 dy = _mm512_sub_ps(y_i, y_j);
    // dz = z_i - z_j;
    const __m512 dz = _mm512_sub_ps(z_i, z_j);
    // r2 = dx*dx + dy*dy + dz*dz;
    const __m512 r2(_mm512_fmadd_ps(dx,dx,_mm512_fmadd_ps(dy,dy,
      _mm512_mul_ps(dz, dz))));
    // Atoms within cutoff
    __mmask16 r2mask = _mm512_cmple_ps_mask(r2, _mm512_set1_ps(_cutoff2)) &
      loopMask;

    // Force, Energy, Virial calculation
    const __m512 r_1 = _mm512_invsqrt_ps(r2);
    __m512 forceSlow, tableDiff, rTableDiff;
    __m512i table_int;
    getOmpSimdTableI(r_1, table_int, tableDiff, rTableDiff);
    forceEnergySlow512<doEnergy>(r2mask, kqq, exclTable, exclEtable,
				 table_int, tableDiff, rTableDiff, forceSlow,
				 energySlow);

    // forceSlow_i_. = d. * forceSlow
    const __m512 forceSlow_i_x = _mm512_mul_ps(dx, forceSlow);
    const __m512 forceSlow_i_y = _mm512_mul_ps(dy, forceSlow);
    const __m512 forceSlow_i_z = _mm512_mul_ps(dz, forceSlow);
    // fNetSlow_. -= forceSlow_i_.
    fNetSlow_x = _mm512_mask_sub_ps(fNetSlow_x, r2mask, 
				    fNetSlow_x, forceSlow_i_x);
    fNetSlow_y = _mm512_mask_sub_ps(fNetSlow_y, r2mask, 
				    fNetSlow_y, forceSlow_i_y);
    fNetSlow_z = _mm512_mask_sub_ps(fNetSlow_z, r2mask, 
				    fNetSlow_z, forceSlow_i_z);

    for (int z = 0; z < 16; z++) {
      // Skip if outside cutoff or remainder
      if (!(r2mask & 1)) {
	r2mask >>= 1;
	continue;
      }
      r2mask >>= 1;

      // WMB: Might be better to check if next i is same and update in mem once
      const int i_z = *((int*)&i + z);
      const int j_z = *((int*)&j + z);
      forcesSlow_i[i_z].x -= *((float*)&forceSlow_i_x + z);
      forcesSlow_i[i_z].y -= *((float*)&forceSlow_i_y + z);
      forcesSlow_i[i_z].z -= *((float*)&forceSlow_i_z + z);
      forcesSlow_j[j_z].x += *((float*)&forceSlow_i_x + z);
      forcesSlow_j[j_z].y += *((float*)&forceSlow_i_y + z);
      forcesSlow_j[j_z].z += *((float*)&forceSlow_i_z + z);
    } // for m
  }
}

template <bool doEnergy, bool doVirial, bool doSlow, bool doList, 
	  int interpMode>
void AVXTileLists::doAll() {

  if (doList) build();

  __m512 energyVdw, energyElec, energySlow;
  __m512 fNet_x, fNet_y, fNet_z;
  __m512 fNetSlow_x, fNetSlow_y, fNetSlow_z;

  // Zero energy and virial vectors used in all routines
  if (doEnergy) {
    energyVdw = _mm512_setzero_ps();
    energyElec = _mm512_setzero_ps();
    if (doSlow) energySlow = _mm512_setzero_ps();
  }
  if (doVirial) {
    fNet_x = _mm512_setzero_ps();
    fNet_y = _mm512_setzero_ps();
    fNet_z = _mm512_setzero_ps();
    if (doSlow) {
      fNetSlow_x = _mm512_setzero_ps();
      fNetSlow_y = _mm512_setzero_ps();
      fNetSlow_z = _mm512_setzero_ps();
    }
  }

  // Calculations from tile lists
  nbAVX512Tiles<doEnergy, doVirial, doSlow, doList, 
		interpMode>(energyVdw, energyElec, energySlow, fNet_x, fNet_y,
			    fNet_z, fNetSlow_x, fNetSlow_y, fNetSlow_z);

  if (doList) delEmptyLists();

  // Calculations from pair lists
  if (NAMD_AVXTILES_PAIR_THRESHOLD > 0 && _numPairLists)
    nbAVX512Pairs<doEnergy, doVirial, doSlow,
 		  interpMode>(energyVdw, energyElec, energySlow, fNet_x,
                              fNet_y, fNet_z, fNetSlow_x, fNetSlow_y,
			      fNetSlow_z);

  // Calculations for modified exclusions
  if (_numModified)
    nbAVX512Modified<doEnergy, doVirial, doSlow,
		     interpMode>(energyVdw, energyElec, energySlow, fNet_x,
				 fNet_y, fNet_z, fNetSlow_x, fNetSlow_y,
				 fNetSlow_z);

  // Calculations for full exclusions
  if (doSlow && _numExcluded)
    nbAVX512Excluded<doEnergy, doVirial>(energySlow, fNetSlow_x,
					 fNetSlow_y, fNetSlow_z);

  // Reduce energy vectors into scalar
  if (doEnergy) {
    _energyVdw = _mm512_reduce_add_ps(energyVdw);
    if (interpMode > 1) _energyVdw *= _paramScale * 0.5f;
    _energyElec = _mm512_reduce_add_ps(energyElec);
    if (doSlow) _energySlow = _mm512_reduce_add_ps(energySlow);
  }

  // Reduce virial vectors into scalars
  if (doVirial) {
    _fNet_x = _mm512_reduce_add_ps(fNet_x);
    _fNet_y = _mm512_reduce_add_ps(fNet_y);
    _fNet_z = _mm512_reduce_add_ps(fNet_z);
    if (doSlow) {
      _fNetSlow_x = _mm512_reduce_add_ps(fNetSlow_x);
      _fNetSlow_y = _mm512_reduce_add_ps(fNetSlow_y);
      _fNetSlow_z = _mm512_reduce_add_ps(fNetSlow_z);
    }
  }

  // Touch "tiles" data structures on patches used to indicate they
  //   need reduction with native NAMD data and zeroing for subsequent use
  if (doList) { 
    if (_numLists || _numModified || _numExcluded ||
	(NAMD_AVXTILES_PAIR_THRESHOLD > 0 && _numPairLists)) {
      tiles_p0->touch();
      tiles_p1->touch();
    }
  } else {
    tiles_p0->touch();
    tiles_p1->touch();
  }
}

//---------------------------------------------------------------------------

void AVXTileLists::nbForceAVX512(const int doEnergy, const int doVirial, 
				 const int doSlow, const int doList) {

#define CALL(DOENERGY, DOVIRIAL, DOSLOW, DOPAIRLIST, INTERPMODE) \
  doAll<DOENERGY, DOVIRIAL, DOSLOW, DOPAIRLIST, INTERPMODE>();

  if (_interpolationMode == 1) {
    if (!doEnergy && !doVirial && !doSlow && !doList) CALL(0, 0, 0, 0, 1);
    if (!doEnergy && !doVirial &&  doSlow && !doList) CALL(0, 0, 1, 0, 1);
    if (!doEnergy &&  doVirial && !doSlow && !doList) CALL(0, 1, 0, 0, 1);
    if (!doEnergy &&  doVirial &&  doSlow && !doList) CALL(0, 1, 1, 0, 1);
    if ( doEnergy && !doVirial && !doSlow && !doList) CALL(1, 0, 0, 0, 1);
    if ( doEnergy && !doVirial &&  doSlow && !doList) CALL(1, 0, 1, 0, 1);
    if ( doEnergy &&  doVirial && !doSlow && !doList) CALL(1, 1, 0, 0, 1);
    if ( doEnergy &&  doVirial &&  doSlow && !doList) CALL(1, 1, 1, 0, 1);

    if (!doEnergy && !doVirial && !doSlow &&  doList) CALL(0, 0, 0, 1, 1);
    if (!doEnergy && !doVirial &&  doSlow &&  doList) CALL(0, 0, 1, 1, 1);
    if (!doEnergy &&  doVirial && !doSlow &&  doList) CALL(0, 1, 0, 1, 1);
    if (!doEnergy &&  doVirial &&  doSlow &&  doList) CALL(0, 1, 1, 1, 1);
    if ( doEnergy && !doVirial && !doSlow &&  doList) CALL(1, 0, 0, 1, 1);
    if ( doEnergy && !doVirial &&  doSlow &&  doList) CALL(1, 0, 1, 1, 1);
    if ( doEnergy &&  doVirial && !doSlow &&  doList) CALL(1, 1, 0, 1, 1);
    if ( doEnergy &&  doVirial &&  doSlow &&  doList) CALL(1, 1, 1, 1, 1);
  } else if (_interpolationMode == 2) {
    if (!doEnergy && !doVirial && !doSlow && !doList) CALL(0, 0, 0, 0, 2);
    if (!doEnergy && !doVirial &&  doSlow && !doList) CALL(0, 0, 1, 0, 2);
    if (!doEnergy &&  doVirial && !doSlow && !doList) CALL(0, 1, 0, 0, 2);
    if (!doEnergy &&  doVirial &&  doSlow && !doList) CALL(0, 1, 1, 0, 2);
    if ( doEnergy && !doVirial && !doSlow && !doList) CALL(1, 0, 0, 0, 2);
    if ( doEnergy && !doVirial &&  doSlow && !doList) CALL(1, 0, 1, 0, 2);
    if ( doEnergy &&  doVirial && !doSlow && !doList) CALL(1, 1, 0, 0, 2);
    if ( doEnergy &&  doVirial &&  doSlow && !doList) CALL(1, 1, 1, 0, 2);

    if (!doEnergy && !doVirial && !doSlow &&  doList) CALL(0, 0, 0, 1, 2);
    if (!doEnergy && !doVirial &&  doSlow &&  doList) CALL(0, 0, 1, 1, 2);
    if (!doEnergy &&  doVirial && !doSlow &&  doList) CALL(0, 1, 0, 1, 2);
    if (!doEnergy &&  doVirial &&  doSlow &&  doList) CALL(0, 1, 1, 1, 2);
    if ( doEnergy && !doVirial && !doSlow &&  doList) CALL(1, 0, 0, 1, 2);
    if ( doEnergy && !doVirial &&  doSlow &&  doList) CALL(1, 0, 1, 1, 2);
    if ( doEnergy &&  doVirial && !doSlow &&  doList) CALL(1, 1, 0, 1, 2);
    if ( doEnergy &&  doVirial &&  doSlow &&  doList) CALL(1, 1, 1, 1, 2);
  } else {
    if (!doEnergy && !doVirial && !doSlow && !doList) CALL(0, 0, 0, 0, 3);
    if (!doEnergy && !doVirial &&  doSlow && !doList) CALL(0, 0, 1, 0, 3);
    if (!doEnergy &&  doVirial && !doSlow && !doList) CALL(0, 1, 0, 0, 3);
    if (!doEnergy &&  doVirial &&  doSlow && !doList) CALL(0, 1, 1, 0, 3);
    if ( doEnergy && !doVirial && !doSlow && !doList) CALL(1, 0, 0, 0, 3);
    if ( doEnergy && !doVirial &&  doSlow && !doList) CALL(1, 0, 1, 0, 3);
    if ( doEnergy &&  doVirial && !doSlow && !doList) CALL(1, 1, 0, 0, 3);
    if ( doEnergy &&  doVirial &&  doSlow && !doList) CALL(1, 1, 1, 0, 3);

    if (!doEnergy && !doVirial && !doSlow &&  doList) CALL(0, 0, 0, 1, 3);
    if (!doEnergy && !doVirial &&  doSlow &&  doList) CALL(0, 0, 1, 1, 3);
    if (!doEnergy &&  doVirial && !doSlow &&  doList) CALL(0, 1, 0, 1, 3);
    if (!doEnergy &&  doVirial &&  doSlow &&  doList) CALL(0, 1, 1, 1, 3);
    if ( doEnergy && !doVirial && !doSlow &&  doList) CALL(1, 0, 0, 1, 3);
    if ( doEnergy && !doVirial &&  doSlow &&  doList) CALL(1, 0, 1, 1, 3);
    if ( doEnergy &&  doVirial && !doSlow &&  doList) CALL(1, 1, 0, 1, 3);
    if ( doEnergy &&  doVirial &&  doSlow &&  doList) CALL(1, 1, 1, 1, 3);
  }

#undef CALL
}

#endif // NAMD_AVXTILES
