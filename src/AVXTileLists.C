#include "AVXTileLists.h"

#ifdef NAMD_AVXTILES

//---------------------------------------------------------------------------

AVXJTiles::AVXJTiles() : _numTiles(0), _numTilesAlloc(0) {
}

AVXJTiles::~AVXJTiles() {
  if (_numTilesAlloc) {
    free(excl);
    free(atomStart);
    free(status);
  }
}

void AVXJTiles::_realloc() {
  if (_numTilesAlloc) {
    free(excl);
    free(atomStart);
    free(status);
  }
  _numTilesAlloc = 1.2 * _numTiles;
  int e = posix_memalign((void **)&excl, 64, 
			 sizeof(unsigned int)*_numTilesAlloc << 4);
  e = e | posix_memalign((void **)&atomStart, 64, sizeof(int)*_numTilesAlloc);
  e = e | posix_memalign((void **)&status, 64, sizeof(int)*_numTilesAlloc);
  if (e) NAMD_die("Could not allocate memory for tiles.");
}

//---------------------------------------------------------------------------

AVXTileLists::AVXTileLists() : _numLists(0), _numListsAlloc(0), 
                               _numModifiedAlloc(0), _numExcludedAlloc(0),
                               _maxPairLists(0), _maxPairs(0)
{
  #ifndef MEM_OPT_VERSION
  _exclFlyListBuffer = 0;
  #endif
}

//---------------------------------------------------------------------------

AVXTileLists::~AVXTileLists() {
  if (_numListsAlloc) {
    free(lists);
    free(listDepth);
  }
  if (_numModifiedAlloc) {
    free(_modified_i);
    free(_modified_j);
  }
  if (_numExcludedAlloc) {
    free(_excluded_i);
    free(_excluded_j);
  }
  if (NAMD_AVXTILES_PAIR_THRESHOLD > 0) {
    if (_maxPairLists) {
      free(_pair_i);
      free(_numPairs);
      free(_pairStart);
      free(_pairLists);
    }
  }
  #ifndef MEM_OPT_VERSION
  delete []_exclFlyListBuffer;
  #endif
}

//---------------------------------------------------------------------------

void AVXTileLists::setSimParams(const float scale, const float scale14,
				const float c1, const float c3,
				const float switchOn2, float *fastTable,
				float *fastEnergyTable, float *slowTable,
				float *slowEnergyTable, float *eps4sigma,
				float *eps4sigma14, float *ljTable, 
				const float ljTableWidth, float *modifiedTable,
				float *modifiedEnergyTable,
				float *excludedTable,
				float *excludedEnergyTable,
				const int interpolationMode) {
  _interpolationMode = interpolationMode;

  if (_interpolationMode > 1)
    _paramScale = 2.0 * scale;
  else
    _paramScale = scale;
  _paramScale14 = scale14;
  _paramC1 = c1;
  _paramC3 = c3;
  _paramSwitchOn2 = switchOn2;
  if (_interpolationMode == 3) {
    _paramFastTable = fastTable;
    _paramFastEnergyTable = fastEnergyTable;
  }
  _paramSlowTable = slowTable;
  _paramSlowEnergyTable = slowEnergyTable;
  _paramModifiedTable = modifiedTable;
  _paramModifiedEnergyTable = modifiedEnergyTable;
  _paramExcludedTable = excludedTable;
  _paramExcludedEnergyTable = excludedEnergyTable;
  if (_interpolationMode > 1) {
    _paramLjTable = ljTable;
    _paramLjWidth = ljTableWidth;
  } else {
    _paramEps4Sigma = eps4sigma;
    _paramEps4Sigma14 = eps4sigma14;
  }
}

//---------------------------------------------------------------------------

// Build tiles lists based on bounding boxes
// - count is true if we are only counting and not storing lists
//     * used first time and when in parition mode
//     * if count is false, can still fail due to insufficient storage
//     * requiring realloc and repeat
// - paritionMode is true if the compute is partitioned for LB
//     * in this case, first call with count=true to paritition based on
//     * number of neighbor tiles. Then reallocate and build only for i-tiles
//     * active in partition
template <bool count, bool partitionMode>
int AVXTileLists::_buildBB() {
  // Get a count of J tiles for initial allocation or for partitioning
  // If we already ran w/ count==true, guaranteed to have enough storage
  // If we already ran w/ count==true for paritions, listDepth is valid
  if (!count)
    if (partitionMode || jTiles.maxTiles() == 0)
      _buildBB<true, partitionMode>();

  int numJtiles = 0;
  const int maxJtiles = jTiles.maxTiles();
  const int numTileLists = tiles_p0->numTiles();
  const int numTiles2 = tiles_p1->numTiles();
  const bool zeroShift = ! (_shx*_shx + _shy*_shy + _shz*_shz > 0);
  const bool self = zeroShift && (tiles_p0 == tiles_p1);

  int out_i;
  if (!count) out_i = 0;

  for (int itileList = 0; itileList < numTileLists; itileList++) {
    if (partitionMode && count==false && listDepth[itileList]==0)
      continue;

    int itileListLen = 0;
    int jinit;
    if (self) jinit = itileList;
    else jinit = 0;

    // Load i-atom data (and shift coordinates)

    // Loading bounding boxes and shift
    const __m512 bbIx = _mm512_set1_ps(tiles_p0->bbox_x[itileList] + _shx);
    const __m512 bbIy = _mm512_set1_ps(tiles_p0->bbox_y[itileList] + _shy);
    const __m512 bbIz = _mm512_set1_ps(tiles_p0->bbox_z[itileList] + _shz);
    const __m512 bbIwx = _mm512_set1_ps(tiles_p0->bbox_wx[itileList]);
    const __m512 bbIwy = _mm512_set1_ps(tiles_p0->bbox_wy[itileList]);
    const __m512 bbIwz = _mm512_set1_ps(tiles_p0->bbox_wz[itileList]);

    // Starting indices for jtiles
    __m512i jAtomStart = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
					   128, 144, 160, 176, 192, 208, 224,
					   240);
    jAtomStart = _mm512_add_epi32(jAtomStart, _mm512_set1_epi32(jinit<<4));
    int jtileStart = numJtiles;

    __mmask16 loopMask = 0xFFFF;
    for (int j=jinit; j < numTiles2; j += 16) {
      // Remainder Predication
      if (numTiles2 - j < 16) 
	loopMask >>= (16 - (numTiles2 - j));

      // Load bounding boxes for j-tiles
      const __m512 bbJx = _mm512_mask_loadu_ps(_mm512_undefined_ps(), 
        loopMask, tiles_p1->bbox_x + j);
      const __m512 bbJy = _mm512_mask_loadu_ps(_mm512_undefined_ps(),
        loopMask, tiles_p1->bbox_y + j);
      const __m512 bbJz = _mm512_mask_loadu_ps(_mm512_undefined_ps(),
        loopMask, tiles_p1->bbox_z + j);
      const __m512 bbJwx = _mm512_mask_loadu_ps(_mm512_undefined_ps(),
        loopMask, tiles_p1->bbox_wx + j);
      const __m512 bbJwy = _mm512_mask_loadu_ps(_mm512_undefined_ps(),
        loopMask, tiles_p1->bbox_wy + j);
      const __m512 bbJwz = _mm512_mask_loadu_ps(_mm512_undefined_ps(),
        loopMask, tiles_p1->bbox_wz + j);

      // dx = max(0.f,abs(bbIx - bbJx) - bbIwx - bbJwx)
      const __m512 dx_one = _mm512_abs_ps(_mm512_sub_ps(bbIx, bbJx));
      const __m512 dx_two = _mm512_add_ps(bbIwx, bbJwx);
      const __mmask16 lxmask = _mm512_cmplt_ps_mask(dx_two, dx_one);
      const __m512 dx = _mm512_mask_sub_ps(_mm512_setzero_ps(), lxmask,
					   dx_one, dx_two);
      // dy = max(0.f,abs(bbIy - bbJy) - bbIwy - bbJwy)
      const __m512 dy_one = _mm512_abs_ps(_mm512_sub_ps(bbIy, bbJy));
      const __m512 dy_two = _mm512_add_ps(bbIwy, bbJwy);
      const __mmask16 lymask = _mm512_cmplt_ps_mask(dy_two, dy_one);
      const __m512 dy = _mm512_mask_sub_ps(_mm512_setzero_ps(), lymask,
					   dy_one, dy_two);
      // dz = max(0.f,abs(bbIz - bbJz) - bbIwz - bbJwz)
      const __m512 dz_one = _mm512_abs_ps(_mm512_sub_ps(bbIz, bbJz));
      const __m512 dz_two = _mm512_add_ps(bbIwz, bbJwz);
      const __mmask16 lzmask = _mm512_cmplt_ps_mask(dz_two, dz_one);
      const __m512 dz = _mm512_mask_sub_ps(_mm512_setzero_ps(), lzmask,
					   dz_one, dz_two);
      // r2bb = dx*dx + dy*dy + dz*dz
      const __m512 r2bb = _mm512_fmadd_ps(dz,dz,
	_mm512_fmadd_ps(dx, dx, _mm512_mul_ps(dy, dy)));
      // Count lanes with tiles within the cutoff
      const __mmask16 keep = _mm512_cmplt_ps_mask(r2bb,
	_mm512_set1_ps(_plcutoff2)) & loopMask;
      const int nKeep = _popcnt32(keep);
      // Store neighbor tiles within cutoff
      if (count == false && nKeep)
	if (partitionMode || jtileStart + itileListLen + nKeep <= maxJtiles)
	  _mm512_mask_compressstoreu_epi32(jTiles.atomStart+jtileStart+
					   itileListLen, keep, jAtomStart);

      itileListLen += nKeep;
      // Next set of j-tile atoms
      jAtomStart = _mm512_add_epi32(jAtomStart, _mm512_set1_epi32(256));
    }
    
    if (count == false) {
      if (itileListLen > 0) {
	listDepth[out_i] = (unsigned int)itileListLen;
	lists[out_i].atomStart_i = itileList<<4;
	lists[out_i].jtileStart = jtileStart;
	out_i++;
      } 
    } else if (partitionMode)
      listDepth[itileList] = (unsigned int)itileListLen;

    numJtiles += itileListLen;
  } // for (int itileList

  // Divide i-tiles between partitions to approximately evenly divide # total
  // number of neighbor tiles between partitions
  if (partitionMode && count) {
    const int jtilesPerPart = static_cast<double>(numJtiles)/_numParts + 0.5;
    int jtileCount = 0, curPart = 0, nextPart = jtilesPerPart;
    numJtiles = 0;

    for (int j = 0; j < numTileLists; j++) {
      jtileCount += listDepth[j];
      if (curPart < _minPart || curPart >= _maxPart)
	listDepth[j] = 0;
      else
	numJtiles += listDepth[j];
      if (jtileCount >= nextPart) {
	curPart++;
	nextPart += jtilesPerPart;
	if (curPart == _numParts) curPart--;
      }
    }
  }

  if (count) {
    jTiles.realloc(numJtiles);
    return numJtiles;
  }

  // If we are in partition mode, guartaneed we had enough storage, else
  // reallocate and repeat the calculation if we ran out of room
  if (!partitionMode && jTiles.realloc(numJtiles))
    return _buildBB<false, false>();

  _numLists = out_i;
  if (_numLists) {
    // Set all neighbor tiles as active, refined in force loop
    memset(jTiles.status, 0, numJtiles * sizeof(int));
    lists[_numLists].jtileStart = lists[_numLists-1].jtileStart + 
      listDepth[_numLists - 1];
  }

  return numJtiles;
}

void AVXTileLists::delEmptyLists() {
  int out_i = 0;
  int totalTiles = 0;
  const int numLists = _numLists;

  __mmask16 tripMask = 0xFFFF;
  int vecTrip = 16;

  for (int ii = 0; ii < numLists; ii += 16) {
    if (numLists - ii < 16) {
      vecTrip = numLists - ii;
      tripMask >>= (16 - vecTrip);
    }

    // Load number of neighbor tiles for each i-tile (depth)
    __m512i depth = _mm512_loadu_epi32(listDepth + ii);
    // Loop predication
    __mmask16 depth_mask = _mm512_cmpneq_epi32_mask(depth,
      _mm512_setzero_epi32()) & tripMask;
    // depth &= 65535
    depth = _mm512_and_epi32(depth, _mm512_set1_epi32((int)65535));
    // Store only active tiles
    _mm512_mask_compressstoreu_epi32(listDepth + out_i, depth_mask, depth);

    for (int vi = 0; vi < vecTrip; vi++) {
      if (*((unsigned int*)&(depth) + vi)) {
	const int i = ii + vi;
	lists[out_i].atomStart_i = lists[i].atomStart_i;

	int start = totalTiles;
	int startOld = lists[i].jtileStart;
	int endOld = lists[i+1].jtileStart;
	lists[out_i].jtileStart = start;

	int jtile0 = start;
	for (int jtileOld=startOld; jtileOld < endOld; jtileOld++) {
	  int jtile = jTiles.status[jtileOld];
	  jtile &= 65535;
	  if (jtile) {
	    jTiles.atomStart[jtile0] = jTiles.atomStart[jtileOld];
	    // Pack tile exclusion data for active tiles
	    __m512i excl;
	    excl = _mm512_loadu_epi32(jTiles.excl + (jtileOld<<4));
	    _mm512_store_epi32((void *)(jTiles.excl + (jtile0<<4)), excl);
      	    jtile0++;
	  }
	}
	out_i++;
	totalTiles += (*((unsigned int*)&(depth) + vi));
      }
    }
  }
  jTiles.realloc(totalTiles);

  _numLists = out_i;
  if (_numLists)
    lists[_numLists].jtileStart = lists[_numLists-1].jtileStart + 
      listDepth[_numLists - 1];
}

void AVXTileLists::build() {
  tiles_p0->buildBoundingBoxes(_lastBuild);
  if (tiles_p0 != tiles_p1)
    tiles_p1->buildBoundingBoxes(_lastBuild);

  if (_maxPart - _minPart < _numParts)
    _buildBB<false,true>();
  else
    _buildBB<false,false>();

  // if hybrid tiles/pairs, set indices for i-atoms for any pairlists
  if (NAMD_AVXTILES_PAIR_THRESHOLD > 0) {
    for (int i = 0; i < _maxPairLists; i++) _numPairs[i] = 0;
    int end = tiles_p0->numAtoms() & 15;
    if (end <= NAMD_AVXTILES_PAIR_THRESHOLD) {
      const int start = tiles_p0->numAtoms() - end;
      end += start;
      for (int i = start; i < end; i++) _pair_i[i-start] = i;
    } 
  }
}

void AVXTileLists::_realloc() {
  if (_numListsAlloc) {
    free(lists);
    free(listDepth);
  }
  _numListsAlloc = 1.2 * _numLists;
  int e = posix_memalign((void **)&lists, 64, sizeof(List)*(_numListsAlloc+1));
  e = e | posix_memalign((void **)&listDepth, 64,
		 sizeof(unsigned int)*(_numListsAlloc+1));
  if (e) NAMD_die("Could not allocate memory for tiles.");

  if (_numModifiedAlloc == 0) {
    int newNum = tiles_p0->numAtoms();
    if (newNum < tiles_p1->numAtoms()) newNum = tiles_p1->numAtoms();
    reallocModified(2 * newNum);
    // WMB: Below is only needed for full electrostatics
    reallocExcluded(2.2 * newNum);
  }
  if (NAMD_AVXTILES_PAIR_THRESHOLD > 0)
    _reallocPairLists(tiles_p0->numAtoms() & 15, NAMD_AVXTILES_IPAIRCOUNT);
}

//---------------------------------------------------------------------------

void AVXTileLists::_reallocModified() {
  int numModifiedAlloc = 1.2 * _numModified;
  int *newp;

  if (posix_memalign((void **)&newp, 64, sizeof(int) * numModifiedAlloc))
    NAMD_die("Could not allocate memory for tiles.");

  if (_numModifiedAlloc) { 
    memcpy(newp, _modified_i, sizeof(int) * _numModifiedAlloc);
    free(_modified_i);
  }
  _modified_i = newp;

  if (posix_memalign((void **)&newp, 64, sizeof(int) * numModifiedAlloc))
    NAMD_die("Could not allocate memory for tiles.");
    
  if (_numModifiedAlloc) { 
    memcpy(newp, _modified_j, sizeof(int) * _numModifiedAlloc);
    free(_modified_j);
  }
  _modified_j = newp;

  _numModifiedAlloc = numModifiedAlloc;
}

//---------------------------------------------------------------------------

void AVXTileLists::_reallocExcluded() {
  int numExcludedAlloc = 1.2 * _numExcluded;
  int *newp;

  if (posix_memalign((void **)&newp, 64, sizeof(int) * numExcludedAlloc))
    NAMD_die("Could not allocate memory for tiles.");

  if (_numExcludedAlloc) { 
    memcpy(newp, _excluded_i, sizeof(int) * _numExcludedAlloc);
    free(_excluded_i);
  }
  _excluded_i = newp;

  if (posix_memalign((void **)&newp, 64, sizeof(int) * numExcludedAlloc))
    NAMD_die("Could not allocate memory for tiles.");

  if (_numExcludedAlloc) { 
    memcpy(newp, _excluded_j, sizeof(int) * _numExcludedAlloc);
    free(_excluded_j);
  }
  _excluded_j = newp;

  _numExcludedAlloc = numExcludedAlloc;
}

//---------------------------------------------------------------------------

void AVXTileLists::_reallocPairLists(const int numPairLists, 
				     const int maxPairs) {
  if (NAMD_AVXTILES_PAIR_THRESHOLD > 0) {
    if (!_maxPairLists) {
      _maxPairLists = NAMD_AVXTILES_PAIR_THRESHOLD;
      _maxPairs = maxPairs;
      int e = posix_memalign((void **)&_pair_i, 64, sizeof(int)*_maxPairLists);
      e = e | posix_memalign((void **)&_numPairs, 64, 
			     sizeof(int)*_maxPairLists);
      e = e | posix_memalign((void **)&_pairStart, 64,
			     sizeof(int)*_maxPairLists);
      e = e | posix_memalign((void **)&_pairLists, 64,
		     sizeof(int)*_maxPairLists*_maxPairs);
      if (e) NAMD_die("Could not allocate memory for tiles.");

      int listStart = 0;
      for (int i = 0; i < _maxPairLists; i++) {
	_pairStart[i] = listStart;
	listStart += _maxPairs;
      }
    } else if (maxPairs > _maxPairs) {
      int *hold;
      if (posix_memalign((void **)&hold, 64, 
			 sizeof(int)*_maxPairLists*maxPairs))
	NAMD_die("Could not allocate memory for tiles.");

      int listStart = 0;
      for (int i = 0; i < _maxPairLists; i++) {
	if (_numPairs[i]) memcpy(hold + listStart, _pairLists + _pairStart[i],
				 sizeof(int) * _numPairs[i]);
	_pairStart[i] = listStart;
	listStart += maxPairs;
      }
      free(_pairLists);
      _pairLists = hold;
      _maxPairs = maxPairs;
    }
  }
}

#endif // NAMD_AVXTILES
