#include "AVXTiles.h"

#include "Molecule.h"
#include "Node.h"

#ifdef NAMD_AVXTILES
#include <immintrin.h>

//---------------------------------------------------------------------------

AVXTiles::AVXTiles() : _numAtoms(0), _numTiles(0), _numTilesAlloc(0),
		       _lastBuild(-1) {
}

//---------------------------------------------------------------------------

AVXTiles::~AVXTiles() {
  if (_numTilesAlloc) {
    free(forces);
    free(forcesSlow);
    free(vdwTypes);
    free(atomIndex);
#ifdef MEM_OPT_VERSION
    free(atomExclIndex);
#endif
    free(exclIndexStart);
    free(exclIndexMaxDiff);
    free(reverseOrder);
    free(bbox_x);
    free(bbox_y);
    free(bbox_z);
    free(bbox_wx);
    free(bbox_wy);
    free(bbox_wz);
  }
}

//---------------------------------------------------------------------------

void AVXTiles::_realloc() {
  if (_numTilesAlloc) {
    free(forces);
    free(forcesSlow);
    free(vdwTypes);
    free(atomIndex);
#ifdef MEM_OPT_VERSION
    free(atomExclIndex);
#endif
    free(exclIndexStart);
    free(exclIndexMaxDiff);
    free(reverseOrder);
    free(bbox_x);
    free(bbox_y);
    free(bbox_z);
    free(bbox_wx);
    free(bbox_wy);
    free(bbox_wz);
  }
  _touched = true;
  _numTilesAlloc = 1.2 * _numTiles;
  const int atomsAlloc = _numTilesAlloc << 4;
  int e = posix_memalign((void **)&forces, 64,
			 sizeof(AVXTilesForce)*atomsAlloc);
  e = e | posix_memalign((void **)&forcesSlow, 64, 
			 sizeof(AVXTilesForce)*atomsAlloc);
  e = e | posix_memalign((void **)&vdwTypes, 64, sizeof(int)*atomsAlloc);
  e = e | posix_memalign((void **)&atomIndex, 64 ,sizeof(int)*atomsAlloc);
#ifdef MEM_OPT_VERSION
  e = e | posix_memalign((void **)&atomExclIndex, 64 ,sizeof(int)*atomsAlloc);
#endif
  e = e | posix_memalign((void **)&exclIndexStart, 64, sizeof(int)*atomsAlloc);
  e = e | posix_memalign((void **)&exclIndexMaxDiff, 64, 
			 sizeof(int)*atomsAlloc);
  e = e | posix_memalign((void **)&reverseOrder, 64, sizeof(int)*atomsAlloc);
  e = e | posix_memalign((void **)&bbox_x, 64, sizeof(float)*_numTilesAlloc);
  e = e | posix_memalign((void **)&bbox_y, 64, sizeof(float)*_numTilesAlloc);
  e = e | posix_memalign((void **)&bbox_z, 64, sizeof(float)*_numTilesAlloc);
  e = e | posix_memalign((void **)&bbox_wx, 64, sizeof(float)*_numTilesAlloc);
  e = e | posix_memalign((void **)&bbox_wy, 64, sizeof(float)*_numTilesAlloc);
  e = e | posix_memalign((void **)&bbox_wz, 64, sizeof(float)*_numTilesAlloc);

  if (e) NAMD_die("Could not allocate memory for tiles.");
}

//---------------------------------------------------------------------------

void AVXTiles::atomUpdate(const CompAtom *compAtom,
			  const CompAtomExt *compAtomExt) {
  Molecule* mol = Node::Object()->molecule;
  int numFreeAtoms = _numAtoms;

  #pragma ivdep
  for (int k=0; k < _numAtoms; ++k) {
    int j = compAtomExt[k].sortOrder;
    int type = compAtom[j].vdwType;
    vdwTypes[k] = type;
    const int id = compAtomExt[j].id;
    atomIndex[k] = id;
    if (compAtomExt[j].atomFixed) --numFreeAtoms;
#ifdef MEM_OPT_VERSION
    const int exclId = compAtomExt[j].exclId;
    atomExclIndex[k] = exclId;
    const ExclusionCheck *exclcheck = mol->get_excl_check_for_idx(exclId);
    exclIndexStart[k] = id + exclcheck->min;
    exclIndexMaxDiff[k] = id + exclcheck->max;
#else // ! MEM_OPT_VERSION
    const ExclusionCheck *exclcheck = mol->get_excl_check_for_atom(id);
    exclIndexStart[k] = exclcheck->min;
    exclIndexMaxDiff[k] = exclcheck->max;
#endif // MEM_OPT_VERSION
    reverseOrder[j] = k;
  }

  _numFreeAtoms = numFreeAtoms;
  const int end = _numTiles << 4;
  for (int i = _numAtoms; i < end; i++) vdwTypes[i] = 0;
}

//---------------------------------------------------------------------------

void AVXTiles::_buildBoundingBoxes(const int step) {
  _lastBuild = step;
  for (int tile = 0; tile < _numTiles; tile ++) {
    // Load x, y, and z for tile
    const float *iptr = (float *)(atoms + (tile << 4));
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
    const __m512 x = _mm512_castsi512_ps(_mm512_mask_blend_epi32(t9, t5i,
								 t7i));
    const __m512 y = _mm512_shuffle_f32x4(t5, t7, 0x4E);
    const __m512i t12 = _mm512_set_epi32(31,27,23,19,15,11,7,3,30,26,22,
				   18,14,10,6,2);
    const __m512 t13 = _mm512_permutex2var_ps(t0, t12, t1);
    const __m512i t14 = _mm512_set_epi32(30,26,22,18,14,10,6,2,31,27,23,
				   19,15,11,7,3);
    const __m512 t15 = _mm512_permutex2var_ps(t2, t14, t3);
    const __m512i t13i = _mm512_castps_si512(t13);
    const __m512i t15i = _mm512_castps_si512(t15);
    const __m512 z = _mm512_castsi512_ps(_mm512_mask_blend_epi32(t9, t13i,
								 t15i));

    // Min x for tile
    const float min_x = _mm512_reduce_min_ps(x);
    // Max x for tile
    const float max_x = _mm512_reduce_max_ps(x);
    // Min y for tile
    const float min_y = _mm512_reduce_min_ps(y);
    // Max y for tile
    const float max_y = _mm512_reduce_max_ps(y);
    // Min z for tile
    const float min_z = _mm512_reduce_min_ps(z);
    // Max z for tile
    const float max_z = _mm512_reduce_max_ps(z);
    
    bbox_x[tile] = 0.5f*(min_x + max_x);
    bbox_y[tile] = 0.5f*(min_y + max_y);
    bbox_z[tile] = 0.5f*(min_z + max_z);
    bbox_wx[tile] = 0.5f*(max_x - min_x);
    bbox_wy[tile] = 0.5f*(max_y - min_y);
    bbox_wz[tile] = 0.5f*(max_z - min_z);
  }
}

//---------------------------------------------------------------------------

template <int doSlow, int doVirial, int touched>
void AVXTiles::_nativeForceVirialUpdate(const CompAtom *p,
       const Vector &center, Force * __restrict__ nativeForces,
       Force * __restrict__ nativeForcesSlow,
       const Force * __restrict__ nativeForcesVirial, 
       const Force * __restrict__ nativeForcesSlowVirial,
       double virial[6], double virialSlow[6]) {

  BigReal v_xx, v_xy, v_xz, v_yy, v_yz, v_zz;
  BigReal vS_xx, vS_xy, vS_xz, vS_yy, vS_yz, vS_zz;
  if (doVirial) {
    v_xx = v_xy = v_xz = v_yy = v_yz = v_zz = 0.0;
    if (doSlow) vS_xx = vS_xy = vS_xz = vS_yy = vS_yz = vS_zz = 0.0;
  }
  
  const int * __restrict__ reverseOrder = this->reverseOrder;
  const AVXTilesForce * __restrict__ forces = this->forces;
  const AVXTilesForce * __restrict__ forcesSlow = this->forcesSlow;
  #pragma omp simd aligned(reverseOrder, forces, forcesSlow:64)	\
      reduction(+:v_xx, v_xy, v_xz, v_yy, v_yz, v_zz,		\
    	        vS_xx, vS_xy, vS_xz, vS_yy, vS_yz, vS_zz)
  for (int i = 0; i < _numAtoms; i++) {
    BigReal f_x = nativeForcesVirial[i].x;
    BigReal f_y = nativeForcesVirial[i].y;
    BigReal f_z = nativeForcesVirial[i].z;
    if (touched) {
      const int order = reverseOrder[i];
      f_x += forces[order].x;
      f_y += forces[order].y;
      f_z += forces[order].z;
    }
    BigReal p_x, p_y, p_z;
    if (doVirial) {
      p_x = p[i].position.x - center.x;
      p_y = p[i].position.y - center.y;
      p_z = p[i].position.z - center.z;
      v_xx += f_x * p_x;
      v_xy += f_x * p_y;
      v_xz += f_x * p_z;
      v_yy += f_y * p_y;
      v_yz += f_y * p_z;
      v_zz += f_z * p_z;
    }
    nativeForces[i].x += f_x;
    nativeForces[i].y += f_y;
    nativeForces[i].z += f_z;
    if (doSlow) {
      BigReal f_x = nativeForcesSlowVirial[i].x;
      BigReal f_y = nativeForcesSlowVirial[i].y;
      BigReal f_z = nativeForcesSlowVirial[i].z;
      if (touched) {
        const int order = reverseOrder[i];
	f_x += forcesSlow[order].x;
	f_y += forcesSlow[order].y;
	f_z += forcesSlow[order].z;
      }
      if (doVirial) {
	vS_xx += f_x * p_x;
	vS_xy += f_x * p_y;
	vS_xz += f_x * p_z;
	vS_yy += f_y * p_y;
	vS_yz += f_y * p_z;
	vS_zz += f_z * p_z;
      }
      nativeForcesSlow[i].x += f_x;
      nativeForcesSlow[i].y += f_y; 
      nativeForcesSlow[i].z += f_z;
    }
  }
  if (doVirial) {
    virial[0] = v_xx;
    virial[1] = v_xy;
    virial[2] = v_xz;
    virial[3] = v_yy;
    virial[4] = v_yz;
    virial[5] = v_zz;
    if (doSlow) {
      virialSlow[0] = vS_xx;
      virialSlow[1] = vS_xy;
      virialSlow[2] = vS_xz;
      virialSlow[3] = vS_yy;
      virialSlow[4] = vS_yz;
      virialSlow[5] = vS_zz;
    }
  }
}

//---------------------------------------------------------------------------

void AVXTiles::nativeForceVirialUpdate(const int doSlow, const int doVirial, 
       const CompAtom *p, const Vector &center,
       Force * __restrict__ nativeForces,
       Force * __restrict__ nativeForcesSlow,
       const Force * __restrict__ nativeForcesVirial, 
       const Force * __restrict__ nativeForcesSlowVirial,
       double virial[6], double virialSlow[6]) {

#define CALL(DOSLOW, DOVIRIAL, TOUCHED) \
  _nativeForceVirialUpdate<DOSLOW, DOVIRIAL, \
                           TOUCHED>(p, center, nativeForces, nativeForcesSlow,\
				    nativeForcesVirial, \
                                    nativeForcesSlowVirial, virial, \
                                    virialSlow);

  if (_touched) {
    if (!doSlow && !doVirial) CALL(0, 0, 1);
    if (!doSlow && doVirial) CALL(0, 1, 1);
    if (doSlow && doVirial) CALL(1, 1, 1);
    if (doSlow && !doVirial) CALL(1, 0, 1);
  } else {
    if (!doSlow && !doVirial) CALL(0, 0, 0);
    if (!doSlow && doVirial) CALL(0, 1, 0);
    if (doSlow && doVirial) CALL(1, 1, 0);
    if (doSlow && !doVirial) CALL(1, 0, 0);
  }
#undef CALL

}
#endif // NAMD_AVXTILES
