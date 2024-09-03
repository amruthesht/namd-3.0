//
// Tuple types that enable fast evaluation on GPU
//
#ifndef TUPLETYPESCUDA_H
#define TUPLETYPESCUDA_H

#ifdef NAMD_CUDA
#include <cuda_runtime.h>  // float3
#endif
#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif

#include "NamdTypes.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
struct CudaBond {
  int i, j, itype;
  // int ivir;
  float scale;
  float3 ioffsetXYZ;
  int fepBondedType;
};
struct CudaBondStage {
  enum { size = 2 };
  int itype;
  // int ivir;
  float scale;
  int fepBondedType;
  int patchIDs[size];
  int index[size];
};
// DMC it is fine to change the size of these structures but make sure you need to
// as we need to copy this data during migration
static_assert(sizeof(CudaBondStage) == 28, "CudaBondStage unexpected size");

struct CudaAngle {
  int i, j, k, itype;
  // int ivir, kvir;
  float scale;
  float3 ioffsetXYZ;
  float3 koffsetXYZ;
  int fepBondedType;
};
struct CudaAngleStage {
  enum { size = 3 };
  int itype;
  // int ivir, kvir;
  float scale;
  int fepBondedType;
  int index[size];
  int patchIDs[size];
};
static_assert(sizeof(CudaAngleStage) == 36, "CudaAngleStage unexpected size");

struct CudaDihedral {
  int i, j, k, l, itype;
  // int ivir, jvir, lvir;
  float scale;
  float3 ioffsetXYZ;
  float3 joffsetXYZ;
  float3 loffsetXYZ;
  int fepBondedType;
};
struct CudaDihedralStage {
  enum { size = 4 };
  int itype;
  // int ivir, jvir, lvir;
  float scale;
  int fepBondedType;
  int patchIDs[size];
  int index[size];
};
static_assert(sizeof(CudaDihedralStage) == 44, "CudaDihedralStage unexpected size");

struct CudaExclusion {
  int i, j, vdwtypei, vdwtypej;
  // int ivir;
  float3 ioffsetXYZ;
  int pswitch;
};
struct CudaExclusionStage {
  enum { size = 2 };
  int vdwtypei, vdwtypej;
  // int ivir;
  int pswitch;
  int patchIDs[size];
  int index[size];
};
static_assert(sizeof(CudaExclusionStage) == 28, "CudaExclusionStage unexpected size");

struct CudaCrossterm {
  int i1, i2, i3, i4, i5, i6, i7, i8, itype;
  float scale;
  float3 offset12XYZ;
  float3 offset23XYZ;
  float3 offset34XYZ;
  float3 offset56XYZ;
  float3 offset67XYZ;
  float3 offset78XYZ;
  int fepBondedType;
};
struct CudaCrosstermStage {
  enum { size = 8 };
  int itype;
  float scale;
  int fepBondedType;
  int patchIDs[size];
  int index[size];
};
static_assert(sizeof(CudaCrosstermStage) == 76, "CudaCrosstermStage unexpected size");

struct CudaBondValue {
  float k;   //  Force constant for the bond
  float x0;  //  Rest distance for the bond
  float x1;  //  Upper wall for harmonic wall potential (with x0 lower wall)
};

struct CudaAngleValue {
  float k;   //  Force constant for angle
  float theta0;  //  Rest angle for angle
  float k_ub;  //  Urey-Bradley force constant
  float r_ub;  //  Urey-Bradley distance
  int normal; // Whether we use harmonic (0) or cos-based (1) angle terms
};

struct CudaDihedralValue {
  float k;     //  Force constant
  float delta; //  Phase shift
  int n;       //  Periodicity*2, if n low bit is set to 0, this is the last in multiplicity
};

// struct CudaCrosstermData { float d00,d01,d10,d11; };

struct CudaCrosstermValue {
  enum {dim=24};
  float4 c[dim][dim][4]; // bicubic interpolation coefficients
};

// struct contains the boolean flags related to alchemical transformation
struct CudaAlchFlags {
  CudaAlchFlags():
    alchOn(false), alchFepOn(false), alchThermIntOn(false), alchWCAOn(false),
    alchLJcorrection(false), alchVdwForceSwitching(false),
    alchDecouple(false), alchBondDecouple(false) {}
  bool alchOn;
  bool alchFepOn;
  bool alchThermIntOn;
  bool alchWCAOn;
  bool alchLJcorrection;
  bool alchVdwForceSwitching;
  bool alchDecouple;
  bool alchBondDecouple;
};

// struct contains the constants value of alchemical transformation
// These values are not changed during the step update.
// But they may change by TCL script update.
struct CudaAlchParameters {
  float switchDist2; // = switchOn2
  // For soft-core potential
  float alchVdwShiftCoeff;
};

// struct contains the lambda values of alchemical transformation
// These values are considered to be changed and copied to GPU every step.
struct CudaAlchLambdas {
  float currentLambda;
  float currentLambda2;
  float bondLambda1;
  float bondLambda2;
  float bondLambda12;
  float bondLambda22;
  float elecLambdaUp;
  float elecLambda2Up;
  float elecLambdaDown;
  float elecLambda2Down;
  float vdwLambdaUp;
  float vdwLambda2Up;
  float vdwLambdaDown;
  float vdwLambda2Down;
};

static constexpr int kNumTupleTypes = 7;

struct TupleCounts {
  int bond;
  int angle;
  int dihedral;
  int improper;
  int modifiedExclusion;
  int exclusion;
  int crossterm;
};

struct TupleSizes {
  size_t bond;
  size_t angle;
  size_t dihedral;
  size_t improper;
  size_t modifiedExclusion;
  size_t exclusion;
  size_t crossterm;
};

struct TupleIntArrays {
  int* bond;
  int* angle;
  int* dihedral;
  int* improper;
  int* modifiedExclusion;
  int* exclusion;
  int* crossterm;
};

struct TupleIntArraysContiguous {
  int* data;  // Pointer to underlying buffer
  size_t offsets[kNumTupleTypes];

  NAMD_HOST_DEVICE int* bond() { return data + offsets[0]; }
  NAMD_HOST_DEVICE int* angle() { return data + offsets[1]; }
  NAMD_HOST_DEVICE int* dihedral() { return data + offsets[2]; }
  NAMD_HOST_DEVICE int* improper() { return data + offsets[3]; }
  NAMD_HOST_DEVICE int* modifiedExclusion() { return data + offsets[4]; }
  NAMD_HOST_DEVICE int* exclusion() { return data + offsets[5]; }
  NAMD_HOST_DEVICE int* crossterm() { return data + offsets[6]; }
};

struct TupleIntArraysPeer {
  int** bond;
  int** angle;
  int** dihedral;
  int** improper;
  int** modifiedExclusion;
  int** exclusion;
  int** crossterm;
};

struct TupleData {
  CudaBond* bond;
  CudaAngle* angle;
  CudaDihedral* dihedral;
  CudaDihedral* improper;
  CudaExclusion* modifiedExclusion;
  CudaExclusion* exclusion;
  CudaCrossterm* crossterm;
};

struct TupleDataStage {
  CudaBondStage* bond;
  CudaAngleStage* angle;
  CudaDihedralStage* dihedral;
  CudaDihedralStage* improper;
  CudaExclusionStage* modifiedExclusion;
  CudaExclusionStage* exclusion;
  CudaCrosstermStage* crossterm;
};

struct TupleDataStagePeer {
  CudaBondStage** bond;
  CudaAngleStage** angle;
  CudaDihedralStage** dihedral;
  CudaDihedralStage** improper;
  CudaExclusionStage** modifiedExclusion;
  CudaExclusionStage** exclusion;
  CudaCrosstermStage** crossterm;
};

#endif // NAMD_CUDA || NAMD_HIP

#endif // TUPLETYPESCUDA_H
