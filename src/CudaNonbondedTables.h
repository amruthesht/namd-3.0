#ifndef CUDANONBONDEDTABLES_H
#define CUDANONBONDEDTABLES_H

#ifdef NAMD_CUDA
#include <cuda.h>
#endif
#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif

#include "HipDefines.h"


#if defined(NAMD_CUDA) || defined(NAMD_HIP)

class CudaNonbondedTables {
private:
  const int deviceID;

  float2 *vdwCoefTable;
  int vdwCoefTableWidth;
#ifndef USE_TABLE_ARRAYS
  cudaTextureObject_t vdwCoefTableTex;
#endif
  int forceAndEnergyTableSize;

  // Non-bonded
#ifndef USE_TABLE_ARRAYS
  cudaArray_t forceArray;
  cudaTextureObject_t forceTableTex;
#endif
  float4* forceTable;

#ifndef USE_TABLE_ARRAYS
  cudaArray_t energyArray;
  cudaTextureObject_t energyTableTex;
#endif
  float4* energyTable;

  // Modified exclusions
#ifndef USE_TABLE_ARRAYS
  cudaArray_t modifiedExclusionForceArray;
  cudaTextureObject_t modifiedExclusionForceTableTex;
#endif
  float4* modifiedExclusionForceTable;
 
#ifndef USE_TABLE_ARRAYS 
  cudaArray_t modifiedExclusionEnergyArray;
  cudaTextureObject_t modifiedExclusionEnergyTableTex;
#endif
  float4* modifiedExclusionEnergyTable;

  // Exclusions
  float2 *exclusionVdwCoefTable;
#ifndef USE_TABLE_ARRAYS
  cudaTextureObject_t exclusionVdwCoefTableTex;
#endif

  // cudaArray_t exclusionForceArray;
  // cudaTextureObject_t exclusionForceTableTex;
  
  // cudaArray_t exclusionEnergyArray;
  // cudaTextureObject_t exclusionEnergyTableTex;

  float4* exclusionTable;
  float* r2_table;
#ifndef USE_TABLE_ARRAYS
  cudaTextureObject_t exclusionTableTex;
  cudaTextureObject_t r2_table_tex;
#endif

  void buildVdwCoefTable(bool update=false);
  void buildForceAndEnergyTables(int tableSize);

public:
  CudaNonbondedTables(const int deviceID);
  ~CudaNonbondedTables();

  float2* getVdwCoefTable() {return vdwCoefTable;}
  int getVdwCoefTableWidth() {return vdwCoefTableWidth;}
  int getForceAndEnergyTableSize() {return forceAndEnergyTableSize;}
#ifndef USE_TABLE_ARRAYS
  cudaTextureObject_t getVdwCoefTableTex() {return vdwCoefTableTex;}
  cudaTextureObject_t getForceTableTex() {return forceTableTex;}
  cudaTextureObject_t getEnergyTableTex() {return energyTableTex;}
#endif
  float4* getForceTable() {return forceTable;}
  float4* getEnergyTable() {return energyTable;}

  void updateTables();

  float2* getExclusionVdwCoefTable() {return exclusionVdwCoefTable;}
#ifndef USE_TABLE_ARRAYS
  cudaTextureObject_t getExclusionVdwCoefTableTex() {return exclusionVdwCoefTableTex;}
  // cudaTextureObject_t getExclusionForceTableTex() {return exclusionForceTableTex;}
  // cudaTextureObject_t getExclusionEnergyTableTex() {return exclusionEnergyTableTex;}
#endif
  float4* getExclusionTable() {return exclusionTable;}
  float* get_r2_table() {return r2_table;}
#ifndef USE_TABLE_ARRAYS
  cudaTextureObject_t getExclusionTableTex() {return exclusionTableTex;}
  cudaTextureObject_t get_r2_table_tex() {return r2_table_tex;}
#endif

#ifndef USE_TABLE_ARRAYS
  cudaTextureObject_t getModifiedExclusionForceTableTex() {return modifiedExclusionForceTableTex;}
  cudaTextureObject_t getModifiedExclusionEnergyTableTex() {return modifiedExclusionEnergyTableTex;}
#endif
  float4* getModifiedExclusionForceTable() {return modifiedExclusionForceTable;}
  float4* getModifiedExclusionEnergyTable() {return modifiedExclusionEnergyTable;}

};

#endif // NAMD_CUDA
#endif // CUDANONBONDEDTABLES_H
