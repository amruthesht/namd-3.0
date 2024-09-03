#ifndef CUDARECORD_H
#define CUDARECORD_H

#ifdef NAMD_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif

#include "HipDefines.h"

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
struct PatchRecord {
  int atomStart;
  int numAtoms;
  int patchID;
};

struct CudaPeerRecord {
  int deviceIndex;
  int patchIndex;
  int bufferOffset;
  int bufferOffsetNBPad;  // Buffer offset with nonbonded padding
};
// 16 bytes

// Cuda Local Record should be 128 bytes so it can be loaded in a single cache line
// inline_peers are used to avoid another level of redirection. However, we'll maintain
// the array of CudaPeerRecords anyways to handle cases when a HomePatch has more than 
// 6 proxies. David Clark doesn't think this will happen, but a patch can technically have
// 13 proxies given we are only looking 1 patch away and taking adcantage of upstream/downstream
// logic
struct CudaLocalRecord {
  static constexpr int num_inline_peer = 5;
  int patchID; 
  int bufferOffset;    
  int bufferOffsetNBPad;  // Buffer offset with nonbonded padding
  int numAtoms;
  int numAtomsNBPad;  // Atom count including the nonbonded padding
  int isProxy;  // Indicates if this local record represent a proxy or home patch
  int numPeerRecord;  // The number of CudaPeerRecords corresponding to this patch
  int peerRecordStartIndex;
  // These four fields are used during GPU atom migration to keep track 
  int bufferOffsetOld; 
  int numMigrationGroups;
  int numAtomsNew;
  int numAtomsLocal;
  // These are used to pad the CudaLocalRecord to 128 bytes and avoid a global memory lookup
  CudaPeerRecord inline_peers[num_inline_peer];
};

static_assert(sizeof(CudaLocalRecord) == 128, "CudaLocalRecord must be 128 bytes");

#endif // NAMD_CUDA

struct CudaAtom {
  float x,y,z,q;
};

struct CudaForce {
  float x, y, z;
};

#endif //CUDARECORD_H
