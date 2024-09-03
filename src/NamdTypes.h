/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef NAMDTYPES_H

#define NAMDTYPES_H

#ifdef NAMD_CUDA
#include <cuda_runtime.h>  // Include vector types
#endif
#ifdef NAMD_HIP
#include <hip/hip_runtime.h>
#endif

#include "common.h"
#include "Vector.h"
#include "CudaRecord.h"
#include "ResizeArray.h"

class Patch;
class Compute;

typedef Vector Position;
typedef Vector Velocity;

//#ifdef ARCH_POWERPC
//typedef AlignVector Force;
//#else
typedef Vector Force;
//#endif

typedef int32 AtomID;
typedef int32 AtomType;
typedef float Mass;
typedef float Charge;

typedef double Coordinate;

struct Transform
{
  int8 i,j,k;
  NAMD_HOST_DEVICE Transform(void) { i=0; j=0; k=0; }

#if defined(NAMD_CUDA) || defined(NAMD_HIP)
  // Allows for implicit conversion from char3
  NAMD_HOST_DEVICE Transform(char3 input)
    : i(input.x), j(input.y), k(input.z) { ; }

  // Allows for implicit conversion to char3
  NAMD_HOST_DEVICE operator char3() const {
     char3 res;
     res.x = i;
     res.y = j;
     res.z = k;
     return res;
  }
#endif

};

#ifndef USE_NO_BITFIELDS

/*
 * 1. "position" field in this structure is very important since it
 * needs to be sent to every patch after every timestep.
 * 2. Anything that is static (value is decided before computation)
 * or only changes after atom migration should be put into the CompAtomExt structure
 * 3. This data structure is 32-byte long which is particularly optimized for some machines
 * (including BG/L) for better cache and message performance. Therefore, changes
 * to this structure should be cautioned for the sake of performance.
 */

struct CompAtom {
  Position position;
  Charge charge;
  int16 vdwType;
  uint8 partition;
  uint8 nonbondedGroupSize : 3;
  /**< ngs is number of atoms, starting with parent,
   * that are all within 0.5*hgroupCutoff.
   * Value is reset before each force calculation.
   * XXX Looks like ngs is unused for CUDA and KNL kernels.
   * XXX Limited to 5.  Why?
   */
  uint8 hydrogenGroupSize : 4;  // could be 3 if unsigned
  uint8 isWater : 1;  // 0 = particle is not in water, 1 = is in water
};

#else

/*
 * Code below from Venkat, removing bitfields to avoid
 * compilation bugs in the post-2016 Intel compilers.
 */

struct CompAtom {
  Position position;
  Charge charge;
  int16 vdwType;
  uint8 partition;
  uint8 nonbondedGroupSize;
  /**< ngs is number of atoms, starting with parent,
   * that are all within 0.5*hgroupCutoff.
   * Value is reset before each force calculation.
   * XXX Looks like ngs is unused for CUDA and KNL kernels.
   * XXX Limited to 5.  Why?
   */
  uint8 hydrogenGroupSize;
  uint8 isWater;// 0 = particle is not in water, 1 = is in water

};

#endif // USE_NO_BITFIELDS

#ifdef NAMD_KNL
struct CompAtomFlt {
  FloatVector position;
  int32 vdwType;
};
#endif

//CompAtomExt is now needed even in normal case
//for changing the packed msg type related to
//ProxyPatch into varsize msg type where
// two types of proxy msgs (originally, the msg 
// for the step where atoms migrate (ProxyAllMsg), 
// and  the msg for normal steps (ProxyDataMsg))
// are declared as the same class (ProxyDataMsg).
// Note that in normal case, the class is needed
// just for passing the compilation, but not involved
// in the actual force calculation.
// --Chao Mei

typedef int32 SigIndex;
typedef int32 AtomSigID;
typedef int32 ExclSigID;

#ifndef USE_NO_BITFIELDS

#define NAMD_ATOM_ID_MASK      0x3FFFFFFFu
#define NAMD_ATOM_FIXED_MASK   0x40000000u
#define NAMD_GROUP_FIXED_MASK  0x80000000u

struct CompAtomExt {
  #if defined(NAMD_CUDA) || defined(NAMD_MIC) || defined(NAMD_AVXTILES) || defined(NAMD_HIP)
  int32 sortOrder;  // used to reorder atoms for CUDA
  #endif
  #ifdef MEM_OPT_VERSION
  int32 id;
  ExclSigID exclId;
  uint32 sigId : 30;  // AtomSigID sigId;
  #else
  uint32 id : 30;  // minimum for 100M atoms is 28 signed, 27 unsigned
  #endif
  uint32 atomFixed : 1;
  uint32 groupFixed : 1;
};

// Use this temporary structure when copying SOA back to AOS
// to avoid loop vectorization bug in patchReady_SOA.
struct CompAtomExtCopy {
  #if defined(NAMD_CUDA) || defined(NAMD_MIC) || defined(NAMD_AVXTILES) || defined(NAMD_HIP)
  int32 sortOrder;  // used to reorder atoms for CUDA
  #endif
  #ifdef MEM_OPT_VERSION
  int32 id;
  ExclSigID exclId;
  uint32 sigId; // must bitwise OR high order bits atomFixed, groupFixed
  #else
  uint32 id; // must bitwise OR high order bits atomFixed, groupFixed
  #endif
};

static_assert(sizeof(CompAtomExt)==sizeof(CompAtomExtCopy), "NAMD fails to compile due to sizeof(CompAtomExt) != sizeof(CompAtomExtCopy)");

#else

struct CompAtomExt {
  #if defined(NAMD_CUDA) || defined(NAMD_HIP) || defined(NAMD_MIC)
  int32 sortOrder;  // used to reorder atoms for CUDA
  #endif
  #ifdef MEM_OPT_VERSION
  int32 id;
  ExclSigID exclId;
  //int sigId : 30;  // AtomSigID sigId;
  int32 sigId;
  #else
  //int id : 30;  // minimum for 100M atoms is 28 signed, 27 unsigned
  int32 id;
  #endif
  uint8 atomFixed;
  uint8 groupFixed;
};

#endif // USE_NO_BITFIELDS

struct FullAtom : CompAtom, CompAtomExt{
  Velocity velocity;
  Position fixedPosition;
  double recipMass;
  /**< The reciprocal mass is set to 1/mass or to 0 for massless particles.
   * Calculating this apriori allows us to remove the divide instruction 
   * from the integration loops and the Langevin velocity updates. 
   */
  Mass mass;
  union{
      Real langevinParam;
#ifdef MEM_OPT_VERSION
      int32 hydVal;
#endif      
  };  
  int32 status;  ///< Atom status bit fields defined in structures.h
  Transform transform;
  int32 migrationGroupSize;
  Real rigidBondLength;

#ifdef MEM_OPT_VERSION
  int32 outputRank;
#endif

#ifdef MEM_OPT_VERSION
  //a HACK to re-sort FullAtom list used in Parallel IO
  //When every home patch processor receives its atoms list for a patch,
  //the atoms inside this patch may not sorted according to hydList value
  //To save space, use anonymous union data structure to share the space
  //of "langevinParam" to store "hydList" from an InputAtom and then sort the 
  //atom list. The "langevinParam" value is not initialized until home 
  //patch creation -Chao Mei
  int32 operator < (const FullAtom &a) const {
      return hydVal < a.hydVal;
  }
#endif
};

//InputAtom is used to contain the info of the atoms
//loaded into input processors.
struct InputAtom: FullAtom{
	bool isValid;
	int16 isGP;
	int16 isMP;
	int32 hydList;
	int32 GPID;
	int32 MPID;
    	
	int32 operator < (const InputAtom &a) const{
		return hydList < a.hydList;
	}
};

#if !(defined(__CUDACC__) || defined(__HIPCC__))
typedef ResizeArray<CudaAtom> CudaAtomList;
typedef ResizeArray<CompAtom> CompAtomList;
typedef ResizeArray<CompAtomExt> CompAtomExtList;
#ifdef NAMD_KNL
typedef ResizeArray<CompAtomFlt> CompAtomFltList;
#endif
typedef ResizeArray<FullAtom> FullAtomList;
typedef ResizeArray<InputAtom> InputAtomList;
typedef ResizeArray<Position> PositionList;
typedef ResizeArray<Velocity> VelocityList;
typedef ResizeArray<Force> ForceList;
typedef ResizeArray<Transform> TransformList;

typedef ResizeArray<AtomID> AtomIDList;
typedef ResizeArray<BigReal> BigRealList;
typedef ResizeArray<Real> RealList;
typedef float GBReal;
typedef ResizeArray<GBReal> GBRealList;
typedef ResizeArray<int> IntList;

typedef int32 PatchID;
typedef int32 ComputeID;
typedef int32 NodeID;

typedef ResizeArray<PatchID> PatchIDList;
typedef ResizeArray<Patch *> PatchList;

typedef ResizeArray<Compute *> ComputeList;

// See AtomMap
struct LocalID
{
  PatchID pid;
  int32 index;
};

typedef ResizeArray<NodeID> NodeIDList;

struct ExtForce {
  int32 replace;
  Force force;
  ExtForce() : replace(0) {;}
};


// DMK - Atom Sort
#if NAMD_ComputeNonbonded_SortAtoms != 0

  typedef struct __sort_entry {
    int32 index;  // Index of atom in CompAtom array
    BigReal sortValue;   // Distance of PAp from P0 (see calculation code)
  } SortEntry;

#endif

//This class represents a tree node of proxy spanning tree
//All pes in this array have the same "nodeID". In other words,
//all those pes are in the same physical node.
//This is a structure for adapting NAMD to multicore processors
struct proxyTreeNode{
    int32 nodeID;
    int32 *peIDs;
    int32 numPes;

    proxyTreeNode(){
        nodeID = -1;
        peIDs = NULL;
        numPes = 0;
    }
    proxyTreeNode(int nid, int numPes_, int *pes){
        nodeID = nid;
        numPes = numPes_;
        peIDs = new int[numPes];
        memcpy(peIDs, pes, sizeof(int)*numPes);
    }

    inline proxyTreeNode(const proxyTreeNode &n){
        nodeID = n.nodeID;
        numPes = n.numPes;
        if(numPes==0) {
            peIDs = NULL;
        }else{
            peIDs = new int[n.numPes];
            memcpy(peIDs, n.peIDs, sizeof(int)*numPes);
        }
    }
    inline proxyTreeNode &operator=(const proxyTreeNode &n){
        nodeID = n.nodeID;
        numPes = n.numPes;
        delete [] peIDs;
        if(numPes==0) {
            peIDs = NULL;
            return (*this);
        }
        peIDs = new int[n.numPes];
        memcpy(peIDs, n.peIDs, sizeof(int)*numPes);
        return (*this);
    }
    ~proxyTreeNode(){
        delete [] peIDs;
    }
};

typedef ResizeArray<proxyTreeNode> proxyTreeNodeList;
#endif // __CUDACC__

struct PatchDataSOA {

  unsigned char *buffer;

  double * pos_x;
  double * pos_y;
  double * pos_z;

  float *  charge;
  int32 *  vdwType;
  int32 *  partition;
  int32 *  nonbondedGroupSize;
  int32 *  hydrogenGroupSize;
  int32 *  isWater;

  int32 *  sortOrder;
  int32 *  unsortOrder;
  int32 *  id;
  int32 *  exclId;
  int32 *  sigId;
  int32 *  atomFixed;
  int32 *  groupFixed;

  double * vel_x;  ///< Jim recommends double precision velocity
  double * vel_y;
  double * vel_z;

  double * fixedPosition_x;
  double * fixedPosition_y;
  double * fixedPosition_z;

  double * recipMass;  ///< derived from mass
  float *  mass;
  float *  langevinParam;

  int32 *  status;

  int32 *  transform_i;
  int32 *  transform_j;
  int32 *  transform_k;

  int32 *  migrationGroupSize;

  float *  rigidBondLength;

  // derived quantities for Langevin damping
  float *  langScalVelBBK2;  ///< derived from langevinParam
  float *  langScalRandBBK2; ///< from langevinParam and recipMass

  // Gaussian distributed random numbers for Langevin damping
  float *  gaussrand_x;  ///< fill with Gaussian distributed random numbers
  float *  gaussrand_y;
  float *  gaussrand_z;

  // forces
  double * f_normal_x;
  double * f_normal_y;
  double * f_normal_z;
  double * f_nbond_x;
  double * f_nbond_y;
  double * f_nbond_z;
  double * f_slow_x;
  double * f_slow_y;
  double * f_slow_z;
  double * f_global_x;
  double * f_global_y;
  double * f_global_z;
  double * f_saved_nbond_x; /**< saved nonbonded force */
  double * f_saved_nbond_y; /**< saved nonbonded force */
  double * f_saved_nbond_z; /**< saved nonbonded force */
  double * f_saved_slow_x; /**< saved slow force */
  double * f_saved_slow_y; /**< saved slow force */
  double * f_saved_slow_z; /**< saved slow force */

  // temporary storage for rigid bond constraints
  double * velNew_x;  ///< temp storage for rigid bond constraints
  double * velNew_y;
  double * velNew_z;
  double * posNew_x;
  double * posNew_y;
  double * posNew_z;

  size_t numBytes;  ///< number of bytes allocated for soa_buffer
  int32 numAtoms;  ///< number of atoms
  int32 maxAtoms;  ///< max number of atoms available, multiple of MAXFACTOR

}; // PatchDataSOA



#endif /* NAMDTYPES_H */

