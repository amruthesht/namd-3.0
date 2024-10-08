/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/**
 * AtomMap maps the global atom ID (int) to the atom's assigned
 * patch ID (int) and local index (int) within that patch.
 * An array of "LocalID" of length (number of atoms) is allocated.
 * The total space required is 2*sizeof(int)*(number of atoms).
 */

#ifndef ATOMMAP_H
#define ATOMMAP_H

#include "NamdTypes.h"

#ifdef MEM_OPT_VERSION

struct AtomMapEntry {
  AtomMapEntry *next;
  int pid;
  short index;
  short aid_upper;
};

#endif

enum { notUsed = -1 };

class AtomMap
{
public:
  static AtomMap *Instance();
  inline static AtomMap *Object() { return CkpvAccess(AtomMap_instance); }
  inline static AtomMap *ObjectOnPe(int pe) {
    return CkpvAccessOther(AtomMap_instance, CmiRankOf(pe));
  }
  ~AtomMap(void);
  void checkMap();

  void allocateMap(int nAtomIDs);

  LocalID localID(AtomID id);

  friend class AtomMapper;
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#ifdef BONDED_CUDA
  friend class ComputeBondedCUDA;
#endif
#endif

protected:
  AtomMap(void);

private:
  int registerIDsCompAtomExt(PatchID pid, const CompAtomExt *begin, const CompAtomExt *end);
  int registerIDsFullAtom(PatchID pid, const FullAtom *begin, const FullAtom *end);
  int unregisterIDsCompAtomExt(PatchID pid, const CompAtomExt *begin, const CompAtomExt *end);
  int unregisterIDsFullAtom(PatchID pid, const FullAtom *begin, const FullAtom *end);

#ifdef MEM_OPT_VERSION
  AtomMapEntry **entries;
  bool onlyUseTbl;
#endif

  LocalID *localIDTable;
  int tableSz;

};

#ifndef MEM_OPT_VERSION
//----------------------------------------------------------------------
// LocalID contains patch pid and local patch atom index
// for a given global atom number
inline LocalID AtomMap::localID(AtomID id)
{
  return localIDTable[id];
}
#endif


class AtomMapper {
public:
  AtomMapper(PatchID _pid) : pid(_pid), mapped(0), map(AtomMap::Object()) {}
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
#ifdef BONDED_CUDA
  AtomMapper(PatchID _pid, AtomMap *_map) : pid(_pid), mapped(0), map(_map) {}
#endif
#endif
  ~AtomMapper() {
    if ( mapped ) NAMD_bug("deleted AtomMapper with atoms still mapped");
  }
  void registerIDsCompAtomExt(const CompAtomExt *begin, const CompAtomExt *end);
  void registerIDsFullAtom(const FullAtom *begin, const FullAtom *end);
  void unregisterIDsCompAtomExt(const CompAtomExt *begin, const CompAtomExt *end);
  void unregisterIDsFullAtom(const FullAtom *begin, const FullAtom *end);

private:
  const PatchID pid;
  int mapped;
  AtomMap *map;
#ifdef MEM_OPT_VERSION
  ResizeArray<AtomMapEntry> entries;
#endif
};


#endif /* ATOMMAP_H */

