/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/*
   Defines a new stream: iout, for "i"nforming consoles.
*/

#include "InfoStream.h"
#include "charm++.h"
#include "Vector.h"
#include "Tensor.h"
#include <stdio.h>

CkpvExtern(infostream, iout_obj);

std::ostream& infostream::access_iout_obj() {
  return CkpvAccess(iout_obj);
}

class CkPrintf_stringbuf : public std::stringbuf {
protected:
  virtual int sync();
};

infostream::infostream() : std::ostream(new CkPrintf_stringbuf) {}

infostream::~infostream() {
  // rdbuf() returns a pointer to the new'ed CkPrintf_stringbuf object
  // https://www.cplusplus.com/reference/ios/ios/rdbuf/
  delete rdbuf();
}

/* output using CkPrintf() (end by inform) */
int CkPrintf_stringbuf::sync() {
  // Even though str() returns a *copy* of the string, the RHS rules 
  // should avoid doing two copies for assignment to infostr.
  //
  // Note that str().c_str() should be avoided due to possibility for
  // leaving a dangling pointer or undefined behavior at best, e.g.,
  // https://stackoverflow.com/questions/21034834/is-there-issue-will-stringstream-str-c-str
  // Copying into a named temporary is considered good practice.
  std::string infostr = str();
  CkPrintf("%s",infostr.c_str());
#ifndef NAMD_NO_STDOUT_FLUSH
  fflush(stdout);  // since CkPrintf doesn't always flush
#endif
  str("");
  return 0;
}

std::ostream& endi(std::ostream& s) {
  // flush() will cause overloaded sync() to be called
  // https://www.cplusplus.com/reference/ostream/ostream/flush/
  s.flush();
  return s;
}

std::ostream& iPE(std::ostream& s) {
  return s << "Pe(" << CkMyPe() << ')';
}

std::ostream& operator<<(std::ostream& strm, const Vector &v1) {
       strm << v1.x << " " << v1.y << " " << v1.z;
       return strm;
}

std::ostream& operator<<(std::ostream& strm, const Tensor &t1) {
       strm << t1.xx << " " << t1.xy << " " << t1.xz << " "
            << t1.yx << " " << t1.yy << " " << t1.yz << " "
            << t1.zx << " " << t1.zy << " " << t1.zz;
       return strm;
}

/** common messages **/
/** iINFO, iWARN, iERROR, iDEBUG provide initial headings. **/
/** iINFOF, iWARNF, iERRORF, iDEBUGF provide initial headings with file name
    and line numbers. **/
std::ostream& iINFO (std::ostream& s)  { return s << "Info: "; }
std::ostream& iWARN (std::ostream& s)  { return s << "Warning: "; }
std::ostream& iERROR(std::ostream& s)  { return s << "ERROR: "; }
std::ostream& iDEBUG(std::ostream& s)  { return s << "DEBUG: "; }


