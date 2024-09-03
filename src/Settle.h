/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef SETTLE_H__
#define SETTLE_H__

#include "Vector.h"
#include "Tensor.h"

/*

Oxygen at a0, hydrogens at b0, c0, center of mass at the origin.

                   |
                   |
                   |
                a0 .-------
                   |   |
                   |   ra
                   |   | 
---------------------------------------
           |       |        |
           rb      |---rc---|
           |       |        |
    b0 .-----------|        c0 .
  
SETTLE for step one of velocity verlet.
ref: positions before unconstrained step
mass: masses
pos: on input: positions after unconstrained step; on output: the new 
               positions.
vel: on input: velocities after unconstrained step; on output: the new
               velocities.
ra, rb, rc: canonical positions of water atoms; see above diagram
*/

/// initialize cached water properties
void settle1init(BigReal pmO, BigReal pmH, BigReal hhdist, BigReal ohdist,
                 BigReal &mO, BigReal &mH,
                 BigReal &mOrmT, BigReal &mHrmT, BigReal &ra,
                 BigReal &rb, BigReal &rc, BigReal &rra);

/// optimized settle1 algorithm, reuses water properties as much as possible
int settle1(const Vector *ref, Vector *pos, Vector *vel, BigReal invdt,
                 BigReal mOrmT, BigReal mHrmT, BigReal ra,
                 BigReal rb, BigReal rc, BigReal rra);

template <int veclen>
void settle1_SIMD(const Vector *ref, Vector *pos,
  BigReal mOrmT, BigReal mHrmT, BigReal ra,
  BigReal rb, BigReal rc, BigReal rra);

struct RattleParam {
  int ia;
  int ib;
  BigReal dsq;
  BigReal rma;
  BigReal rmb;
};

template <int veclen>
void rattlePair(const RattleParam* rattleParam,
  const BigReal *refx, const BigReal *refy, const BigReal *refz,
  BigReal *posx, BigReal *posy, BigReal *posz, bool& consFailure);

void rattleN(const int icnt, const RattleParam* rattleParam,
  const BigReal *refx, const BigReal *refy, const BigReal *refz,
  BigReal *posx, BigReal *posy, BigReal *posz,
  const BigReal tol2, const int maxiter,
  bool& done, bool& consFailure);
//#ifdef NAMD_MSHAKE
void MSHAKEIterate(const int icnt, const RattleParam* rattleParam,
  const BigReal *refx, const BigReal *refy, const BigReal *refz,
  BigReal *posx, BigReal *posy, BigReal *posz,
  const BigReal tol2, const int maxiter,
  bool& done, bool& consFailure);
void LINCS(const int icnt, const RattleParam* rattleParam,
  const BigReal *refx, const BigReal *refy, const BigReal *refz,
  BigReal *posx, BigReal *posy, BigReal *posz,
  const BigReal tol2, const int maxiter,
  bool& done, bool& consFailure);
//#endif
extern int settle2(BigReal mO, BigReal mH, const Vector *pos,
                   Vector *vel, BigReal dt, Tensor *virial); 

void settle1_SOA(
    const double * __restrict ref_x,
    const double * __restrict ref_y,
    const double * __restrict ref_z,
    double       * __restrict pos_x,
    double       * __restrict pos_y,
    double       * __restrict pos_z,
    int numWaters,
    BigReal mOrmT,
    BigReal mHrmT,
    BigReal ra,
    BigReal rb,
    BigReal rc,
    BigReal rra
    );

#endif
