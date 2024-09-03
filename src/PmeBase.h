/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef PME_BASE_H__
#define PME_BASE_H__

#ifdef DEBUGM
#include <stdio.h>
#endif
#include <math.h>
#include "MathArray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct PmeGrid {
  int K1, K2, K3;
  int dim2, dim3;
  int order;
  int block1, block2, block3;
  int xBlocks, yBlocks, zBlocks;
#ifdef DEBUGM
  void print() {
    printf("K1=%d  K2=%d  K3=%d  dim2=%d  dim3=%d  order=%d\n"
        "block1=%d  block2=%d  block3=%d  xBlocks=%d  yBlocks=%d  zBlock=%d\n",
        K1, K2, K3, dim2, dim3, order, block1, block2, block3,
        xBlocks, yBlocks, zBlocks);
  }
#endif
};

struct PmeParticle {
  double x, y, z;
  double cg;
};

#define PME_MAX_EVALS 255 
typedef MathArray<double,7> PmeReduction;

#ifndef SQRT_PI
#define SQRT_PI 1.7724538509055160273 /* mathematica 15 digits*/
#endif

#endif
