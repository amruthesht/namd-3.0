/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef LATTICE_H
#define LATTICE_H

#include <stdlib.h>
#include "NamdTypes.h"
#include <math.h>
#include "Tensor.h"

typedef Vector ScaledPosition;

class Lattice
{
public:
  Lattice(void) : a1(0,0,0), a2(0,0,0), a3(0,0,0),
                  b1(0,0,0), b2(0,0,0), b3(0,0,0),
                  o(0,0,0), p1(0), p2(0), p3(0) {};

  // maps a transformation triplet onto a single integer
  NAMD_HOST_DEVICE static int index(int i=0, int j=0, int k=0)
  {
    return 9 * (k+1) + 3 * (j+1) + (i+1);
  }

  // sets lattice basis vectors but not origin (fixed center)
  NAMD_HOST_DEVICE void set(Vector A, Vector B, Vector C)
  {
    set(A,B,C,o);
  }

  // sets lattice basis vectors and origin (fixed center)
  NAMD_HOST_DEVICE void set(Vector A, Vector B, Vector C, Position Origin)
  {
    a1 = A; a2 = B; a3 = C; o = Origin;
    p1 = ( a1.length2() ? 1 : 0 );
    p2 = ( a2.length2() ? 1 : 0 );
    p3 = ( a3.length2() ? 1 : 0 );
    if ( ! p1 ) a1 = Vector(1.0,0.0,0.0);
    if ( ! p2 ) {
      Vector u1 = a1 / a1.length();
      Vector e_z(0.0,0.0,1.0);
      if ( fabs(e_z * u1) < 0.9 ) { a2 = cross(e_z,a1); }
      else { a2 = cross(Vector(1.0,0.0,0.0),a1); }
      a2 /= a2.length();
    }
    if ( ! p3 ) {
      a3 = cross(a1,a2);
      a3 /= a3.length();
    }
    if ( volume() < 0.0 ) a3 *= -1.0;
    recalculate();
  }

  // rescale lattice dimensions by factor, origin doesn't move
  NAMD_HOST_DEVICE void rescale(Tensor factor)
  {
    a1 = factor * a1;
    a2 = factor * a2;
    a3 = factor * a3;
    recalculate();
  }

  // rescale a position, keeping origin constant, assume 3D
  NAMD_HOST_DEVICE void rescale(Position &p, Tensor factor) const
  {
    p -= o;
    p = factor * p;
    p += o;
  }

  // transform scaled position to unscaled position
  NAMD_HOST_DEVICE Position unscale(ScaledPosition s) const
  {
    return (o + a1*s.x + a2*s.y + a3*s.z);
  }

  // transform unscaled position to scaled position
  NAMD_HOST_DEVICE ScaledPosition scale(Position p) const
  {
    p -= o;
    return Vector(b1*p,b2*p,b3*p);
  }

  NAMD_HOST_DEVICE Vector scale_force(Vector f) const
  {
    return (f.x*b1 + f.y*b2 + f.z*b3);  // calculating A^(-1)f for PME force contributions
  }

  // transforms a position nearest to a SCALED reference position
  NAMD_HOST_DEVICE Position nearest(Position data, ScaledPosition ref) const
  {
    ScaledPosition sn = scale(data);
    if ( p1 ) {
      sn.x -= namdnearbyint(sn.x - ref.x);
    }
    if ( p2 ) {
      sn.y -= namdnearbyint(sn.y - ref.y);
    }
    if ( p3 ) {
      sn.z -= namdnearbyint(sn.z - ref.z);
    }
    return unscale(sn);
  }

  // transforms a position nearest to a SCALED reference position
  // adds transform for later reversal
  NAMD_HOST_DEVICE Position nearest(Position data, ScaledPosition ref, Transform *t) const
  {
    ScaledPosition sn = scale(data);
    if ( p1 ) {
      BigReal tmp = sn.x - ref.x;
      BigReal rit = namdnearbyint(tmp);
      sn.x -= rit;
      t->i -= (int) rit;
    }
    if ( p2 ) {
      BigReal tmp = sn.y - ref.y;
      BigReal rit = namdnearbyint(tmp);
      sn.y -= rit;
      t->j -= (int) rit;
    }
    if ( p3 ) {
      BigReal tmp = sn.z - ref.z;
      BigReal rit = namdnearbyint(tmp);
      sn.z -= rit;
      t->k -= (int) rit;
    }
    return unscale(sn);
  }

  // applies stored transform to original coordinates
  NAMD_HOST_DEVICE Position apply_transform(Position data, const Transform &t) const
  {
    return ( data + t.i*a1 + t.j*a2 + t.k*a3 );
  }

  // reverses cumulative transformations for output
  NAMD_HOST_DEVICE Position reverse_transform(Position data, const Transform &t) const
  {
    return ( data - t.i*a1 - t.j*a2 - t.k*a3 );
  }

  // calculates shortest vector from p2 to p1 (equivalent to p1 - p2)
  NAMD_HOST_DEVICE Vector delta(const Position &pos1, const Position &pos2) const
  {
    Vector diff = pos1 - pos2;
    return delta_from_diff(diff);
  }

  // calculates shortest vector for given distance vector
  NAMD_HOST_DEVICE Vector delta_from_diff(const Position &diff_in) const
  {
    Vector diff = diff_in;
#ifdef ARCH_POWERPC   //Prevents stack temporaries
    Vector result = diff;
    if ( p1 ) {
      BigReal fval = namdnearbyint(b1*diff); 
      result.x -= a1.x *fval;    
      result.y -= a1.y *fval;    
      result.z -= a1.z *fval;    
    }
    if ( p2 ) {
      BigReal fval = namdnearbyint(b2*diff);
      result.x -= a2.x * fval;
      result.y -= a2.y * fval;
      result.z -= a2.z * fval;
    }
    if ( p3 ) {
      BigReal fval = namdnearbyint(b3*diff);
      result.x -= a3.x * fval;
      result.y -= a3.y * fval;
      result.z -= a3.z * fval;
    }
    return result;
#else
    BigReal f1 = p1 ? namdnearbyint(b1*diff) : 0.;
    BigReal f2 = p2 ? namdnearbyint(b2*diff) : 0.;
    BigReal f3 = p3 ? namdnearbyint(b3*diff) : 0.;
    diff.x -= f1*a1.x + f2*a2.x + f3*a3.x;
    diff.y -= f1*a1.y + f2*a2.y + f3*a3.y;
    diff.z -= f1*a1.z + f2*a2.z + f3*a3.z;
    return diff;
#endif
  }

  // calculates scaled vector v such that vector pos1 - pos2 + v*a is the shortest
  NAMD_HOST_DEVICE Vector wrap_delta_scaled(const Position &pos1, const Position &pos2) const
  {
    Vector diff = pos1 - pos2;
    Vector result(0.,0.,0.);
   
    if ( p1 ) result.x = -namdnearbyint(b1*diff);
    if ( p2 ) result.y = -namdnearbyint(b2*diff);
    if ( p3 ) result.z = -namdnearbyint(b3*diff);
    return result;
  }
  
  NAMD_HOST_DEVICE Vector wrap_delta_scaled_fast(const Position &pos1, const Position &pos2) const
  {
    Vector diff = pos1 - pos2;
    Vector result(-namdnearbyint(b1*diff), -namdnearbyint(b2*diff), -namdnearbyint(b3*diff));
    return result;
  }
  
  // calculates shortest vector from origin to p1 (equivalent to p1 - o)
  NAMD_HOST_DEVICE Vector delta(const Position &pos1) const
  {
    Vector diff = pos1 - o;
    Vector result = diff;
    if ( p1 ) result -= a1*namdnearbyint(b1*diff);
    if ( p2 ) result -= a2*namdnearbyint(b2*diff);
    if ( p3 ) result -= a3*namdnearbyint(b3*diff);
    return result;
  }

  // calculates vector to bring p1 closest to origin
  NAMD_HOST_DEVICE Vector wrap_delta(const Position &pos1) const
  {
    Vector diff = pos1 - o;
    Vector result(0.,0.,0.);
    if ( p1 ) result -= a1*namdnearbyint(b1*diff);
    if ( p2 ) result -= a2*namdnearbyint(b2*diff);
    if ( p3 ) result -= a3*namdnearbyint(b3*diff);
    return result;
  }

  // calculates vector to bring p1 closest to origin in unscaled coordinates
  NAMD_HOST_DEVICE Vector wrap_nearest_delta(Position pos1) const
  {
    Vector diff = pos1 - o;
    Vector result0(0.,0.,0.);
    if ( p1 ) result0 -= a1*namdnearbyint(b1*diff);
    if ( p2 ) result0 -= a2*namdnearbyint(b2*diff);
    if ( p3 ) result0 -= a3*namdnearbyint(b3*diff);
    diff += result0;
    BigReal dist = diff.length2();
    Vector result(0.,0.,0.);
    for ( int i1=-p1; i1<=p1; ++i1 ) {
      for ( int i2 =-p2; i2<=p2; ++i2 ) {
        for ( int i3 =-p3; i3<=p3; ++i3 ) {
          Vector newresult = i1*a1+i2*a2+i3*a3;
          BigReal newdist = (diff+newresult).length2();
          if ( newdist < dist ) {
            dist = newdist;
            result = newresult;
          }
        }
      }
    }
    return result0 + result;
  }

  NAMD_HOST_DEVICE Vector offset(int i) const
  {
    return ( (i%3-1) * a1 + ((i/3)%3-1) * a2 + (i/9-1) * a3 );
  }

  NAMD_HOST_DEVICE static int offset_a(int i) { return (i%3-1); }
  NAMD_HOST_DEVICE static int offset_b(int i) { return ((i/3)%3-1); }
  NAMD_HOST_DEVICE static int offset_c(int i) { return (i/9-1); }

  // lattice vectors
  NAMD_HOST_DEVICE Vector a() const { return a1; }
  NAMD_HOST_DEVICE Vector b() const { return a2; }
  NAMD_HOST_DEVICE Vector c() const { return a3; }

  // only if along x y z axes
  NAMD_HOST_DEVICE int orthogonal() const {
    return ( ! ( a1.y || a1.z || a2.x || a2.z || a3.x || a3.y ) );
  }

  // origin (fixed center of cell)
  NAMD_HOST_DEVICE Vector origin() const
  {
    return o;
  }

  // reciprocal lattice vectors
  NAMD_HOST_DEVICE Vector a_r() const { return b1; }
  NAMD_HOST_DEVICE Vector b_r() const { return b2; }
  NAMD_HOST_DEVICE Vector c_r() const { return b3; }

  // periodic along this direction
  NAMD_HOST_DEVICE int a_p() const { return p1; }
  NAMD_HOST_DEVICE int b_p() const { return p2; }
  NAMD_HOST_DEVICE int c_p() const { return p3; }

  NAMD_HOST_DEVICE BigReal volume(void) const
  {
    return ( p1 && p2 && p3 ? cross(a1,a2) * a3 : 0.0 );
  }

private:
  Vector a1,a2,a3; // real lattice vectors
  Vector b1,b2,b3; // reciprocal lattice vectors (more or less)
  Vector o; // origin (fixed center of cell)
  int p1, p2, p3; // periodic along this lattice vector?

  // calculate reciprocal lattice vectors
  NAMD_HOST_DEVICE void recalculate(void) {
    {
      Vector c = cross(a2,a3);
      b1 = c / ( a1 * c );
    }
    {
      Vector c = cross(a3,a1);
      b2 = c / ( a2 * c );
    }
    {
      Vector c = cross(a1,a2);
      b3 = c / ( a3 * c );
    }
  }

};

#if !(defined(__NVCC__) || defined(__HIPCC__))
#include <pup.h>
PUPbytes (Lattice);  
#endif


#endif

