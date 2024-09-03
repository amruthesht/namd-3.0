/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#include "InfoStream.h"
#include "ComputeNonbondedCUDAExcl.h"
#include "Molecule.h"
#include "Parameters.h"
#include "LJTable.h"
#include "Node.h"
#include "ReductionMgr.h"
#include "Lattice.h"
#include "PressureProfile.h"
#include "Debug.h"


// static initialization
int ExclElem::pressureProfileSlabs = 0;
int ExclElem::pressureProfileAtomTypes = 1;
BigReal ExclElem::pressureProfileThickness = 0;
BigReal ExclElem::pressureProfileMin = 0;
int ExclElem::pswitchTable[3*3] = {0,1,2,1,1,99,2,99,2};

void ExclElem::getMoleculePointers
    (Molecule* mol, int* count, int32*** byatom, Exclusion** structarray)
{
#ifdef MEM_OPT_VERSION
  NAMD_die("Should not be called in ExclElem::getMoleculePointers in memory optimized version!");
#else
  *count = mol->numExclusions;
  *byatom = mol->exclusionsByAtom;
  *structarray = mol->exclusions;
#endif
}

void ExclElem::getParameterPointers(Parameters *p, const int **v) {
  *v = 0;
}


void ExclElem::computeForce(
    ExclElem *tuples,
    int ntuple,
    BigReal *reduction,
    BigReal *pressureProfileData
    )
{
  //
  // Use the following from ComputeNonbondedUtil:
  //   cutoff2
  //   switchOn2
  //   alchFepOn
  //   alchVdwShiftCoeff
  //   alchDecouple
  //   scaling
  //   scale14
  //   dielectric_1
  //   ljTable
  //   r2_table
  //   table_noshort
  //   fast_table
  //   slow_table
  //
  const Lattice & lattice = tuples[0].p[0]->p->lattice;
  const Flags &flags = tuples[0].p[0]->p->flags;
  if ( ! flags.doNonbonded ) return;
  SimParameters *simParams = Node::Object()->simParameters;
  const int doFull = flags.doFullElectrostatics;
  const int doEnergy = flags.doEnergy;

 /* ALCH STUFF */
  BigReal lambdaUp, lambdaDown, lambda2Up, lambda2Down, switchfactor;
  BigReal vdwLambdaUp, vdwLambdaDown, elecLambdaUp, elecLambdaDown;
  BigReal elecLambda2Up, elecLambda2Down, vdwLambda2Up, vdwLambda2Down;
  BigReal vdwShiftUp, vdwShift2Up, vdwShiftDown, vdwShift2Down;
  BigReal myVdwLambda, myVdwLambda2, myElecLambda, myElecLambda2, myVdwShift, myVdwShift2;	
  //bool isAlch = simParams->alchFepOn;
  int pswitch, ref, alch, dec, up, down;
	
  if (alchFepOn || alchThermIntOn) {
    BigReal lambdaUp    = simParams->getCurrentLambda(flags.step);
    BigReal lambda2Up   = simParams->getCurrentLambda2(flags.step);
    BigReal lambdaDown  = 1 - lambdaUp;
    BigReal lambda2Down = 1 - lambda2Up; 
    switchfactor      = 1./((cutoff2 - switchOn2)*
        (cutoff2 - switchOn2)*(cutoff2 - switchOn2));
    vdwLambdaUp       = simParams->getVdwLambda(lambdaUp);
    vdwLambdaDown     = simParams->getVdwLambda(lambdaDown);
    elecLambdaUp      = simParams->getElecLambda(lambdaUp);
    elecLambdaDown    = simParams->getElecLambda(lambdaDown);
    elecLambda2Up     = simParams->getElecLambda(lambda2Up);
    elecLambda2Down   = simParams->getElecLambda(lambda2Down);
    vdwLambda2Up      = simParams->getVdwLambda(lambda2Up);
    vdwLambda2Down    = simParams->getVdwLambda(lambda2Down);
    vdwShiftUp        = alchVdwShiftCoeff*(1 - vdwLambdaUp);
    vdwShiftDown      = alchVdwShiftCoeff*(1 - vdwLambdaDown);
    vdwShift2Up       = alchVdwShiftCoeff*(1 - vdwLambda2Up);
    vdwShift2Down     = alchVdwShiftCoeff*(1 - vdwLambda2Down);
     
    if (alchDecouple) {
      // decoupling: PME calculates extra grids so that while PME 
      // interaction with the full system is switched off, a new PME grid
      // containing only alchemical atoms is switched on. Full interactions 
      // between alchemical atoms are maintained; potentials within one 
      // partition need not be scaled here.
      pswitchTable[1+3*1] = 0;
      pswitchTable[2+3*2] = 0;
    }
  }

  for ( int ituple=0; ituple<ntuple; ++ituple ) {
    // ENERGIES FOR REDUCTION
    BigReal energyVdw = 0.0,   energyElec = 0.0,   energySlow = 0.0;
    // FEP Energies
    BigReal energyVdw_s = 0.0, energyElec_s = 0.0, energySlow_s = 0.0;

    const ExclElem &tup = tuples[ituple];
    enum { size = 2 };
    const AtomID (&atomID)[size](tup.atomID);
    const int    (&localIndex)[size](tup.localIndex);
    TuplePatchElem * const(&p)[size](tup.p);
    const Real (&scale)(tup.scale);
    const int (&modified)(tup.modified);

    const CompAtom &p_i = p[0]->x[localIndex[0]];
    const CompAtom &p_j = p[1]->x[localIndex[1]];

    const unsigned char p1 = p_i.partition;
    const unsigned char p2 = p_j.partition;

    BigReal alch_vdw_energy   = 0.0;
    BigReal alch_vdw_energy_2 = 0.0;
    BigReal alch_vdw_force    = 0.0;
    BigReal alch_vdw_dUdl     = 0.0;

    // compute vectors between atoms and their distances
    const Vector r12 = lattice.delta(p_i.position, p_j.position);
    BigReal r2 = r12.length2();

    if ( r2 > cutoff2 ) continue;

    if ( modified && r2 < 1.0 ) r2 = 1.0;  // match CUDA interpolation

    r2 += r2_delta;

    pswitch = pswitchTable[p1 + 3*p2];
    if (alchFepOn || alchThermIntOn) {
      switch (pswitch) {
        case 0:
          myVdwLambda  = 1.0;  myElecLambda  = 1.0;
          myVdwLambda2 = 1.0;  myElecLambda2 = 1.0;
          break;
        case 1:
          myVdwLambda  = vdwLambdaUp;  myElecLambda  = elecLambdaUp;  myVdwShift = vdwShiftUp; 
          myVdwLambda2 = vdwLambda2Up; myElecLambda2 = elecLambda2Up; myVdwShift2 = vdwShift2Up;
          break;
        case 2:
          myVdwLambda  = vdwLambdaDown;   myElecLambda  = elecLambdaDown;  myVdwShift  = vdwShiftDown;
          myVdwLambda2 = vdwLambda2Down;  myElecLambda2 = elecLambda2Down; myVdwShift2 = vdwShift2Down;
          break;
	default:
          myVdwLambda  = 0.0;  myElecLambda  = 0.0; 
          myVdwLambda2 = 0.0;  myElecLambda2 = 0.0;
          break;
      }
    }
    else {
      myVdwLambda  = 1.0;  myElecLambda  = 1.0;
      myVdwLambda2 = 1.0;  myElecLambda2 = 1.0;
    }

    union { double f; int64 i; } r2i;
    r2i.f = r2;
    const int r2_delta_expc = 64 * (r2_delta_exp - 1023);
    int table_i = (r2i.i >> (32+14)) + r2_delta_expc;  // table_i >= 0

    const BigReal* const table_four_i = table_noshort + 16*table_i;

    BigReal diffa = r2 - r2_table[table_i];

    BigReal fast_a = 0., fast_b = 0., fast_c = 0., fast_d = 0.;
    BigReal slow_a, slow_b, slow_c, slow_d;

    if ( modified ) {  // fix modified 1-4 interactions

      const LJTable::TableEntry * lj_pars =
        ljTable->table_row(p_i.vdwType) + 2 * p_j.vdwType;

      // modified - normal = correction
      const BigReal A = scaling * ( (lj_pars+1)->A - lj_pars->A );
      const BigReal B = scaling * ( (lj_pars+1)->B - lj_pars->B );

      BigReal vdw_d = A * table_four_i[0] - B * table_four_i[4];
      BigReal vdw_c = A * table_four_i[1] - B * table_four_i[5];
      BigReal vdw_b = A * table_four_i[2] - B * table_four_i[6];
      BigReal vdw_a = A * table_four_i[3] - B * table_four_i[7];

      const BigReal kqq = (1.0 - scale14) *
        COULOMB * p_i.charge * p_j.charge * scaling * dielectric_1;

      fast_a =      kqq * fast_table[4*table_i+0];  // not used!
      fast_b = 2. * kqq * fast_table[4*table_i+1];
      fast_c = 4. * kqq * fast_table[4*table_i+2];
      fast_d = 6. * kqq * fast_table[4*table_i+3];

      if ( doFull ) {
        slow_a =      kqq * slow_table[4*table_i+3];  // not used!
        slow_b = 2. * kqq * slow_table[4*table_i+2];
        slow_c = 4. * kqq * slow_table[4*table_i+1];
        slow_d = 6. * kqq * slow_table[4*table_i+0];
      }

      if ( doEnergy ) {
        energyElec  = (( ( diffa * (1./6.)*fast_d + 0.25*fast_c ) * diffa
              + 0.5*fast_b ) * diffa + fast_a);
        if ( doFull ) {
          energySlow = (( ( diffa * (1./6.)*slow_d + 0.25*slow_c ) * diffa
                + 0.5*slow_b ) * diffa + slow_a);
        }
      }

      if (pswitch == 0) {
        fast_a += vdw_a;
        fast_b += vdw_b;
        fast_c += vdw_c;
        fast_d += vdw_d;
        if (doEnergy) {
          energyVdw = (( ( diffa * (1./6.)*vdw_d + 0.25*vdw_c ) * diffa
                + 0.5*vdw_b ) * diffa + vdw_a);
        }
      }
      else if(pswitch != 99) {
        // Special alch forces should be calculated here
	if(alchFepOn){
          const BigReal r2_1 = 1./(r2 + myVdwShift);
          const BigReal r2_2 = 1./(r2 + myVdwShift2);
          const BigReal r6_1 = r2_1*r2_1*r2_1;
          const BigReal r6_2 = r2_2*r2_2*r2_2;
          const BigReal U1 = A*r6_1*r6_1 - B*r6_1; // NB: unscaled, shorthand only
          const BigReal U2 = A*r6_2*r6_2 - B*r6_2;

          const BigReal switchmul  = (r2 > switchOn2 ?
            switchfactor * (cutoff2 - r2) * (cutoff2 - r2) *
            (cutoff2 - 3.*switchOn2 + 2.*r2) : 1.);
 	
          const BigReal switchmul2 = (r2 > switchOn2 ?
            12.*switchfactor * (cutoff2 - r2) * (r2 - switchOn2) : 0.);
          BigReal rinv = namd_rsqrt(r2);
          alch_vdw_energy   = myVdwLambda*switchmul*U1;
          alch_vdw_energy_2 = myVdwLambda2*switchmul*U2;
          alch_vdw_force    = myVdwLambda*((switchmul*(12.*U1 + 6.*B*r6_1)*r2_1 +
              switchmul2*U1));
          if (doEnergy) {
            reduction[vdwEnergyIndex]   += alch_vdw_energy;
            reduction[vdwEnergyIndex_s] += alch_vdw_energy_2;
          }
         }else if(alchThermIntOn){
	    const BigReal r2_1 = 1./(r2 + myVdwShift);
  	    const BigReal r6_1 = r2_1*r2_1*r2_1;
  	    // switching function (this is correct whether switching is active or not)
  	    const BigReal switchmul = (r2 > switchOn2 ? switchfactor*(cutoff2 - r2) \
		    *(cutoff2 - r2) \
		    *(cutoff2 - 3.*switchOn2 + 2.*r2) : 1.);
  	    const BigReal switchmul2 = (r2 > switchOn2 ?                      \
                          12.*switchfactor*(cutoff2 - r2)       \
                           *(r2 - switchOn2) : 0.);

	    // separation-shifted vdW force and energy
	    const BigReal U = A*r6_1*r6_1 - B*r6_1; // NB: unscaled! for shorthand only!
	    alch_vdw_energy = myVdwLambda*switchmul*U;
	    alch_vdw_force = (myVdwLambda*(switchmul*(12.*U + 6.*B*r6_1)*r2_1 \
                                  + switchmul2*U));
	    alch_vdw_dUdl = (switchmul*(U + myVdwLambda*alchVdwShiftCoeff \
                                   *(6.*U + 3.*B*r6_1)*r2_1));
	    if (doEnergy) {
               reduction[vdwEnergyIndex]   += alch_vdw_energy;
	       reduction[vdwEnergyIndex_ti_1] += alch_vdw_dUdl*(pswitch == 1);
	       reduction[vdwEnergyIndex_ti_2] += alch_vdw_dUdl*(pswitch == 2);
            }
	 }//alchFepOn
      }
    } // end if modified
    else if ( doFull ) {  // full exclusion

      const BigReal kqq = 
        COULOMB * p_i.charge * p_j.charge * scaling * dielectric_1;

      slow_d = kqq * ( table_four_i[8]  - table_four_i[12] );
      slow_c = kqq * ( table_four_i[9]  - table_four_i[13] );
      slow_b = kqq * ( table_four_i[10] - table_four_i[14] );
      slow_a = kqq * ( table_four_i[11] - table_four_i[15] );  // not used!

      if ( doEnergy ) {
        energySlow = (( ( diffa * (1./6.)*slow_d + 0.25*slow_c ) * diffa
              + 0.5*slow_b ) * diffa + slow_a);
      }
    }

    register BigReal fast_dir = (diffa * fast_d + fast_c) * diffa + fast_b;

    fast_dir = (fast_dir*myElecLambda) + alch_vdw_force*(pswitch == 1 || pswitch == 2);
    const Force f12 = fast_dir * r12;

    // Now add the forces to each force vector
    p[0]->r->f[Results::nbond][localIndex[0]] += f12;
    p[1]->r->f[Results::nbond][localIndex[1]] -= f12;

    // reduction[nonbondedEnergyIndex] += energy;
    reduction[virialIndex_XX] += f12.x * r12.x;
    //reduction[virialIndex_XY] += f12.x * r12.y;
    //reduction[virialIndex_XZ] += f12.x * r12.z;
    reduction[virialIndex_YX] += f12.y * r12.x;
    reduction[virialIndex_YY] += f12.y * r12.y;
    //reduction[virialIndex_YZ] += f12.y * r12.z;
    reduction[virialIndex_ZX] += f12.z * r12.x;
    reduction[virialIndex_ZY] += f12.z * r12.y;
    reduction[virialIndex_ZZ] += f12.z * r12.z;

    if ( doFull ) {
      register BigReal slow_dir = (diffa * slow_d + slow_c) * diffa + slow_b;

      slow_dir = (slow_dir * myElecLambda);
      const Force slow_f12 = slow_dir * r12;

      p[0]->r->f[Results::slow][localIndex[0]] += slow_f12;
      p[1]->r->f[Results::slow][localIndex[1]] -= slow_f12;

      // reduction[nonbondedEnergyIndex] += energy;
      reduction[slowVirialIndex_XX] += slow_f12.x * r12.x;
      //reduction[slowVirialIndex_XY] += slow_f12.x * r12.y;
      //reduction[slowVirialIndex_XZ] += slow_f12.x * r12.z;
      reduction[slowVirialIndex_YX] += slow_f12.y * r12.x;
      reduction[slowVirialIndex_YY] += slow_f12.y * r12.y;
      //reduction[slowVirialIndex_YZ] += slow_f12.y * r12.z;
      reduction[slowVirialIndex_ZX] += slow_f12.z * r12.x;
      reduction[slowVirialIndex_ZY] += slow_f12.z * r12.y;
      reduction[slowVirialIndex_ZZ] += slow_f12.z * r12.z;
    }

    // Scale the energies here
    if (doEnergy) {
      reduction[vdwEnergyIndex] -= energyVdw*myElecLambda;
      reduction[electEnergyIndex] -= energyElec*myElecLambda;
      if (doFull) {
        reduction[fullElectEnergyIndex] -= energySlow*myElecLambda;
      }
      if (alchFepOn) {
        reduction[vdwEnergyIndex_s] -= energyVdw*myElecLambda2;
        reduction[electEnergyIndex_s] -= energyElec*myElecLambda2;
        if (doFull) {
          reduction[fullElectEnergyIndex_s] -= energySlow*myElecLambda2;
        }
      }else if (alchThermIntOn){
	if(pswitch == 1){
	   //Those aren't defined here. What should I do?
	   //reduction[vdwEnergyIndex_ti_1]   -= alch_vdw_dUdl;
	   reduction[electEnergyIndex_ti_1] -= energyElec;
	   if (doFull) reduction[fullElectEnergyIndex_ti_1] -= energySlow;
	}
	else if (pswitch == 2){
	   //reduction[vdwEnergyIndex_ti_2]   -= alch_vdw_dUdl;
	   reduction[electEnergyIndex_ti_2] -= energyElec;
	   if (doFull) reduction[fullElectEnergyIndex_ti_2] -= energySlow; //should I be scaling those?
        }
      }
    } // end if doEnergy
  } // end for loop
} // end computeForce()


void ExclElem::submitReductionData(BigReal *data, SubmitReduction *reduction)
{
  //bool isAlch = Node::Object()->simParameters->alchFepOn;

  reduction->item(REDUCTION_ELECT_ENERGY) += data[electEnergyIndex];
  reduction->item(REDUCTION_LJ_ENERGY) += data[vdwEnergyIndex];
  reduction->item(REDUCTION_ELECT_ENERGY_SLOW) += data[fullElectEnergyIndex];
  if (alchFepOn) {
    reduction->item(REDUCTION_ELECT_ENERGY_F)      += data[electEnergyIndex_s];
    reduction->item(REDUCTION_ELECT_ENERGY_SLOW_F) += data[fullElectEnergyIndex_s];
    reduction->item(REDUCTION_LJ_ENERGY_F)         += data[vdwEnergyIndex_s];
  }else if (alchThermIntOn){
    reduction->item(REDUCTION_ELECT_ENERGY_TI_1) += data[electEnergyIndex_ti_1];
    reduction->item(REDUCTION_ELECT_ENERGY_TI_2) += data[electEnergyIndex_ti_2];
    reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_1) += data[fullElectEnergyIndex_ti_1];
    reduction->item(REDUCTION_ELECT_ENERGY_SLOW_TI_2) += data[fullElectEnergyIndex_ti_2];
    reduction->item(REDUCTION_LJ_ENERGY_TI_1) += data[vdwEnergyIndex_ti_1];
    reduction->item(REDUCTION_LJ_ENERGY_TI_2) += data[vdwEnergyIndex_ti_2];
  }
  //transposes the virial tensor after calculation
  data[virialIndex_XY] = data[virialIndex_YX];
  data[virialIndex_XZ] = data[virialIndex_ZX];
  data[virialIndex_YZ] = data[virialIndex_ZY];

  data[slowVirialIndex_XY] = data[slowVirialIndex_YX];
  data[slowVirialIndex_XZ] = data[slowVirialIndex_ZX];
  data[slowVirialIndex_YZ] = data[slowVirialIndex_ZY];

  ADD_TENSOR(reduction,REDUCTION_VIRIAL_NBOND,data,virialIndex);
  ADD_TENSOR(reduction,REDUCTION_VIRIAL_SLOW,data,slowVirialIndex);
}

