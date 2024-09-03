/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#include "ComputeGridForce.h"
#include "GridForceGrid.h"
#include "Node.h"

#define MIN_DEBUG_LEVEL 3
//#define DEBUGM
#include "Debug.h"
#include "GridForceGrid.inl"
#include "MGridforceParams.h"

//#define GF_FORCE_OUTPUT
//#define GF_FORCE_OUTPUT_FREQ 100
#define GF_OVERLAPCHECK_FREQ 1000


ComputeGridForce::ComputeGridForce(ComputeID c, PatchID pid)
    : ComputeHomePatch(c,pid)
{

    reduction = ReductionMgr::Object()->willSubmit(REDUCTIONS_BASIC);
    idxChecked=false;
}
/*			END OF FUNCTION ComputeGridForce	*/


ComputeGridForce::~ComputeGridForce()
{
    delete reduction;
}
/*			END OF FUNCTION ~ComputeGridForce	*/


//! if fewer than half the atoms in the patch need grid forces, use a list
int ComputeGridForce::checkGridForceRatio()
{
  //  Loop through and check each atom
  int numGridForcedAtoms=0;
  Molecule *mol = Node::Object()->molecule;
  int numGrids = mol->numGridforceGrids;
  int numAtoms = homePatch->getNumAtoms();
  FullAtom* a = homePatch->getAtomList().begin();
  for (int gridnum = 0; gridnum < numGrids; gridnum++) {
    for (int i = 0; i < numAtoms; i++)
      if (mol->is_atom_gridforced(a[i].id, gridnum)){
	numGridForcedAtoms++;
      }
  }
  return numGridForcedAtoms;
}

void ComputeGridForce::createGridForcedIdxList(int numGridForcedAtoms)
{
  //  Loop through and check each atom
  Molecule *mol = Node::Object()->molecule;
  int numGrids = mol->numGridforceGrids;
  int numAtoms = homePatch->getNumAtoms();
  if(gridForcedAtomIdxList.size()>0)
    {
      //empty everything
      for (int gridnum = 0; gridnum < gridForcedAtomIdxList.size(); gridnum++) {
	gridForcedAtomIdxList[gridnum].clear();
      }
      gridForcedAtomIdxList.clear();
    }
  else
    {
      gridForcedAtomIdxList.reserve(numGrids);
    }
  FullAtom* a = homePatch->getAtomList().begin();

  for (int gridnum = 0; gridnum < numGrids; gridnum++) {
    std::vector <int> thisGrid;
    thisGrid.reserve(numGridForcedAtoms);
    for (int i = 0; i < numAtoms; i++)
      if (mol->is_atom_gridforced(a[i].id, gridnum)){
	thisGrid.push_back(i);
      }
    gridForcedAtomIdxList.push_back(thisGrid);
  }
}


template <class T> void ComputeGridForce::do_calc(T *grid, int gridnum, FullAtom *p, int numAtoms, Molecule *mol, Force *forces, BigReal &energy, Force &extForce, Tensor &extVirial)
{
    Real scale;			// Scaling factor
    Charge charge;		// Charge
    Vector dV;
    float V;
    
    Vector gfScale = grid->get_scale();
    DebugM(3, "doCalc()\n" << endi);
#ifdef DEBUGM
    int gridForcedCount=0;
#endif
    if(useIndexList)
      {
	// loop through the atoms we know need gridforce
	DebugM(3, "doCalc() using index \n" << endi);
	std::vector<int> &thisGrid=gridForcedAtomIdxList[gridnum];
	for (int idx = 0; idx < thisGrid.size(); idx++)
	  {
	    int i=thisGrid[idx];
#ifdef DEBUGM
	    gridForcedCount++;
#endif
	    DebugM(1, "Atom " << p[i].id << " is gridforced\n" << endi);
	    
	    mol->get_gridfrc_params(scale, charge, p[i].id, gridnum);
	    
	    // Wrap coordinates using grid center
	    Position pos = grid->wrap_position(p[i].position, homePatch->lattice);
	    DebugM(1, "pos = " << pos << "\n" << endi);
	    
	    // Here's where the action happens
	    int err = grid->compute_VdV(pos, V, dV);
	    
	    if (err) {
	      DebugM(2, "V = 0\n" << endi);
	      DebugM(2, "dV = 0 0 0\n" << endi);
	      continue;  // This means the current atom is outside the potential
	    }
	    
	    //Force force = scale * Tensor::diagonal(gfScale) * (-charge * dV);
	    Force force = -charge * scale * Vector(gfScale.x * dV.x, gfScale.y * dV.y, gfScale.z * dV.z);
	    
#ifdef DEBUGM
	    DebugM(2, "scale = " << scale << " gfScale = " << gfScale << " charge = " << charge << "\n" << endi);
	    
	    DebugM(2, "V = " << V << "\n" << endi);
	    DebugM(2, "dV = " << dV << "\n" << endi);
	    DebugM(2, "grid = " << gridnum << " force = " << force << " pos = " << pos << " V = " << V << " dV = " << dV << " step = " << homePatch->flags.step << " index = " << p[i].id << "\n" << endi);
	    
	    DebugM(1, "transform = " << (int)p[i].transform.i << " "
		   << (int)p[i].transform.j << " " << (int)p[i].transform.k << "\n" << endi);
	    
	    if (V != V) {
	      iout << iWARN << "V is NaN!\natomid = " << p[i].id << " loc = " << p[i].position << " V = " << V << "\n" << endi;
	    }
#endif
	    
	    forces[i] += force;
	    extForce += force;
	    Position vpos = homePatch->lattice.reverse_transform(p[i].position, p[i].transform);
	    
	    //energy -= force * (vpos - homePatch->lattice.origin());
	    if (gfScale.x == gfScale.y && gfScale.x == gfScale.z)
	      {
		// only makes sense when scaling is isotropic
		energy += scale * gfScale.x * (charge * V);
		
		// add something when we're off the grid? I'm thinking no
	      }
	    extVirial += outer(force,vpos);
	  }
      }
    else
      {
	//  Loop through and check each atom
	for (int i = 0; i < numAtoms; i++) {
	  if (mol->is_atom_gridforced(p[i].id, gridnum))
	    {
	      DebugM(1, "Atom " << p[i].id << " is gridforced\n" << endi);
#ifdef DEBUGM
              gridForcedCount++;
#endif
	      mol->get_gridfrc_params(scale, charge, p[i].id, gridnum);

	      // Wrap coordinates using grid center
	      Position pos = grid->wrap_position(p[i].position, homePatch->lattice);
	      DebugM(1, "pos = " << pos << "\n" << endi);

	      // Here's where the action happens
	      int err = grid->compute_VdV(pos, V, dV);

	      if (err) {
		DebugM(2, "V = 0\n" << endi);
		DebugM(2, "dV = 0 0 0\n" << endi);
		continue;  // This means the current atom is outside the potential
	      }

	      //Force force = scale * Tensor::diagonal(gfScale) * (-charge * dV);
	      Force force = -charge * scale * Vector(gfScale.x * dV.x, gfScale.y * dV.y, gfScale.z * dV.z);

#ifdef DEBUGM
	      DebugM(2, "scale = " << scale << " gfScale = " << gfScale << " charge = " << charge << "\n" << endi);

	      DebugM(2, "V = " << V << "\n" << endi);
	      DebugM(2, "dV = " << dV << "\n" << endi);
	      DebugM(2, "grid = " << gridnum << " force = " << force << " pos = " << pos << " V = " << V << " dV = " << dV << " step = " << homePatch->flags.step << " index = " << p[i].id << "\n" << endi);

	      DebugM(1, "transform = " << (int)p[i].transform.i << " "
		     << (int)p[i].transform.j << " " << (int)p[i].transform.k << "\n" << endi);

	      if (V != V) {
		iout << iWARN << "V is NaN!\natomid = " << p[i].id << " loc = " << p[i].position << " V = " << V << "\n" << endi;
	      }
#endif

	      forces[i] += force;
	      extForce += force;
	      Position vpos = homePatch->lattice.reverse_transform(p[i].position, p[i].transform);

	      //energy -= force * (vpos - homePatch->lattice.origin());
	      if (gfScale.x == gfScale.y && gfScale.x == gfScale.z)
		{
		  // only makes sense when scaling is isotropic
		  energy += scale * gfScale.x * (charge * V);

		  // add something when we're off the grid? I'm thinking no
		}
	      extVirial += outer(force,vpos);
	    }
	}
      }
    DebugM(3, "doCalc() patch "<<homePatch->getPatchID()<<" computed "<< gridForcedCount<<" of "<< numAtoms <<"\n" << endi);
    DebugM(3, "doCalc() done\n" << endi);
}

void ComputeGridForce::doForce(FullAtom* p, Results* r)
{
    SimParameters *simParams = Node::Object()->simParameters;
    Molecule *mol = Node::Object()->molecule;
    
    Force *forces = r->f[Results::normal];
    BigReal energy = 0;
    Force extForce = 0.;
    Tensor extVirial;
    
    int numAtoms = homePatch->getNumAtoms();

    if ( mol->numGridforceGrids < 1 ) NAMD_bug("No grids loaded in ComputeGridForce::doForce()");

    DebugM(3, "doForce()\n" << endi);
    if(!homePatch->gridForceIdxChecked){
      DebugM(3, "doForce() patch " << homePatch->getPatchID()<<" checking grid \n" << endi);
      int numGridForcedAtoms= checkGridForceRatio();
      int numAtoms = homePatch->getNumAtoms();
      useIndexList=(float)numGridForcedAtoms/(float)numAtoms<0.5;
      if(useIndexList)
	{
	  createGridForcedIdxList(numGridForcedAtoms);
	}
      homePatch->gridForceIdxChecked=true;
    }
    
    for (int gridnum = 0; gridnum < mol->numGridforceGrids; gridnum++) {
	GridforceGrid *grid = mol->get_gridfrc_grid(gridnum);

        const Vector gfScale = grid->get_scale();
        if ((gfScale.x == 0.0) && (gfScale.y == 0.0) && (gfScale.z == 0.0)) {
          DebugM(3, "Skipping grid index " << gridnum << "\n" << endi);
          continue;
        }
	
	if (homePatch->flags.step % GF_OVERLAPCHECK_FREQ == 0) {
	    // only check on node 0 and every GF_OVERLAPCHECK_FREQ steps
	  if (simParams->langevinPistonOn || simParams->berendsenPressureOn) {
		// check for grid overlap if pressure control is on
		// not needed without pressure control, since the check is also performed on startup
      if (!grid->fits_lattice(homePatch->lattice)) {
        char errmsg[512];
        if (grid->get_checksize()) {
          sprintf(errmsg, "Warning: Periodic cell basis too small for Gridforce grid %d.  Set gridforcechecksize off in configuration file to ignore.\n", gridnum);
          NAMD_die(errmsg);      
        }
      }
	 }
	}
	
	if (homePatch->flags.step % 100 == 1) {
            Position center = grid->get_center();
	    DebugM(3, "center = " << center << "\n" << endi);
	    DebugM(3, "e = " << grid->get_e() << "\n" << endi);
	}
	
	if (grid->get_grid_type() == GridforceGrid::GridforceGridTypeFull) {
	    GridforceFullMainGrid *g = (GridforceFullMainGrid *)grid;
	    do_calc(g, gridnum, p, numAtoms, mol, forces, energy, extForce, extVirial);
	} else if (grid->get_grid_type() == GridforceGrid::GridforceGridTypeLite) {
	    GridforceLiteGrid *g = (GridforceLiteGrid *)grid;
	    do_calc(g, gridnum, p, numAtoms, mol, forces, energy, extForce, extVirial);
	}
    }
    reduction->item(REDUCTION_MISC_ENERGY) += energy;
    ADD_VECTOR_OBJECT(reduction,REDUCTION_EXT_FORCE_NORMAL,extForce);
    ADD_TENSOR_OBJECT(reduction,REDUCTION_VIRIAL_NORMAL,extVirial);
    reduction->submit();
    DebugM(3, "doForce() done\n" << endi);
}
/*			END OF FUNCTION force				*/
