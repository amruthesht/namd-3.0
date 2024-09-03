#include "ComputeSMDCUDA.h"
#include "ComputeSMDCUDAKernel.h"
#include "SimParameters.h"
#include "PDB.h"
#include "PDBData.h"
#include "Node.h"
#include "Molecule.h"

#ifdef NODEGROUP_FORCE_REGISTER

ComputeSMDCUDA::ComputeSMDCUDA(
  std::vector<HomePatch*> &patchList, 
  double springConstant, 
  double transverseSpringConstant, 
  double velocity,
  double3 direction,
  int outputFrequency,
  int firstTimeStep,
  const char* filename, 
  int numAtoms){

  // I could use an initializer list, but I don't like them
  this->patchList = &patchList;
  this->springConstant = springConstant;
  this->transverseSpringConstant = transverseSpringConstant;
  this->velocity = velocity;
  this->direction = direction;
  this->outputFrequency = outputFrequency;
  this->firstTimeStep = firstTimeStep;
  this->filename = filename;
  this->numAtoms = numAtoms;
  smdAtomsGlobalIndex.clear();
  // I need to save the global index of atoms. That way I can quickly rebuild the SMD index vector
  allocate_host<double3>(&curCOM, 1);
  parseAtoms();
  
  smdAtomsSOAIndex.resize(this->numSMDAtoms);
  allocate_device<unsigned int>(&d_tbcatomic, 1);
  allocate_device<double3>(&d_curCOM, 1);
  allocate_device<int>(&d_smdAtomsSOAIndex, this->numSMDAtoms);
  // set the current COM value to {0, 0, 0}
  curCOM->x = 0.0;
  curCOM->y = 0.0;
  curCOM->z = 0.0;
  copy_HtoD<double3>(curCOM, d_curCOM, 1);
  cudaCheck(cudaMemset(d_tbcatomic, 0, sizeof(unsigned int)));
}

ComputeSMDCUDA::~ComputeSMDCUDA(){
  deallocate_host<double3>(&curCOM);
  deallocate_device<unsigned int>(&d_tbcatomic);
  deallocate_device<int>(&d_smdAtomsSOAIndex);
}

// This builds the global vector index - swiped from GlobalMasterSMD.C
void ComputeSMDCUDA::parseAtoms(){
  PDB smdpdb(filename);
  origCOM.x = origCOM.y = origCOM.z = 0;
  Molecule *mol = Node::Object()->molecule; // to get masses
  int numPDBAtoms = smdpdb.num_atoms(); 
  if(numPDBAtoms < 1 ) NAMD_die("No Atoms found in SMDFile\n");

  BigReal imass = 0; 
  
  if (numPDBAtoms != this->numAtoms){
    fprintf(stderr, "Error, wrong numPDB (%d vs %d)\n",numPDBAtoms, this->numAtoms);
    NAMD_die("The number of atoms in SMDFile must be equal to the total number of atoms in the structure!\n"); 
  }
  
  // Would this work on PDB atoms? Is the data replicated for everyone?
  for(int i = 0; i < numPDBAtoms; i++){
    // MEMOPT obviously doesn't work with CUDASOA, so we can just use this
    PDBAtom *atom = smdpdb.atom(i);
    if(atom->occupancy()){ // It's a SMD atom! Add it to the list
      smdAtomsGlobalIndex.push_back(i);

      // compute the center of mass
      BigReal mass = mol->atommass(i); 
      origCOM.x += atom->xcoor()*mass;
      origCOM.y += atom->ycoor()*mass;
      origCOM.z += atom->zcoor()*mass;
      imass += mass; 
    }
  }

  inv_group_mass = 1.0 / imass;
  origCOM.x *= inv_group_mass;
  origCOM.y *= inv_group_mass;
  origCOM.z *= inv_group_mass;

  if (imass == 0) // we didn't find any!
    NAMD_die("SMDFile contained no SMD atoms (atoms w/ nonzero occupancy)\n");
  
  this->numSMDAtoms = smdAtomsGlobalIndex.size();
}

void ComputeSMDCUDA::updateAtoms(
  std::vector<AtomMap*> &atomMapsList,
  std::vector<CudaLocalRecord> &localRecords,
  const int* h_globalToLocalID) {
  
  for(int i = 0 ; i < this->numSMDAtoms; i++){
    int gid = smdAtomsGlobalIndex[i];
    LocalID lid;
    // Search for a valid localID in all atoms
    for(int j = 0 ; j < atomMapsList.size(); j++){
      lid = atomMapsList[j]->localID(gid);
      if( lid.pid != -1) break; 
    }

    //JM NOTE: Fields of lid need to be != -1, bc the atom needs to be somewhere
    //          otherwise we have a bug
    if(lid.pid == -1){
      NAMD_bug(" LocalAtomID not found in patchMap");      
    }

    int soaPid = h_globalToLocalID[lid.pid]; // Converts global patch ID to its local position in our SOA data structures
    int soaIndex = localRecords[soaPid].bufferOffset + lid.index;

    smdAtomsSOAIndex[i] = soaIndex;
  }
  // Sort vector for better coalesce memory access
  std::sort(smdAtomsSOAIndex.begin(), smdAtomsSOAIndex.end());
  copy_HtoD<int>(smdAtomsSOAIndex.data(), d_smdAtomsSOAIndex, this->numSMDAtoms);
}

void ComputeSMDCUDA::doForce(
      const int timeStep, 
      const Lattice &lat, 
      const bool        doEnergy, 
      const float*      d_mass, 
      const double*     d_pos_x,
      const double*     d_pos_y,
      const double*     d_pos_z,
      const char3*      d_transform,
      double*           d_f_normal_x,
      double*           d_f_normal_y,
      double*           d_f_normal_z,
      cudaTensor*       d_extVirial,
      double*           h_extEnergy,  
      double3*          h_extForce, 
      cudaTensor*       h_extVirial, 
      cudaStream_t      stream
    )
{
  const int doOutput = (timeStep % this->outputFrequency) == 0;
  computeSMDForce(
    lat, 
    this->inv_group_mass,
    this->springConstant,
    this->transverseSpringConstant, 
    this->velocity, 
    this->direction,
    doEnergy || doOutput, 
    timeStep,
    this->origCOM,
    d_mass, 
    d_pos_x, 
    d_pos_y,
    d_pos_z, 
    d_transform, 
    d_f_normal_x, 
    d_f_normal_y, 
    d_f_normal_z, 
    this->numSMDAtoms, 
    this->d_smdAtomsSOAIndex, 
    this->d_curCOM, 
    this->curCOM, 
    d_extVirial,
    h_extEnergy, 
    h_extForce, 
    h_extVirial, 
    this->d_tbcatomic,
    stream
   );

    if(doOutput){
      cudaCheck(cudaStreamSynchronize(stream));
      Vector p(curCOM->x, curCOM->y, curCOM->z);
      Vector f(h_extForce->x, h_extForce->y, h_extForce->z);
      if(timeStep % (100*this->outputFrequency) == 0) {
        iout << "SMDTITLE: TS   CURRENT_POSITION         FORCE\n" << endi; 
      }
      iout << "SMD  " << timeStep << ' ' << p << ' ' << f*PNPERKCALMOL << '\n' << endi;
    }
}

#endif // NODEGROUP_FORCE_REGISTER
