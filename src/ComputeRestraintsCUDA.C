#include "ComputeRestraintsCUDA.h"
#include "Molecule.h"
#include "Node.h"
#include "HomePatch.h"
#include "SimParameters.h"
#include "ComputeRestraintsCUDAKernel.h"

#ifdef NODEGROUP_FORCE_REGISTER

ComputeRestraintsCUDA::ComputeRestraintsCUDA(
  std::vector<HomePatch*> &patchList, 
  std::vector<AtomMap*> &atomMapsList,
  cudaStream_t stream
) {

  Molecule *molecule       = Node::Object()->molecule;
  SimParameters *simParams = Node::Object()->simParameters;
  nConstrainedAtoms = 0;
  // Set the constraints flags
  consExp = simParams->constraintExp;
  // consScaling = simParams->constraintScaling;
  
  // Selective constraints flags
  selConsOn = simParams->selectConstraintsOn;

  if (selConsOn){
    consSelectX = simParams->constrXOn;
    consSelectY = simParams->constrYOn;
    consSelectZ = simParams->constrZOn;
  }

  // Moving Constraints flags
  movConsOn = simParams->movingConstraintsOn;
  if (movConsOn) {
    moveVel.x = simParams->movingConsVel.x;
    moveVel.y = simParams->movingConsVel.y;
    moveVel.z = simParams->movingConsVel.z;
  }

  // Rotating cosntraints flags
  rotConsOn = simParams->rotConstraintsOn;
  if (rotConsOn) {
    rotVel = simParams->rotConsVel; // velocity is a scalar here
    rotAxis.x = simParams->rotConsAxis.x;
    rotAxis.y = simParams->rotConsAxis.y;
    rotAxis.z = simParams->rotConsAxis.z;
  }

  // Spherical Constraints flags
  spheConsOn = simParams->sphericalConstraintsOn;
  if(spheConsOn){
    spheConsCenter.x = simParams->sphericalConstrCenter.x;
    spheConsCenter.y = simParams->sphericalConstrCenter.y;
    spheConsCenter.z = simParams->sphericalConstrCenter.z;
  }

  // Set all flags, now allocate constraints data structures
  // We need to go through the patches and add stuff to a constraint vector
  for(int i = 0; i < patchList.size(); i++){
    HomePatch *p = patchList[i];
    for(int j = 0; j < p->getNumAtoms(); j++){
      int gid = p->getAtomList()[j].id; // Gets global ID of corresponding FullAtom
      if (molecule->is_atom_constrained(gid)){
        nConstrainedAtoms++;
        h_constrainedID.push_back(gid); // Pushes back global ID vector
        float k;
        Vector refPos;
        molecule->get_cons_params(k, refPos, gid);
        h_k.push_back(k);
        h_cons_x.push_back(refPos.x);
        h_cons_y.push_back(refPos.y);
        h_cons_z.push_back(refPos.z);
      }
    }
  }
  this->stream = stream;

  allocate_host<int>(&h_constrainedSOA, nConstrainedAtoms);
  allocate_device<unsigned int>(&d_tbcatomic, 1);
  allocate_device<int>(&d_constrainedSOA, nConstrainedAtoms);
  allocate_device<int>(&d_constrainedID, nConstrainedAtoms);
  allocate_device<double>(&d_k, nConstrainedAtoms);
  allocate_device<double>(&d_cons_x, nConstrainedAtoms);
  allocate_device<double>(&d_cons_y, nConstrainedAtoms);
  allocate_device<double>(&d_cons_z, nConstrainedAtoms);

  copy_HtoD_sync<double>(h_k.data(), d_k, nConstrainedAtoms);
  copy_HtoD_sync<double>(h_cons_x.data(), d_cons_x, nConstrainedAtoms);
  copy_HtoD_sync<double>(h_cons_y.data(), d_cons_y, nConstrainedAtoms);
  copy_HtoD_sync<double>(h_cons_z.data(), d_cons_z, nConstrainedAtoms);
  copy_HtoD_sync<int>(h_constrainedID.data(), d_constrainedID, nConstrainedAtoms);
  
  cudaCheck(cudaMemset(d_tbcatomic, 0, sizeof(unsigned int))); // sets the scalar to zero

  // this->updateRestrainedAtoms(atomMapsList, h_globalToLocalID, h_patchOffsets);
}

void ComputeRestraintsCUDA::updateRestrainedAtoms(
  std::vector<AtomMap*> &atomMapsList, 
  std::vector<CudaLocalRecord> &localRecords,
  const int* h_globalToLocalID
){
  // JM NOTE: This gets called for every migration step, so it would be good to have this somewhat fast
  // This is serialized, so if we have a lot of constrained atoms, it might become a bottleneck in the future

  for(int i = 0; i < nConstrainedAtoms; i++){
    // translates the global ID to the SOA ID.
    int gid = h_constrainedID[i];
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

    // JM: Now that we have a patchID and a localPosition inside the patch, I can figure out 
    //     the SOA position for each constrained atom
    
    int soaPid = h_globalToLocalID[lid.pid]; // Converts global patch ID to its local position in our SOA data structures
    int soaIndex = localRecords[soaPid].bufferOffset + lid.index;

    h_constrainedSOA[i] = soaIndex;
  }

  // Copy the h_constrainedSOA data structure over to the GPU
  copy_HtoD_sync<int>(h_constrainedSOA, d_constrainedSOA, nConstrainedAtoms);  

}

// doForce is called every time step, so no copies here
void ComputeRestraintsCUDA::doForce(
  const Lattice *lat, 
  const bool doEnergy,
  const bool doVirial, 
  const int timeStep, 
  double* d_pos_x,
  double* d_pos_y, 
  double* d_pos_z, 

  double* f_normal_x, 
  double* f_normal_y,
  double* f_normal_z, 
  double* d_bcEnergy, 
  double* h_bcEnergy,
  double3* d_netForce,
  double3* h_netForce,
  cudaTensor* d_virial, 
  cudaTensor* h_virial
){
  SimParameters *simParams = Node::Object()->simParameters;

  computeRestrainingForce(
    doEnergy, 
    doVirial, 
    timeStep, 
    nConstrainedAtoms, 
    consExp, 
    // consScaling, 
    simParams->constraintScaling, // read directly from SimParameters
        // to make sure we always calculate using latest value
    movConsOn, 
    rotConsOn, 
    selConsOn, 
    spheConsOn,
    consSelectX,
    consSelectY, 
    consSelectZ, 
    rotVel, 
    rotAxis,
    rotPivot, 
    moveVel, 
    spheConsCenter, 
    d_constrainedSOA, 
    d_constrainedID, 
    d_pos_x, 
    d_pos_y, 
    d_pos_z, 
    d_k, 
    d_cons_x,
    d_cons_y, 
    d_cons_z, 
    f_normal_x, 
    f_normal_y, 
    f_normal_z, 
    d_bcEnergy, 
    h_bcEnergy, 
    d_netForce, 
    h_netForce, 
    lat, 
    d_virial, 
    h_virial, 
    rotationMatrix, 
    d_tbcatomic,
    stream
  );

}


ComputeRestraintsCUDA::~ComputeRestraintsCUDA()
{
  deallocate_device<unsigned int>(&d_tbcatomic);
  deallocate_device<int>(&d_constrainedSOA);
  deallocate_device<int>(&d_constrainedID);
  deallocate_device<double>(&d_cons_x);
  deallocate_device<double>(&d_cons_y);
  deallocate_device<double>(&d_cons_z);
  deallocate_device<double>(&d_k);
}

#endif // NODEGROUP_FORCE_REGISTER
