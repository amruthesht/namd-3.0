#include "ComputeGroupRestraintsCUDA.h"
#include "ComputeGroupRes1GroupCUDAKernel.h"
#include "ComputeGroupRes2GroupCUDAKernel.h"
#include "SimParameters.h"
#include "Node.h"
#include "Molecule.h"

#ifdef NODEGROUP_FORCE_REGISTER
GroupRestraintsCUDA::GroupRestraintsCUDA(const GroupRestraintParam *param) {
    resParam = param;
    groupName = param->GetGroupName();
    restraintExp = param->GetExponent();
    restraintK = param->GetForce();
    inv_group1_mass = 0.0;
    inv_group2_mass = 0.0;
    const std::vector<int> &group1Index = param->GetGroup1AtomIndex();
    const std::vector<int> &group2Index = param->GetGroup2AtomIndex();
    numRestrainedGroup1 = group1Index.size();
    numRestrainedGroup2 = group2Index.size();
    totalNumRestrained = numRestrainedGroup1 + numRestrainedGroup2; 
    // If we defined a list of atoms for group 1, then we have to
    // calculate the COM for group 1 at every steps
    calcGroup1COM = (numRestrainedGroup1 ? true : false);
    // groupAtomsSOAIndex stores SOAindex of group 1, followed by 
    // SOAindex of group 2 
    groupAtomsSOAIndex.resize(totalNumRestrained);
    useDistMagnitude = param->GetUseDistMagnitude();
    Vector center = param->GetResCenter();
    Vector dir = param->GetResDirection();
    resDirection = make_double3(dir.x, dir.y, dir.z);
    resCenterVec = make_double3(center.x, center.y, center.z);

    // Allocate host and device memory
    allocate_host<double>(&h_resEnergy, 1);
    allocate_host<double3>(&h_diffCOM, 1);
    allocate_host<double3>(&h_group1COM, 1);
    allocate_host<double3>(&h_group2COM, 1);
    allocate_host<double3>(&h_resForce, 1);
    allocate_device<double3>(&d_group1COM, 1);
    allocate_device<double3>(&d_group2COM, 1);
    allocate_device<int>(&d_groupAtomsSOAIndex, totalNumRestrained);
    allocate_device<unsigned int>(&d_tbcatomic, 1);
    // Set the counter to zero
    cudaCheck(cudaMemset(d_tbcatomic, 0, sizeof(unsigned int)));

    // Check the atom index and calculate he inverse mass
    Molecule *mol = Node::Object()->molecule;
    int totalAtoms = mol->numAtoms;
    double total_mass = 0.0;

    for (int i = 0; i < numRestrainedGroup2; ++i) {
        int index = group2Index[i]; 
        if (index > -1 && index < totalAtoms) {
            total_mass += mol->atommass(index);    
        } else {
            char err_msg[512];
            sprintf(err_msg, "Group restraints: Bad atom index for %s!"
                " Atom indices must be within [%d, %d].\n", groupName, 0, totalAtoms - 1);
            NAMD_die(err_msg);
        }
    }
    inv_group2_mass = 1.0 / total_mass; 

    // Do we need to calculate COM of group 1, or we have a
    // reference position for it?
    if (calcGroup1COM) {
        total_mass = 0.0;
        for (int i = 0; i < numRestrainedGroup1; ++i) {
            int index = group1Index[i]; 
            if (index > -1 && index < totalAtoms) {
                total_mass += mol->atommass(index);    
            } else {
                char err_msg[512];
                sprintf(err_msg, "Group restraints: Bad atom index for %s!"
                    " Atom indices must be within [%d, %d].\n", groupName, 0, totalAtoms - 1);
                NAMD_die(err_msg);
            }
        }
        inv_group1_mass = 1.0 / total_mass; 
    } else {
        // We defined the reference point for COM of group 1, so no need
        // to calculate it, just copy it.
        // Set the h_group1COM to reference COM position of group 1
        Vector ref = param->GetGroupRes1Position();
        h_group1COM->x = ref.x;
        h_group1COM->y = ref.y;
        h_group1COM->z = ref.z;
    }
}

GroupRestraintsCUDA::~GroupRestraintsCUDA() {
    deallocate_host<double>(&h_resEnergy);
    deallocate_host<double3>(&h_resForce);
    deallocate_host<double3>(&h_diffCOM);
    deallocate_host<double3>(&h_group1COM);
    deallocate_host<double3>(&h_group2COM);
    deallocate_device<double3>(&d_group1COM); 
    deallocate_device<double3>(&d_group2COM); 
    deallocate_device<int>(&d_groupAtomsSOAIndex);
    deallocate_device<unsigned int>(&d_tbcatomic);
}

/*! Update the global index to local position in SOA data structure for a group */
void GroupRestraintsCUDA::updateAtoms(
      std::vector<AtomMap*> &atomMapsList,
      std::vector<CudaLocalRecord> &localRecords,
      const int *h_globalToLocalID) {

    // If we need to calculate COM of group 1, we have to store
    // index to SOA data structures for group 1 
    if(calcGroup1COM) {
        const std::vector<int> &group1Index = resParam->GetGroup1AtomIndex();
        if(numRestrainedGroup1 != group1Index.size()) {
            char err_msg[512];
            sprintf("Number of atoms in group 1 restraint for '%s' is changed!", groupName);
            NAMD_bug(err_msg);
        }

        // Map the global index to local position in SOA data structure 
        for(int i = 0 ; i < numRestrainedGroup1; ++i){
            int gid = group1Index[i];
            LocalID lid;
            // Search for a valid localID in all atoms
            for(int j = 0 ; j < atomMapsList.size(); ++j){
                lid = atomMapsList[j]->localID(gid);
                if( lid.pid != -1) {
                    break;
                } 
            }

            // Fields of lid need to be != -1, bc the atom needs to be somewhere
            //  otherwise we have a bug
            if(lid.pid == -1){
                NAMD_bug("LocalAtomID not found in patchMap");      
            }

            // Converts global patch ID to its local position in our SOA data structures
            int soaPid = h_globalToLocalID[lid.pid]; 
            int soaIndex = localRecords[soaPid].bufferOffset + lid.index;

            groupAtomsSOAIndex[i] = soaIndex;
        }
        // Sort vector for better coalesce memory access. Just sort only group 1
        std::sort(groupAtomsSOAIndex.begin(), groupAtomsSOAIndex.begin() + numRestrainedGroup1);
    }

    // We always calculate the COM of group 2, so we store
    // SOAIndex of group 2, after group 1 index
    const std::vector<int> &group2Index = resParam->GetGroup2AtomIndex();
    if(numRestrainedGroup2 != group2Index.size()) {
        char err_msg[512];
        sprintf("Number of atoms in group 2 restraint for '%s' is changed!", groupName);
        NAMD_bug(err_msg);
    }

    // Map the global index to local position in SOA data structure 
    for(int i = 0 ; i < numRestrainedGroup2; ++i){
        int gid = group2Index[i];
        LocalID lid;
        // Search for a valid localID in all atoms
        for(int j = 0 ; j < atomMapsList.size(); ++j){
            lid = atomMapsList[j]->localID(gid);
            if( lid.pid != -1) {
                break;
            } 
        }

        // Fields of lid need to be != -1, bc the atom needs to be somewhere
        //  otherwise we have a bug
        if(lid.pid == -1){
            NAMD_bug("LocalAtomID not found in patchMap");      
        }

        // Converts global patch ID to its local position in our SOA data structures
        int soaPid = h_globalToLocalID[lid.pid]; 
        int soaIndex = localRecords[soaPid].bufferOffset + lid.index;
        // store the index for group 2, after group 1 index
        groupAtomsSOAIndex[i + numRestrainedGroup1] = soaIndex;
    }
    // Sort vector for better coalesce memory access. Sort only for group 2
    std::sort(groupAtomsSOAIndex.begin() + numRestrainedGroup1, groupAtomsSOAIndex.end());
    
    // Update the SOA index in device
    copy_HtoD<int>(groupAtomsSOAIndex.data(), d_groupAtomsSOAIndex, totalNumRestrained);
}

/*! Compute harmonic restraint energy, force, and virial for restrained groups */   
void GroupRestraintsCUDA::doForce(
    const int timeStep,
    const int doEnergy,
    const int doVirial,
    const int doOutput,  
    const Lattice &lat, 
    const char3*  d_transform, 
    const float*  d_mass, 
    const double* d_pos_x,
    const double* d_pos_y,
    const double* d_pos_z,
    double*       d_f_normal_x,
    double*       d_f_normal_y,
    double*       d_f_normal_z,
    cudaTensor*   d_virial,
    double*       h_extEnergy, 
    double3*      h_extForce, 
    cudaTensor*   h_extVirial, 
    cudaStream_t  stream) {

    if (calcGroup1COM) {
        computeGroupRestraint_2Group(
        useDistMagnitude,
        doEnergy,
        doVirial,
        this->numRestrainedGroup1,
        this->totalNumRestrained,
        this->restraintExp,
        this->restraintK,
        this->resCenterVec,
        this->resDirection,
        this->inv_group1_mass,
        this->inv_group2_mass,
        this->d_groupAtomsSOAIndex,
        lat,
        d_transform,
        d_mass,
        d_pos_x,
        d_pos_y,
        d_pos_z,
        d_f_normal_x,
        d_f_normal_y,
        d_f_normal_z,
        d_virial,
        h_extVirial,
        this->h_resEnergy,
        this->h_resForce,
        this->h_group1COM,
        this->h_group2COM,
        this->h_diffCOM,
        this->d_group1COM,
        this->d_group2COM,
        this->d_tbcatomic,
        stream);
    } else {
        computeGroupRestraint_1Group(
        useDistMagnitude,
        doEnergy,
        doVirial,
        this->numRestrainedGroup2,
        this->restraintExp,
        this->restraintK,
        this->resCenterVec,
        this->resDirection,
        this->inv_group2_mass,
        this->d_groupAtomsSOAIndex,
        lat,
        d_transform,
        d_mass,
        d_pos_x,
        d_pos_y,
        d_pos_z,
        d_f_normal_x,
        d_f_normal_y,
        d_f_normal_z,
        d_virial,
        h_extVirial,
        this->h_resEnergy,
        this->h_resForce,
        this->h_group1COM,
        this->h_group2COM,
        this->h_diffCOM,
        this->d_group2COM,
        this->d_tbcatomic,
        stream);
    }
    

    if(doOutput){
        cudaCheck(cudaStreamSynchronize(stream));
        // sum up external energy and virial from this group
        h_extEnergy[0] += h_resEnergy[0];
        // If we have restraint to reference point, then we have net force
        // otherwise the net force is zero for restraining two atom groups
        if (!calcGroup1COM) {
            h_extForce->x += h_resForce->x;
            h_extForce->y += h_resForce->y;
            h_extForce->z += h_resForce->z;
        }
      
        char msg[1024];
        sprintf(msg,"GRES: %9d %14s %14.4f %14.4f %14.4f %19.4f %14.4f %14.4f %14.4f\n",
            timeStep, groupName, h_diffCOM->x, h_diffCOM->y, h_diffCOM->z,
            h_resForce->x, h_resForce->y, h_resForce->z, h_resEnergy[0]);
        iout << msg << endi;

        // {
        //     printf("!!!Accu. exForce: %14.8f %14.8f %14.8f, Vir.x: %14.8f %14.8f %14.8f,"
        //     "Vir.y: %14.8f %14.8f %14.8f, Vir.z: %14.8f %14.8f %14.8f \n",
        //     h_extForce->x, h_extForce->y, h_extForce->z,
        //     h_extVirial->xx, h_extVirial->xy, h_extVirial->xz,
        //     h_extVirial->yx, h_extVirial->yy, h_extVirial->yz,
        //     h_extVirial->zx, h_extVirial->zy, h_extVirial->zz);
        // }
    }
}

// ###########################################################################
// # ComputeGroupRestraintsCUDA functions
// ###########################################################################

ComputeGroupRestraintsCUDA::ComputeGroupRestraintsCUDA(const int ouputFreq,
    const GroupRestraintList &resList) {
    gResOutputFreq = ouputFreq; 
    const std::map<std::string, GroupRestraintParam*> & groupMap = resList.GetGroupResMap();
    for (auto it = groupMap.begin(); it != groupMap.end(); ++it) {
        GroupRestraintsCUDA * gResCUDA = new GroupRestraintsCUDA(it->second);
        restraintsCUDAList.push_back(gResCUDA);
    }
}

ComputeGroupRestraintsCUDA::~ComputeGroupRestraintsCUDA() {
    int numGroup = restraintsCUDAList.size(); 
    for (int i = 0; i < numGroup; ++i) {
        delete restraintsCUDAList[i]; 
    }
    restraintsCUDAList.clear(); 
}

/*! Update the global index to local position in SOA data structure for all groups */
void ComputeGroupRestraintsCUDA::updateAtoms(
        std::vector<AtomMap*> &atomMapsList,
        std::vector<CudaLocalRecord> &localRecords,
        const int *h_globalToLocalID) {

    int numGroup = restraintsCUDAList.size(); 
    for (int i = 0; i < numGroup; ++i) {
        restraintsCUDAList[i]->updateAtoms(atomMapsList, localRecords, h_globalToLocalID); 
    }
}

/*! Compute harmonic restraint energy, force, and virial for all groups */  
void ComputeGroupRestraintsCUDA::doForce(
        const int timeStep, 
        const int doEnergy,
        const int doVirial,
        const Lattice &lat, 
        const char3*  d_transform, 
        const float*  d_mass, 
        const double* d_pos_x,
        const double* d_pos_y,
        const double* d_pos_z,
        double*       d_f_normal_x,
        double*       d_f_normal_y,
        double*       d_f_normal_z,
        cudaTensor*   d_virial,
        double*       h_extEnergy, 
        double3*      h_extForce, 
        cudaTensor*   h_extVirial, 
        cudaStream_t  stream) {

    const int doOutput = (timeStep % gResOutputFreq) == 0;
    // Since output freq is same as energyOutputFrq, we need to calculate virial
    // for outputting energy data
    int doVirCalc = (doOutput ? 1 : doVirial);
    int numGroup = restraintsCUDAList.size();

    // Reset the values before we add the energy, force, and virial value
    // for each restraint group
    h_extEnergy[0]  = 0.0;
    h_extForce->x   = 0.0;
    h_extForce->y   = 0.0; 
    h_extForce->z   = 0.0;  
    h_extVirial->xx = 0.0;
    h_extVirial->xy = 0.0;
    h_extVirial->xz = 0.0;
    h_extVirial->yx = 0.0;
    h_extVirial->yy = 0.0;
    h_extVirial->yz = 0.0;
    h_extVirial->zx = 0.0;
    h_extVirial->zy = 0.0;
    h_extVirial->zz = 0.0;

    if(doOutput){
      if(timeStep % (100 * gResOutputFreq) == 0) {
        char msg[1024];
        sprintf(msg,"\nGRES_TITLE: %3s %14s %14s %14s %14s %19s %14s %14s %14s\n",
            "TS", "GROUP_NAME", "DISTANCE.X", "DISTANCE.Y", "DISTANCE.Z",
            "FORCE.X", "FORCE.Y", "FORCE.Z", "ENERGY");
        iout << msg << endi;
      }
    }

    for (int gIdx = 0; gIdx < numGroup; ++gIdx) { 
        restraintsCUDAList[gIdx]->doForce( 
            timeStep, doEnergy, doVirCalc, doOutput, lat, d_transform, 
            d_mass, d_pos_x, d_pos_y, d_pos_z, 
            d_f_normal_x, d_f_normal_y, d_f_normal_z, d_virial, 
            h_extEnergy, h_extForce, h_extVirial, stream);
    }
    if(doOutput) {
        iout <<"\n" << endi;;
    }
}




#endif
