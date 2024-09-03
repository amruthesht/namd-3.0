#include "GroupRestraintsParam.h"


GroupRestraintParam::GroupRestraintParam(void) {
    groupNameDefined = false;
    restraintForceDefined = false;
    group1RefPositionDefined = false;
    restraintCenterDefined = false;
    groupName = NULL;
    group1Idx.clear();
    group2Idx.clear();
    restraintExp = 2;          /**< Default restraint exponent is 2 */
    restraintForce = 0.0;      /**< Default force */
    group1RefPosition = 0.0;   /**< Default reference COM position for group 1 */
    restraintDir = 1.0;        /**< Default restraint in X,Y, and Z direction */
    restraintCenter = 0.0;     /**< Default center of restraint */
    useDistanceMagnitude = false; /**< Default is to use vector distance, not magnitude */
}

GroupRestraintParam::~GroupRestraintParam(void) {
    if (groupName) delete [] groupName;
}

/*! Read the restrained atom indices for group 1 from file and store it */
void GroupRestraintParam::SetGroup1AtomFileIndices(const char *fileName) {
    char err_msg[512]; //  Buffer for error message
    char buffer[512];  //  Buffer for file reading

    if(group1Idx.size()) {
        group1Idx.clear();
        sprintf(err_msg, "Group restraints: Redefining existing group 1 indices for %s!\n",
            groupName);
        iout << iWARN << err_msg << "\n" << endi;   
    }

    if (! fileName) {
        sprintf(err_msg, "Group restraints: No group 1 restraint file is defined for '%s'!\n", groupName);
        NAMD_die(err_msg); 
    }

    FILE *file = fopen(fileName, "r");
    if (!file) {
        sprintf(err_msg, "Group restraints: Unable to open group 1 restraint file '%s'!\n", fileName);
        NAMD_die(err_msg); 
    } else {
        iout << iINFO << "Reading group 1 restraint file " << fileName << " for group " <<
                groupName << "\n" << endi; 
    }

    while (true) {
        int index;
        int badLine = 0;
        int return_code = 0;
        do {
            return_code = NAMD_read_line(file, buffer, 512);
        } while ( (return_code == 0) && (NAMD_blank_string(buffer)) );

        if (return_code) {
            break;
        }
        if (sscanf(buffer, "%d", &index) != 1) {
            sprintf(err_msg, "Group restraints: Bad line in group 1 restraint file '%s': %s!\n", fileName, buffer);
            NAMD_die(err_msg);
        } else {
            group1Idx.push_back(index);
        }
    }
    fclose(file);
}

/*! Read the restrained atom indices for group 2 from file and store it */
void GroupRestraintParam::SetGroup2AtomFileIndices(const char *fileName) {
    char err_msg[512]; //  Buffer for error message
    char buffer[512];  //  Buffer for file reading

    if(group2Idx.size()) {
        group2Idx.clear();
        sprintf(err_msg, "Group restraints: Redefining existing group 2 indices for %s!\n",
            groupName);
        iout << iWARN << err_msg << "\n" << endi;   
    }

    if (! fileName) {
        sprintf(err_msg, "Group restraints: No group 2 restraint file is defined for '%s'!\n", groupName);
        NAMD_die(err_msg); 
    }

    FILE *file = fopen(fileName, "r");
    if (!file) {
        sprintf(err_msg, "Group restraints: Unable to open group 2 restraint file '%s'!\n", fileName);
        NAMD_die(err_msg); 
    } else {
        iout << iINFO << "Reading group 2 restraint file " << fileName << " for group " <<
                groupName << "\n" << endi; 
    }

    while (true) {
        int index;
        int badLine = 0;
        int return_code = 0;
        do {
            return_code = NAMD_read_line(file, buffer, 512);
        } while ( (return_code == 0) && (NAMD_blank_string(buffer)) );

        if (return_code) {
            break;
        }
        if (sscanf(buffer, "%d", &index) != 1) {
            sprintf(err_msg, "Group restraints: Bad line in group 2 restraint file '%s': %s!\n", fileName, buffer);
            NAMD_die(err_msg);
        } else {
            group2Idx.push_back(index);
        }
    }
    fclose(file);
}

/*! Set the restrained atom indices for group 1 from a list*/
void GroupRestraintParam::SetGroup1AtomListIndices(const char *list) {
    char err_msg[512]; //  Buffer for error message
    int index, numRead;
    int i = 0;

    if(group1Idx.size()) {
        group1Idx.clear();
        sprintf(err_msg, "Group restraints: Redefining existing group 1 restraint indices for %s!\n",
            groupName);
        iout << iWARN << err_msg << "\n" << endi;   
    }

    // Get the length of string
    int strLength = strlen(list);
    // Read the index and advance the number of read char
    // until we reach to the end of it
    while (i < strLength) {
        if(sscanf(list,"%d%n", &index, &numRead) != 1) {
            sprintf(err_msg, "Group restraints: Bad line in group 1 restraint list for '%s': %s!\n",
                groupName, list);
            NAMD_die(err_msg);
        } else {
            list += numRead + 1;
            i += numRead + 1;
            group1Idx.push_back(index);
        }
    } 
}

/*! Set the restrained atom indices for group 2 from a list*/
void GroupRestraintParam::SetGroup2AtomListIndices(const char *list) {
    char err_msg[512]; //  Buffer for error message
    int index, numRead;
    int i = 0;

    if(group2Idx.size()) {
        group2Idx.clear();
        sprintf(err_msg, "Group restraints: Redefining existing group 2 restraint indices for %s!\n",
            groupName);
        iout << iWARN << err_msg << "\n" << endi;   
    }

    // Get the length of string
    int strLength = strlen(list);
    // Read the index and advance the number of read char
    // until we reach to the end of it
    while (i < strLength) {
        if(sscanf(list,"%d%n", &index, &numRead) != 1) {
            sprintf(err_msg, "Group restraints: Bad line in group 2 restraint list for '%s': %s!\n",
                groupName, list);
            NAMD_die(err_msg);
        } else {
            list += numRead + 1;
            i += numRead + 1;
            group2Idx.push_back(index);
        }
    } 
}

/*! Set restraint group name */
void GroupRestraintParam::SetGroupName(const char *name) {
    char err_msg[512]; //  Buffer for error message
    if (groupName) {
        sprintf(err_msg, "Group restraints: Redefining existing group restraint name from %s to %s!\n",
            groupName, name);
        NAMD_die(err_msg);
    } else {
        groupNameDefined = true;
        int nameLength = strlen(name);
        groupName = new char[nameLength + 1];
        strncpy(groupName, name, nameLength + 1);
    }
}

/*! Set the restraint force constant */
void GroupRestraintParam::SetForce(const BigReal force) {
    char err_msg[512]; //  Buffer for error message
    if (restraintForceDefined) {
        sprintf(err_msg, "Group restraints: Redefining restraint force for %s!\n",
            groupName);
        iout << iWARN << err_msg << "\n" << endi;   
    }
    restraintForce = force; 
    restraintForceDefined = true;
}

/*! Set the restraint exponent */
void GroupRestraintParam::SetExponent(const int exponent) {
    restraintExp = exponent; 
}

/*! Set the reference COM position for group 1 */
void GroupRestraintParam::SetGroup1RefPosition(const char *vec) {
    char err_msg[512]; //  Buffer for error message
    if (group1RefPositionDefined) {
        sprintf(err_msg, "Group restraints: Redefining group 1 reference COM position for %s!\n",
            groupName);
        iout << iWARN << err_msg << "\n" << endi;   
    }

    if(group1RefPosition.set(vec)) {
        group1RefPositionDefined = true; 
    } else {
        sprintf(err_msg, "Group restraints: Bad reference COM position for %s: { %s }. %s\n",
            groupName, vec, "Expect a {x y z} vector.");
        NAMD_die(err_msg);
    }
}

/*! Set the center or equilibrium value of restraint */
void GroupRestraintParam::SetResCenter(const char *vec) {
    char err_msg[512]; //  Buffer for error message
    if (restraintCenterDefined) {
        sprintf(err_msg, "Group restraints: Redefining restraint center for %s!\n",
            groupName);
        iout << iWARN << err_msg << "\n" << endi;   
    }

    if(restraintCenter.set(vec)) {
        restraintCenterDefined = true; 
    } else {
        sprintf(err_msg, "Group restraints: Bad restraint center value for %s: { %s }. %s\n",
            groupName, vec, "Expect a {x y z} vector.");
        NAMD_die(err_msg);
    }
}

/*! Set restraint vector component (X, Y, Z) */
void GroupRestraintParam::SetResDirection(const char *status, const int component) {
    // Only modify the resDir if we turn off the specific component
    // because the default behaviour is that we apply restraint in
    // all direction 
    BigReal value = this->CheckStatus<BigReal>(status);

    switch (component) {
        case 0 :
            restraintDir.x = value;
            break;
        case 1 :
            restraintDir.y = value;
            break; 
        case 2 :
            restraintDir.z = value;
            break;   
        default :
            NAMD_die("Group restraints: Unknown vector component in SetResDirection function! \n");
    }
}

/*! Set restraint distance mode (magnitude or vector) */
void GroupRestraintParam::SetUseDistMagnitude(const char *status) {
    useDistanceMagnitude = this->CheckStatus<bool>(status); 
}

/*! Check all necessary parameters are set*/
void GroupRestraintParam::CheckParam(void) {
    char err_msg[512];

    if (!groupNameDefined) {
        sprintf(err_msg, "Group restraints: Key or tag name is not defined!\n");
        NAMD_die(err_msg);
    }

    if (!restraintForceDefined) {
        sprintf(err_msg, "Group restraints: Restraint Force constant is not defined for %s!\n", groupName);
        NAMD_die(err_msg);
    }

    if (!restraintCenterDefined) {
        sprintf(err_msg, "Group restraints: Restraint center is not defined for %s!\n", groupName);
        NAMD_die(err_msg);
    }

    if (!(restraintExp > 0)) {
        sprintf(err_msg, "Group restraints: Restraint exponent must be positive value for %s!\n", groupName);
        NAMD_die(err_msg);
    }

    if (restraintExp % 2) {
        sprintf(err_msg, "Group restraints: Restraint exponent must be an even number for %s!\n", groupName);
        NAMD_die(err_msg);
    }

    if (!(restraintForce > 0)) {
        sprintf(err_msg, "Group restraints: Restraint Force constant must be positive value for %s!\n", groupName);
        NAMD_die(err_msg);
    }

    if (!group1RefPositionDefined && !group1Idx.size()) {
        sprintf(err_msg, "Group restraints: Either reference COM position or atom indices for group 1 must be defined for %s!\n", groupName);
        NAMD_die(err_msg);
    }

    if (group1RefPositionDefined && group1Idx.size()) {
        sprintf(err_msg, "Group restraints: Reference COM position and atom indices for group 1 cannot be defined together for %s!\n", groupName);
        NAMD_die(err_msg);
    }

    if (!(group2Idx.size())) {
        sprintf(err_msg, "Group restraints: No atom is defined for group 2 to be restrained for %s!\n", groupName);
        NAMD_die(err_msg);
    }

    if (!(restraintDir.length2())) {
        sprintf(err_msg, "Group restraints: At least one component of restraint distance "
            "must be selected for %s!\n", groupName);
        NAMD_die(err_msg);
    }
}

/*! Print summary of parameters in group restraint */
void GroupRestraintParam::PrintSummary() {
    iout << iINFO << "GROUP RESTRAINT KEY                  " << groupName << "\n" << endi;
    if (useDistanceMagnitude) {
        iout << iINFO << "       RESTRAINT DISTANCE MAGNITUDE  " << "ACTIVE\n" << endi;
        iout << iINFO << "       RESTRAINT CENTER              " << restraintCenter.length() << "\n" << endi;
    } else {
        iout << iINFO << "       RESTRAINT DISTANCE VECTOR     " << "ACTIVE\n" << endi;
        iout << iINFO << "       RESTRAINT CENTER              " << restraintCenter << "\n" << endi;
    }
    iout << iINFO << "       RESTRAINT FORCE               " << restraintForce << "\n" << endi;
    iout << iINFO << "       RESTRAINT EXPONENT            " << restraintExp << "\n" << endi;
    iout << iINFO << "       RESTRAINT COMPONENT X         " << (restraintDir.x ? "YES" : "NO") << "\n" << endi;
    iout << iINFO << "       RESTRAINT COMPONENT Y         " << (restraintDir.y ? "YES" : "NO") << "\n" << endi;
    iout << iINFO << "       RESTRAINT COMPONENT Z         " << (restraintDir.z ? "YES" : "NO") << "\n" << endi;
    iout << iINFO << "       RESTRAINED ATOMS IN GROUP 2   " << group2Idx.size() << "\n" << endi;
    if (group1RefPositionDefined) {
        iout << iINFO << "       COM REF. POSITION IN GROUP 1  " << group1RefPosition << "\n" << endi;
    } else {
        iout << iINFO << "       RESTRAINED ATOMS IN GROUP 1   " << group1Idx.size() << "\n" << endi;
    }
}

/*! Convert char status (on, off, ..) to type T */
template<typename Type>
Type GroupRestraintParam::CheckStatus(const char *status) const {
    Type state = static_cast<Type>(1);
    if (0 == strcasecmp(status, "no") || 
        0 == strcasecmp(status, "off") || 
        0 == strcasecmp(status, "false")) {
        state = static_cast<Type>(0);
    } else if (0 == strcasecmp(status, "yes") || 
        0 == strcasecmp(status, "on") || 
        0 == strcasecmp(status, "true")) {
        state = static_cast<Type>(1);
    } else {
        char err_msg[512];
        sprintf(err_msg, "Group restraints: Unknown status keyword '%s'!" 
            " Options are: no, off, false, yes, on, true.\n", status);
        NAMD_die(err_msg); 
    }
    return state;
}

// ###########################################################################
// # GroupRestraintList functions
// ###########################################################################

GroupRestraintList::~GroupRestraintList(){
    for (auto it = groupRestraints.begin(); it != groupRestraints.end(); ++it) {
        delete it->second;
    }
}

/*! Set the restrained atom indices for group 1 using a text file */
void GroupRestraintList::SetGroup1AtomFileIndices(const char *groupTag, const char *fileName)  {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetGroup1AtomFileIndices(fileName); 
}

/*! Set the restrained atom indices for group 1 using a list */
void GroupRestraintList::SetGroup1AtomListIndices(const char *groupTag, const char *list)  {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetGroup1AtomListIndices(list); 
}

/*! Set the reference COM position for group 1 */
void GroupRestraintList::SetGroup1RefPosition(const char *groupTag, const char *vec) {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetGroup1RefPosition(vec); 
}

/*! Set the restrained atom indices for group 2 using a text file */
void GroupRestraintList::SetGroup2AtomFileIndices(const char *groupTag, const char *fileName)  {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetGroup2AtomFileIndices(fileName); 
}

/*! Set the restrained atom indices for group 2 using a list */
void GroupRestraintList::SetGroup2AtomListIndices(const char *groupTag, const char *list)  {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetGroup2AtomListIndices(list); 
}

/*! Set the center or equilibrium value of the restraint */
void GroupRestraintList::SetResCenter(const char *groupTag, const char *vec) {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetResCenter(vec); 
}

/*! Set the restraint force constant */
void GroupRestraintList::SetForce(const char *groupTag, const BigReal force) {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetForce(force); 
}

/*! Set the restraint exponent */
void GroupRestraintList::SetExponent(const char *groupTag, const int exponent) {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetExponent(exponent); 
}

/*! Set restraint vector component X */
void GroupRestraintList::SetResDirectionX(const char *groupTag, const char *status) {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetResDirection(status, 0); 
}

/*! Set restraint vector component Y */
void GroupRestraintList::SetResDirectionY(const char *groupTag, const char *status) {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetResDirection(status, 1); 
}

/*! Set restraint vector component Z */
void GroupRestraintList::SetResDirectionZ(const char *groupTag, const char *status) {
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetResDirection(status, 2); 
}

/*! Set restraint distance mode (magnitude or vector) */
void GroupRestraintList::SetUseDistMagnitude(const char *groupTag, const char *status){
    GroupRestraintParam *resParam = FindGroupRestraint(groupTag);
    resParam->SetUseDistMagnitude(status); 
}

/*! Check all necessary parameters in group restraint are set */ 
void GroupRestraintList::CheckGroupRestraints() {
    for (auto it = groupRestraints.begin(); it != groupRestraints.end(); ++it) {
        it->second->CheckParam();
    }
}

/*! Print summary of parameters in group restraint */
void GroupRestraintList::PrintGroupRestraints() {
    for (auto it = groupRestraints.begin(); it != groupRestraints.end(); ++it) {
        it->second->PrintSummary();
    }
}

/*! Find a restraint group parameter with tag key.
    If no key was found in the map, insert a new object */
GroupRestraintParam* GroupRestraintList::FindGroupRestraint(const char *tag) {
    std::string groupName(tag);
    auto it = groupRestraints.find(groupName);
    if (it == groupRestraints.end()) {
        GroupRestraintParam *resParam = new GroupRestraintParam();
        resParam->SetGroupName(tag);
        groupRestraints.insert(std::make_pair(groupName, resParam));
        return groupRestraints.find(groupName)->second; 
    } else {
        return it->second;
    }
}
