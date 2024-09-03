#ifndef GROUP_RESTRAINTS_PARAM
#define GROUP_RESTRAINTS_PARAM 

#include "strlib.h"
#include "common.h"
#include "Vector.h"
#include "InfoStream.h"
#include "MStream.h"
#include "strlib.h"
#include <vector>
#include <string>
#include <iterator>
#include <map>


/*! 
    Group restraint parameter class.
    Stores parameters required for harmonic restraints.
    Make sure that all necessary parameters are provided.
    Read atom indicies from file.
    Print summary of parameters.
*/
class GroupRestraintParam {

public:
    GroupRestraintParam(void);

    ~GroupRestraintParam(void);

    /*! Read the restrained atom indicies for group 1 from file and store it */
    void SetGroup1AtomFileIndices(const char *fileName);

    /*! Set the restrained atom indicies for group 1 from a list */
    void SetGroup1AtomListIndices(const char *list);

    /*! Set the reference COM position for group 1 */
    void SetGroup1RefPosition(const char *vec);

    /*! Read the restained atom indicies for group 2 from file and store it */
    void SetGroup2AtomFileIndices(const char *fileName);

    /*! Set the restrained atom indicies for group 2 from a list */
    void SetGroup2AtomListIndices(const char *list);

    /*! Set restraint group name */
    void SetGroupName(const char *name);

    /*! Set the restraint force constant */
    void SetForce(const BigReal force);

    /*! Set the restraint exponent */
    void SetExponent(const int exponent);

    /*! Set the center of equilibrium value of restraint */
    void SetResCenter(const char *vec);

    /*! Set restraint vector component (X, Y, Z) */
    void SetResDirection(const char *status, const int component);

    /*! Set restraint distance mode (magnitude or vector) */
    void SetUseDistMagnitude(const char *status);

    /*! Check all necessary parameters in group restraint are set */
    void CheckParam(void);

    /*! Print summary of parameters in group restraint */
    void PrintSummary();

    /*! Get restraint group name */
    const char* GetGroupName() const {return groupName;}

    /*! Get the restraint exponent */
    int GetExponent() const {return restraintExp;}

    /*! Get the restraint force constant */
    double GetForce() const {return restraintForce;}

    /*! Get the reference COM position for group 1 */
    Vector GetGroupRes1Position() const {return group1RefPosition;}

    /*! Get restraint vector component (X, Y, Z) */
    Vector GetResDirection() const {return restraintDir;}

    /*! Get the center of restraint */
    Vector GetResCenter() const {return restraintCenter;}

    /*! Get reference to restrained atom indicies in group 1 */
    const std::vector<int> &GetGroup1AtomIndex() const {return group1Idx;}

    /*! Get reference to restrained atom indicies in group 2 */
    const std::vector<int> &GetGroup2AtomIndex() const {return group2Idx;}

    /* Get the restraint distance mode true for magnitude, false for vector */
    bool GetUseDistMagnitude() const {return useDistanceMagnitude;}

private:
    /*! Convert char status (on, off, ..) to type T */
    template<typename Type>
    Type CheckStatus(const char *status) const;

    char *groupName;                 /**< Restraint group name */  
    bool restraintForceDefined;      /**< To verify if force constant is defined */
    bool group1RefPositionDefined;   /**< To verify if reference COM position for group 1 is defined */
    bool groupNameDefined;           /**< To verify if group name is defined */ 
    bool restraintCenterDefined;     /**< To verify if center of restraint is defined */
    bool useDistanceMagnitude;       /**< Restraining the group 1 and 2, using distance magnitude or vector */ 

    int restraintExp;                /**< Restraint exponent */
    double restraintForce;           /**< Restraint force constant */
    Vector restraintCenter;          /**< Restraint center or equilibrium vector */
    Vector group1RefPosition;        /**< Reference COM position for group 1 */
    Vector restraintDir;             /**< Restraint distance component, to decide if we aplly restaint in X, Y, or Z direction */
    std::vector<int> group1Idx;      /**< Indices of atoms in group 1 */
    std::vector<int> group2Idx;      /**< Indices of atoms in group 2 */
};

/*!
    Group restraints class.
    Set restraints parameters for specific group tag.
*/
class GroupRestraintList {

public:
    ~GroupRestraintList();

    /*! Set the restrained atom indices for group 1 using a text file */
    void SetGroup1AtomFileIndices(const char *groupTag, const char *fileName);

    /*! Set the restrained atom indices for group 1 using a list */
    void SetGroup1AtomListIndices(const char *groupTag, const char *list);

    /*! Set the reference COM position for group 1 */
    void SetGroup1RefPosition(const char *groupTag, const char *vec);

    /*! Set the restrained atom indices for group 2 using a text file */
    void SetGroup2AtomFileIndices(const char *groupTag, const char *fileName);

    /*! Set the restrained atom indices for group 2 using a list */
    void SetGroup2AtomListIndices(const char *groupTag, const char *list);

    /*! Set the restraint force constant */
    void SetForce(const char *groupTag, const BigReal force);

    /*! Set the restraint exponent */
    void SetExponent(const char *groupTag, const int exponent);

    /*! Set the center or equilibrium value of the restraint */
    void SetResCenter(const char *groupTag, const char *vec);

    /*! Set component X of restraint vector */
    void SetResDirectionX(const char *groupTag, const char *status);

    /*! Set component Y of restraint vector */
    void SetResDirectionY(const char *groupTag, const char *status);

    /*! Set component Z of restraint vector */
    void SetResDirectionZ(const char *groupTag, const char *status);

    /*! Set restraint distance mode (magnitude or vector) */
    void SetUseDistMagnitude(const char *groupTag, const char *status);

    /*! Check all necessary parameters in group restraint are set */ 
    void CheckGroupRestraints();

    /*! Print summary of parameters in group restraint */
    void PrintGroupRestraints();

    const std::map<std::string, GroupRestraintParam*> &GetGroupResMap() const {
        return groupRestraints; 
    } 


private:
    /*! Find a restraint group parameter with tag key.
    If no key was found in the map, insert a new object */
    GroupRestraintParam* FindGroupRestraint(const char *tag);

    /**< Map the group restraint tag name to it's parameters */
    std::map<std::string, GroupRestraintParam*> groupRestraints;

};

#endif
