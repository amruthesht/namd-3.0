#ifndef LINCSSOLVER_H_
#define LINCSSOLVER_H_
#include "common.h"
typedef struct sparseMatrix sparseMatrix;
typedef struct constraintTuple constraintTuple;
typedef struct atomData atomData;
/**
 * This class is for solving the long-chain constraints
 * for the Martini model with LINCS algorithm by the 
 * conjugate gradient method.
 */

class LincsSolver
{
  private:
    sparseMatrix* B;                 //!< a sparse matrix stored in coordinate format.
    constraintTuple* constraintList; //!< a list of constraint tuples.
    int* globalIndex;      //!< global index 
    int* patchID;          //!< patch ID
    int* offset;           //!< local offset
    BigReal* inv_mass;     //!< inverse of masses
    BigReal* ref;          //!< the positions of particles at the previous time step

    int num_atoms;         //!< number of atoms in the chain
    int num_constraints;   //!< number of constraints in the chain
    int maxLoops;          //!< maximum of loops for the CG solver
    BigReal tol;           //!< tolerance for the CG solver
 
    int  Map(int);      //!< use binary search to map from global ID to local ID
    void multiplyR(BigReal* dest, BigReal* src);
    void multiplyC(BigReal* dest, BigReal* src);
    void matMultiply(BigReal* dest, BigReal* src);
    void vecAdd(BigReal* dest, BigReal* src1, BigReal* src2, BigReal a);
    BigReal vecDot(BigReal*, BigReal*);
    void conjugateGradient(BigReal*, BigReal*);
    int  partition(int left, int right);
    void sortAtom(int left, int right);
    void sortConstraints();
    void copy(void*, void*);
    void destroy();
    void buildBMatrix();
    #ifdef DEBUG
    void showDataRead(void*,void*);
    void checkConvergence();
    #endif
  public:
    void setUpSolver(atomData*, constraintTuple*); //!< interface to set up the solver with two given arrays of data.
    void solve(); //!< interface to run the solver with CG.
    /**
    * The constructor of the class.
    */ 
    LincsSolver(int loop=10000, BigReal r=1e-12) : B(NULL), constraintList(NULL), globalIndex(NULL),
                    patchID(NULL), offset(NULL), inv_mass(NULL), ref(NULL), num_atoms(0), num_constraints(0), maxLoops(loop), tol(r) {}
    ~LincsSolver()
    {
        destroy();
    }
};
#endif
