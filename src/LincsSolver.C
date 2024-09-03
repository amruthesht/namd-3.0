/*
 * This class is for solving the long-chain constraints
 * for the Martini model with LINCS algorithm by CG method.
 */
#include <cstdlib>
#include <cerrno>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include "LincsSolver.h"
#include <cmath>
#define BUFFSIZE 1024

struct sparseMatrix
{
    BigReal* aij;
    int* IA; //rows
    int* JA; //colums
}; //store in coordinate format

typedef struct atomData
{
    int globalIndex;
    int patchID;
    int offset;
    BigReal posx, posy, posz;
    BigReal inv_mass;

} atomData;

typedef struct constraintTuple
{
    int i, j;
    BigReal dist;
} constraintTuple;

static void* Malloc(size_t size)
{
    void* ptr = malloc(size);
    if(errno != 0)
    {
        perror("malloc error");
        NAMD_die("malloc error");
    }
    return ptr;
}

static void* Realloc(void* src, size_t size)
{
    if(src == NULL)
        return Malloc(size);
    void* ptr = realloc(src, size);
    if(errno != 0)
    {
        perror("realloc error");
        NAMD_die("realloc error");
    }
    return ptr;
}

static void* Calloc(size_t size, size_t elem)
{
    void* ptr = calloc(size, elem);
    if(errno != 0)
    {
        perror("calloc error");
        NAMD_die("calloc error");
    }
    return ptr;
}

void LincsSolver::multiplyR(BigReal* dest, BigReal* src)
{
    memset(dest, 0, sizeof(BigReal)*num_constraints);
    for(int i = 0; i < 6*num_constraints; i++)
    {
        int col = B->JA[i];
        int row = B->IA[i];
        
        dest[row] += B->aij[i] * src[col];
    }
}

void LincsSolver::multiplyC(BigReal* dest, BigReal* src)
{
    memset(dest, 0, sizeof(BigReal)*num_atoms*3);
    for(int i = 0; i < 6*num_constraints; i++)
    {
        int row = B->JA[i];
        int col = B->IA[i];

        dest[row] += B->aij[i] * src[col];
    }
}

void LincsSolver::matMultiply(BigReal* dest, BigReal* src)
{
    BigReal* tmp = (BigReal*)Malloc(sizeof(BigReal)*3*num_atoms);
    multiplyC(tmp, src);
    for(int i = 0; i < num_atoms; ++i)
    {
        BigReal m = inv_mass[i];
        tmp[3*i+0] *= m;
        tmp[3*i+1] *= m;
        tmp[3*i+2] *= m;
    }
    multiplyR(dest, tmp);
    free(tmp);
}

void LincsSolver::vecAdd(BigReal* dest, BigReal* src1, BigReal* src2, BigReal a)
{
    for(int i = 0; i < num_constraints; ++i)
        dest[i] = src1[i]+a*src2[i];
}

BigReal LincsSolver::vecDot(BigReal* a, BigReal* b)
{
    BigReal x = 0.;
    for(int i = 0; i < num_constraints; ++i)
        x += a[i] * b[i];
    return x;
}

//solve (B^M^-1B^T)x=b
void LincsSolver::conjugateGradient(BigReal* x, BigReal* b)
{
    BigReal* r   = (BigReal*)Malloc(sizeof(BigReal)*num_constraints);
    BigReal* p   = (BigReal*)Malloc(sizeof(BigReal)*num_constraints);
    BigReal* tmp = (BigReal*)Malloc(sizeof(BigReal)*num_constraints);
    
    matMultiply(r, x);
    vecAdd(r,b,r,-1);
    memcpy(p, r, sizeof(BigReal)*num_constraints);
    for(int i = 0; i < maxLoops; ++i)
    {
        matMultiply(tmp, p);
        BigReal r2_old = vecDot(r,r);
        BigReal alpha = r2_old/vecDot(p, tmp);
        vecAdd(x,x,p,alpha);
        vecAdd(r,r,tmp,-alpha);
        BigReal r2 = vecDot(r,r);
        if(r2 < tol*tol)
            break;
        BigReal beta = r2/r2_old;
        vecAdd(p,r,p,beta);
    }
    matMultiply(r,x);
    vecAdd(r,b,r,-1); 
    free(tmp);
    free(p);
    free(r);
}

static int intPairCmp(const void* a1, const void* a2)
{
    constraintTuple* a = (constraintTuple*)a1;
    constraintTuple* b = (constraintTuple*)a2;
    if(a->i < b->i)
        return -1;
    else if(a->i > b->i)
        return 1;
    else
    {
        if(a->j < b->j)
            return -1;
        else
            return 1;
    }
}

static int intCmp(const void* a1, const void* a2)
{
    int* a = (int*)a1;
    int* b = (int*)a2;
    if(*a < *b)
        return -1;
    else if(*a > *b)
        return 1;
    else
        return 0;
}

int LincsSolver::Map(int i)
{
    int* ptr = (int*)bsearch(&i, globalIndex, num_atoms, sizeof(int), &intCmp); 
    if(ptr == NULL)
    {
        fprintf(stderr, "error mapping for %d\n", i);
        NAMD_die("Error in mapping");
    }
    return (int)((size_t)(ptr-globalIndex));
}

static void swapInt(int* v, int a, int b)
{
    int tmp = v[a];
    v[a] = v[b];
    v[b] = tmp;
}

static void swapBigReal(BigReal* v, int a, int b)
{
    BigReal tmp = v[a];
    v[a] = v[b];
    v[b] = tmp;
}

static void swapVector3(BigReal* v, int a, int b)
{
    BigReal tmp;
    for(int i = 0; i < 3; ++i)
    {
        tmp = v[3*a+i];
        v[3*a+i] = v[3*b+i];
        v[3*b+i] = tmp;
    }
}

int LincsSolver::partition(int left, int right) 
{ 
    int pivot = globalIndex[right];
    int i = (left-1);
  
    for (int j = left; j <= right- 1; j++) 
    { 
        // If current element is smaller than or 
        // equal to pivot 
        if (globalIndex[j] <= pivot) 
        { 
            i++;// increment index of smaller element 
            swapInt(globalIndex, i, j);
            swapInt(patchID,  i, j);
            swapInt(offset,   i, j);
            swapBigReal(inv_mass, i, j);
            swapVector3(ref,  i, j);
        } 
    } 
    swapInt(globalIndex, i+1, right);
    swapInt(patchID,  i+1, right);
    swapInt(offset,   i+1, right);
    swapBigReal(inv_mass, i+1, right);
    swapVector3(ref,  i+1, right);
    return i + 1; 
} 
  
void LincsSolver::sortAtom(int left, int right) 
{ 
    if (left < right) 
    { 
        int pi = partition(left, right);  
        // Separately sort elements before 
        // partition and after partition 
        sortAtom(left, pi-1); 
        sortAtom(pi+1, right); 
    } 
}

void LincsSolver::sortConstraints()
{
    qsort(constraintList, num_constraints, sizeof(constraintTuple), &intPairCmp);
}
#ifdef DEBUG
void LincsSolver::showDataRead(void* Data, void* Constraints)
{
    atomData* data = (atomData*)Data;
    constraintTuple* constraints = (constraintTuple*)Constraints;

    fprintf(stdout, "Atom data:\n");
    for(int i = 0; i < num_atoms; ++i)
        fprintf(stdout,"%d %f %f %f\n", data[i].globalIndex, data[i].posx, data[i].posy, data[i].posz);

    fprintf(stdout, "\n");
    for(int i = 0; i < num_atoms; ++i)
        fprintf(stdout,"%d %f %f %f\n", globalIndex[i], ref[3*i+0], ref[3*i+1], ref[3*i+2]);

    fprintf(stdout, "\nconstraints:\n");
    for(int i = 0; i < num_constraints; ++i)
        fprintf(stdout, "%d %d %f\n", constraints[i].i, constraints[i].j, constraints[i].dist);
    
    fprintf(stdout, "\n");
    for(int i = 0; i < num_constraints; ++i)
        fprintf(stdout, "%d %d %f\n", constraintList[i].i, constraintList[i].j, constraintList[i].dist);

}

void LincsSolver::checkConvergence()
{
    for(int idx = 0; idx < num_constraints; ++idx)
    {
        int i = constraintList[idx].i;
        int j = constraintList[idx].j;
        BigReal dist = constraintList[idx].dist;
        int col_i = Map(i);
        int col_j = Map(j);
        BigReal x1, x2, y1, y2, z1, z2;
        x1 = ref[3*col_i+0];
        x2 = ref[3*col_j+0];
        y1 = ref[3*col_i+1];
        y2 = ref[3*col_j+1];
        z1 = ref[3*col_i+2];
        z2 = ref[3*col_j+2];
        BigReal r = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));
    }
}
#endif
void LincsSolver::copy(void* Data, void* Constraints)
{
     int size = BUFFSIZE;
     int count = 0;

     globalIndex = (int*)Malloc(sizeof(int)*size);
     patchID     = (int*)Malloc(sizeof(int)*size);
     offset      = (int*)Malloc(sizeof(int)*size);
     inv_mass    = (BigReal*)Malloc(sizeof(BigReal)*size);
     ref         = (BigReal*)Malloc(sizeof(BigReal)*size*3);
     //let's cast it
     atomData* data = (atomData*)Data;
     while(data->globalIndex > -1)
     {
         globalIndex[count]     = data->globalIndex;
         patchID    [count]     = data->patchID;
         offset     [count]     = data->offset;
         inv_mass   [count]     = data->inv_mass;
         ref        [3*count+0] = data->posx;
         ref        [3*count+1] = data->posy;
         ref        [3*count+2] = data->posz;
         ++count;
         ++data;
         if(count >= size)
         {
              size <<= 1;
              //alloc more memory
              globalIndex = (int*)Realloc(globalIndex, sizeof(int)*size);
              patchID     = (int*)Realloc(patchID,     sizeof(int)*size);
              offset      = (int*)Realloc(offset,      sizeof(int)*size);
              inv_mass    = (BigReal*)Realloc(inv_mass,    sizeof(BigReal)*size);
              ref         = (BigReal*)Realloc(ref,         sizeof(BigReal)*size*3);
         }
     }
     num_atoms   = count;
     globalIndex = (int*)Realloc(globalIndex, sizeof(int)*num_atoms);
     patchID     = (int*)Realloc(patchID,     sizeof(int)*num_atoms);
     offset      = (int*)Realloc(offset,      sizeof(int)*num_atoms);
     inv_mass    = (BigReal*)Realloc(inv_mass,    sizeof(BigReal)*num_atoms);
     ref         = (BigReal*)Realloc(ref,         sizeof(BigReal)*num_atoms*3);
 
     size = BUFFSIZE;
     constraintList = (constraintTuple*)Malloc(sizeof(constraintTuple)*size);
     count = 0;
     constraintTuple* constraints = (constraintTuple*)Constraints;
     while(constraints->i > -1)
     {
 
         int i = constraints->i;
         int j = constraints->j;
         if(i > j)
         {
             int k = i;
             i = j;
             j = k;
         }       
         constraintList[count].i = i;
         constraintList[count].j = j;
         constraintList[count++].dist = (constraints++)->dist;
         
         if(count >= size)
         {
              size <<= 1;
              constraintList = (constraintTuple*)Realloc(constraintList,size*sizeof(constraintTuple));
         }
     }
     num_constraints = count;
     constraintList = (constraintTuple*)Realloc(constraintList, sizeof(constraintTuple)*num_constraints);
}


void LincsSolver::buildBMatrix()
{
    B = (sparseMatrix*)Malloc(sizeof(sparseMatrix));
    B->aij = (BigReal*)Malloc(sizeof(BigReal)*6*num_constraints);
    B->IA  = (int*)Malloc(sizeof(int)*6*num_constraints);
    B->JA  = (int*)Malloc(sizeof(int)*6*num_constraints);
    for(int idx = 0; idx < num_constraints; ++idx)
    {
        int i = constraintList[idx].i;
        int j = constraintList[idx].j;
        //record all idexes, this is not memory optimized
        int col_i = Map(i);
        int col_j = Map(j);
        for(int k = 0; k < 6; ++k)
            B->IA[6*idx+k] = idx;
        for(int k = 0; k < 3; ++k)
        {
            B->JA[6*idx+k]   = 3*col_i+k;
            B->JA[6*idx+3+k] = 3*col_j+k; 
        }
        //build matrix element
        BigReal x1,x2,y1,y2,z1,z2,d2;
        x1 = ref[3*col_i+0];
        y1 = ref[3*col_i+1];
        z1 = ref[3*col_i+2];

        x2 = ref[3*col_j+0];
        y2 = ref[3*col_j+1];
        z2 = ref[3*col_j+2]; 

        d2 = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2);
        d2 = sqrt(d2);
        B->aij[6*idx+0] = (x1-x2)/d2;
        B->aij[6*idx+1] = (y1-y2)/d2;
        B->aij[6*idx+2] = (z1-z2)/d2;
        B->aij[6*idx+3] = -B->aij[6*idx+0];
        B->aij[6*idx+4] = -B->aij[6*idx+1];
        B->aij[6*idx+5] = -B->aij[6*idx+2];
    }    
}

void LincsSolver::setUpSolver(atomData* data, constraintTuple* constraints)
{
    destroy();
    copy(data, constraints);
    sortAtom(0, num_atoms-1);
    sortConstraints();   
    buildBMatrix();
    #ifdef DEBUG
    showDataRead(data, constraints);
    checkConvergence();
    #endif
}

void LincsSolver::solve()
{
    BigReal* b = (BigReal*)Malloc(sizeof(BigReal)*num_constraints);
    BigReal* x = (BigReal*)Malloc(sizeof(BigReal)*num_constraints);
    multiplyR(b, ref);
    for(int i = 0; i < num_constraints; ++i)
        b[i] -= constraintList[i].dist; 
    memset(x, 0 , sizeof(BigReal)*num_constraints);
    conjugateGradient(x, b);
    BigReal* tmp = (BigReal*)Malloc(sizeof(BigReal)*num_atoms*3);
    multiplyC(tmp, x);
    for(int i = 0; i < num_atoms; ++i)
    {
        BigReal m = inv_mass[i];
        ref[3*i+0] -= tmp[3*i+0]*m;
        ref[3*i+1] -= tmp[3*i+1]*m;
        ref[3*i+2] -= tmp[3*i+2]*m;
    }
    multiplyR(b, ref);
    for(int idx = 0; idx < num_constraints; ++idx)
    {
        int i = constraintList[idx].i;
        int j = constraintList[idx].j;
        BigReal d = constraintList[idx].dist;
        int col_i = Map(i);
        int col_j = Map(j);
        BigReal x1, x2, y1, y2, z1, z2;
        x1 = ref[3*col_i+0];
        x2 = ref[3*col_j+0];
        y1 = ref[3*col_i+1];
        y2 = ref[3*col_j+1];
        z1 = ref[3*col_i+2];
        z2 = ref[3*col_j+2];
        BigReal l = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2);

        b[idx] -= sqrt(2.0*d*d-l);
    }
    memset(x, 0 , sizeof(BigReal)*num_constraints);
    conjugateGradient(x, b);
    multiplyC(tmp, x);
    for(int i = 0; i < num_atoms; ++i)
    {
        BigReal m = inv_mass[i];
        ref[3*i+0] -= tmp[3*i+0]*m;
        ref[3*i+1] -= tmp[3*i+1]*m;
        ref[3*i+2] -= tmp[3*i+2]*m;
    }
    #ifdef DEBUG
    checkConvergence();
    #endif
    free(b);
    free(x);
    free(tmp); 
}
void LincsSolver::destroy()
{
    if(ref != NULL)
        free(ref);
    ref = NULL;
    if(inv_mass != NULL)
        free(inv_mass);
    inv_mass = NULL;
    if(offset != NULL)
        free(offset);
    offset = NULL;
    if(patchID != NULL)
        free(patchID);
    patchID = NULL;
    if(globalIndex != NULL)
        free(globalIndex);
    globalIndex = NULL;
    if(constraintList != NULL)
        free(constraintList);
    constraintList = NULL;
    if(B != NULL)
    {
        free(B->aij);
        B->aij = NULL;
        free(B->IA);
        B->IA = NULL;
        free(B->JA);
        B->JA = NULL;
        free(B);
        B = NULL;
    }
}

