/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "converse.h"
#include "Priorities.h"
#include "PatchTypes.h"
#include "PatchMgr.h"
#include "SequencerCUDA.h"

/// Simplify Sequencer function SOA arguments
/// by accessing patch->patchDataSOA arrays
/// directly within integrator routines.
///
/// Earlier tests with passing PatchDataSOA by reference
/// made code about 1% slower for STMV benchmark.
///
/// Current version passes no PatchDataSOA arguments
/// and seems to be slightly faster than explicitly passing
/// all of the PatchDataSOA arrays, producing some of the
/// fastest timings yet, although probably not quite 1% improvement.
/// All of the array pointer assignments within the routines
/// use the __restrict directive.
#define SOA_SIMPLIFY_PARAMS

class HomePatch;
class SimParameters;
class SubmitReduction;
class CollectionMgr;
class ControllerBroadcasts;
class LdbCoordinator;
class Random;
class SequencerCUDA;
#ifdef SOA_SIMPLIFY_PARAMS
struct PatchDataSOA;
#endif

class Sequencer
{
    friend class HomePatch;
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
    friend class SequencerCUDA;
#endif
public:
    Sequencer(HomePatch *p);
    virtual ~Sequencer(void);
    void run(void);             // spawn thread, etc.
    void awaken(void) {
      CthAwakenPrio(thread, CK_QUEUEING_IFIFO, PRIORITY_SIZE, &priority);
    }
    void suspend(void);

protected:
    virtual void algorithm(void);	// subclasses redefine this method

#ifdef SEQUENCER_SOA
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
    void integrate_CUDA_SOA(int scriptTask);
    void initialize_integrate_CUDA_SOA(int scriptTask, int step, BigReal timestep, 
       int numberOfSteps, int nbondstep, int slowstep, int maxForceUsed);
    void suspendULTs();
    void wakeULTs();
    void runComputeObjectsCUDA(int doMigration, int doGlobal, int pairlists, int nstep, int startup);
    void constructDevicePatchMap();
    void updateDevicePatchMap(int startup);
    void printDevicePatchMap();
    void clearDevicePatchMap();
    void updateDeviceData(const int startup, const int maxForceUsed, const int doGlobal);
    void doMigrationGPU(const int startup, const int doGlobal, const int updatePatchMap);
        
    // apply MC pressure control
    void monteCarloPressureControl(
        const int step,
        const int doMigration,
        const int doEnergy,
        const int doVirial,
        const int maxForceNumber);
#endif
    void integrate_SOA(int); // Verlet integrator using SOA data structures
    void rattle1_SOA(BigReal,int);
    void addForceToMomentum_SOA(
        const double scaling,
        double       dt_normal,               // timestep Results::normal = 0
        double       dt_nbond,                // timestep Results::nbond  = 1
        double       dt_slow,                 // timestep Results::slow   = 2
#ifndef SOA_SIMPLIFY_PARAMS
        const double * __restrict recipMass,
        const double * __restrict f_normal_x, // force    Results::normal = 0
        const double * __restrict f_normal_y,
        const double * __restrict f_normal_z,
        const double * __restrict f_nbond_x,  // force    Results::nbond  = 1
        const double * __restrict f_nbond_y,
        const double * __restrict f_nbond_z,
        const double * __restrict f_slow_x,   // force    Results::slow   = 2
        const double * __restrict f_slow_y,
        const double * __restrict f_slow_z,
        double       * __restrict vel_x,
        double       * __restrict vel_y,
        double       * __restrict vel_z,
        int numAtoms,
#endif
        int maxForceNumber
    );
    void addVelocityToPosition_SOA(
        const double dt   ///< scaled timestep
#ifndef SOA_SIMPLIFY_PARAMS
        ,
        const double * __restrict vel_x,
        const double * __restrict vel_y,
        const double * __restrict vel_z,
        double *       __restrict pos_x,
        double *       __restrict pos_y,
        double *       __restrict pos_z,
        int numAtoms      ///< number of atoms
#endif
    );
    void submitHalfstep_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        const int    * __restrict hydrogenGroupSize,
        const float  * __restrict mass,
        const double * __restrict vel_x,
        const double * __restrict vel_y,
        const double * __restrict vel_z,
        int numAtoms
#endif
    );
    void submitReductions_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        const int    * __restrict hydrogenGroupSize,
        const float  * __restrict mass,
        const double * __restrict pos_x,
        const double * __restrict pos_y,
        const double * __restrict pos_z,
        const double * __restrict vel_x,
        const double * __restrict vel_y,
        const double * __restrict vel_z,
        const double * __restrict f_normal_x,
        const double * __restrict f_normal_y,
        const double * __restrict f_normal_z,
        const double * __restrict f_nbond_x,
        const double * __restrict f_nbond_y,
        const double * __restrict f_nbond_z,
        const double * __restrict f_slow_x,
        const double * __restrict f_slow_y,
        const double * __restrict f_slow_z,
        int numAtoms
#endif
    );
    void submitCollections_SOA(int step, int zeroVel = 0);
    void maximumMove_SOA(
        const double dt,  ///< scaled timestep
        const double maxvel2  ///< square of bound on velocity
#ifndef SOA_SIMPLIFY_PARAMS
        ,
        const double * __restrict vel_x,
        const double * __restrict vel_y,
        const double * __restrict vel_z,
        int numAtoms      ///< number of atoms
#endif
    );
    void langevinVelocitiesBBK1_SOA(
        BigReal timestep 
#ifndef SOA_SIMPLIFY_PARAMS
        ,
        const float * __restrict langevinParam,
        double      * __restrict vel_x,
        double      * __restrict vel_y,
        double      * __restrict vel_z,
        int numAtoms
#endif
    );
    void langevinVelocitiesBBK2_SOA(
        BigReal timestep 
#ifndef SOA_SIMPLIFY_PARAMS
        ,
        const float * __restrict langevinParam,
        const float * __restrict langScalVelBBK2,
        const float * __restrict langScalRandBBK2,
        float       * __restrict gaussrand_x,
        float       * __restrict gaussrand_y,
        float       * __restrict gaussrand_z,
        double      * __restrict vel_x,
        double      * __restrict vel_y,
        double      * __restrict vel_z,
        int numAtoms
#endif
    );

    void berendsenPressure_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        const int    * __restrict hydrogenGroupSize,
        const float  * __restrict mass,
        double       * __restrict pos_x,
        double       * __restrict pos_y,
        double       * __restrict pos_z,
        int numAtoms,
#endif
        int step);

    void langevinPiston_SOA(
#ifndef SOA_SIMPLIFY_PARAMS
        const int    * __restrict hydrogenGroupSize,
        const float  * __restrict mass,
        double       * __restrict pos_x,
        double       * __restrict pos_y,
        double       * __restrict pos_z,
        double       * __restrict vel_x,
        double       * __restrict vel_y,
        double       * __restrict vel_z,
        int numAtoms,
#endif
        int step
    );
    void stochRescaleVelocities_SOA(int step);
    void runComputeObjects_SOA(int migration, int pairlists, int step);
#endif

    void integrate(int); // Verlet integrator
    void minimize(); // CG minimizer
      SubmitReduction *min_reduction;

    void runComputeObjects(int migration = 1, int pairlists = 0, int pressureStep = 0);
    int pairlistsAreValid;
    int pairlistsAge;
    int pairlistsAgeLimit;  // constant based on fixed simParams values

    void calcFixVirial(Tensor& fixVirialNormal, Tensor& fixVirialNbond, Tensor& fixVirialSlow,
      Vector& fixForceNormal, Vector& fixForceNbond, Vector& fixForceSlow);

    void submitReductions(int);
    void submitHalfstep(int);
    void submitMinimizeReductions(int, BigReal fmax2);
    void submitCollections(int step, int zeroVel = 0);

    void submitMomentum(int step);
    void correctMomentum(int step, BigReal drifttime);

    void saveForce(const int ftag = Results::normal);
    void addForceToMomentum(BigReal, const int ftag = Results::normal, const int useSaved = 0);
    void addForceToMomentum3(const BigReal timestep1, const int ftag1, const int useSaved1,
        const BigReal timestep2, const int ftag2, const int useSaved2,
        const BigReal timestep3, const int ftag3, const int useSaved3);
    void addVelocityToPosition(BigReal);
    
    void addRotDragToPosition(BigReal);
    void addMovDragToPosition(BigReal);

    void minimizeMoveDownhill(BigReal fmax2);
    void newMinimizeDirection(BigReal);
    void newMinimizePosition(BigReal);
    void quenchVelocities();

    void hardWallDrude(BigReal,int);

    void rattle1(BigReal,int);
    // void rattle2(BigReal,int);

    void maximumMove(BigReal);
    void minimizationQuenchVelocity(void);

    void reloadCharges();
    void rescaleSoluteCharges(BigReal);

    BigReal adaptTempT;         // adaptive tempering temperature
    void adaptTempUpdate(int); // adaptive tempering temperature update

    void rescaleVelocities(int);
    void rescaleaccelMD(int, int, int); // for accelMD
    int rescaleVelocities_numTemps;
    void reassignVelocities(BigReal,int);
    void reinitVelocities(void);
    void rescaleVelocitiesByFactor(BigReal);
    void tcoupleVelocities(BigReal,int);

    /**
     * When doing stochastic velocity rescaling, every stochRescaleFreq
     * steps we receive the globally computed rescaling coefficient and 
     * apply it to the velocities of all the atoms in our patch.
     */
    void stochRescaleVelocities(int);

    int stochRescale_count;
    /**< Count time steps until next stochastic velocity rescaling. */

    void berendsenPressure(int);
      int berendsenPressure_count;
      int checkpoint_berendsenPressure_count;
    void langevinPiston(int);
      int slowFreq;
    void newtonianVelocities(BigReal, const BigReal, const BigReal, 
                             const BigReal, const int, const int, const int);
    void langevinVelocities(BigReal);
    void langevinVelocitiesBBK1(BigReal);
    void langevinVelocitiesBBK2(BigReal);
    // Multigrator
    void scalePositionsVelocities(const Tensor& posScale, const Tensor& velScale);
    void multigratorPressure(int step, int callNumber);
    void scaleVelocities(const BigReal velScale);
    BigReal calcKineticEnergy();
    void multigratorTemperature(int step, int callNumber);
    SubmitReduction *multigratorReduction;
    int doKineticEnergy;
    int doMomenta;
    // End of Multigrator
    
    void cycleBarrier(int,int);
	void traceBarrier(int);
#ifdef MEASURE_NAMD_WITH_PAPI
	void papiMeasureBarrier(int);
#endif
    void terminate(void);

    Random *random;
    SimParameters *const simParams;	// for convenience
    HomePatch *const patch;		// access methods in patch
    SubmitReduction *reduction;
    SubmitReduction *pressureProfileReduction;

    CollectionMgr *const collection;
    ControllerBroadcasts * broadcast;

    int ldbSteps;
    bool masterThread;
    void rebalanceLoad(int timestep);


private:
    CthThread thread;
    unsigned int priority;
    static void threadRun(Sequencer*);

    LdbCoordinator *ldbCoordinator;
#if (defined(NAMD_CUDA) || defined(NAMD_HIP)) && defined(SEQUENCER_SOA)
    SequencerCUDA *CUDASequencer;
    PatchData *patchData;
#endif
};

#endif
