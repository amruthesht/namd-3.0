/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "converse.h"
#include "Node.h"
#include "common.h"
#include "fstream_namd.h"
#include "ReductionMgr.h"
#include <string>
#include <map>
#include <vector>

/// Calculate running average and standard deviation.
/// Supplement timing output statements with average performance.
/// Implements stable and easily verified recurrence relations from
/// https://en.wikipedia.org/wiki/Standard_deviation
class RunningAverage {
  double a;  ///< for running average
  double q;  ///< for running variance/stddev
  unsigned int n;  ///< count number of samples
public:
  RunningAverage() : a(0), q(0), n(0) { }
  void add_sample(double x) {
    double a_old = a;
    n++;
    a = a_old + (x - a_old) / n;
    q = q + (x - a_old) * (x - a);
  }
  double average() const { return a; }
  /// For calculating sample variance,
  /// divide by n-1.
  double variance() const { return (n > 1 ? q/(n-1) : 0); }
  double standard_deviation() const { return sqrt(variance()); }
};

/// Maintain moving window average for a collection of samples.
/// Samples contained in a circular queue (newest replacing oldest)
/// implemented using a fixed length vector.
/// Average is updated for each new sample with O(1) work.
class MovingAverage {
  std::vector<double> sample;  // vector of sample values to average
  double a;  // contains the average
  int n;     // count number of samples
  int i;     // index to next available slot in sample vector
  int m;     // max window size
public:
  MovingAverage(int mws = 20) : a(0), n(0), i(0), m(mws) {
    sample.resize(m);
  }
  void reset(int mws = 20) {
    a=0, n=0, i=0;
    if (m != mws) {
      m = mws;
      sample.resize(m);
    }
  }
  void addSample(double x) {
    if (n < m) {  // use running average formula
      sample[i] = x;
      // XXX I think that the index increment below is faster than
      // i = (i+1) % m due to avoiding an integer division
      i = (i < m-1 ? i+1 : 0);  // increment i to next available slot
      double a_old = a;
      n++;
      a = a_old + (x - a_old) / n;
    }
    else {  // use update average formula
      double x_old = sample[i];
      sample[i] = x;
      i = (i < m-1 ? i+1 : 0);  // increment i to next available slot
      a = a + (x - x_old) / n;
    }
  }
  double average() const { return a; }
};

class ControllerBroadcasts;
class NamdState;
class SimParameters;
class RequireReduction;
class SubmitReduction;

#ifdef MEM_OPT_VERSION
class CollectionMasterHandler;
#else
class CollectionMaster;
#endif

class Random;
class PressureProfileReduction;

struct ControllerState {
    Tensor langevinPiston_strainRate;
    Tensor berendsenPressure_avg;
    // x element: x-axis, constant ratio, or isotropic fluctuations
    // y element: y-axis fluctuations
    // z element: z-axis or constant area fluctuations
    Vector monteCarloMaxVolume; 
    int berendsenPressure_count;
    BigReal smooth2_avg;
};

class Controller : protected ControllerState
{
public:
    Controller(NamdState *s);
    virtual ~Controller(void);
    void run(void);             // spawn thread, etc.
    void awaken(void) { CthAwaken(thread); };
    void resumeAfterTraceBarrier(int);
#ifdef MEASURE_NAMD_WITH_PAPI
	void resumeAfterPapiMeasureBarrier(int step);
#endif
    BigReal accelMDdV; // this is used for on-the-fly reweighting in colvars
  private:
    void computeTIderivative(void); // for lambda-dynamics
    BigReal TIderivative;
  public:
    BigReal getTIderivative(void) const { return TIderivative; }

    void resetMovingAverage();
#ifdef NODEGROUP_FORCE_REGISTER
    void printStep(int step, NodeReduction* nr);
    void piston1(int step) { this->langevinPiston1(step); }
    void berendsenPressureController(int step) { this->berendsenPressure(step); }
    void mcPressure_prepare(int step) { this->monteCarloPressure_prepare(step); }
    void mcPressure_accept(int step) { this->monteCarloPressure_accept(step); }
    Tensor getPositionRescaleFactor(int step); 
    int getMCAcceptance(int step);
#endif

protected:
    friend class ScriptTcl;
    friend class Node;
    friend class CheckpointMsg;
    virtual void algorithm(void);	// subclasses redefine this method

    void integrate(int); // Verlet integrator
    void minimize(); // CG minimizer
      RequireReduction *min_reduction;

    void receivePressure(int step, int minimize = 0);
    void calcPressure(int step, int minimize,
      const Tensor& virial_normal_in, const Tensor& virial_nbond_in, const Tensor& virial_slow_in,
      const Tensor& intVirial_normal, const Tensor& intVirial_nbond, const Tensor& intVirial_slow,
      const Vector& extForce_normal, const Vector& extForce_nbond, const Vector& extForce_slow);

      Tensor pressure_normal;
      Tensor pressure_nbond;
      Tensor pressure_slow;
      Tensor pressure_amd;
      Tensor virial_amd;
      Tensor groupPressure_normal;
      Tensor groupPressure_nbond;
      Tensor groupPressure_slow;
      Tensor controlPressure_normal;
      Tensor controlPressure_nbond;
      Tensor controlPressure_slow;
      int nbondFreq;
      int slowFreq;
      BigReal temp_avg;
      BigReal pressure_avg;
      BigReal groupPressure_avg;
      int avg_count;
      Tensor pressure_tavg;
      Tensor groupPressure_tavg;
      int tavg_count;
    void compareChecksums(int,int=0);
      int computeChecksum;
      int marginViolations;
      int pairlistWarnings;
    // Returns the total potential energy  
    BigReal getTotalPotentialEnergy(int step);
    void printTiming(int);
    void printMinimizeEnergies(int);
      BigReal min_energy;
      BigReal min_f_dot_f;
      BigReal min_f_dot_v;
      BigReal min_v_dot_v;
      int min_huge_count;
    void printDynamicsEnergies(int);
    void printEnergies(int step, int minimize);
      int64_t numDegFreedom;
      int stepInFullRun;
      BigReal totalEnergy;
      BigReal electEnergy;
      BigReal electEnergySlow;
      BigReal ljEnergy;
      BigReal groLJEnergy;
      BigReal groGaussEnergy;
      BigReal goNativeEnergy;
      BigReal goNonnativeEnergy;
      BigReal goTotalEnergy;
//fepb
      BigReal bondedEnergyDiff_f;
      BigReal electEnergy_f;
      BigReal electEnergySlow_f;
      BigReal ljEnergy_f;
      BigReal ljEnergy_f_left;  // used by WCA repulsive, [s1,s2]
      BigReal exp_dE_ByRT;
      BigReal dE;
      BigReal net_dE;
      BigReal dG;
      int FepNo;
      void printFepMessage(int);
      BigReal fepSum;
//fepe
      BigReal bondedEnergy_ti_1;
      BigReal bondedEnergy_ti_2;
      BigReal electEnergy_ti_1;
      BigReal electEnergySlow_ti_1;
      BigReal ljEnergy_ti_1;
      BigReal electEnergy_ti_2;
      BigReal electEnergySlow_ti_2;
      BigReal ljEnergy_ti_2;
      BigReal net_dEdl_bond_1;
      BigReal net_dEdl_bond_2;
      BigReal net_dEdl_elec_1;
      BigReal net_dEdl_elec_2;
      BigReal net_dEdl_lj_1;
      BigReal net_dEdl_lj_2;
      BigReal cumAlchWork;
      BigReal electEnergyPME_ti_1;
      BigReal electEnergyPME_ti_2;
      int TiNo;
      BigReal recent_dEdl_bond_1;
      BigReal recent_dEdl_bond_2;
      BigReal recent_dEdl_elec_1;
      BigReal recent_dEdl_elec_2;
      BigReal recent_dEdl_lj_1;
      BigReal recent_dEdl_lj_2;
      BigReal recent_alchWork;
      BigReal alchWork;
      int recent_TiNo;
      void printTiMessage(int);

      BigReal drudeBondTemp; // temperature of Drude bonds
      BigReal drudeBondTempAvg;

      BigReal kineticEnergy;
      BigReal kineticEnergyHalfstep;
      BigReal kineticEnergyCentered;
      BigReal temperature;
      BigReal heat;
      /**< heat exchanged with the thermostat since firstTimestep */
      BigReal totalEnergy0; /**< totalEnergy at firstTimestep */
      // BigReal smooth2_avg;
      BigReal smooth2_avg2;  // avoid internal compiler error
      Tensor pressure;
      Tensor groupPressure;
      int controlNumDegFreedom;
      Tensor controlPressure;
    void enqueueCollections(int);
    void correctMomentum(int step);
    void rescaleVelocities(int);
      BigReal rescaleVelocities_sumTemps;
      int rescaleVelocities_numTemps;
    void reassignVelocities(int);
    void tcoupleVelocities(int);

public:
    /**
     * The Controller routine for stochastic velocity rescaling uses
     * the most recent temperature reduction to calculate the velocity
     * rescaling coefficient that is then broadcast to all patches.
     */
    void stochRescaleVelocities(int);

    /**
     * Calculate new coefficient for stochastic velocity rescaling
     * and update heat.
     */
    double stochRescaleCoefficient();

    int stochRescale_count;
    /**< Count time steps until next stochastic velocity rescaling. */ 

    BigReal stochRescaleTimefactor;
    /**< The timefactor for stochastic velocity rescaling depends on
     * fixed configuration parameters, so can be precomputed. */

protected:
    void berendsenPressure(int);
      // Tensor berendsenPressure_avg;
      // int berendsenPressure_count;
    void langevinPiston1(int);
    void langevinPiston2(int);
      Tensor langevinPiston_origStrainRate;
      Tensor strainRate_old;  // for langevinPistonBarrier no
      Tensor positionRescaleFactor;  // for langevinPistonBarrier no
#ifdef NODEGROUP_FORCE_REGISTER
      std::map<int, Tensor> publishedRescaleFactors;
      // map the acceptance status of MC volume change trial to step number
      std::map<int, int> publishedMCAcceptance; 
#endif
    /**
     * Perform a random walk in volume and calculate the rescale 
     * factor for lattice and atom coordinates.
    */ 
    void monteCarloPressure_prepare(int);
    /**
     * Calculate the MC acceptance criteria for volume change and
     * determin if this volume change is accepted or not.
    */ 
    void monteCarloPressure_accept(int);
    enum mc_axis_pick {
      MC_X = 0, // for x-axis, constant ratio, or isotropic fluctuation
      MC_Y, // for y-axis
      MC_Z, // for z-axis or constant area fluctuations
      MC_AXIS_TOTAL
    };
    // keep track of attempted and accepted volume fluctuation in MC barostat
    int mc_trial[MC_AXIS_TOTAL], mc_accept[MC_AXIS_TOTAL];
    int mc_totalTry, mc_totalAccept;
    int mc_picked_axis;
    BigReal mc_totalEnergyOld; // total energy of old micro state
    Lattice mc_oldLattice; // lattice of old micro state

    void multigratorPressure(int step, int callNumber);
    BigReal multigratorXi;
    BigReal multigratorXiT;
    Tensor momentumSqrSum;
    void multigratorTemperature(int step, int callNumber);
    std::vector<BigReal> multigratorNu;
    std::vector<BigReal> multigratorNuT;
    std::vector<BigReal> multigratorOmega;
    std::vector<BigReal> multigratorZeta;
    RequireReduction *multigratorReduction;
    BigReal multigatorCalcEnthalpy(BigReal potentialEnergy, int step, int minimize);

    int ldbSteps;
    void rebalanceLoad(int);
      int fflush_count;
    void cycleBarrier(int,int);	
	
	void traceBarrier(int, int);

#ifdef MEASURE_NAMD_WITH_PAPI
	void papiMeasureBarrier(int, int);
#endif

    void suspend(void) { CthSuspend(); };
    void terminate(void);
    Random *random;
    SimParameters *const simParams;	// for convenience
    NamdState *const state;		// access data in state
    RequireReduction *reduction;
    RequireReduction *amd_reduction;
    SubmitReduction *submit_reduction;
#if defined(NAMD_CUDA) || defined(NAMD_HIP)
    NodeReduction *nodeReduction;
#endif

    // data for pressure profile reductions and output
    PressureProfileReduction *ppbonded;
    PressureProfileReduction *ppnonbonded;
    PressureProfileReduction *ppint;
    int pressureProfileSlabs;
    int pressureProfileCount;
    BigReal *pressureProfileAverage;

    CollectionMaster *const collection;
    
    ControllerBroadcasts * broadcast;
    ofstream_namd xstFile;
    void outputExtendedSystem(int step);
    void writeExtendedSystemLabels(ofstream_namd &file);
    void writeExtendedSystemData(int step, ofstream_namd &file);

//fepb
    ofstream_namd fepFile;
    void outputFepEnergy(int step);
    void writeFepEnergyData(int step, ofstream_namd &file);
//fepe
    ofstream_namd tiFile;
    void outputTiEnergy(int step);
    BigReal computeAlchWork(const int step);
    void writeTiEnergyData(int step, ofstream_namd &file);

    // for checkpoint/revert
    int checkpoint_stored;
    Lattice checkpoint_lattice;
    ControllerState checkpoint_state;

    struct checkpoint {
      Lattice lattice;
      ControllerState state;
    };
    std::map<std::string,checkpoint*> checkpoints;
    int checkpoint_task;
    void recvCheckpointReq(const char *key, int task, checkpoint &cp);
    void recvCheckpointAck(checkpoint &cp);

    Lattice origLattice;

//for accelMD
   inline void calc_accelMDG_mean_std
   (BigReal testV, int step_n, 
    BigReal *Vmax, BigReal *Vmin, BigReal *Vavg, BigReal *M2, BigReal *sigmaV);

   inline void calc_accelMDG_E_k
   (int iE, int V_n, BigReal sigma0, BigReal Vmax, BigReal Vmin, BigReal Vavg, BigReal sigmaV, 
    BigReal* k0, BigReal* k, BigReal* E, int* iEused, char *warn);

   inline void calc_accelMDG_force_factor
   (BigReal k, BigReal E, BigReal testV, Tensor vir_orig,
    BigReal *dV, BigReal *factor, Tensor *vir);

   void write_accelMDG_rest_file
       (int step_n, char type, int V_n, BigReal Vmax, BigReal Vmin, BigReal Vavg, BigReal sigmaV, BigReal M2,
	BigReal E, BigReal k, bool write_topic, bool lasttime);

   void rescaleaccelMD (int step, int minimize = 0);
   BigReal accelMDdVAverage;

//JS for adaptive temperature sampling
   void adaptTempInit(int step);
   void adaptTempUpdate(int step, int minimize = 0);
   void adaptTempWriteRestart(int step);
   BigReal *adaptTempPotEnergyAveNum;
   BigReal *adaptTempPotEnergyAveDen;
   BigReal *adaptTempPotEnergyVarNum;
   BigReal *adaptTempPotEnergyAve;
   BigReal *adaptTempPotEnergyVar;
   int     *adaptTempPotEnergySamples;
   BigReal *adaptTempBetaN;
   BigReal adaptTempT;
   BigReal adaptTempDTave;
   BigReal adaptTempDTavenum;
   BigReal adaptTempBetaMin;
   BigReal adaptTempBetaMax;
   int     adaptTempBin;
   int     adaptTempBins;
   BigReal adaptTempDBeta;
   BigReal adaptTempCg;
   BigReal adaptTempDt;
   Bool    adaptTempAutoDt;
   BigReal adaptTempDtMin;
   BigReal adaptTempDtMax;
   ofstream_namd adaptTempRestartFile;

// Average performance stats as supplement to output timing statements.
   RunningAverage perfstats;
  
// Moving averages of reductions for GPU-resident mode
   MovingAverage totalEnergyAverage;
   MovingAverage temperatureAverage;
   MovingAverage pressureAverage;
   MovingAverage groupPressureAverage;

// Moving averages for pressure tensor output
   MovingAverage pressureAverage_xx;
   MovingAverage pressureAverage_yx;
   MovingAverage pressureAverage_yy;
   MovingAverage pressureAverage_zx;
   MovingAverage pressureAverage_zy;
   MovingAverage pressureAverage_zz;
   MovingAverage groupPressureAverage_xx;
   MovingAverage groupPressureAverage_xy;
   MovingAverage groupPressureAverage_xz;
   MovingAverage groupPressureAverage_yx;
   MovingAverage groupPressureAverage_yy;
   MovingAverage groupPressureAverage_yz;
   MovingAverage groupPressureAverage_zx;
   MovingAverage groupPressureAverage_zy;
   MovingAverage groupPressureAverage_zz;

private:
    CthThread thread;
    static void threadRun(Controller*);

    double startCTime;
    double startWTime;
    double firstCTime;
    double firstWTime;
    double startBenchTime;

    int computesPartitioned;
};

//Modifications for alchemical fep
static char *FEPTITLE(int X)
{
  static char tmp_string[21];
  sprintf(tmp_string, "FepEnergy: %6d ",X);
  return tmp_string;
}

static char *FEPTITLE_BACK(int X)
{
  static char tmp_string[21];
  sprintf(tmp_string, "FepE_back: %6d ",X);
  return tmp_string;
}

static char *FEPTITLE2(int X)
{
  static char tmp_string[21];
  sprintf(tmp_string, "FEP:    %7d",X);
  return tmp_string;
}

static char *TITITLE(int X)
{
  static char tmp_string[21];
  sprintf(tmp_string, "TI:     %7d",X);
  return tmp_string;
}
//fepe

#endif // CONTROLLER_H

