/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef GLOBALMASTERIMD_H
#define GLOBALMASTERIMD_H

#include "GlobalMaster.h"
#include "imd.h"
#include "ResizeArray.h"

class FloatVector;

class GlobalMasterIMD : public GlobalMaster {
 public: 
  /* initializes this according to the simulation parameters */
  GlobalMasterIMD();
  ~GlobalMasterIMD();

  void send_energies(IMDEnergies *);
  void send_fcoords(int, FloatVector *);
  void send_velocities(int, FloatVector *);
  void send_forces(int, FloatVector *);
  void send_box(IMDBox *);
  void send_time(IMDTime *);

 protected:

  friend class IMDOutput;

  virtual void calculate();

  // Simple function for getting MDComm-style forces from VMD
  void get_vmd_forces();

  // IMD protocol version
  int IMDversion;

  // flag for whether to proceed with simulation when there are no connections
  int IMDwait;

  // flag for whether to ignore all user input
  int IMDignore;

  // flag for whether to ignore only forces
  int IMDignoreForces;

  // IMD session info i.e. send settings
  IMDSessionInfo IMDsendsettings;

  // My server socket handle
  void *sock;

  // Connected sockets
  ResizeArray<void *>clients;

  // temporaries in case 3*sizeof(float) != sizeof(FloatVector)
  float *coordtmp;
  int coordtmpsize;

  float *veltmp;
  int veltmpsize;

  float *forcetmp;
  int forcetmpsize;
};

#endif

