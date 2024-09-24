
#ifndef IMD_H__
#define IMD_H__

#include <limits.h>
#include <vector>
#include <common.h>

#if ( INT_MAX == 2147483647 )
typedef int     int32;
#else
typedef short   int32;
#endif

enum IMDType {
  IMD_DISCONNECT, //  = 0
  IMD_ENERGIES, //  = 1
  IMD_FCOORDS, // = 2
  IMD_GO, // = 3
  IMD_HANDSHAKE, // = 4
  IMD_KILL, // = 5
  IMD_MDCOMM, // = 6
  IMD_PAUSE, // = 7
  IMD_TRATE, // = 8
  IMD_IOERROR, // = 9
  //  New in IMD v3
  IMD_SESSIONINFO, // = 10
  IMD_RESUME, // = 11
  IMD_TIME, // = 12
  IMD_BOX, // = 13
  IMD_VELOCITIES, // = 14
  IMD_FORCES, // = 15
};

typedef struct {
  int32 tstep;
  float T;
  float Etot;
  float Epot;
  float Evdw;
  float Eelec;
  float Ebond;
  float Eangle;
  float Edihe;
  float Eimpr;
} IMDEnergies;

typedef struct {
  BigReal dt;
  BigReal time;
  long int tstep;
} IMDTime;

typedef struct {
  float ax, ay, az;
  float bx, by, bz;
  float cx, cy, cz;
} IMDBox;

typedef struct {
  int time_switch;
  int energies_switch;
  int box_switch;
  int fcoords_switch;
  int wrap_switch;
  int velocities_switch;
  int forces_switch;

  // Method to convert IMDSessionInfo into a std::vector<Type>
  template <typename Type = char>
  std::vector<Type> toTypeVector() const {
    std::vector<Type> data;
    data.push_back(static_cast<Type>(time_switch));
    data.push_back(static_cast<Type>(energies_switch));
    data.push_back(static_cast<Type>(box_switch));
    data.push_back(static_cast<Type>(fcoords_switch));
    data.push_back(static_cast<Type>(wrap_switch));
    data.push_back(static_cast<Type>(velocities_switch));
    data.push_back(static_cast<Type>(forces_switch));
    return data;
  }

} IMDSessionInfo;

// Send simple messages - these consist of a header with no subsequent data
extern int   imd_disconnect(void *);
extern int   imd_pause(void *);
extern int   imd_kill(void *);
extern int   imd_handshake(void *, const int);
extern int   imd_trate(void *, int32);
template <typename Type = char>
extern int   imd_sessioninfo(void *, const IMDSessionInfo *);

// Send data
extern int   imd_send_mdcomm(void *, int32, const int32 *, const float *);
extern int   imd_send_energies(void *, const IMDEnergies *);
extern int   imd_send_fcoords(void *, int32, const float *);
extern int   imd_send_velocities(void *, int32, const float *);
extern int   imd_send_forces(void *, int32, const float *);
extern int   imd_send_box(void *, const IMDBox *);
extern int   imd_send_time(void *, const IMDTime *);

/// Receive header and data 

// recv_handshake returns 0 if server and client have the same relative 
// endianism; returns 1 if they have opposite endianism, and -1 if there
// was an error in the handshake process.
extern int imd_recv_handshake(void *);

extern IMDType imd_recv_header(void *, int32 *);
extern int imd_recv_mdcomm(void *, int32, int32 *, float *);
extern int imd_recv_energies(void *, IMDEnergies *);
extern int imd_recv_fcoords(void *, int32, float *);
extern int imd_recv_velocities(void *, int32, float *);
extern int imd_recv_forces(void *, int32, float *);
extern int imd_recv_box(void *, IMDBox *);
extern int imd_recv_time(void *, IMDTime *);

#endif

