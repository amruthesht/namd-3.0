#ifdef GPUBUF_H
#define GPUBUF_H
#include "cudaUtils.h"
#include "DeviceCUDA.h"

/* Nodewide global variable definitions */

class GBData(){
public:
  double* _gpos_x;
  double* _gpos_y;
  double* _gpos_z;

  double* _gvel_x;
  double* _gvel_y;
  double* _gvel_z;

  double* _gf_normal_x;
  double* _gf_normal_y;
  double* _gf_normal_z;

  double* _gf_nbond_x;
  double* _gf_nbond_y;
  double* _gf_nbond_z;

  double* _gf_slow_x;
  double* _gf_slow_y;
  double* _gf_slow_z;
  
  int *offsets;
  int *peList;
  
  GBData(){}
  
  GBData(int deviceID, int natoms){
    cudaCheck(cudaSetDevice(deviceID));
   
    // allocates everything here
    allocate_device<double>(&_gpos_x, natoms);
    allocate_device<double>(&_gpos_y, natoms);
    allocate_device<double>(&_gpos_z, natoms);
    allocate_device<double>(&_gvel_x, natoms);
    allocate_device<double>(&_gvel_y, natoms);
    allocate_device<double>(&_gvel_z, natoms);
    
    allocate_device<double>(&_gf_normal_x, natoms);
    allocate_device<double>(&_gf_normal_y, natoms);
    allocate_device<double>(&_gf_normal_z, natoms);
    
    allocate_device<double>(&_gf_nbond_x, natoms);
    allocate_device<double>(&_gf_nbond_y, natoms);
    allocate_device<double>(&_gf_nbond_z, natoms);
    
    allocate_device<double>(&_gf_slow_x, natoms);
    allocate_device<double>(&_gf_slow_y, natoms);
    allocate_device<double>(&_gf_slow_z, natoms);
  }
    
};

#endif