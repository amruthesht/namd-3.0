# IMD v3 developement

The IMDv3-dev branch focuses on the developement of a new IMD protocol for sending additional information via IMD.
Details of the protocol can be found [here](https://github.com/Becksteinlab/imdclient/blob/protocolv3/docs/source/protocol_v3.rst).

## Instructions for compiling and building NAMD

### Dependencies and requirements

Building a complete NAMD binary from source code requires:

- working C and C++ compilers;
- a compiled version of the Charm++/Converse library;
- a compiled version of the TCL library and its header files;
- a compiled version of the FFTW library and its header files;
- a C shell (csh/tcsh) to run the script used to configure the build.

NAMD can be compiled without TCL or FFTW, but certain features will be
disabled.  Fortunately, precompiled TCL and FFTW libraries are available from
http://www.ks.uiuc.edu/Research/namd/libraries/.  You may disable
these options by specifying --without-tcl --without-fftw as options
when you run the config script.  Some files in arch may need editing
to set the path to TCL and FFTW libraries correctly.

### Building and Compiling Charm++

As an example, here is the build sequence for 64-bit Linux workstations:

Unpack NAMD and matching Charm++ source code:

``` 
    git clone https://github.com/amruthesht/namd-3.0.git
    cd namd-3.0
    tar xf charm-8.0.0.tar
```

Build and test the Charm++/Converse library (testing is optional)

When using single-node multicore version:
```
    cd charm-8.0.0
    ./build charm++ multicore-linux-x86_64 --with-production
    cd multicore-linux-x86_64/tests/charm++/megatest
    make
    ./megatest +p4   (multicore does not support multiple nodes)
    cd ../../../../..
```

When using InfiniBand UCX OpenMPI PMIx version:
(Make sure modules and corresponding headers for openmpi and pmix are available and accessible before running this)
```
    cd charm-8.0.0
    ./build charm++ ucx-linux-x86_64 ompipmix --with-production
    cd ucx-linux-x86_64-ompipmix/tests/charm++/megatest
    make
    mpiexec -n 4 ./megatest
    cd ../../../../..
```

When using MPI version:
```
    cd charm-8.0.0
    env MPICXX=mpicxx ./build charm++ mpi-linux-x86_64 --with-production
    cd mpi-linux-x86_64/tests/charm++/megatest
    make pgm
    mpiexec -n 4 ./megatest   (run as any other MPI program on your cluster)
    cd ../../../../..
```

The `mpiexec` command is run as any other OpenMPI or MPI program on your cluster on an intercative node. Make sure to request appropriate resources for it.

### Installing header-only dependencies

Download and install TCL and FFTW libraries:
  (cd to `namd-3.0` if you're not already there)
```
    wget http://www.ks.uiuc.edu/Research/namd/libraries/fftw-linux-x86_64.tar.gz
    tar xzf fftw-linux-x86_64.tar.gz
    mv linux-x86_64 fftw
    wget http://www.ks.uiuc.edu/Research/namd/libraries/tcl8.6.13-linux-x86_64.tar.gz
    wget http://www.ks.uiuc.edu/Research/namd/libraries/tcl8.6.13-linux-x86_64-threaded.tar.gz
    tar xzf tcl8.6.13-linux-x86_64.tar.gz
    tar xzf tcl8.6.13-linux-x86_64-threaded.tar.gz
    mv tcl8.6.13-linux-x86_64 tcl
    mv tcl8.6.13-linux-x86_64-threaded tcl-threaded
```

### Compiling and Building NAMD

Set up build directory and compile for NAMD:
multicore version:
```
  ./config Linux-x86_64-g++ --charm-arch multicore-linux-x86_64
```
InfiniBand UCX version:
```
  ./config Linux-x86_64-g++ --charm-arch ucx-linux-x86_64-ompipmix
```
MPI version:
```
  ./config Linux-x86_64-g++ --charm-arch mpi-linux-x86_64
```
GPU-resident CUDA multicore version:
``` 
    ./config Linux-x86_64-g++ --charm-arch multicore-linux-x86_64 --with-single-node-cuda
```
GPU-resident CUDA ethernet version:
```
  ./config Linux-x86_64-g++ --charm-arch netlrts-linux-x86_64 --with-single-node-cuda
```
GPU-resident HIP multicore version:
```
  ./config Linux-x86_64-g++ --charm-arch multicore-linux-x86_64 --with-single-node-hip
```
GPU-resident HIP ethernet version:
```
  ./config Linux-x86_64-g++ --charm-arch netlrts-linux-x86_64 --with-single-node-hip
```

You might also need `--cuda-prefix CUDA_DIRECTORY` for the GPU ones, if needed.

Finally, `make` the namd source files to create an executable

```
  cd Linux-x86_64-g++
  make  
```

## Input configuration options

We have introduced new configuration settings in the input file `*.conf` in accordance with the protocol defined [here](https://github.com/Becksteinlab/imdclient/blob/protocolv3/docs/source/protocol_v3.rst). The following options and settings can be set in accordance with the user's need to stream data and trajectory information.

#### General IMD options (same as v2)
1. `IMDon` - streaming on or off (yes/no)
2. `IMDport` - port number to listen on (typically 8888)
3. `IMDfreq` - frequency to send data
4. `IMDwait` - wait for client to connect before starting simulation (on/off)

#### New IMD options (new in v3)
5. `IMDversion` - 2 for VMD and 3 for latest protocol
#### IMD session info settings (yes/no)
6. `IMDsendPositions` - sending positions of entire system
7. `IMDsendEnergies` - sending energies and bonded, non-bonded and other contributions
8. `IMDsendTime` - sending time information (time, dt, step)
9. `IMDsendBoxDimensions` - sending box dimensions (lattice vectors a, b, c). If box dimensions are not defined, default unit box is sent
10. `IMDsendVelocities` - sending velocities of entire system
11. `IMDsendForces` - sending forces on all atoms
12. `IMDwrapPositions` - wrapping positions to box; applicable when IMDsendPositions is yes

## Example Run

Run an example simulation (this is a 66-atom simulation) and compare to the `-sample` output files (for seed 12345) in the dierctory:
```
  ./namd3 ../examples/alanin/shorter-run/alanin.conf
```

If running on a cluster with $N$ cores allocated for the job, one might have to run something like
```
  mpiexec -n N ./namd3 ../examples/alanin/shorter-run/alanin.conf
```