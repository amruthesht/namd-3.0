# IMD v3 developement

The IMDv3-dev branch focuses on the developement of a new IMD protocol for sending additional information via IMD.
Details of the protocol can be found [here](https://heyden-lab-asu.atlassian.net/wiki/x/AQA4).

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

Build and test the Charm++/Converse library (InfiniBand UCX OpenMPI PMIx version):
(Make sure modules and corresponding headers for openmpi and pmix are available and accessible before running this)
```
    cd charm-8.0.0
    ./build charm++ ucx-linux-x86_64 ompipmix --with-production
    cd ucx-linux-x86_64-ompipmix/tests/charm++/megatest
    make
    mpiexec -n 4 ./megatest
    cd ../../../../..
```
The `mpiexec` command is run as any other OpenMPI program on your cluster on ana intercative node. Make sure to request appropriate resources for it.

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

InfiniBand UCX version:
```
  ./config Linux-x86_64-g++ --charm-arch ucx-linux-x86_64-ompipmix
```

```
  cd Linux-x86_64-g++
  make  
```

## Example Run

Run an example simulation (this is a 66-atom simulation) and compare to the sample output (for seed 12345) in the dierctory:
```
  ./namd3 ../examples/alanin/shorter-run/alanin
```

If running on a cluster with N cores allocated for the job, one might have to run something like
```
  mpiexec -n N ./namd3 ../examples/alanin/shorter-run/alanin
```