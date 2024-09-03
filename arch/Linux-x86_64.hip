ifneq ($(ROCM_DIR),)
   ROCMDIR = $(ROCM_DIR)
endif
HIPDIR  = $(ROCMDIR)

#ifeq (,$(wildcard $(HIPDIR)/bin/hipconfig))
#   HIPDIR = $(ROCMDIR)/hip
#endif

ifeq ($(PLATFORM),)
    PLATFORM = $(shell $(HIPDIR)/bin/hipconfig -P)
endif
CUDADIR = $(HIPDIR)
export HIP_PLATFORM=$(PLATFORM)
HIPCCOPTS = -std=c++17 -O3 -fPIE -fexpensive-optimizations -ffast-math $(shell export HIP_PLATFORM=$(PLATFORM); $(HIPDIR)/bin/hipconfig --cpp_config) 
ifneq ($(HIPCUB_DIR),)
    HIPCCOPTS += -I$(HIPCUB_DIR)
endif
ifneq ($(PLATFORM),nvcc)
    ifneq ($(ROCPRIM_DIR),)
        HIPCCOPTS += -I$(ROCPRIM_DIR)
    endif
    #-fnogpu-rdc means we do not need to specify a specific linker, 
    #as the kernels will be generated at compile time rather than link time.
    #The (optional) --amdgpu-target directives are useful for cross-compiling to different AMD architectures.
    HIPCCOPTS += -march=native -ffast-math -fno-gpu-rdc -fcuda-flush-denormals-to-zero -Wno-invalid-command-line-argument -Wno-unused-command-line-argument -Wno-invalid-constexpr -Wno-ignored-optimization-argument
    HIPCCOPTS += -Wno-unused-private-field
    HIPCCOPTS += -fdenormal-fp-math=ieee -munsafe-fp-atomics --offload-arch=gfx906,gfx908,gfx90a,gfx1030 -fno-slp-vectorize

else
    # include CUB
    CUBDIR=.rootdir/cub
    HIPCCOPTS += --use_fast_math  -I$(CUBDIR) -DNAMD_CUDA
endif

HIPCC = $(HIPDIR)/bin/hipcc -m64

ROCMV = $(shell export HIP_PLATFORM=$(PLATFORM); $(HIPDIR)/bin/hipconfig -v | cut -d "." -f 1,2 )

CUDAFLAGS = -DNAMD_HIP
ifneq ($(PLATFORM),nvcc)
    #ROCM version 3.3 was the last one to use the old hcc compiler.
    ifeq ($(ROCMV),3.3)
    	CUDALIB = -L$(HIPDIR)/lib -lhip_hcc -lhipfft -lhiprand -lroctx64 -lroctracer64
    else
    	CUDALIB = -L$(HIPDIR)/lib -lamdhip64 -lhipfft -lrocfft -lhiprand -lroctx64 -lroctracer64
    endif
    # CHARMLD = -ld++ $(HIPCC)
    CUDAFLAGS += -I$(HIPDIR)/include -I$(HIPDIR)/include/hipfft -I$(HIPDIR)/include/rocfft -I$(HIPDIR)/include/hiprand -I$(HIPDIR)/include/rocrand -I$(HIPDIR)/include/roctracer #-DDISABLE_P2P
else
    CUDALIB = -L$(CUDADIR)/lib64 -lcudart -lrt -lcufft
    CUDAFLAGS += -DNAMD_CUDA
endif
CUDAOBJS = $(CUDAOBJSRAW)
CUDA = $(CUDAFLAGS) -I. $(shell export HIP_PLATFORM=$(PLATFORM); $(HIPDIR)/bin/hipconfig --cpp_config) 
CUDACC = $(HIPCC)
CUDACCOPTS = $(HIPCCOPTS) $(CUDAFLAGS)
