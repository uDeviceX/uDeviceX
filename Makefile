NVCC ?= nvcc
CCBIN ?= mpicxx
ARCH ?= -arch compute_90 -code sm_90
OPT ?= -O2 -g

TARGET := build/udx
UNITS := $(wildcard udx.*.cu)
OBJECTS := $(UNITS:.cu=.o)
DEVICE := build/device.o

HDF5_ROOT ?= $(firstword $(wildcard /scratch/slitvinov/build/hdf5 /scratch/slitvinov/miniforge3/pkgs/hdf5-*/))
NVTX_INCLUDE ?= $(firstword $(wildcard /usr/local/cuda-*/targets/*/include /scratch/slitvinov/.grace/Linux_aarch64/26.3/cuda/*/targets/sbsa-linux/include))

CPPFLAGS += -I.
ifneq ($(strip $(HDF5_ROOT)),)
CPPFLAGS += -I$(HDF5_ROOT)/include
LIBS += -L$(HDF5_ROOT)/lib -lhdf5
endif
ifneq ($(strip $(NVTX_INCLUDE)),)
CPPFLAGS += -I$(NVTX_INCLUDE)
endif
LIBS += ${HDF5_LIBS} ${NVCC_LIBS} ${LIBCONFIG_LIBS}
NVCCFLAGS := -std=c++11 $(OPT) -use_fast_math -restrict
MPI_LIBS := $(filter-out -Wl% -pthread,$(shell $(CCBIN) --showme:link 2>/dev/null)) -lopen-rte -lopen-pal

.SUFFIXES: .cu .o

.cu.o:
	$(NVCC) $(ARCH) -ccbin $(CCBIN) $(NVCCFLAGS) $(CPPFLAGS) -dc -c $< -o $@

all: $(TARGET)

$(TARGET): $(OBJECTS) $(DEVICE) | build
	$(NVCC) $(ARCH) -ccbin $(CCBIN) $(NVCCFLAGS) $(OBJECTS) $(DEVICE) $(LIBS) $(MPI_LIBS) -lcudart -lcurand -o $@

$(DEVICE): $(OBJECTS) | build
	$(NVCC) $(ARCH) -ccbin $(CCBIN) -dlink $(OBJECTS) -o $@

build:
	mkdir -p $@

clean:
	rm -f $(OBJECTS) $(DEVICE) $(TARGET)

.PHONY: all clean
