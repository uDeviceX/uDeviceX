# Root build for the editable CUDA compilation units.

NVCC = nvcc
CCBIN = mpicxx
ARCH = -arch compute_90 -code sm_90
OPT = -O2 -g
CPPFLAGS = -I.
NVCCFLAGS = -std=c++11 $(OPT) -use_fast_math -restrict

MPI_LIBS = -lopen-rte -lopen-pal

OBJECTS = \
 udx.contact-complete.o \
 udx.runtime-config-complete.o \
 udx.support-grid-algo-complete.o \
 udx.control-complete.o \
 udx.distr-exch.o \
 udx.distr-flu.o \
 udx.distr-rbc.o \
 udx.distr-rig.o \
 udx.exch-flu.o \
 udx.exch-mesh.o \
 udx.exch-obj.o \
 udx.flu-halo.o \
 udx.fluforces-bulk-main.o \
 udx.io-fraghe-complete.o \
 udx.pair-fsi.o \
 udx.mesh-struct-complete.o \
 udx.rbc-core-adj-com.o \
 udx.rbc-force-all.o \
 udx.rbc-force-bending-all.o \
 udx.rig-complete.o \
 udx.scheme-system-complete.o \
 udx.sim.o \
 udx.wall-complete.o \
 udx.wall-sdf-complete.o

TARGET = build/udx
DEVICE = build/device.o

.SUFFIXES: .cu .o

.cu.o:
	$(NVCC) $(ARCH) -ccbin $(CCBIN) $(NVCCFLAGS) $(CPPFLAGS) -dc -c $< -o $@

all: $(TARGET)

$(TARGET): $(OBJECTS) $(DEVICE)
	mkdir -p build
	$(NVCC) $(ARCH) -ccbin $(CCBIN) $(NVCCFLAGS) $(OBJECTS) $(DEVICE) $(MPI_LIBS) -lcudart -lcurand -o $@

$(DEVICE): $(OBJECTS)
	mkdir -p build
	$(NVCC) $(ARCH) -ccbin $(CCBIN) -dlink $(OBJECTS) -o $@

clean:
	rm -f $(OBJECTS) $(DEVICE) $(TARGET)

.PHONY: all clean
