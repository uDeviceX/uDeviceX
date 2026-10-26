# Root build for the editable CUDA compilation units.

NVCC = nvcc
CCBIN = mpicxx
ARCH = -arch compute_90 -code sm_90
OPT = -O2 -g
CPPFLAGS = -I.
NVCCFLAGS = -std=c++11 $(OPT) -use_fast_math -restrict

MPI_LIBS = -lopen-rte -lopen-pal

OBJECTS = \
 udx.clist.o \
 udx.cnt.o \
 udx.color.o \
 udx.comm.o \
 udx.conf-libconfig.o \
 udx.coords.o \
 udx.d-mpi.o \
 udx.dbg.o \
 udx.algo-all.o \
 udx.algo-edge-accumulators-scan.o \
 udx.control-den-outflow.o \
 udx.control-inflow.o \
 udx.control-vel.o \
 udx.distr-common.o \
 udx.distr-exch-map.o \
 udx.distr-flu.o \
 udx.distr-rbc.o \
 udx.distr-rig.o \
 udx.exch-common.o \
 udx.exch-flu.o \
 udx.exch-mesh.o \
 udx.exch-obj.o \
 udx.flu.o \
 udx.fluforces-bulk-main.o \
 udx.fluforces-halo.o \
 udx.frag-he.o \
 udx.fsi.o \
 udx.grid_sampler.o \
 udx.inter-color.o \
 udx.io-bop.o \
 udx.io-text-diagnostics.o \
 udx.io-point-grid.o \
 udx.io-mesh-rig-restart.o \
 udx.io-vtk-output.o \
 udx.main.o \
 udx.utils-math.o \
 udx.mesh-complete.o \
 udx.pair.o \
 udx.rbc-core-adj-com.o \
 udx.rbc-force-all.o \
 udx.rbc-force-bending-juelicher.o \
 udx.rbc-force-bending-kantor.o \
 udx.rig-complete.o \
 udx.scheme-all.o \
 udx.sim-objects.o \
 udx.sim-objinter.o \
 udx.sim-opt.o \
 udx.sim-walls.o \
 udx.sim.o \
 udx.struct-arrays.o \
 udx.struct-pfarrays.o \
 udx.support.o \
 udx.wall-exchange-force.o \
 udx.wall-sdf-transform-bounce.o \
 udx.wall-sdf-label.o \
 udx.wall-sdf.o \
 udx.wall.o

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
