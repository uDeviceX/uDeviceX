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
 udx.conf.o \
 udx.coords.o \
 udx.d.o \
 udx.dbg.o \
 udx.algo-convert.o \
 udx.algo-edg.o \
 udx.algo-force_stat.o \
 udx.algo-kahan_sum.o \
 udx.algo-key_list.o \
 udx.algo-minmax.o \
 udx.algo-scan.o \
 udx.algo-scalars.o \
 udx.algo-vectors.o \
 udx.control-den.o \
 udx.control-inflow.o \
 udx.control-outflow.o \
 udx.control-vel.o \
 udx.distr-common.o \
 udx.distr-flu-status.o \
 udx.distr-flu.o \
 udx.distr-map.o \
 udx.distr-rbc.o \
 udx.distr-rig.o \
 udx.exch-common.o \
 udx.exch-flu.o \
 udx.exch-map.o \
 udx.exch-mesh.o \
 udx.exch-obj.o \
 udx.flu.o \
 udx.fluforces-bulk.o \
 udx.fluforces-halo.o \
 udx.fluforces.o \
 udx.frag.o \
 udx.fsi.o \
 udx.grid_sampler.o \
 udx.he.o \
 udx.inter-color.o \
 udx.io-bop.o \
 udx.io-text.o \
 udx.io-diagnostics.o \
 udx.io-grid-output.o \
 udx.io-mesh.o \
 udx.io-mesh-reader.o \
 udx.io-point.o \
 udx.io-restart.o \
 udx.io-rig.o \
 udx.io-vtk-output.o \
 udx.io-write.o \
 udx.libconfig.o \
 udx.main.o \
 udx.math.o \
 udx.mesh-geometry.o \
 udx.mesh-measures.o \
 udx.mesh-collision.o \
 udx.mesh-coordinates.o \
 udx.mesh-distance-props.o \
 udx.mesh-eng_julicher.o \
 udx.mesh-eng_kantor.o \
 udx.mesh-force-kantor0.o \
 udx.mesh-force-kantor1.o \
 udx.mesh-gen-matrices.o \
 udx.mesh-gen.o \
 udx.mesh-scatter.o \
 udx.mesh-topology.o \
 udx.mesh_bounce.o \
 udx.mpi.o \
 udx.pair.o \
 udx.rbc-adj.o \
 udx.rbc-com.o \
 udx.rbc-force-area_volume.o \
 udx.rbc-force-bending-juelicher.o \
 udx.rbc-force-bending-kantor.o \
 udx.rbc-force-bending.o \
 udx.rbc-force-rnd-api.o \
 udx.rbc-force-rnd.o \
 udx.rbc-force.o \
 udx.rbc-params.o \
 udx.rbc-shape.o \
 udx.rbc-stretch.o \
 udx.rbc.o \
 udx.rig-gen.o \
 udx.rig.o \
 udx.rigid.o \
 udx.scheme-force.o \
 udx.scheme-move.o \
 udx.scheme-restrain.o \
 udx.scheme-time_line.o \
 udx.scheme-time_step.o \
 udx.sim-objects.o \
 udx.sim-objinter.o \
 udx.sim-opt.o \
 udx.sim-walls.o \
 udx.sim.o \
 udx.struct-arrays.o \
 udx.struct-pfarrays.o \
 udx.support.o \
 udx.utils-string.o \
 udx.utils.o \
 udx.wall-exch.o \
 udx.wall-force.o \
 udx.wall-sdf-array3d.o \
 udx.wall-sdf-bounce.o \
 udx.wall-sdf-field.o \
 udx.wall-sdf-label.o \
 udx.wall-sdf-tex3d.o \
 udx.wall-sdf-tform.o \
 udx.wall-sdf.o \
 udx.wall-wvel.o \
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
