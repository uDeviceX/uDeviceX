# Root build for the editable CUDA compilation units.

NVCC = nvcc
CCBIN = mpicxx
ARCH = -arch compute_90 -code sm_90
OPT = -O2 -g
CPPFLAGS = -I.
NVCCFLAGS = -std=c++11 $(OPT) -use_fast_math -restrict

MPI_LIBS = -lopen-rte -lopen-pal

OBJECTS = \
 config.o \
 mpi.o \
 os.o \
 tri.o \
 scalar.o \
 vector.o \
 kahan.o \
 cuda.o \
 mpicheck.o \
 text.o \
 linalg.o \
 random.o \
 timestep.o \
 timeline.o \
 restrain.o \
 iopoint.o \
 txtio.o \
 diagpart.o \
 restart.o \
 meshio.o \
	halfedge.o \
	util.o \
	bforce.o \
	vtk.o \
	fragmenthost.o \
	meshwrite.o \
	meshread.o \
 transform.o \
 coords.o \
 comm.o \
 color.o \
 error.o \
 message.o \
 edge.o \
 stats.o \
 convert.o \
 minmax.o \
 keylist.o \
 scan.o \
 contactutil.o \
 clist.o \
 contact.o \
 debug.o \
 colorhost.o \
 settings.o \
 inflow.o \
 control.o \
 filter.o \
 bop.o \
 gridsampler.o \
 memory.o \
 halo.o \
 mesh.o \
 force.o \
 io.o \
 body.o \
 main.o \
 wall.o

TARGET = build/udx
DEVICE = build/device.o
DEPS = $(OBJECTS:.o=.d)

.SUFFIXES: .cu .o

.cu.o:
	$(NVCC) $(ARCH) -ccbin $(CCBIN) $(NVCCFLAGS) $(CPPFLAGS) -MMD -MP -MF $(@:.o=.d) -dc -c $< -o $@

all: $(TARGET)

$(OBJECTS): Makefile

$(TARGET): $(OBJECTS) $(DEVICE)
	mkdir -p build
	$(NVCC) $(ARCH) -ccbin $(CCBIN) $(NVCCFLAGS) $(OBJECTS) $(DEVICE) $(MPI_LIBS) -lcudart -lcurand -o $@

$(DEVICE): $(OBJECTS)
	mkdir -p build
	$(NVCC) $(ARCH) -ccbin $(CCBIN) -dlink $(OBJECTS) -o $@

clean:
	rm -f $(OBJECTS) $(DEPS) $(DEVICE) $(TARGET)

.PHONY: all clean

-include $(DEPS)
