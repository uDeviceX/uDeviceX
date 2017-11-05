.SUFFIXES:

%.o : %.mod

%.o: %.f90
	$(F95) $(F95FLAGS) -o $@ -c $<

%: %.o
	$(F95) $(F95LINKFLAGS) -o $@ $^ $(LIBS)

vpath %.f90 ../Src

# Now define the compiler and option flags.
F95FLAGS = -O2 -g # -gline -u -dcfuns -C -C=undefined
F95LINKFLAGS = $(F95FLAGS)

F95 = gfortran

executables = polynomialRootsTimings polynomialRootsCompare

all: $(executables)

OBJS = SetWorkingPrecision.o JenkinsTraubSolver.o \
       Polynomial234RootSolvers.o

polynomialRootsTimings: $(OBJS) PolynomialRootsTimings.o
	$(F95) $(F95LINKFLAGS) -o $@ $^ $(LIBS)

polynomialRootsCompare: $(OBJS) PolynomialRootsCompare.o
	$(F95) $(F95LINKFLAGS) -o $@ $^ $(LIBS)

PolynomialRootsTimings.o: JenkinsTraubSolver.o \
			Polynomial234RootSolvers.o \
			SetWorkingPrecision.o

PolynomialRootsCompare.o: JenkinsTraubSolver.o \
			Polynomial234RootSolvers.o \
			SetWorkingPrecision.o

JenkinsTraubSolver.o: SetWorkingPrecision.o

Polynomial234rootSolvers.o: SetWorkingPrecision.o

SetWorkingPrecision.o:

.PHONY: clean
clean:
	rm -f *.o *.d *.g90 *.mod
	rm -f $(executables)
