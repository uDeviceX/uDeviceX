F95FLAGS = -O2 -g # -gline -u -dcfuns -C -C=undefined
F95LINKFLAGS = $(F95FLAGS)
F95 = gfortran
P = Timings Compare
OBJS = Precision.o JenkinsTraubSolver.o \
       Solvers.o

all: $P
%.o : %.mod
%.o: %.f90; $(F95) $(F95FLAGS) -o $@ -c $<
%: %.o; $(F95) $(F95LINKFLAGS) -o $@ $^

Timings: $(OBJS) Timings.o
	$(F95) $(F95LINKFLAGS) -o $@ $^

Compare: $(OBJS) Compare.o
	$(F95) $(F95LINKFLAGS) -o $@ $^

Timings.o: JenkinsTraubSolver.o \
			Solvers.o \
			Precision.o

Compare.o: JenkinsTraubSolver.o \
			Solvers.o \
			Precision.o

JenkinsTraubSolver.o: Precision.o
Solvers.o: Precision.o

Precision.o:

.PHONY: clean
clean:
	rm -f *.o *.d *.g90 *.mod
	rm -f $E
