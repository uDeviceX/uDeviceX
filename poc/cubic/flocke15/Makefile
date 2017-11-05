F95FLAGS = -O2 -g # -gline -u -dcfuns -C -C=undefined
F95LINKFLAGS = $(F95FLAGS)
F95 = gfortran
P = timings compare
OBJS = precision.o jenkins.o \
       solvers.o

all: $P
%.o : %.mod
%.o: %.f90; $(F95) $(F95FLAGS) -o $@ -c $<
%: %.o; $(F95) $(F95LINKFLAGS) -o $@ $^

timings: $(OBJS) timings.o
	$(F95) $(F95LINKFLAGS) -o $@ $^

compare: $(OBJS) compare.o
	$(F95) $(F95LINKFLAGS) -o $@ $^

timings.o: jenkins.o \
			solvers.o \
			precision.o

compare.o: jenkins.o \
			solvers.o \
			precision.o

jenkins.o: precision.o
solvers.o: precision.o

precision.o:

.PHONY: clean
clean:
	rm -f *.o *.d *.g90 *.mod
	rm -f $E
