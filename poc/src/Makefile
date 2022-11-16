.POSIX:
.SUFFIXES:
.SUFFIXES: .o
.SUFFIXES: .cpp
.SUFFIXES: .cu

MPICXX=mpicxx
LIBCONFIG_CXXFLAGS=

include var.mk
all: $O
.cpp.o:; $(MPICXX) -c -I. -I`dirname $<` $(CXXFLAGS) $(LIBCONFIG_CXXFLAGS) $< -o $@


