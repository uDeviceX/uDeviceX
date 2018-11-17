# a fragment of makefile
# see src/make/ and tools/udeps
# B: binary dir
# S: source dir
#
# O: list of objects

# commands
# D: create directories
# N: nvcc compile
# X: c++ compile
# L: link
# DL: device link
# A: archive

PREFIX    = $(HOME)
BIN       = $(PREFIX)/bin
NVCC     ?= nvcc
ARCH     ?= -arch compute_35 -code sm_35

NCFLAGS     = $(ARCH)   $(CXXFLAGS)
XCFLAGS     =           $(CXXFLAGS)
NVCCFLAGS  += -use_fast_math -restrict
NVCCLIBS   += -lcudart -lcurand -lnvToolsExt

P = \
udx_dep.pc\
udx_cpu.pc\
udx_cuda.pc

udx_dep.pc: udx_dep.pc.in
	CFLAGS="$(XCFLAGS)" LIBS="$(LIBS)" ; \
	sed -e "s|@CFLAGS@|$$CFLAGS|g" -e "s|@LIBS@|$$LIBS|g"       $< > $@

udx_cpu.pc: udx_cpu.pc.in
	sed "s|@PREFIX@|$(PREFIX)|g" $< > $@

udx_cuda.pc: udx_cuda.pc.in
	CFLAGS="$(NVCCFLAGS)" LIBS="$(NVCCLIBS)" ; \
	sed -e "s|@PREFIX@|$(PREFIX)|g" \
            -e "s|@CFLAGS@|$$CFLAGS|g" \
            -e "s|@LIBS@|$$LIBS|g" $< > $@

install: $P
	u.install $P "$(PREFIX)/lib/pkgconfig"

PHONY: clean
clean:; rm -rf $P
