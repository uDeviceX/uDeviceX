PREFIX    = $(HOME)
BIN       = $(PREFIX)/bin
ARCH      ?= -arch compute_35 -code sm_35

NVCCFLAGS  += -use_fast_math -restrict
NVCCLIBS   += $(NVCC_LIBS) -lcudart -lcurand -lnvToolsExt

P = \
udx_dep.pc\
udx_cpu.pc\
udx_cuda.pc

udx_dep.pc: udx_dep.pc.in
	CFLAGS="$(CXXFLAGS)" LIBS="$(LIBS)" ; \
	sed -e "s|@CFLAGS@|$$CFLAGS|g" -e "s|@LIBS@|$$LIBS|g"       $< > $@

udx_cpu.pc: udx_cpu.pc.in
	sed "s|@PREFIX@|$(PREFIX)|g" $< > $@

udx_cuda.pc: udx_cuda.pc.in
	CFLAGS="$(ARCH) --compiler-options $(shell ./coma $(CXXFLAGS))" ; LIBS="$(NVCCLIBS)" ; \
	sed -e "s|@PREFIX@|$(PREFIX)|g" \
            -e "s|@CFLAGS@|$$CFLAGS|g" \
            -e "s|@LIBS@|$$LIBS|g" $< > $@

install: $P
	u.install $P "$(PKG)"

show:; @echo $(NVCCLIBS)

PHONY: clean
clean:; rm -rf $P
