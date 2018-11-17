include ../../../conf.mk
LD = $(CXX)
UDX_CXXFLAGS = $(shell u.pkg-config udx_cuda --cflags)
UDX_LDFLAGS  = $(shell u.pkg-config udx_cpu --libs) $(shell u.pkg-config udx_cuda --libs)

main: main.o; $(LD) $^ $(UDX_LDFLAGS) -o $@
main.o: main.cu; $(NCC) $(UDX_CXXFLAGS) $< -c -o $@
install: main; u.install main $(BIN)

main.o: dev.h

clean:; rm -f main main.o
.PHONY: install clean
