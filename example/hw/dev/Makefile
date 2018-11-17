include ../../../conf.mk
LD = $(NCC)
UDX_CXXFLAGS = `u.pkg-config udx_cuda --cflags` \
--compiler-options '`u.pkg-config udx_cpu --cflags`' \
--compiler-options '$(CXXFLAGS)'

UDX_LDFLAGS  = `u.pkg-config udx_cuda --libs`   --compiler-options   '`u.pkg-config udx_cpu --libs`'

main: main.o; $(LD) $^ $(UDX_LDFLAGS) -o $@
main.o: main.cu; $(NCC) $(CXXFLAGS) $(UDX_CXXFLAGS) $< -c -o $@
install: main; u.install main $(BIN)

main.o: dev.h

clean:; rm -f main main.o
.PHONY: install clean
