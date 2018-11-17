include ../../../conf.mk
LD = $(CXX)
UDX_CXXFLAGS = `u.pkg-config udx_cpu --cflags`
UDX_LDFLAGS  = `u.pkg-config udx_cpu --libs`

main: main.o; $(LD) $< $(UDX_LDFLAGS) -o $@

main.o: main.cpp; $(CXX) $(CXXFLAGS) $(UDX_CXXFLAGS) $^ -c -o $@
install: main; u.install main $(BIN)

clean:; rm -f main main.o
.PHONY: install clean
