LD = g++
UDX_CXXFLAGS = `u.pkg-config udx_cpu --cflags`
UDX_LDFLAGS  = `u.pkg-config udx_cuda --libs`

main: main.o
	$(LD) $^ $(UDX_LDFLAGS) -o $@

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(UDX_CXXFLAGS) $< -c -o $@

