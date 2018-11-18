LINK = $(CXX)
UDX_CXXFLAGS = $(shell u.pkg-config udx_cpu --cflags)
UDX_NCCFLAGS = $(shell u.pkg-config udx_cuda --cflags)
UDX_LDFLAGS  = $(shell u.pkg-config udx_cpu --libs) $(shell u.pkg-config udx_cuda --libs)

%.o: %.cpp; $(NCC) $(CXXFLAGS) $(UDX_NCCFLAGS) $< -c -o $@
%.o: %.cu;  $(NCC) $(CXXFLAGS) $(UDX_NCCFLAGS) $< -c -o $@
%: %.o; $(LINK) $^ $(UDX_LDFLAGS) $(LDLIBS) -o $@
