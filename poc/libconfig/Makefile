CXX = g++

CXXFLAGS += -Wall -O3
CXXFLAGS += `pkg-config --cflags libconfig`

libconfigL  = `pkg-config --libs-only-l libconfig`
libconfigD  = `pkg-config --variable=libdir libconfig`

LDFLAGS += -L${libconfigD} -Wl,-rpath -Wl,${libconfigD} ${libconfigL}

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean test
clean: ; rm -rf main
test:  ; atest main.cpp
