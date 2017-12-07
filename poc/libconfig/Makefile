CXX=g++

LIBCONFIG = ${HOME}/prefix/libconfig

CXXFLAGS= -Wall -O3
CXXFLAGS += `pkg-config --cflags libconfig`
LDFLAGS  += `pkg-config --libs libconfig`

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean: ; rm -rf main
