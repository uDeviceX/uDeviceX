CXX=g++
CXXFLAGS=-std=c++11 -O3 -Wall -Wpedantic

srcs := $(wildcard *.cpp)
execs := $(srcs:%.cpp=%)

default: $(execs)

% :: %.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm -rf $(execs)

