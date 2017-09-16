CXX = gcc
CXXFLAGS = -O3 -Wall -Wpedantic

P = deltas frag
all: $P

%: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ 

clean:
	rm -rf $P
