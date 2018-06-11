CXX=g++
CXXFLAGS=-O3 -Wall
LDFLAGS=-lm

main: main.o com.o disp.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^


clean:; rm -rf *.o main
.PHONY: clean
