CXX=g++
CXXFLAGS=-O3 -Wall
LDFLAGS=

main: main.o com.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $^


clean:; rm -rf *.o main
.PHONY: clean
