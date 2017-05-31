CXX=g++
CXXFLAGS= -O3 -Wall -Wpedantic -std=c++11

EXECS = gensphere genellipsoid gencylinder genellipse gendisk

default: all
all: $(EXECS)

gensphere: gensphere.o mesh.o ply.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

genellipsoid: genellipsoid.o mesh.o ply.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

gencylinder: gencylinder.o mesh.o ply.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

genellipse: genellipse.o mesh.o ply.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

gendisk: gendisk.o mesh.o ply.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXECS)
