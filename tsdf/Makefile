CXX?=g++
CXXFLAGS?=-O2 -g -Wall -Wextra

all: sdf2volume sdf2volume2 sdf2vtk mergesdf

%: %.cpp
	$(CXX) $< $(CXXFLAGS) -o $@


.PHONY: clean
clean:
	-rm sdf2volume sdf2volume sdf2vtk mergesdf
