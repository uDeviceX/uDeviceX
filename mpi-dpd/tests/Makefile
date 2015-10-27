-include ../.cache.Makefile

NVCC ?= nvcc -ccbin $(CXX)
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35

NVCCFLAGS += -I$(HDF5_DIR)/include -Xcudafe "--diag_suppress=unrecognized_gcc_pragma"
NVCCFLAGS +=  -arch $(ARCH_VAL) -code $(CODE_VAL) -O3 -use_fast_math -g -DNDEBUG -Xcompiler "-fopenmp"
CXXFLAGS += -L../../cuda-dpd/dpd -L../../cuda-rbc/ -L../../cuda-ctc/ -O3 -g  -std=c++11 -DNDEBUG -fopenmp
NVCCFLAGS += -I../../cuda-dpd/dpd -I../../cuda-dpd/ -I../../cuda-rbc/ -I../../cuda-ctc -I../
LIBS = -lcuda-dpd -lcuda-rbc -lcuda-ctc -lcudart -ldl -lz -lhdf5 -fopenmp

testcontact: ../test testcontact.o
	rm -f main.o
	$(CXX) $(CXXFLAGS) *.o $(LIBS) -o testcontact
	
../test:
	make -C ../
	cp ../*.o .
	
testcontact.o: testcontact.cu ../argument-parser.h ../common.h ../containers.h ../contact.h ../../cuda-dpd/dpd-rng.h
		$(NVCC) $(NVCCFLAGS) testcontact.cu -c -o testcontact.o
	
.PHONY: ../test