-include .cache.Makefile

CXXFLAGS +=  -L$(subst :, -L,$(LD_LIBRARY_PATH))
CXXFLAGS+= -fopenmp -std=c++11 -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated
GPULIBS+= -lcudart
GPUFLAGS+= -arch=$(ARCH_VAL)  -code=$(CODE_VAL)  
GPUFLAGS+= --use_fast_math
CXXFLAGS+= -O3 -g -fno-tree-vectorize -funroll-loops -DNDEBUG
GPUFLAGS+= -O4 -DNDEBUG

.PHONY: clean

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(wildcard $(DIRS:%/=%/*.cpp))))
endif

ifeq "$(findstring cuda,$(MAKECMDGOALS))" "cuda"
-include $(notdir $(patsubst %.cu,%.dcu,$(wildcard $(DIRS:%/=%/*.cu))))
endif

vpath %.cpp %.h %.cu $(DIRS)

clean:
	rm -f *.cudaobj
	rm -f *.o *.so *.a
	rm -f *.d *.dcu
	rm -f test-rbc

libcuda-rbc.a: rbc-cuda.cudaobj
	ar rcs $@ $^

%.o:%.cpp
	$(CXX)  $(CXXFLAGS)  $(INCLUDES) -fPIC -c $< -o $@

%.cudaobj:%.cu
	$(NVCC)  $(GPUFLAGS)  $(INCLUDES) -c $< -o $@

%.d: %.cpp
	$(CXX)  $(CXXFLAGS) $(THRUST_TARGET_CPU) -MM $< > $*.d

%.dcu: %.cu
	$(NVCC)  $(GPUFLAGS)  $(INCLUDES) -M $< -o $@

