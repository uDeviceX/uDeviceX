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
	rm -f *.o *.a
	rm -f *.d *.dcu

libcuda-rbc.a: rbc-cuda.cudaobj
	ar rcs $@ $^

%.o:%.cpp
	$(CXX)  $(CXXFLAGS)  $(INCLUDES) -fPIC -c $< -o $@

%.cudaobj:%.cu
	$(NVCC)  $(NVCCFLAGS)  $(INCLUDES) -c $< -o $@

%.d: %.cpp
	$(CXX)  $(CXXFLAGS)  -MM $< > $*.d

%.dcu: %.cu
	$(NVCC)  $(NVCCFLAGS)  $(INCLUDES) -M $< -o $@

