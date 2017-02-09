.PHONY: clean

clean:
	rm -f *.cudaobj
	rm -f *.o *.a

libcuda-rbc.a: rbc-cuda.cudaobj
	ar rcs $@ $^

%.o:%.cpp
	$(CXX)  $(CXXFLAGS)  $(INCLUDES) -fPIC -c $< -o $@

%.cudaobj:%.cu
	$(NVCC)  $(NVCCFLAGS)  $(INCLUDES) -c $< -o $@
