CXXFLAGS = \
	-I$(CRAY_MPICH2_DIR)/include \
	-L$(CRAY_MPICH2_DIR)/lib \
	-I$(HDF5_DIR)/include \
	-L$(HDF5_DIR)/lib \
	-std=c++11 -O3

vrate: main.cpp
	$(CXX) $(CXXFLAGS) $< -o vrate

clean: 
	rm vrate

.PHONY = clean