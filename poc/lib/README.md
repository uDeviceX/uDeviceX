make MPICXX=mpicxx HDF5_CXXFLAGS="`pkg-config --cflags hdf5`" CXXFLAGS=-I/usr/local/cuda/include

make MPICXX=mpicxx
nvcc -lcurand -ccbin=mpicxx --linker-options=../lib/libudx.a,/apps/spack/opt/spack/linux-centos7-skylake_avx512/gcc-10.3.0/hdf5-1.12.2-mmf6eesafa3jcbdmuaqy4c2lowhxnoz5/lib/libhdf5.a,-lsz,-lz main.o -o main
