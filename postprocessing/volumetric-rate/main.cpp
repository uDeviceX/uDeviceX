#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>

using namespace std;

#include <hdf5.h>

#include "../mpi-check.h"

int main(int argc, char ** argv)
{
    if (argc != 4)
    {
	perror("usage: ./vrate <xextent> <yextent> <zextent>\n"); 
	return EXIT_FAILURE;
    }

    //const char * const path2h5 = argv[1]; //"flowfields-0629.h5";
    const int xsize = atoi(argv[1]);//40 * 56;
    const int ysize = atoi(argv[2]);//10 * 52;
    const int zsize = atoi(argv[3]);//56;

    MPI_CHECK( MPI_Init(&argc, &argv) );

    MPI_Comm comm = MPI_COMM_WORLD;

    auto udotnx = [&](const char * const path2h5, unsigned int x)
	{
	    id_t plist_id_access = H5Pcreate(H5P_FILE_ACCESS);
	    H5Pset_fapl_mpio(plist_id_access, comm, MPI_INFO_NULL);
	    hid_t file_id = H5Fopen(path2h5, H5F_ACC_RDONLY, plist_id_access);
	    H5Pclose(plist_id_access);

	    hsize_t globalsize[4] = { (hsize_t)zsize, (hsize_t)ysize, (hsize_t)xsize, 1};
	    hid_t filespace_simple = H5Screate_simple(4, globalsize, NULL);

	    hid_t dset_id = H5Dopen(file_id, "u", H5P_DEFAULT);
	    id_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	    hsize_t start[4] = { 0, 0, x, 0};
	    hsize_t extent[4] = { globalsize[0], globalsize[1], 1,  1};
	    hid_t filespace = H5Dget_space(dset_id);
	    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, extent, NULL);

	    hid_t memspace = H5Screate_simple(4, extent, NULL);

	    const int N = globalsize[0] * globalsize[1];
	    float * data = new float[N];
	    herr_t status = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data);
    
	    double U = 0;
	    for(int i = 0; i < N; ++i)
		U += data[i];

	    delete [] data;
    
	    H5Sclose(memspace);
	    H5Pclose(plist_id);
	    H5Dclose(dset_id);
	    H5Sclose(filespace_simple);
	    H5Fclose(file_id);

	    return U;
	};

    for (string line; getline(cin, line);)
    {
	double avgU = 0;
	int nslices = 0;

	for(int i = xsize - 3; i < xsize; ++i, ++nslices)
	    avgU += udotnx(line.c_str(), i);

	printf("%e\n", avgU / nslices);
    }

    MPI_CHECK( MPI_Finalize() );

    return 0;
}
