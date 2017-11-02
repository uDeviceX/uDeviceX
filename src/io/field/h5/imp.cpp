#include <hdf5.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "imp.h"

namespace h5 {

static hid_t create(const char *const path) {
    hid_t id, file;
    id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(id, m::cart, MPI_INFO_NULL);
    file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, id);
    H5Pclose(id);
    return file;
}

static void close(hid_t file_id) {
    herr_t rc;
    rc = H5Fclose(file_id);
}

static void write0(hid_t file_id,
                   const float * const channeldata[],
                   const char * const * const channelnames, const int nchannels) {
    int i;
    const int L[3] = { XS, YS, ZS };
    hsize_t globalsize[4] = {(hsize_t) m::dims[2] * L[2], (hsize_t) m::dims[1] * L[1], (hsize_t) m::dims[0] * L[0], 1};
    hid_t filespace_simple = H5Screate_simple(4, globalsize, NULL);

    for(i = 0; i < nchannels; ++i) {
        hid_t dset_id = H5Dcreate(file_id, channelnames[i], H5T_NATIVE_FLOAT, filespace_simple,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);

        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        hsize_t start[4]  = { (hsize_t) m::coords[2] * L[2], (hsize_t) m::coords[1] * L[1], (hsize_t) m::coords[0] * L[0], 0};
        hsize_t extent[4] = { (hsize_t) L[2], (hsize_t) L[1], (hsize_t) L[0],  1};
        hid_t filespace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, extent, NULL);

        hid_t memspace = H5Screate_simple(4, extent, NULL);
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, channeldata[i]);

        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Pclose(plist_id);
        H5Dclose(dset_id);
    }

    H5Sclose(filespace_simple);
}

void write(const char * const path,
           const float * const channeldata[],
           const char * const * const channelnames, const int nchannels) {
    hid_t file_id;
    file_id = create(path);
    write0(file_id, channeldata, channelnames, nchannels);
    close(file_id);
}

} /* namespace */
