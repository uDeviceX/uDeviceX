#include <assert.h>
#include <hdf5.h>

#include <conf.h>
#include "inc/conf.h"

#include "msg.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "imp.h"

namespace h5 {

struct IDs {
    hid_t plist;
    hid_t file;
};

void create(const char *const path, /**/ IDs *ids) {
    hid_t plist, file;
    herr_t rc;

    plist = H5Pcreate(H5P_FILE_ACCESS);
    if (plist < 0) ERR("fail to create plist for <%s>", path);

    rc = H5Pset_fapl_mpio(plist, m::cart, MPI_INFO_NULL);
    if (rc < 0) ERR("file to store MPI information for <%s>", path);

    file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, plist);
    if (file < 0) ERR("fail to create file <%s>", path);

    ids->plist = plist;
    ids->file  = file;
}

static void close(IDs ids, const char *path) {
    if (H5Pclose(ids.plist) < 0)
        ERR("fail to close file  id <%s>", path);

    if (H5Fclose(ids.file) < 0)
        ERR("fail to close plist id <%s>", path);
}

static void write_float(hid_t dataset_id,
                        hid_t mem_space_id,
                        hid_t file_space_id,
                        hid_t xfer_plist_id,
                        const void *buf) {
    herr_t rc;
    rc = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, mem_space_id, file_space_id, xfer_plist_id, buf);
    if (rc < 0)
        ERR("fail to write data set");
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
        write_float(dset_id, memspace, filespace, plist_id, channeldata[i]);

        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Pclose(plist_id);
        H5Dclose(dset_id);
    }

    H5Sclose(filespace_simple);
}

void write(const char * const path,
           const float * const data[],
           const char * const * const names, const int n) {
    IDs ids;
    create(path, /**/ &ids);
    write0(ids.file, data, names, n);
    close(ids, path);
}

} /* namespace */
