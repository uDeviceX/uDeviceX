void fields(const char * const path2h5,
            const float * const channeldata[],
            const char * const * const channelnames, const int nchannels) {
#ifndef NO_H5
    hid_t plist_id_access = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id_access, m::cart, MPI_INFO_NULL);

    hid_t file_id = H5Fcreate(path2h5, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_access);
    H5Pclose(plist_id_access);

    const int L[3] = { XS, YS, ZS };
    hsize_t globalsize[4] = {(hsize_t) m::dims[2] * L[2], (hsize_t) m::dims[1] * L[1], (hsize_t) m::dims[0] * L[0], 1};
    hid_t filespace_simple = H5Screate_simple(4, globalsize, NULL);

    for(int ichannel = 0; ichannel < nchannels; ++ichannel)
    {
        hid_t dset_id = H5Dcreate(file_id, channelnames[ichannel], H5T_NATIVE_FLOAT, filespace_simple,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);

        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        hsize_t start[4]  = { (hsize_t) m::coords[2] * L[2], (hsize_t) m::coords[1] * L[1], (hsize_t) m::coords[0] * L[0], 0};
        hsize_t extent[4] = { (hsize_t) L[2], (hsize_t) L[1], (hsize_t) L[0],  1};
        hid_t filespace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, extent, NULL);

        hid_t memspace = H5Screate_simple(4, extent, NULL);
        herr_t status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, channeldata[ichannel]);

        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Pclose(plist_id);
        H5Dclose(dset_id);
    }

    H5Sclose(filespace_simple);
    H5Fclose(file_id);

    if (!m::rank) wrapper(path2h5, channelnames, nchannels);
#endif // NO_H5
}
