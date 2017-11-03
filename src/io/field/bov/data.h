void write_values_seq(const char *fdname, int nx, int ny, int nz, const float *data) {
    long n;
    f = safe_open(fdname, "wb");

    n = nx * ny * nz;
    fwrite(d, sizeof(float), n, f);
    
    fclose(f);
}

static void write_line(MPI_File f, MPI_Offset base, long nx, const float *data) {
    MPI_Status status;
    MC( MPI_File_write_at_all(f, base, data, nx, MPI_FLOAT, &status) );
}

static void write_plane(MPI_File f, MPI_Offset base, long nx, long ny, const float *data) {
    MPI_OFFSET offset;
    long iy, lnline, nline, strt;

    lnline = nx;
    nline = lnline * dims[0];
    
    for (iy = 0; iy < ny; ++iy) {
        strt = (ny * coords[1] + iy) * nline;
        offset = strt * sizeof(float);
        write_line(f, base + offset, nx, data + iy * lnline);
    }
}

static void write_bulk(const int coords[3], const int dims[3], MPI_File f, MPI_Offset base, long nx, long ny, long nz, const float *data) {
    MPI_OFFSET offset;
    long iz, lnplane, nplane, strt;

    lnplane = nx * ny;
    nplane = lnplane * dims[0] * dims[1];
    
    for (iz = 0; iz < nz; ++iz) {
        strt = (nz * coords[2] + iz) * nplane;
        offset = strt * sizeof(float);
        write_plane(f, base + offset, nx, ny, data + iz * lnplane);
    }
}

void write_values(MPI_Comm comm, const int coords[3], const int dims[3], const char *fname, int nx, int ny, int nz, const float *data) {
    MPI_File f;
    MPI_Offset base, offset = 0;
    long i;

    MC(MPI_File_open(comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f));
    MC(MPI_File_set_size(f, 0));
    MC(MPI_File_get_position(f, &base)); 

    write_planes(coords, dims, f, base, nx, ny, nz, data);
    
    MC( MPI_File_close(&f) );
    return ntot;
}
