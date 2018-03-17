int write_file_open(MPI_Comm comm, const char *fn, /**/ WriteFile **fp) {
    EMALLOC(1, fp);
    MC(MPI_File_open(comm, fn, MPI_MODE_WRONLY |  MPI_MODE_CREATE, MPI_INFO_NULL, &(*fp)->f));
    MC(MPI_File_set_size((*fp)->f, 0));
    return 0;
}

int write_file_close(WriteFile *fp) {
    MPI_File f;
    f = fp->f;
    MC(MPI_File_close(&f));
    EFREE(fp);
    return 0;
}

void write_all(MPI_Comm comm, const void * const ptr, const int nbytes32, WriteFile *fp) {
    MPI_File f = fp->f;
    MPI_Offset base;
    MPI_Offset offset = 0, nbytes = nbytes32;
    MPI_Status status;
    MPI_Offset ntotal = 0;
    MC(MPI_File_get_position(f, &base));
    MC(MPI_Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, comm));
    MC(MPI_File_write_at_all(f, base + offset, ptr, nbytes, MPI_CHAR, &status));
    MC(MPI_Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, comm) );
    MC(MPI_File_seek(f, ntotal, MPI_SEEK_CUR));
}

int write_master(MPI_Comm comm, const void * const ptr, int sz0, WriteFile *fp) {
    int sz;
    sz = m::is_master(comm) ? sz0 : 0;
    write_all(comm, ptr, sz, fp);
    return 0;
}

int write_shift_indices(MPI_Comm comm, int n, int *shift0) {
    *shift0 = 0;
    MC(MPI_Exscan(&n, shift0, 1, MPI_INTEGER, MPI_SUM, comm));
    return 0;
}

int write_reduce(MPI_Comm comm, int n0, /**/ int *n) {
    *n = 0;
    MC(m::Reduce(&n0, n, 1, MPI_INT, MPI_SUM, 0, comm));
    return 0;
}
