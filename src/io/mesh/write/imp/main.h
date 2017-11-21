void all(const void * const ptr, const int nbytes32, File *fp) {
    MPI_File f = fp->f;
    MPI_Offset base;
    MPI_Offset offset = 0, nbytes = nbytes32;
    MPI_Status status;
    MPI_Offset ntotal = 0;
    MC(MPI_File_get_position(f, &base));
    MC(MPI_Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, m::cart));
    MC(MPI_File_write_at_all(f, base + offset, ptr, nbytes, MPI_CHAR, &status));
    MC(MPI_Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, m::cart) );
    MC(MPI_File_seek(f, ntotal, MPI_SEEK_CUR));
}
/* root predicate */
int rootp() { return m::rank == 0; }
void one(const void * const ptr, int sz0, File *fp) {
    int sz;
    sz = rootp() ? sz0 : 0;
    all(ptr, sz, fp);
}

int shift(int n, int *shift0) {
    *shift0 = 0;
    MC(MPI_Exscan(&n, shift0, 1, MPI_INTEGER, MPI_SUM, m::cart));
    return 0;
}

int fopen(const char *fn, /**/ File *fp) {
    MC(MPI_File_open(m::cart, fn, MPI_MODE_WRONLY |  MPI_MODE_CREATE, MPI_INFO_NULL, &fp->f));
    MC(MPI_File_set_size(fp->f, 0));
    return 0;
}

int fclose(File *fp) {
    MPI_File f;
    f = fp->f;
    MC(MPI_File_close(&f));
    return 0;
}
