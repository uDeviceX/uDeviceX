struct File { MPI_File f; };

static void all(const void * const ptr, const int nbytes32, MPI_File *fp) {
    MPI_File f = *fp;

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
static int rootp() { return m::rank == 0; }
static void one(const void * const ptr, int sz0, MPI_File *fp) {
    int sz;
    sz = (rootp()) ? sz0 : 0;
    all(ptr, sz, fp);
}

static int fopen(const char *fn, /**/ MPI_File *fp) {
    MPI_File f;
    f = *fp;
    MC(MPI_File_open(m::cart, fn, MPI_MODE_WRONLY |  MPI_MODE_CREATE, MPI_INFO_NULL, &f));
    MC(MPI_File_set_size(f, 0));
    return 0;
}

static int fclose(MPI_File *fp) {
    MPI_File f;
    f = *fp;
    MC(MPI_File_close(&f));
    return 0;
}
