static void write(const void * const ptr, const int nbytes32, MPI_File f) {
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
static void write_once(const void * const ptr, int sz0, MPI_File f) {
    int sz;
    sz = (rootp()) ? sz0 : 0;
    write(ptr, sz, f);
}
