struct File { MPI_File f; };
void all(MPI_Comm cart, const void * const ptr, const int nbytes32, File *fp) {
    MPI_File f = fp->f;
    MPI_Offset base;
    MPI_Offset offset = 0, nbytes = nbytes32;
    MPI_Status status;
    MPI_Offset ntotal = 0;
    MC(MPI_File_get_position(f, &base));
    MC(MPI_Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, cart));
    MC(MPI_File_write_at_all(f, base + offset, ptr, nbytes, MPI_CHAR, &status));
    MC(MPI_Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, cart) );
    MC(MPI_File_seek(f, ntotal, MPI_SEEK_CUR));
}
int one(MPI_Comm cart, const void * const ptr, int sz0, File *fp) {
    int sz;
    sz = m::is_master(cart) ? sz0 : 0;
    all(cart, ptr, sz, fp);
    return 0;
}

int shift(MPI_Comm cart, int n, int *shift0) {
    *shift0 = 0;
    MC(MPI_Exscan(&n, shift0, 1, MPI_INTEGER, MPI_SUM, cart));
    return 0;
}

int reduce(MPI_Comm cart,int n0, /**/ int* n) {
    *n = 0;
    MC(m::Reduce(&n0, n, 1, MPI_INT, MPI_SUM, 0, cart));
    return 0;
}

int fopen(MPI_Comm cart, const char *fn, /**/ File **fp) {
    *fp = (File*)malloc(sizeof(File));
    MC(MPI_File_open(cart, fn, MPI_MODE_WRONLY |  MPI_MODE_CREATE, MPI_INFO_NULL, &(*fp)->f));
    MC(MPI_File_set_size((*fp)->f, 0));
    return 0;
}

int fclose(File *fp) {
    MPI_File f;
    f = fp->f;
    MC(MPI_File_close(&f));
    free(fp);
    return 0;
}
