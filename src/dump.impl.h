namespace dump
{
void ini() {
    if (m::rank == 0)
    mkdir(DUMP_BASE "/bop", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    const int L[3] = {XS, YS, ZS};        
    for (int c = 0; c < 3; ++c) mi[c] = (m::coords[c] + 0.5) * L[c];

    w_pp = new float[7*MAX_PART_NUM];

    MC(MPI_Type_contiguous(global_ids ? 7 : 6, MPI_FLOAT, &dumptype));
    MC(MPI_Type_commit(&dumptype));
}

void fin() {
    delete[] w_pp;
    MC(MPI_Type_free(&dumptype)); 
}

static void copy_shift(const Particle *pp, const long n, float *w) {
    for (int j = 0; j < n; ++j)
    for (int d = 0; d < 3; ++d) {
        w[6 * j + d]     = pp[j].r[d] + mi[d];
        w[6 * j + 3 + d] = pp[j].v[d];
    }
}

static void copy_shift_id(const Particle *pp, const int *ii, const long n, float *w) {
    for (int j = 0; j < n; ++j) {
        for (int d = 0; d < 3; ++d) {
            w[7 * j + d]     = pp[j].r[d] + mi[d];
            w[7 * j + 3 + d] = pp[j].v[d];
        }
        w[7 * j + 6] = ii[j];
    }
}

#define PATTERN "%s-%05d"
    
static void header(const bool dumpid, const long n, const char *name, const int step) {
    char fname[256] = {0};
    sprintf(fname, DUMP_BASE "/bop/" PATTERN ".bop", name, step / part_freq);
        
    FILE *f = fopen(fname, "w");

    if (f == NULL)
    ERR("could not open <%s>\n", fname);

    fprintf(f, "%ld\n", n);
    fprintf(f, "DATA_FILE: " PATTERN ".values\n", name, step / part_freq);
    fprintf(f, "DATA_FORMAT: float\n");
    if (dumpid) fprintf(f, "VARIABLES: x y z vx vy vz id\n");
    else        fprintf(f, "VARIABLES: x y z vx vy vz\n");
    fclose(f);
}
    
void parts(const Particle *pp, const long n, const char *name, const int step) {
    copy_shift(pp, n, /**/ w_pp);
        
    char fname[256] = {0};
    sprintf(fname, DUMP_BASE "/bop/" PATTERN ".values", name, step / part_freq);

    MPI_File f;
    MPI_Status status;
    MPI_Offset base, offset = 0;
    MPI_Offset len = n * sizeof(Particle);

    long ntot = 0;
    MC( l::m::Reduce(&n, &ntot, 1, MPI_LONG, MPI_SUM, 0, m::cart) );
    MC( MPI_File_open(m::cart, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f) );
    MC( MPI_File_set_size(f, 0) );
    MC( MPI_File_get_position(f, &base) ); 

    if (m::rank == 0) header(false, ntot, name, step);

    MC( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, m::cart) );
    MC( MPI_File_write_at_all(f, base + offset, w_pp, n, Particle::datatype(), &status) ); 
    MC( MPI_File_close(&f) );
}

void parts_ids(const Particle *pp, const int *ii, const long n, const char *name, const int step) {
    copy_shift_id(pp, ii, n, /**/ w_pp);
        
    char fname[256] = {0};
    sprintf(fname, DUMP_BASE "/bop/" PATTERN ".values", name, step / part_freq);

    MPI_File f;
    MPI_Status status;
    MPI_Offset base, offset = 0;
    MPI_Offset len = n * (sizeof(Particle) + sizeof(float));

    long ntot = 0;
    MC( l::m::Reduce(&n, &ntot, 1, MPI_LONG, MPI_SUM, 0, m::cart) );
    MC( MPI_File_open(m::cart, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f) );
    MC( MPI_File_set_size(f, 0) );
    MC( MPI_File_get_position(f, &base) ); 

    if (m::rank == 0) header(true, ntot, name, step);

    MC( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, m::cart) );
    MC( MPI_File_write_at_all(f, base + offset, w_pp, n, dumptype, &status) ); 
    MC( MPI_File_close(&f) );
}

#undef PATTERN
}
