static const int NVP = 3; /* number of vertices per face */

static void copy_v(const Particle *f, /**/ Particle *t) {
    enum {X, Y, Z};
    t->v[X] = f->v[X];
    t->v[Y] = f->v[Y];
    t->v[Z] = f->v[Z];
}
static void shift(const Particle *f, int n, /**/ Particle *t) {
    /* f, t: from, to
       see mesh/shift/     */
    int i;
    for (i = 0; i < n; i++) {
        shift0(f, t); copy_v(f, t); f++; t++;
    }
}

static void write(const void * const ptr, const int nbytes32, MPI_File f) {
    MPI_Offset base;
    MC(MPI_File_get_position(f, &base));
    MPI_Offset offset = 0, nbytes = nbytes32;
    MC(MPI_Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, m::cart));
    MPI_Status status;
    MC(MPI_File_write_at_all(f, base + offset, ptr, nbytes, MPI_CHAR, &status));
    MPI_Offset ntotal = 0;
    MC(MPI_Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, m::cart) );
    MC(MPI_File_seek(f, ntotal, MPI_SEEK_CUR));
}

static void header(int nc0, int nv, int nt, MPI_File f) {
    int nc; /* total number of cells */
    int sz = 0;
    char s[BUFSIZ] = {0};
    nc = 0;
    m::Reduce(&nc0, &nc, 1, MPI_INT, MPI_SUM, 0, m::cart) ;
    if (m::rank == 0)
        sz = sprintf(s,
                     "ply\n"
                     "format binary_little_endian 1.0\n"
                     "element vertex %d \n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property float u\nproperty float v\nproperty float w\n"
                     "element face  %d  \n"
                     "property list int int vertex_index\n"
                     "end_header\n", nv*nc, nt*nc);
    write(s, sz, f);
}

static void vert(const Particle *pp, int nc, int nv, MPI_File f) {
    int n;
    n = nc * nv;
    write(pp, sizeof(Particle) * n, f);
}

static void wfaces0(int *buf, const int4 *faces, int nc, int nv, int nt, MPI_File f) {
    /* write faces */
    int c, t, b;  /* cell, triangle, buffer index */
    int n, shift;
    n = nc * nv;
    shift = 0;
    MPI_Exscan(&n, &shift, 1, MPI_INTEGER, MPI_SUM, m::cart);

    b = 0;
    int4 tri;
    for(c = 0; c < nc; ++c)
        for(t = 0; t < nt; ++t) {
            tri = faces[t];
            buf[b++] = NVP;
            buf[b++] = shift + nv*c + tri.x;
            buf[b++] = shift + nv*c + tri.y;
            buf[b++] = shift + nv*c + tri.z;
        }
    write(buf, b*sizeof(buf[0]), f);
}

static void wfaces(const int4 *faces, int nc, int nv, int nt, MPI_File f) {
    int *buf; /* buffer for faces */
    int sz;
    sz = (1 + NVP) * nc * nt * sizeof(int);
    UC(emalloc(sz, (void**) &buf));
    wfaces0(buf, faces, nc, nv, nt, f);
    free(buf);
}

static void dump0(const Particle *pp, const int4 *faces,
                  int nc, int nv, int nt, MPI_File f) {
    header(nc,        nv, nt, f);
    vert(pp,      nc, nv,     f);
    wfaces(faces, nc, nv, nt, f);
}

static void dump1(const Particle *pp, const int4 *faces, int nc, int nv, int nt, MPI_File f) {
    int sz, n;
    Particle *pp0;
    n = nc * nv;
    sz = n*sizeof(Particle);
    UC(emalloc(sz, (void**) &pp0));
    shift(pp, n, /**/ pp0); /* copy-shift to global coordinates */
    dump0(pp0, faces, nc, nv, nt, f);
    free(pp0);
}

static void dump2(const Particle *pp, const int4 *faces, int nc, int nv, int nt, const char *fn) {
    MPI_File f;
    MPI_File_open(m::cart, fn, MPI_MODE_WRONLY |  MPI_MODE_CREATE, MPI_INFO_NULL, &f);
    MPI_File_set_size(f, 0);
    dump1(pp, faces, nc, nv, nt, f);
    MPI_File_close(&f);
}

void rbc_mesh_dump(const Particle *pp, const int4 *faces, int nc, int nv, int nt, int id) {
    const char *fmt = DUMP_BASE "/r/%05d.ply";
    char f[BUFSIZ]; /* file name */
    sprintf(f, fmt, id);
    if (m::rank == 0) os::mkdir(DUMP_BASE "/r");
    dump2(pp, faces, nc, nv, nt, f);
}

void rig_mesh_dump(const Particle *pp, const int4 *faces, int nc, int nv, int nt, int id) {
    const char *fmt = DUMP_BASE "/s/%05d.ply";
    char f[BUFSIZ]; /* file name */
    sprintf(f, fmt, id);
    if (m::rank == 0) os::mkdir(DUMP_BASE "/s");
    dump2(pp, faces, nc, nv, nt, f);
}
