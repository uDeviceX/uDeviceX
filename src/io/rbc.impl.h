static const int NVP = 3; /* number of vertices per face */

static void shift0(Particle* f, /**/ Particle* t) {
    enum {X, Y, Z};
    int *co;
    co = m::coords;
    t->r[X] = f->r[X] + 0.5*XS + co[X]*XS;
    t->r[Y] = f->r[Y] + 0.5*YS + co[Y]*YS;
    t->r[Z] = f->r[Z] + 0.5*ZS + co[Z]*ZS;
}

static void shift(Particle *f, int n, /**/ Particle *t) {
    /* f, t: from, to */
    int i;
    for (i = 0; i < n; i++) shift0(f++, t++);
}

static void write(const void *const ptr, int nbytes, MPI_File f) {
    MPI_Offset base, offset, ntotal;
    MPI_Status status;
    MC(MPI_File_get_position(f, &base));
    offset = 0;
    MC(MPI_Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, m::cart));
    MC(MPI_File_write_at_all(f, base + offset, ptr, nbytes, MPI_CHAR, &status));
    ntotal = 0;
    MC(MPI_Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, m::cart) );
    MC(MPI_File_seek(f, ntotal, MPI_SEEK_CUR));
}

static void header(int nc0, int nv, int nt, MPI_File f) {
    int nc; /* total number of cells */
    nc = 0;
    l::m::Reduce(&nc0, &nc, 1, MPI_INT, MPI_SUM, 0, m::cart) ;
    std::stringstream ss;
    if (m::rank == 0) {
        ss <<  "ply\n";
        ss <<  "format binary_little_endian 1.0\n";
        ss <<  "element vertex " << nv * nc << "\n";
        ss <<  "property float x\nproperty float y\nproperty float z\n";
        ss <<  "property float u\nproperty float v\nproperty float w\n";
        ss <<  "element face " << nt * nc << "\n";
        ss <<  "property list int int vertex_index\n";
        ss <<  "end_header\n";
    }
    std::string content = ss.str();
    write(content.c_str(), content.size(), f);
}

static void vert(Particle *pp, int nc, int nv, MPI_File f) {
    int n;
    n = nc * nv;
    write(pp, sizeof(Particle) * n, f);
}

static void wfaces0(int *buf, int *faces, int nc, int nv, int nt, MPI_File f) {
    /* write faces */
    int c, t, b;  /* cell, triangle, buffer index */
    int n, shift;
    n = nc * nv;
    shift = 0;
    MPI_Exscan(&n, &shift, 1, MPI_INTEGER, MPI_SUM, m::cart);

    b = 0;
    for(c = 0; c < nc; ++c)
    for(t = 0; t < nt; ++t) {
        buf[b++] = NVP;
        buf[b++] = shift + nv*c + faces[3*t    ];
        buf[b++] = shift + nv*c + faces[3*t + 1];
        buf[b++] = shift + nv*c + faces[3*t + 2];        
    }
    write(buf, sizeof(&buf) * b, f);
}

static void wfaces(int *faces, int nc, int nv, int nt, MPI_File f) {
    int *buf; /* buffer for faces */
    int sz;
    sz = (1 + NVP) * nc * nt * sizeof(int);
    buf = (int*)malloc(sz);
    wfaces0(buf, faces, nc, nv, nt, f);
    free(buf);
}

static void dump0(Particle *pp, int *faces,
                  int nc, int nv, int nt, MPI_File f) {
    header(nc,        nv, nt, f);
    vert(pp,      nc, nv,     f);
    wfaces(faces, nc, nv, nt, f);
}

static void dump1(Particle  *pp, int *faces, int nc, int nv, int nt, MPI_File f) {
    int sz, n;
    Particle* pp0;
    n = nc * nv;
    sz = n*sizeof(Particle);
    pp0 = (Particle*)malloc(sz);
    shift(pp, n, /**/ pp0); /* copy-shift to global coordinates */
    dump0(pp0, faces, nc, nv, nt, f);
    free(pp0);
}

static void dump2(Particle  *pp, int *faces, int nc, int nv, int nt, const char *fn) {
    MPI_File f;
    MPI_File_open(m::cart, fn, MPI_MODE_WRONLY |  MPI_MODE_CREATE, MPI_INFO_NULL, &f);
    MPI_File_set_size(f, 0);
    dump1(pp, faces, nc, nv, nt, f);
    MPI_File_close(&f);
}

void rbc_dump(Particle *pp, int *faces, int nc, int nv, int nt, int id) {
    const char *fmt = DUMP_BASE "/r/%05d.ply";
    char f[BUFSIZ]; /* file name */
    sprintf(f, fmt, id);
    if (m::rank == 0) os::mkdir(DUMP_BASE "/r");
    dump2(pp, faces, nc, nv, nt, f);
}
