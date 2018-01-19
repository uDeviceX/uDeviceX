static const int NVP = 3; /* number of vertices per face */

static void copy_v(const Particle *f, /**/ Particle *t) {
    enum {X, Y, Z};
    t->v[X] = f->v[X];
    t->v[Y] = f->v[Y];
    t->v[Z] = f->v[Z];
}
static void shift(Coords c, const Particle *f, int n, /**/ Particle *t) {
    /* f, t: from, to
       see mesh/shift/     */
    int i;
    for (i = 0; i < n; i++) {
        shift0(c, f, t); copy_v(f, t); f++; t++;
    }
}

static void header(MPI_Comm cart, int nc0, int nv, int nt, write::File *f) {
    int nc; /* total number of cells */
    int sz = 0;
    char s[BUFSIZ] = {0};
    write::reduce(cart, nc0, &nc);
    if (m::is_master(cart))
        sz = sprintf(s,
                     "ply\n"
                     "format binary_little_endian 1.0\n"
                     "element vertex %d \n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property float u\nproperty float v\nproperty float w\n"
                     "element face  %d  \n"
                     "property list int int vertex_index\n"
                     "end_header\n", nv*nc, nt*nc);
    write::one(cart, s, sz, f);
}

static void vert(MPI_Comm cart,const Particle *pp, int nc, int nv, write::File *f) {
    int n;
    n = nc * nv;
    write::all(cart, pp, sizeof(Particle) * n, f);
}

static void wfaces0(MPI_Comm cart, int *buf, const int4 *faces, int nc, int nv, int nt, write::File *f) {
    /* write faces */
    int c, t, b;  /* cell, triangle, buffer index */
    int n, shift;
    n = nc * nv;
    write::shift(cart, n, &shift);

    int4 tri;
    for(b = c = 0; c < nc; ++c)
        for(t = 0; t < nt; ++t) {
            tri = faces[t];
            buf[b++] = NVP;
            buf[b++] = shift + nv*c + tri.x;
            buf[b++] = shift + nv*c + tri.y;
            buf[b++] = shift + nv*c + tri.z;
        }
    write::all(cart, buf, b*sizeof(buf[0]), f);
}

static void wfaces(MPI_Comm cart, const int4 *faces, int nc, int nv, int nt, write::File *f) {
    int *buf; /* buffer for faces */
    int sz;
    sz = (1 + NVP) * nc * nt * sizeof(int);
    UC(emalloc(sz, (void**) &buf));
    wfaces0(cart, buf, faces, nc, nv, nt, f);
    free(buf);
}

static void main0(MPI_Comm cart, const Particle *pp, const int4 *faces,
                  int nc, int nv, int nt, write::File *f) {
    header(cart, nc,        nv, nt, f);
    vert(cart, pp,      nc, nv,     f);
    wfaces(cart, faces, nc, nv, nt, f);
}

static void main1(MPI_Comm cart, Coords c, const Particle *pp, const int4 *faces, int nc, int nv, int nt, write::File *f) {
    int sz, n;
    Particle *pp0;
    n = nc * nv;
    sz = n*sizeof(Particle);
    UC(emalloc(sz, (void**) &pp0));
    shift(c, pp, n, /**/ pp0); /* copy-shift to global coordinates */
    main0(cart, pp0, faces, nc, nv, nt, f);
    free(pp0);
}

static void main(MPI_Comm cart, Coords coords, const Particle *pp, const int4 *faces, int nc, int nv, int nt, const char *fn) {
    write::File *f;
    write::fopen(cart, fn, /**/ &f);
    main1(cart, coords, pp, faces, nc, nv, nt, f);
    write::fclose(f);
}

void rbc(MPI_Comm cart, Coords coords, const Particle *pp, const int4 *faces, int nc, int nv, int nt, int id) {
    const char *fmt = DUMP_BASE "/r/%05d.ply";
    char f[BUFSIZ]; /* file name */
    sprintf(f, fmt, id);
    if (m::is_master(cart)) UC(os_mkdir(DUMP_BASE "/r"));
    main(cart, coords, pp, faces, nc, nv, nt, f);
}

void rig(MPI_Comm cart, Coords coords, const Particle *pp, const int4 *faces, int nc, int nv, int nt, int id) {
    const char *fmt = DUMP_BASE "/s/%05d.ply";
    char f[BUFSIZ]; /* file name */
    sprintf(f, fmt, id);
    if (m::is_master(cart)) UC(os_mkdir(DUMP_BASE "/s"));
    main(cart, coords, pp, faces, nc, nv, nt, f);
}
