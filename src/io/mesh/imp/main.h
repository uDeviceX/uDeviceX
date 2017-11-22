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

static void header(int nc0, int nv, int nt, write::File *f) {
    int nc; /* total number of cells */
    int sz = 0;
    char s[BUFSIZ] = {0};
    write::reduce(nc0, &nc);
    if (write::rootp())
        sz = sprintf(s,
                     "ply\n"
                     "format binary_little_endian 1.0\n"
                     "element vertex %d \n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property float u\nproperty float v\nproperty float w\n"
                     "element face  %d  \n"
                     "property list int int vertex_index\n"
                     "end_header\n", nv*nc, nt*nc);
    write::one(s, sz, f);
}

static void vert(const Particle *pp, int nc, int nv, write::File *f) {
    int n;
    n = nc * nv;
    write::all(pp, sizeof(Particle) * n, f);
}

static void wfaces0(int *buf, const int4 *faces, int nc, int nv, int nt, write::File *f) {
    /* write faces */
    int c, t, b;  /* cell, triangle, buffer index */
    int n, shift;
    n = nc * nv;
    write::shift(n, &shift);

    int4 tri;
    for(b = c = 0; c < nc; ++c)
        for(t = 0; t < nt; ++t) {
            tri = faces[t];
            buf[b++] = NVP;
            buf[b++] = shift + nv*c + tri.x;
            buf[b++] = shift + nv*c + tri.y;
            buf[b++] = shift + nv*c + tri.z;
        }
    write::all(buf, b*sizeof(buf[0]), f);
}

static void wfaces(const int4 *faces, int nc, int nv, int nt, write::File *f) {
    int *buf; /* buffer for faces */
    int sz;
    sz = (1 + NVP) * nc * nt * sizeof(int);
    UC(emalloc(sz, (void**) &buf));
    wfaces0(buf, faces, nc, nv, nt, f);
    free(buf);
}

static void main0(const Particle *pp, const int4 *faces,
                  int nc, int nv, int nt, write::File *f) {
    header(nc,        nv, nt, f);
    vert(pp,      nc, nv,     f);
    wfaces(faces, nc, nv, nt, f);
}

static void main1(const Particle *pp, const int4 *faces, int nc, int nv, int nt, write::File *f) {
    int sz, n;
    Particle *pp0;
    n = nc * nv;
    sz = n*sizeof(Particle);
    UC(emalloc(sz, (void**) &pp0));
    shift(pp, n, /**/ pp0); /* copy-shift to global coordinates */
    main0(pp0, faces, nc, nv, nt, f);
    free(pp0);
}

void main(const Particle *pp, const int4 *faces, int nc, int nv, int nt, const char *fn) {
    write::File *f;
    write::fopen(fn, /**/ &f);
    main1(pp, faces, nc, nv, nt, f);
    write::fclose(f);
}

void rbc(const Particle *pp, const int4 *faces, int nc, int nv, int nt, int id) {
    const char *fmt = DUMP_BASE "/r/%05d.ply";
    char f[BUFSIZ]; /* file name */
    sprintf(f, fmt, id);
    if (write::rootp()) os::mkdir(DUMP_BASE "/r");
    main(pp, faces, nc, nv, nt, f);
}

void rig(const Particle *pp, const int4 *faces, int nc, int nv, int nt, int id) {
    const char *fmt = DUMP_BASE "/s/%05d.ply";
    char f[BUFSIZ]; /* file name */
    sprintf(f, fmt, id);
    if (write::rootp()) os::mkdir(DUMP_BASE "/s");
    main(pp, faces, nc, nv, nt, f);
}
