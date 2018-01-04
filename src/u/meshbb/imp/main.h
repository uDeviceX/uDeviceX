/* max number of vertices and triangles */
#define NV 1000
#define NT 1000

struct MeshHst {
    int nf, nv;
    float *vert;
    int4  *faces;
};

struct MeshDev {
    int nf, nv;
    Particle *pp;
    int4  *faces;
};

static clist::Clist cells;
static clist::Map  mcells;
static meshbb::BBdata bbd;

static int    argc;
static char **argv;
/* left shift */
static void lshift() {
    argc--;
    if (argc < 1) {
        fprintf(stderr, "h5: not enough args\n");
        exit(2);
    }
}

enum {OK, END, FAIL};
int read_rv0(const char *s, float *r, float *v) {
    int n;
    enum {X, Y, Z};
    n = sscanf(s, "%f %f %f %f %f %f",
               &r[X], &r[Y], &r[Z],
               &v[X], &v[Y], &v[Z]);
    if (n != 6) return FAIL;
    return OK;
}
static int read_rv(float *r, float *v) {
    char s[BUFSIZ];
    if (fgets(s, BUFSIZ - 1, stdin) == NULL) return END;
    if (read_rv0(s, /**/ r, v) == FAIL) return FAIL;
    return OK;
}
static void write_particle(Particle p) {
    enum {X, Y, Z};
    msg_print("%g %g %g %g %g %g",
              p.r[X], p.r[Y], p.r[Z],
              p.v[X], p.v[Y], p.v[Z]);
}

static void read_off(const char *path, /**/ MeshHst *M) {
    int nf, nv;
    off_read_vert(path,  NV, /**/ &nv, M->vert);
    off_read_faces(path, NT, /**/ &nf, M->faces);
    M->nv = nv; M->nf = nf;
}

static void vert2pp(int n, float *pp0, /**/ Particle *pp) {
    enum {X, Y, Z};
    float *r0, *r, *v;
    Particle *p;
    int i;
    for (i = 0; i < n; i++) {
        p = &pp[i];
        r0 = &pp0[3*i];
        r = p->r; v = p->v;
        r[X] = r0[X]; r[Y] = r0[Y]; r[Z] = r0[Z];
        v[X] = v[Y] = v[Z] = 0;
    }
}

static void read_particles0(/**/ int *pn, Particle *pp) {
    enum {X, Y, Z};
    Particle p;
    float r[3], v[3];
    int rc, i;
    for (i = 0; /**/ ; i++) {
        rc = read_rv(r, v);
        if (rc == END) break;
        if (rc == FAIL) ERR("fail to read particle");
        p.r[X] = r[X]; p.r[Y] = r[Y]; p.r[Z] = r[Z];
        p.v[X] = v[X]; p.v[Y] = v[Y]; p.v[Z] = v[Z];
        pp[i] = p;
    }
    *pn = i;
}

static void read_particles(/**/ int *pn, Particle *d) {
    int n;
    Particle h[NV];
    read_particles0(/**/ &n, h);
    write_particle(h[0]);
    cH2D(d, h, n);
    *pn = n;
}

static void main0(int n, meshbb::BBdata bdb) {
    int i;
    int ncols[MAX_PART_NUM];
    float4 datacol[MAX_PART_NUM];

    cD2H(ncols,   bdb.ncols,   n);
    cD2H(datacol, bdb.datacol, n);

    for (i = 0; i < n; i++) {
        if (!ncols[i]) continue;
        msg_print("ncols[%d] = %d", i, ncols[i]    );
        msg_print("t    [%d] = %.16e", i, datacol[i].x);
        msg_print("u    [%d] = %.16e", i, datacol[i].y);
        msg_print("v    [%d] = %.16e", i, datacol[i].z);
    }
}

static void main1(MeshDev m, int n, Particle *pp, Force *ff, clist::Clist cells,
                  meshbb::BBdata bdb) {
    int ns;
    ns = 1;
    int3 L;
    L = make_int3(XS, YS, ZS);

    meshbb::ini(MAX_PART_NUM, &bbd);
    meshbb::reini(n, /**/ bbd);
    meshbb::find_collisions(ns, m.nf, m.nv, m.faces, m.pp,   L, cells.starts, cells.counts,   pp, ff, /**/ bbd);

    main0(n, bbd);

    meshbb::fin(&bbd);
}

static void main2(MeshDev m, int n, Particle *pp, Force *ff, clist::Clist cells) {
    meshbb::BBdata bbd;
    meshbb::ini(MAX_PART_NUM, /**/ &bbd);
    main1(m, n, pp, ff, cells, /**/ bbd);
    meshbb::fin(&bbd);
}

static void main3(MeshDev m, int n, Particle *pp, Force *ff, Particle *pp0) {
    clist::Clist cells;
    clist::Map  mcells;

    clist::ini(XS, YS, ZS, /**/ &cells);
    clist::ini_map(n, 2, &cells, /**/ &mcells);
    clist::build(n, n, pp, /**/ pp0, &cells, &mcells);

    main2(m, n, pp, ff, cells);

    clist::fin(&cells);
    clist::fin_map(&mcells);
}

static void main4(MeshDev m, int n, Particle* pp) {
    Particle *pp0;
    Force *ff;
    Dalloc(&pp0, n); Dalloc(&ff, n);
    main3(m, n, pp, ff, pp0);
    Dfree(pp0); Dfree(ff);
}

static void main5(MeshDev m) {
    enum {X, Y, Z};
    int n;
    Particle *pp;
    Dalloc(&pp, NV);

    read_particles(/**/ &n, pp);
    main4(m, n, pp);

    Dfree(pp);
}

static void main6(MeshHst h) {
    MeshDev d;
    Particle pp[NV];
    vert2pp(h.nv, h.vert, /**/ pp);

    Dalloc(&d.pp,    h.nv);
    Dalloc(&d.faces, h.nf);

    cH2D(d.pp,    pp,      h.nv);
    cH2D(d.faces, h.faces, h.nf);
    d.nv = h.nv; d.nf = h.nf;
    main5(d);

    Dfree(d.pp);
    Dfree(d.faces);
}

static void main7(const char *path) {
    float vert[3*NV];
    int4  faces[NT];
    MeshHst M;
    M.vert = vert; M.faces = faces;

    read_off(path, /**/ &M);
    main6(M);
}

static void main8() {
    const char *path;
    path = argv[argc - 1]; lshift();

    m::ini(&argc, &argv);
    main7(path);
    m::fin();
}

int main(int argc0, char **argv0) {
    msg_ini(0);
    argc = argc0;
    argv = argv0;
    main8();
}
