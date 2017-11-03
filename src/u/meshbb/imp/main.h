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

static void read_particle0(const char *s, float *r) {
    enum {X, Y, Z};
    sscanf(s, "%f %f %f", &r[X], &r[Y], &r[Z]);
}

enum {OK, END, FAIL};
static int read_particle(float *r) {
    char s[BUFSIZ];
    if (fgets(s, BUFSIZ - 1, stdin) == NULL) return END;
    read_particle0(s, /**/ r);
    return OK;
}
static void write_particle(float *r, int inside) {
    enum {X, Y, Z};
    printf("%g %g %g %d\n", r[X], r[Y], r[Z], inside);
}

static void read_off(const char *path, /**/ MeshHst *M) {
    int nf, nv;
    nv = off::vert(path,  M->vert);
    nf = off::faces(path, M->faces);
    M->nv = nv; M->nf = nf;
}

static void main0(const char *path) {
    float r[3];
    int inside;
    int n, ns, nv, nt;
    int4 *tt;

    Particle *i_pp, *pp, *pp0;
    Force *ff;
    int *ss, *cc;

    int3 L = make_int3(XS, YS, ZS);

    n = 1;
    ns = 1;
    //    nv = M.nv;
    //    nt = M.nf;
    //    tt = M.faces;

    Dalloc(&ff, MAX_PART_NUM);
    Dalloc(&i_pp, MAX_PART_NUM);
    Dalloc(&pp, MAX_PART_NUM);
    Dalloc(&pp0, MAX_PART_NUM);

    clist::ini(XS, YS, ZS, /**/ &cells);
    clist::ini_map(2, &cells, /**/ &mcells);
    cc = cells.counts;
    ss = cells.starts;

    clist::build(n, n, pp, /**/ pp0, &cells, &mcells);
    meshbb::ini(MAX_PART_NUM, /**/ &bbd);

    while (read_particle(r) != END) {
        meshbb::reini(n, /**/ bbd);
        meshbb::find_collisions(ns, nt, nv, tt, i_pp, L, ss, cc, pp, ff, /**/ bbd);
        write_particle(r, inside);
    }
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

static void read_particles(/**/ int *pn, Particle *pp) {
    enum {X, Y, Z};
    Particle p;
    float r[3];
    int i;
    for (i = 0; /**/ ; i++) {
        if (read_particle(r) == END) break;
        p.r[X] = r[X]; p.r[Y] = r[Y]; p.r[Z] = r[Z];
        pp[i] = p;
    }
    *pn = i;
}

static void main1(MeshDev m) {
    int n;
    Particle pp[NV];
    read_particles(/**/ &n, pp);
}

static void main2(MeshHst h) {
    MeshDev d;
    Particle pp[NV];
    vert2pp(h.nv, h.vert, /**/ pp);

    Dalloc(&d.pp,    h.nv);
    Dalloc(&d.faces, h.nf);

    cH2D(d.pp,    pp,      h.nv);
    cH2D(d.faces, h.faces, h.nf);
    d.nv = h.nv; d.nf = h.nf;
    main1(d);
}

static void main3(const char *path) {
    float vert[3*NV];
    int4  faces[NT];
    MeshHst M;
    M.vert = vert; M.faces = faces;
    
    read_off(path, /**/ &M);
    main2(M);
}

static void main4() {
    const char *path;
    path = argv[argc - 1]; lshift();

    m::ini(argc, argv);
    main3(path);
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0;
    argv = argv0;
    main4();
}
