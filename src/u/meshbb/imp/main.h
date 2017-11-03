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

static void read_point0(const char *s, float *r) {
    enum {X, Y, Z};
    sscanf(s, "%f %f %f", &r[X], &r[Y], &r[Z]);
}

enum {OK, END, FAIL};
static int read_point(float *r) {
    char s[BUFSIZ];
    if (fgets(s, BUFSIZ - 1, stdin) == NULL) return END;
    read_point0(s, /**/ r);
    return OK;
}
static void write_point(float *r, int inside) {
    enum {X, Y, Z};
    printf("%g %g %g %d\n", r[X], r[Y], r[Z], inside);
}

static void read_off(const char *path, /**/ MeshHst *M) {
    int nf, nv;
    float vert[3*NV];
    int4  faces[NT];

    nv = off::vert(path,  vert);
    nf = off::faces(path, faces);

    M->nv = nv; M->nf = nf;
    M->vert = vert; M->faces = faces;
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

    while (read_point(r) != END) {
        meshbb::reini(n, /**/ bbd);
        meshbb::find_collisions(ns, nt, nv, tt, i_pp, L, ss, cc, pp, ff, /**/ bbd);

        write_point(r, inside);
    }
}

static void main1(MeshHst h) {
    MeshDev d;
    int nf, nv, i;
    Particle pp[NV];
    nv = h.nv; nf = h.nf;

    Dalloc(&d.pp,    nv);
    Dalloc(&d.faces, nf);
}

static void main2(const char *path) {
    MeshHst M;
    read_off(path, &M);
    main1(M);
}

static void main3() {
    const char *path;
    path = argv[argc - 1]; lshift();

    m::ini(argc, argv);
    main2(path);
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0;
    argv = argv0;
    main3();
}
