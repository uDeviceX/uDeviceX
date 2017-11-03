/* max number of vertices and triangles */
#define NV 1000
#define NT 1000

static struct Mesh {
    int nf, nv;
    float vert[3*NV];
    int4  faces[NT];
} M;

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

static void read_off(const char *path) {
    M.nv = off::vert(path,  M.vert);
    M.nf = off::faces(path, M.faces);
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

    read_off(path);

    n = 1;
    ns = 1;
    nv = M.nv;
    nt = M.nf;
    tt = M.faces;

    cc = o::q.cells.counts;
    ss = o::q.cells.starts;

    Dalloc(ff);
    Dalloc(i_pp, MAX_PART_NUM);
    Dalloc(pp, MAX_PART_NUM);
    Dalloc(pp0, MAX_PART_NUM);
    
    clist::ini(XS, YS, ZS, /**/ &cells);
    clist::ini_map(2, &cells, /**/ &mcells);

    
    clist::build(n, n, pp, /**/ pp0, cells, mcells);
    meshbb::ini(MAX_PART_NUM, /**/ &bbd);

    while (read_point(r) != END) {
        meshbb::reini(n, /**/ bbd);
        meshbb::find_collisions(ns, nt, nv, tt, i_pp, L, ss, cc, pp, ff, /**/ bbd);

        write_point(r, inside);
    }
}

static void main1() {
    const char *path;
    path = argv[argc - 1]; lshift();

    m::ini(argc, argv);
    main0(path);
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0;
    argv = argv0;
    main1();
}
