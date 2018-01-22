/* max number of vertices and triangles */
#define NV 1000
#define NT 1000

struct Mesh {
    int nf, nv;
    float vert[3*NV];
    int4  faces[NT];
} M;

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
    off_read_vert(path,  NV,  /**/ &M.nv, M.vert);
    off_read_faces(path, NT,  /**/ &M.nf, M.faces);
}

static void main0(const char *path) {
    float r[3];
    int inside;
    read_off(path);
    while (read_point(r) != END) {
        inside = collision_inside_1p(r, M.vert, M.faces, M.nf);
        write_point(r, inside);
    }
}

static void main1() {
    const char *path;
    path = argv[argc - 1]; lshift();
    
    m::ini(&argc, &argv);
    main0(path);
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0;
    argv = argv0;
    main1();
}
