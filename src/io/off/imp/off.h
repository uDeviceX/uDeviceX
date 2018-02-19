#define SIZE 1024
static int emptyp(char *s) {
    for (;;) {
        if      (s[0] == '\0')                return 1;
        else if (s[0] != ' ' && s[0] != '\t') return 0;
        else s++;
    }
}
static int commentp(char *s) {
    return s[0] != '\0' && s[0] == '#';
}
static void unend(char *s) { /* rm \n */
    if (s[0] == '\0') return;
    while (s[1] != '\0') s++;
    if (s[0] == '\n') s[0] = '\0';
}
enum {OK, ERR};
static int line(FILE *f, /**/ char *s) { /* read line */
    for (;;) {
        if (fgets(s, SIZE, f) == NULL)  return ERR;
        unend(s);
        if (!emptyp(s) && !commentp(s)) break;
    }
    return OK;
}

static int header(FILE *f, char *s) {
    char key[SIZE];
    if (line(f, /**/ s) == ERR) return 0;
    sscanf(s, "%s", key);
    return same_str(key, "OFF");
}

static int sizes(FILE *f, char *s, /**/ int *nv, int *nt) {
    int r;
    if (line(f, /**/ s) == ERR) return 0;
    r = sscanf(s, "%d %d %*d", nv, nt);
    return r == 2;
}
static int vert(FILE *f, char *s, int n, float *rr) {
    enum {X, Y, Z};
    int i;
    float *r;
    for (i = 0; i < n; i++) {
        if (line(f, /**/ s) == ERR) return 0;
        r = &rr[3*i];
        if (sscanf(s, "%f %f %f", &r[X], &r[Y], &r[Z]) != 3) return 0;
    }
    return 1;
}
static int good(int4 *t, int n) {
    int x, y, z;
    x = t->x; y = t->y; z = t->z;
    return
        (0 <= x  && x < n) &&
        (0 <= y  && y < n) &&
        (0 <= z  && z < n) &&
        (x != y  && y != z);
}
static int tri(FILE *f, char *s, int nt, int nv, int4 *tt) {
    int i;
    int4 *t;
    int nvpt; /* number of vertices per face (triangle) */
    for (i = 0; i < nt; i++) {
        if (line(f, /**/ s) == ERR) return 0;
        t = &tt[i];
        if (sscanf(s, "%d %d %d %d", &nvpt, &t->x, &t->y, &t->z) != 4) {
            msg_print("expecting triangle:'%s'", s);
            return 0;
        }
        if (nvpt != 3) {
            msg_print("expecting triangle: '%s'", s);
            return 0;
        }
        if (!good(t, nv)) {
            msg_print("wrong triangle: '%s'", s);
            msg_print("nv = %d, nt = %d", nv, nt);
            return 0;
        }
    }
    return 1;
}
static void read_off(FILE *f, const char *path, /**/ OffRead *q) {
    int nv, nt;
    char s[SIZE];
    float *rr;
    int4  *tt;
    if (!header(f, s)) {
        msg_print("not an OFF file: '%s'", s);
        ERR("failed to read: '%s'", path);
    }
    if (!sizes(f, s, &nv, &nt)) {
        msg_print("wrong sizes: '%s'", s);
        ERR("failed to read: '%s'", path);
    }
    EMALLOC(3*nv, &rr);
    EMALLOC(  nt, &tt);
    if (!vert(f, s, nv, /**/ rr))
        ERR("failed to read vertices: '%s'", path);
    if (!tri(f, s, nt, nv, /**/ tt))
        ERR("failed to read triangles: '%s'", path);
    q->nv = nv; q->nt = nt;
    q->rr = rr; q->tt = tt;
}
