#define SIZE 1024

static void ini(OffRead **pq) {
    OffRead *p;
    UC(emalloc(sizeof(OffRead), (void**)&p));
    *pq = p;
}

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
static void read(FILE *f, const char *path, /**/ OffRead *q) {
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

void off_read_off(const char *path, OffRead **pq) {
    FILE *f;
    OffRead *q;
    UC(ini(&q));
    UC(efopen(path, "r", /**/ &f));
    read(f, path, /**/ q);
    UC(efclose(f));
    msg_print("read '%s'", path);
    *pq = q;
}

void off_fin(OffRead* q) { EFREE(q->rr); EFREE(q->tt); EFREE(q); }
int off_get_nv(OffRead *q) { return q->nv; }
int off_get_nt(OffRead *q) { return q->nt; }
static int amax(int *a, int n) {
    int i, m;
    if (n <= 0) ERR("amax called with size: %d\n", n);
    m = a[0];
    for (i = 1; i < n; i++)
        if (a[i] > m) m = a[i];
    return m;
}
int off_get_md(OffRead *q) {
    int *d;
    int i, m, nt, nv;
    int x, y, z;
    int4 *tt;
    nt = q->nt; nv = q->nv; tt = q->tt;

    if (nv == 0) {
        msg_print("off_get_md called for nv = 0");
        return 0;
    }
    UC(emalloc(nv*sizeof(d[0]), (void**)&d));
    for (i = 0; i < nv; i++) d[i] = 0;
    for (i = 0; i < nt; i++) {
        x = tt[i].x; y = tt[i].y; z = tt[i].z;
        d[x]++; d[y]++; d[z]++;
    }
    m = amax(d, nv);
    UC(efree(d));
    return m;
}

const float *off_get_vert(OffRead *q) { return q->rr; }
const int4  *off_get_tri(OffRead *q) { return q->tt; }
