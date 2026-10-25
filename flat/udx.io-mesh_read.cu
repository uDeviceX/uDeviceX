// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/io/mesh_read/imp.cpp */
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

/* begin src/utils/error.h */
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)

/* udx check */
#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

/* [B]o[p] [c]heck */
#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
/* end src/utils/error.h */
/* begin src/utils/imp.h */
// tag::mem[]
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>
// end::mem[]

// tag::macros[]
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))
// end::macros[]

// tag::stdio[]
void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);
// end::stdio[]

// tag::tools[]
bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
// end::tools[]
/* end src/utils/imp.h */

/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
/* begin src/io/mesh_read/edg/imp.h */
struct Edg;

void e_ini(int md, int nv, /**/ Edg**);
void e_fin(Edg*);

void e_set(Edg*, int i, int j, int val);
int  e_get(Edg*, int i, int j);
int  e_valid(Edg*, int i, int j);
/* end src/io/mesh_read/edg/imp.h */

/* begin src/io/mesh_read/imp.h */
struct int4;
struct MeshRead;

// tag::interface[]
void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);
// end::interface[]
/* end src/io/mesh_read/imp.h */

/* begin src/io/mesh_read/imp/type.h */
struct MeshRead {
    int nv, nt, nd; /* vertices, triangles, dihidrals */
    float *rr; /* [x y z] [x y z] */
    int4  *tt; /* triangles: [v0 v1 v2 dummy] */
    int4  *dd; /* dihidrals: [v0 (v1 v2) v3]  */

    int dd_computed;
};
/* end src/io/mesh_read/imp/type.h */
/* begin src/io/mesh_read/imp/off.h */
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
static void read_off(FILE *f, const char *path, /**/ MeshRead *q) {
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
/* end src/io/mesh_read/imp/off.h */
/* begin src/io/mesh_read/imp/ply.h */
static bool prop(const char *pname, const char *str) {
    const int l1 = strlen(pname);
    const int l2 = strlen(str);

    if (l1 > l2) return false;
    for (int i = 0; i < l1; ++i) if (pname[i] != str[i]) return false;
    return true;
}

static void read_ply(FILE *f, const char *fname, MeshRead *q) {
    int nv, nt;
    int4 *tt;
    float *rr;
    int l = 0;
#define BUFSIZE 256 // max number of chars per line
#define MAXLINES 64 // max number of line for header

    // https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html#Stringizing
#define xstr(s) str(s)
#define str(s) #s

    nv = nt = -1;
    while (l++ < MAXLINES) {
        char cbuf[BUFSIZE + 1] = {0}; // + 1 for \0

        const int checker = fscanf(f, " %[^\n]" xstr(BUFSIZE) "c", cbuf);
        if (checker != 1)
            ERR("Fail to read '%s'", fname);

        int ibuf;
        if    (sscanf(cbuf, "element vertex %d", &ibuf) == 1) nv = ibuf;
        else if (sscanf(cbuf, "element face %d", &ibuf) == 1) nt = ibuf;
        else if (prop("end_header", cbuf)) break;
    }
    if (l >= MAXLINES || nt == -1 || nv == -1)
        ERR("Fail to read '%s': did not catch end_header", fname);
    EMALLOC(  nt, &tt);
    EMALLOC(3*nv, &rr);
    for (int i = 0; i < nv; ++i)
        fscanf(f, "%f %f %f\n",
               rr + 3*i + 0, rr + 3*i + 1, rr + 3*i + 2);
    int4 t; t.z = 0;
    for (int i = 0; i < nt; ++i) {
        fscanf(f, "%*d %d %d %d\n", &t.x, &t.y, &t.z);
        tt[i] = t;
    }
    q->nv = nv; q->nt = nt;
    q->rr = rr; q->tt = tt;
}
/* end src/io/mesh_read/imp/ply.h */
/* begin src/io/mesh_read/imp/main.h */
struct Q { int4 *d; }; /* queue */

static void q_push(Q *q, int i0, int i1, int i2, int i3) {
    int4 *d;
    d = q->d;
    d->x = i0; d->y = i1; d->z = i2; d->w = i3;
    (q->d)++;
}

static void reg(int i1, int i2, Edg *nxt, Edg *seen, /**/ Q *q) {
    int i0, i3;
    if (e_valid(seen, i1, i2)) return;
    if (e_valid(seen, i2, i1)) return;
    i0 = e_get(nxt, i1, i2);
    i3 = e_get(nxt, i2, i1); /* previous */
    q_push(q, i0, i1, i2, i3);
    UC(e_set(seen, i1, i2, 1));
}

static void ini_dd0(int nt, const int4 *tt, int nv, int md, /**/ int4 *dd) {
    Q q;
    Edg *nxt, *seen;
    int i, i0, i1, i2;
    q.d = dd;

    UC(e_ini(md, nv, &nxt));
    UC(e_ini(md, nv, &seen));

    for (i = 0; i < nt; i++) { /* save who is next for every edge */
        i0 = tt[i].x; i1 = tt[i].y; i2 = tt[i].z;
        e_set(nxt, i0, i1, i2);
        e_set(nxt, i1, i2, i0);
        e_set(nxt, i2, i0, i1);
    }

    for (i = 0; i < nt; i++) { /* register dihidrals */
        i0 = tt[i].x; i1 = tt[i].y; i2 = tt[i].z;
        reg(i0, i1, nxt, /**/ seen, &q);
        reg(i1, i2, nxt, /**/ seen, &q);
        reg(i2, i0, nxt, /**/ seen, &q);
    }

    UC(e_fin(nxt); e_fin(seen));
}

static void ini_dd(/**/ MeshRead *q) {
    int nt, nv, nd, md;
    const int4 *tt;
    nt = mesh_read_get_nt(q);
    nv = mesh_read_get_nv(q);
    nd = mesh_read_get_ne(q);
    md = mesh_read_get_md(q);
    tt = mesh_read_get_tri(q);
    q->nd = nd;
    EMALLOC(nd, &q->dd);
    UC(ini_dd0(nt, tt, nv, md, /**/ q->dd));
}

void mesh_read_ini_off(const char *path, MeshRead **pq) {
    FILE *f;
    MeshRead *q;
    EMALLOC(1, &q);
    UC(efopen(path, "r", /**/ &f));
    UC(read_off(f, path, /**/ q));
    UC(efclose(f));
    msg_print("read off '%s'", path);
    q->dd_computed = 0;
    q->dd = NULL;
    *pq = q;
}

void mesh_read_ini_ply(const char *path, MeshRead **pq) {
    FILE *f;
    MeshRead *q;
    EMALLOC(1, &q);
    UC(efopen(path, "r", /**/ &f));
    UC(read_ply(f, path, /**/ q));
    UC(efclose(f));
    msg_print("read ply '%s'", path);
    q->dd = NULL;
    *pq = q;
}

void mesh_read_fin(MeshRead* q) {
    EFREE(q->rr);
    EFREE(q->tt);
    if (q->dd) EFREE(q->dd);
    EFREE(q);
}

int mesh_read_get_nv(const MeshRead *q) { return q->nv; }
int mesh_read_get_nt(const MeshRead *q) { return q->nt; }
int mesh_read_get_ne(const MeshRead *q) {
    int nt;
    nt = q->nt;
    if (nt % 2 != 0) ERR("nt=%d % 2 != 0", nt);
    return 3 * nt / 2;
}

static int amax(int *a, int n) {
    int i, m;
    if (n <= 0) ERR("amax called with size: %d\n", n);
    m = a[0];
    for (i = 1; i < n; i++)
        if (a[i] > m) m = a[i];
    return m;
}

int mesh_read_get_md(const MeshRead *q) {
    int *d;
    int i, m, nt, nv;
    int x, y, z;
    int4 *tt;
    nt = q->nt; nv = q->nv; tt = q->tt;

    if (nv == 0) {
        msg_print("mesh_get_md called for nv = 0");
        return 0;
    }
    EMALLOC(nv, &d);
    for (i = 0; i < nv; i++) d[i] = 0;
    for (i = 0; i < nt; i++) {
        x = tt[i].x; y = tt[i].y; z = tt[i].z;
        d[x]++; d[y]++; d[z]++;
    }
    m = amax(d, nv);
    EFREE(d);
    return m;
}

const float *mesh_read_get_vert(const MeshRead *q) { return q->rr; }
const int4  *mesh_read_get_tri(const MeshRead *q) { return q->tt; }
const int4  *mesh_read_get_dih(MeshRead *q) {
    if (!q->dd_computed) {
        UC(ini_dd(q));
        q->dd_computed = 1;
    }
    return q->dd;
}
/* end src/io/mesh_read/imp/main.h */
/* end src/io/mesh_read/imp.cpp */

#line 1 "src/io/point/imp.cpp"
