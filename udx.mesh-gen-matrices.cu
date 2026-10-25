// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/mesh/gen/matrices/imp.cpp */
#include <stdio.h>
#include <string.h>

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
/* begin src/coords/imp.h */
struct Coords;
struct Coords_v;
struct float3;
struct int3;

// tag::view[]
void coords_get_view(const Coords *c, Coords_v *v);
// end::view[]

// tag::domainsz[]
int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);
// end::domainsz[]

// tag::local2center[]
float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl, /**/ float3 *rc);
// end::local2center[]

// tag::center2local[]
float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc, /**/ float3 *rl);
// end::center2local[]

// tag::local2global[]
float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl, /**/ float3 *rg);
// end::local2global[]

// tag::global2local[]
float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);
// end::global2local[]

/* edges of the sub-domain in global coordinates */

// tag::bblocal[]
int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);
// end::bblocal[]

// tag::subdomainsz[]
int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);
// end::subdomainsz[]

// tag::int[]
/* rank predicates */
bool is_end(const Coords *c, int dir);

/* a string unique for a rank */
void coord_stamp(const Coords *c, /**/ char *s);

/* number of subdomains */
int coords_size(const Coords *c);
// end::int[]
/* end src/coords/imp.h */

/* begin src/mesh/gen/matrices/imp.h */
struct Matrices;
struct Coords;

// tag::interface[]
void matrices_read(const char *path, /**/ Matrices**); // <1>
void matrices_read_filter(const char *path, const Coords*, /**/ Matrices**); // <2>

void matrices_get(const Matrices*, int i, /**/ double**); // <3>
void matrices_get_r(const Matrices*, int i, /**/ double r[3]); // <4>
int  matrices_get_n(const Matrices*); // <5>
void matrices_log(const Matrices*);   // <6>

void matrices_fin(Matrices*);
// end::interface[]
/* end src/mesh/gen/matrices/imp.h */

/* begin src/mesh/gen/matrices/imp/type.h */
struct Matrix { double D[4*4]; };
struct Matrices {
    int n;
    Matrix *m;
};
/* end src/mesh/gen/matrices/imp/type.h */
/* begin src/mesh/gen/matrices/imp/main.h */
static const int MAX_N = 100000;

static void matrix2r(const double A[16], double r[3]) {
    enum {X, Y, Z};
    int i;
    i = 0;
    i++; i++; i++; r[X] = A[i++];
    i++; i++; i++; r[Y] = A[i++];
    i++; i++; i++; r[Z] = A[i++];
}

static int read_matrix(FILE *f, double A[16]) {
    int i;
    for (i = 0; i < 16; i++)
        if (fscanf(f, "%lf", &A[i]) != 1) return 0;
    return 1;
}

void matrices_read(const char *path, /**/ Matrices **pq) {
    int n;
    FILE *f;
    Matrices *q;
    EMALLOC(1, &q);
    EMALLOC(MAX_N, &q->m);
    msg_print("read '%s'", path);
    UC(efopen(path, "r", /**/ &f));
    n = 0;
    while (read_matrix(f, /**/ q->m[n].D)) {
        n++;
        if (n > MAX_N) ERR("n=%d > MAX_N=%d", n, MAX_N);
    }
    UC(efclose(f));
    q->n = n;
    *pq = q;
}

static int good(const Coords *c, const double A[16]) {
    enum {X, Y, Z};
    double r[3];
    matrix2r(A, /**/ r);
    return
        xlo(c) <= r[X] && r[X] < xhi(c) &&
        ylo(c) <= r[Y] && r[Y] < yhi(c) &&
        zlo(c) <= r[Z] && r[Z] < zhi(c);
}
void matrices_read_filter(const char *path, const Coords *c, /**/ Matrices **pq) {
    int m, n;
    FILE *f;
    Matrices *q;
    EMALLOC(1, &q);
    EMALLOC(MAX_N, &q->m);
    msg_print("read '%s'", path);
    UC(efopen(path, "r", /**/ &f));
    n = m = 0;
    while (read_matrix(f, /**/ q->m[n].D)) {
        m++;
        if (good(c, q->m[n].D)) n++;
        if (n > MAX_N) ERR("n=%d > MAX_N=%d", n, MAX_N);
    }
    msg_print("took %d/%d matrices", n, m);
    UC(efclose(f));
    q->n = n;
    *pq = q;
}

void matrices_get(const Matrices *q, int i, /**/ double **pq) {
    int n;
    n = q->n;
    if (i >= n) ERR("i=%d >= n=%d", i, n);
    *pq = q->m[i].D;
}

void matrices_get_r(const Matrices *q, int k, /**/ double r[3]) {
    enum {X, Y, Z};
    int n;
    double *A;
    n = q->n;
    if (k >= n) ERR("k=%d >= n=%d", k, n);
    A = q->m[k].D;
    matrix2r(A, /**/ r);
}

int matrices_get_n(const Matrices *q) { return q->n; }

static void log(double A[16]) {
    int i, j;
    double a, b, c, d;
    for (i = j = 0; j < 4; j++) {
        a = A[i++]; b = A[i++]; c = A[i++]; d = A[i++];
        msg_print("%g %g %g %g", a, b, c, d);
    }
}
void matrices_log(const Matrices *q) {
    int n, i;
    n = q->n;
    msg_print("<matrices_log");
    for (i = 0; i < n; i++) {
        if (i > 0) msg_print("");
        log(q->m[i].D);
    }
    msg_print(">matrices_log");
}

void matrices_fin(Matrices *q) { EFREE(q->m); EFREE(q); }
/* end src/mesh/gen/matrices/imp/main.h */
/* end src/mesh/gen/matrices/imp.cpp */

#line 1 "src/mesh/props/imp.cpp"
