// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/rbc/shape/imp.cpp */
#include <stdio.h>
#include <math.h>

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

/* begin src/rbc/adj/imp.h */
struct Adj;
struct Adj_v;
struct AdjMap;
struct int4;

// tag::interface[]
void adj_ini(int md, int nt, int nv, const int4 *tt, /**/ Adj**);
void adj_fin(Adj*);
int  adj_get_map(int i, const Adj*, /**/ AdjMap *m);
int  adj_get_max(const Adj*);
void adj_get_anti(const Adj*, /**/ int *anti);

void adj_view_ini(const Adj*, Adj_v**);
void adj_view_fin(Adj_v*);
// end::interface[]
/* end src/rbc/adj/imp.h */
#ifndef UDX_RBC_ADJ_MAP_TYPE
#define UDX_RBC_ADJ_MAP_TYPE
/* begin src/rbc/adj/type/common.h */
// tag::interface[]
struct AdjMap {
    int i0, i1, i2, i3, i4;
    int rbc;
};
// end::interface[]
/* end src/rbc/adj/type/common.h */
#endif
/* begin src/math/tri/imp.h */
namespace tri_hst {
// tag::interface[]
double kahan_area0(double a, double b, double c); // <1>
double kahan_area(const double a[3], const double b[3], const double c[3]); // <2>
void   ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]); // <3>
double shewchuk_area(const double a[3], const double b[3], const double c[3]); // <4>
double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]); // <5>
void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                   double *x, double *y); // <6>
// end::interface[]
}
/* end src/math/tri/imp.h */

/* begin src/rbc/shape/imp.h */
struct RbcShape;
struct MeshRead;
struct Adj;

/* rr: vertices: x0 y0 z0   x1 y1 z1 ... */

// tag::mem[]
void rbc_shape_ini(const Adj*, const float *rr, /**/ RbcShape**);
void rbc_shape_fin(/**/ RbcShape*);
// end::mem[]

// tag::int[]
void rbc_shape_edg (RbcShape*, /**/ float**); // <1>
void rbc_shape_area(RbcShape*, /**/ float**); // <2>

void rbc_shape_total_area(RbcShape*, /**/ float*);    // <3>
void rbc_shape_total_volume(RbcShape*, /**/ float*);  // <4>
// end::int[]
/* end src/rbc/shape/imp.h */

/* begin src/rbc/shape/imp/type.h */
struct RbcShape {
    float *edg, *area;
    float A, V;
};
/* end src/rbc/shape/imp/type.h */
/* begin src/rbc/shape/imp/main.h */
static void diff(const float *a, const float *b, /**/ float *c) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
static double vabs(float *a) {
    enum {X, Y, Z};
    double r;
    r = a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z];
    return sqrt(r);
}
static void compute_edg(const Adj *adj, const float *rr, /**/ float *o) {
    int i, valid, n;
    const float *r0, *r1;
    float r01[3];
    float a;
    AdjMap m;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        r0 = &rr[3*m.i0]; r1 = &rr[3*m.i1];
        diff(r0, r1, /**/ r01);
        a = vabs(r01);
        o[i] = a;
    }
}

static void copy(const float f[3], double d[3]) {
    enum {X, Y, Z};
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double area(const float af[3], const float bf[3], const float cf[3]) {
    double ad[3], bd[3], cd[3];
    copy(af, ad); copy(bf, bd); copy(cf, cd);
    return tri_hst::kahan_area(ad, bd, cd);
}
static void compute_area(const Adj *adj, const float *rr, /**/ float *o) {
    int i, i0, i1, i2, valid, n;
    const float *r0, *r1, *r2;
    AdjMap m;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1; i2 = m.i2;
        r0 = &rr[3*i0]; r1 = &rr[3*i1]; r2 = &rr[3*i2];
        o[i] = area(r0, r1, r2);
    }
}

static void compute_total_volume(const Adj *adj, const float *rr, /**/ float *pV) {
    enum {X, Y, Z};
    int i, i0, i1, i2, valid, n;
    const float *r0, *r1, *r2;
    float x0, y0, z0, x1, y1, z1, x2, y2, z2;
    AdjMap m;
    double V;
    V = 0;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1; i2 = m.i2;
        r0 = &rr[3*i0]; r1 = &rr[3*i1]; r2 = &rr[3*i2];
        x0 = r0[X]; y0 = r0[Y]; z0 = r0[Z];
        x1 = r1[X]; y1 = r1[Y]; z1 = r1[Z];
        x2 = r2[X]; y2 = r2[Y]; z2 = r2[Z];
        V += (x0*y1-x1*y0)*z2+x2*(y0*z1-y1*z0)+y2*(x1*z0-x0*z1);
    }
    V /= 6; /* from the formula */
    V /= 3; /* every triangle was visited three times */
    msg_print("V: %g", V);
    *pV = V;
}

static void compute_total_area(const Adj *adj, const float* area, /**/ float *pA) {
    int n, i, valid;
    AdjMap m;
    double A;

    A = 0;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        A += area[i];
    }
    A /= 3; /* every triangle was visited three times */
    *pA = A;
}

void rbc_shape_ini(const Adj *adj, const float *rr, /**/ RbcShape **pq) {
    int n;
    RbcShape *q;
    n = adj_get_max(adj);
    EMALLOC(1, &q);
    EMALLOC(n, &q->edg);
    EMALLOC(n, &q->area);

    compute_edg(adj, rr, /**/ q->edg);
    compute_area(adj, rr, /**/ q->area);
    compute_total_area(adj, q->area, /**/ &q->A);
    compute_total_volume(adj, rr, /**/ &q->V);

    msg_print("A: %g", q->A);

    *pq = q;
}

void rbc_shape_fin(RbcShape *q) {
    EFREE(q->edg);
    EFREE(q->area);
    EFREE(q);
}

void rbc_shape_edg  (RbcShape *q, /**/ float** pe) { *pe = q->edg; }
void rbc_shape_area (RbcShape *q, /**/ float** pe) { *pe = q->area; }

void rbc_shape_total_area(RbcShape *q, /**/ float *pe)   { *pe = q->A; }
void rbc_shape_total_volume(RbcShape *q, /**/ float *pe) {
    ERR("rbc_shape_total_volume is not implimented");
    *pe = q->V;
}
/* end src/rbc/shape/imp/main.h */
/* end src/rbc/shape/imp.cpp */

#line 1 "src/rbc/stretch/imp.cu"
