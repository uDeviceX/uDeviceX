// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/wall/sdf/tform/imp.cpp */
#include <stdio.h>
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
/* begin src/math/tform/imp.h */
struct Tform;
struct Tform_v;

// tag::ini[]
void tform_ini(Tform**);
void tform_fin(Tform*);
// end::ini[]

// tag::def[]
void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3], /**/ Tform*); // <1>
void tform_chain(const Tform *A, const Tform *B, /**/ Tform *C); // <2>
void tform_to_grid(const float lo[3], const float hi[3], const int n[3], /**/ Tform*); // <3>
void tform_from_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform*); // <4>
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3],
                     Tform*); // <5>
// end::def[]


// tag::convert[]
void tform_convert(const Tform*, const float a0[3], /**/ float a1[3]);
// end::convert[]

// tag::view[]
void tform_to_view(const Tform*, /**/ Tform_v*);
// end::view[]

// tag::log[]
void tform_log(Tform*);
void tform_dump(Tform*, FILE*);
// end::log[]
/* end src/math/tform/imp.h */

/* begin src/wall/sdf/tform/imp.h */
struct Tform;
struct Coords;

// tag::int[]
/* T: texture size, N: sdf file size, M: wall margin */
void tex2sdf_ini(const Coords*, const int T[3], const int N[3], const int M[3], /**/ Tform*); // <1>
void out2sdf_ini(const Coords*, const int N[3], /**/ Tform*); // <2>
void sub2tex_ini(const Coords*, const int T[3], const int M[3], /**/ Tform*); // <3>
// end::int[]

/* end src/wall/sdf/tform/imp.h */

struct TGrid {
    float lo[3], hi[3];
    int n[3];
};

enum {OK, BAD};
static int goodp(const int n[3]) {
    enum {X, Y, Z};
    const int H = 1000000;
    int cl, ch;
    cl = n[X] > 0 && n[Y] > 0 && n[Z] > 0;
    ch = n[X] < H && n[Y] < H && n[Z] < H;
    return (cl && ch) ? OK : BAD;
}


static void grid2grid(TGrid *from, TGrid *to, /**/ Tform *t) {
    tform_grid2grid(from->lo, from->hi, from->n,
                    to->lo, to->hi,   to->n, /**/ t);
}

static void sdf_ini(const Coords *c, const int N[3], /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = t->lo[Y] = t->lo[Z] = 0;
    t->hi[X] = xdomain(c);
    t->hi[Y] = ydomain(c);
    t->hi[Z] = zdomain(c);

    t->n[X] = N[X]; t->n[Y] = N[Y]; t->n[Z] = N[Z];
}

static void tex_ini(const Coords *c, const int T[3], const int M[3], /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = xlo(c) - M[X];
    t->lo[Y] = ylo(c) - M[Y];
    t->lo[Z] = zlo(c) - M[Z];
    t->hi[X] = xhi(c) + M[X];
    t->hi[Y] = yhi(c) + M[Y];
    t->hi[Z] = zhi(c) + M[Z];

    t->n[X] = T[X]; t->n[Y] = T[Y]; t->n[Z] = T[Z];
}

static void out_ini(const Coords *c, /**/ TGrid *t) {
    enum {X, Y, Z};
    t->lo[X] = xlo(c);
    t->lo[Y] = ylo(c);
    t->lo[Z] = zlo(c);

    t->hi[X] = xhi(c);
    t->hi[Y] = yhi(c);
    t->hi[Z] = zhi(c);

    t->n[X] = xs(c);
    t->n[Y] = ys(c);
    t->n[Z] = zs(c);
}

static void sub_ini(const Coords *c, /**/ Tform *t) {
    enum {X, Y, Z};
    float a0[3], a1[3], b0[3], b1[3];
    a0[X] = -xs(c)/2; a0[Y] = -ys(c)/2; a0[Z] = -zs(c)/2;
    b0[X] =  xs(c)/2; b0[Y] =  ys(c)/2; b0[Z] =  zs(c)/2;

    a1[X] = xlo(c); a1[Y] = ylo(c); a1[Z] = zlo(c);
    b1[X] = xhi(c); b1[Y] = yhi(c); b1[Z] = zhi(c);
    UC(tform_vector(a0, a1,    b0, b1, /**/ t));
}

void tex2sdf_ini(const Coords *c,
                 const int T[3], const int N[3], const int M[3],
                 /**/ Tform *t) {
    enum {X, Y, Z};
    TGrid tex, sdf;

    if (goodp(T) == BAD) ERR("bad T = [%d %d %d]", T[X], T[Y], T[Z]);
    if (goodp(N) == BAD) ERR("bad N = [%d %d %d]", N[X], N[Y], N[Z]);
    if (goodp(M) == BAD) ERR("bad M = [%d %d %d]", M[X], M[Y], M[Z]);

    tex_ini(c, T, M, /**/ &tex);
    sdf_ini(c, N, /**/ &sdf);

    UC(grid2grid(&tex, &sdf, /**/ t));
}

void out2sdf_ini(const Coords *c, const int N[3], /**/ Tform* t) {
    enum {X, Y, Z};
    TGrid sub, sdf;
    if (goodp(N) == BAD) ERR("bad N = [%d %d %d]", N[X], N[Y], N[Z]);
    out_ini(c, /**/ &sub);
    sdf_ini(c, N, /**/ &sdf);
    UC(grid2grid(&sub, &sdf, /**/ t));
}

void sub2tex_ini(const Coords *c, const int T[3], const int M[3], /**/ Tform *t) {
    enum {X, Y, Z, D};
    float lo[D], hi[D];

    if (goodp(T) == BAD) ERR("bad T = [%d %d %d]", T[X], T[Y], T[Z]);
    if (goodp(M) == BAD) ERR("bad M = [%d %d %d]", M[X], M[Y], M[Z]);

    lo[X] = -xs(c)/2 - M[X];
    lo[Y] = -ys(c)/2 - M[Y];
    lo[Z] = -zs(c)/2 - M[Z];

    hi[X] = xs(c)/2 + M[X];
    hi[Y] = ys(c)/2 + M[Y];
    hi[Z] = zs(c)/2 + M[Z];

    tform_to_grid(lo, hi, T, /**/ t);
}
/* end src/wall/sdf/tform/imp.cpp */

#line 1 "src/wall/wvel/conf.cpp"
