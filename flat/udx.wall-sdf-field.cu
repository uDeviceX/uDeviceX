// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/wall/sdf/field/imp.cpp */
#include <mpi.h>
#include <math.h>
#include <vector_types.h>

/* begin src/inc/conf.h */
/* [k]ernel [l]aunch macros */
#if !defined(KL_RELEASE)    && !defined(KL_NONE)   &&   \
    !defined(KL_SYNC)
#define KL_RELEASE
#endif

/* [c]uda [c]heck macro */
#if !defined(CC_RELEASE) && !defined(CC_SYNC)
  #define CC_RELEASE
#endif

/* who plays as device? */
#if !defined(DEV_CUDA) && !defined(DEV_CPU)
  #define DEV_CUDA
#endif

/* cuda memory allocation calls */
#if !defined(CUDA_ALLOC_RELEASE) && !defined(CUDA_ALLOC_DEBUG)
  #define CUDA_ALLOC_RELEASE
#endif

/* compute rbc force in double or float */
#if !defined(RBC_DOUBLE) && !defined(RBC_FLOAT)
  #define RBC_DOUBLE
#endif

/* what to do if r > lmax for a spring? */
#if !defined(RBC_SPRING_IGNORE) && !defined(RBC_SPRING_FAIL)
  #define RBC_SPRING_IGNORE
#endif

#ifndef DUMP_BASE
#define DUMP_BASE "."
#endif
/* end src/inc/conf.h */


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
/* begin src/io/grid/imp.h */
// tag::int[]
void grid_write(int3 subGrid, int3 subDomain, MPI_Comm, const char *path, int ncmp, const float **data, const char **names);
// end::int[]
/* end src/io/grid/imp.h */
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

/* begin src/wall/sdf/field/imp.h */
struct Coords;
struct Tform;
struct Field;

// tag::mem[]
void field_ini(const char *path, /**/ Field**); // <1>
void field_fin(Field*);                         // <2>
// end::mem[]

// tag::int[]
void field_sample(const Field*, const Tform*, const int N[3], /**/ Field**); // <1>
// end::int[]

// tag::get[]
void field_size(const Field*, /**/ int N[3]);       // <1>
void field_extend(const Field*, /**/ float ext[3]); // <2>
void field_data(const Field*, /**/ float**);        // <3>
// end::get[]

// tag::int[]
void field_dump(const Field*, const Coords*, MPI_Comm cart); // <2>
void field_scale(Field*, float scale); // <3>
// end::int[]
/* end src/wall/sdf/field/imp.h */

/* begin src/wall/sdf/field/imp/type.h */
struct Field {  /* size, extension, and data */
    int N[3];
    float ext[3];
    float *D;
};
/* end src/wall/sdf/field/imp/type.h */
/* begin src/wall/sdf/field/imp/ini.h */
static void read_extents(FILE *f, /**/ float *x, float *y, float *z) {
    int r;
    char s[2048];
    UC(efgets(s, sizeof(s), f));
    r = sscanf(s, "%f %f %f", x, y, z);
    if (r != 3)
        ERR("expect three floats, got '%s'", s);
}

static void read_grid_size(FILE *f, /**/ int *x, int *y, int *z) {
    int r;
    char s[2048];
    UC(efgets(s, sizeof(s), f));
    r = sscanf(s, "%d %d %d", x, y, z);
    if (r != 3)
        ERR("expect three integers, got '%s'", s);
}

void field_ini(const char *path, /**/ Field **pq) {
    enum {X, Y, Z};
    Field *q;
    int n, *N;
    float *ext, *D;
    FILE *f;

    EMALLOC(1, &q);
    N = q->N; ext = q->ext;
    msg_print("reading '%s'", path);
    UC(efopen(path, "r", /**/ &f));

    read_extents  (f, /**/ &ext[X], &ext[Y], &ext[Z]);
    read_grid_size(f, /**/ &N[X],     &N[Y],   &N[Z]);

    n = N[X]*N[Y]*N[Z];
    EMALLOC(n, &q->D);
    D = q->D;
    UC(efread(D, sizeof(D[0]), n, f));
    msg_print("size:   %d %d %d", N[X], N[Y], N[Z]);
    msg_print("extend: %g %g %g", ext[X], ext[Y], ext[Z]);
    msg_print("value: D[0,n-1] = %g %g", D[0], D[n-1]);

    *pq = q;
}
/* end src/wall/sdf/field/imp/ini.h */
/* begin src/wall/sdf/field/imp/fin.h */
void field_fin(Field *q) {
    EFREE(q->D);
    EFREE(q);
}
/* end src/wall/sdf/field/imp/fin.h */
/* begin src/wall/sdf/field/imp/sample.h */
struct Fi { /* field */
    const Tform *t;
    const int *n;
    float *D;
};

static void fi_ini(const Tform *t, const int *n, float *D, /**/ Fi *fi) {
    fi->t = t; fi->n = n; fi->D = D;
}

static void fi_r(const Fi *fi, int ix, int iy, int iz, /**/ float *r) {
    enum {X, Y, Z};
    const Tform *t;
    t = fi->t;
    float a[3] = {(float)ix, (float)iy, (float)iz};
    tform_convert(t, a, /**/ r);
}

static void fi_set(const Fi *fi, int ix, int iy, int iz, float v) {
    enum {X, Y};
    float *D;
    const int *n;
    int i;
    D = fi->D; n = fi->n;
    i = ix + n[X] * (iy + n[Y] * iz);
    D[i] = v;
}

static float spl(float x) { /* b-spline (see poc/spline/main.mac) */
    return
        x <= 0 ? 0.0 :
        x <= 1 ? x*x*x/6 :
        x <= 2 ? (x*((12-3*x)*x-12)+4)/6 :
        x <= 3 ? (x*(x*(3*x-24)+60)-44)/6 :
        x <= 4 ? (x*((12-x)*x-48)+64)/6 :
        0.0;
}

static float get(const int N[3], const float *D, const float *r) {
    enum {X, Y, Z};
#define DDD(ix, iy, iz) (D [ix + N[X] * (iy + N[Y] * iz)])
    int i, c, sx, sy, sz, anchor[3], g[3];
    float val, s, w[3][4], tmp[4][4], partial[4];

    for (c = 0; c < 3; ++c)
        anchor[c] = (int)floor(r[c]);

    for (c = 0; c < 3; ++c)
        for (i = 0; i < 4; ++i)
            w[c][i] = spl(r[c] - (anchor[c] - 1 + i) + 2);

    for (sz = 0; sz < 4; ++sz) {
        for (sy = 0; sy < 4; ++sy) {
            s = 0;
            for (sx = 0; sx < 4; ++sx) {
                int l[3] = {sx, sy, sz};
                for (c = 0; c < 3; ++c)
                    g[c] = (l[c] - 1 + anchor[c] + N[c]) % N[c];
                s += w[0][sx] * DDD(g[X], g[Y], g[Z]);
            }
            tmp[sz][sy] = s;
        }
    }
    for (sz = 0; sz < 4; ++sz) {
        s = 0;
        for (sy = 0; sy < 4; ++sy) s += w[1][sy] * tmp[sz][sy];
        partial[sz] = s;
    }
    val = 0;
    for (sz = 0; sz < 4; ++sz) val += w[2][sz] * partial[sz];
    return val;
}

static void sample(const Tform *t, const int N0[3], const float *D0, const int N1[3], float *D1) {
    enum {X, Y, Z};
    int ix, iy, iz;
    float val, r[3];
    Fi fi;
    fi_ini(t, N1, D1, /**/ &fi);
    for (iz = 0; iz < N1[Z]; ++iz) {
        for (iy = 0; iy < N1[Y]; ++iy) {
            for (ix = 0; ix < N1[X]; ++ix) {
                fi_r(&fi, ix, iy, iz, /**/ r);
                val = get(N0, D0, r);
                fi_set(&fi, ix, iy, iz, val);
            }
        }
    }
}
/* end src/wall/sdf/field/imp/sample.h */
/* begin src/wall/sdf/field/imp/main.h */
static void dump0(const Coords *coords, const int N0[3], const float *D0, /**/ float *D1) {
    int L[3] = {xs(coords), ys(coords), zs(coords)};
    Tform *t;
    UC(tform_ini(&t));
    UC(out2sdf_ini(coords, N0, /**/ t));
    UC(sample(t, N0, D0,   L, /**/ D1));
    UC(tform_fin(t));
}

static void dump1(const Coords *coords, MPI_Comm cart, const int N[3], const float *D, /*w*/ float *W) {
    int3 L = subdomain(coords);
    const char *names[] = {"wall"};
    const float *data[] = {W};

    UC(dump0(coords, N, D, /**/ W));
    UC(grid_write(L, L, cart, DUMP_BASE "/h5/wall.h5", 1, data, names));
}

static void dump(const Coords *coords, MPI_Comm cart, const int N[], const float *D) {
    float *W;
    int ngrid = xs(coords) * ys(coords) * zs(coords);
    EMALLOC(ngrid, &W);
    UC(dump1(coords, cart, N, D, /*w*/ W));
    EFREE(W);
}

void field_size(const Field *q, /**/ int N[3]) {
    enum {X, Y, Z};
    N[X] = q->N[X]; N[Y] = q->N[Y]; N[Z] = q->N[Z];
}
void field_extend(const Field *q, /**/ float ext[3]) {
    enum {X, Y, Z};
    ext[X] = q->ext[X]; ext[Y] = q->ext[Y]; ext[Z] = q->ext[Z];
}
void field_data(const Field *q, /**/ float **pD) {
    *pD = q->D;
}
void field_scale(Field *q, float scale) {
    enum {X, Y, Z};
    float *D;
    int i, n, *N;
    N = q->N; D = q->D;
    n = N[X]*N[Y]*N[Z];
    for (i = 0; i < n; i++) D[i] *= scale;
}

void field_sample(const Field *F, const Tform *t, const int N1[3], /**/ Field **pq) {
    enum {X, Y, Z};
    int n;
    Field *q;
    EMALLOC(1, &q);
    n = N1[X]*N1[Y]*N1[Z];
    EMALLOC(n, &q->D);
    sample(t, F->N, F->D, N1, /**/ q->D);
    q->N[X] = N1[X];
    q->N[Y] = N1[Y];
    q->N[Z] = N1[Z];
    /* q->ext ?  */
    *pq = q;
}

void field_dump(const Field *q, const Coords *c, MPI_Comm cart) {
    dump(c, cart, q->N, q->D);
}
/* end src/wall/sdf/field/imp/main.h */
/* end src/wall/sdf/field/imp.cpp */

#line 1 "src/wall/sdf/imp.cu"
