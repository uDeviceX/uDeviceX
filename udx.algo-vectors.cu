// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/algo/vectors/imp.cpp */
#include <stdio.h>

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
/* begin src/inc/type.h */
#ifndef UDX_GRID_INTER_TYPE_COMMON
#define UDX_GRID_INTER_TYPE_COMMON
template <typename T, int N>
struct Sarray { /* [s]tatic array */
    T d[N];
};

struct Particle {
    float r[3], v[3];
};

struct Rigid {
    float Iinv[6],            /* moment of inertia            6        */
        mass,                 /* mass of the solid            7        */
        com[3],               /* [c]enter [o]f [m]ass         10       */
        v[3], om[3],          /* linear and angular velocity  13 16    */
        e0[3], e1[3], e2[3],  /* local referential            19 22 25 */
        fo[3], to[3],         /* force, torque                28 31    */
        id;                   /* id of the solid              32       */
};

struct Force {
    float f[3];
};

struct Momentum {
    float P[3], L[3]; /* linear and angular momentum */
};

struct Positioncp {
    float rc[3]; /* current position  */
    float rp[3]; /* previous position */
};

typedef Sarray<int, 26>       int26;
typedef Sarray<int, 27>       int27;

typedef Sarray<int*, 26>     intp26;
typedef Sarray<Particle*, 26> Pap26;
typedef Sarray<Force*,    26> Fop26;
typedef Sarray<Momentum*, 26> Mop26;
typedef Sarray<Positioncp*, 26> Pocpp26;

/* particles wrap */
struct PaWrap {
    int n;
    const Particle *pp;
};

/* forces wrap */
struct FoWrap {
    int n;
    Force *ff;
};
/* end src/inc/type.h */
#endif /* UDX_GRID_INTER_TYPE_COMMON */
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

/* begin src/algo/vectors/imp.h */
struct Particle;
struct Vectors;
struct Coords;

// tag::interface[]
void vectors_float_ini(int n, const float*, /**/ Vectors**); // <1>
void vectors_postions_ini(int n, const Particle*, /**/ Vectors**); // <2>
void vectors_postions_edge_ini  (const Coords*, int n, const Particle*, /**/ Vectors**); // <3>
void vectors_postions_center_ini(const Coords*, int n, const Particle*, /**/ Vectors**); // <4>
void vectors_velocities_ini(int n, const Particle*, /**/ Vectors**); // <5>
void vectors_zero_ini(int n, /**/ Vectors**); // <6>

void vectors_fin(Vectors*);
void vectors_get(const Vectors*, int i, /**/ float r[3]); // <7>
// end::interface[]
/* end src/algo/vectors/imp.h */
/* begin src/algo/vectors/imp/type.h */
struct Vectors {
    int type;
    int n;
    union {
        const float *rr;
        const Particle *pp;
    } D;
    Tform *tform;
};

static void float_get(const Vectors*, int i, float r[3]);

static void positions_get(const Vectors*, int i, float r[3]);
static void positions_edge_get(const Vectors*, int i, float r[3]);
static void positions_center_get(const Vectors*, int i, float r[3]);
static void velocities_get(const Vectors*, int i, float r[3]);
static void zero_get(const Vectors*, int i, float r[3]);

enum {VECTOR_FLOAT, VECTOR_POSITIONS, VECTOR_POSITIONS_EDGE, VECTOR_POSITIONS_CENTER, VECTOR_VELOCITIES, VECTOR_ZERO};
static void (*vector_get_table[])(const Vectors*, int i, float r[3]) =
{ float_get,  positions_get, positions_edge_get, positions_center_get, velocities_get, zero_get };

typedef  void (*Local2Global)(const Coords *coords, float a[3], /**/ float b[3]);
/* end src/algo/vectors/imp/type.h */
/* begin src/algo/vectors/imp/main.h */
void vectors_float_ini(int n, const float *rr, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_FLOAT; q->n = n; q->D.rr = rr;
    *pq = q;
}

void vectors_postions_ini(int n, const Particle *pp, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_POSITIONS; q->n = n; q->D.pp = pp;
    *pq = q;
}

static void l2edge(const Coords *coords, float a[3], /**/ float b[3]) {
    enum {X, Y, Z};
    b[X] = xl2xg(coords, a[X]);
    b[Y] = yl2yg(coords, a[Y]);
    b[Z] = zl2zg(coords, a[Z]);
}
static void l2center(const Coords *coords, float a[3], /**/ float b[3]) {
    enum {X, Y, Z};
    b[X] = xl2xc(coords, a[X]);
    b[Y] = yl2yc(coords, a[Y]);
    b[Z] = zl2zc(coords, a[Z]);
}

static void edge_ini_tform(const Coords *coords, Local2Global local2global, Tform **pq) {
    enum {X, Y, Z};
    Tform *q;
    float a0[3], a1[3], b0[3], b1[3];
    a0[X] = a0[Y] = a0[Z] = 0;
    b0[X] = b0[Y] = b0[Z] = 1;
    local2global(coords, a0, /**/ a1);
    local2global(coords, b0, /**/ b1);
    tform_ini(&q);
    UC(tform_vector(a0, a1, b0, b1, /**/ q));
    *pq = q;
}
void vectors_postions_edge_ini(const Coords *coords, int n, const Particle *pp, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_POSITIONS_EDGE; q->n = n; q->D.pp = pp;
    edge_ini_tform(coords, l2edge, /**/ &q->tform);
    *pq = q;
}

void vectors_postions_center_ini(const Coords *coords, int n, const Particle *pp, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_POSITIONS_EDGE; q->n = n; q->D.pp = pp;
    edge_ini_tform(coords, l2center, /**/ &q->tform);
    *pq = q;
}

void vectors_velocities_ini(int n, const Particle *pp, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_VELOCITIES; q->n = n; q->D.pp = pp;
    *pq = q;
}

void vectors_zero_ini(int n, /**/ Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_ZERO; q->n = n;
    *pq = q;
}

void vectors_fin(Vectors *q) {
    if (q->type == VECTOR_POSITIONS_EDGE ||
        q->type == VECTOR_POSITIONS_CENTER)
        tform_fin(q->tform);
    EFREE(q);
}

static void float_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const float *rr;
    rr = q->D.rr;
    r[X] = rr[3*i + 0];
    r[Y] = rr[3*i + 1];
    r[Z] = rr[3*i + 2];
}
static void positions_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    r[X] = pp[i].r[X];
    r[Y] = pp[i].r[Y];
    r[Z] = pp[i].r[Z];
}

static void shift_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    UC(tform_convert(q->tform, pp[i].r, /**/ r));
}
static void positions_edge_get(const Vectors *q, int i, float r[3]) { shift_get(q, i, r); }
static void positions_center_get(const Vectors *q, int i, float r[3]) { shift_get(q, i, r); }
static void velocities_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    r[X] = pp[i].v[X];
    r[Y] = pp[i].v[Y];
    r[Z] = pp[i].v[Z];
}
static void zero_get(const Vectors*, int, float r[3]) {
    enum {X, Y, Z};
    r[X] = r[Y] = r[Z] = 0;
}
void vectors_get(const Vectors *q, int i, /**/ float r[3]) {
    int n;
    n = q->n;
    if (i >= n) ERR("i = %d    >=   n = %d", i, n);
    if (i < 0)  ERR("i = %d    < 0", i);
    vector_get_table[q->type](q, i, r);
}
/* end src/algo/vectors/imp/main.h */
/* end src/algo/vectors/imp.cpp */

#line 1 "src/clist/imp.cu"
