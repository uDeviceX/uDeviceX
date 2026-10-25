// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/wall/exch/imp.cpp */
#include <stdio.h>
#include <string.h>
#include <mpi.h>

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

#ifndef UDX_WALL_DEF_COMMON
#define UDX_WALL_DEF_COMMON
/* begin src/inc/def.h */
/* solvent colors */
#define XMACRO_COLOR(_)                         \
    _(BLUE)                                     \
    _(RED)

#define ENUM_COLOR(a) a##_COLOR
#define make_enum_color(a) ENUM_COLOR(a) ,
enum {
    XMACRO_COLOR(make_enum_color)
    N_COLOR
};
#undef make_enum_color

enum {
    /* wall margins */
    XWM = 6,
    YWM = 6,
    ZWM = 6,

    SAFETY_FACTOR_MAXP = 3,

    /* maximum number of particles per solid */
    MAX_PSOLID_NUM = 12000,

    /* maximum number of solids per node */
    MAX_SOLIDS = 100,

    /* maximum number of object types (solid, rbc, ...) */
    MAX_MBR_TYPES = 2,
    MAX_RIG_TYPES = 2,
    MAX_OBJ_TYPES = MAX_MBR_TYPES + MAX_RIG_TYPES,

    /* maximum number density of particles of the objects */
    MAX_OBJ_DENSITY = 30,

    /* maximum number of red blood cells per node */
    MAX_CELL_NUM = 1500,

    /* maximum texture size in bytes */
    MAX_TEXO_SIZE = 100000000,

    /* safety factor for dpd halo interactions */
    HSAFETY_FACTOR = 10,

    /* safety factor for odist fragments */
    ODSTR_FACTOR = 3,
};

/* for the spring forces r = min(lmax*RBC_SPRING_CAP, r) */
#define RBC_SPRING_CAP (0.99)
/* end src/inc/def.h */
#endif

/* begin src/frag/imp.h */
enum {frag_bulk = 26};

struct int3;

namespace frag_hst {

// tag::i2d[]
int i2dx(int i);                 // <1>
int i2dy(int i);                 // <2>
int i2dz(int i);                 // <3>
void i2d3(int i, /**/ int d[3]); // <4>
// end::i2d[]

// tag::d2i[]
int d2i(int x, int y, int z); // <1>
int d32i(const int d[3]);     // <2>
// end::d2i[]

// tag::ncell[]
int ncell(int3 L, int i); // <1>
// end::ncell[]

// tag::anti[]
int antid2i(int x, int y, int z); // <1>
int anti(int i);                  // <2>
// end::anti[]

// tag::hst[]
void estimates(int3 L, int nfrags, float maxd, /**/ int *cap); // <1>
// end::hst[]

} // frag_hst
/* end src/frag/imp.h */
/* begin src/comm/imp.h */
typedef unsigned char data_t;

enum {
    NFRAGS = 26, /* number of fragments for one halo */
    BULK   = 26, /* bulk id                          */
    NBAGS  = 27, /* fragments + bulk                 */
};

/* Allocation mod for bags */
enum AllocMod {
    // tag::AllocMod[]
    HST_ONLY,   /* only host bags allocated                 */
    DEV_ONLY,   /* only device bags allocated               */
    PINNED,     /* both host and device pinned              */
    PINNED_HST, /* host pinned; no device memory            */
    PINNED_DEV, /* host pinned; device global memory on gpu */
    NONE        /* no allocation                            */
    // end::AllocMod[]
};

// tag::dBags[]
struct dBags {
    data_t *data[NBAGS]; /* data on the device         */
};
// end::dBags[]

// tag::hBags[]
struct hBags {
    data_t *data[NBAGS]; /* data on the host                    */
    int         *counts; /* size of the data                    */
    int capacity[NBAGS]; /* capacity of each frag (elem number) */
    size_t bsize;        /* size of one datum in bytes          */
};
// end::hBags[]

struct Comm;
struct CommBuffer;

// tag::alloc[]
void comm_bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS], /**/ hBags *hb, dBags *db);
void comm_bags_fin(AllocMod fmod, AllocMod bmod, /**/ hBags *hb, dBags *db);
void comm_ini(MPI_Comm cart, /**/ Comm **c);
void comm_fin(/**/ Comm *c);
void comm_buffer_ini(int nbags, const hBags *hbb, CommBuffer**);
void comm_buffer_fin(CommBuffer*);
// end::alloc[]


void comm_buffer_set(int nbags, const hBags*, CommBuffer*);
void comm_buffer_get(const CommBuffer*, int nbags, hBags*);

// tag::communication[]
void comm_post_recv(hBags *b, Comm *c);           // <1>
void comm_post_send(const hBags *b, Comm *c);     // <2>

void comm_wait_recv(Comm *c, /**/ hBags *b);      // <3>
void comm_wait_send(Comm *c);                     // <4>
// end::communication[]

// tag::communication_b[]
void comm_post_recv(CommBuffer *cb, Comm *c);
void comm_post_send(const CommBuffer *cb, Comm *c);

void comm_wait_recv(Comm *c, /**/ CommBuffer *cb);
void comm_wait_send(Comm *c);
// end::communication_b[]

// tag::get[]
int    comm_get_number_capacity(int i, const hBags *b); // <1>
size_t comm_get_byte_capacity(int i, const hBags *b);   // <2>
// end::get[]

/* end src/comm/imp.h */

#ifndef UDX_WALL_CORE_TYPE
#define UDX_WALL_CORE_TYPE
/* begin src/inc/type.h */
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
#endif

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
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */

static void exch_estimate(int numdensity, int3 L, int cap[]) {
    enum {SAFETY_FACTOR = 2};
    int i, nc, c;
    for (i = 0; i < NFRAGS; ++i) {
        nc = frag_hst::ncell(L, i);
        nc *= frag_hst::i2dx(i) ? XWM : 1;
        nc *= frag_hst::i2dy(i) ? YWM : 1;
        nc *= frag_hst::i2dz(i) ? ZWM : 1;
        c = numdensity * nc;
        cap[i] = SAFETY_FACTOR * c;
    }
}

static void shift(int3 L, int fid, float r[3]) {
    enum {X, Y, Z};
    int d[3];
    frag_hst::i2d3(fid, d);
    r[X] += d[X] * L.x;
    r[Y] += d[Y] * L.y;
    r[Z] += d[Z] * L.z;
}

static bool is_inside(int3 LW, const Particle p) {
    enum {X, Y, Z};
    return
        p.r[X] >= -0.5 * LW.x && p.r[X] < 0.5 * LW.x &&
        p.r[Y] >= -0.5 * LW.y && p.r[Y] < 0.5 * LW.y &&
        p.r[Z] >= -0.5 * LW.z && p.r[Z] < 0.5 * LW.z;
}

static void fill_bags(int3 L, int n, const Particle *pp, hBags *b) {
    int i, j, *cc, c;
    Particle p0, p, **dst;
    int3 LW;
    LW.x = L.x + 2*XWM;
    LW.y = L.y + 2*YWM;
    LW.z = L.z + 2*ZWM;

    cc  = b->counts;
    dst = (Particle **) b->data;

    memset(cc, 0, NBAGS * sizeof(int));

    for (i = 0; i < n; ++i) {
        p0 = pp[i];
        for (j = 0; j < NFRAGS; ++j) {
            p = p0;
            shift(L, j, p.r);
            if (is_inside(LW, p)) {
                c = cc[j] ++;
                dst[j][c] = p;
            }
        }
    }
}

static void communicate(const hBags *s, Comm *c, hBags *r) {
    UC(comm_post_send(s, c));
    UC(comm_post_recv(r, c));
    UC(comm_wait_send(c));
    UC(comm_wait_recv(c, /**/ r));
}

static void check_counts(int maxn, int n0, const hBags *b) {
    int i, c = n0;
    for (i = 0; i < NFRAGS; ++i) c += b->counts[i];
    if (c > maxn)
        ERR("Too many particles for wall : %d / %d\n", c, maxn);
}

static void unpack(const hBags *b, /*io*/ int *n, Particle *pp) {
    int i, j, k, c;
    const Particle *src;
    k = *n;
    for (j = 0; j < NFRAGS; ++j) {
        c = b->counts[j];
        src = (const Particle *) b->data[j];
        for (i = 0; i < c; ++i) {
            pp[k] = src[i];
            ++k;
        }
    }
    *n = k;
}

/* exchange pp(hst) between processors to get a wall margin */
void wall_exch_pp(MPI_Comm cart, int numdensity, int3 L, int maxn, /*io*/ Particle *pp, int *n) {
    hBags send, recv;
    Comm *com;
    int capacity[NBAGS];

    exch_estimate(numdensity, L, capacity);

    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Particle), capacity, &send, NULL));
    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Particle), capacity, &recv, NULL));
    UC(comm_ini(cart, &com));

    fill_bags(L, *n, pp, /**/ &send);
    communicate(&send, /**/ com, &recv);
    check_counts(maxn, *n, &recv);
    unpack(&recv, /**/ n, pp);

    UC(comm_bags_fin(HST_ONLY, NONE, &send, NULL));
    UC(comm_bags_fin(HST_ONLY, NONE, &recv, NULL));
    UC(comm_fin(com));
}
/* end src/wall/exch/imp.cpp */

#line 1 "src/wall/force/imp.cu"
