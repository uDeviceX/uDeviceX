// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/grid_sampler/imp.cu */
#include <mpi.h>
#include <stdio.h>

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

/* begin src/utils/kl.h */
/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
/* begin src/utils/kl/release.h */
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
/* end src/utils/kl/release.h */
#elif  defined(KL_NONE)
/* begin src/utils/kl/none.h */
#define KL_BEFORE(s, C)
#define KL_CALL(F, C, A)
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
/* end src/utils/kl/none.h */
#elif  defined(KL_SYNC)
/* begin src/utils/kl/sync.h */
#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)                             \
    do {                                        \
        dSync();                                \
        CC(d::PeekAtLastError());               \
    } while(0)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
/* end src/utils/kl/sync.h */
#else
  #error KL_* is undefined
#endif

/* begin src/utils/kl/common.h */
namespace kl {
inline int safe0(dim3 i, dim3 j) {
    msg(i.x, i.y, i.z,   j.x, j.y, j.z);
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
}
/* end src/utils/kl/common.h */
/* begin src/utils/kl/macro.h */
#define ESC(...) __VA_ARGS__
#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, C);                       \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)
/* end src/utils/kl/macro.h */

/* TODO: header includes header */
/* end src/utils/kl.h */
/* begin src/utils/cc.h */
/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
/* begin src/utils/cc/sync.h */
/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
/* end src/utils/cc/sync.h */
#elif defined(CC_RELEASE)
/* begin src/utils/cc/release.h */
/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
/* end src/utils/cc/release.h */
#else
  #error CC_* is undefined
#endif

/* begin src/utils/cc/common.h */
namespace cc {
void check(int rc, const char *file, int line);
}


/* end src/utils/cc/common.h */

/* TODO: header includes header */
/* end src/utils/cc.h */

/* begin src/d/api.h */
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

// tag::more[]
const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);
// end::more[]

// tag::api[]
int GetSymbolAddress(void **devPtr, const void *symbol);
int Malloc(void **p, size_t);
int MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset=0, int kind=MemcpyHostToDevice);
int MemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset=0, int kind=MemcpyDeviceToHost);
int HostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
int Memcpy (void *dst, const void *src, size_t count, int kind);
int MemsetAsync (void *devPtr, int value, size_t count, Stream_t stream=0);
int Memset (void *devPtr, int value, size_t count);
int MemcpyAsync (void * dst, const void * src, size_t count, int kind, Stream_t stream = 0);
int Free (void *devPtr);
int FreeHost (void *hstPtr);
int DeviceSynchronize (void);
int PeekAtLastError(void);
// end::api[]
}
/* end src/d/api.h */

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
/* begin src/inc/dev.h */
/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())

#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault /* "[a]ny to [a]ny" */

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))  /* [t]to, [f]rom */
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))  /* [t]to, [f]rom */
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))

/* device allocation */
#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))

/* [d]evice set */
#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)

/* [d]evice [a]synchronous set */
#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)
/* end src/inc/dev.h */

/* begin src/io/grid/imp.h */
// tag::int[]
void grid_write(int3 subGrid, int3 subDomain, MPI_Comm, const char *path, int ncmp, const float **data, const char **names);
// end::int[]
/* end src/io/grid/imp.h */

/* begin src/grid_sampler/imp.h */
struct GridSampler;
struct GridSampleData;

struct Particle;

// tag::data_mem[]
void grid_sampler_data_ini(GridSampleData **);
void grid_sampler_data_fin(GridSampleData  *);
// end::data_mem[]

// tag::data_int[]
void grid_sampler_data_reset(GridSampleData *); // <1>
void grid_sampler_data_push(long n, const Particle *pp, const int *cc,
                            const float *ss, GridSampleData *); // <2>
// end::data_int[]


// tag::mem[]
void grid_sampler_ini(bool colors, bool stress, int3 L, int3 N, GridSampler**); // <1>
void grid_sampler_fin(GridSampler*); // <2>
// end::mem[]

// tag::int[]
void grid_sampler_reset(GridSampler*); // <1>
void grid_sampler_add(const GridSampleData*, GridSampler*); // <2>
void grid_sampler_dump(MPI_Comm, const char *dir, long id, GridSampler*); // <3>
// end::int[]
/* end src/grid_sampler/imp.h */
/* begin src/grid_sampler/imp/type.h */
enum {
    RHO, VX, VY, VZ,
    NFIELDS_P
};

enum {
    SXX, SXY, SXZ,
    SYY, SYZ, SZZ,
    NFIELDS_S
};

enum {
    NFIELDS_C = N_COLOR
};

enum {
    TOT_NFIELDS = NFIELDS_P + NFIELDS_S + NFIELDS_C
};

static const char *names_p[NFIELDS_P] =
    {"density", "u", "v", "w"};

static const char *names_s[NFIELDS_S] =
    {"sxx", "sxy", "sxz",
     "syy", "syz", "szz"};

#define make_str(a) #a ,
static const char *names_c[NFIELDS_C] =
    {XMACRO_COLOR(make_str)}; /* see inc/def.h */
#undef make_str

struct Grid {
    int3 L, N; /* subdomain size, grid size */
    float *p[NFIELDS_P]; /* particle density and velocity */
    float *s[NFIELDS_S]; /* stress                        */
    float *c[NFIELDS_C]; /* color density                 */
    bool colors, stress;
};

struct GridSampler {
    Grid sdev, stdev, hst;
    long nsteps;
};

struct SampleDatum {
    long n;
    const Particle *pp;
    const float *ss;
    const int *cc;
};

enum {
    MAX_N_DATA = 10
};

struct GridSampleData {
    int n;
    SampleDatum d[MAX_N_DATA];
};
/* end src/grid_sampler/imp/type.h */

namespace sampler_dev {
/* begin src/grid_sampler/dev/type.h */
struct Part {
    float3 r, v;
};

struct Stress {
    float3 s1, s2;
};
/* end src/grid_sampler/dev/type.h */
/* begin src/grid_sampler/dev/fetch.h */
#define _S_ static __device__
#define _I_ static __device__

_I_ void fetch_part(int i, const SampleDatum *d, Part *p) {
    float2 p0, p1, p2;
    const float2 *pp = (const float2*) d->pp;
    p0 = pp[3*i+0]; p1 = pp[3*i+1]; p2 = pp[3*i+2];

    p->r = make_float3(p0.x, p0.y, p1.x);
    p->v = make_float3(p1.y, p2.x, p2.y);
}

_I_ void fetch_color(int i, const SampleDatum *d, int *c) {
    *c = d->cc[i];
}

_I_ void fetch_stress(int i, const SampleDatum *d, Stress *s) {
    float2 s0, s1, s2;
    const float2 *ss = (const float2*) d->ss;
    s0 = ss[3*i+0]; s1 = ss[3*i+1]; s2 = ss[3*i+2];

    s->s1 = make_float3(s0.x, s0.y, s1.x);
    s->s2 = make_float3(s1.y, s2.x, s2.y);
}

#undef _S_
#undef _I_
/* end src/grid_sampler/dev/fetch.h */
/* begin src/grid_sampler/dev/add.h */
#define _S_ static __device__
#define _I_ static __device__

_I_ void add_part(int i, const Part *p, const Grid *g) {
    atomicAdd(g->p[RHO] + i, 1.f);
    atomicAdd(g->p[VX]  + i, p->v.x);
    atomicAdd(g->p[VY]  + i, p->v.y);
    atomicAdd(g->p[VZ]  + i, p->v.z);
}

_I_ void add_color(int i, const int c, const Grid *g) {
#define ADD_COLOR(a)                               \
    if (ENUM_COLOR(a) == c)                        \
        atomicAdd(g->c[ENUM_COLOR(a)] + i, 1.f);

    XMACRO_COLOR(ADD_COLOR)

#undef ADD_COLOR
}

_I_ void add_stress(int i, const Stress *s, const Grid *g) {
    atomicAdd(g->s[SXX] + i, s->s1.x);
    atomicAdd(g->s[SXY] + i, s->s1.y);
    atomicAdd(g->s[SXZ] + i, s->s1.z);

    atomicAdd(g->s[SYY] + i, s->s2.x);
    atomicAdd(g->s[SYZ] + i, s->s2.y);
    atomicAdd(g->s[SZZ] + i, s->s2.z);
}

#undef _S_
#undef _I_
/* end src/grid_sampler/dev/add.h */
/* begin src/grid_sampler/dev/main.h */
#define _S_ static __device__

_S_ float get_spacing(int L, int N) {
    return (float) L / (float) N;
}

_S_ int get_coord(float x, int L, int N) {
    int i;
    float dx;
    dx = get_spacing(L, N);
    i = (x + L/2) / dx;
    return min(N-1, max(0, i));
}

_S_ int3 get_cell_coords(float3 r, int3 L, int3 N) {
    return make_int3(get_coord(r.x, L.x, N.x),
                     get_coord(r.y, L.y, N.y),
                     get_coord(r.z, L.z, N.z));
}

_S_ int get_grid_id(int3 gc, int3 N) {
    return gc.x + N.x * (gc.y + N.y * gc.z);
}

_S_ int get_grid_id(const Part *p, const Grid *g) {
    int3 gcoords;
    gcoords = get_cell_coords(p->r, g->L, g->N);
    return get_grid_id(gcoords, g->N);
}

__global__ void add(SampleDatum data, /**/ Grid grid) {
    int pid, gid; /* particle and grid id */
    Part p;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= data.n) return;

    fetch_part(pid, &data, &p);
    gid = get_grid_id(&p, &grid);

    add_part(gid, &p, &grid);

    if (grid.colors) {
        int c;
        fetch_color(pid, &data, &c);
        add_color(gid, c, &grid);
    }

    if (grid.stress) {
        Stress s;
        fetch_stress(pid, &data, &s);
        add_stress(gid, &s, &grid);
    }
}

_S_ long get_size(const Grid *g) {
    int3 N = g->N;
    return N.x * N.y * N.z;
}

_S_ float cell_volume(const Grid *g) {
    float dx, dy, dz;
    dx = get_spacing(g->L.x, g->N.x);
    dy = get_spacing(g->L.y, g->N.y);
    dz = get_spacing(g->L.z, g->N.z);
    return dx * dy * dz;
}

__global__ void space_avg(Grid g) {
    long i;
    float inv_rho;
    float rho, vol;
    float sv, sr, ss; /* scales for v, rho, stress */
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= get_size(&g)) return;

    rho = g.p[RHO][i];
    vol = cell_volume(&g);

    inv_rho = fabs(rho) > 1e-6 ? 1.f / rho : 0;

    sv = inv_rho;
    sr = 1.f / vol;

    g.p[RHO][i] = sr * rho;

    g.p[VX][i] *= sv;
    g.p[VY][i] *= sv;
    g.p[VZ][i] *= sv;

    if (g.colors) {
#define scale(a) g.c[ENUM_COLOR(a)][i] *= inv_rho;
        XMACRO_COLOR(scale)
#undef scale
    }

    if (g.stress) {
        ss = sr;
        g.s[SXX][i] *= ss;
        g.s[SXY][i] *= ss;
        g.s[SXZ][i] *= ss;
        g.s[SYY][i] *= ss;
        g.s[SYZ][i] *= ss;
        g.s[SZZ][i] *= ss;
    }
}

__global__ void add_to_grid(const Grid src, Grid dst) {
    long i, j;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= get_size(&src)) return;

    for (j = 0; j < NFIELDS_P; ++j)
        dst.p[j][i] += src.p[j][i];

    if (src.colors) {
        for (j = 0; j < NFIELDS_C; ++j)
            dst.c[j][i] += src.c[j][i];
    }

    if (src.stress) {
        for (j = 0; j < NFIELDS_S; ++j)
            dst.s[j][i] += src.s[j][i];
    }
}

__global__ void time_avg(int nsteps, Grid g) {
    long i, j;
    float s;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= get_size(&g)) return;

    s = 1.0 / nsteps;

    for (j = 0; j < NFIELDS_P; ++j)
        g.p[j][i] *= s;

    if (g.colors)
        for (j = 0; j < NFIELDS_C; ++j)
            g.c[j][i] *= s;

    if (g.stress)
        for (j = 0; j < NFIELDS_S; ++j)
            g.s[j][i] *= s;
}

#undef _S_
/* end src/grid_sampler/dev/main.h */
}

/* begin src/grid_sampler/imp/main.h */
static long get_size(const Grid *g) {
    int3 N = g->N;
    return N.x * N.y * N.z;
}

static void dev_alloc_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) Dalloc(&a[i], n);
}

static void ini_dev_grid(bool colors, bool stress, int3 L, int3 N, Grid *g) {
    long n;
    g->N = N;
    g->L = L;
    g->colors = colors;
    g->stress = stress;

    n = get_size(g);

    UC(             dev_alloc_n(NFIELDS_P, n, g->p) );
    if (colors) UC( dev_alloc_n(NFIELDS_C, n, g->c) );
    if (stress) UC( dev_alloc_n(NFIELDS_S, n, g->s) );
}

static void hst_alloc_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) EMALLOC(n, &a[i]);
}

static void ini_hst_grid(bool colors, bool stress, int3 L, int3 N, Grid *g) {
    long n;
    g->N = N;
    g->L = L;
    g->colors = colors;
    g->stress = stress;

    n = get_size(g);
    UC(             hst_alloc_n(NFIELDS_P, n, g->p) );
    if (colors) UC( hst_alloc_n(NFIELDS_C, n, g->c) );
    if (stress) UC( hst_alloc_n(NFIELDS_S, n, g->s) );
}

static void dev_free_n(int na, float *a[]) {
    for (int i = 0; i < na; ++i) Dfree(a[i]);
}

static void fin_dev_grid(Grid *g) {
    UC(                dev_free_n(NFIELDS_P, g->p) );
    if (g->colors) UC( dev_free_n(NFIELDS_C, g->c) );
    if (g->stress) UC( dev_free_n(NFIELDS_S, g->s) );
}

static void hst_free_n(int na, float *a[]) {
    for (int i = 0; i < na; ++i) EFREE(a[i]);
}

static void fin_hst_grid(Grid *g) {
    UC(                hst_free_n(NFIELDS_P, g->p) );
    if (g->colors) UC( hst_free_n(NFIELDS_C, g->c) );
    if (g->stress) UC( hst_free_n(NFIELDS_S, g->s) );
}

void grid_sampler_ini(bool colors, bool stress, int3 L, int3 N, GridSampler **s0) {
    GridSampler *s;
    EMALLOC(1, s0);
    s = *s0;
    UC(ini_dev_grid(colors, stress, L, N, &s->sdev));
    UC(ini_dev_grid(colors, stress, L, N, &s->stdev));
    UC(ini_hst_grid(colors, stress, L, N, &s->hst));
    UC(grid_sampler_reset(s));
}

void grid_sampler_fin(GridSampler *s) {
    UC(fin_dev_grid(&s->sdev));
    UC(fin_dev_grid(&s->stdev));
    UC(fin_hst_grid(&s->hst));
    EFREE(s);
}

static void reset_dev_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) DzeroA(a[i], n);
}

static void reset_dev_grid(Grid *g) {
    long n = get_size(g);
    UC(                reset_dev_n(NFIELDS_P, n, g->p) );
    if (g->colors) UC( reset_dev_n(NFIELDS_C, n, g->c) );
    if (g->stress) UC( reset_dev_n(NFIELDS_S, n, g->s) );
}

void grid_sampler_reset(GridSampler *s) {
    s->nsteps = 0;
    UC(reset_dev_grid(&s->stdev));
}

static void add_data_to_grid(const GridSampleData *data, Grid *g) {
    long i, n;
    SampleDatum d;
    for (i = 0; i < data->n; ++i) {
        d = data->d[i];
        n = d.n;
        KL(sampler_dev::add, (k_cnf(n)), (d, *g));
    }
}

static void space_avg(Grid *g) {
    long n = get_size(g);
    KL(sampler_dev::space_avg, (k_cnf(n)), (*g));
}

static void add_to_grid(const Grid *s, Grid *st) {
    long n = get_size(s);
    KL(sampler_dev::add_to_grid, (k_cnf(n)), (*s, *st));
}

void grid_sampler_add(const GridSampleData *data, GridSampler *s) {
    Grid *sg, *stg;
    sg  = &s->sdev;
    stg = &s->stdev;
    UC(reset_dev_grid(sg));
    UC(add_data_to_grid(data, sg));
    UC(space_avg(sg));
    UC(add_to_grid(sg, stg));
    s->nsteps ++;
}

static void time_avg(int nsteps, Grid *g) {
    long n = get_size(g);
    KL(sampler_dev::time_avg, (k_cnf(n)), (nsteps, *g));
}

static void download_n(int na, long n, float * const dev[], float *hst[]) {
   for (int i = 0; i < na; ++i) aD2H(hst[i], dev[i], n);
}

static void download(const Grid *dev, Grid *hst) {
    long n = get_size(dev);
    UC(                  download_n(NFIELDS_P, n, dev->p, hst->p) );
    if (dev->colors) UC( download_n(NFIELDS_C, n, dev->c, hst->c) );
    if (dev->stress) UC( download_n(NFIELDS_S, n, dev->s, hst->s) );
}

static void build_desc(int ncmp, const char * const name[], const float * const data[],
                       int *j, const char *dst_name[], const float *dst_data[]) {
    for (int i = 0; i < ncmp; ++i) {
        dst_data[*j] = data[i];
        dst_name[*j] = name[i];
        ++(*j);
    }
}

static void dump(MPI_Comm cart, const char *dir, long id, const Grid *g) {
    char path[FILENAME_MAX] = {'\0'};
    int ncmp;
    const float *data[TOT_NFIELDS];
    const char *names[TOT_NFIELDS];

    ncmp = 0;
    build_desc               (NFIELDS_P, names_p, g->p, /**/ &ncmp, names, data);
    if (g->colors) build_desc(NFIELDS_C, names_c, g->c, /**/ &ncmp, names, data);
    if (g->stress) build_desc(NFIELDS_S, names_s, g->s, /**/ &ncmp, names, data);

    snprintf(path, FILENAME_MAX, DUMP_BASE "/%s/%04ld.h5", dir, id);

    UC(grid_write(g->N, g->L, cart, path, ncmp, data, names));
}

void grid_sampler_dump(MPI_Comm cart, const char *dir, long id, GridSampler *s) {
    UC(time_avg(s->nsteps, &s->stdev));
    UC(download(&s->stdev, &s->hst));
    dSync();
    UC(dump(cart, dir, id, &s->hst));
}
/* end src/grid_sampler/imp/main.h */
/* begin src/grid_sampler/imp/data.h */
void grid_sampler_data_ini(GridSampleData **s) {
    EMALLOC(1, s);
    grid_sampler_data_reset(*s);
}

void grid_sampler_data_fin(GridSampleData  *s) {
    EFREE(s);
}

void grid_sampler_data_reset(GridSampleData *s) {
    s->n = 0;
}

void grid_sampler_data_push(long n, const Particle *pp, const int *cc, const float *ss, GridSampleData *s) {
    int id;
    SampleDatum *d;
    id = s->n++;

    if (s->n >= MAX_N_DATA)
        ERR("Too many data: %d/%d", s->n / MAX_N_DATA);

    d = s->d + id;
    d->n = n;
    d->pp = pp;
    d->cc = cc;
    d->ss = ss;
}

/* end src/grid_sampler/imp/data.h */
/* end src/grid_sampler/imp.cu */

#line 1 "src/he/err.cpp"
