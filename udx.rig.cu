// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/rig/imp.cpp */
#include <mpi.h>
#include <stdio.h>
#include <vector_types.h>
#include <assert.h>
#include <string.h>
#include <math.h>

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


/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
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
/* begin src/io/restart/imp.h */
struct Coords;
struct Particle;
struct Rigid;

// tag::id[]
enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};
// end::id[]

// tag::stream[]
typedef void (*StreamWriter)(const void*, FILE*);
typedef void (*StreamReader)(FILE*, void*);
// end::stream[]

// tag::write[]
void restart_write_pp(MPI_Comm, const char *base, const char *code, int id, long n, const Particle *pp); // <1>
void restart_write_ii(MPI_Comm, const char *base, const char *code, int id, long n, const int *ii);      // <2>
void restart_write_ss(MPI_Comm, const char *base, const char *code, int id, long n, const Rigid *ss);    // <3>
void restart_write_stream_one_node(MPI_Comm, const char *base, const char *code, int id, const void *data, StreamWriter); // <4>
 // end::write[]

// tag::read[]
void restart_read_pp(MPI_Comm, const char *base, const char *code, int id, int *n, Particle *pp);
void restart_read_ii(MPI_Comm, const char *base, const char *code, int id, int *n, int *ii);
void restart_read_ss(MPI_Comm, const char *base, const char *code, int id, int *n, Rigid *ss);
void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader, void *data);
// end::read[]
/* end src/io/restart/imp.h */

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
/* begin src/mesh/gen/imp.h */
struct Coords;
struct Particle;
struct Matrices;

// tag::int[]
/* vv : vertices of the mesh template : x y z x y z, ...   */
int  mesh_gen_from_file(const Coords*, const float *vv, const char *ic, int nv, /**/ Particle*); // <1>
void mesh_gen_from_matrices(int nv, const float *vv, const Matrices*, /**/ int *pn, Particle*);  // <2>
void mesh_shift(const Coords*, int n, /**/ Particle*); // <3>
// end::int[]
/* end src/mesh/gen/imp.h */

/* begin src/rig/imp.h */
// tag::quants[]
struct RigQuants {
    int n, ns, nps;              /* number of particles (total), solid, particle per solid        */
    Particle *pp_hst, *pp;       /* particles on hst and device                                   */
    Rigid    *ss_hst, *ss;       /* rigid strutures                                               */
    float   *rr0_hst, *rr0;      /* frozen particle templates                                     */

    /* mesh related quantities */
    int nt, nv;                   /* number of [t]riangles and [v]ertices                          */
    int4 *dtt;                    /* triangle indices on device                                    */
    float *dvv;                   /* vertices on device (template)                                 */
    Particle *i_pp_hst, *i_pp;    /* particles representing all meshes of all solids of that node  */

    Rigid *ss_dmp;

    long maxp; /* maximum particle number */
};
// end::quants[]

struct Coords;
struct RigPinInfo;
struct MeshRead;

// tag::mem[]
void rig_ini(long maxs, long maxp, const MeshRead*, RigQuants*);
void rig_fin(RigQuants *q);
// end::mem[]

// tag::gen[]
void rig_gen_mesh(const Coords*, MPI_Comm, const MeshRead*, const char *ic, /**/ RigQuants *); // <1>
void rig_gen_freeze(const Coords*, bool empty_pp, int numdensity, float rig_mass, const RigPinInfo*,
                    MPI_Comm, const MeshRead*, Particle *opp, int *on, RigQuants *q);          // <2>
void rig_strt_quants(MPI_Comm, const MeshRead*, const char *base, const char *name, const int id, RigQuants *q); // <3>
// end::gen[]

// tag::io[]
void rig_strt_dump_templ(MPI_Comm, const char *base, const char *name, const RigQuants *q);         // <1>
void rig_strt_dump(MPI_Comm, const char *base, const char *name, const int id, const RigQuants *q); // <2>
// end::io[]
/* end src/rig/imp.h */

/* begin src/rig/gen/imp.h */
struct Coords;
struct RigPinInfo;
struct MeshRead;

// tag::struct[]
struct RigGenInfo {
    float mass;
    int numdensity;
    const RigPinInfo *pi;
    int nt, nv;
    const int4 *tt;     /* on device */
    const Particle *pp; /* on device */
    bool empty_pp;
    const MeshRead *mesh;
};

struct FluInfo {
    Particle *pp;
    int *n;
};

/* everything on host */
struct RigInfo {
    int ns, *nps;
    float *rr0;
    Rigid *ss;
    Particle *pp;
};
// end::struct[]

// tag::int[]
void rig_gen_from_solvent(const Coords*, MPI_Comm, RigGenInfo, /*io*/ FluInfo, /**/ RigInfo);
// end::int[]
/* end src/rig/gen/imp.h */

/* begin src/rig/imp/ini.h */
void rig_ini(long maxs, long maxp, const MeshRead *mesh, RigQuants *q) {
    q->n = q->ns = q->nps = 0;
    q->maxp = maxp;

    Dalloc(&q->pp,      maxp);
    Dalloc(&q->ss,      maxs);
    Dalloc(&q->rr0, 3 * maxp);
    Dalloc(&q->i_pp,    maxp);

    EMALLOC(maxp, &q->pp_hst);
    EMALLOC(maxs, &q->ss_hst);
    EMALLOC(3 * maxp, &q->rr0_hst);
    EMALLOC(maxp, &q->i_pp_hst);

    EMALLOC(maxs, &q->ss_dmp);

    q->nt = mesh_read_get_nt(mesh);
    q->nv = mesh_read_get_nv(mesh);

    Dalloc(&q->dtt,     q->nt);
    Dalloc(&q->dvv, 3 * q->nv);

    cH2D(q->dtt, mesh_read_get_tri (mesh),     q->nt);
    cH2D(q->dvv, mesh_read_get_vert(mesh), 3 * q->nv);
}
/* end src/rig/imp/ini.h */
/* begin src/rig/imp/fin.h */
void rig_fin(RigQuants *q) {
    EFREE(q->pp_hst);
    EFREE(q->ss_hst);
    EFREE(q->rr0_hst);
    EFREE(q->i_pp_hst);

    Dfree(q->pp);
    Dfree(q->ss);
    Dfree(q->rr0);
    Dfree(q->i_pp);

    Dfree(q->dtt);
    Dfree(q->dvv);

    EFREE(q->ss_dmp);
}
/* end src/rig/imp/fin.h */
/* begin src/rig/imp/common.h */
enum {X, Y, Z};

static void update_r_hst(const float *rr0, const int n, const float *com, const float *e0, const float *e1, const float *e2, /**/ Particle *pp) {
    for (int ip = 0; ip < n; ++ip) {
        float *r0 = pp[ip].r;
        const float* ro = &rr0[3*ip];
        float x = ro[X], y = ro[Y], z = ro[Z];
        r0[X] = x*e0[X] + y*e1[X] + z*e2[X];
        r0[Y] = x*e0[Y] + y*e1[Y] + z*e2[Y];
        r0[Z] = x*e0[Z] + y*e1[Z] + z*e2[Z];

        r0[X] += com[X]; r0[Y] += com[Y]; r0[Z] += com[Z];
    }
}

static void reinit_ft_hst(const int nsolid, /**/ Rigid *ss) {
    for (int i = 0; i < nsolid; ++i) {
        Rigid *s = ss + i;

        s->fo[X] = s->fo[Y] = s->fo[Z] = 0;
        s->to[X] = s->to[Y] = s->to[Z] = 0;
    }
}

static void generate_hst(const Rigid *ss_hst, const int ns, const float *rr0, const int nps, /**/ Particle *pp) {
    int j, start;
    for (j = start = 0; j < ns; ++j) {
        update_r_hst(rr0, nps, ss_hst[j].com, ss_hst[j].e0, ss_hst[j].e1, ss_hst[j].e2, /**/ pp + start);
        start += nps;
    }
}

static void mesh2pp_hst(const Rigid *ss_hst, const int ns, int nv, const float *vv, /**/ Particle *pp) {
    int i, j;
    for (j = 0; j < ns; ++j) {
        const Rigid *s = ss_hst + j;
        update_r_hst(vv, nv, s->com, s->e0, s->e1, s->e2, /**/ pp + j * nv);

        for (i = 0; i < nv; ++i) {
            float *v = pp[j*nv + i].v;
            v[X] = v[Y] = v[Z] = 0;
        }
    }
}

static void gen_pp_hst(const int ns, const float *rr0_hst, const int nps, /**/ Rigid *ss_hst, Particle *pp_hst) {
    generate_hst(ss_hst, ns, rr0_hst, nps, /**/ pp_hst);
    reinit_ft_hst(ns, /**/ ss_hst);
}

static void gen_ipp_hst(const Rigid *ss_hst, const int ns, int nv, const float *vv, Particle *i_pp_hst) {
    mesh2pp_hst(ss_hst, ns, nv, vv, /**/ i_pp_hst);
}

static void cpy_H2D(const RigQuants *q) {
    cH2D(q->i_pp, q->i_pp_hst, q->ns * q->nv);
    cH2D(q->ss,   q->ss_hst,   q->ns);
    cH2D(q->rr0,  q->rr0_hst,  q->nps * 3);
    cH2D(q->pp,   q->pp_hst,   q->n);
}
/* end src/rig/imp/common.h */
/* begin src/rig/imp/generate.h */
static void check(const float e[]) {
    enum {X, Y, Z};
    float ee, err;
    ee = e[X]*e[X] + e[Y]*e[Y] + e[Z]*e[Z];
    err = fabs(ee - 1.f);
    if (err > 1e-3)
        ERR("rigid ic: Matrix should be rotation only. Do not support scaling\n"
            "error: %g, e = %g %g %g", err, e[X], e[Y], e[Z]);
}

static Rigid gen_from_matrix(const double *A) {
    enum {X, Y, Z, W, M};
    enum {D = W};
    int c;
    Rigid s; memset(&s, 0, sizeof(s));
    for (c = 0; c < D; ++c) {
        s.com[c] = A[M * c + W];
        s.e0[c]  = A[M * c + X];
        s.e1[c]  = A[M * c + Y];
        s.e2[c]  = A[M * c + Z];
    }
    UC(check(s.e0));
    UC(check(s.e1));
    UC(check(s.e2));
    return s;
}

static void gen_from_matrices(Matrices *matrices, Rigid *ss) {
    int i, n;
    double *A;
    n = matrices_get_n(matrices);

    for (i = 0; i < n; ++i) {
        matrices_get(matrices, i, &A);
        ss[i] = gen_from_matrix(A);
    }
}

static void shift(const Coords *c, int n, Rigid *ss) {
    enum {X, Y, Z};
    float *r;
    int i;
    for (i = 0; i < n; ++i) {
        r = ss[i].com;
        r[X] = xg2xl(c, r[X]);
        r[Y] = yg2yl(c, r[Y]);
        r[Z] = zg2zl(c, r[Z]);
    }
}

void rig_gen_mesh(const Coords *coords, MPI_Comm comm, const MeshRead *mesh, const char *ic, /**/ RigQuants *q) {
    const float *vv;
    int n, nm, nv;
    Matrices *matrices;
    nv = mesh_read_get_nv(mesh);
    vv = mesh_read_get_vert(mesh);

    UC(matrices_read_filter(ic, coords, /**/ &matrices));

    q->ns = nm = matrices_get_n(matrices);
    if (nm > MAX_SOLIDS)
        ERR("nm=%d >= MAX_SOLIDS=%d", nm, MAX_SOLIDS);

    UC(mesh_gen_from_matrices(nv, vv, matrices, /**/ &n, q->i_pp_hst));
    UC(mesh_shift(coords, n, q->i_pp_hst));

    UC(gen_from_matrices(matrices, q->ss_hst));
    UC(shift(coords, nm, q->ss_hst));

    if (n)  cH2D(q->i_pp, q->i_pp_hst, n);
    if (nm) cH2D(q->ss, q->ss_hst, nm);

    UC(matrices_fin(matrices));
}

void rig_gen_freeze(const Coords *coords, bool empty_pp, int numdensity, float rig_mass, const RigPinInfo *pi, MPI_Comm comm,
                    const MeshRead *mesh, /* io */ Particle *opp, int *on, /**/ RigQuants *q) {
    RigGenInfo rgi;
    FluInfo fluinfo;
    RigInfo riginfo;

    rgi.mass = rig_mass;
    rgi.numdensity = numdensity;
    rgi.pi = pi;
    rgi.nt = q->nt;
    rgi.nv = q->nv;
    rgi.tt = q->dtt;
    rgi.pp = q->i_pp;
    rgi.empty_pp = empty_pp;
    rgi.mesh = mesh;

    fluinfo.pp = opp;
    fluinfo.n = on;

    riginfo.ns = q->ns;
    riginfo.nps = &q->nps;
    riginfo.rr0 = q->rr0_hst;
    riginfo.ss = q->ss_hst;
    riginfo.pp = q->pp_hst;

    UC(rig_gen_from_solvent(coords, comm, rgi, /*io*/ fluinfo, /**/ riginfo));
    gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, mesh_read_get_vert(mesh), /**/ q->i_pp_hst);
    cpy_H2D(q);
}
/* end src/rig/imp/generate.h */
/* begin src/rig/imp/start.h */
#define PP ".pp"
#define SS ".ss"

static void gen_code(const char *name, const char *ext, char *code) {
    strcpy(code, name);
    strcat(code, ext);
}

static void pp2rr(const Particle *pp, const int n, float *rr) {
    int i, c;
    for (i = 0; i < n; ++i)
        for ( c = 0; c < 3; ++c)
            rr[3*i + c] = pp[i].r[c];
}

static void gen_from_strt(int maxp, MPI_Comm comm, const char *base, const char *name, const int id, int *ns, int *nps, int *n, float *rr0_hst, Rigid *ss_hst) {
    Particle *pp;
    char code[FILENAME_MAX];

    EMALLOC(maxp, &pp);
    gen_code(name, PP, code);
    restart_read_pp(comm, base, code, RESTART_TEMPL, nps, pp);
    pp2rr(pp, *nps, rr0_hst);
    EFREE(pp);

    gen_code(name, SS, code);
    restart_read_ss(comm, base, code, id, ns, ss_hst);
    *n = *ns * (*nps);
}

void rig_strt_quants(MPI_Comm comm, const MeshRead *mesh, const char *base, const char *name, const int id, RigQuants *q) {
    gen_from_strt(q->maxp, comm, base, name, id, /**/ &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst);
    gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, mesh_read_get_vert(mesh), /**/ q->i_pp_hst);
    cpy_H2D(q);
}

static void rr2pp(const float *rr, const int n, Particle *pp) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) {
        pp[i].r[c] = rr[3*i + c];
        pp[i].v[c] = 0;
    }
}

static void strt_dump_templ0(MPI_Comm comm, const char *base, const char *name, const int nps, const float *rr0_hst) {
    Particle *pp;
    char code[FILENAME_MAX];
    EMALLOC(nps, &pp);
    rr2pp(rr0_hst, nps, pp);

    gen_code(name, PP, code);
    restart_write_pp(comm, base, code, RESTART_TEMPL, nps, pp);

    EFREE(pp);
}

void rig_strt_dump_templ(MPI_Comm comm, const char *base, const char *name, const RigQuants *q) {
    strt_dump_templ0(comm, base, name, q->nps, q->rr0_hst);
}

void rig_strt_dump(MPI_Comm comm, const char *base, const char *name, const int id, const RigQuants *q) {
    char code[FILENAME_MAX];
    gen_code(name, SS, code);
    restart_write_ss(comm, base, code, id, q->ns, q->ss_hst);
}

#undef PP
#undef SS
/* end src/rig/imp/start.h */
/* end src/rig/imp.cpp */

#line 1 "src/rigid/conf.cpp"
