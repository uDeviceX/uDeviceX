// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/sim/objects/imp.cpp */
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
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

/* begin src/inc/type.h */
#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
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
/* begin src/inc/def.h */
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
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

/* begin src/d/api.h */
#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
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
#endif
/* begin src/mpi/wrapper.h */
namespace m {
int Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Barrier(MPI_Comm comm);
int Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int Cancel(MPI_Request *request);
int Cart_rank(MPI_Comm comm, const int coords[], int *rank);
int Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart);
int Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
int Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]);
int Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int Comm_free(MPI_Comm *comm);
int Comm_rank(MPI_Comm comm, int *rank);
int Comm_size( MPI_Comm comm, int *size );
int Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
int File_close(MPI_File *fh);
int File_get_position(MPI_File fh, MPI_Offset *offset);
int File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);
int File_seek(MPI_File fh, MPI_Offset offset, int whence);
int File_set_size(MPI_File fh, MPI_Offset size);
int File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
int Finalize(void);
int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
           int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
            const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm);
int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
              int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
               const int displs[], MPI_Datatype recvtype, MPI_Comm comm);
int Init(int *argc, char ***argv);
int Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
int Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
int Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
int Type_commit(MPI_Datatype *type);
int Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);
int Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int Error_string(int errorcode, char *string, int *resultlen);

/* local */
bool is_master(MPI_Comm comm);
int is_success(int errorcode);
int is_pending(int errorcode);
int is_err_in_status(int errorcode);
int status2errcode(MPI_Status *status);
}
/* end src/mpi/wrapper.h */

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
/* begin src/utils/os.h */
// tag::int[]
void os_mkdir(const char* path); // <1>
long os_time();                  // <2>
void os_srand(long int seedval); // <3>
double os_drand();               // <4>
void os_sleep(unsigned int seconds); // <5>
void os_malloc_stats();              // TODO
// end::int[]
/* end src/utils/os.h */
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
/* begin src/utils/mc.h */
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
/* end src/utils/mc.h */
/* begin src/dbg/imp.h */
struct Dbg;

struct Config;
struct Force;
struct Particle;
struct int3;
struct Coords;

// tag::kind[]
enum {
    DBG_POS,      // <1>
    DBG_POS_SOFT, // <2>
    DBG_VEL,      // <3>
    DBG_FORCES,   // <4>
    DBG_COLORS,   // <5>
    DBG_CLIST,    // <6>
    DBG_NKIND_
};
// end::kind[]

// tag::mem[]
void dbg_ini(Dbg**);
void dbg_fin(Dbg*);
// end::mem[]

// tag::ini[]
void dbg_enable(int kind, Dbg *dbg);
void dbg_disable(int kind, Dbg *dbg);
void dbg_set_verbose(bool, Dbg *dbg);
void dbg_set_dump(bool, Dbg *dbg);
// end::ini[]

// tag::cnf[]
void dbg_set_conf(const Config*, Dbg*);
// end::cnf[]

// tag::int[]
void dbg_check_pos(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_pos_soft(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_vel(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_forces(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp, const Force *ff);
void dbg_check_colors(const Coords *c, const Dbg *dbg, int n, const int *cc);
void dbg_check_clist(const Coords *c, const Dbg *dbg, int3 L, const int *starts, const int *counts, const Particle *pp);
// end::int[]
/* end src/dbg/imp.h */

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
/* begin src/algo/convert/imp.h */
struct Particle;
struct Positioncp;
struct float4;

void convert_pp2rr_current (long n, const Particle *pp, Positioncp *rr);
void convert_pp2rr_previous(long n, const Particle *pp, Positioncp *rr);
void convert_pp2f4_pos(int n, const float *pp, /**/ float4 *zpp);
/* end src/algo/convert/imp.h */

/* begin src/coords/ini.h */
struct Coords;
struct Config;

// tag::mem[]
void coords_ini(MPI_Comm cart, int Lx, int Ly, int Lz, Coords **c); // <1>
void coords_fin(Coords *c); // <2>
// end::mem[]

// tag::cnf[]
void coords_ini_conf(MPI_Comm cart, const Config *cfg, Coords **c);
// end::cnf[]

/* end src/coords/ini.h */
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

/* begin src/pair/imp.h */
struct PairParams;
struct Config;

struct PairDPD;
struct PairDPDC;
struct PairDPDCM;
struct PairDPDLJ;
struct PairAdhesion;

// tag::mem[]
void pair_ini(PairParams **);
void pair_fin(PairParams *);
// end::mem[]

// tag::set[]
void pair_set_dpd(int ncol, const float a[], const float g[], float spow, PairParams *p); // <1>
void pair_set_lj(float sigma, float eps, PairParams *p); // <2>
// end::set[]
void pair_set_adhesion(float k1, float k2, PairParams *p);

// tag::cnf[]
void pair_set_conf(const Config *, const char *name_space, PairParams *);
// end::cnf[]

// tag::upd[]
void pair_compute_dpd_sigma(float kBT, float dt, PairParams *p);
// end::upd[]

// tag::view[]
void pair_get_view_dpd(const PairParams*, PairDPD*);
void pair_get_view_dpd_color(const PairParams*, PairDPDC*);
void pair_get_view_dpd_mirrored(const PairParams*, PairDPDCM*);
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
// end::view[]
void pair_get_view_adhesion(const PairParams*, PairAdhesion*);
/* end src/pair/imp.h */
/* begin src/struct/parray/imp.h */
#ifndef UDX_SIM_OBJECTS_PARRAY_COMMON
#define UDX_SIM_OBJECTS_PARRAY_COMMON
struct PaArray {
    bool colors;
    const float *pp;
    const int *cc;
};

struct PaArray_v;
struct PaCArray_v;

struct Particle;

// tag::push[]
void parray_push_pp(const Particle *pp, PaArray *a);
void parray_push_cc(const int *cc, PaArray *a);
// end::push[]

// tag::get[]
bool parray_is_colored(const PaArray *a);
// end::get[]

// tag::view[]
void parray_get_view(const PaArray *a, PaArray_v *v);  // <1>
void parray_get_view(const PaArray *a, PaCArray_v *v); // <2>
// end::view[]
/* end src/struct/parray/imp.h */
#endif
/* begin src/struct/farray/imp.h */
#ifndef UDX_SIM_OBJECTS_FARRAY_COMMON
#define UDX_SIM_OBJECTS_FARRAY_COMMON
struct FoArray {
    bool stress;
    float *ff;
    float *ss; /* stress */
};

struct FoArray_v;
struct FoSArray_v;

struct Force;

// tag::push[]
void farray_push_ff(Force *ff, FoArray *a);
void farray_push_ss(float *ss, FoArray *a);
// end::push[]

void farray_clear(FoArray *);

// tag::get[]
bool farray_has_stress(const FoArray *a);
// end::get[]

// tag::view[]
void farray_get_view(const FoArray *a, FoArray_v *v);  // <1>
void farray_get_view(const FoArray *a, FoSArray_v *v); // <2>
// end::view[]
/* end src/struct/farray/imp.h */
#endif
/* begin src/struct/pfarrays/imp.h */
#ifndef UDX_SIM_OBJECTS_PFARRAYS_COMMON
#define UDX_SIM_OBJECTS_PFARRAYS_COMMON
struct PFarray {
    PaArray p;
    FoArray f;
    long n;
};

struct PFarrays;

struct PaArray;
struct FoArray;

void pfarrays_ini(PFarrays**);
void pfarrays_fin(PFarrays*);

void pfarrays_clear(PFarrays*);
void pfarrays_push(PFarrays*, long n, PaArray, FoArray);
void pfarrays_push(PFarrays*, PFarray);
int  pfarrays_size(const PFarrays*);
void pfarrays_get(int i, const PFarrays*, long *n, PaArray*, FoArray*);
void pfarrays_get(int i, const PFarrays*, PFarray*);
/* end src/struct/pfarrays/imp.h */
#endif
/* begin src/struct/partlist/type.h */
enum {
    ALIVE = 0,
    DEAD  = 1
};

/* structure of particle                           */
/* optional "deathlist" for particles to be killed */

// tag::type[]
struct PartList {
    const Particle *pp;   /* particles array          */
    const int *deathlist; /* what pp is dead or alive */
};
// end::type[]
/* end src/struct/partlist/type.h */
/* begin src/mesh/triangles/type.h */
// tag::interface[]
struct Triangles {
    int4 *tt;
    int nt;
};
// end::interface[]
/* end src/mesh/triangles/type.h */
/* begin src/mesh/triangles/imp.h */
struct Triangles;
struct MeshRead;

// tag::interface[]
void triangles_ini(MeshRead*, /**/ Triangles**);
void triangles_fin(Triangles*);
// end::interface[]
/* end src/mesh/triangles/imp.h */
/* begin src/clist/imp.h */
struct Clist;
struct ClistMap;

// tag::mem[]
void clist_ini(int LX, int LY, int LZ, /**/ Clist **c);
void clist_fin(/**/ Clist *c);

void clist_ini_map(int maxp, int nA, const Clist *c, /**/ ClistMap **m);
void clist_fin_map(ClistMap *m);
// end::mem[]

// tag::access[]
const int* clists_get_ss(const Clist *c);
const int* clists_get_cc(const Clist *c);
int3       clists_get_dim(const Clist *c);
int        clists_get_n(const Clist *c);
const uint* clist_get_ids(const ClistMap *m);
// end::access[]

// tag::int[]
void clist_ini_counts(Clist *c);
void clist_subindex(bool project, int aid, int n, const PartList lp, /**/ Clist *c, ClistMap *m);
void clist_build_map(const int nn[], /**/ Clist *c, ClistMap *m);

/* special for fluid distribution */
void clist_subindex_local(int n, const PartList lp, /**/ Clist *c, ClistMap *m);
void clist_subindex_remote(int n, const PartList lp, /**/ Clist *c, ClistMap *m);

void clist_gather_pp(const Particle *pplo, const Particle *ppre, const ClistMap *m, long nout, /**/ Particle *ppout);
void clist_gather_ii(const int *iilo, const int *iire, const ClistMap *m, long nout, /**/ int *iiout);

/* quick cell build for single array */
void clist_build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, ClistMap *m);
// end::int[]
/* end src/clist/imp.h */

/* begin src/algo/minmax/imp.h */
struct Particle;
struct float3;
// tag::interface[]
void minmax(const Particle *pp, int nv, int nobj, /**/ float3 *lo, float3 *hi);
// end::interface[]
/* end src/algo/minmax/imp.h */

/* begin src/conf/imp.h */
struct Config;
struct float3;
struct int3;

// tag::mem[]
void conf_ini(/**/ Config **c);
void conf_fin(/**/ Config *c);
// end::mem[]

// tag::ini[]
void conf_read(int argc, char **argv, /**/ Config *cfg);
// end::ini[]

// tag::lookup[]
void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);
// end::lookup[]

// tag::ns[]
void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);
// end::ns[]

// tag::opt[]
bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);
// end::opt[]

// tag::set[]
void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);
// end::set[]

// tag::dump[]
void conf_write_exe(const Config *cfg, FILE *stream);     // <1>
void conf_write_history(const Config *cfg, FILE *stream); // <2>
// end::dump[]
/* end src/conf/imp.h */
/* begin src/io/com/imp.h */
struct Coords;
struct float3;
void io_com_dump(MPI_Comm, const Coords*, const char *name, long id, int n, const int *ii, const float3 *rr);
/* end src/io/com/imp.h */
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
/* begin src/io/mesh/imp.h */
#ifndef UDX_SIM_MESH_IO_COMMON
#define UDX_SIM_MESH_IO_COMMON
struct Coords;
struct Particle;
struct int4;
struct MeshWrite;
struct MeshRead;
struct Vectors;

enum {MESH_WRITE_EDGE, MESH_WRITE_CENTER};

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite**);
void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead*, const char *directory, /**/ MeshWrite**);

void mesh_write_particles(MeshWrite*, MPI_Comm, const Coords*, int nm, const Particle*, int id);
void mesh_write_vectors(MeshWrite*, MPI_Comm, int nm, Vectors *pos, Vectors *vel, int id);

void mesh_write_fin(MeshWrite*);
/* end src/io/mesh/imp.h */
#endif
/* begin src/io/restart/imp.h */
#ifndef UDX_SIM_RESTART_COMMON
#define UDX_SIM_RESTART_COMMON
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
#endif
/* begin src/io/rig/imp.h */
struct Coords;
struct Rigid;

void io_rig_dump(MPI_Comm comm, const Coords *c, const char *name, long id, int ns, const Rigid *ss);

/* end src/io/rig/imp.h */
/* begin src/io/bop/imp.h */
struct IoBop;

void io_bop_ini(MPI_Comm comm, int maxp, IoBop **t);
void io_bop_fin(IoBop *t);

void io_bop_parts       (MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const char *name, int id, IoBop *t);
void io_bop_parts_forces(MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const Force *ff, const char *name, int id, IoBop *t);
void io_bop_stresses    (MPI_Comm cart, long n, const float *ss, const char *name, int id, IoBop *t);
void io_bop_ids         (MPI_Comm cart, long n, const int   *ii, const char *name, int id, IoBop *t);
void io_bop_colors      (MPI_Comm cart, long n, const int   *cc, const char *name, int id, IoBop *t);
/* end src/io/bop/imp.h */

/* begin src/rbc/params/imp.h */
struct Config;
struct RbcParams;
struct RbcParams_v {
    float gammaC, gammaT, kBT;
    float kb, phi;
    float ks, x0, mpow;
    float ka, kd, kv;
    float totArea, totVolume;
};

// tag::mem[]
void rbc_params_ini(RbcParams **);
void rbc_params_fin(RbcParams *);
// end::mem[]

// tag::set[]
void rbc_params_set_fluct(float gammaC, float gammaT, float kBT, RbcParams *);
void rbc_params_set_bending(float kb, float phi, RbcParams *);
void rbc_params_set_spring(float ks, float x0, float mpow , RbcParams *);
void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *);
void rbc_params_set_tot_area_volume(float totArea, float totVolume, RbcParams *);
// end::set[]

// tag::get[]
double rbc_params_get_tot_volume(const RbcParams *);
// end::get[]

// tag::cnf[]
void rbc_params_set_conf(const Config *c, const char *name, RbcParams *);
// end::cnf[]

// tag::view[]
RbcParams_v rbc_params_get_view(const RbcParams *);
// end::view[]
/* end src/rbc/params/imp.h */
/* begin src/rbc/type.h */
struct AreaVolume;
struct Particle;

// tag::quants[]
struct RbcQuants {
    int n, nc;               /* number of particles, cells                */
    int nt, nv;              /* number of triangles and vertices per mesh */
    int md;                  /* maximum valence of the mesh               */
    Particle *pp, *pp_hst;   /* vertices particles on host and device     */
    bool ids;                /* global ids active ?                       */
    int *ii;                 /* global ids on host                        */
    AreaVolume *area_volume; /* to compute area and volume */
};
// end::quants[]

/* end src/rbc/type.h */
/* begin src/rbc/imp.h */
struct Coords;
struct MeshRead;

// tag::mem[]
void rbc_ini(long maxnc, bool ids, const MeshRead*, RbcQuants*);
void rbc_fin(RbcQuants*);
// end::mem[]

// tag::ini[]
void rbc_gen_mesh(const Coords*, MPI_Comm, const MeshRead*, const char *ic, /**/ RbcQuants*); // <1>
void rbc_gen_freeze(MPI_Comm, /**/ RbcQuants*);                                               // <2>
void rbc_strt_quants(MPI_Comm, const MeshRead*, const char *base, const char *name, const int id, RbcQuants*);  // <3>
void rbc_strt_dump(MPI_Comm, const char *base, const char *name, int id, const RbcQuants *q); // <4>
// end::ini[]

/* end src/rbc/imp.h */
/* begin src/rbc/force/imp.h */
struct RbcParams;
struct RbcForce;
struct Force;
struct Config;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void rbc_force_ini(const MeshRead*, RbcForce**);
void rbc_force_fin(RbcForce*);
// end::mem[]

// tag::set[]
void rbc_force_set_stressful(int nt, float totArea, /**/ RbcForce*); // <1>
void rbc_force_set_stressfree(const char *fname, /**/ RbcForce*);    // <2>

void rbc_force_set_rnd0(RbcForce*);           // <3>
void rbc_force_set_rnd1(int seed, RbcForce*); // <4>
void rbc_force_set_bending(const MeshRead*, const char*, RbcForce*);
// end::set[]

// tag::cnf[]
void rbc_force_set_conf(const MeshRead*, const Config *cfg, const char *name, RbcForce*);
// end::cnf[]

// tag::apply[]
void rbc_force_apply(RbcForce*, const RbcParams*, float dt, const RbcQuants*, /**/ Force*); // <1>
void rbc_force_stat(/**/ float *pArea, float *pVolume); // <2>
// end::apply[]
/* end src/rbc/force/imp.h */
/* begin src/rbc/force/area_volume/imp.h */
struct AreaVolume;
struct Particle;
struct int4;

// tag::interface[]
void area_volume_ini(int nv, int nt, const int4 *tt, int max_cell, AreaVolume**); // <1>
void area_volume_fin(AreaVolume*);

void area_volume_compute(AreaVolume*, int nm, const Particle*, /**/ float **av); // <2>
void area_volume_host(AreaVolume*, /**/ float **av); // <3>
// end::interface[]
/* end src/rbc/force/area_volume/imp.h */
/* begin src/rbc/stretch/imp.h */
struct RbcStretch;
struct Force;

// tag::interface[]
void rbc_stretch_ini(const char *path, int nv, /**/ RbcStretch**); /* `nv` is for error check */
void rbc_stretch_fin(RbcStretch*);
void rbc_stretch_apply(int nm, const RbcStretch*, /**/ Force*);
// end::interface[]
/* end src/rbc/stretch/imp.h */
/* begin src/rbc/com/imp.h */
struct RbcCom;
struct float3;
struct Particle;

// tag::interface[]
void rbc_com_ini(int nv, int nm_max, /**/ RbcCom**);
void rbc_com_fin(/**/ RbcCom*);
void rbc_com_apply(RbcCom*, int nm, const Particle*, /**/ float3 **rr, float3 **vv);
// end::interface[]
/* end src/rbc/com/imp.h */

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
/* begin src/rigid/imp.h */
struct RigPinInfo;
struct int3;
struct Config;

// tag::enum[]
enum {NOT_PERIODIC = -1};
// end::enum[]

// tag::pin[]
void rig_pininfo_ini(RigPinInfo **); // <1>
void rig_pininfo_fin(RigPinInfo *); // <2>
void rig_pininfo_set(int3 com, int3 axis, RigPinInfo *); // <3>
void rig_pininfo_set_pdir(int pdir, RigPinInfo *); // <4>
void rig_pininfo_set_conf(const Config *cfg, const char *ns, RigPinInfo *); // <5>

int rig_pininfo_get_pdir(const RigPinInfo *); // <6>
// end::pin[]

struct Particle;
struct Rigid;
struct Force;

// tag::upd[]
void rig_reinit_ft(const int ns, /**/ Rigid *ss); // <1>
void rig_update(const RigPinInfo *pi, float dt, int n, const Force *ff, const float *rr0, int ns, /**/ Particle *pp, Rigid *ss); // <2>
void rig_generate(int ns, const Rigid *ss, int nps, const float *rr0, /**/ Particle *pp); // <3>
void rig_update_mesh(float dt, int ns, const Rigid *ss, int nv, const float *vv, /**/ Particle *pp); // <4>
// end::upd[]

/* end src/rigid/imp.h */

/* begin src/wall/imp.h */
#ifndef UDX_SIM_WALLS_COMMON
#define UDX_SIM_WALLS_COMMON
struct PairParams;
struct PaArray;
struct FoArray;
struct WvelStep;
struct Coords;
struct Sdf;
struct int3;
struct float4;
struct Config;

// tag::struct[]
struct WallQuants {
    float4 *pp; /* particle positions xyzo xyzo ... */
    int n;      /* number of particles              */
    int3 L;     /* subdomain size                   */
};

struct WallTicket;
struct WallRepulsePrm;

// end::struct[]

// tag::mem[]
void wall_ini_quants(int3 L, WallQuants*);
void wall_ini_ticket(int3 L, WallTicket**);

void wall_fin_quants(WallQuants*);
void wall_fin_ticket(WallTicket*);
// end::mem[]

// tag::gen[]
void wall_gen_quants(MPI_Comm, int numdensity, int maxn, const Sdf*, /* io */ int *o_n, Particle *o_pp, /**/ WallQuants*); // <1>
void wall_gen_ticket(const WallQuants*, WallTicket*); // <2>
// end::gen[]

// tag::start[]
void wall_strt_quants(MPI_Comm, const char *base, int maxn, WallQuants*); // <1>
void wall_strt_dump_templ(MPI_Comm, const char *base, const WallQuants*); // <2>
// end::start[]

// tag::int[]
void wall_force(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                const WallTicket*, int n, const PaArray*, const FoArray*); // <1>

void wall_force_adhesion(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                         const WallTicket*, int n, const PaArray*, const FoArray*); // <2>

void wall_repulse(const Sdf*, const WallRepulsePrm*, long n, const PaArray*, const FoArray*); // <3>
// end::int[]

// tag::repulse[]
void wall_repulse_prm_ini(float lambda, WallRepulsePrm**); // <1>
void wall_repulse_prm_ini_conf(const Config*, const char *ns, WallRepulsePrm**); // <2>
void wall_repulse_prm_fin(WallRepulsePrm*); // <3>
// end::repulse[]
/* end src/wall/imp.h */
#endif

/* begin src/scheme/move/imp.h */
struct Force;
struct Particle;

// tag::int[]
void scheme_move_apply(float dt, float mass, int n, const Force*, Particle*); // <1>
void scheme_move_clear_vel(int n, Particle*); // <2>
// end::int[]
/* end src/scheme/move/imp.h */
/* begin src/scheme/force/imp.h */
struct Particle;
struct Force;
struct BForce;
struct float3;
struct Config;
struct Coords;

// tag::mem[]
void bforce_ini(BForce **p);
void bforce_fin(BForce *p);
// end::mem[]

// tag::set[]
void bforce_set_none(/**/ BForce *p);            // <1>
void bforce_set_cste(float3 f, /**/ BForce *p);  // <2>
void bforce_set_dp(float a, /**/ BForce *p);     // <3>
void bforce_set_shear(float a, /**/ BForce *p);  // <4>
void bforce_set_rol(float a, /**/ BForce *p);    // <5>
void bforce_set_rad(float a, /**/ BForce *p);    // <6>
// end::set[]

// tag::cnf[]
void bforce_set_conf(const Config *cfg, /**/ BForce *bf);
// end::cnf[]

// tag::interface[]
void bforce_adjust(float3 f, /**/ BForce *fpar); // <1>
void bforce_apply(const Coords *c, float mass, const BForce *bf, int n, const Particle *pp, /**/ Force *ff); // <2>
// end::interface[]
/* end src/scheme/force/imp.h */
/* begin src/scheme/time_step/imp.h */
struct TimeStep;
struct TimeStepAccel;
struct Force;
struct Config;

// tag::amem[]
void time_step_accel_ini(/**/ TimeStepAccel**);
void time_step_accel_fin(TimeStepAccel*);
// end::amem[]

// tag::aint[]
void time_step_accel_push(TimeStepAccel*, float m, int n, Force*); // <1>
void time_step_accel_reset(TimeStepAccel*); // <2>
// end::aint[]


// tag::mem[]
void time_step_ini(const Config*, /**/ TimeStep**); // <1>
void time_step_fin(TimeStep*);
// end::mem[]

// tag::int[]
float time_step_dt(TimeStep*, MPI_Comm, TimeStepAccel*); // <1>
float time_step_dt0(TimeStep*);                          // <2>
void time_step_log(TimeStep*);                           // <3>
// end::int[]
/* end src/scheme/time_step/imp.h */

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
/* begin src/distr/rbc/imp.h */
struct DRbcPack;
struct DRbcComm;
struct DRbcUnpack;

// tag::interface[]
void drbc_pack_ini(bool ids, int3 L, int maxc, int nv, DRbcPack **p);
void drbc_comm_ini(MPI_Comm comm, /**/ DRbcComm **c);
void drbc_unpack_ini(bool ids, int3 L, int maxc, int nv, DRbcUnpack **u);

void drbc_pack_fin(DRbcPack *p);
void drbc_comm_fin(DRbcComm *c);
void drbc_unpack_fin(DRbcUnpack *u);

void drbc_build_map(int nc, int nv, const Particle *pp, DRbcPack *p);
void drbc_pack(const RbcQuants *q, /**/ DRbcPack *p);
void drbc_download(DRbcPack *p);

void drbc_post_recv(DRbcComm *c, DRbcUnpack *u);
void drbc_post_send(DRbcPack *p, DRbcComm *c);
void drbc_wait_recv(DRbcComm *c, DRbcUnpack *u);
void drbc_wait_send(DRbcComm *c);

void drbc_unpack_bulk(const DRbcPack *p, /**/ RbcQuants *q);
void drbc_unpack_halo(const DRbcUnpack *u, /**/ RbcQuants *q);
// end::interface[]
/* end src/distr/rbc/imp.h */
/* begin src/distr/rig/imp.h */
struct DRigPack;
struct DRigComm;
struct DRigUnpack;

// tag::interface[]
void drig_pack_ini(int3 L, int maxns, int nv, DRigPack **p);
void drig_comm_ini(MPI_Comm comm, /**/ DRigComm **c);
void drig_unpack_ini(int3 L, int maxns, int nv, DRigUnpack **u);

void drig_pack_fin(DRigPack *p);
void drig_comm_fin(DRigComm *c);
void drig_unpack_fin(DRigUnpack *u);

void drig_build_map(int ns, const Rigid *ss, /**/ DRigPack *p);
void drig_pack(int ns, int nv, const Rigid *ss, const Particle *ipp, /**/ DRigPack *p);
void drig_download(DRigPack *p);

void drig_post_recv(DRigComm *c, DRigUnpack *u);
void drig_post_send(DRigPack *p, DRigComm *c);
void drig_wait_recv(DRigComm *c, DRigUnpack *u);
void drig_wait_send(DRigComm *c);

void drig_unpack_bulk(const DRigPack *p, /**/ RigQuants *q);
void drig_unpack_halo(const DRigUnpack *u, /**/ RigQuants *q);
// end::interface[]
/* end src/distr/rig/imp.h */
/* begin src/exch/mesh/imp.h */
struct EMeshPack;
struct EMeshComm;
struct EMeshUnpack;
struct EMeshPackM;
struct EMeshCommM;
struct EMeshUnpackM;

/* mesh exchanger */

// tag::mem[]
void emesh_pack_ini(int3 L, int nv, int max_mesh_num, EMeshPack **p);
void emesh_comm_ini(MPI_Comm comm, /**/ EMeshComm **c);
void emesh_unpack_ini(int3 L, int nv, int max_mesh_num, EMeshUnpack **u);

void emesh_pack_fin(EMeshPack *p);
void emesh_comm_fin(EMeshComm *c);
void emesh_unpack_fin(EMeshUnpack *u);
// end::mem[]

// tag::map[]
void emesh_build_map(int nm, int nv, const Particle *pp, /**/ EMeshPack *p);
// end::map[]

// tag::pack[]
void emesh_pack_pp(int nv, const Particle *pp, /**/ EMeshPack *p);
void emesh_pack_rrcp(int nv, const Positioncp *rr, /**/ EMeshPack *p);
void emesh_download(EMeshPack *p);
// end::pack[]

// tag::com[]
void emesh_post_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_post_send(EMeshPack *p, EMeshComm *c);
void emesh_wait_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_wait_send(EMeshComm *c);
// end::com[]

// tag::unpack[]
void emesh_unpack_pp(int nv, const EMeshUnpack *u, /**/ int *nmhalo, Particle *pp);
void emesh_unpack_rrcp(int nv, const EMeshUnpack *u, /**/ int *nmhalo, Positioncp *rr);
// end::unpack[]

// tag::get[]
void emesh_get_num_frag_mesh(const EMeshUnpack *u, /**/ int cc[NFRAGS]);
// end::get[]

/* optional: (back) momentum exchanger */

// tag::memback[]
void emesh_packm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshPackM **p);
void emesh_commm_ini(MPI_Comm comm, /**/ EMeshCommM **c);
void emesh_unpackm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshUnpackM **u);

void emesh_packm_fin(EMeshPackM *p);
void emesh_commm_fin(EMeshCommM *c);
void emesh_unpackm_fin(EMeshUnpackM *u);
// end::memback[]

// tag::packback[]
void emesh_packM(int nt, const int counts[NFRAGS], const Momentum *mm, /**/ EMeshPackM *p);
void emesh_downloadM(const int counts[NFRAGS], EMeshPackM *p);
// end::packback[]

// tag::comback[]
void emesh_post_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_post_send(EMeshPackM *p, EMeshCommM *c);
void emesh_wait_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_wait_send(EMeshCommM *c);
// end::comback[]

// tag::unpackback[]
void emesh_upload(EMeshUnpackM *u);
void emesh_unpack_mom(int nt, const EMeshPack *p, const EMeshUnpackM *u, /**/ Momentum *mm);
// end::unpackback[]
/* end src/exch/mesh/imp.h */
//#include "meshbb/imp.h"
/* begin src/mesh_bounce/imp.h */
struct MeshBB;

struct Particle;
struct Positioncp;
struct Force;
struct Momentum;
struct Rigid;
struct int4;

struct MeshInfo {
    int nt, nv;
    const int4 *tt;
};

// tag::mem[]
void mesh_bounce_ini(int maxpp, /**/ MeshBB **d);
void mesh_bounce_fin(/**/ MeshBB *d);
// end::mem[]

// tag::int[]
void mesh_bounce_reini(int n, /**/ MeshBB *d);  // <1>
void mesh_bounce_find_collisions(float dt, int nm, MeshInfo meshinfo, const Positioncp *i_rr, int3 L,
                                 const int *starts, const int *counts, const Particle *pp, const Particle *pp0, /**/ MeshBB *d);  // <2>
void mesh_bounce_select_collisions(int n, /**/ MeshBB *d);  // <3>
void mesh_bounce_bounce(float dt, float mass, int n, const MeshBB *d, MeshInfo meshinfo,
                        const Positioncp *i_rr, const Particle *pp0, /**/ Particle *pp, Momentum *mm);  // <4>
// end::int[]

// tag::collect[]
void mesh_bounce_collect_rig_momentum(int ns, MeshInfo meshinfo, const Particle *pp, const Momentum *mm, /**/ Rigid *ss); // <1>
void mesh_bounce_collect_rbc_momentum(float mass, int nc, MeshInfo mi, const Momentum *mm, /**/ Particle *pp);            // <2>
// end::collect[]
/* end src/mesh_bounce/imp.h */
/* begin src/mesh/collision/imp.h */
struct float3;
struct Particle;
struct int4;
struct Triangles;

int collision_inside_1p(int spdir, const float *r, const float *vv, const int4 *tt, const int nt);

void collision_label_ini(long n, int lab_out, /**/ int *labels);
void collision_label(int pdir, int n, const Particle *pp, const Triangles *tri,
                     int nv, int nm, const Particle *i_pp,
                     const float3 *minext, const float3 *maxext,
                     int lab_in, /**/ int *labels);
/* end src/mesh/collision/imp.h */

/* begin src/wall/sdf/imp.h */
struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

// tag::mem[]
void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);
// end::mem[]

// tag::int[]
void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*); // <1>
void sdf_get_view(const Sdf*, /**/ Sdf_v*);                  // <2>
// end::int[]

// tag::tools[]
void sdf_bulk_wall(const Sdf*, /*io*/ int *s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp); // <1>
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv, /**/ int *stay);          // <2>
double sdf_compute_volume(MPI_Comm, int3 L, const Sdf*, long nsamples);                          // <3>
// end::tools[]

// tag::bounce[]
void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n, /**/ Particle*);
// end::bounce[]
/* end src/wall/sdf/imp.h */

/* begin src/sim/opt/imp.h */
#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON
/* solvent options */
struct OptFlu {
    bool colors, ids, ss, push;
};

/* common parameters for membranes and rigid objects */
struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};

/* membrane options */
struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};

/* rigid options */
struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};

/* global physical parameters */
struct OptParams {
    int3 L;          /* subdomain sizes */
    float kBT;       /* temperature     */
    int numdensity;  /* number density  */
};

struct OptDump {
    bool       parts,      mesh,      field,      strt, forces;
    float freq_parts, freq_mesh, freq_field, freq_strt, freq_diag;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
};

struct Opt {
    OptFlu flu;
    OptMbr mbr[MAX_MBR_TYPES];
    OptRig rig[MAX_RIG_TYPES];
    int nmbr, nrig;
    OptWall wall;
    OptParams params;
    OptDump dump;
    bool inflow, outflow, denoutflow, vcon;
    int recolor_freq;
    int sampler_npdump;
    int3 sampler_grid_ref;
    bool restart;
    bool tracers;
};

struct Config;

void opt_read(const Config*, Opt*);
void opt_check(const Opt*);
long opt_estimate_maxp(const Opt*);
/* end src/sim/opt/imp.h */
#endif

/* begin src/sim/objects/imp.h */
struct Objects;

struct Sdf;
struct Config;
struct Coords;
struct Opt;
struct PFarray;
struct PFarrays;
struct BForce;
struct TimeStepAccel;
struct Dbg;
struct IoBop;
struct PairParams;
struct WallRepulsePrm;

// tag::mem[]
void objects_ini(const Config*, const Opt*, MPI_Comm, const Coords*, int maxp, Objects**);
void objects_fin(Objects*);
// end::mem[]

void objects_save_mesh(Objects*);
// tag::upd[]
void objects_clear_vel(Objects*);         // <1>
void objects_advance(float dt, Objects*); // <2>
void objects_distribute(Objects*);        // <3>
void objects_update_dpd_prms(float dt, float kBT, Objects*); // <4>
// end::upd[]

// tag::force[]
void objects_clear_forces(Objects*);                 // <1>
void objects_body_forces(const BForce*, Objects *o); // <2>
// end::force[]

// tag::dump[]
void objects_mesh_dump(Objects*);                    // <1>
void objects_diag_dump(float t, Objects*);           // <2>
void objects_part_dump(long id, Objects*, IoBop*);   // <3>

void objects_strt_templ(const char *base, Objects*);         // <4>
void objects_strt_dump(const char *base, long id, Objects*); // <5>
// end::dump[]

// tag::get[]
bool objects_have_bounce(const Objects*);

void objects_get_particles_all(Objects*, PFarrays*);    // <1>
void objects_get_particles_mbr(Objects*, PFarrays*);    // <2>
void objects_get_accel(const Objects*, TimeStepAccel*); // <3>

void objects_get_params_fsi(const Objects*, const PairParams*[]);           // <4>
void objects_get_params_adhesion(const Objects*, const PairParams*[]);      // <5>
void objects_get_params_repulsion(const Objects*, const WallRepulsePrm*[]); // <6>
void objects_get_params_cnt(const Objects*, const PairParams***);           // <7>
// end::get[]

// tag::gen[]
void objects_gen_mesh(Objects*);                           // <1>
void objects_remove_from_wall(const Sdf *sdf, Objects *o); // <2>
void objects_gen_freeze(PFarray*, Objects*);               // <3>
// end::gen[]

// tag::strt[]
void objects_restart(Objects*);
// end::strt[]

// tag::tools[]
void objects_activate_mbr_bounce(Objects*);
void objects_bounce(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, Objects *obj); // <1>
void objects_recolor_flu(Objects*, PFarray *flu); // <2>
double objects_mbr_tot_volume(const Objects*);    // <3>
// end::tools[]

// tag::check[]
void objects_check_size(const Objects*);
void objects_check_vel(const Objects*, const Dbg*, float dt);
void objects_check_forces(const Objects*, const Dbg*, float dt);
// end::check[]

/* end src/sim/objects/imp.h */
/* begin src/sim/objects/imp/type.h */
/* membrane distribution */
struct MbrDistr {
    DRbcPack *p;
    DRbcComm *c;
    DRbcUnpack *u;
};

/* rigid distribution */
struct RigDistr {
    DRigPack *p;
    DRigComm *c;
    DRigUnpack *u;
};

/* mesh exchanger */
struct MeshExch {
    EMeshPack *p;
    EMeshComm *c;
    EMeshUnpack *u;
};

/* mesh momentum exchanger */
struct MeshMomExch {
    EMeshPackM *p;
    EMeshCommM *c;
    EMeshUnpackM *u;
};

/* workspace for solvent coloring */
struct Colorer {
    Particle *pp_mesh; /* particle workspace */
    float3 *lo, *hi;   /* bounding boxes     */
};

/* workspace for mesh bounce back */
struct BounceBackData {
    Positioncp *rr_cp;
    MeshMomExch *e;
    Momentum *mm;
};

/* common parts for membranes and rigid objects */
struct Obj {
    char name[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    float mass;           /* mass of one particle                      */
    MeshRead   *mesh;     /* cell template                             */
    MeshWrite  *mesh_write;
    MeshExch   *mesh_exch;
    BounceBackData *bbdata;

    PairParams *fsi;
    PairParams *adhesion;
    WallRepulsePrm *wall_rep_prm;
};

/* data holder for cell membranes */
struct Mbr : Obj {
    RbcQuants q;
    MbrDistr d;
    Force *ff;            /* large time scale forces                   */
    Force *ff_fast;       /* small time scale forces                   */
    RbcForce   *force;    /* helper to compute membrane forces         */
    RbcParams  *params;   /* model parameters                          */
    RbcCom     *com;      /* helper to compute center of masses        */
    RbcStretch *stretch;  /* helper to apply stretching force to cells */
    Triangles  *tri;      /* triangles for one cell on devices         */

    bool active_bounce;
    Colorer  *colorer;
};

/* data holder for rigid objects */
struct Rig : Obj {
    RigQuants q;
    Force *ff;

    RigPinInfo *pininfo;
    RigDistr d;
};

struct Dump {
    Particle *pp; /* workspace on host */
    long id, id_diag;
};

struct Objects {
    int nmbr, nrig;
    Mbr **mbr;
    Rig **rig;
    Opt opt;
    Dump *dump;
    Coords *coords;
    MPI_Comm cart;
    MeshBB *bb;
    bool active; /* false before completely generated */
    PairParams **cnt_params;
};
/* end src/sim/objects/imp/type.h */
/* begin src/sim/objects/imp/ini.h */
static void gen_name_mesh_dir(const char *name, char *mesh_dir) {
    strcpy(mesh_dir, "ply/");
    strcat(mesh_dir, name);
}

static void ini_mbr_distr(bool ids, int nv, MPI_Comm comm, int3 L, /**/ MbrDistr *d) {
    UC(drbc_pack_ini(ids, L, MAX_CELL_NUM, nv, /**/ &d->p));
    UC(drbc_comm_ini(comm, /**/ &d->c));
    UC(drbc_unpack_ini(ids, L, MAX_CELL_NUM, nv, /**/ &d->u));
}

static void ini_rig_distr(int nv, MPI_Comm comm, int3 L, /**/ RigDistr *d) {
    UC(drig_pack_ini(L, MAX_SOLIDS, nv, /**/ &d->p));
    UC(drig_comm_ini(comm, /**/ &d->c));
    UC(drig_unpack_ini(L,MAX_SOLIDS, nv, /**/ &d->u));
}

static void ini_mesh_exch(int3 L, int nv, int max_m, MPI_Comm comm, /**/ MeshExch **me) {
    MeshExch *e;
    EMALLOC(1, me);
    e = *me;
    UC(emesh_pack_ini(L, nv, max_m, /**/ &e->p));
    UC(emesh_comm_ini(comm, /**/ &e->c));
    UC(emesh_unpack_ini(L, nv, max_m, /**/ &e->u));
}

static void ini_mesh_mom_exch(int nt, int max_m, MPI_Comm comm, /**/ MeshMomExch **mme) {
    MeshMomExch *e;
    EMALLOC(1, mme);
    e = *mme;
    UC(emesh_packm_ini(nt, max_m, /**/ &e->p));
    UC(emesh_commm_ini(comm, /**/ &e->c));
    UC(emesh_unpackm_ini(nt, max_m, /**/ &e->u));
}

static void ini_bbdata(int nt, int nv, int max_m, MPI_Comm cart, /**/ BounceBackData **bbdata) {
    BounceBackData *bb;
    EMALLOC(1, bbdata);
    bb = *bbdata;
    UC(ini_mesh_mom_exch(nt, max_m, cart, &bb->e));
    Dalloc(&bb->mm,    max_m * nt);
    Dalloc(&bb->rr_cp, max_m * nv);
}

static void ini_colorer(int nv, int max_m, /**/ Colorer **col) {
    Colorer *c;
    EMALLOC(1, col);
    c = *col;
    Dalloc(&c->pp_mesh, nv * max_m);
    Dalloc(&c->lo, max_m);
    Dalloc(&c->hi, max_m);
}

static void read_params(const Config *cfg, const char *ns, PairParams **par) {
    PairParams *p;
    UC(pair_ini(par));
    p = *par;
    UC(pair_set_conf(cfg, ns, p));
}

static void ini_params(const Config *cfg, const char *name, const char *pair, PairParams **par) {
    const char *ns;
    UC(conf_lookup_string_ns(cfg, name, pair, &ns));

    if (same_str(ns, "none")) *par = NULL;
    else UC(read_params(cfg, ns, par));
}

static void ini_repulsion_params(const Config *cfg, const char *name, WallRepulsePrm **par) {
    const char *ns;
    UC(conf_lookup_string_ns(cfg, name, "repulsion", &ns));

    if (same_str(ns, "none")) *par = NULL;
    else UC(wall_repulse_prm_ini_conf(cfg, ns, par));
}

static void read_mesh(const char *filename, MeshRead **mesh) {
    const char *ext;
    ext = get_filename_ext(filename);

    if      (same_str(ext, "off"))
        UC(mesh_read_ini_off(filename, mesh));
    else if (same_str(ext, "ply"))
        UC(mesh_read_ini_ply(filename, mesh));
    else
        ERR("%s : unrecognised extension <%s>\n", filename, ext);
}

static void ini_obj_common(const Config *cfg, const OptObj *opt, MPI_Comm cart, int max_m, int3 L, Obj *obj) {
    char mesh_dir[FILENAME_MAX];
    int nt, nv;

    strcpy(obj->name, opt->name);
    strcpy(obj->ic_file, opt->ic_file);
    gen_name_mesh_dir(obj->name, mesh_dir);

    UC(read_mesh(opt->templ_file, &obj->mesh));
    UC(mesh_write_ini_from_mesh(cart, opt->shifttype, obj->mesh, mesh_dir, /**/ &obj->mesh_write));

    obj->mass = opt->mass;

    UC(ini_params(cfg, obj->name, "fsi", &obj->fsi));
    UC(ini_params(cfg, obj->name, "adhesion", &obj->adhesion));
    UC(ini_repulsion_params(cfg, obj->name, &obj->wall_rep_prm));

    nt = mesh_read_get_nt(obj->mesh);
    nv = mesh_read_get_nv(obj->mesh);

    obj->mesh_exch = NULL;
    obj->bbdata    = NULL;

    if (opt->bounce) {
        UC(ini_bbdata(nt, nv, max_m, cart, /**/ &obj->bbdata));
        UC(ini_mesh_exch(L, nv, max_m, cart, /**/ &obj->mesh_exch));
    }
}

static void ini_mbr(const Config *cfg, const OptMbr *opt, MPI_Comm cart, int3 L,
                    bool recolor, /**/ Mbr **membrane) {
    int nv, max_m;
    Mbr *m;
    EMALLOC(1, membrane);
    m = *membrane;

    max_m = MAX_CELL_NUM;

    m->com       = NULL;
    m->stretch   = NULL;
    m->colorer   = NULL;
    m->active_bounce = false;

    UC(ini_obj_common(cfg, opt, cart, max_m, L, m));

    nv = mesh_read_get_nv(m->mesh);

    Dalloc(&m->ff,      max_m * nv);
    Dalloc(&m->ff_fast, max_m * nv);

    UC(triangles_ini(m->mesh, /**/ &m->tri));
    UC(rbc_ini(max_m, opt->ids, m->mesh, &m->q));
    UC(ini_mbr_distr(opt->ids, nv, cart, L, /**/ &m->d));

    if (opt->dump_com) UC(rbc_com_ini(nv, max_m, /**/ &m->com));
    if (opt->stretch)  UC(rbc_stretch_ini(opt->stretch_file, nv, /**/ &m->stretch));

    UC(rbc_params_ini(&m->params));
    UC(rbc_params_set_conf(cfg, m->name, m->params));

    UC(rbc_force_ini(m->mesh, /**/ &m->force));
    UC(rbc_force_set_conf(m->mesh, cfg, m->name, m->force));

    if (recolor) {
        if (!m->mesh_exch)
            UC(ini_mesh_exch(L, nv, max_m, cart, /**/ &m->mesh_exch));
        UC(ini_colorer(nv, max_m, /**/ &m->colorer));
    }
}

static void ini_rig(const Config *cfg, const OptRig *opt, MPI_Comm cart, int maxp, int3 L, /**/ Rig **rigid) {
    Rig *r;
    long max_m = MAX_SOLIDS;
    int nv;
    EMALLOC(1, rigid);
    r = *rigid;

    UC(ini_obj_common(cfg, opt, cart, max_m, L, r));

    UC(rig_ini(max_m, maxp, r->mesh, &r->q));

    Dalloc(&r->ff, maxp);

    nv = r->q.nv;

    UC(ini_rig_distr(nv, cart, L, /**/ &r->d));

    UC(rig_pininfo_ini(&r->pininfo));
    UC(rig_pininfo_set_conf(cfg, r->name, r->pininfo));
}

static void ini_dump(long maxp, Dump **dump) {
    Dump *d;
    EMALLOC(1, dump);
    d = *dump;
    EMALLOC(maxp, &d->pp);
    d->id = d->id_diag = 0;
}

static bool need_bb(int nm, int nr, const OptMbr *m, const OptRig *r) {
    int i;
    for (i = 0; i < nm; ++i) if (m[i].bounce) return true;
    for (i = 0; i < nr; ++i) if (r[i].bounce) return true;
    return false;
}

static void get_all_names(int nmbr, int nrig, Mbr **mm, Rig **rr, const char **names) {
    int i, j = 0;
    for (i = 0; i < nmbr; ++i) names[j++] = mm[i]->name;
    for (i = 0; i < nrig; ++i) names[j++] = rr[i]->name;
}

enum {NOT_FOUND=-1};

static int get_name_id(const char *s, int n, const char **names) {
    int i;
    for (i = 0; i < n; ++i)
        if (same_str(s, names[i]))
            return i;
    ERR("object <%s> does not exist\n", s);
    return NOT_FOUND;
}

static int get_id_inter(int a, int b) {
    int i, j;
    i = a > b ? a : b;
    j = a > b ? b : a;
    return j + i*(i+1)/2;
}

static void ini_cnt_params(const Config *cfg, int nmbr, int nrig, Mbr **mm, Rig **rr, PairParams ***prms_ptr) {
    int i, n, ninter, ninter_read;
    int id_obja, id_objb, id_inter;
    const char **names, **prms_ns, *sobja, *sobjb, *pair_ns;
    PairParams **prms;

    n = nmbr + nrig;
    ninter = n * (n+1) / 2;

    EMALLOC(ninter, &prms_ns);
    EMALLOC(ninter, prms_ptr);
    EMALLOC(n, &names);

    prms = *prms_ptr;
    get_all_names(nmbr, nrig, mm, rr, names);

    UC(conf_lookup_vstring(cfg, "contact", ninter, &ninter_read, prms_ns));

    for (i = 0; i < ninter; ++i) prms[i] = NULL;

    for (i = 0; i < ninter_read; ++i) {
        UC(conf_lookup_string_ns(cfg, prms_ns[i], "obja", &sobja));
        UC(conf_lookup_string_ns(cfg, prms_ns[i], "objb", &sobjb));
        UC(conf_lookup_string_ns(cfg, prms_ns[i], "pair", &pair_ns));

        id_obja = get_name_id(sobja, n, names);
        id_objb = get_name_id(sobjb, n, names);


        id_inter = get_id_inter(id_obja, id_objb);

        read_params(cfg, pair_ns, &prms[id_inter]);
    }

    EFREE(prms_ns);
    EFREE(names);
}

void objects_ini(const Config *cfg, const Opt *opt, MPI_Comm cart, const Coords *coords, int maxp, Objects **objects) {
    Objects *obj;
    int3 L;
    bool recolor = opt->flu.colors;
    int i;
    EMALLOC(1, objects);
    obj = *objects;
    obj->opt = *opt;
    obj->active = false;

    MC(m::Comm_dup(cart, &obj->cart));
    L = subdomain(coords);
    UC(coords_ini(cart, L.x, L.y, L.z, &obj->coords));

    obj->bb  = NULL;

    obj->nmbr = opt->nmbr;
    obj->nrig = opt->nrig;

    if (obj->nmbr) EMALLOC(obj->nmbr, &obj->mbr);
    if (obj->nrig) EMALLOC(obj->nrig, &obj->rig);

    for (i = 0; i < obj->nmbr; ++i) UC(ini_mbr(cfg, &opt->mbr[i], cart, L, recolor, &obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(ini_rig(cfg, &opt->rig[i], cart, maxp, L, &obj->rig[i]));

    if (need_bb(opt->nmbr, opt->nrig, opt->mbr, opt->rig))
        UC(mesh_bounce_ini(maxp, /**/ &obj->bb));

    UC(ini_dump(maxp, &obj->dump));
    UC(ini_cnt_params(cfg, obj->nmbr, obj->nrig, obj->mbr, obj->rig, &obj->cnt_params));
}
/* end src/sim/objects/imp/ini.h */
/* begin src/sim/objects/imp/fin.h */
static void fin_mbr_distr(/**/ MbrDistr *d) {
    UC(drbc_pack_fin(/**/ d->p));
    UC(drbc_comm_fin(/**/ d->c));
    UC(drbc_unpack_fin(/**/ d->u));
}

static void fin_rig_distr(/**/ RigDistr *d) {
    UC(drig_pack_fin(/**/ d->p));
    UC(drig_comm_fin(/**/ d->c));
    UC(drig_unpack_fin(/**/ d->u));
}

static void fin_mesh_exch(/**/ MeshExch *e) {
    UC(emesh_pack_fin(   /**/ e->p));
    UC(emesh_comm_fin(   /**/ e->c));
    UC(emesh_unpack_fin( /**/ e->u));
    EFREE(e);
}

static void fin_mesh_mom_exch(/**/ MeshMomExch *e) {
    UC(emesh_packm_fin(   /**/ e->p));
    UC(emesh_commm_fin(   /**/ e->c));
    UC(emesh_unpackm_fin( /**/ e->u));
    EFREE(e);
}

static void fin_bbdata(/**/ BounceBackData *bb) {
    UC(fin_mesh_mom_exch(/**/ bb->e));
    Dfree(bb->mm);
    Dfree(bb->rr_cp);
    EFREE(bb);
}

static void fin_colorer(/**/ Colorer *c) {
    Dfree(c->pp_mesh);
    Dfree(c->lo);
    Dfree(c->hi);
    EFREE(c);
}

static void fin_obj_common(Obj *o) {
    UC(mesh_read_fin(o->mesh));
    UC(mesh_write_fin(o->mesh_write));

    if (o->fsi)      UC(pair_fin(o->fsi));
    if (o->adhesion) UC(pair_fin(o->adhesion));
    if (o->wall_rep_prm)
        UC(wall_repulse_prm_fin(o->wall_rep_prm));
}

static void fin_mbr(Mbr *m) {
    UC(rbc_fin(&m->q));
    UC(rbc_force_fin(m->force));

    UC(fin_mbr_distr(/**/ &m->d));

    Dfree(m->ff);
    Dfree(m->ff_fast);
    UC(triangles_fin(m->tri));

    if (m->com)     UC(rbc_com_fin(/**/ m->com));
    if (m->stretch) UC(rbc_stretch_fin(/**/ m->stretch));

    UC(rbc_params_fin(m->params));

    if (m->colorer)   UC(fin_colorer(/**/ m->colorer));
    if (m->mesh_exch) UC(fin_mesh_exch(/**/ m->mesh_exch));

    UC(fin_obj_common(m));

    EFREE(m);
}

static void fin_rig(Rig *r) {
    UC(rig_fin(&r->q));
    Dfree(r->ff);

    UC(fin_rig_distr(/**/ &r->d));
    UC(rig_pininfo_fin(r->pininfo));

    if (r->bbdata) UC(fin_bbdata(r->bbdata));

    UC(fin_obj_common(r));

    EFREE(r);
}

static void fin_dump(Dump *d) {
    EFREE(d->pp);
    EFREE(d);
}

static void fin_cnt_params(int nmbr, int nrig, PairParams **prms) {
    int i, n, ninter;
    n = nmbr + nrig;
    ninter = n * (n+1) / 2;
    for (i = 0; i < ninter; ++i) {
        if (prms[i])
            UC(pair_fin(prms[i]));
    }
    EFREE(prms);
}

void objects_fin(Objects *obj) {
    int i;
    for (i = 0; i < obj->nmbr; ++i) UC(fin_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(fin_rig(obj->rig[i]));
    if (obj->nmbr) EFREE(obj->mbr);
    if (obj->nrig) EFREE(obj->rig);
    if (obj->bb)  UC(mesh_bounce_fin(/**/ obj->bb));
    UC(fin_cnt_params(obj->nmbr, obj->nrig, obj->cnt_params));
    UC(fin_dump(obj->dump));
    UC(coords_fin(obj->coords));
    MC(m::Comm_free(&obj->cart));
    EFREE(obj);
}
/* end src/sim/objects/imp/fin.h */
/* begin src/sim/objects/imp/check.h */
static void check_size(long n, long nmax) {
    if (n < 0 || n > nmax)
        ERR("wrong size: %ld / %ld", n, nmax);
}

static void check_size_mbr(const Mbr *m) {
    UC(check_size(m->q.nc, MAX_CELL_NUM));
}

static void check_size_rig(const Rig *r) {
    UC(check_size(r->q.ns, MAX_SOLIDS));
}

void objects_check_size(const Objects *obj) {
    int i;
    for (i = 0; i < obj->nmbr; ++i) UC(check_size_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(check_size_rig(obj->rig[i]));
}

static void check_vel_mbr(const Dbg *dbg, float dt, const Coords *coords, const Mbr *m) {
    UC(dbg_check_vel(dt, coords, "rbc", dbg, m->q.n, m->q.pp));
}

static void check_vel_rig(const Dbg *dbg, float dt, const Coords *coords, const Rig *r) {
    UC(dbg_check_vel(dt, coords, "rig", dbg, r->q.n, r->q.pp));
}

void objects_check_vel(const Objects *o, const Dbg *dbg, float dt) {
    int i;
    for (i = 0; i < o->nmbr; ++i) UC(check_vel_mbr(dbg, dt, o->coords, o->mbr[i]));
    for (i = 0; i < o->nrig; ++i) UC(check_vel_rig(dbg, dt, o->coords, o->rig[i]));
}

static void check_forces_mbr(const Dbg *dbg, float dt, const Coords *coords, const Mbr *m) {
    UC(dbg_check_forces(dt, coords, "rbc.ff", dbg, m->q.n, m->q.pp, m->ff));
}

static void check_forces_rig(const Dbg *dbg, float dt, const Coords *coords, const Rig *r) {
    UC(dbg_check_forces(dt, coords, "rig.ff", dbg, r->q.n, r->q.pp, r->ff));
}

void objects_check_forces(const Objects *o, const Dbg *dbg, float dt) {
    int i;
    for (i = 0; i < o->nmbr; ++i) UC(check_forces_mbr(dbg, dt, o->coords, o->mbr[i]));
    for (i = 0; i < o->nrig; ++i) UC(check_forces_rig(dbg, dt, o->coords, o->rig[i]));
}
/* end src/sim/objects/imp/check.h */
/* begin src/sim/objects/imp/forces.h */
static void clear_forces(int n, Force* ff) {
    if (n) DzeroA(ff, n);
}

static void clear_mbr_forces(Mbr *m) {
    UC(clear_forces(m->q.n, m->ff));
}

static void clear_rig_forces(Rig *r) {
    UC(clear_forces(r->q.n, r->ff));
    UC(rig_reinit_ft(r->q.ns, /**/ r->q.ss));
}

void objects_clear_forces(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(clear_mbr_forces(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(clear_rig_forces(obj->rig[i]));
}

// used in main.h
static void internal_forces_mbr(float dt, const OptMbr *opt, Mbr *m) {
    UC(rbc_force_apply(m->force, m->params, dt, &m->q, /**/ m->ff_fast));
    if (opt->stretch) UC(rbc_stretch_apply(m->q.nc, m->stretch, /**/ m->ff_fast));
}

static void bforce_mbr(const Coords *c, const BForce *bf, Mbr *m) {
    UC(bforce_apply(c, m->mass, bf, m->q.n, m->q.pp, /**/ m->ff));
}

static void bforce_rig(const Coords *c, const BForce *bf, Rig *r) {
    UC(bforce_apply(c, r->mass, bf, r->q.n, r->q.pp, /**/ r->ff));
}

void objects_body_forces(const BForce *bf, Objects *o) {
    int i;
    const Opt *opt = &o->opt;
    if (!o->active) return;

    for (i = 0; i < o->nmbr; ++i)
        if (opt->mbr[i].push)
            bforce_mbr(o->coords, bf, o->mbr[i]);

    for (i = 0; i < o->nrig; ++i)
        if (opt->rig[i].push)
            bforce_rig(o->coords, bf, o->rig[i]);
}
/* end src/sim/objects/imp/forces.h */
/* begin src/sim/objects/imp/main.h */
static void save_mesh_mbr(Mbr *m) {
    if (m->bbdata)
        convert_pp2rr_previous(m->q.n, m->q.pp, m->bbdata->rr_cp);
}

static void save_mesh_rig(Rig *r) {
    if (r->bbdata)
        convert_pp2rr_previous(r->q.ns * r->q.nv, r->q.i_pp, r->bbdata->rr_cp);
}

void objects_save_mesh(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(save_mesh_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(save_mesh_rig(obj->rig[i]));
}

static void clear_vel_mbr(Mbr *m) {
    UC(scheme_move_clear_vel(m->q.n, m->q.pp));
}

static void clear_vel_rig(Rig *r) {
    UC(scheme_move_clear_vel(r->q.n, r->q.pp));
}

void objects_clear_vel(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(clear_vel_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(clear_vel_rig(obj->rig[i]));
}

static void advance_rig(float dt, Rig *r) {
    if (!r->q.n) return;
    RigQuants *q = &r->q;
    UC(rig_update(r->pininfo, dt, q->n, r->ff, q->rr0, q->ns, /**/ q->pp, q->ss));
    UC(rig_update_mesh(dt, q->ns, q->ss, q->nv, q->dvv, /**/ q->i_pp));
}

static void ini_mbr_fast_forces(Mbr *m) {
    aD2D(m->ff_fast, m->ff, m->q.n);
}

static void sub_advance_mbr(float dt, const OptMbr *opt, Mbr *m) {
    UC(ini_mbr_fast_forces(m));
    UC(internal_forces_mbr(dt, opt, m));
    UC(scheme_move_apply(dt, m->mass, m->q.n, m->ff_fast, m->q.pp));
}

static void advance_mbr(float dt, const OptMbr *opt, Mbr *m) {
    int i, n = opt->substeps;
    float sub_dt = dt / n;
    for (i = 0; i < n; ++i)
        UC(sub_advance_mbr(sub_dt, opt, m));
}

void objects_advance(float dt, Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(advance_mbr(dt, &obj->opt.mbr[i], obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(advance_rig(dt, obj->rig[i]));
}

static void distribute_mbr(Mbr *m) {
    RbcQuants *q = &m->q;
    MbrDistr  *d = &m->d;

    UC(drbc_build_map(q->nc, q->nv, q->pp, /**/ d->p));
    UC(drbc_pack(q, /**/ d->p));
    UC(drbc_download(/**/d->p));

    UC(drbc_post_send(d->p, d->c));
    UC(drbc_post_recv(d->c, d->u));

    UC(drbc_unpack_bulk(d->p, /**/ q));

    UC(drbc_wait_send(d->c));
    UC(drbc_wait_recv(d->c, d->u));

    UC(drbc_unpack_halo(d->u, /**/ q));
    dSync();
}

static void distribute_rig(Rig *r) {
    RigQuants *q = &r->q;
    RigDistr  *d = &r->d;
    int nv = q->nv;

    UC(drig_build_map(q->ns, q->ss, /**/ d->p));
    UC(drig_pack(q->ns, nv, q->ss, q->i_pp, /**/ d->p));
    UC(drig_download(/**/d->p));

    UC(drig_post_send(d->p, d->c));
    UC(drig_post_recv(d->c, d->u));

    UC(drig_unpack_bulk(d->p, /**/ q));

    UC(drig_wait_send(d->c));
    UC(drig_wait_recv(d->c, d->u));

    UC(drig_unpack_halo(d->u, /**/ q));

    q->n = q->ns * q->nps;
    UC(rig_generate(q->ns, q->ss, q->nps, q->rr0, /**/ q->pp));
    dSync();
}

void objects_distribute (Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(distribute_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(distribute_rig(obj->rig[i]));
}

static void update_dpd_prms(float dt, float kBT, Obj *o) {
    if (o->fsi) UC(pair_compute_dpd_sigma(kBT, dt, /**/ o->fsi));
}

void objects_update_dpd_prms(float dt, float kBT, Objects *obj) {
    int i, n, ninter;
    for (i = 0; i < obj->nmbr; ++i) update_dpd_prms(dt, kBT, obj->mbr[i]);
    for (i = 0; i < obj->nrig; ++i) update_dpd_prms(dt, kBT, obj->rig[i]);

    n = obj->nmbr + obj->nrig;
    ninter = n*(n+1)/2;
    for (i = 0; i < ninter; ++i) {
        if (obj->cnt_params[i])
            UC(pair_compute_dpd_sigma(kBT, dt, /**/ obj->cnt_params[i]));
    }
}

bool objects_have_bounce(const Objects *o) {
    return o->bb != NULL;
}

static void get_mbr(Mbr *m, PFarrays *pf) {
    PaArray p;
    FoArray f;
    parray_push_pp(m->q.pp, &p);
    farray_push_ff(m->ff, &f);
    UC(pfarrays_push(pf, m->q.n, p, f));
}

static void get_rig(Rig *r, PFarrays *pf) {
    PaArray p;
    FoArray f;
    parray_push_pp(r->q.pp, &p);
    farray_push_ff(r->ff, &f);
    UC(pfarrays_push(pf, r->q.n, p, f));
}

void objects_get_particles_all(Objects *obj, PFarrays *pf) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) get_mbr(obj->mbr[i], pf);
    for (i = 0; i < obj->nrig; ++i) get_rig(obj->rig[i], pf);
}

void objects_get_particles_mbr(Objects *obj, PFarrays *pf) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) get_mbr(obj->mbr[i], pf);
}

static void get_mbr_accel(const Mbr *m, TimeStepAccel *aa) {
    if(m->q.n) UC(time_step_accel_push(aa, m->mass, m->q.n, m->ff));
}

static void get_rig_accel(const Rig *r, TimeStepAccel *aa) {
    if (r->q.n) UC(time_step_accel_push(aa, r->mass, r->q.n, r->ff));
}

void objects_get_accel(const Objects *obj, TimeStepAccel *aa) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) get_mbr_accel(obj->mbr[i], aa);
    for (i = 0; i < obj->nrig; ++i) get_rig_accel(obj->rig[i], aa);
}

static void get_params_fsi(const Obj *o, const PairParams **prm) {*prm = o->fsi;}

void objects_get_params_fsi(const Objects *obj, const PairParams *prms[]) {
    int i, j = 0;
    for (i = 0; i < obj->nmbr; ++i) get_params_fsi(obj->mbr[i], &prms[j++]);
    for (i = 0; i < obj->nrig; ++i) get_params_fsi(obj->rig[i], &prms[j++]);
}

static void get_params_adhesion(const Obj *o, const PairParams **prm) {*prm = o->adhesion;}

void objects_get_params_adhesion(const Objects *obj, const PairParams *prms[]) {
    int i, j = 0;
    for (i = 0; i < obj->nmbr; ++i) get_params_adhesion(obj->mbr[i], &prms[j++]);
    for (i = 0; i < obj->nrig; ++i) get_params_adhesion(obj->rig[i], &prms[j++]);
}

static void get_params_repulsion(const Obj *o, const WallRepulsePrm **prm) {*prm = o->wall_rep_prm;}

void objects_get_params_repulsion(const Objects *obj, const WallRepulsePrm *prms[]) {
    int i, j = 0;
    for (i = 0; i < obj->nmbr; ++i) get_params_repulsion(obj->mbr[i], &prms[j++]);
    for (i = 0; i < obj->nrig; ++i) get_params_repulsion(obj->rig[i], &prms[j++]);
}

void objects_get_params_cnt(const Objects *o, const PairParams ***prms) {
    *prms = (const PairParams**) o->cnt_params;
}

static void restart_mbr(MPI_Comm cart, const char *base, Mbr *m) {
    UC(rbc_strt_quants(cart, m->mesh, base, m->name, RESTART_BEGIN, &m->q));
}

static void restart_rig(MPI_Comm cart, const char *base, Rig *r) {
    UC(rig_strt_quants(cart, r->mesh, base, r->name, RESTART_BEGIN, &r->q));
}

void objects_restart(Objects *o) {
    int i;
    const char *base = o->opt.dump.strt_base_read;
    for (i = 0; i < o->nmbr; ++i) restart_mbr(o->cart, base, o->mbr[i]);
    for (i = 0; i < o->nrig; ++i) restart_rig(o->cart, base, o->rig[i]);
    o->active = true;
}

static double local_vol_mbr(const Mbr *m) {
    long nc;
    double V0;
    nc = m->q.nc;
    V0 = rbc_params_get_tot_volume(m->params);
    return nc * V0;
}

double objects_mbr_tot_volume(const Objects *o) {
    int i;
    double loc, tot;
    if (!o->active) return 0;

    loc = 0;

    for (i = 0; i < o->nmbr; ++i)
        loc += local_vol_mbr(o->mbr[i]);

    tot = 0;
    MC(m::Allreduce(&loc, &tot, 1, MPI_DOUBLE, MPI_SUM, o->cart));

    return tot;
}

static void activate_mbr_bounce(Mbr *m) {
    if (m->bbdata)
        m->active_bounce = true;
}

void objects_activate_mbr_bounce(Objects *o) {
    int i;
    for (i = 0; i < o->nmbr; ++i) activate_mbr_bounce(o->mbr[i]);
}
/* end src/sim/objects/imp/main.h */
/* begin src/sim/objects/imp/bounce.h */
static MeshInfo mesh_info_mbr(const Mbr *m) {
    MeshInfo mi;
    mi.nv = m->q.nv;
    mi.nt = m->q.nt;
    mi.tt = m->tri->tt;
    return mi;
}

static MeshInfo mesh_info_rig(const Rig *r) {
    MeshInfo mi;
    mi.nv = r->q.nv;
    mi.nt = r->q.nt;
    mi.tt = r->q.dtt;
    return mi;
}

static void mesh_pack_and_send_mbr(Mbr *m) {
    const RbcQuants *q = &m->q;
    MeshExch *e = m->mesh_exch;

    UC(emesh_build_map(q->nc, q->nv, q->pp, /**/ e->p));
    UC(emesh_pack_rrcp(q->nv, m->bbdata->rr_cp, /**/ e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));
}

static void mesh_pack_and_send_rig(Rig *r) {
    const RigQuants *q = &r->q;
    MeshExch  *e = r->mesh_exch;

    UC(emesh_build_map(q->ns, q->nv, q->i_pp, /**/ e->p));
    UC(emesh_pack_rrcp(q->nv, r->bbdata->rr_cp, /**/ e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));
}

static int mesh_recv_unpack_mbr(Mbr *m) {
    MeshExch *e = m->mesh_exch;
    RbcQuants *q = &m->q;
    int nmhalo;
    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    /* unpack at the end of current mesh buffer */
    UC(emesh_unpack_rrcp(q->nv, e->u, /**/ &nmhalo, m->bbdata->rr_cp + q->n));
    return nmhalo;
}

static int mesh_recv_unpack_rig(Rig *r) {
    MeshExch *e = r->mesh_exch;
    RigQuants *q = &r->q;
    int nmhalo;
    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    /* unpack at the end of current mesh buffer */
    UC(emesh_unpack_rrcp(q->nv, e->u, /**/ &nmhalo, r->bbdata->rr_cp + q->ns * q->nv));
    return nmhalo;
}

static void clear_momentum(long n, Momentum *mm) {
    if (n) DzeroA(mm, n);
}

static void clear_momentum_mbr(int nmhalo, Mbr *m) {
    long ntri = m->q.nt * (m->q.nc + nmhalo);
    UC(clear_momentum(ntri, m->bbdata->mm));
}

static void clear_momentum_rig(int nmhalo, Rig *r) {
    long ntri = r->q.nt * (r->q.ns + nmhalo);
    UC(clear_momentum(ntri, r->bbdata->mm));
}


static void find_and_select_collisions_rig(float dt, long nflu, const Clist *cflu, const Particle *flu_pp, const Particle *flu_pp0, int nm, Rig *r, MeshBB *bb) {
    MeshInfo mi = mesh_info_rig(r);
    UC(mesh_bounce_find_collisions(dt, nm, mi, r->bbdata->rr_cp,
                                   clists_get_dim(cflu),
                                   clists_get_ss(cflu),
                                   clists_get_cc(cflu),
                                   flu_pp, flu_pp0, /**/ bb));
    UC(mesh_bounce_select_collisions(nflu, /**/ bb));
}

static void find_and_select_collisions_mbr(float dt, long nflu, const Clist *cflu, const Particle *flu_pp, const Particle *flu_pp0, int nm, Mbr *m, MeshBB *bb) {
    MeshInfo mi = mesh_info_mbr(m);
    UC(mesh_bounce_find_collisions(dt, nm, mi, m->bbdata->rr_cp,
                                   clists_get_dim(cflu),
                                   clists_get_ss(cflu),
                                   clists_get_cc(cflu),
                                   flu_pp, flu_pp0, /**/ bb));
    UC(mesh_bounce_select_collisions(nflu, /**/ bb));
}

static void bounce_mbr(float dt, float flu_mass, const MeshBB *bb, long n, const Particle *pp0, Particle *pp, Mbr *m) {
    MeshInfo mi = mesh_info_mbr(m);
    UC(mesh_bounce_bounce(dt, flu_mass, n, bb, mi, m->bbdata->rr_cp, pp0, /**/ pp, m->bbdata->mm));
}

static void bounce_rig(float dt, float flu_mass, const MeshBB *bb, long n, const Particle *pp0, Particle *pp, Rig *r) {
    MeshInfo mi = mesh_info_rig(r);
    UC(mesh_bounce_bounce(dt, flu_mass, n, bb, mi, r->bbdata->rr_cp, pp0, /**/ pp, r->bbdata->mm));
}

static void mom_pack_and_send_mbr(Mbr *m) {
    const RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshMomExch *e = bb->e;
    int counts[NFRAGS];

    UC(emesh_get_num_frag_mesh(m->mesh_exch->u, /**/ counts));
    UC(emesh_packM(q->nt, counts, bb->mm + q->n, /**/ e->p));
    UC(emesh_downloadM(counts, /**/ e->p));

    UC(emesh_post_recv(e->c, e->u));
    UC(emesh_post_send(e->p, e->c));
}

static void mom_pack_and_send_rig(Rig *r) {
    const RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshMomExch *e = bb->e;
    int counts[NFRAGS];

    UC(emesh_get_num_frag_mesh(r->mesh_exch->u, /**/ counts));
    UC(emesh_packM(q->nt, counts, bb->mm + q->ns * q->nt, /**/ e->p));
    UC(emesh_downloadM(counts, /**/ e->p));

    UC(emesh_post_recv(e->c, e->u));
    UC(emesh_post_send(e->p, e->c));
}

static void mom_recv_unpack_mbr(Mbr *m) {
    const RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshMomExch *e = bb->e;
    MeshExch *meshe = m->mesh_exch;

    UC(emesh_wait_recv(e->c, e->u));
    UC(emesh_wait_send(e->c));

    UC(emesh_upload(e->u));
    UC(emesh_unpack_mom(q->nt, meshe->p, e->u, /**/ bb->mm));
}

static void mom_recv_unpack_rig(Rig *r) {
    const RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshMomExch *e = bb->e;
    MeshExch *meshe = r->mesh_exch;

    UC(emesh_wait_recv(e->c, e->u));
    UC(emesh_wait_send(e->c));

    UC(emesh_upload(e->u));
    UC(emesh_unpack_mom(q->nt, meshe->p, e->u, /**/ bb->mm));
}

static void collect_mom_mbr(float dt, Mbr *m) {
    RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshInfo mi = mesh_info_mbr(m);

    UC(mesh_bounce_collect_rbc_momentum(m->mass, q->nc, mi, bb->mm, /**/ q->pp));
}

static void collect_mom_rig(Rig *r) {
    RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshInfo mi = mesh_info_rig(r);

    UC(mesh_bounce_collect_rig_momentum(q->ns, mi, q->i_pp, bb->mm, /**/ q->ss));
}

/* TODO less brutal implementation */
static void bounce_mbr(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, MeshBB *bb, Mbr *m) {
    int nmhalo;
    if (!m->bbdata || !m->active_bounce) return;

    mesh_pack_and_send_mbr(m);
    nmhalo = mesh_recv_unpack_mbr(m);

    UC(mesh_bounce_reini(n, /**/ bb));
    clear_momentum_mbr(nmhalo, m);

    find_and_select_collisions_mbr(dt, n, flu_cells, flu_pp, flu_pp0, nmhalo + m->q.nc, m, bb);

    bounce_mbr(dt, flu_mass, bb, n, flu_pp0, /**/ flu_pp, m);

    mom_pack_and_send_mbr(m);
    mom_recv_unpack_mbr(m);
    collect_mom_mbr(dt, m);
}

static void bounce_rig(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, MeshBB *bb, Rig *r) {
    int nmhalo;
    if (!r->bbdata) return;

    mesh_pack_and_send_rig(r);
    nmhalo = mesh_recv_unpack_rig(r);

    UC(mesh_bounce_reini(n, /**/ bb));
    clear_momentum_rig(nmhalo, r);

    find_and_select_collisions_rig(dt, n, flu_cells, flu_pp, flu_pp0, nmhalo + r->q.ns, r, bb);

    bounce_rig(dt, flu_mass, bb, n, flu_pp0, /**/ flu_pp, r);

    mom_pack_and_send_rig(r);
    mom_recv_unpack_rig(r);
    collect_mom_rig(r);
}

static void save_mesh_mbr_current(Mbr *m) {
    if (m->bbdata && m->active_bounce)
        convert_pp2rr_current(m->q.n, m->q.pp, m->bbdata->rr_cp);
}

static void save_mesh_rig_current(Rig *r) {
    if (r->bbdata)
        convert_pp2rr_current(r->q.ns * r->q.nv, r->q.i_pp, r->bbdata->rr_cp);
}

static void objects_save_mesh_current(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(save_mesh_mbr_current(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(save_mesh_rig_current(obj->rig[i]));
}

void objects_bounce(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, Objects *obj) {
    MeshBB *bb = obj->bb;
    int i;

    if (!obj->active) return;
    if (!bb) return;

    UC(objects_save_mesh_current(obj));

    for (i = 0; i < obj->nmbr; ++i)
        bounce_mbr(dt, flu_mass, flu_cells, n, flu_pp0, flu_pp, bb, obj->mbr[i]);

    for (i = 0; i < obj->nrig; ++i)
        bounce_rig(dt, flu_mass, flu_cells, n, flu_pp0, flu_pp, bb, obj->rig[i]);
}
/* end src/sim/objects/imp/bounce.h */
/* begin src/sim/objects/imp/recolor.h */
static void reini_colors(PFarray *flu) {
    UC(collision_label_ini(flu->n, BLUE_COLOR, /**/ (int*) flu->p.cc));
}

static int exchange_mesh_mbr(Mbr *m, Particle *pp) {
    int nm, nv, nmhalo;
    const RbcQuants *q = &m->q;
    MeshExch *e = m->mesh_exch;
    nm = q->nc;
    nv = q->nv;

    UC(emesh_build_map(nm, nv, q->pp, /**/ e->p));
    UC(emesh_pack_pp(nv, q->pp, /**/ e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));

    if (nm*nv) aD2D(pp, q->pp, nm*nv);

    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    UC(emesh_unpack_pp(nv, e->u, /**/ &nmhalo, pp + nm * nv));
    return nm + nmhalo;
}

static void recolor_flu_from_mbr(Mbr *m, PFarray *flu) {
    int nm, nv;
    Colorer *c = m->colorer;
    if (c == NULL)
        ERR("colorer was not initialised");

    nv = m->q.nv;
    nm = exchange_mesh_mbr(m, c->pp_mesh);

    UC(minmax(c->pp_mesh, nv, nm, /**/ c->lo, c->hi));

    UC(collision_label(NOT_PERIODIC, flu->n, (const Particle*) flu->p.pp, m->tri, nv, nm,
                       c->pp_mesh, c->lo, c->hi, RED_COLOR, /**/ (int*) flu->p.cc));
}

void objects_recolor_flu(Objects *obj, PFarray *flu) {
    int i;
    if (!obj->active) return;
    if (!obj->nmbr) return;
    if (flu->n == 0) return;

    UC(reini_colors(flu));
    for (i = 0; i < obj->nmbr; ++i) UC(recolor_flu_from_mbr(obj->mbr[i], flu));
}
/* end src/sim/objects/imp/recolor.h */
/* begin src/sim/objects/imp/dump.h */
static void dump_mesh_mbr(MPI_Comm cart, const Coords *coords, Particle *pp, long id, Mbr *m) {
    cD2H(pp, m->q.pp, m->q.n);
    UC(mesh_write_particles(m->mesh_write, cart, coords, m->q.nc, pp, id));
}

static void dump_mesh_rig(MPI_Comm cart, const Coords *coords, Particle *pp, long id, Rig *r) {
    cD2H(pp, r->q.i_pp, r->q.ns * r->q.nv);
    UC(mesh_write_particles(r->mesh_write, cart, coords, r->q.ns, pp, id));
}

void objects_mesh_dump(Objects *obj) {
    int i;
    Dump *d = obj->dump;
    if (!obj->active) return;

    for (i = 0; i < obj->nmbr; ++i)
        UC(dump_mesh_mbr(obj->cart, obj->coords, d->pp, d->id, obj->mbr[i]));

    for (i = 0; i < obj->nrig; ++i)
        UC(dump_mesh_rig(obj->cart, obj->coords, d->pp, d->id, obj->rig[i]));
    ++ d->id;
}

static void dump_diag_mbr(MPI_Comm cart, const Coords *coords, Mbr *m, Dump *d) {
    RbcQuants *q = &m->q;
    long nc = q->nc;
    float3 *rr, *vv;
    if (m->com) {
        UC(rbc_com_apply(m->com, nc, q->pp, /**/ &rr, &vv));
        UC(io_com_dump(cart, coords, m->name, d->id_diag, nc, q->ii, rr));
    }
}

static void dump_diag_rig(MPI_Comm cart, const Coords *coords, Rig *r, Dump *d) {
    RigQuants *q = &r->q;
    cD2H(q->ss_dmp, q->ss, q->ns);
    UC(io_rig_dump(cart, coords, r->name, d->id_diag, q->ns, q->ss_dmp));
}

void objects_diag_dump(float t, Objects *obj) {
    int i;
    Dump *d = obj->dump;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) dump_diag_mbr(obj->cart, obj->coords, obj->mbr[i], d);
    for (i = 0; i < obj->nrig; ++i) dump_diag_rig(obj->cart, obj->coords, obj->rig[i], d);
    ++ d->id_diag;
}

static void dump_part_rig(MPI_Comm cart, const Coords *coords, long id, Rig *r, IoBop *bop) {
    RigQuants *q = &r->q;
    cD2H(q->pp_hst, q->pp, q->n);
    UC(io_bop_parts(cart, coords, q->n, q->pp_hst, r->name, id, bop));
}

void objects_part_dump(long id, Objects *o, IoBop *bop) {
    int i;
    if (!o->active) return;
    for (i = 0; i < o->nrig; ++i) dump_part_rig(o->cart, o->coords, id, o->rig[i], bop);
}

void objects_strt_templ(const char *base, Objects *o) {
    int i;
    if (!o->active) return;
    for (i = 0; i < o->nrig; ++i) UC(rig_strt_dump_templ(o->cart, base, o->rig[i]->name, &o->rig[i]->q));
}

void objects_strt_dump(const char *base, long id, Objects *o) {
    int i;
    if (!o->active) return;
    for (i = 0; i < o->nmbr; ++i) UC(rbc_strt_dump(o->cart, base, o->mbr[i]->name, id, &o->mbr[i]->q));
    for (i = 0; i < o->nrig; ++i) UC(rig_strt_dump(o->cart, base, o->rig[i]->name, id, &o->rig[i]->q));
}
/* end src/sim/objects/imp/dump.h */
/* begin src/sim/objects/imp/gen.h */
static void gen_mesh_mbr(Coords *coords, MPI_Comm cart, Mbr *m) {
    MeshRead *mesh = m->mesh;
    UC(rbc_gen_mesh(coords, cart, mesh, m->ic_file, /**/ &m->q));
}

static void gen_mesh_rig(Coords *coords, MPI_Comm cart, Rig *r) {
    MeshRead *mesh = r->mesh;
    UC(rig_gen_mesh(coords, cart, mesh, r->ic_file, /**/ &r->q));
}

void objects_gen_mesh(Objects *o) {
    int i;
    for (i = 0; i < o->nmbr; ++i) gen_mesh_mbr(o->coords, o->cart, o->mbr[i]);
    for (i = 0; i < o->nrig; ++i) gen_mesh_rig(o->coords, o->cart, o->rig[i]);
}

template <typename T>
static void remove(T *data, int nv, int *stay, int nc) {
    int c; /* c: cell index */
    for (c = 0; c < nc; c++)
        cA2A(data + nv*c, data + nv * stay[c], nv);
}

static void remove_mbr(const Sdf *sdf, Mbr *m) {
    int nc0, stay[MAX_CELL_NUM];
    RbcQuants *q = &m->q;
    q->nc = sdf_who_stays(sdf, q->n, q->pp, nc0 = q->nc, q->nv, /**/ stay);
    q->n = q->nc * q->nv;
    remove(q->pp, q->nv, stay, q->nc);
    msg_print("%s: %d/%d survived", m->name, q->nc, nc0);
}

static void remove_rig(const Sdf *sdf, Rig *r) {
    int nip, ns0, stay[MAX_SOLIDS];
    RigQuants *q = &r->q;
    nip = q->ns * q->nv;
    q->ns = sdf_who_stays(sdf, nip, q->i_pp, ns0 = q->ns, q->nv, /**/ stay);
    q->n  = q->ns * q->nps;
    remove(q->pp,       q->nps,      stay, q->ns);
    remove(q->pp_hst,   q->nps,      stay, q->ns);

    remove(q->ss,       1,           stay, q->ns);
    remove(q->ss_hst,   1,           stay, q->ns);

    remove(q->i_pp,     q->nv, stay, q->ns);
    remove(q->i_pp_hst, q->nv, stay, q->ns);
    msg_print("%s: %d/%d survived", r->name, q->ns, ns0);
}

void objects_remove_from_wall(const Sdf *sdf, Objects *o) {
    int i;
    if (!sdf) return;
    for (i = 0; i < o->nmbr; ++i) remove_mbr(sdf, o->mbr[i]);
    for (i = 0; i < o->nrig; ++i) remove_rig(sdf, o->rig[i]);
}

static void gen_freeze_mbr(MPI_Comm cart, Mbr *m) {
    UC(rbc_gen_freeze(cart, /**/ &m->q));
}

static void gen_freeze_rig(const Coords *coords, MPI_Comm cart, bool empty_pp, int numdensity, PFarray *flu, Rig *r) {
    const MeshRead *mesh = r->mesh;
    int n;
    Particle *pp = (Particle*) flu->p.pp;
    n = flu->n;

    UC(rig_gen_freeze(coords, empty_pp, numdensity, r->mass, r->pininfo,
                      cart, mesh, pp, &n, /**/ &r->q));

    flu->n = n;
}

void objects_gen_freeze(PFarray *flu, Objects *o) {
    int i;
    const Opt *opt = &o->opt;

    for (i = 0; i < o->nmbr; ++i)
        gen_freeze_mbr(o->cart, o->mbr[i]);

    for (i = 0; i < o->nrig; ++i)
        gen_freeze_rig(o->coords, o->cart, opt->rig[i].empty_pp, opt->params.numdensity, flu, o->rig[i]);

    o->active = true;
}
/* end src/sim/objects/imp/gen.h */
/* end src/sim/objects/imp.cpp */

#line 1 "src/sim/objinter/imp.cpp"
