// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/flu/imp.cpp */
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <mpi.h>

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
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
/* begin src/inc/def.h */
#ifndef UDX_FLU_HALO_DEF_COMMON
#define UDX_FLU_HALO_DEF_COMMON
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
/* begin src/inc/type.h */
#ifndef UDX_FLU_HALO_TYPE_COMMON
#define UDX_FLU_HALO_TYPE_COMMON
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

/* begin src/d/api.h */
#ifndef UDX_FLU_HALO_DEVICE_API_COMMON
#define UDX_FLU_HALO_DEVICE_API_COMMON
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

/* begin src/utils/mc.h */
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
/* end src/utils/mc.h */
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
/* begin src/algo/scan/imp.h */
struct Scan;

// tag::interface[]
void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Scan*); // <1>
void scan_ini(int size, /**/ Scan**);                                       // <2>
void scan_fin(Scan*);                                                       // <3>
// end::interface[]
/* end src/algo/scan/imp.h */
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
/* begin src/inter/color/imp.h */
struct GenColor;
struct Particle;
struct Coords;
struct Config;

// tag::mem[]
void inter_color_ini(GenColor**);
void inter_color_fin(GenColor*);
// end::mem[]

// tag::set[]
void inter_color_set_drop(float R, GenColor*); // <1>
void inter_color_set_uniform(GenColor*); // <2>
// end::set[]

// tag::cfg[]
void inter_color_set_conf(const Config*, GenColor*);
// end::cfg[]

// tag::int[]
void inter_color_apply_hst(const Coords*, const GenColor*, int n, const Particle *pp, /**/ int *cc);
void inter_color_apply_dev(const Coords*, const GenColor*, int n, const Particle *pp, /**/ int *cc);
// end::int[]

/* end src/inter/color/imp.h */

/* begin src/io/txt/imp.h */
struct Particle;
struct Force;
struct TxtRead;

// tag::write[]
void txt_write_pp(long n, const Particle*, const char *path);
void txt_write_pp_ff(long n, const Particle*, const Force*, const char *path);
// end::write[]

// tag::read[]
void txt_read_pp(const char *path, TxtRead **);
void txt_read_pp_ff(const char *path, TxtRead **);
void txt_read_ff(const char *path, TxtRead **);
void txt_read_fin(TxtRead*);

int txt_read_get_n(const TxtRead *);
const Particle* txt_read_get_pp(const TxtRead *);
const Force*    txt_read_get_ff(const TxtRead *);
// end::read[]
/* end src/io/txt/imp.h */

/* begin src/flu/imp.h */
// tag::struct[]
struct FluQuants {
    Particle *pp, *pp0;    /* particles on device  */
    int       n;           /* particle number      */
    Clist     *cells;      /* cell lists           */
    ClistMap *mcells;      /* cell lists map       */
    Particle *pp_hst;      /* particles on host    */

    /* optional data */

    bool ids, colors;

    int *ii, *ii0;  /* global ids on device */
    int *ii_hst;    /* global ids on host   */

    int *cc, *cc0;  /* colors on device */
    int *cc_hst;    /* colors on host   */

    int maxp; /* maximum particle number */
};
// end::struct[]

struct Coords;
struct GenColor;

// tag::mem[]
void flu_ini(bool colors, bool ids, int3 L, int maxp, FluQuants *q);
void flu_fin(FluQuants *q);
// end::mem[]

// tag::gen[]
void flu_gen_quants(const Coords*, int numdensity, const GenColor *gc, FluQuants *q); // <1>
void flu_gen_ids(MPI_Comm, const int n, FluQuants *q); // <2>
// end::gen[]

// tag::start[]
void flu_strt_quants(MPI_Comm, const char *base, const int id, FluQuants *q);     // <1>
void flu_strt_dump(MPI_Comm, const char *base, const int id, const FluQuants *q); // <2>
// end::start[]

// tag::tools[]
void flu_txt_dump(const Coords*, const FluQuants *q); // <1>

/* build cells only from one array of particles fully contained in the domain */
/* warning: this will delete particles which are outside                      */
void flu_build_cells(/**/ FluQuants *q); // <2>
// end::tools[]
/* end src/flu/imp.h */

/* begin src/flu/imp/ini.h */
static void ini_ii(int maxp, int **ii, int **ii0, int **ii_hst) {
    Dalloc(ii, maxp);
    Dalloc(ii0, maxp);
    EMALLOC(maxp, ii_hst);
}

void flu_ini(bool colors, bool ids, int3 L, int maxp, FluQuants *q) {
    q->n = 0;
    q->maxp = maxp;
    Dalloc(&q->pp, maxp);
    Dalloc(&q->pp0, maxp);
    UC(clist_ini(L.x, L.y, L.z, /**/ &q->cells));
    UC(clist_ini_map(maxp, 2, q->cells, /**/ &q->mcells));

    EMALLOC(maxp, &q->pp_hst);

    if (ids)    ini_ii(maxp, &q->ii, &q->ii0, &q->ii_hst);
    if (colors) ini_ii(maxp, &q->cc, &q->cc0, &q->cc_hst);

    q->ids = ids;
    q->colors = colors;
}
/* end src/flu/imp/ini.h */
/* begin src/flu/imp/fin.h */
static void fin_ii(int *ii, int *ii0, int *ii_hst) {
    CC(d::Free(ii));
    CC(d::Free(ii0));
    EFREE(ii_hst);
}

void flu_fin(FluQuants *q) {
    CC(d::Free(q->pp));
    CC(d::Free(q->pp0));
    UC(clist_fin(q->cells));
    UC(clist_fin_map(q->mcells));
    EFREE(q->pp_hst);

    if (q->ids)    UC(fin_ii(q->ii, q->ii0, q->ii_hst));
    if (q->colors) UC(fin_ii(q->cc, q->cc0, q->cc_hst));
}
/* end src/flu/imp/fin.h */
/* begin src/flu/imp/generate.h */
static void assert_np(int n, int m) {
    if (n > m) ERR("too many particles: n = %d < m = %d", n, m);
}

static int gen0(int3 L, int numdensity, int maxp, Particle *pp) { /* generate particle positions and velocities */
    enum {X, Y, Z};
    UC(assert_np(L.x * L.y * L.z * numdensity, maxp));
    os_srand(123456);
    int iz, iy, ix, l, nd = numdensity;
    int n = 0; /* particle index */
    float x, y, z, dr = 0.99;
    for (iz = 0; iz < L.z; iz++)
    for (iy = 0; iy < L.y; iy++)
    for (ix = 0; ix < L.x; ix++) {
        /* edge of a cell */
        int xlo = -0.5*L.x + ix, ylo = -0.5*L.y + iy, zlo = -0.5*L.z + iz;
        for (l = 0; l < nd; l++) {
            Particle p;
            x = xlo + dr * os_drand(), y = ylo + dr * os_drand(), z = zlo + dr * os_drand();
            p.r[X] = x; p.r[Y] = y; p.r[Z] = z;
            p.v[X] = 0; p.v[Y] = 0; p.v[Z] = 0;
            pp[n++] = p;
        }
    }
    msg_print("ic::gen: %d solvent particles", n);
    return n;
}

static int genColor(int3 L, int numdensity, int maxp, const Coords *coords, const GenColor *gc, /*o*/ Particle *pp, int *color, /*w*/ Particle *pp_hst, int *color_hst) {
    int n = gen0(L, numdensity, maxp, pp_hst);
    inter_color_apply_hst(coords, gc, n, pp_hst, /**/ color_hst);
    cH2D(color, color_hst, n);
    cH2D(   pp,    pp_hst, n);
    return n;
}

static int genGrey(int3 L, int numdensity, int maxp, /*o*/ Particle *dev, /*w*/ Particle *hst) {
    int n = gen0(L, numdensity, maxp, hst);
    cH2D(dev, hst, n);
    return n;
}

void flu_gen_quants(const Coords *coords, int numdensity, const GenColor *gc, FluQuants *q) {
    int3 L = subdomain(coords);
    if (q->colors)
        q->n = genColor(L, numdensity, q->maxp, coords, gc, q->pp, q->cc, /*w*/ q->pp_hst, q->cc_hst);
    else
        q->n = genGrey(L, numdensity, q->maxp, q->pp, /*w*/ q->pp_hst);
}

static void ii_gen0(MPI_Comm comm, const long n, int *ii) {
    long i0 = 0;
    MC(m::Exscan(&n, &i0, 1, MPI_LONG, MPI_SUM, comm));
    for (long i = 0; i < n; ++i) ii[i] = i + i0;
}

static void ii_gen(MPI_Comm comm, const int n, int *ii_dev, int *ii_hst) {
    ii_gen0(comm, n, ii_hst);
    cH2D(ii_dev, ii_hst, n);
}

void flu_gen_ids(MPI_Comm comm, const int n, FluQuants *q) {
    ii_gen(comm, n, q->ii, q->ii_hst);
}
/* end src/flu/imp/generate.h */
/* begin src/flu/imp/start.h */
#define CODE "flu"
#define PP  CODE ".pp"
#define COL CODE ".cc"
#define IDS CODE ".ii"

static int strt_pp(MPI_Comm comm, const char *base, const int id, Particle *dev, /*w*/ Particle *hst) {
    int n;
    restart_read_pp(comm, base, PP, id, &n, hst);
    if (n) cH2D(dev, hst, n);
    return n;
}

static int strt_ii(MPI_Comm comm, const char *base, const char *code, const int id, int *dev, /*w*/ int *hst) {
    int n;
    restart_read_ii(comm, base, code, id, &n, hst);
    if (n) cH2D(dev, hst, n);
    return n;
}

void flu_strt_quants(MPI_Comm comm, const char *base, const int id, FluQuants *q) {
    q->n =         strt_pp(comm, base,     id, /**/ q->pp, /* w */ q->pp_hst);
    if (q->ids)    strt_ii(comm, base, IDS, id, /**/ q->ii, /* w */ q->ii_hst);
    if (q->colors) strt_ii(comm, base, COL, id, /**/ q->cc, /* w */ q->cc_hst);
}

static void strt_dump_pp(MPI_Comm comm, const char *base, const int id, const int n, const Particle *dev, Particle *hst) {
    if (n) cD2H(hst, dev, n);
    restart_write_pp(comm, base, PP, id, n, hst);
}

static void strt_dump_ii(MPI_Comm comm, const char *base, const char *code, const int id, const int n, const int *dev, int *hst) {
    if (n) cD2H(hst, dev, n);
    restart_write_ii(comm, base, code, id, n, hst);
}

void flu_strt_dump(MPI_Comm comm, const char *base, const int id, const FluQuants *q) {
    strt_dump_pp(comm, base, id, q->n, q->pp, /* w */ q->pp_hst);
    if (q->ids)    strt_dump_ii(comm, base, IDS, id, q->n, q->ii, /* w */ q->ii_hst);
    if (q->colors) strt_dump_ii(comm, base, COL, id, q->n, q->cc, /* w */ q->cc_hst);
}

#undef CODE
#undef COl
#undef IDS
#undef PP
/* end src/flu/imp/start.h */
/* begin src/flu/imp/txt.h */
static void local2global_p(const Coords *c, long n, Particle *pp) {
    enum {X, Y, Z};
    float *r;
    int i;
    for (i = 0; i < n; i++) {
        r = pp[i].r;
        r[X] = xl2xg(c, r[X]);
        r[Y] = yl2yg(c, r[Y]);
        r[Z] = zl2zg(c, r[Z]);
    }
}

static void gen_name(const Coords *c, /**/ char *name) {
    int r;
    char stamp[FILENAME_MAX];
    coord_stamp(c, /**/ stamp);
    r = sprintf(name, "%s.txt", stamp);
    if (r < 0) ERR("sprintf failed");
}

void flu_txt_dump(const Coords *c, const FluQuants *q) {
    Particle *dev, *hst;
    int n;
    char name[FILENAME_MAX];
    n = q->n; dev = q->pp; hst = q->pp_hst;
    if (n) cD2H(hst, dev, n);
    local2global_p(c, n, /**/ hst);
    UC(gen_name(c, /**/ name));
    UC(txt_write_pp(n, hst, name));
}
/* end src/flu/imp/txt.h */
/* begin src/flu/imp/cells.h */
void flu_build_cells(/**/ FluQuants *q) {
    UC(clist_build(q->n, q->n, q->pp, /**/ q->pp0, q->cells, q->mcells));
    // swap
    Particle *tmp = q->pp;
    q->pp = q->pp0; q->pp0 = tmp;
}
/* end src/flu/imp/cells.h */
/* end src/flu/imp.cpp */

#line 1 "src/fluforces/bulk/imp.cu"
