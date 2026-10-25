// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/wall/sdf/imp.cu */
#include <mpi.h>

#include <curand.h>

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


/* begin src/d/q.h */
/* function and variable type [q]ualifiers */

#if   defined(DEV_CUDA)
/* begin src/d/cuda/q.h */
/* B.1. Function Type Qualifiers */
/* B.2. Variable Type Qualifiers */
/* end src/d/cuda/q.h */
#elif defined(DEV_CPU)
/* begin src/d/cpu/q.h */
/* B.1. Function Type Qualifiers */
#ifdef __device__
  #undef  __device__
  #define __device__
#endif

#ifdef __global__
  #undef  __global__
  #define __global__
#endif

#ifdef __host__
  #undef  __host__
  #define __host__
#endif

#ifdef __noinline__
  #undef  __noinline__
  #define __noinline__
#endif

#ifdef __forceinline__
  #undef  __forceinline__
  #define __forceinline__
#endif

/* B.2. Variable Type Qualifiers */
/* __device__ : also a function qualifier */
#ifdef __constant__
  #undef  __constant__
  #define __constant__
#endif

#ifdef __shared__
  #undef  __shared__
  #define __shared__
#endif

#ifdef __managed__
  #undef  __managed__
  #define __managed__
#endif

#ifdef __restrict__
  #undef  __restrict__
  #define __restrict__
#endif
/* end src/d/cpu/q.h */
#else
  #error DEV_* is undefined
#endif
/* end src/d/q.h */
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
/* begin src/d/ker.h */
#if   defined(DEV_CUDA)
/* begin src/d/cuda/ker.h */
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
/* end src/d/cuda/ker.h */
#elif defined(DEV_CPU)
/* begin src/d/cpu/ker.h */
#define atomicAdd(...) (0)

#define atomicExch(...)
#define __float2half_rn(...) (0)
#define __fmaf_rz(...) (0.0)
#define __ldg(p) (*(p))
#define __popc(...)
#define __shfl(...) (0.0)
#define __shfl_down(...) (0.0)
#define __shfl_up(...) (0)
#define __shfl_xor(...) (0)

#define __syncthreads(...)

#define tex3D(...)

#define  Ifetch(t, i)  (0);
#define F4fetch(t, i)  (make_float4(0, 0, 0, 0))
#define F2fetch(t, i)  (make_float2(0, 0))
#define Ffetch(t, i)   (0.0)
#define Tfetch(T, t, i) (T{})

#define Ttex3D(T, to, i, j, k) (T{})
/* end src/d/cpu/ker.h */
/* begin src/d/cpu/wrap.h */
namespace d {
static uint3 threadIdx, blockIdx, gridDim;
static dim3  blockDim;
static int   warpSize = 1;
}

#define threadIdx d::threadIdx
#define blockDim  d::blockDim
#define gridDim   d::gridDim
#define blockIdx  d::blockIdx
#define warpSize  d::warpSize
/* end src/d/cpu/wrap.h */
#else
  #error DEV_* is undefined
#endif
/* end src/d/ker.h */
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
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
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

/* begin src/wall/sdf/array3d/imp.h */
struct Array3d;
void array3d_ini(Array3d**, size_t x, size_t y, size_t z);                // <1>
void array3d_fin(Array3d*);                                               // <2>
void array3d_copy(size_t x, size_t y, size_t z, float *D, /**/ Array3d*); // <3>
/* end src/wall/sdf/array3d/imp.h */
/* begin src/wall/sdf/tex3d/type.h */
struct Tex3d_v { cudaTextureObject_t t; }; // <1>
struct Tex3d   { cudaTextureObject_t t; }; // <2>
/* end src/wall/sdf/tex3d/type.h */
/* begin src/wall/sdf/tex3d/imp.h */
struct Tex3d;
struct Tex3d_v;
struct Array3d;

// tag::mem[]
void tex3d_ini(Tex3d**);
void tex3d_fin(Tex3d*);
// end::mem[]

// tag::int[]
void tex3d_copy(const Array3d*, /**/ Tex3d*);    // <1>
void tex3d_to_view(const Tex3d*, /**/ Tex3d_v*); // <2>
// end::int[]
/* end src/wall/sdf/tex3d/imp.h */

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
/* begin src/wall/sdf/bounce/imp.h */
struct Sdf;
struct WvelStep;
struct Coords;
struct Particle;

// tag::int[]
void bounce_back(float dt, const WvelStep *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp);
// end::int[]

/* end src/wall/sdf/bounce/imp.h */
/* begin src/wall/sdf/label/imp.h */
enum {
    // tag::enum[]
    LABEL_BULK,  /* remains in bulk */
    LABEL_WALL,  /* becomes wall particle */
    LABEL_DEEP   /* deep inside the wall */
    // end::enum[]
};

// tag::int[]
void wall_label_dev(const Sdf *sdf, int n, const Particle*, /**/ int *labels); // <1>
void wall_label_hst(const Sdf *sdf, int n, const Particle*, /**/ int *labels); // <2>
// end::int[]
/* end src/wall/sdf/label/imp.h */

/* begin src/math/tform/type.h */
struct Tform_v { float o[3], s[3]; };
/* end src/math/tform/type.h */
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
/* begin src/math/tform/dev.h */
static __device__ void tform_convert_dev(const Tform_v *t, const float a0[3], /**/ float a1[3]) {
    enum {X, Y, Z};
    const float *o, *s;
    o = t->o; s = t->s;
    a1[X] = s[X]*a0[X] + o[X];
    a1[Y] = s[Y]*a0[Y] + o[Y];
    a1[Z] = s[Z]*a0[Z] + o[Z];
}

static __device__ void tform_spacing_dev(const Tform_v *t, /**/ float s[3]) {
    enum {X, Y, Z};
    s[X] = t->s[X];
    s[Y] = t->s[Y];
    s[Z] = t->s[Z];
}
/* end src/math/tform/dev.h */
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
/* begin src/algo/utils/shfl.h */
#define _I_ static __device__

#if __CUDACC_VER_MAJOR__ >= 9
template <typename T>
_I_ T shfl_down(T val, unsigned int width) {
    return __shfl_down_sync(0xFFFFFFFF, val, width);
}

template <typename T>
_I_ T shfl_up(T val, unsigned int width) {
    return __shfl_up_sync(0xFFFFFFFF, val, width);
}
template <typename T>
_I_ T shfl(T val, unsigned int width) {
    return __shfl_sync(0xFFFFFFFF, val, width);
}
#else
template <typename T>
_I_ T shfl_down(T val, unsigned int width) {
    return __shfl_down(val, width);
}

template <typename T>
_I_ T shfl_up(T val, unsigned int width) {
    return __shfl_up(val, width);
}
template <typename T>
_I_ T shfl(T val, unsigned int width) {
    return __shfl(val, width);
}
#endif

#undef _I_
/* end src/algo/utils/shfl.h */
/* begin src/algo/utils/dev.h */
#define _S_ static __device__
#define _I_ static __device__

// tag::int[]
template <typename T>
_I_ T warpReduceSum(T v)  // <1>
// end::int[]
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += shfl_down(v, offset);
    return v;
}

// tag::int[]
_I_ float2 warpReduceSum(float2 val)  // <2>
// end::int[]
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
    }
    return val;
}

// tag::int[]
_I_ float3 warpReduceSum(float3 val)  // <3>
// end::int[]
{
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
        val.z += shfl_down(val.z, offset);
    }
    return val;
}

#undef _S_
#undef _I_

/* end src/algo/utils/dev.h */

/* begin src/wall/sdf/type.h */
/* view */
struct Sdf_v {
    Tex3d_v tex;
    Tform_v t;
    float cheap_threshold;
};
/* end src/wall/sdf/type.h */
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

/* begin src/wall/sdf/dev.h */
#define _S_ static __device__
#define _I_ static __device__

_S_ float fetch(const Sdf_v *sdf, float i, float j, float k) {
    return Ttex3D(float, sdf->tex.t, i, j, k);
}

_S_ int iround(float x) { return (x > 0.5) ? (x + 0.5) : (x - 0.5); }

_S_ void convert(const Sdf_v *sdf, const float a[3], /**/ float b[3]) {
    tform_convert_dev(&sdf->t, a, /**/ b);
}

_S_ void convert_floor(const Sdf_v *sdf, const float a[3], /**/ int i[3]) {
    enum {X, Y, Z};
    float f[3];
    convert(sdf, a, /**/ f);
    i[X] = int(f[X]); i[Y] = int(f[Y]); i[Z] = int(f[Z]);
}

_S_ void convert_round(const Sdf_v *sdf, const float a[3], /**/ int i[3]) {
    enum {X, Y, Z};
    float f[3];
    convert(sdf, a, /**/ f);
    i[X] = iround(f[X]); i[Y] = iround(f[Y]); i[Z] = iround(f[Z]);
}

_S_ void spacing(const Sdf_v *sdf, float s[3]) {
    tform_spacing_dev(&sdf->t, /**/ s);
}

_S_ float3 grad(const Sdf_v *sdf, const float3 *pos) {
    int tc[3];
    float fcts[3], r[3] = {pos->x, pos->y, pos->z};
    float myval, gx, gy, gz;
    convert_floor(sdf, r, /**/ tc);
    spacing(sdf, /**/ fcts);

#define tex0(ix, iy, iz) (fetch(sdf, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    myval = tex0(0, 0, 0);
    gx = fcts[0] * (tex0(1, 0, 0) - myval);
    gy = fcts[1] * (tex0(0, 1, 0) - myval);
    gz = fcts[2] * (tex0(0, 0, 1) - myval);
#undef tex0
    return make_float3(gx, gy, gz);
}

// tag::int[]
_I_ float3 sdf_ugrad(const Sdf_v *texsdf, const float3 *r) // <1>
// end::int[]
{
    float mag, eps;
    float3 g;
    eps = 1e-6;
    g = grad(texsdf, r);
    mag = sqrt(g.x*g.x + g.y*g.y + g.z*g.z);
    if (mag > eps) {
        g.x /= mag;
        g.y /= mag;
        g.z /= mag;
    }
    return g;
}

_S_ float cheap_sdf(const Sdf_v *sdf, float x, float y, float z)  {
    int tc[3];
    float r[3] = {x, y, z};
    convert_round(sdf, r, /**/ tc);
    return fetch(sdf, tc[0], tc[1], tc[2]);
}

// tag::int[]
_I_ bool sdf_far(const Sdf_v *sdf, float x, float y, float z) // <2>
// end::int[]
{
    return cheap_sdf(sdf, x, y, z) <= sdf->cheap_threshold - 1 ;
}

// tag::int[]
_I_ float sdf_eval(const Sdf_v *sdf0, float x, float y, float z)  // <3>
// end::int[]
{
    int c;
    float s000, s001, s010, s100, s101, s011, s110, s111;
    float s00x, s01x, s10x, s11x;
    float s0yx, s1yx;
    float szyx;
    float tc0[3], tc[3], lmbd[3], r[3] = {x, y, z};
    convert(sdf0, r, /**/ tc0);
    for (c = 0; c < 3; ++c) {
        lmbd[c] = tc0[c] - (int)tc0[c];
        tc[c] = (int)tc0[c] + 0.5;
    }
#define tex0(ix, iy, iz) (fetch(sdf0, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    s000 = tex0(0, 0, 0), s001 = tex0(1, 0, 0), s010 = tex0(0, 1, 0);
    s011 = tex0(1, 1, 0), s100 = tex0(0, 0, 1), s101 = tex0(1, 0, 1);
    s110 = tex0(0, 1, 1), s111 = tex0(1, 1, 1);
#undef tex0

#define wavrg(A, B, p) A*(1-p) + p*B /* weighted average */
    s00x = wavrg(s000, s001, lmbd[0]);
    s01x = wavrg(s010, s011, lmbd[0]);
    s10x = wavrg(s100, s101, lmbd[0]);
    s11x = wavrg(s110, s111, lmbd[0]);

    s0yx = wavrg(s00x, s01x, lmbd[1]);
    s1yx = wavrg(s10x, s11x, lmbd[1]);

    szyx = wavrg(s0yx, s1yx, lmbd[2]);
#undef wavrg
    return szyx;
}

#undef _S_
#undef _I_
/* end src/wall/sdf/dev.h */

namespace sdf_dev {
/* begin src/wall/sdf/dev/main.h */
static __device__ float3 get_r(int3 L, const float *U) {
    enum {X, Y, Z};
    float3 r;
    r.x = L.x * (U[X] - 0.5);
    r.y = L.y * (U[Y] - 0.5);
    r.z = L.z * (U[Z] - 0.5);
    return r;
}

__global__ void count_inside(Sdf_v sv, int3 L, int n, const float *UU, int *counts) {
    int i, c, chunkid, warpmaster;
    float3 r;
    float s;

    c = 0;
    warpmaster = (0 == threadIdx.x % warpSize);
    i   = threadIdx.x + blockIdx.x * blockDim.x;
    chunkid = blockDim.x;

    if (i < n) {
        r = get_r(L, UU + 3 * i);
        s = sdf_eval(&sv, r.x, r.y, r.z);
        c = (s <= 0.f);
    }

    c = warpReduceSum(c);

    if (warpmaster)
        atomicAdd(&counts[chunkid], c);
}
/* end src/wall/sdf/dev/main.h */
}

/* begin src/wall/sdf/imp/type.h */
struct Array3d;
struct Tform;
struct Sdf {
    Array3d   *arr;
    Tex3d     *tex;
    Tform     *t;
    float cheap_threshold;

    int3 Lte; /* texture grid size */
};
/* end src/wall/sdf/imp/type.h */
/* begin src/wall/sdf/imp/gen.h */
static void gen0(const Coords *coords, Field *F, /**/ Sdf *sdf) {
    enum {X, Y, Z};
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */
    int N[3];
    float *D;
    field_size(F, /**/  N);
    field_data(F, /**/ &D);
    UC(array3d_copy(N[X], N[Y], N[Z], D, /**/ sdf->arr));
    UC(tex3d_ini(&sdf->tex));
    UC(tex3d_copy(sdf->arr, /**/ sdf->tex));
    UC(sub2tex_ini(coords, N, M, /**/ sdf->t));
}

static void gen1(const Coords *coords, const int T[3], const Field *F0, /**/ Sdf *sdf) {
    Tform *t;
    int N[3];
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */
    Field *F1;

    UC(tform_ini(&t));
    field_size(F0, /**/ N);
    UC(tex2sdf_ini(coords, T, N, M, /**/ t));
    UC(field_sample(F0, t, T, /**/ &F1));
    UC(gen0(coords, F1, sdf));

    UC(field_fin(F1));
    UC(tform_fin(t));
}

static void gen2(const Coords *coords, const Field *F, /**/ Sdf *sdf) {
    int3 Lte = sdf->Lte;
    int T[] = {Lte.x, Lte.y, Lte.z};
    UC(gen1(coords, T, F, /**/ sdf));
}

static void gen3(const Coords *coords, bool dump, MPI_Comm cart, Field *F, /**/ Sdf *sdf) {
    enum {X, Y, Z};
    float sc, G, ext[3];
    G = xdomain(coords);
    UC(field_extend(F, /**/ ext));
    sc = G / ext[X];
    UC(field_scale(F, sc));
    if (dump)
        UC(field_dump(F, coords, cart));
    UC(gen2(coords, F, /**/ sdf));
}

void sdf_gen(const Coords *coords, MPI_Comm cart, bool dump, Sdf *sdf) {
    enum {X, Y, Z};
    const char *f = "sdf.dat";
    Field *F;
    UC(field_ini(f, &F));
    UC(gen3(coords, dump, cart, F, /**/ sdf));
    UC(field_fin(F));
}
/* end src/wall/sdf/imp/gen.h */
/* begin src/wall/sdf/imp/split.h */
/* sort solvent particle into remaining in solvent and turning into wall according to keys (all on hst) */
static void split_wall_solvent(const int *keys, /*io*/ int *s_n, Particle *s_pp, /**/ int *w_n, Particle *w_pp) {
    int n = *s_n;
    Particle p;
    int k, ia = 0, is = 0, iw = 0; /* all, solvent, wall */

    for (ia = 0; ia < n; ++ia) {
        k = keys[ia];
        p = s_pp[ia];

        if      (k == LABEL_BULK) s_pp[is++] = p;
        else if (k == LABEL_WALL) w_pp[iw++] = p;
    }
    *s_n = is;
    *w_n = iw;
}

/* sort solvent particle (dev) into remaining in solvent (dev) and turning into wall (hst)*/
void sdf_bulk_wall(const Sdf *sdf, /*io*/ int* s_n, Particle *s_pp, /*o*/ int *w_n, Particle *w_pp) {
    int n = *s_n, *labels;
    Particle *s_pp_hst;
    EMALLOC(n, &s_pp_hst);
    EMALLOC(n, &labels);

    UC(wall_label_hst(sdf, n, s_pp, labels));
    cD2H(s_pp_hst, s_pp, n);

    UC(split_wall_solvent(labels, /*io*/ s_n, s_pp_hst, /**/ w_n, w_pp));
    cH2D(s_pp, s_pp_hst, *s_n);

    EFREE(s_pp_hst);
    EFREE(labels);
}

/* bulk predicate : is in bulk? */
static bool bulkp(int *keys, int i) { return keys[i] == LABEL_BULK; }

static int who_stays0(int *keys, int nc, int nv, /*o*/ int *stay) {
    int c, v;  /* cell and vertex */
    int s = 0; /* how many stays? */
    for (c = 0; c < nc; ++c) {
        v = 0;
        while (v  < nv && bulkp(keys, v + nv * c)) v++;
        if    (v == nv) stay[s++] = c;
    }
    return s;
}

int sdf_who_stays(const Sdf *sdf, int n, const Particle *pp, int nc, int nv, /**/ int *stay) {
    int *labels;
    EMALLOC(n, &labels);
    UC(wall_label_hst(sdf, n, pp, /**/ labels));
    nc = who_stays0(labels, nc, nv, /**/ stay);
    EFREE(labels);
    return nc;
}
/* end src/wall/sdf/imp/split.h */
/* begin src/wall/sdf/imp/main.h */
static int3 texture_grid_size(int3 L, int3 M) {
    enum {PAD = 16};
    int3 T;
    int x, y, z; /* grid extents in real space */
    int ty, tz;

    x = L.x + 2 * M.x;
    y = L.y + 2 * M.y;
    z = L.z + 2 * M.z;

    T.x = PAD * PAD;
    ty = ceiln(y * T.x, x);
    tz = ceiln(z * T.x, x);
    T.y = PAD * ceiln(ty, PAD);
    T.z = PAD * ceiln(tz, PAD);

    return T;
};

void sdf_ini(int3 L, Sdf **pq) {
    Sdf *q;
    int3 Lte, M;
    M = make_int3(XWM, YWM, ZWM);
    Lte = texture_grid_size(L, M);
    EMALLOC(1, &q);
    UC(array3d_ini(&q->arr, Lte.x, Lte.y, Lte.z));
    UC(  tform_ini(&q->t));

    q->Lte = Lte;
    q->cheap_threshold = - sqrt(3.f) * ((float) (L.x + 2*M.x) / (float)Lte.x);

    *pq = q;
}

void sdf_fin(Sdf *q) {
    UC(array3d_fin(q->arr));
    UC(  tex3d_fin(q->tex));
    UC(  tform_fin(q->t));
    EFREE(q);
}

void sdf_bounce(float dt, const WvelStep *wv, const Coords *c, const Sdf *sdf, int n, /**/ Particle *pp) {
    UC(bounce_back(dt, wv, c, sdf, n, /**/ pp));
}


void sdf_get_view(const Sdf *q, /**/ Sdf_v *v) {
    tex3d_to_view(q->tex, &v->tex);
    tform_to_view(q->t  , &v->t);
    v->cheap_threshold = q->cheap_threshold;
}

enum {NTHREADS = 128};

static void gen_rnd(long n, float *UU) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, UU, n);
    curandDestroyGenerator(gen);
}

static void chunk_counts(int3 L, const Sdf *sdf, int nsamples, int nchunks, int *counts_hst) {
    int *counts_dev;
    Sdf_v view;
    float *UU;
    sdf_get_view(sdf, &view);

    Dalloc(&UU, 3 * nsamples);
    Dalloc(&counts_dev, nchunks);
    DzeroA(counts_dev, nchunks);

    gen_rnd(3 * nsamples, UU);

    KL(sdf_dev::count_inside,
       (nchunks, NTHREADS),
       (view, L, nsamples, UU, /**/ counts_dev));

    cD2H(counts_hst, counts_dev, nchunks);
    Dfree(counts_dev);
    Dfree(UU);
}

static long reduce(int n, int *d) {
    long c, i;
    for (i = c = 0; i < n; ++i) c += d[i];
    return c;
}

static long subdomain_counts(int3 L, const Sdf *sdf, long nsamples) {
    int *counts_hst;
    long nin;
    int nchunks;

    nchunks = ceiln(nsamples, NTHREADS);
    EMALLOC(nchunks, &counts_hst);

    chunk_counts(L, sdf, nsamples, nchunks, counts_hst);
    nin = reduce(nchunks, counts_hst);

    EFREE(counts_hst);
    return nin;
}

static double subdomain_volume(int3 L, long nin, long nsamples) {
    double V = (double) nin / (double) nsamples;
    V *= L.x * L.y * L.z;
    return V;
}

double sdf_compute_volume(MPI_Comm comm, int3 L, const Sdf *sdf, long nsamples) {
    double loc, tot = 0;
    long nin;

    nin = subdomain_counts(L, sdf, nsamples);
    loc = subdomain_volume(L, nin, nsamples);

    MC(m::Allreduce(&loc, &tot, 1, MPI_DOUBLE, MPI_SUM, comm));

    return tot;
}
/* end src/wall/sdf/imp/main.h */
/* end src/wall/sdf/imp.cu */

#line 1 "src/wall/sdf/label/imp.cu"
