// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/control/vel/conf.cpp */
#include <stdio.h>
#include <mpi.h>
#include <vector_types.h>

#ifndef UDX_CONTROL_UTILS_COMMON
#define UDX_CONTROL_UTILS_COMMON
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
#endif /* UDX_CONTROL_UTILS_COMMON */
#ifndef UDX_CONTROL_ERROR_COMMON
#define UDX_CONTROL_ERROR_COMMON
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
#endif /* UDX_CONTROL_ERROR_COMMON */
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

/* begin src/control/vel/imp.h */
struct PidVCont;
struct Coords;
struct Config;
struct float3;
struct int3;
struct Particle;

// tag::mem[]
void vcont_ini(MPI_Comm comm, int3 L, /**/ PidVCont **c);
void vcont_fin(/**/ PidVCont *cont);
// end::mem[]

// tag::set[]
void vcont_set_params(float factor, float Kp, float Ki, float Kd, /**/ PidVCont *c); // <1>
void vcont_set_target(float3 vtarget, /**/ PidVCont *c);  // <2>
void vcont_set_cart(/**/ PidVCont *c);  // <3>
void vcont_set_radial(/**/ PidVCont *c);  // <4>
// end::set[]

// tag::cnf[]
void vcont_set_conf(const Config *cfg, /**/ PidVCont *c);
// end::cnf[]

// tag::start[]
void vcont_strt_dump(MPI_Comm, const char *base, int id, const PidVCont*); // <1>
void vcont_strt_read(          const char *base, int id, PidVCont*);       // <2>
// end::start[]

// tag::int[]
void   vcont_sample(const Coords *coords, int n, const Particle *pp, const int *starts, const int *counts, /**/ PidVCont *c); // <1>
float3 vcont_adjustF(/**/ PidVCont *c); // <2>
void   vcont_log(const PidVCont *c);    // <3>
// end::int[]
/* end src/control/vel/imp.h */

void vcont_set_conf(const Config *cfg, /**/ PidVCont *c) {
    const char *type;
    float3 U;
    float factor, Kp, Ki, Kd;

    UC(conf_lookup_string(cfg, "vcon.type", &type));
    UC(conf_lookup_float3(cfg, "vcon.U", &U));
    UC(conf_lookup_float(cfg, "vcon.factor", &factor));
    UC(conf_lookup_float(cfg, "vcon.Kp", &Kp));
    UC(conf_lookup_float(cfg, "vcon.Ki", &Ki));
    UC(conf_lookup_float(cfg, "vcon.Kd", &Kd));

    UC(vcont_set_params(factor, Kp, Ki, Kd, /**/ c));
    UC(vcont_set_target(U, /**/ c));

    if      (same_str(type, "cart"))
        UC(vcont_set_cart(/**/ c));
    else if (same_str(type, "rad"))
        UC(vcont_set_radial(/**/ c));
    else
        ERR("Unrecognised type <%s>", type);

}
/* end src/control/vel/conf.cpp */

#line 1 "src/control/vel/imp.cu"


/* begin src/control/vel/imp.cu */
#include <stdio.h>
#include <mpi.h>
#include <assert.h>

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

#ifndef UDX_CONTROL_ERROR_COMMON
#define UDX_CONTROL_ERROR_COMMON
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
#endif /* UDX_CONTROL_ERROR_COMMON */


#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
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
#endif /* UDX_CONTROL_TYPE_COMMON */
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
#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
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
#endif /* UDX_CONTROL_D_API_COMMON */
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
#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
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
#endif /* UDX_CONTROL_KL_COMMON */
/* begin src/utils/mc.h */
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
/* end src/utils/mc.h */

#ifndef UDX_CONTROL_MATH_COMMON
#define UDX_CONTROL_MATH_COMMON
/* begin src/math/dev.h */
/* use templates here because we might have mixed float/double */

#define _HD_ __host__ __device__

template <typename T1, typename T2>
_HD_ void scal(const T1 a, /*io*/ T2 *b) {
    b->x *= a;
    b->y *= a;
    b->z *= a;
}

template <typename T1, typename T2>
_HD_ void add(const T1 *a, /*io*/ T2 *b) {
    b->x += a->x;
    b->y += a->y;
    b->z += a->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void diff(const T1 *a, const T2 *b, /**/ T3 *c) {
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ T1 dot(const T2 *a, const T3 *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void cross(const T1 *a, const T2 *b, /**/ T3 *c) {
    c->x = a->y * b->z - a->z * b->y;
    c->y = a->z * b->x - a->x * b->z;
    c->z = a->x * b->y - a->y * b->x;
}

template <typename T1, typename T2, typename T3>
_HD_ void scalmult(const T1 *a, const T2 x, /**/ const T3 *b) {
    b->x = a->x * x;
    b->y = a->y * x;
    b->z = a->z * x;
}


template <typename T1, typename T2, typename T3, typename T4>
_HD_ void apxb(const T1 *a, const T2 x, const T3 *b, /**/ T4 *c) {
    c->x = a->x + x * b->x;
    c->y = a->y + x * b->y;
    c->z = a->z + x * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void axpy(const T1 a, const T2 *x, /*io*/ T3 *y) {
    y->x += a * x->x;
    y->y += a * x->y;
    y->z += a * x->z;
}
/* end src/math/dev.h */
#endif /* UDX_CONTROL_MATH_COMMON */
#ifndef UDX_CONTROL_COORDS_COMMON
#define UDX_CONTROL_COORDS_COMMON
/* begin src/coords/type.h */
// tag::view[]
struct Coords_v {
    int xc, yc, zc; /* rank coordinates        */
    int xd, yd, zd; /* rank sizes              */
    int Lx, Ly, Lz; /* [L]ocal: subdomain size */
};
// end::view[]
/* end src/coords/type.h */
#endif /* UDX_CONTROL_COORDS_COMMON */
/* begin src/coords/dev.h */
/* domain sizes */

static __device__ int xdomain(const Coords_v c) {
    return c.Lx * c.xd;
}

static __device__ int ydomain(const Coords_v c) {
    return c.Ly * c.yd;
}

static __device__ int zdomain(const Coords_v c) {
    return c.Lz * c.zd;
}

/* [l]ocal to [c]enter */

static __device__ float xl2xc(const Coords_v c, float xl) {
    return xl - 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
}

static __device__ float yl2yc(const Coords_v c, float yl) {
    return yl - 0.5f * c.Ly * (c.yd - 2 * c.yc - 1);
}

static __device__ float zl2zc(const Coords_v c, float zl) {
    return zl - 0.5f * c.Lz * (c.zd - 2 * c.zc - 1);
}

static __device__ void local2center(const Coords_v c, float3 rl, /**/ float3 *rc) {
    rc->x = xl2xc(c, rl.x);
    rc->y = yl2yc(c, rl.y);
    rc->z = zl2zc(c, rl.z);
}

/* [c]enter to [l]ocal  */

static __device__ float xc2xl(const Coords_v c, float xc) {
    return xc + 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
}

static __device__ float yc2yl(const Coords_v c, float yc) {
    return yc + 0.5f * c.Ly * (c.yd - 2.f * c.yc - 1);
}

static __device__ float zc2zl(const Coords_v c, float zc) {
    return zc + 0.5f * c.Lz * (c.zd - 2.f * c.zc - 1);
}

static __device__ void center2local(Coords_v c, float3 rc, /**/ float3 *rl) {
    rl->x = xc2xl(c, rc.x);
    rl->y = yc2yl(c, rc.y);
    rl->z = zc2zl(c, rc.z);
}

/* [l]ocal to [g]lobal */

static __device__ float xl2xg(const Coords_v c, float xl) {
    return (c.xc + 0.5f) * c.Lx + xl;
}

static __device__ float yl2yg(const Coords_v c, float yl) {
    return (c.yc + 0.5f) * c.Ly + yl;
}

static __device__ float zl2zg(const Coords_v c, float zl) {
    return (c.zc + 0.5f) * c.Lz + zl;
}

static __device__ void local2global(const Coords_v c, float3 rl, /**/ float3 *rg) {
    rg->x = xl2xg(c, rl.x);
    rg->y = yl2yg(c, rl.y);
    rg->z = zl2zg(c, rl.z);
}

/* [g]lobal to [l]ocal */

static __device__ float xg2xl(const Coords_v c, float xg) {
    return xg - (c.xc + 0.5f) * c.Lx;
}

static __device__ float yg2yl(const Coords_v c, float yg) {
    return yg - (c.yc + 0.5f) * c.Ly;
}

static __device__ float zg2zl(const Coords_v c, float zg) {
    return zg - (c.zc + 0.5f) * c.Lz;
}

static __device__ void global2local(const Coords_v c, float3 rg, /**/ float3 *rl) {
    rl->x = xg2xl(c, rg.x);
    rl->y = yg2yl(c, rg.y);
    rl->z = zg2zl(c, rg.z);
}

/* end src/coords/dev.h */
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

/* begin src/algo/utils/shfl.h */
#ifndef UDX_CONTROL_SHFL_COMMON
#define UDX_CONTROL_SHFL_COMMON
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
#endif /* UDX_CONTROL_SHFL_COMMON */

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



/* begin src/control/vel/imp/type.h */
enum {
    VCONT_TYPE_NONE,
    VCONT_TYPE_CART,
    VCONT_TYPE_RAD
};

// empty for now
struct TCart {};
struct TRad {};

/* parameters for transformation */
union Trans {
    TCart cart;
    TRad  rad;
};


struct Sampler {
    int3 L;                   /* subdomain size                                  */
    long nsamples;            /* number of "pending" avg on grid                 */

    float3 *gridvel;          /* sum of velocity per grid point                  */
    int    *gridnum;          /* sum of number of particles per grid point       */
    float3 *totvel;           /* chunk sums (pinned memory)                      */
    int    *totnum;
    float3 *dtotvel;          /* device pointer of the above                     */
    int    *dtotnum;
};

struct State {
    float3 cur;               /* current average velocity                        */
    float3 olde, sume;        /* previous error, sum of all previous errors      */
    float3 f;                 /* force estimate                                  */
};

struct Param {
    float3 target;            /* target average velocity                         */
    float Kp, Ki, Kd;         /* parameters of the pid controller                */
};

/* pid velocity controller */
struct PidVCont {
    Param param;
    State state;
    Sampler sampler;          /* helper to average velocity on device            */

    MPI_Comm comm;

    FILE *fdump;              /* output file for logging info                    */

    Trans trans;              /* transformation of the velocity before averaging */
    int type;                 /* type of the transformation                      */
};
/* end src/control/vel/imp/type.h */

namespace vcont_dev {
/* begin src/control/vel/dev/common.h */
__global__ void reduceByWarp(const float3 *gvel, const int *gnum, const uint ncells, /**/ float3 *vel, int *num) {
    assert(blockDim.x == 32);
    int i, c;
    float3 v = make_float3(0, 0, 0);
    int n = 0;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    c = blockIdx.x; /* [c]hunk id */

    if (i < ncells) {
        v = gvel[i];
        n = gnum[i];
    }

    v = warpReduceSum(v);
    n = warpReduceSum(n);

    if ((threadIdx.x % warpSize) == 0) {
        vel[c] = v;
        num[c] = n;
    }
}
/* end src/control/vel/dev/common.h */
/* begin src/control/vel/dev/cart.h */
static __device__ float3 transform(Coords_v, TCart, const Particle p) {
    enum {X, Y, Z};
    float3 u;
    u.x = p.v[X];
    u.y = p.v[Y];
    u.z = p.v[Z];
    return u;
}
/* end src/control/vel/dev/cart.h */
/* begin src/control/vel/dev/radial.h */
static __device__ float3 transform(Coords_v c, TRad, const Particle p) {
    enum {X, Y, Z};
    float3 u; // radial coordinates
    float x, y, r, rinv;
    float cost, sint;
    x = xl2xc(c, p.r[X]);
    y = yl2yc(c, p.r[Y]);

    r = sqrt(x*x + y*y);
    rinv = 1 / r;
    cost = rinv * x;
    sint = rinv * y;

    u.x = r * (  cost * p.v[X] + sint * p.v[Y]);
    u.y = r * (- sint * p.v[X] + cost * p.v[Y]);
    u.z = p.v[Z];
    return u;
}
/* end src/control/vel/dev/radial.h */
/* begin src/control/vel/dev/sample.h */
static __device__ bool valid_cell(int3 L, int3 c) {
    return c.x < L.x && c.y < L.y && c.z < L.z;
}

static __device__ int get_cid(int3 L, int3 c) {
    return c.x + L.x * (c.y + L.y * c.z);
}

template<typename Trans>
__global__ void sample(Coords_v coords, Trans t, int3 L, const int *cellsstart, const int *cellscount, const Particle *pp,
                       /**/ float3 *gridv, int *gridn) {
    Particle p;
    float3 u;
    int pid, cid, num;
    const int3 c = make_int3(threadIdx.x + blockIdx.x * blockDim.x,
                             threadIdx.y + blockIdx.y * blockDim.y,
                             threadIdx.z + blockIdx.z * blockDim.z);

    if (valid_cell(L, c)) {
        cid = get_cid(L, c);
        num = cellscount[cid];

        for (pid = cellsstart[cid]; pid < cellsstart[cid] + num; pid++) {
            p = pp[pid];
            u = transform(coords, t, p);

            gridv[cid].x += u.x;
            gridv[cid].y += u.y;
            gridv[cid].z += u.z;
        }
        gridn[cid] += num;
    }
}
/* end src/control/vel/dev/sample.h */
}

/* begin src/control/vel/imp/main.h */
enum {WARPSIZE = 32};

static void ini_dump(int rank, /**/ FILE **f) {
    *f = NULL;
    if (rank) return;
    UC(efopen(DUMP_BASE "/vcont.txt", "w", /**/ f));
    fprintf(*f, "#vx vy vz fx fy fz\n");
    msg_print("Velocity controller: dump to " DUMP_BASE "/vcont.txt");
}

static void fin_dump(FILE *f) {
    if (f) UC(efclose(f));
}

static void reini_sampler(/**/ Sampler *s) {
    int3 L = s->L;
    int ncells = L.x * L.y * L.z;

    if (ncells) {
        CC(d::MemsetAsync(s->gridvel, 0, ncells * sizeof(float3)));
        CC(d::MemsetAsync(s->gridnum, 0, ncells * sizeof(int)));
    }

    s->nsamples = 0;
}

static void ini_sampler(int3 L, Sampler *s) {
    int ncells, nchunks;
    s->L = L;
    ncells = L.x * L.y * L.z;

    Dalloc(&s->gridvel, ncells);
    Dalloc(&s->gridnum, ncells);

    nchunks = ceiln(ncells, WARPSIZE);

    CC(d::alloc_pinned((void **) &s->totvel, nchunks * sizeof(float3)));
    CC(d::alloc_pinned((void **) &s->totnum, nchunks * sizeof(int)));
    CC(d::HostGetDevicePointer((void **) &s->dtotvel, s->totvel, 0));
    CC(d::HostGetDevicePointer((void **) &s->dtotnum, s->totnum, 0));
}

static void ini_state(State *s) {
    s->f = s->sume = s->cur = make_float3(0, 0, 0);
}

static void ini(MPI_Comm comm, int3 L, /**/ PidVCont *c) {
    int rank;

    UC(ini_sampler(L, &c->sampler));
    UC(reini_sampler(&c->sampler));
    UC(ini_state(&c->state));

    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_dup(comm, &c->comm));

    UC(ini_dump(rank, /**/ &c->fdump));

    c->type = VCONT_TYPE_NONE;
}

void vcont_ini(MPI_Comm comm, int3 L, /**/ PidVCont **c) {
    PidVCont *vc;
    EMALLOC(1, c);
    vc = *c;
    UC(ini(comm, L, /**/ vc));
}

static void fin_sampler(Sampler *s) {
    Dfree(s->gridvel);
    Dfree(s->gridnum);
    CC(d::FreeHost(s->totvel));
    CC(d::FreeHost(s->totnum));
}

void vcont_fin(/**/ PidVCont *c) {
    UC(fin_sampler(&c->sampler));
    MC(m::Comm_free(&c->comm));
    UC(fin_dump(c->fdump));
    EFREE(c);
}

static void set_params(float f, float Kp, float Ki, float Kd, Param *p) {
    p->Kp = f * Kp;
    p->Ki = f * Ki;
    p->Kd = f * Kd;
}

void vcont_set_params(float factor, float Kp, float Ki, float Kd, /**/ PidVCont *c) {
    set_params(factor, Kp, Ki, Kd, &c->param);
}

void vcont_set_target(float3 vtarget, /**/ PidVCont *c) {
    c->param.target = vtarget;
    c->state.olde   = vtarget;
}

void vcont_set_cart(/**/ PidVCont *cont) {
    cont->type = VCONT_TYPE_CART;
}

void vcont_set_radial(/**/ PidVCont *cont) {
    cont->type = VCONT_TYPE_RAD;
}

template <typename Trans>
static void sample(Trans t, const Coords *coords, const Particle *pp, const int *starts, const int *counts, /**/ Sampler *s) {
    int3 L = s->L;
    Coords_v coordsv;

    coords_get_view(coords, &coordsv);

    dim3 block(8, 8, 1);
    dim3 grid(ceiln(L.x, block.x),
              ceiln(L.y, block.y),
              ceiln(L.z, block.z));

    KL(vcont_dev::sample, (grid, block), (coordsv, t, L, starts, counts, pp, /**/ s->gridvel, s->gridnum));
    s->nsamples++;
}

void vcont_sample(const Coords *coords, int, const Particle *pp, const int *starts, const int *counts, /**/ PidVCont *c) {
    switch (c->type) {
    case VCONT_TYPE_NONE:
        break;
    case VCONT_TYPE_CART:
        sample(c->trans.cart, coords, pp, starts, counts, /**/ &c->sampler);
        break;
    case VCONT_TYPE_RAD:
        sample(c->trans.cart, coords, pp, starts, counts, /**/ &c->sampler);
        break;
    default:
        ERR("Unknown type");
        break;
    };
}

static void reduce_local(Sampler *s, double3 *v, long *n) {
    int3 L = s->L;
    int ncells, nchunks, i;
    ncells = L.x * L.y * L.z;
    nchunks = ceiln(ncells, WARPSIZE);

    KL(vcont_dev::reduceByWarp, (nchunks, WARPSIZE), (s->gridvel, s->gridnum, ncells, /**/ s->dtotvel, s->dtotnum));
    dSync();

    *v = make_double3(0, 0, 0);
    *n = 0;

    for (i = 0; i < nchunks; ++i) {
        add(s->totvel + i, /**/ v);
        *n += s->totnum[i];
    }
}

static float3 average_samples(MPI_Comm comm, Sampler *s) {
    long ncur;
    double3 vcur;
    double fac;

    UC(reduce_local(s, &vcur, &ncur));

    MC(m::Allreduce(MPI_IN_PLACE, &vcur.x, 3, MPI_DOUBLE, MPI_SUM, comm));
    MC(m::Allreduce(MPI_IN_PLACE, &ncur,   1, MPI_LONG,   MPI_SUM, comm));

    fac = ncur ? (1.0 / ncur) : 1.0;
    scal(fac, /**/ &vcur);

    return make_float3(vcur.x, vcur.y, vcur.z);
}

static float3 update_state(const Param *p, float3 vcur, State *s) {
    float3 e, de, f;

    s->cur = vcur;
    diff(&p->target, &vcur, /**/ &e);
    diff(&e, &s->olde, /**/ &de);
    add(&e, /**/ &s->sume);
    s->olde = e;

    f = make_float3(0, 0, 0);

    axpy(p->Kp, &e,       /**/ &f);
    axpy(p->Ki, &s->sume, /**/ &f);
    axpy(p->Kd, &de,      /**/ &f);
    s->f = f;
    return f;
}

float3 vcont_adjustF(/**/ PidVCont *c) {
    float3 vcur;
    vcur = average_samples(c->comm, &c->sampler);
    UC(reini_sampler(&c->sampler));

    return update_state(&c->param, vcur, &c->state);
}

void vcont_log(const PidVCont *c) {
    if (c->fdump == NULL) return;
    float3 v = c->state.cur;
    float3 f = c->state.f;
    fprintf(c->fdump, "%.3e %.3e %.3e %.3e %.3e %.3e\n",
            v.x, v.y, v.z, f.x, f.y, f.z);
    fflush(c->fdump);
}
/* end src/control/vel/imp/main.h */
/* begin src/control/vel/imp/start.h */
#define CODE "vcon"
#define PAT "%g %g %g\n"

static void writef3(float3 v, FILE *f) {
    fprintf(f, PAT, v.x, v.y, v.z);
}

static void write_state(const State *s, FILE *f) {
    writef3(s->cur, f);
    writef3(s->olde, f);
    writef3(s->sume, f);
    writef3(s->f, f);
}

static void stream_write_state(const void *data, FILE *f) {
    const PidVCont *c = (const PidVCont*) data;
    write_state(&c->state, f);
}

void vcont_strt_dump(MPI_Comm comm, const char *base, int id, const PidVCont *c) {
    StreamWriter sw = stream_write_state;
    restart_write_stream_one_node(comm, base, CODE, id, (void*) c, sw);
}

static void readf3(FILE *f, float3 *v) {
    if (3 != fscanf(f, PAT, &v->x, &v->y, &v->z))
        ERR("wrong format\n");
}

static void read_state(FILE *f, State *s) {
    UC(readf3(f, &s->cur));
    UC(readf3(f, &s->olde));
    UC(readf3(f, &s->sume));
    UC(readf3(f, &s->f));
}

static void stream_read_state(FILE *f, void *data) {
    PidVCont *c = (PidVCont*) data;
    UC(read_state(f, &c->state));
}

void vcont_strt_read(const char *base, int id, PidVCont *c) {
    StreamReader sr = stream_read_state;
    restart_read_stream_one_node(base, CODE, id, sr, (void*) c);
}

#undef CODE
#undef PAT
/* end src/control/vel/imp/start.h */
/* end src/control/vel/imp.cu */

#line 1 "src/coords/conf.cpp"
