// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/rig/gen/imp.cpp */
#include <mpi.h>
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

/* begin src/math/linal/imp.h */
// tag::interface[]
void linal_inv3x3(const float m0[6], /**/ float minv0[6]); // <1>
// end::interface[]
/* end src/math/linal/imp.h */
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
/* begin src/algo/minmax/imp.h */
struct Particle;
struct float3;
// tag::interface[]
void minmax(const Particle *pp, int nv, int nobj, /**/ float3 *lo, float3 *hi);
// end::interface[]
/* end src/algo/minmax/imp.h */
/* begin src/mesh/triangles/type.h */
// tag::interface[]
struct Triangles {
    int4 *tt;
    int nt;
};
// end::interface[]
/* end src/mesh/triangles/type.h */
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
/* begin src/mesh/props/imp.h */
struct int4;
float mesh_volume0(int nt, const int4 *tt, const float *vv);
void mesh_center_of_mass(int nt, const int4 *tt, const float *vv, /**/ float *com);
void mesh_inertia_tensor(int nt, const int4 *tt, const float *vv, const float *com, const float density, /**/ float *I);
/* end src/mesh/props/imp.h */
/* begin src/mesh/dist/imp.h */
struct int4;
float dist_from_mesh(int nt, const int4 *tt, const float *vv, const float *r0);
/* end src/mesh/dist/imp.h */
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

#define _I_ static
#define _S_ static

/* begin src/rig/gen/imp/common.h */
enum {
    IN  = -2,
    OUT = -1
};

_I_ int get_root(MPI_Comm comm, bool hasid0) {
    int rank, root0, root;
    MC(m::Comm_rank(comm, &rank));
    root0 = hasid0 ? rank : 0;
    root = 0;
    MC(m::Allreduce(&root0, &root, 1, MPI_INT, MPI_SUM, comm));
    if (hasid0 && root != rank) ERR("More than one rank has id 0");
    return root;
}

_I_ void exchange_mesh(int maxm, int3 L, MPI_Comm cart, int nv, /*io*/ int *nm, Particle *pp, /**/ int *cc) {
    EMeshPack *pack;
    EMeshComm *comm;
    EMeshUnpack *unpack;
    int nm0, nmhalo;
    nm0 = *nm;

    UC(emesh_pack_ini(L, nv, maxm, &pack));
    UC(emesh_comm_ini(cart, /**/ &comm));
    UC(emesh_unpack_ini(L, nv, maxm, &unpack));

    UC(emesh_build_map(nm0, nv, pp, /**/ pack));
    UC(emesh_pack_pp(nv, pp, /**/ pack));
    UC(emesh_download(pack));

    UC(emesh_post_recv(comm, unpack));
    UC(emesh_post_send(pack, comm));
    UC(emesh_wait_recv(comm, unpack));
    UC(emesh_wait_send(comm));

    UC(emesh_unpack_pp(nv, unpack, /**/ &nmhalo, pp + nm0 * nv));
    if (cc) UC(emesh_get_num_frag_mesh(unpack, /**/ cc));
    *nm += nmhalo;

    UC(emesh_pack_fin(pack));
    UC(emesh_comm_fin(comm));
    UC(emesh_unpack_fin(unpack));
}

_I_ void compute_labels(int pdir, int n, const Particle *pp, int nt, int nv, int nm, const int4 *tt, const Particle *pp_mesh, int, int, /**/ int *ll) {
    float3 *lo, *hi;
    Triangles tri;

    if (nm) Dalloc(&lo, nm);
    if (nm) Dalloc(&hi, nm);

    tri.nt = nt;
    tri.tt = (int4*) tt;

    if (nm) UC(minmax(pp_mesh, nv, nm, /**/ lo, hi));
    UC(collision_label_ini(n, OUT, /**/ ll));
    UC(collision_label(pdir, n, pp, &tri, nv, nm, pp_mesh, lo, hi, IN, /**/ ll));

    if (nm) Dfree(lo);
    if (nm) Dfree(hi);
}
/* end src/rig/gen/imp/common.h */
/* begin src/rig/gen/imp/template.h */
_S_ int3 fid2shift(int3 L, int fid) {
    int3 s;
    using namespace frag_hst;
    s.x = - i2dx(fid) * L.x;
    s.y = - i2dy(fid) * L.y;
    s.z = - i2dz(fid) * L.z;
    return s;
}

_S_ void extract_and_shift_hst(int3 s, int n, const Particle *pp, const int *labels, int *nx, float *rrx) {
    enum {X, Y, Z};
    int i, j;
    const float *r;
    float *rx;
    j = *nx;
    for (i = 0; i < n; ++i) {
        if (labels[i] == IN) {
            r = pp[i].r;
            rx = &rrx[3 * j];
            rx[X] = r[X] + s.x;
            rx[Y] = r[Y] + s.y;
            rx[Z] = r[Z] + s.z;
            ++j;
        }
    }
    *nx = j;
}

_S_ void label_extract_and_shift(int3 shift, int pdir, int n, const Particle *pp_dev, const Particle *pp_hst, int nt, int nv,
                                 int nm, const int4 *tt, const Particle *pp_mesh,
                                 /**/ int *ntempl, float *rrtempl, /*w*/ int *ll_dev, int *ll_hst) {
    UC(compute_labels(pdir, n, pp_dev, nt, nv, nm, tt, pp_mesh, IN, OUT, /**/ ll_dev));
    cD2H(ll_hst, ll_dev, n);
    extract_and_shift_hst(shift, n, pp_hst, ll_hst, ntempl, rrtempl);
}

_S_ void collect_and_broadcast_template(MPI_Comm comm, int *n, float *rr) {
    int i, rank, size, *starts, *counts, ntot;
    float *rr_recv;
    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_size(comm, &size));

    EMALLOC(size, &counts);
    EMALLOC(size, &starts);

    MC(m::Allgather(n, 1, MPI_INT, counts, 1, MPI_INT, comm));

    starts[0] = 0;
    for (i = 0; i < size; ++i) counts[i] *= 3;
    for (i = 1; i < size; ++i) starts[i] = starts[i-1] + counts[i-1];
    ntot = starts[size-1] + counts[size-1];

    EMALLOC(ntot, &rr_recv);

    MC(m::Allgatherv(rr, counts[rank], MPI_FLOAT, rr_recv, counts, starts, MPI_INT, comm));

    *n = ntot/3;
    memcpy(rr, rr_recv, ntot * sizeof(float));

    EFREE(rr_recv);
    EFREE(starts);
    EFREE(counts);
}

_S_ void label_template_dev(int pdir, int3 L, MPI_Comm cart, int nt, int nv, int nm, const int4 *tt, const Particle *pp_mesh,
                            int nflu, const Particle *pp_dev, const Particle *pp_hst, /**/ int *nps, float *rr0, /*w*/ int *ll_dev, int *ll_hst) {
    int i, maxm, nmall, n, cc[NFRAGS];
    int3 shift;
    Particle *pp0, *pp;

    maxm = NFRAGS + 1;
    Dalloc(&pp, nv * maxm);
    pp0 = pp;

    nmall = nm;
    n = nm * nv;
    if (n) cD2D(pp, pp_mesh, n);

    UC(exchange_mesh(maxm, L, cart, nv, /* io */ &nmall, pp, /**/ cc));

    // bulk mesh
    shift = fid2shift(L, frag_bulk);
    if (nm) UC(label_extract_and_shift(shift, pdir, nflu, pp_dev, pp_hst, nt, nv, nm, tt, pp, /**/ nps, rr0, /*w*/ ll_dev, ll_hst));
    pp += nm * nv;

    // halo meshes
    for (i = 0; i < NFRAGS; ++i) {
        nm = cc[i];
        shift = fid2shift(L, i);
        if (nm) UC(label_extract_and_shift(shift, pdir, nflu, pp_dev, pp_hst, nt, nv, nm, tt, pp, /**/ nps, rr0, /*w*/ ll_dev, ll_hst));
        pp += nm * nv;
    }

    UC(collect_and_broadcast_template(cart, nps, rr0));

    Dfree(pp0);
}

struct Transf {
    float s[3];
    float e0[3], e1[3], e2[3];
};

_S_ void get_transf(MPI_Comm comm, bool hasid0, const Rigid *ss, Transf *T) {
    int root, sz;
    if (hasid0) {
        sz = 3 * sizeof(float);
        memcpy(T->s, ss[0].com, sz);
        memcpy(T->e0, ss[0].e0, sz);
        memcpy(T->e1, ss[0].e1, sz);
        memcpy(T->e2, ss[0].e2, sz);
    }
    root = get_root(comm, hasid0);
    MC(m::Bcast(T->s, 3, MPI_FLOAT, root, comm));
    MC(m::Bcast(T->e0, 3, MPI_FLOAT, root, comm));
    MC(m::Bcast(T->e1, 3, MPI_FLOAT, root, comm));
    MC(m::Bcast(T->e2, 3, MPI_FLOAT, root, comm));
}

_S_ float dot(const float a[], const float b[]) {
    enum {X, Y, Z};
    return a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z];
}

_S_ void transform(const Transf *T, float *r) {
    enum {X, Y, Z};
    int c;
    float r0[3];
    for (c = 0; c < 3; ++c) r0[c] = r[c] - T->s[c];
    r[X] = dot(r0, T->e0);
    r[Y] = dot(r0, T->e1);
    r[Z] = dot(r0, T->e2);
}

_S_ void transf_template(const Transf *T, int n, float *rr) {
    int i;
    for (i = 0; i < n; ++i) transform(T, &rr[3*i]);
}

_I_ void extract_template(int3 L, MPI_Comm cart, RigGenInfo rgi, int n, const Particle *flu_pp_dev, const Particle *flu_pp_hst,
                             int, bool hasid0, const Rigid *ss, /**/ int *nps, float *rr0, /*w*/ int *ll_dev, int *ll_hst) {
    int nm, pdir;
    Transf T;
    *nps = 0;
    nm = hasid0 ? 1 : 0;
    pdir = rig_pininfo_get_pdir(rgi.pi);

    UC(label_template_dev(pdir, L, cart, rgi.nt, rgi.nv, nm, rgi.tt, rgi.pp, n, flu_pp_dev, flu_pp_hst, /**/ nps, rr0, /*w*/ ll_dev, ll_hst));
    UC(get_transf(cart, hasid0, ss, &T));
    UC(transf_template(&T, *nps, rr0));
}

_I_ void empty_solid(const MeshRead *mesh, /* io */ int *n, float *rr) {
    enum {X, Y, Z};
    int n0, i, j, nt;
    const float *r, *vv;
    const int4 *tt;
    float d;
    n0 = *n;
    nt = mesh_read_get_nt(mesh);
    tt = mesh_read_get_tri(mesh);
    vv = mesh_read_get_vert(mesh);

    for (i = j = 0; i < n0; ++i) {
        r = rr + 3*i;
        d = dist_from_mesh(nt, tt, vv, r);
        if (d <= 1.0) {
            rr[3*j + X] = r[X];
            rr[3*j + Y] = r[Y];
            rr[3*j + Z] = r[Z];
            ++j;
        }
    }
    if (j == 0)
        ERR("No particle remaining in solid template (size = %d)", n0);

    msg_print("Template solid: keep %d out of %d particles", j, n0);
    *n = j;
}
/* end src/rig/gen/imp/template.h */
/* begin src/rig/gen/imp/kill.h */
_S_ void remove_hst(const int *labels, int *n, Particle *pp) {
    int i, j;
    for (i = j = 0; i < *n; ++i) {
        if (labels[i] == OUT)
            pp[j++] = pp[i];
    }
    *n = j;
}

_I_ void kill_solvent(RigGenInfo rgi, int maxm, int3 L, MPI_Comm cart, int nm,
                         /*io*/ int *n, Particle *pp_hst, Particle *pp_dev,
                         /*w*/ int *ll_dev, int *ll_hst) {
    int nmall, pdir;
    Particle *i_ppall;
    nmall = nm;
    pdir = rig_pininfo_get_pdir(rgi.pi);

    Dalloc(&i_ppall, maxm * rgi.nv);
    cD2D(i_ppall, rgi.pp, nm * rgi.nv);

    UC(exchange_mesh(maxm, L, cart, rgi.nv, /*io*/ &nmall, i_ppall, NULL));
    UC(compute_labels(pdir, *n, pp_dev, rgi.nt, rgi.nv, nmall, rgi.tt, i_ppall, IN, OUT, /**/ ll_dev));

    cD2H(ll_hst, ll_dev, *n);
    remove_hst(ll_hst, n, pp_hst);
    cH2D(pp_dev, pp_hst, *n);

    Dfree(i_ppall);
}
/* end src/rig/gen/imp/kill.h */
/* begin src/rig/gen/imp/props.h */
_S_ void init_I_from_pos(int n, const float *rr, float pmass, /**/ float *I) {
    enum {X, Y, Z};
    enum {XX, XY, XZ, YY, YZ, ZZ};
    enum {YX = XY, ZX = XZ, ZY = YZ};
    int c, i;
    const float *r;
    float x, y, z;

    for (c = 0; c < 6; ++c) I[c] = 0;
    for (i = 0; i < n; ++i) {
        r = &rr[3*i];
        x = r[X]; y = r[Y]; z = r[Z];
        I[XX] += y*y + z*z;
        I[YY] += z*z + x*x;
        I[ZZ] += x*x + y*y;
        I[XY] -= x*y;
        I[XZ] -= z*x;
        I[YZ] -= y*z;
    }
    for (c = 0; c < 6; ++c) I[c] *= pmass;
}

_S_ void init_I_from_mesh(float density, int nt, const int4 *tt, const float *vv, /**/ float *I) {
    float com[3] = {0};
    mesh_center_of_mass(nt, tt, vv, /**/ com);
    mesh_inertia_tensor(nt, tt, vv, com, density, /**/ I);
}

_S_ void compute_properties(const RigPinInfo *pi, int n, const float *rr0, float pmass,
                               float numdensity, const MeshRead *mesh, /**/ Rigid *s) {
    enum {X, Y, Z};
    int spdir, nt;
    const int4 *tt;
    const float *vv;
    float I[6], rho;

    spdir = rig_pininfo_get_pdir(pi);

    /* ini inertia tensor */

    if (spdir == NOT_PERIODIC) {
        rho = pmass * numdensity;
        nt = mesh_read_get_nt(mesh);
        tt = mesh_read_get_tri(mesh);
        vv = mesh_read_get_vert(mesh);
        init_I_from_mesh(rho, nt, tt, vv, /**/ I);
        s->mass = rho * mesh_volume0(nt, tt, vv);
    }
    else {
        init_I_from_pos(n, rr0, pmass, /**/ I);
        s->mass = n * pmass;
    }

    UC(linal_inv3x3(I, /**/ s->Iinv));
}

_S_ void copy_props(const Rigid *s0, Rigid *s) {
    s->mass = s0->mass;
    memcpy(s->Iinv, s0->Iinv, 6*sizeof(float));
}

_S_ void clear_vel(Rigid *s) {
    enum {X, Y, Z};
    s->v[X] = s->v[Y] = s->v[Z] = 0;
    s->om[X] = s->om[Y] = s->om[Z] = 0;
}

_S_ void clear_forces(Rigid *s) {
    enum {X, Y, Z};
    s->fo[X] = s->fo[Y] = s->fo[Z] = 0;
    s->to[X] = s->to[Y] = s->to[Z] = 0;
}

_I_ void set_properties(MPI_Comm, RigGenInfo rgi, int n, const float *rr0, int ns, const int *ids, /**/ Rigid *ss) {
    Rigid s_props, *s;
    int i;

    compute_properties(rgi.pi, n, rr0, rgi.mass, rgi.numdensity, rgi.mesh, &s_props);

    for (i = 0; i < ns; ++i) {
        s = &ss[i];
        copy_props(&s_props, s);
        clear_vel(s);
        clear_forces(s);
        s->id = ids[i];
    }
}
/* end src/rig/gen/imp/props.h */
/* begin src/rig/gen/imp/main.h */
static void gen_ids(MPI_Comm comm, int n, int *ii) {
    int i, i0 = 0, count = 1;
    MC(m::Exscan(&n, &i0, count, MPI_INT, MPI_SUM, comm));
    for (i = 0; i < n; ++i) ii[i] = i + i0;
}

void rig_gen_from_solvent(const Coords *coords, MPI_Comm cart, RigGenInfo rgi, /*io*/ FluInfo fi, /**/ RigInfo ri) {
    int3 L;
    int *ids, nflu, *ll_hst, *ll_dev;
    Particle *pp_flu_hst;
    bool hasid0;

    L = subdomain(coords);
    nflu = *fi.n;

    EMALLOC(ri.ns, &ids);
    EMALLOC(nflu, &pp_flu_hst);
    EMALLOC(nflu, &ll_hst);
    Dalloc(&ll_dev, nflu);
    cD2H(pp_flu_hst, fi.pp, nflu);

    UC(gen_ids(cart, ri.ns, ids));

    hasid0 = (ri.ns && ids[0] == 0);

    UC(extract_template(L, cart, rgi, nflu, fi.pp, pp_flu_hst, ri.ns, hasid0, ri.ss,
                        /**/ ri.nps, ri.rr0, /*w*/ ll_dev, ll_hst));

    UC(kill_solvent(rgi, MAX_SOLIDS, L, cart, ri.ns,
                    /*io*/ fi.n, pp_flu_hst, fi.pp,
                    /*w*/ ll_dev, ll_hst));

    UC(set_properties(cart, rgi, *ri.nps, ri.rr0, ri.ns, ids, /**/ ri.ss));

    if (rgi.empty_pp)
        UC(empty_solid(rgi.mesh, /*io*/ ri.nps, ri.rr0));
    Dfree(ll_dev);
    EFREE(pp_flu_hst);
    EFREE(ll_hst);
    EFREE(ids);
}
/* end src/rig/gen/imp/main.h */

#undef _I_
#undef _S_
/* end src/rig/gen/imp.cpp */

#line 1 "src/rig/imp.cpp"
