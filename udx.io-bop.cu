// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/io/bop/imp.cpp */
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

/* begin src/utils/mc.h */
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
/* end src/utils/mc.h */

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

/* begin src/inc/type.h */
#ifndef UDX_IO_OUTPUT_TYPE_COMMON
#define UDX_IO_OUTPUT_TYPE_COMMON
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
/* begin src/io/bop/imp/type.h */
struct IoBop {
    BopData *ff; /* float data */
    BopData *ii; /*   int data */
};
/* end src/io/bop/imp/type.h */
/* begin src/io/bop/imp/main.h */
enum {
    NVARP = 6,
    NVARF = 3,
    NVARS = 6,
    NVARI = 1,
    NVARC = 1
};

#define VARP "x y z vx vy vz"
#define VARF "fx fy fz"
#define VARS "sxx sxy sxz syy syz szz"
#define VARI "ids"
#define VARC "color"

#define PATTERN "%s-%05d"

void io_bop_ini(MPI_Comm comm, int maxp, IoBop **ib) {
    IoBop *t;
    EMALLOC(1, ib);
    t = *ib;

    BPC( bop_ini(&t->ff) );
    BPC( bop_ini(&t->ii) );

    BPC( bop_set_n(maxp, t->ff) );
    BPC( bop_set_n(maxp, t->ii) );

    BPC( bop_set_vars(NVARP + NVARF, VARP " " VARF, t->ff) );
    BPC( bop_set_vars(NVARI, VARI, t->ii) );

    BPC( bop_set_type(BopFLOAT, t->ff) );
    BPC( bop_set_type(  BopINT, t->ii) );

    BPC( bop_alloc(t->ff) );
    BPC( bop_alloc(t->ii) );

    if (m::is_master(comm))
        UC(os_mkdir(DUMP_BASE "/bop"));
}

void io_bop_fin(IoBop *t) {
    BPC( bop_fin(t->ff) );
    BPC( bop_fin(t->ii) );
    EFREE(t);
}

static void p2f3(const Particle p, /**/ float3 *r, float3 *v) {
    enum {X, Y, Z};
    r->x = p.r[X];
    r->y = p.r[Y];
    r->z = p.r[Z];

    v->x = p.v[X];
    v->y = p.v[Y];
    v->z = p.v[Z];
}

static void f2f3(const Force fo, /**/ float3 *f) {
    enum {X, Y, Z};
    f->x = fo.f[X];
    f->y = fo.f[Y];
    f->z = fo.f[Z];
}

static void copy_shift(const Coords *c, long n, const Particle *pp, /**/ float3 *w) {
    float3 r, v;
    for (int j = 0; j < n; ++j) {
        p2f3(pp[j], /**/ &r, &v);
        local2global(c, r, /**/ &w[2*j]);
        w[2*j + 1] = v;
    }
}

static void copy_shift_with_forces(const Coords *c, long n, const Particle *pp, const Force *ff, /**/ float3 *w) {
    float3 r, v, f;
    for (int j = 0; j < n; ++j) {
        p2f3(pp[j], /**/ &r, &v);
        f2f3(ff[j], /**/ &f);
        local2global(c, r, /**/ &w[3*j]);
        w[3*j + 1] = v;
        w[3*j + 2] = f;
    }
}

static void set_name(const char *base, int id, /**/ char *name) {
    sprintf(name, DUMP_BASE "/bop/" PATTERN, base, id);
}

static void write(MPI_Comm cart, const char *base, int id, BopData *bop) {
    char fname[FILENAME_MAX];
    set_name(base, id, fname);

    BPC( bop_write_header(cart, fname, bop) );
    BPC( bop_write_values(cart, fname, bop) );
}

void io_bop_parts (MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const char *name, int id, IoBop *t) {
    BopData *bop  = t->ff;
    float3 *ppout = (float3*) bop_get_data(bop);

    copy_shift(coords, n, pp, /**/ ppout);
    BPC( bop_set_n(n, bop) );
    BPC( bop_set_vars(NVARP, VARP, bop) );

    write(cart, name, id, bop);
}

void io_bop_parts_forces(MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const Force *ff, const char *name, int id, IoBop *t) {
    BopData *bop = t->ff;
    float3 *out  = (float3*) bop_get_data(bop);

    copy_shift_with_forces(coords, n, pp, ff, /**/ out);
    BPC( bop_set_n(n, bop) );
    BPC( bop_set_vars(NVARP + NVARF, VARP " " VARF, bop) );

    write(cart, name, id, bop);
}

void io_bop_stresses(MPI_Comm cart, long n, const float *ss, const char *name, int id, IoBop *t) {
    BopData *bop = t->ff;
    float   *out = (float*) bop_get_data(bop);

    memcpy(out, ss, n * NVARS * sizeof(float));
    BPC( bop_set_n(n, bop) );
    BPC( bop_set_vars(NVARS, VARS, bop) );

    write(cart, name, id, bop);
}

void io_bop_ids(MPI_Comm cart, long n, const int *ii, const char *name, int id, IoBop *t) {
    BopData *bop = t->ii;
    int     *out = (int*) bop_get_data(bop);

    memcpy(out, ii, n * NVARI * sizeof(int));
    BPC( bop_set_n(n, bop) );
    BPC( bop_set_vars(NVARI, VARI, bop) );

    write(cart, name, id, bop);
}

void io_bop_colors(MPI_Comm cart, long n, const int *cc, const char *name, int id, IoBop *t) {
    BopData *bop = t->ii;
    int     *out = (int*) bop_get_data(bop);

    memcpy(out, cc, n * NVARC * sizeof(int));
    BPC( bop_set_n(n, bop) );
    BPC( bop_set_vars(NVARC, VARC, bop) );

    write(cart, name, id, bop);
}

#undef VARP
#undef VARF
#undef VARS
#undef VARI
#undef VARC

#undef PATTERN
/* end src/io/bop/imp/main.h */
/* end src/io/bop/imp.cpp */

#line 1 "src/io/com/imp.cpp"
