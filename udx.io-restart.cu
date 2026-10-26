// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/io/restart/imp.cpp */
#include <stdio.h>
#include <string.h>
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
#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
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
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
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

/* begin src/io/restart/imp.h */
struct Coords;
struct Particle;
struct Rigid;

// tag::id[]
#ifndef UDX_SIM_RESTART_COMMON
#define UDX_SIM_RESTART_COMMON
enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};
#endif
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
/* begin src/io/restart/imp/main.h */
/*
   pattern :
   base/code/id.ext
   base depends on read/write
*/

#define PATTERN "%s/%s/%s"

// buff size
enum { BS = 1024 };

// check fprintf (BS-1 for \0 character)
#define CSPR(a) do {                                        \
        int check = a;                                      \
        if (check < 0 || check >= BS-1)                     \
        ERR("Buffer too small to handle this format\n");    \
    } while (0)

static void id2code(const int id, char *code) {
    switch (id) {
    case RESTART_TEMPL:
        CSPR(sprintf(code, "templ"));
        break;
    case RESTART_FINAL:
        CSPR(sprintf(code, "final"));
        break;
    default:
        CSPR(sprintf(code, "%05d", id));
        break;
    }
}

static void gen_base_name(const char *pattern, const char *base, const char *code, const int id, /**/ char *name) {
    char idcode[BS] = {0};
    id2code(id, /**/ idcode);
    CSPR(sprintf(name, pattern, base, code, idcode));
}

static void gen_base_name_dump(const char *base, const char *code, const int id, /**/ char *name) {
    gen_base_name(PATTERN, base, code, id, name);
}

static void gen_base_name_read(const char *base, const char *code, const int id, /**/ char *name) {
    gen_base_name(PATTERN ".bop", base, code, id, name);
}

template <typename T>
static void write(MPI_Comm comm, const char *base, const char *code, int id, long n, const T *tt, BopType type, int nvars, const char *vars) {
    char name[BS];
    BopData *bop;
    T *tt_bop;
    gen_base_name_dump(base, code, id, /**/ name);

    BPC(bop_ini(&bop));
    BPC(bop_set_n(n, bop));
    BPC(bop_set_type(type, bop));
    BPC(bop_set_vars(nvars, vars, bop));
    BPC(bop_alloc(bop));

    tt_bop = (T*) bop_get_data(bop);
    memcpy(tt_bop, tt, n * sizeof(T));

    BPC(bop_write_header(comm, name, bop));
    BPC(bop_write_values(comm, name, bop));

    BPC(bop_fin(bop));
}

template <typename T>
static void read(MPI_Comm comm, const char *base, const char *code, const int id, int *n, T *tt) {
    long np = 0;
    char hname[BS], dname[BS];
    BopData *bop;
    const T *tt_bop;
    gen_base_name_read(base, code, id, /**/ hname);
    msg_print("reading '%s'", hname);

    BPC(bop_ini(&bop));
    BPC(bop_read_header(comm, hname, bop, dname));
    BPC(bop_alloc(bop));
    BPC(bop_read_values(comm, dname, bop));

    BPC(bop_get_n(bop, &np));
    tt_bop = (const T*) bop_get_data(bop);
    memcpy(tt, tt_bop, np * sizeof(T));
    *n = np;

    BPC(bop_fin(bop));
}

void restart_write_pp(MPI_Comm comm, const char *base, const char *code, int id, long n, const Particle *pp) {
    write(comm, base, code, id, n, pp, BopFLOAT, 6, "x y z vx vy vz");
}

void restart_read_pp(MPI_Comm comm, const char *base, const char *code, int id, int *n, Particle *pp) {
    read(comm, base, code, id, n, pp);
}

void restart_write_ii(MPI_Comm comm, const char *base, const char *code, int id, long n, const int *ii) {
    write(comm, base, code, id, n, ii, BopINT, 1, "i");
}

void restart_read_ii(MPI_Comm comm, const char *base, const char *code, int id, int *n, int *ii) {
    read(comm, base, code, id, n, ii);
}

void restart_write_ss(MPI_Comm comm, const char *base, const char *code, int id, long n, const Rigid *ss) {
    static const char *vars =
        "Ixx Ixy Ixz Iyy Iyz Izz "
        "mass x y z vx vy vz omx omy omz "
        "e0x e0y e0z e1x e1y e1z e2x e2y e2z "
        "fx fy fz tx ty tz i";
    enum {NVARS=32};
    write(comm, base, code, id, n, ss, BopFLOAT, NVARS, vars);
}

void restart_read_ss(MPI_Comm comm, const char *base, const char *code, int id, int *n, Rigid *ss) {
    read(comm, base, code, id, n, ss);
}

void restart_write_stream_one_node(MPI_Comm comm, const char *base, const char *code, int id, const void *data, StreamWriter sw) {
    char name[BS];
    FILE *f;
    if (!m::is_master(comm)) return;
    gen_base_name(PATTERN ".dat", base, code, id, name);
    UC(efopen(name, "w", &f));
    UC(sw(data, f));
    UC(efclose(f));
}

void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader sr, void *data) {
    char name[BS];
    FILE *f;
    gen_base_name(PATTERN ".dat", base, code, id, name);
    msg_print("reading '%s'", name);
    UC(efopen(name, "r", &f));
    UC(sr(f, data));
    UC(efclose(f));
}

#undef PATTERN
/* end src/io/restart/imp/main.h */
/* end src/io/restart/imp.cpp */

#line 1 "src/io/rig/imp.cpp"
