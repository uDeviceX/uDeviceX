// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/io/mesh/imp.cpp */
#include <mpi.h>
#include <stdio.h>
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
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
/* begin src/utils/mc.h */
/* [m]pi [c]heck macros */
#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
/* end src/utils/mc.h */

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
/* begin src/algo/vectors/imp.h */
struct Particle;
struct Vectors;
struct Coords;

// tag::interface[]
void vectors_float_ini(int n, const float*, /**/ Vectors**); // <1>
void vectors_postions_ini(int n, const Particle*, /**/ Vectors**); // <2>
void vectors_postions_edge_ini  (const Coords*, int n, const Particle*, /**/ Vectors**); // <3>
void vectors_postions_center_ini(const Coords*, int n, const Particle*, /**/ Vectors**); // <4>
void vectors_velocities_ini(int n, const Particle*, /**/ Vectors**); // <5>
void vectors_zero_ini(int n, /**/ Vectors**); // <6>

void vectors_fin(Vectors*);
void vectors_get(const Vectors*, int i, /**/ float r[3]); // <7>
// end::interface[]
/* end src/algo/vectors/imp.h */

/* begin src/io/write/imp.h */
struct WriteFile;

// tag::interface[]
int write_file_open(MPI_Comm, const char *path, /**/ WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*); // <1>
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*); // <2>

int write_shift_indices(MPI_Comm, int, /**/ int*);
int write_reduce(MPI_Comm, int, /**/ int*);
// end::interface[]
/* end src/io/write/imp.h */

/* begin src/io/mesh/imp.h */
struct Coords;
struct Particle;
struct int4;
struct MeshWrite;
struct MeshRead;
struct Vectors;

#ifndef UDX_SIM_MESH_WRITE_COMMON
#define UDX_SIM_MESH_WRITE_COMMON
enum {MESH_WRITE_EDGE, MESH_WRITE_CENTER};
#endif

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite**);
void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead*, const char *directory, /**/ MeshWrite**);

void mesh_write_particles(MeshWrite*, MPI_Comm, const Coords*, int nm, const Particle*, int id);
void mesh_write_vectors(MeshWrite*, MPI_Comm, int nm, Vectors *pos, Vectors *vel, int id);

void mesh_write_fin(MeshWrite*);
/* end src/io/mesh/imp.h */
/* begin src/io/mesh/imp/type.h */
struct MeshWrite {
    int4 *tt;
    int nv, nt;
    char directory[FILENAME_MAX];
    int shift_type;
};
/* end src/io/mesh/imp/type.h */

/* begin src/io/mesh/imp/util.h */
#ifndef UDX_IO_OUTPUT_COMMON_UTIL
#define UDX_IO_OUTPUT_COMMON_UTIL
static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
static char *cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }
#endif

static void mkdir0(const char *p, const char *s) {
    char path[FILENAME_MAX] = {'\0'};
    cpy(path, p);
    cat(path, "/");
    cat(path, s);
    msg_print("mkdir -p '%s'", path);
    UC(os_mkdir(path));
}

#ifndef UDX_IO_OUTPUT_MKDIR_COMMON
#define UDX_IO_OUTPUT_MKDIR_COMMON
static void mkdir(MPI_Comm comm, const char *p, const char *s) {
    if (m::is_master(comm)) UC(mkdir0(p, s));
}
#endif
/* end src/io/mesh/imp/util.h */
/* begin src/io/mesh/imp/main.h */
#define PATTERN "%s/%s/%05d.ply"
enum { NVP = 3 /* number of vertices per face */ };

static void ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    int i;
    MeshWrite *q;
    EMALLOC(1, &q);

    q->nv = nv; q->nt = nt;
    cpy(q->directory, directory);
    UC(mkdir(comm, DUMP_BASE, directory));
    EMALLOC(nt, &q->tt);
    for (i = 0; i < nt; i++)
        q->tt[i] = tt[i];
    q->shift_type = shifttype;
    *pq = q;
}

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    UC(ini(comm, shifttype, tt, nv, nt, directory, /**/ pq));
}

void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead *cell, const char *directory, /**/ MeshWrite **pq) {
    int nv, nt;
    const int4 *tt;
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    tt = mesh_read_get_tri(cell);
    UC(ini(comm, shifttype, tt, nv, nt, directory, /**/ pq));
}

void mesh_write_fin(MeshWrite *q) {
    EFREE(q->tt);
    EFREE(q);
}

static void header(MPI_Comm comm, int nc0, int nv, int nt, WriteFile *f) {
    int nc; /* total number of cells */
    int sz = 0;
    char s[BUFSIZ];
    UC(write_reduce(comm, nc0, &nc));
    if (m::is_master(comm))
        sz = snprintf(s, BUFSIZ - 1,
                     "ply\n"
                     "format binary_little_endian 1.0\n"
                     "element vertex %d\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property float u\nproperty float v\nproperty float w\n"
                     "element face  %d\n"
                     "property list int int vertex_index\n"
                     "end_header\n", nv*nc, nt*nc);
    write_master(comm, s, sz, f);
}

static void vert(MPI_Comm cart, int n, const Vectors *pos, const Vectors *vel, WriteFile *f) {
    float *D, *r, *v;
    int i, j, m;
    m = (3 + 3) * n; /* pos[3] + vel[3] */
    EMALLOC(m, &D);
    for (i = j = 0; i < n; i++) {
        r = &D[j]; j += 3;
        v = &D[j]; j += 3;
        UC(vectors_get(pos, i, /**/ r));
        UC(vectors_get(vel, i, /**/ v));
    }
    UC(write_all(cart, D, m*sizeof(D[0]), f));
    EFREE(D);
}

static void wfaces0(MPI_Comm comm, int *buf, const int4 *faces, int nc, int nv, int nt, WriteFile *f) {
    /* write faces */
    int c, t, b;  /* cell, triangle, buffer index */
    int n, shift;
    int4 tri;
    n = nc * nv;
    UC(write_shift_indices(comm, n, &shift));
    for (b = c = 0; c < nc; ++c)
        for (t = 0; t < nt; ++t) {
            tri = faces[t];
            buf[b++] = NVP;
            buf[b++] = shift + nv*c + tri.x;
            buf[b++] = shift + nv*c + tri.y;
            buf[b++] = shift + nv*c + tri.z;
        }
    UC(write_all(comm, buf, b*sizeof(buf[0]), f));
}

static void wfaces(MPI_Comm cart, const int4 *faces, int nc, int nv, int nt, WriteFile *f) {
    int *buf; /* buffer for faces */
    int sz;
    sz = (1 + NVP) * nc * nt;
    EMALLOC(sz, &buf);
    UC(wfaces0(cart, buf, faces, nc, nv, nt, f));
    EFREE(buf);
}

static void mesh_write(MPI_Comm comm, int nc, int nv, int nt,
                        const Vectors *pos, const Vectors *vel, const int4 *faces,
                        WriteFile *f) {
    int n;
    n = nc * nv;
    UC(header(comm, nc, nv, nt, f));
    UC(vert(comm, n, pos, vel, f));
    UC(wfaces(comm, faces, nc, nv, nt, f));
}

void mesh_write_particles(MeshWrite *q, MPI_Comm comm, const Coords *coords, int nc, const Particle *pp, int id) {
    int nv, nt, n;
    const int4 *tt;
    Vectors *pos, *vel;
    WriteFile *f;
    const char *directory;
    char path[FILENAME_MAX] = {'\0'} ;

    nv = q->nv; nt = q->nt; tt = q->tt; directory = q->directory;
    n = nv * nc;
    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, directory, id) < 0)
        ERR("sprintf failed");
    UC(write_file_open(comm, path, /**/ &f));
    switch (q->shift_type) {
    case MESH_WRITE_EDGE:
        vectors_postions_edge_ini(coords, n, pp, /**/ &pos);
        break;
    case MESH_WRITE_CENTER:
        vectors_postions_center_ini(coords, n, pp, /**/ &pos);
        break;
    default:
        ERR("unkown q->type: %d", q->shift_type);
    }
    vectors_velocities_ini(n, pp, /**/ &vel);
    UC(mesh_write(comm, nc, nv, nt, pos, vel, tt, f));
    vectors_fin(pos);
    vectors_fin(vel);
    UC(write_file_close(f));
}

void mesh_write_vectors(MeshWrite *q, MPI_Comm comm, int nc, Vectors *pos, Vectors *vel, int id) {
    int nv, nt;
    const int4 *tt;
    WriteFile *f;
    const char *directory;
    char path[FILENAME_MAX] = {'\0'};

    nv = q->nv; nt = q->nt; tt = q->tt; directory = q->directory;
    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, directory, id) < 0)
        ERR("snprintf failed");
    UC(write_file_open(comm, path, /**/ &f));
    UC(mesh_write(comm, nc, nv, nt, pos, vel, tt, f));
    UC(write_file_close(f));
}
/* end src/io/mesh/imp/main.h */
/* end src/io/mesh/imp.cpp */

#line 1 "src/io/mesh_read/edg/imp.cpp"
