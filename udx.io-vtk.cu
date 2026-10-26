// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/io/vtk/imp.cpp */
#include <stdio.h>
#include <stdarg.h>
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

/* begin src/algo/key_list/imp.h */
struct KeyList;

void key_list_ini(KeyList**);
void key_list_copy(KeyList*, /**/ KeyList**);
void key_list_fin(KeyList*);

void key_list_push(KeyList*, const char*);
int key_list_has(KeyList*, const char*);
int key_list_offset(KeyList*, const char*);
int key_list_width(KeyList*, const char*);
int key_list_size(KeyList*);
const char* key_list_name(KeyList*, int i);

void key_list_mark(KeyList*, const char*);
int  key_list_marked(KeyList*);
void key_list_unmark(KeyList*);

void key_list_log(KeyList*);
/* end src/algo/key_list/imp.h */
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
/* begin src/algo/scalars/imp.h */
struct Particle;
struct Scalars;
struct Vectors;

// tag::interface[]
void scalars_float_ini(int n, const float*, /**/ Scalars**);
void scalars_double_ini(int n, const double*, /**/ Scalars**);
void scalars_vectors_ini(int n, const Vectors*, int dim, /**/ Scalars**);

void scalars_zero_ini(int n, /**/ Scalars**);
void scalars_one_ini(int n, /**/ Scalars**);

void scalars_fin(Scalars*);
double scalars_get(const Scalars*, int i);
// end::interface[]
/* end src/algo/scalars/imp.h */

/* begin src/io/vtk/mesh/imp.h */
struct Mesh;
struct MeshRead;

void mesh_ini(MeshRead*, /**/ Mesh**);
void mesh_copy(const Mesh*, /**/ Mesh**);
void mesh_fin(Mesh*);

int mesh_nv(Mesh*);
int mesh_nt(Mesh*);
int mesh_ne(Mesh*);
const int* mesh_tt(Mesh*);
/* end src/io/vtk/mesh/imp.h */
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

/* begin src/io/vtk/imp.h */
struct VTK;
struct VTKConf;

struct MeshRead;
struct Vectors;
struct Scalars;

// tag::conf[]
void vtk_conf_ini(MeshRead*, /**/ VTKConf**); // <1>
void vtk_conf_fin(VTKConf*);
void vtk_conf_vert(VTKConf*, const char *keys); // <2>
void vtk_conf_tri(VTKConf*, const char *keys); // <3>
// end::conf[]

// tag::main[]
void vtk_ini(MPI_Comm, int maxm, char const *dir, VTKConf*, /**/ VTK**); // <1>
void vtk_points(VTK*, int nm, const Vectors*); // <2>

void vtk_vert(VTK*, int nm, const Scalars*, const char *keys); // <3>
void vtk_tri(VTK*, int nm, const Scalars*, const char *keys);  // <4>

void vtk_write(VTK*, MPI_Comm, int id); // <5>
void vtk_fin(VTK*);
// tag::main[]
/* end src/io/vtk/imp.h */

/* begin src/io/vtk/imp/type.h */
struct VTKConf {
    Mesh *mesh;
    KeyList *tri, *vert;
};

enum { MAGIC = 1234 };
#ifndef UDX_IO_OUTPUT_COMMON_LIMITS
#define UDX_IO_OUTPUT_COMMON_LIMITS
#ifndef UDX_IO_OUTPUT_COMMON_UNSET
#define UDX_IO_OUTPUT_COMMON_UNSET
enum { UNSET = - 1 };
#endif
enum { N_MAX = 100 };
#endif
struct VTK {
    int nm; /* current number of meshes */
    Mesh *mesh;
    KeyList *tri, *vert;

    int nbuf; /* maximum bufer size */
    double *dbuf;
    int *ibuf;

    double *TRI[N_MAX];
    double *VERT[N_MAX];

    int rr_set;
    char path[FILENAME_MAX];

    int stamp; /* MAGIC */
};

struct Out {
    WriteFile *file;
    MPI_Comm comm;
};
/* end src/io/vtk/imp/type.h */
/* begin src/io/vtk/imp/util.h */
#ifndef UDX_IO_OUTPUT_COMMON_UTIL
#define UDX_IO_OUTPUT_COMMON_UTIL
static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
static char *cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }
#endif

static void mkdir(MPI_Comm comm, const char *p, const char *s) {
    char path[FILENAME_MAX] = {'\0'};
    if (!m::is_master(comm)) return;
    cpy(path, p);
    cat(path, "/");
    cat(path, s);
    msg_print("mkdir -p '%s'", path);
    UC(os_mkdir(path));
}

static int max3(int a, int b, int c) {
    if (a > b) b = a;
    if (b > c) c = b;
    return c;
}
static int get_nbuf(int nm, int nv, int nt, int ne) {
    return max3(3*nm*nv, 4*nm*nt, 2*nm*ne);
}

static int little_p0() {
    int a;
    unsigned char *b;
    a = 1;
    b = (unsigned char*)&a;
    return (*b) != 0;
}

static int little_p() {
    static int done = 0;
    static int little = 0;
    if (done) return little;
    little = little_p0();
    done = 1;
    return little;
}

static void big_endian_dbl0(double *px) {
    const int n = sizeof(double);
    int i;
    unsigned char *c, b[n];
    if (!little_p()) return;
    c = (unsigned char*)px;
    for (i = 0; i < n; i++) b[i] = c[i];
    for (i = 0; i < n; i++) c[i] = b[n - i - 1];
}

static void big_endian_dbl(int n, double *d) {
    int i;
    for (i = 0; i < n; i++)
        big_endian_dbl0(&d[i]);
}

static void big_endian_int0(int *px) {
    const int n = sizeof(int);
    int i;
    unsigned char *c, b[n];
    if (!little_p()) return;
    c = (unsigned char*)px;
    for (i = 0; i < n; i++) b[i] = c[i];
    for (i = 0; i < n; i++) c[i] = b[n - i - 1];
}

static void big_endian_int(int n, int *d) {
    int i;
    for (i = 0; i < n; i++)
        big_endian_int0(&d[i]);
}
/* end src/io/vtk/imp/util.h */
/* begin src/io/vtk/imp/print.h */
enum { BUF = 9999 };

static void print(Out *out, const char *fmt, ...) {
    char s[BUF];
    int size;
    va_list args;
    size = 0;
    if (m::is_master(out->comm)) {
        va_start(args, fmt);
        size = vsnprintf(s, BUF - 1, fmt, args);
        va_end(args);
    }
    UC(write_master(out->comm, s, size, out->file));
}
/* end src/io/vtk/imp/print.h */
/* begin src/io/vtk/imp/file.h */
static void header(Out *o) {
    print(o, "# vtk DataFile Version 2.0\n");
    print(o, "created with uDeviceX\n");
    print(o, "BINARY\n");
    print(o, "DATASET POLYDATA\n");
}

static void points(Out *o, int n, double *rr) {
    int n_total;
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    big_endian_dbl(3*n, /**/ rr);
    UC(write_reduce(comm, n, &n_total));
    print(o, "POINTS %d double\n", n_total);
    UC(write_all(comm, rr, 3*n*sizeof(rr[0]), file));
    print(o, "\n");
}

static void tri(Out *o, int nm, int nv, int nt, const int *tt, int *buf) {
    enum {NVP = 3};
    int n, m, t, i, shift, nm_total;
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    n = nm * nv;
    UC(write_shift_indices(comm, n, &shift));
    for (t = m = 0; m < nm; m++)
        for (i = 0; i < nt; i++) {
            buf[t++] = NVP;
            buf[t++] = shift + nv*m + tt[3*i + 0];
            buf[t++] = shift + nv*m + tt[3*i + 1];
            buf[t++] = shift + nv*m + tt[3*i + 2];
        }
    big_endian_int(t, /**/ buf);
    UC(write_reduce(comm, nm, &nm_total));
    print(o, "POLYGONS %d %d\n", nt*nm_total, 4*nt*nm_total);
    UC(write_all(comm, buf, t*sizeof(buf[0]), file));
    print(o, "\n");
}

static void cell_header(Out *o, int n) {
    int n_total;
    MPI_Comm comm;
    comm = o->comm;
    UC(write_reduce(comm, n, &n_total));
    print(o, "CELL_DATA %d\n", n_total);
}

static void cell_data(Out *o, int n, double *data, const char *name) {
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    print(o, "SCALARS %s double 1\n", name);
    print(o, "LOOKUP_TABLE default\n");

    big_endian_dbl(n, /**/ data);
    UC(write_all(comm, data, n*sizeof(data[0]), file));
    print(o, "\n");
}

static void point_header(Out *o, int n) {
    int n_total;
    MPI_Comm comm;
    comm = o->comm;
    UC(write_reduce(comm, n, &n_total));
    print(o, "POINT_DATA %d\n", n_total);
}

static void point_data(Out *o, int n, double *data, const char *name) {
    MPI_Comm comm;
    WriteFile *file;
    comm = o->comm;
    file = o->file;
    print(o, "SCALARS %s double 1\n", name);
    print(o, "LOOKUP_TABLE default\n");

    big_endian_dbl(n, /**/ data);
    UC(write_all(comm, data, n*sizeof(data[0]), file));
    print(o, "\n");
}
/* end src/io/vtk/imp/file.h */
/* begin src/io/vtk/imp/main.h */
#define PATTERN "%s/%s/%05d.vtk"

void vtk_conf_ini(MeshRead *mesh, /**/ VTKConf **pq) {
    VTKConf *q;
    EMALLOC(1, &q);
    mesh_ini(mesh, &q->mesh);
    key_list_ini(&q->tri);
    key_list_ini(&q->vert);
    *pq = q;
}

void vtk_conf_fin(VTKConf *q) {
    key_list_fin(q->tri);
    key_list_fin(q->vert);
    mesh_fin(q->mesh);
    EFREE(q);
}

void vtk_conf_tri(VTKConf *q, const char *keys) {
    int n;
    n = key_list_size(q->tri);
    if (n > N_MAX)
        ERR("n=%d > N_MAX=%d", n, N_MAX);
    UC(key_list_push(q->tri, keys));
}

void vtk_conf_vert(VTKConf *q, const char *keys) {
    int n;
    n = key_list_size(q->tri);
    if (n > N_MAX)
        ERR("n=%d > N_MAX=%d", n, N_MAX);
    UC(key_list_push(q->vert, keys));
}

void vtk_ini(MPI_Comm comm, int maxm, char const *path, VTKConf *c, /**/ VTK **pq) {
    int i;
    VTK *q;
    int nbuf, nk;
    Mesh *mesh;
    EMALLOC(1, &q);
    mesh = c->mesh;
    nbuf = get_nbuf(maxm, mesh_nv(mesh), mesh_nt(mesh), mesh_ne(mesh));
    EMALLOC(nbuf, &q->dbuf);
    EMALLOC(nbuf, &q->ibuf);
    UC(mkdir(comm, DUMP_BASE, path));
    cpy(q->path, path);
    UC(nk = key_list_size(c->tri));
    for (i = 0; i < nk; i++) EMALLOC(nbuf, &q->TRI[i]);
    UC(nk = key_list_size(c->vert));
    for (i = 0; i < nk; i++) EMALLOC(nbuf, &q->VERT[i]);

    q->stamp = MAGIC;
    q->nbuf = nbuf;
    q->nm       = UNSET;
    q->rr_set   = 0;
    UC(mesh_copy(mesh, /**/ &q->mesh));
    UC(key_list_copy(c->tri, &q->tri));
    UC(key_list_copy(c->vert, &q->vert));

    *pq = q;
}

void vtk_points(VTK *q, int nm, const Vectors *pos) {
    enum {X, Y, Z};
    float r[3];
    int nv, n, i, j;
    if (q->nm != UNSET && q->nm != nm)
        ERR("q->nm=%d  != nm=%d", q->nm, nm);

    nv = mesh_nv(q->mesh);
    n = nv * nm;
    for (i = j = 0; i < n; i++) {
        UC(vectors_get(pos, i, r));
        q->dbuf[j++] = r[X];
        q->dbuf[j++] = r[Y];
        q->dbuf[j++] = r[Z];
    }

    q->nm = nm;
    q->rr_set = 1;
}

void vtk_tri(VTK *q, int nm, const Scalars *sc, const char *keys) {
    int i, j, nt, n;
    if (!key_list_has(q->tri, keys)) {
        msg_print("unkown key '%s'", keys);
        key_list_log(q->tri);
        ERR("");
    }
    j = key_list_offset(q->tri, keys);
    nt = mesh_nt(q->mesh);
    n = nm * nt;
    for (i = 0; i < n; i++)
        q->TRI[j][i] = scalars_get(sc, i);
    UC(key_list_mark(q->tri, keys));
}

void vtk_vert(VTK *q, int nm, const Scalars *sc, const char *keys) {
    int i, j, nv, n;
    if (!key_list_has(q->vert, keys)) {
        msg_print("unkown key '%s'", keys);
        key_list_log(q->vert);
        ERR("wrong vtk_vert call");
    }
    j = key_list_offset(q->vert, keys);
    nv = mesh_nv(q->mesh);
    n = nm * nv;
    for (i = 0; i < n; i++)
        UC(q->VERT[j][i] = scalars_get(sc, i));
    UC(key_list_mark(q->vert, keys));
}

void vtk_write(VTK *q, MPI_Comm comm, int id) {
    char path[FILENAME_MAX] = {'\0'};
    int i, nk, n, nm, nv, nt, nbuf;
    const int *tt;
    const char *keys;
    Out out;
    if (q->stamp != MAGIC)
        ERR("VTK is not initialized");

    if (!key_list_marked(q->tri)) {
        msg_print("missing triangle data");
        key_list_log(q->tri);
        ERR("vtk_write is called is not enough data");
    }
    if (!key_list_marked(q->vert)) {
        msg_print("missing vertices data");
        key_list_log(q->vert);
        ERR("vtk_write is called is not enough data");
    }
    if (!q->rr_set) ERR("points are unset");

    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, q->path, id) < 0)
        ERR("snprintf failed");
    out.comm = comm;
    UC(write_file_open(comm, path, &out.file));

    nm = q->nm; nbuf = q->nbuf; nv = mesh_nv(q->mesh);
    nt = mesh_nt(q->mesh); tt = mesh_tt(q->mesh);

    n = nm * nv;
    if (3 * n > nbuf)
        ERR("3*n=%d > nbuf=%d", n, nbuf);
    if (4 * nm * nt > nbuf)
        ERR("nm=%d * nt=%d > nbuf=%d", nm, nt, nbuf);

    header(&out);
    points(&out, n, q->dbuf);
    tri(&out, nm, nv, nt, tt, q->ibuf);

    UC(nk = key_list_size(q->tri));
    if (nk > 0) cell_header(&out, nm*nt);
    for (i = 0; i < nk; i++) {
        keys = key_list_name(q->tri, i);
        cell_data(&out, nm*nt, q->TRI[i], keys);
    }

    UC(nk = key_list_size(q->vert));
    if (nk > 0) point_header(&out, n);
    for (i = 0; i < nk; i++) {
        keys = key_list_name(q->vert, i);
        point_data(&out, n, q->VERT[i], keys);
    }

    UC(write_file_close(out.file));

    q->rr_set = 0;
    q->nm = UNSET;
    key_list_unmark(q->tri);
    key_list_unmark(q->vert);
}

void vtk_fin(VTK *q) {
    int i;
    if (q->stamp != MAGIC)
        ERR("VTK is not initialized");
    EFREE(q->dbuf);
    EFREE(q->ibuf);
    for (i = 0; i < key_list_size(q->tri); i++)
        EFREE(q->TRI[i]);
    for (i = 0; i < key_list_size(q->vert); i++)
        EFREE(q->VERT[i]);
    key_list_fin(q->tri);
    key_list_fin(q->vert);
    EFREE(q);
}
/* end src/io/vtk/imp/main.h */
/* end src/io/vtk/imp.cpp */

#line 1 "src/io/vtk/mesh/imp.cpp"
