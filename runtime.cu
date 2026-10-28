





















#include "config.cuh"

#include <stdio.h>
#include <assert.h>




void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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
void msg_ini(int rank);
void msg_print(const char *fmt, ...);

#ifndef UDX_D_MPI_COLOR_D_API
#define UDX_D_MPI_COLOR_D_API
#include "device.cuh"
#endif




#include <stdlib.h>
#include <string.h>
#include <mpi.h>






#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

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


bool is_master(MPI_Comm comm);
int is_success(int errorcode);
int is_pending(int errorcode);
int is_err_in_status(int errorcode);
int status2errcode(MPI_Status *status);
}
namespace m {
enum {X, Y, Z, D};

void ini(int *argc, char ***argv) {
    if (m::Init(argc, argv) != MPI_SUCCESS) {
        fprintf(stderr, ": m::Init failed\n");
        exit(2);
    }
    if (m::Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN) != MPI_SUCCESS) {
        fprintf(stderr, ": m::Errhandler_set failed\n");
        exit(2);
    }
}

static void shift(int *argc, char ***argv) {
    (*argc)--;
    (*argv)++;
}
static int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

void get_dims(int *argc, char ***argv, int dims[]) {
    int i, d, ac;
    char **av;


    dims[X] = dims[Y] = dims[Z] = 1;
    ac = *argc; av = *argv;


    shift(&ac, &av);

    for (i = 0; ac > 0 && i <= D; i++) {
        if (eq(av[0], "--")) {
            shift(&ac, &av);
            break;
        }
        if ( (d = atoi(av[0])) == 0 ) break;
        dims[i] = d;
        shift(&ac, &av);
    }

    *argc = ac; *argv = av;
}

void get_cart(MPI_Comm comm, const int dims[3],      MPI_Comm *cart) {
    enum {X, Y, Z};
    const int reorder = 0;
    const int periods[D] = {1, 1, 1};
    int size;
    MC(m::Comm_size(comm, &size));
    if (size != dims[X]*dims[Y]*dims[Z]) {
        fprintf(stderr, "%s:%d: fail to create cartesian geometry\n", __FILE__, __LINE__);
        fprintf(stderr, "size=%d  !=  dims[X]=%d  * dims[Y]=%d  * dims[Z]=%d\n",
                size, dims[X], dims[Y], dims[Z]);
        exit(2);
    }
    MC(m::Cart_create(comm, D, dims, periods, reorder, cart));
}

void fin() { MC(m::Finalize()); }

}



#include <mpi.h>


namespace m {
int Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    return MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

int Barrier(MPI_Comm comm) {
    return MPI_Barrier(comm);
}

int Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    return MPI_Bcast(buffer, count, datatype, root, comm);
}

int Cancel(MPI_Request *request) {
    return MPI_Cancel(request);
}

int Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
                const int periods[], int reorder, MPI_Comm *comm_cart) {
    return MPI_Cart_create(comm_old, ndims, dims, periods, reorder, comm_cart);
}

int Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]) {
    return MPI_Cart_coords(comm, rank, maxdims, coords);
}

int Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]) {
    return MPI_Cart_get(comm, maxdims, dims, periods, coords);
}

int Cart_rank(MPI_Comm comm, const int coords[], int *rank) {
    return MPI_Cart_rank(comm, coords, rank);
}

int Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
    return MPI_Comm_dup(comm, newcomm);
}

int Comm_free(MPI_Comm *comm) {
    return MPI_Comm_free(comm);
}

int Comm_rank(MPI_Comm comm, int *rank) {
    return MPI_Comm_rank(comm, rank);
}

int Comm_size( MPI_Comm comm, int *size ) {
    return MPI_Comm_size(comm, size);
}

int Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler) {
    return MPI_Errhandler_set(comm, errhandler);
}

int Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    return MPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm);
}

int File_close(MPI_File *fh) {
    return MPI_File_close(fh);
}

int File_get_position(MPI_File fh, MPI_Offset *offset) {
    return MPI_File_get_position(fh, offset);
}

int File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh) {
    return MPI_File_open(comm, filename, amode, info, fh);
}

int File_seek(MPI_File fh, MPI_Offset offset, int whence) {
    return MPI_File_seek(fh, offset, whence);
}

int File_set_size(MPI_File fh, MPI_Offset size) {
    return MPI_File_set_size(fh, size);
}

int File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
    return MPI_File_write_at_all(fh, offset, buf, count, datatype, status);
}

int Finalize(void) {
    return MPI_Finalize();
}

int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
           int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
    return MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
}

int Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
            const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm) {
    return MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
}

int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
              int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    return MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
}

int Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
               const int displs[], MPI_Datatype recvtype, MPI_Comm comm) {
    return MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
}

int Init(int *argc, char ***argv) {
    return MPI_Init(argc, argv);
}

int Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request) {
    return MPI_Irecv(buf, count, datatype, source, tag, comm, request);
}

int Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) {
    return MPI_Recv(buf, count, datatype, source, tag, comm, status);
}

int Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    return MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

int Type_commit(MPI_Datatype *type) {
    return MPI_Type_commit(type);
}

int Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype) {
    return MPI_Type_contiguous(count, oldtype, newtype);
}

int Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses) {
    return MPI_Waitall(count, array_of_requests, array_of_statuses);
}

int Probe(int source, int tag, MPI_Comm comm, MPI_Status *status) {
    return MPI_Probe(source, tag, comm, status);
}

int Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count) {
    return MPI_Get_count(status, datatype, count);
}

int Error_string(int errorcode, char *string, int *resultlen) {
    return MPI_Error_string(errorcode, string, resultlen);
}

bool is_master(MPI_Comm comm) {
    int rank;
    Comm_rank(comm, &rank);
    return rank == 0;
}

int is_success(int errorcode) { return errorcode == MPI_SUCCESS; }
int is_pending(int errorcode) { return errorcode == MPI_ERR_PENDING; }
int is_err_in_status(int errorcode) { return errorcode == MPI_ERR_IN_STATUS; }

int status2errcode(MPI_Status *status) { return status->MPI_ERROR; }
}









#include <stdio.h>
#include <curand_kernel.h>
#include <assert.h>






#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}




#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
#include "launch.cuh"
#endif
#ifndef UDX_D_MPI_COLOR_D_API
#define UDX_D_MPI_COLOR_D_API
#endif
#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif


#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())

#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))


#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))


#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)


#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)
#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#include "types.cuh"
#endif

#ifndef UDX_CONTROL_COORDS_COMMON
#define UDX_CONTROL_COORDS_COMMON
struct Coords_v {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};
struct Coords {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};
#endif
struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);

struct int3;
struct Particle;
struct Coords;

struct Rnd;

void color_linear_flux(const Coords*, int3 L, int dir, int color, int n, const Particle *pp,     int *cc);

void color_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc);
void decolor_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc);

void tracers_ini(const int n);

void tracers_fin();
struct Rnd {
    curandState_t * state;
} color_rnd;
__global__ void color_linear_flux_kernel(int3 L, int dir, int color, int n, const Particle *pp, int *cc) {
    int i;
    Particle p;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    p = pp[i];
    const int HL[] = {L.x/2, L.y/2, L.z/2};

    if (p.r[dir] >= HL[dir])
        cc[i] = color;
}

__global__ void color_radial_flux_kernel(const Coords_v c, const float R, const float Po, int color, int n, const Particle *pp, int *cc, curandState_t *rg) {

    long i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t rnd_state = rg[i];
    double rand = curand_normal(&rnd_state);
    double cdf = normcdf(rand);

    if (cdf <= Po)
    {
        int i;
        Particle p;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n) return;
        p = pp[i];

        float pxc = p.r[0] - 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
        float pyc = p.r[1] - 0.5f * c.Ly * (c.yd - 2 * c.yc - 1);
        float pzc = p.r[2] - 0.5f * c.Lz * (c.zd - 2 * c.zc - 1);

        const float r = sqrtf( pxc*pxc + pyc*pyc + pzc*pzc );

        if (r <= R) {
            cc[i] = color;
        }
    }
}

__global__ void color_decolor_tracer_kernel(const Coords_v c, const float R, const float Po, const int color, const int n, const Particle *pp, int *cc, curandState_t *rg) {

    int i;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n)
        return;

    if (cc[i] == color)
        return;

    Particle p;
    p = pp[i];

    curandState_t rnd_state = rg[i];
    double rand = curand_normal(&rnd_state);
    double cdf = normcdf(rand);

    if (cdf <= Po)
    {
        float pxc = p.r[0] - 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
        float pyc = p.r[1] - 0.5f * c.Ly * (c.yd - 2 * c.yc - 1);
        float pzc = p.r[2] - 0.5f * c.Lz * (c.zd - 2 * c.zc - 1);

        const float r = sqrtf( pxc*pxc + pyc*pyc + pzc*pzc );

        if (r > R)
            cc[i] = color;
    }
}

__global__ void color_ini_rnd_kernel(long seed, int n, curandState_t *state) {
    long i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &state[i]);
}

void color_linear_flux(const Coords *coords, int3 L, int dir, int color, int n, const Particle *pp, int *cc) {
    assert(dir >= 0 && dir <= 2);
    if (is_end(coords, dir))
        KL(color_linear_flux_kernel, (k_cnf(n)), (L, dir, color, n, pp, cc));
}

void color_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc) {

    int l[3] = {xlo(coords), ylo(coords), zlo(coords)};
    int h[3] = {xhi(coords), yhi(coords), zhi(coords)};


    bool inradius = 1;
    for (int i=0; i<2; ++i)
        inradius *= ( sqrtf(l[i]*l[i]) <= R || sqrtf(h[i]*h[i]) <= R );

    if (inradius) {
        Coords_v view;
        coords_get_view(coords, &view);
    KL(color_radial_flux_kernel, (k_cnf(n)), (view, R, Po, color, n, pp, cc, color_rnd.state));
    }
}

void decolor_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc) {
    Coords_v view;
    coords_get_view(coords, &view);
    KL(color_decolor_tracer_kernel, (k_cnf(n)), (view, R, Po, color, n, pp, cc, color_rnd.state));
}

static void color_ini_rnd(int n, curandState_t *state) {
    long seed = 1234567;
    KL(color_ini_rnd_kernel, (k_cnf(n)), (seed, n, state));
}

void tracers_ini(const int n) {
    Dalloc(&color_rnd.state, n);
    color_ini_rnd(n, color_rnd.state);
}

void tracers_fin() {
    Dfree(color_rnd.state);
}






#include <stdio.h>
#include <mpi.h>

#include <vector_types.h>






struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);

struct Coords;
struct Config;

void coords_ini(MPI_Comm cart, int Lx, int Ly, int Lz, Coords **c);
void coords_fin(Coords *c);

void coords_ini_conf(MPI_Comm cart, const Config *cfg, Coords **c);


void coords_ini_conf(MPI_Comm cart, const Config *cfg, Coords **c) {
    int3 L;
    UC(conf_lookup_int3(cfg, "glb/L", &L));
    UC(coords_ini(cart, L.x, L.y, L.z, c));
}



#include <vector_types.h>
#include <mpi.h>










#ifndef UDX_CONTROL_COORDS_COMMON
#define UDX_CONTROL_COORDS_COMMON
struct Coords_v {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};

struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);
struct Coords {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};
#endif
enum {X, Y, Z, D};

void coords_get_view(const Coords *c, Coords_v *v) {
    v->xc = c->xc;
    v->yc = c->yc;
    v->zc = c->zc;

    v->xd = c->xd;
    v->yd = c->yd;
    v->zd = c->zd;

    v->Lx = c->Lx;
    v->Ly = c->Ly;
    v->Lz = c->Lz;
}

void coords_ini(MPI_Comm cart, int Lx, int Ly, int Lz, Coords **c0) {
    Coords *c;
    int dims[D], periods[D], coords[D];
    MC(m::Cart_get(cart, D, dims, periods, coords));

    EMALLOC(1, c0);
    c = *c0;

    c->xc = coords[X];
    c->yc = coords[Y];
    c->zc = coords[Z];

    c->xd = dims[X];
    c->yd = dims[Y];
    c->zd = dims[Z];

    c->Lx = Lx;
    c->Ly = Ly;
    c->Lz = Lz;
}

void coords_fin(Coords *c) {
    EFREE(c);
}



int xdomain(const Coords *c) {
    return c->Lx * c->xd;
}

int ydomain(const Coords *c) {
    return c->Ly * c->yd;
}

int zdomain(const Coords *c) {
    return c->Lz * c->zd;
}



float xl2xc(const Coords *c, float xl) {
    return xl - 0.5f * c->Lx * (c->xd - 2 * c->xc - 1);
}

float yl2yc(const Coords *c, float yl) {
    return yl - 0.5f * c->Ly * (c->yd - 2 * c->yc - 1);
}

float zl2zc(const Coords *c, float zl) {
    return zl - 0.5f * c->Lz * (c->zd - 2 * c->zc - 1);
}

void local2center(const Coords *c, float3 rl,      float3 *rc) {
    rc->x = xl2xc(c, rl.x);
    rc->y = yl2yc(c, rl.y);
    rc->z = zl2zc(c, rl.z);
}



float xc2xl(const Coords *c, float xc) {
    return xc + 0.5f * c->Lx * (c->xd - 2 * c->xc - 1);
}

float yc2yl(const Coords *c, float yc) {
    return yc + 0.5f * c->Ly * (c->yd - 2 * c->yc - 1);
}

float zc2zl(const Coords *c, float zc) {
    return zc + 0.5f * c->Lz * (c->zd - 2 * c->zc - 1);
}

void center2local(const Coords *c, float3 rc,      float3 *rl) {
    rl->x = xc2xl(c, rc.x);
    rl->y = yc2yl(c, rc.y);
    rl->z = zc2zl(c, rc.z);
}



float xl2xg(const Coords *c, float xl) {
    return (c->xc + 0.5f) * c->Lx + xl;
}

float yl2yg(const Coords *c, float yl) {
    return (c->yc + 0.5f) * c->Ly + yl;
}

float zl2zg(const Coords *c, float zl) {
    return (c->zc + 0.5f) * c->Lz + zl;
}

void local2global(const Coords *c, float3 rl,      float3 *rg) {
    rg->x = xl2xg(c, rl.x);
    rg->y = yl2yg(c, rl.y);
    rg->z = zl2zg(c, rl.z);
}



float xg2xl(const Coords *c, float xg) {
    return xg - (c->xc + 0.5f) * c->Lx;
}

float yg2yl(const Coords *c, float yg) {
    return yg - (c->yc + 0.5f) * c->Ly;
}

float zg2zl(const Coords *c, float zg) {
    return zg - (c->zc + 0.5f) * c->Lz;
}

void global2local(const Coords *c, float3 rg,      float3 *rl) {
    rl->x = xg2xl(c, rg.x);
    rl->y = yg2yl(c, rg.y);
    rl->z = zg2zl(c, rg.z);
}



int xlo(const Coords *c) { return xl2xg(c, 0) - 0.5*c->Lx; }
int ylo(const Coords *c) { return yl2yg(c, 0) - 0.5*c->Ly; }
int zlo(const Coords *c) { return zl2zg(c, 0) - 0.5*c->Lz; }

int xhi(const Coords *c) { return xl2xg(c, 0) + 0.5*c->Lx; }
int yhi(const Coords *c) { return yl2yg(c, 0) + 0.5*c->Ly; }
int zhi(const Coords *c) { return zl2zg(c, 0) + 0.5*c->Lz; }



int xs(const Coords *c) { return c->Lx; }
int ys(const Coords *c) { return c->Ly; }
int zs(const Coords *c) { return c->Lz; }
int3 subdomain(const Coords *c) {
    int3 L;
    L.x = xs(c);
    L.y = ys(c);
    L.z = zs(c);
    return L;
}



bool is_end(const Coords *c, int dir) {
    enum {X, Y, Z};
    switch (dir) {
    case X: return c->xc == c->xd - 1;
    case Y: return c->yc == c->yd - 1;
    case Z: return c->zc == c->zd - 1;
    }
    return false;
}

static int bigp(int max, int x, int y, int z) {
    return x > max || y > max || z > max;
}
void coord_stamp(const Coords *c,      char *s) {
    int x, y, z;
    x = c->xc; y = c->yc; z = c->zc;
    if (bigp(999, x, y, z))
        ERR("too big domain [%d %d %d]", x, y, z);
    if (sprintf(s, "%03d.%03d.%03d", x, y, z) < 3)
        ERR("sprintf failed [%d %d %d]", x, y, z);
}

int coords_size(const Coords *c) {
    return c->xd * c->yd * c->zd;
}










#include <assert.h>
#include <string.h>
#include <mpi.h>




#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())










#ifndef UDX_SHARED_FRAG_COMMON
#define UDX_SHARED_FRAG_COMMON
enum {frag_bulk = 26};

struct int3;

namespace frag_hst {

int i2dx(int i);
int i2dy(int i);
int i2dz(int i);
void i2d3(int i,      int d[3]);

int d2i(int x, int y, int z);
int d32i(const int d[3]);

int ncell(int3 L, int i);

int antid2i(int x, int y, int z);
int anti(int i);

void estimates(int3 L, int nfrags, float maxd,      int *cap);

}
#endif















#ifndef UDX_D_MPI_COLOR_D_API
#define UDX_D_MPI_COLOR_D_API
#endif

typedef unsigned char data_t;

enum {
    NFRAGS = 26,
    BULK   = 26,
    NBAGS  = 27,
};


enum AllocMod {
    HST_ONLY,
    DEV_ONLY,
    PINNED,
    PINNED_HST,
    PINNED_DEV,
    NONE
};

struct dBags {
    data_t *data[NBAGS];
};

struct hBags {
    data_t *data[NBAGS];
    int         *counts;
    int capacity[NBAGS];
    size_t bsize;
};

struct Comm;
struct CommBuffer;

void comm_bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS],      hBags *hb, dBags *db);
void comm_bags_fin(AllocMod fmod, AllocMod bmod,      hBags *hb, dBags *db);
void comm_ini(MPI_Comm cart,      Comm **c);
void comm_fin(     Comm *c);
void comm_buffer_ini(int nbags, const hBags *hbb, CommBuffer**);
void comm_buffer_fin(CommBuffer*);


void comm_buffer_set(int nbags, const hBags*, CommBuffer*);
void comm_buffer_get(const CommBuffer*, int nbags, hBags*);

void comm_post_recv(hBags *b, Comm *c);
void comm_post_send(const hBags *b, Comm *c);

void comm_wait_recv(Comm *c,      hBags *b);
void comm_wait_send(Comm *c);

void comm_post_recv(CommBuffer *cb, Comm *c);
void comm_post_send(const CommBuffer *cb, Comm *c);

void comm_wait_recv(Comm *c,      CommBuffer *cb);
void comm_wait_send(Comm *c);

int    comm_get_number_capacity(int i, const hBags *b);
size_t comm_get_byte_capacity(int i, const hBags *b);


struct Comm {
    MPI_Request sreq[NBAGS];
    MPI_Request rreq[NBAGS];
    MPI_Comm cart;
    int ranks[NFRAGS];
    int  tags[NFRAGS];
};

struct CommBuffer {
    data_t *buf[NBAGS];
    size_t sz[NBAGS], cap[NBAGS];
};


static void alloc_counts(int n,      int **hc) {
    size_t sz = n * sizeof(int);
    EMALLOC(n, hc);
    memset(*hc, 0, sz);
}


static void alloc_pair(int i, AllocMod mod,      hBags *hb, dBags *db) {
    size_t n = hb->capacity[i] * hb->bsize;

    switch (mod) {
    case HST_ONLY:
        if (n) UC(emalloc(n, (void**) &hb->data[i]));
        else   hb->data[i] = NULL;
        break;
    case DEV_ONLY:
        if (n) CC(d::Malloc((void**) &db->data[i], n));
        else db->data[i] = NULL;
        break;
    case PINNED:
        if (n) {
            CC(d::alloc_pinned((void**) &hb->data[i], n));
            CC(d::HostGetDevicePointer((void**) &db->data[i], hb->data[i], 0));
        } else {
            hb->data[i] = db->data[i] = NULL;
        }
        break;
    case PINNED_HST:
        if (n) CC(d::alloc_pinned((void**) &hb->data[i], n));
        else hb->data[i] = NULL;
        break;
    case PINNED_DEV:
        if (n) {
            CC(d::alloc_pinned((void**) &hb->data[i], n));
            CC(d::Malloc((void **) &db->data[i], n));
        } else {
            hb->data[i] = db->data[i] = NULL;
        }
        break;
    case NONE:
    default:
        break;
    }
}

void comm_bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS],      hBags *hb, dBags *db) {
    hb->bsize = bsize;
    memcpy(hb->capacity, capacity, NBAGS * sizeof(int));


    for (int i = 0; i < NFRAGS; ++i)
        UC(alloc_pair(i, fmod,      hb, db));


    UC(alloc_pair(frag_bulk, bmod,      hb, db));

    UC(alloc_counts(NBAGS,      &hb->counts));
}

static void comm_buffer_ini_one_frag(int fid, int nbags, const hBags *hbb, CommBuffer *b) {
    size_t bs, cap;
    int i;
    cap = nbags * sizeof(int);
    for (i = 0; i < nbags; ++i) {
        bs = hbb[i].bsize;
        cap += hbb[i].capacity[fid] * bs;
    }
    EMALLOC(cap, &b->buf[fid]);
    b->cap[fid] = cap;
    b->sz[fid] = 0;
}

void comm_buffer_ini(int nbags, const hBags *hbb, CommBuffer **cb) {
    EMALLOC(1, cb);
    for (int i = 0; i < NFRAGS; ++i)
        comm_buffer_ini_one_frag(i, nbags, hbb, *cb);
}

void comm_ini(MPI_Comm cart,      Comm **com_p) {
    int i, c, crd_rnk[3], d[3];
    int coords[3], periods[3], dims[3];
    Comm *com;

    EMALLOC(1, com_p);
    com = *com_p;

    MC(m::Cart_get(cart, 3, dims, periods, coords));

    for (i = 0; i < NFRAGS; ++i) {
        frag_hst::i2d3(i, d);
        for (c = 0; c < 3; ++c)
            crd_rnk[c] = coords[c] + d[c];
        MC(m::Cart_rank(cart, crd_rnk, com->ranks + i));
        com->tags[i] = frag_hst::anti(i);
    }
    MC(m::Comm_dup(cart, &com->cart));
}

static void free_counts(int **hc) {
    EFREE(*hc);
    *hc = NULL;
}

static void free_pair(int i, AllocMod mod,      hBags *hb, dBags *db) {
    switch (mod) {
    case HST_ONLY:
        if (hb->data[i])
            EFREE(hb->data[i]);
        break;
    case DEV_ONLY:
        if (db->data[i])
            CC(d::Free(db->data[i]));
        break;
    case PINNED_HST:
    case PINNED:
        if (hb->data[i])
            CC(d::FreeHost(hb->data[i]));
        break;
    case PINNED_DEV:
        if (hb->data[i])
            CC(d::FreeHost(hb->data[i]));
        if (db->data[i])
            CC(d::Free(db->data[i]));
        break;
    case NONE:
    default:
        break;
    }
}

void comm_bags_fin(AllocMod fmod, AllocMod bmod,      hBags *hb, dBags *db) {

    for (int i = 0; i < NFRAGS; ++i)
        free_pair(i, fmod,      hb, db);


    free_pair(frag_bulk, bmod,      hb, db);
    free_counts(     &hb->counts);
}

void comm_buffer_fin(CommBuffer *cb) {
    int i;
    for (i = 0; i < NFRAGS; ++i)
        EFREE(cb->buf[i]);
    EFREE(cb);
}

void comm_fin(     Comm *c) {
    MC(m::Comm_free(&c->cart));
    EFREE(c);
}
static void frag_set_buffer(int fid, int nbags, const hBags *hbb, CommBuffer *cb) {
    size_t tot_sz, sz, bs;
    int i, c, *counts;
    data_t *d;
    const hBags *hb;

    d = cb->buf[fid];
    counts = (int*) d;
    tot_sz = sz = nbags * sizeof(int);
    d += sz;

    for (i = 0; i < nbags; ++i) {
        hb = &hbb[i];
        bs = hb->bsize;
        c  = hb->counts[fid];
        sz = c * bs;
        memcpy(d, hb->data[fid], sz);
        counts[i] = c;
        d += sz;
        tot_sz += sz;
    }
    cb->sz[fid] = tot_sz;

    if (tot_sz > cb->cap[fid]) ERR("fragment %d: exceed buffer size %ld/%ld\n", fid, tot_sz, cb->cap[fid]);
}

void comm_buffer_set(int nbags, const hBags *hbb, CommBuffer *cb) {
    for (int fid = 0; fid < NFRAGS; ++fid)
        frag_set_buffer(fid, nbags, hbb, cb);
}

static void frag_get_buffer(int fid, const CommBuffer *cb, int nbags, hBags *hbb) {
    size_t sz;
    int i, c, *counts;
    data_t *d;
    hBags *hb;

    d = cb->buf[fid];
    counts = (int*) d;
    d += nbags * sizeof(int);

    for (i = 0; i < nbags; ++i) {
        hb = &hbb[i];
        c = counts[i];
        hb->counts[fid] = c;
        sz = c * hb->bsize;
        memcpy(hb->data[fid], d, sz);
        d += sz;
    }
}

void comm_buffer_get(const CommBuffer *cb, int nbags, hBags *hbb) {
    for (int fid = 0; fid < NFRAGS; ++fid)
        frag_get_buffer(fid, cb, nbags, hbb);
}
void comm_post_recv(hBags *b, Comm *com) {
    int i, c, tag;
    for (i = 0; i < NFRAGS; ++i) {
        c = b->capacity[i] * b->bsize;
        tag = com->tags[i];
        MC(m::Irecv(b->data[i], c, MPI_BYTE, com->ranks[i], tag, com->cart, com->rreq + i));
    }
}

static void fail_over(int i, long c, long cap) {
    enum {X, Y, Z};
    int d[3];
    frag_hst::i2d3(i, d);
    ERR("over capacity, fragment %d = [%d %d %d]: %ld/%ld",
        i, d[X], d[Y], d[Z], c, cap);
}

void comm_post_send(const hBags *b, Comm *com) {
    int i, n, c, cap, tag;
    for (i = 0; i < NFRAGS; ++i) {
        c = b->counts[i];
        cap = b->capacity[i];
        n = c * b->bsize;
        tag = i;
        if (c > cap) UC(fail_over(i, c, cap));
        MC(m::Isend(b->data[i], n, MPI_BYTE, com->ranks[i], tag, com->cart, com->sreq + i));
    }
}

static void get_counts_bytes(const MPI_Status ss[NFRAGS],      int counts[NFRAGS]) {
    for (int i = 0; i < NFRAGS; ++i)
        MC(m::Get_count(ss + i, MPI_BYTE, counts + i));
}

static void get_counts(const MPI_Status ss[NFRAGS],      hBags *b) {
    int cap, i, c, cc[NFRAGS];
    get_counts_bytes(ss,      cc);

    for (i = 0; i < NFRAGS; ++i) {
        c = cc[i] / b->bsize;
        b->counts[i] = c;
        cap = b->capacity[i];
        if (c > cap) UC(fail_over(i, c, cap));
    }
}

static void get_counts(const MPI_Status ss[NFRAGS],      CommBuffer *b) {
    int i, cc[NFRAGS];
    size_t cap, c;
    get_counts_bytes(ss,      cc);

    for (i = 0; i < NFRAGS; ++i) {
        c = cc[i];
        cap = b->cap[i];
        b->sz[i] = c;
        if (c > cap) UC(fail_over(i, c, cap));
    }
}

static void fail_wait_normal(int code) {
    int sz;
    char msg[BUFSIZ];
    m::Error_string(code, msg, &sz);
    ERR(msg);
}

static void fail_wait_status(int n, MPI_Status *ss) {

    enum {X, Y, Z};
    int i, sz, code, d[3];
    char msg[BUFSIZ];
    for (i = 0; i < n; i++) {
        code = m::status2errcode(&ss[i]);
        if (m::is_success(code) || m::is_pending(code)) continue;
        frag_hst::i2d3(i, d);
        m::Error_string(code, msg, &sz);
        ERR("mpi error in fragment %d = [%d %d %d], %s", i,
            d[X], d[Y], d[Z], msg);
    }
    ERR("assert");
}

static void fail_wait(int code, int n, MPI_Status *ss) {
    if (m::is_err_in_status(code)) UC(fail_wait_status(n, ss));
    else  UC(fail_wait_normal(code));
}

void comm_wait_recv(Comm *com,      hBags *b) {
    int errorcode;
    MPI_Status ss[NFRAGS];
    errorcode = m::Waitall(NFRAGS, com->rreq,      ss);
    if (!m::is_success(errorcode)) fail_wait(errorcode, NFRAGS, ss);
    get_counts(ss,      b);
}

void comm_post_recv(CommBuffer *cb, Comm *com) {
    int i, tag;
    for (i = 0; i < NFRAGS; ++i) {
        tag = com->tags[i];
        MC(m::Irecv(cb->buf[i], cb->cap[i], MPI_BYTE, com->ranks[i], tag, com->cart, com->rreq + i));
    }
}

void comm_post_send(const CommBuffer *cb, Comm *com) {
    int i, tag;
    long c, cap;
    for (i = 0; i < NFRAGS; ++i) {
        c = cb->sz[i];
        cap = cb->cap[i];
        tag = i;
        if (c > cap) UC(fail_over(i, c, cap));
        MC(m::Isend(cb->buf[i], c, MPI_BYTE, com->ranks[i], tag, com->cart, com->sreq + i));
    }
}


void comm_wait_recv(Comm *com,      CommBuffer *cb) {
    int errorcode;
    MPI_Status ss[NFRAGS];
    errorcode = m::Waitall(NFRAGS, com->rreq,      ss);
    if (!m::is_success(errorcode)) fail_wait(errorcode, NFRAGS, ss);
    get_counts(ss,      cb);
}

void comm_wait_send(Comm *com) {
    MPI_Status ss[NFRAGS];
    MC(m::Waitall(NFRAGS, com->sreq, ss));
}



int comm_get_number_capacity(int i, const hBags *b) {
    return b->capacity[i];
}

size_t comm_get_byte_capacity(int i, const hBags *b) {
    return b->bsize * comm_get_number_capacity(i, b);
}





#include <mpi.h>
#include <stdio.h>







namespace m {
void ini(int *argc, char ***argv);
void get_dims(int *argc, char ***argv, int dims[3]);
void get_cart(MPI_Comm comm, const int dims[3],      MPI_Comm *cart);
void fin();
}
struct TimeLine;

void time_line_ini(float start,      TimeLine**);
void time_line_fin(TimeLine*);

void time_line_advance(float dt, TimeLine*);
int  time_line_cross(const TimeLine*, float interval);

void time_line_strt_dump(MPI_Comm, const char *base, int id, const TimeLine*);
void time_line_strt_read(const char *base, int id, TimeLine*);

float time_line_get_current  (const TimeLine*);
long  time_line_get_iteration(const TimeLine*);
struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);
struct Sim;
struct Config;

void sim_ini(const Config*, MPI_Comm,      Sim**);
void sim_fin(Sim*);

void sim_gen(Sim*);
void sim_strt(Sim*);

int main(int argc, char **argv) {
    Sim *sim;
    Config *cfg;
    int rank, size, dims[3];
    MPI_Comm cart;
    int restart;

    m::ini(&argc, &argv);

    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));
    msg_ini(rank);
    msg_print("mpi rank/size: %d/%d", rank, size);
    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));

    UC(sim_ini(cfg, cart, &sim));
    UC(conf_lookup_bool(cfg, "glb.restart", &restart));
    msg_print("read restart: %s", restart ? "YES" : "NO" );
    if (restart) UC(sim_strt(sim));
    else         UC(sim_gen(sim));
    UC(sim_fin(sim));

    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
    msg_print("end");
}









#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector_types.h>







struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);

enum {
    EXE,
    ARG,
    OPT,
    DEF,
    NCFG
};

struct Config {
    config_t c[NCFG];
    config_t *r;
};

enum {
    OK,
    NOTFOUND,
    WTYPE
};

enum {
    MAX_LEVEL = 10
};

struct CBuf {
    char c[MAX_LEVEL][FILENAME_MAX];
};
static void set_write_options(config_t *c) {
    config_set_option(c, CONFIG_OPTION_SEMICOLON_SEPARATORS,            CONFIG_FALSE);
    config_set_option(c, CONFIG_OPTION_COLON_ASSIGNMENT_FOR_GROUPS,     CONFIG_FALSE);
    config_set_option(c, CONFIG_OPTION_COLON_ASSIGNMENT_FOR_NON_GROUPS, CONFIG_FALSE);
    config_set_option(c, CONFIG_OPTION_OPEN_BRACE_ON_SEPARATE_LINE,     CONFIG_FALSE);
    config_set_option(c, CONFIG_OPTION_ALLOW_SCIENTIFIC_NOTATION,       CONFIG_TRUE );
}

void conf_ini(     Config **pq) {
    Config *q;
    EMALLOC(1, &q);
    for (int i = 0; i < NCFG; ++i) {
        config_init(q->c + i);
        set_write_options(q->c + i);
    }

    EMALLOC(1, &q->r);
    config_init(q->r);
    set_write_options(q->r);
    *pq = q;
}

void conf_fin(     Config *q) {
    int i;
    for (i = 0; i < NCFG; ++i)
        config_destroy(&q->c[i]);
    config_destroy(q->r);
    EFREE(q->r);
    EFREE(q);
}

static void concatenate(int n, char **ss,      char *a) {
    char *s;
    a[0] = '\0';

    for(int i = 0; i < n; ++i) {
        s = ss[i];
        strcat(a, s);
        strcat(a, " ");
    }
}

static void read_file(const char *fname,      config_t *c) {
    msg_print("read '%s'", fname);
    if (!config_read_file(c, fname))
        ERR( "%s:%d - %s\n", config_error_file(c),
             config_error_line(c), config_error_text(c));
}

static void read_args(int argc, char **argv,      config_t *c) {
   enum {MAX_CHAR = 100000};
   char *args;

   EMALLOC(MAX_CHAR, &args);

   concatenate(argc, argv,      args);
   if (!config_read_string(c, args)) {
       msg_print("read args: %s", args);
       ERR("%d - %s\n",
           config_error_line(c), config_error_text(c));
   }

   EFREE(args);
}

static void shift(int *c, char ***v) {
    (*c) --;
    (*v) ++;
}

static int get_opt_file(int *argc, char ***argv,      char fname[]) {
    char *lastpnt, *a;
    int differ;

    if (*argc) {
        a = (*argv)[0];
        lastpnt = strrchr(a, '.');

        if (lastpnt != NULL) {
            differ = strcmp(lastpnt, ".cfg");
            if (differ) return 0;
            strcpy(fname, a);
            shift(argc, argv);
            return 1;
        }
    }
    return 0;
}

static void set_include_dir(const char *path, Config *cfg) {
    for (int i = 0; i < NCFG; ++i)
        config_set_include_dir(&cfg->c[i], path);
}

void conf_read(int argc, char **argv,      Config *cfg) {
    char *home, defname[1024] = {0}, optname[1024], definclude[1024] = {0};
    home = getenv("HOME");


    strcpy(definclude, home);
    strcat(definclude, "/.udx/");
    set_include_dir(definclude, cfg);


    strcpy(defname, home);
    strcat(defname, "/.udx/default.cfg");

    UC(read_file(defname,      &cfg->c[DEF]));

    if (get_opt_file(&argc, &argv,      optname)) {
        UC(read_file(optname,      &cfg->c[OPT]));
    }

    if (argc)
        UC(read_args(argc, argv,      &cfg->c[ARG]));
}

void conf_write_exe(const Config *cfg, FILE *stream) {
    config_write(&cfg->c[EXE], stream);
}

void conf_write_history(const Config *cfg, FILE *stream) {
    config_write(cfg->r, stream);
}
static config_setting_t* subsetting(const char *desc, int type, config_setting_t *group) {
    config_setting_t *sub;
    sub = config_setting_lookup(group, desc);
    if (NULL == sub)
        sub = config_setting_add(group, desc, type);
    return sub;
}

static config_setting_t* get_subgroup_setting(int n, const char *desc[], config_t *c) {
    config_setting_t *group;
    group = config_root_setting(c);

    for (int i = 0; i < n; ++i)
        group = subsetting(desc[i], CONFIG_TYPE_GROUP, group);
    return group;
}

static bool is_end(char c) {return c == '\0';}
static bool is_sep(char c) {return c == '.' || c == '/';}

static void cpy(const char *in, char *out) {
    while ( !is_end(*in) && !is_sep(*in)) {
        *out = *in;
        ++in; ++out;
    }
    *out = '\0';
}

static void split_str(const char *in, int *n, CBuf *out) {
    int i = 0;
    cpy(in, out->c[i++]);

    while ( !is_end(*in) ) {
        if ( is_sep(*in) ) cpy(++in, out->c[i++]);
        if (i > MAX_LEVEL) ERR("Too many levels in desc <%s> : found %d/%d\n", in, i, MAX_LEVEL);
        ++in;
    }
    *n = i;
}

static void to_ptr_array(const CBuf *buf, const char *ptr[]) {
    int i;
    for (i = 0; i < MAX_LEVEL; ++i) ptr[i] = buf->c[i];
}

static void set_int(const char *name, int a, config_t *c) {
    int n;
    CBuf buf;
    const char *desc[MAX_LEVEL];
    config_setting_t *group, *setting;
    int status;

    UC(split_str(name, &n, &buf));
    to_ptr_array(&buf, desc);

    group = get_subgroup_setting(n-1, desc,      c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_INT,      group);

    status = config_setting_set_int(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

static void set_vint(const char *name, int nelem, const int a[], config_t *c) {
    enum {APPEND = -1};
    int n;
    CBuf buf;
    const char *desc[MAX_LEVEL];
    config_setting_t *group, *setting, *status;
    int i;

    UC(split_str(name, &n, &buf));
    to_ptr_array(&buf, desc);

    group = get_subgroup_setting(n-1, desc,      c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_ARRAY,      group);

    for (i = 0; i < nelem; ++i) {
        status = config_setting_set_int_elem(setting, APPEND, a[i]);
        if (NULL == status)
            ERR("could not set element %d/%d of <%s>", i, nelem, desc[n-1]);
    }
}

static void set_int3(const char *desc, int3 a, config_t *c) {
    const int a3[] = {a.x, a.y, a.z};
    UC(set_vint(desc, 3, a3, c));
}

static void set_float(const char *name, float a, config_t *c) {
    int n;
    CBuf buf;
    const char *desc[MAX_LEVEL];
    config_setting_t *group, *setting;
    int status;

    UC(split_str(name, &n, &buf));
    to_ptr_array(&buf, desc);

    group = get_subgroup_setting(n-1, desc,      c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_FLOAT,      group);

    status = config_setting_set_float(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

static void set_vfloat(const char *name, int nelem, const float a[], config_t *c) {
    enum {APPEND = -1};
    int n;
    CBuf buf;
    const char *desc[MAX_LEVEL];
    config_setting_t *group, *setting, *status;
    int i;

    UC(split_str(name, &n, &buf));
    to_ptr_array(&buf, desc);

    group = get_subgroup_setting(n-1, desc,      c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_ARRAY,      group);

    for (i = 0; i < nelem; ++i) {
        status = config_setting_set_float_elem(setting, APPEND, a[i]);
        if (NULL == status)
            ERR("could not set element %d/%d of <%s>", i, nelem, desc[n-1]);
    }
}

static void set_float3(const char *name, float3 a, config_t *c) {
    const float a3[] = {a.x, a.y, a.z};
    UC(set_vfloat(name, 3, a3, c));
}

static void set_bool(const char *name, int a, config_t *c) {
    int n;
    CBuf buf;
    const char *desc[MAX_LEVEL];
    config_setting_t *group, *setting;
    int status;
    UC(split_str(name, &n, &buf));
    to_ptr_array(&buf, desc);

    group = get_subgroup_setting(n-1, desc,      c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_BOOL,      group);

    status = config_setting_set_bool(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

static void set_string(const char *name, const char *a, config_t *c) {
    int n;
    CBuf buf;
    const char *desc[MAX_LEVEL];
    config_setting_t *group, *setting;
    int status;
    UC(split_str(name, &n, &buf));
    to_ptr_array(&buf, desc);

    group = get_subgroup_setting(n-1, desc,      c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_STRING,      group);

    status = config_setting_set_string(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

static void set_vstring(const char *name, int nelem, const char *ss[], config_t *c) {
    enum {APPEND = -1};
    int n;
    CBuf buf;
    const char *desc[MAX_LEVEL];
    config_setting_t *group, *setting, *status;
    int i;

    UC(split_str(name, &n, &buf));
    to_ptr_array(&buf, desc);

    group = get_subgroup_setting(n-1, desc,      c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_ARRAY,      group);

    for (i = 0; i < nelem; ++i) {
        status = config_setting_set_string_elem(setting, APPEND, ss[i]);
        if (NULL == status)
            ERR("could not set element %d/%d of <%s>", i, nelem, desc[n-1]);
    }
}


void conf_set_int(const char *desc, int a, Config *cfg) {
    UC(set_int(desc, a, &cfg->c[EXE]));
}

void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg) {
    UC(set_vint(desc, nelem, a, &cfg->c[EXE]));
}

void conf_set_int3(const char *desc, int3 a, Config *cfg) {
    UC(set_int3(desc, a, &cfg->c[EXE]));
}

void conf_set_float(const char *desc, float a, Config *cfg) {
    UC(set_float(desc, a, &cfg->c[EXE]));
}

void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg) {
    UC(set_vfloat(desc, nelem, a, &cfg->c[EXE]));
}

void conf_set_float3(const char *desc, float3 a, Config *cfg) {
    UC(set_float3(desc, a, &cfg->c[EXE]));
}

void conf_set_bool(const char *desc, int a, Config *cfg) {
    UC(set_bool(desc, a, &cfg->c[EXE]));
}

void conf_set_string(const char *desc, const char *a, Config *cfg) {
    UC(set_string(desc, a, &cfg->c[EXE]));
}

static bool found(const config_setting_t *s) {return s != NULL;}
static void treat_status(int status, const char *desc, const char *type) {
    if (status == NOTFOUND)
        ERR("Could not find <%s>\n", desc);
    else if (status == WTYPE)
        ERR("<%s> has wrong type: expected %s\n", desc, type);
}

static int lookup_setting(const Config *c, const char *desc, config_setting_t **s) {
    int i;
    for (i = 0; i < NCFG; ++i) {
        *s = config_lookup(c->c + i, desc);
        if ( found(*s) ) return OK;
    }
    return NOTFOUND;
}

static int lookup_int(const Config *c, const char *desc, int *a) {
    config_setting_t *s;
    int status;
    status = lookup_setting(c, desc,      &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_INT) return WTYPE;
    *a = config_setting_get_int(s);
    set_int(desc, *a, c->r);
    return status;
}

static int lookup_float(const Config *c, const char *desc, float *a) {
    config_setting_t *s;
    int status;
    status = lookup_setting(c, desc,      &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_FLOAT) return WTYPE;
    *a = config_setting_get_float(s);
    set_float(desc, *a, c->r);
    return status;
}

static int lookup_bool(const Config *c, const char *desc, int *a) {
    config_setting_t *s;
    int status;
    status = lookup_setting(c, desc,      &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_BOOL) return WTYPE;
    *a = config_setting_get_bool(s);
    set_bool(desc, *a, c->r);
    return status;
}

static int lookup_string(const Config *c, const char *desc, const char **a) {
    config_setting_t *s;
    int status;
    status = lookup_setting(c, desc,      &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_STRING) return WTYPE;
    *a = config_setting_get_string(s);
    set_string(desc, *a, c->r);
    return status;
}

static int lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]) {
    config_setting_t *s, *e;
    int j, status;
    status = lookup_setting(c, desc,      &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_ARRAY) return WTYPE;
    *n = config_setting_length(s);
    if (*n > maxn) ERR("%s: too many components: %d / %d", desc, *n, maxn);
    for (j = 0; j < *n; ++j) {
        e = config_setting_get_elem(s, j);
        if (config_setting_type(e) != CONFIG_TYPE_INT) return WTYPE;
        a[j] = config_setting_get_int(e);
    }
    set_vint(desc, *n, a, c->r);
    return status;
}

static int lookup_int3(const Config *c, const char *desc, int3 *a) {
    enum {X, Y, Z};
    int status, n = 0, f[3];
    status = lookup_vint(c, desc, 3, &n, f);
    if (status != OK) return status;
    if (n != 3)
        ERR("fail to read `%s`: int3 must have 3 components, found %d", desc, n);
    a->x = f[X];
    a->y = f[Y];
    a->z = f[Z];
    return status;
}

static int lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]) {
    config_setting_t *s, *e;
    int j, status;
    status = lookup_setting(c, desc,      &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_ARRAY) return WTYPE;
    *n = config_setting_length(s);

    if (*n > maxn) ERR("%s: too many components: %d / %d", desc, *n, maxn);
    for (j = 0; j < *n; ++j) {
        e = config_setting_get_elem(s, j);
        if (config_setting_type(e) != CONFIG_TYPE_FLOAT) return WTYPE;
        a[j] = config_setting_get_float(e);
    }
    set_vfloat(desc, *n, a, c->r);
    return status;
}

static int lookup_float3(const Config *c, const char *desc, float3 *a) {
    enum {X, Y, Z};
    int n = 0, status;
    float f[3];
    status = lookup_vfloat(c, desc, 3, &n, f);
    if (status != OK) return status;
    if (n != 3)
        ERR("fail to read `%s`: float3 must have 3 components, found %d", desc, n);
    a->x = f[X];
    a->y = f[Y];
    a->z = f[Z];
    return status;
}

static int lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss) {
    config_setting_t *s, *e;
    int j, status;
    status = lookup_setting(c, desc,      &s);
    if (status != OK) return status;
    if (config_setting_type(s) != CONFIG_TYPE_ARRAY) return WTYPE;
    *n = config_setting_length(s);

    if (*n > maxn) ERR("%s: too many components: %d / %d", desc, *n, maxn);
    for (j = 0; j < *n; ++j) {
        e = config_setting_get_elem(s, j);
        if (config_setting_type(e) != CONFIG_TYPE_STRING) return WTYPE;
        ss[j] = config_setting_get_string(e);
    }
    set_vstring(desc, *n, ss, c->r);
    return status;
}

void conf_lookup_int(const Config *c, const char *desc, int *a) {
    int status = lookup_int(c, desc, a);
    UC(treat_status(status, desc, "int"));
}

void conf_lookup_float(const Config *c, const char *desc, float *a) {
    int status = lookup_float(c, desc, a);
    UC(treat_status(status, desc, "float"));
}

void conf_lookup_bool(const Config *c, const char *desc, int *a) {
    int status = lookup_bool(c, desc, a);
    UC(treat_status(status, desc, "bool"));
}

void conf_lookup_string(const Config *c, const char *desc, const char **a) {
    int status = lookup_string(c, desc, a);
    UC(treat_status(status, desc, "string"));
}

void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]) {
    int status = lookup_vint(c, desc, maxn, n, a);
    UC(treat_status(status, desc, "array of int"));
}

void conf_lookup_int3(const Config *c, const char *desc, int3 *a) {
    int status = lookup_int3(c, desc, a);
    UC(treat_status(status, desc, "array of int"));
}

void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]) {
    int status = lookup_vfloat(c, desc, maxn, n, a);
    UC(treat_status(status, desc, "array of float"));
}

void conf_lookup_float3(const Config *c, const char *desc, float3 *a) {
    int status = lookup_float3(c, desc, a);
    UC(treat_status(status, desc, "array of float"));
}

void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss) {
    int status = lookup_vstring(c, desc, maxn, n, ss);
    UC(treat_status(status, desc, "array of string"));
}


static void get_desc(const char *ns, const char *d, char *desc) {
    sprintf(desc, "%s.%s", ns, d);
}

void conf_lookup_int_ns(const Config *c, const char *ns, const char *d, int *a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_int(c, desc, a));
}

void conf_lookup_float_ns(const Config *c, const char *ns, const char *d, float *a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_float(c, desc, a));
}

void conf_lookup_bool_ns(const Config *c, const char *ns, const char *d, int *a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_bool(c, desc, a));
}

void conf_lookup_string_ns(const Config *c, const char *ns, const char *d, const char **a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_string(c, desc, a));
}

void conf_lookup_vint_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, int a[]) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_vint(c, desc, maxn, n, a));
}

void conf_lookup_int3_ns(const Config *c, const char *ns, const char *d, int3 *a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_int3(c, desc, a));
}

void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, float a[]) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_vfloat(c, desc, maxn, n, a));
}

void conf_lookup_float3_ns(const Config *c, const char *ns, const char *d, float3 *a) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_float3(c, desc, a));
}

void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss) {
    char desc[FILENAME_MAX];
    get_desc(ns, d, desc);
    UC(conf_lookup_vstring(c, desc, maxn, n, ss));
}



bool conf_opt_int(const Config *c, const char *desc, int *a) {
    return OK == lookup_int(c, desc, a);
}

bool conf_opt_float(const Config *c, const char *desc, float *a)  {
    return OK == lookup_float(c, desc, a);
}

bool conf_opt_bool(const Config *c, const char *desc, int *a)  {
    return OK == lookup_bool(c, desc, a);
}

bool conf_opt_string(const Config *c, const char *desc, const char **a)  {
    return OK == lookup_string(c, desc, a);
}

bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]) {
    return OK == lookup_vint(c, desc, maxn, n, a);
}

bool conf_opt_int3(const Config *c, const char *desc, int3 *a) {
    return OK == lookup_int3(c, desc, a);
}

bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]) {
    return OK == lookup_vfloat(c, desc, maxn, n, a);
}

bool conf_opt_float3(const Config *c, const char *desc, float3 *a) {
    return OK == lookup_float3(c, desc, a);
}

bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss) {
    return OK == lookup_vstring(c, desc, maxn, n, ss);
}





































#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdlib.h>
#include <stdio.h>
#include <vector_types.h>

#ifndef UDX_CONTROL_COORDS_COMMON
#define UDX_CONTROL_COORDS_COMMON
struct Coords_v {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};
#endif
struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);
#ifndef UDX_CONTROL_UTILS_COMMON
#define UDX_CONTROL_UTILS_COMMON




#endif
#ifndef UDX_CONTROL_ERROR_COMMON
#define UDX_CONTROL_ERROR_COMMON

#endif

struct Inflow;
struct Config;
struct Particle;
struct int2;
struct float3;
struct Coords;

void inflow_ini(int2 nc, Inflow **i);
void inflow_fin(Inflow *i);

void inflow_ini_velocity(Inflow *i);

void inflow_ini_params_plate(const Coords *c, float3 o, int dir, float L1, float L2,
                             float3 u, bool upoiseuille, bool vpoiseuille,
                                  Inflow *i);
void inflow_ini_params_circle(const Coords *c, float3 o, float R, float H, float u, bool poiseuille,
                                   Inflow *i);

void inflow_ini_params_conf(const Coords *coords, const Config *cfg,      Inflow *i);

void inflow_create_pp(float kBT, int numdensity, float dt, Inflow *i, int *n, Particle *pp);
void inflow_create_pp_cc(float kBT, int numdensity, float dt, int newcolor, Inflow *i, int *n, Particle *pp, int *cc);

void inflow_ini_params_conf(const Coords *coords, const Config *cfg,      Inflow *i) {
    const char *type;
    UC(conf_lookup_string(cfg, "outflow.type", &type));

    if      (same_str(type, "circle")) {
        float R, H, U;
        int poiseuille;
        float3 center;

        UC(conf_lookup_float(cfg, "inflow.R", &R));
        UC(conf_lookup_float(cfg, "inflow.H", &H));
        UC(conf_lookup_float(cfg, "inflow.U", &U));
        UC(conf_lookup_bool(cfg, "inflow.poiseuille", &poiseuille));
        UC(conf_lookup_float3(cfg, "inflow.center", &center));
        UC(inflow_ini_params_circle(coords, center, R, H, U, poiseuille,      i));
    }
    else if (same_str(type, "plate")) {
        int upois, vpois, dir;
        float3 origin, u;
        float L1, L2;

        UC(conf_lookup_float(cfg, "inflow.L1", &L1));
        UC(conf_lookup_float(cfg, "inflow.L2", &L2));
        UC(conf_lookup_int(cfg, "inflow.direction", &dir));
        UC(conf_lookup_bool(cfg, "inflow.upoiseuille", &upois));
        UC(conf_lookup_bool(cfg, "inflow.vpoiseuille", &vpois));
        UC(conf_lookup_float3(cfg, "inflow.origin", &origin));
        UC(conf_lookup_float3(cfg, "inflow.u", &u));

        UC(inflow_ini_params_plate(coords, origin, dir, L1, L2, u, upois, vpois,      i));
    }
    else {
        ERR("unknown inflow type <%s>", type);
    }
}




#include <stdio.h>
#include <curand_kernel.h>




#ifndef UDX_CONTROL_ERROR_COMMON
#define UDX_CONTROL_ERROR_COMMON

#endif








#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
#endif
#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif


#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())









#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif


struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);
#ifndef UDX_CONTROL_MATH_COMMON
#define UDX_CONTROL_MATH_COMMON


#define _HD_ __host__ __device__

template <typename T1, typename T2>
_HD_ void scal(const T1 a,        T2 *b) {
    b->x *= a;
    b->y *= a;
    b->z *= a;
}

template <typename T1, typename T2>
_HD_ void add(const T1 *a,        T2 *b) {
    b->x += a->x;
    b->y += a->y;
    b->z += a->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void diff(const T1 *a, const T2 *b,      T3 *c) {
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ T1 dot(const T2 *a, const T3 *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void cross(const T1 *a, const T2 *b,      T3 *c) {
    c->x = a->y * b->z - a->z * b->y;
    c->y = a->z * b->x - a->x * b->z;
    c->z = a->x * b->y - a->y * b->x;
}

template <typename T1, typename T2, typename T3>
_HD_ void scalmult(const T1 *a, const T2 x,      const T3 *b) {
    b->x = a->x * x;
    b->y = a->y * x;
    b->z = a->z * x;
}


template <typename T1, typename T2, typename T3, typename T4>
_HD_ void apxb(const T1 *a, const T2 x, const T3 *b,      T4 *c) {
    c->x = a->x + x * b->x;
    c->y = a->y + x * b->y;
    c->z = a->z + x * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void axpy(const T1 a, const T2 *x,        T3 *y) {
    y->x += a * x->x;
    y->y += a * x->y;
    y->z += a * x->z;
}
#endif



struct ParamsPlate {
    float3 o, a, b;
};

struct VParamsPlate {
    float3 u;
    bool upoiseuille, vpoiseuille;
};
static __device__ void coords2pos(ParamsPlate p, float2 xi,      float3 *r) {
    *r = p.o;
    axpy(xi.x, &p.a,      r);
    axpy(xi.y, &p.b,      r);
}

static __device__ float3 get_normal(ParamsPlate p, int2 nc, int i, int j) {
    float3 n;
    cross(&p.a, &p.b,      &n);
    scal(1.f / (nc.x * nc.y),      &n);
    return n;
}

static __device__ void coords2vel(VParamsPlate vp, ParamsPlate p, float2 xi,      float3 *u) {
    float fact;
    fact = 1.f;
    if (vp.upoiseuille)
        fact *= 4 * xi.x * (1 - xi.x);

    if (vp.vpoiseuille)
        fact *= 4 * xi.y * (1 - xi.y);

    *u = vp.u;
    scal(fact,      u);
}

struct ParamsCircle {
    float3 o;
    float R, H;
};

struct VParamsCircle {
    float u;
    bool poiseuille;
};
static __device__ void coords2pos(ParamsCircle p, float2 xi,      float3 *r) {
    float th, h, cth, sth;
    th = xi.x * 2 * M_PI;
    h = xi.y * p.H;
    cth = cos(th);
    sth = sin(th);

    *r = p.o;
    r->x += p.R * cth;
    r->y += p.R * sth;
    r->z += h;
}

static __device__ float3 get_normal(ParamsCircle p, int2 nc, int i, int j) {
    float3 n;
    float th, cth, sth, area;
    th = (i + 0.5f) / (nc.x) * 2 * M_PI;
    cth = cos(th);
    sth = sin(th);

    n = make_float3(cth, sth, 0);
    area = p.H * p.R * 2 * M_PI / (nc.x * nc.y);
    scal(area,      &n);
    return n;
}

static __device__ void coords2vel(VParamsCircle vp, ParamsCircle p, float2 xi,      float3 *u) {
    float th, cth, sth, fact;
    th = xi.x * 2 * M_PI;
    cth = cos(th);
    sth = sin(th);

    fact = 1.f;
    if (vp.poiseuille) {
        float y = xi.y;
        fact *= 4 * y * (1 - y);
    }

    *u = make_float3(cth, sth, 0);
    scal(fact * vp.u,      u);
}


enum Type {
    TYPE_NONE,
    TYPE_PLATE,
    TYPE_CIRCLE,
};

struct Desc {
    curandState_t *rnds;
    float3 *uu;
    float *cumflux;
    int *ndev;
    int2 nc;
};

union ParamsU {
    ParamsPlate plate;
    ParamsCircle circle;
};

union VParamsU {
    VParamsPlate plate;
    VParamsCircle circle;
};

struct Inflow {
    Desc d;
    Type t;
    ParamsU p;
    VParamsU vp;
};

struct SolventWrap {
    Particle *pp;
    int *cc;
    bool multisolvent;
    int color;
};

__global__ void inflow_ini_rnd_kernel(long seed, int n, curandState_t *rr) {
    long i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &rr[i]);
}

__global__ void inflow_ini_flux_kernel(int n,      curandState_t *rr, float *cumflux) {
    long i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    cumflux[i] = curand_uniform(rr + i);
}
template <typename Par, typename VPar>
__global__ void inflow_ini_vel_kernel(VPar vparams, Par params, int2 nc,      float3 *uu) {
    int i, xcid, ycid;
    float3 u;
    float2 xi;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= nc.x * nc.y) return;

    xcid = i % nc.x;
    ycid = i / nc.x;

    xi.x = (xcid + 0.5f) / nc.x;
    xi.y = (ycid + 0.5f) / nc.y;

    coords2vel(vparams, params, xi,      &u);

    uu[i] = u;
}

template <typename Par>
__global__ void inflow_cumulative_flux_kernel(float nxdt, Par params, int2 nc, const float3 *uu,      float *cumflux) {
    int i, xcid, ycid;
    float dn;
    float3 normal, u;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= nc.x * nc.y) return;

    xcid = i % nc.x;
    ycid = i / nc.x;


    normal = get_normal(params, nc, xcid, ycid);
    u      = uu[i];
    dn = nxdt * dot<float>(&normal, &u);

    cumflux[i] += dn;
}

template <typename Par>
static __device__ Particle create_particle(float kBT, Par params, int2 nc,
                                           int xcid, int ycid, float3 u, curandState_t *rg) {
    float2 xi;
    float3 r;
    float sigma;
    xi.x = (xcid + curand_uniform(rg)) / nc.x;
    xi.y = (ycid + curand_uniform(rg)) / nc.y;

    coords2pos(params, xi,      &r);

    sigma = sqrtf(kBT);
    u.x += curand_normal(rg) * sigma;
    u.y += curand_normal(rg) * sigma;
    u.z += curand_normal(rg) * sigma;



    return Particle({r.x, r.y, r.z, u.x, u.y, u.z});
}

template <typename Par>
__global__ void inflow_create_particles_kernel(float kBT, Par params, int2 nc, const float3 *flux,
                                        curandState_t *rnds, float *cumflux,      int *n, SolventWrap wrap) {
    int i, xcid, ycid, j, nnew, strt;
    float c;
    float3 f;
    curandState_t rndstate;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    xcid = i % nc.x;
    ycid = i / nc.x;

    if (i >= nc.x * nc.y) return;

    c = cumflux[i];
    nnew = int(c);

    if (nnew == 0) return;

    c -= nnew;
    cumflux[i] = c;

    f        = flux[i];
    rndstate = rnds[i];

    strt = atomicAdd(n, nnew);

    for (j = strt; j < strt + nnew; ++j) {
        wrap.pp[j] = create_particle(kBT, params, nc, xcid, ycid, f,        &rndstate);
        if (wrap.multisolvent)
            wrap.cc[j] = wrap.color;
    }

    rnds[i] = rndstate;
}

static float crop(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static float3 crop(float3 r, float3 lo, float3 hi) {
    r.x = crop(r.x, lo.x, hi.x);
    r.y = crop(r.y, lo.y, hi.y);
    r.z = crop(r.z, lo.z, hi.z);
    return r;
}

void inflow_ini_params_plate(const Coords *c, float3 o, int dir, float L1, float L2,
                             float3 u, bool upoiseuille, bool vpoiseuille,
                                  Inflow *i) {
    enum {X, Y, Z};
    ParamsPlate *pp;
    VParamsPlate *vpp;
    float3 a, b, lo, hi;
    float3 co, ca, cb;

    lo = make_float3(-xs(c)/2, -ys(c)/2, -zs(c)/2);
    hi = make_float3( xs(c)/2,  ys(c)/2,  zs(c)/2);

    pp = &i->p.plate;
    vpp = &i->vp.plate;

    a = b = make_float3(0, 0, 0);

    switch(dir) {
    case X:
        a.y = L1;
        b.z = L2;
        break;
    case Y:
        a.x = L1;
        b.z = L2;
        break;
    case Z:
        a.x = L1;
        b.y = L2;
        break;
    default:
        ERR("wrong direction");
        break;
    };


    global2local(c, o,      &o);

    co = ca = cb = o;
    add(&a,      &ca);
    add(&b,      &cb);

    co = crop(co, lo, hi);
    ca = crop(ca, lo, hi);
    cb = crop(cb, lo, hi);

    pp->o = co;
    diff(&ca, &co,      &pp->a);
    diff(&cb, &co,      &pp->b);

    vpp->u = u;
    vpp->upoiseuille = upoiseuille;
    vpp->vpoiseuille = vpoiseuille;

    i->t = TYPE_PLATE;
}
static bool is_inside_subdomain(const Coords *c, float3 r) {
    bool inside = true;
    inside &= (xlo(c) <= r.x) && (r.x < xhi(c));
    inside &= (ylo(c) <= r.y) && (r.y < yhi(c));
    inside &= (zlo(c) <= r.z) && (r.z < zhi(c));
    return inside;
}

void inflow_ini_params_circle(const Coords *c, float3 o, float R, float H, float u, bool poiseuille,
                                   Inflow *i) {
    enum {X, Y, Z};
    ParamsCircle *pp;
    VParamsCircle *vpp;

    if (is_inside_subdomain(c, o)) {
        pp  = &i->p.circle;
        vpp = &i->vp.circle;

        global2local(c, o,      &o);

        pp->o = o;
        pp->R = R;
        pp->H = H;

        if (R > xs(c)/2 || R > ys(c)/2 || H > zs(c))
            ERR("dimensions should be smaller");

        vpp->u = u;
        vpp->poiseuille = poiseuille;

        i->t = TYPE_CIRCLE;
    } else {
        i->t = TYPE_NONE;
    }
}

static void ini_rnd(int n, curandState_t *rr) {
    long seed = 1234567;
    KL(inflow_ini_rnd_kernel, (k_cnf(n)), (seed, n, rr));
}

static void ini_flux(int n, curandState_t *rr, float *cumflux) {
    KL(inflow_ini_flux_kernel, (k_cnf(n)), (n, rr, cumflux));
}

void inflow_ini(int2 nc, Inflow **i) {
    int n;
    Inflow *ip;
    Desc *d;

    EMALLOC(1, i);

    ip = *i;
    d = &ip->d;

    n = nc.x * nc.y;
    d->nc = nc;

    Dalloc(&d->rnds,    n);
    Dalloc(&d->uu,      n);
    Dalloc(&d->cumflux, n);
    Dalloc(&d->ndev,    1);

    UC(ini_rnd(n, d->rnds));
    UC(ini_flux(n, d->rnds, d->cumflux));

    ip->t = TYPE_NONE;
}

static void ini_velocity(Type t, int2 nc, const ParamsU *p, const VParamsU *vp,      float3 *uu) {
    int n = nc.x * nc.y;
    switch(t) {
    case TYPE_PLATE:
        KL(inflow_ini_vel_kernel, (k_cnf(n)), (vp->plate, p->plate, nc,      uu));
        break;
    case TYPE_CIRCLE:
        KL(inflow_ini_vel_kernel, (k_cnf(n)), (vp->circle, p->circle, nc,      uu));
        break;
    case TYPE_NONE:
        break;
    default:
        ERR("No inflow type is set");
        break;
    };
}

void inflow_ini_velocity(Inflow *i) {
    ini_velocity(i->t, i->d.nc, &i->p, &i->vp,      i->d.uu);
}

void inflow_fin(Inflow *i) {
    Desc *d = &i->d;
    Dfree(d->rnds);
    Dfree(d->uu);
    Dfree(d->cumflux);
    Dfree(d->ndev);
    EFREE(i);
}

static void create_solvent(float kBT, int numdensity, float dt, Inflow *i, int *n, SolventWrap wrap) {
    int2 nc;
    Desc *d;
    int nctot;
    float nxdt;

    d = &i->d;
    nc = d->nc;
    nctot = nc.x * nc.y;
    nxdt = numdensity * dt;

    CC(d::MemcpyAsync(d->ndev, n, sizeof(int), H2D));

    switch(i->t) {
    case TYPE_PLATE:
        KL(inflow_cumulative_flux_kernel, (k_cnf(nctot)), (nxdt, i->p.plate, nc, d->uu,      d->cumflux));
        KL(inflow_create_particles_kernel, (k_cnf(nctot)),
           (kBT, i->p.plate, nc, d->uu,        d->rnds, d->cumflux,      d->ndev, wrap));
        break;
    case TYPE_CIRCLE:
        KL(inflow_cumulative_flux_kernel, (k_cnf(nctot)), (nxdt, i->p.circle, nc, d->uu,      d->cumflux));
        KL(inflow_create_particles_kernel, (k_cnf(nctot)),
           (kBT, i->p.circle, nc, d->uu,        d->rnds, d->cumflux,      d->ndev, wrap));
        break;
    case TYPE_NONE:
        break;
    default:
        ERR("No inflow type is set");
        break;
    };

    CC(d::MemcpyAsync(n, d->ndev, sizeof(int), D2H));
    dSync();
}

void inflow_create_pp(float kBT, int numdensity, float dt, Inflow *i, int *n, Particle *pp) {
    SolventWrap wrap;

    wrap.pp = pp;
    wrap.cc = NULL;
    wrap.multisolvent = false;
    wrap.color = -1;

    create_solvent(kBT, numdensity, dt, i, n, wrap);
}

void inflow_create_pp_cc(float kBT, int numdensity, float dt, int newcolor, Inflow *i, int *n, Particle *pp, int *cc) {
    SolventWrap wrap;

    wrap.pp = pp;
    wrap.cc = cc;
    wrap.multisolvent = true;
    wrap.color = newcolor;
    create_solvent(kBT, numdensity, dt, i, n, wrap);
}




#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>
#include <mpi.h>
#include <vector_types.h>

#ifndef UDX_CONTROL_UTILS_COMMON
#define UDX_CONTROL_UTILS_COMMON




#endif
#ifndef UDX_CONTROL_ERROR_COMMON
#define UDX_CONTROL_ERROR_COMMON

#endif
struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);

struct PidVCont;
struct Coords;
struct Config;
struct float3;
struct int3;
struct Particle;

void vcont_ini(MPI_Comm comm, int3 L,      PidVCont **c);
void vcont_fin(     PidVCont *cont);

void vcont_set_params(float factor, float Kp, float Ki, float Kd,      PidVCont *c);
void vcont_set_target(float3 vtarget,      PidVCont *c);
void vcont_set_cart(     PidVCont *c);
void vcont_set_radial(     PidVCont *c);

void vcont_set_conf(const Config *cfg,      PidVCont *c);

void vcont_strt_dump(MPI_Comm, const char *base, int id, const PidVCont*);
void vcont_strt_read(          const char *base, int id, PidVCont*);

void   vcont_sample(const Coords *coords, int n, const Particle *pp, const int *starts, const int *counts,      PidVCont *c);
float3 vcont_adjustF(     PidVCont *c);
void   vcont_log(const PidVCont *c);

void vcont_set_conf(const Config *cfg,      PidVCont *c) {
    const char *type;
    float3 U;
    float factor, Kp, Ki, Kd;

    UC(conf_lookup_string(cfg, "vcon.type", &type));
    UC(conf_lookup_float3(cfg, "vcon.U", &U));
    UC(conf_lookup_float(cfg, "vcon.factor", &factor));
    UC(conf_lookup_float(cfg, "vcon.Kp", &Kp));
    UC(conf_lookup_float(cfg, "vcon.Ki", &Ki));
    UC(conf_lookup_float(cfg, "vcon.Kd", &Kd));

    UC(vcont_set_params(factor, Kp, Ki, Kd,      c));
    UC(vcont_set_target(U,      c));

    if      (same_str(type, "cart"))
        UC(vcont_set_cart(     c));
    else if (same_str(type, "rad"))
        UC(vcont_set_radial(     c));
    else
        ERR("Unrecognised type <%s>", type);

}



#include <stdio.h>
#include <mpi.h>
#include <assert.h>




#ifndef UDX_CONTROL_ERROR_COMMON
#define UDX_CONTROL_ERROR_COMMON

#endif


#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())









#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif







#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
#endif



#ifndef UDX_CONTROL_MATH_COMMON
#define UDX_CONTROL_MATH_COMMON


#define _HD_ __host__ __device__

template <typename T1, typename T2>
_HD_ void scal(const T1 a,        T2 *b) {
    b->x *= a;
    b->y *= a;
    b->z *= a;
}

template <typename T1, typename T2>
_HD_ void add(const T1 *a,        T2 *b) {
    b->x += a->x;
    b->y += a->y;
    b->z += a->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void diff(const T1 *a, const T2 *b,      T3 *c) {
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ T1 dot(const T2 *a, const T3 *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void cross(const T1 *a, const T2 *b,      T3 *c) {
    c->x = a->y * b->z - a->z * b->y;
    c->y = a->z * b->x - a->x * b->z;
    c->z = a->x * b->y - a->y * b->x;
}

template <typename T1, typename T2, typename T3>
_HD_ void scalmult(const T1 *a, const T2 x,      const T3 *b) {
    b->x = a->x * x;
    b->y = a->y * x;
    b->z = a->z * x;
}


template <typename T1, typename T2, typename T3, typename T4>
_HD_ void apxb(const T1 *a, const T2 x, const T3 *b,      T4 *c) {
    c->x = a->x + x * b->x;
    c->y = a->y + x * b->y;
    c->z = a->z + x * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void axpy(const T1 a, const T2 *x,        T3 *y) {
    y->x += a * x->x;
    y->y += a * x->y;
    y->z += a * x->z;
}
#endif
#ifndef UDX_CONTROL_COORDS_COMMON
#define UDX_CONTROL_COORDS_COMMON
struct Coords_v {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};
#endif


static __device__ int xdomain(const Coords_v c) {
    return c.Lx * c.xd;
}

static __device__ int ydomain(const Coords_v c) {
    return c.Ly * c.yd;
}

static __device__ int zdomain(const Coords_v c) {
    return c.Lz * c.zd;
}



static __device__ float xl2xc(const Coords_v c, float xl) {
    return xl - 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
}

static __device__ float yl2yc(const Coords_v c, float yl) {
    return yl - 0.5f * c.Ly * (c.yd - 2 * c.yc - 1);
}

static __device__ float zl2zc(const Coords_v c, float zl) {
    return zl - 0.5f * c.Lz * (c.zd - 2 * c.zc - 1);
}

static __device__ void local2center(const Coords_v c, float3 rl,      float3 *rc) {
    rc->x = xl2xc(c, rl.x);
    rc->y = yl2yc(c, rl.y);
    rc->z = zl2zc(c, rl.z);
}



static __device__ float xc2xl(const Coords_v c, float xc) {
    return xc + 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
}

static __device__ float yc2yl(const Coords_v c, float yc) {
    return yc + 0.5f * c.Ly * (c.yd - 2.f * c.yc - 1);
}

static __device__ float zc2zl(const Coords_v c, float zc) {
    return zc + 0.5f * c.Lz * (c.zd - 2.f * c.zc - 1);
}

static __device__ void center2local(Coords_v c, float3 rc,      float3 *rl) {
    rl->x = xc2xl(c, rc.x);
    rl->y = yc2yl(c, rc.y);
    rl->z = zc2zl(c, rc.z);
}



static __device__ float xl2xg(const Coords_v c, float xl) {
    return (c.xc + 0.5f) * c.Lx + xl;
}

static __device__ float yl2yg(const Coords_v c, float yl) {
    return (c.yc + 0.5f) * c.Ly + yl;
}

static __device__ float zl2zg(const Coords_v c, float zl) {
    return (c.zc + 0.5f) * c.Lz + zl;
}

static __device__ void local2global(const Coords_v c, float3 rl,      float3 *rg) {
    rg->x = xl2xg(c, rl.x);
    rg->y = yl2yg(c, rl.y);
    rg->z = zl2zg(c, rl.z);
}



static __device__ float xg2xl(const Coords_v c, float xg) {
    return xg - (c.xc + 0.5f) * c.Lx;
}

static __device__ float yg2yl(const Coords_v c, float yg) {
    return yg - (c.yc + 0.5f) * c.Ly;
}

static __device__ float zg2zl(const Coords_v c, float zg) {
    return zg - (c.zc + 0.5f) * c.Lz;
}

static __device__ void global2local(const Coords_v c, float3 rg,      float3 *rl) {
    rl->x = xg2xl(c, rg.x);
    rl->y = yg2yl(c, rg.y);
    rl->z = zg2zl(c, rg.z);
}

struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);

#ifndef UDX_CONTROL_SHFL_COMMON
#define UDX_CONTROL_SHFL_COMMON
#define _I_ static __device__

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

#undef _I_
#define _S_ static __device__
#define _I_ static __device__

template <typename T>
_I_ T warpReduceSum(T v)
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += shfl_down(v, offset);
    return v;
}

_I_ float2 warpReduceSum(float2 val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
    }
    return val;
}

_I_ float3 warpReduceSum(float3 val)
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

#endif

struct Coords;
struct Particle;
struct Rigid;

enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};

typedef void (*StreamWriter)(const void*, FILE*);
typedef void (*StreamReader)(FILE*, void*);

void restart_write_pp(MPI_Comm, const char *base, const char *code, int id, long n, const Particle *pp);
void restart_write_ii(MPI_Comm, const char *base, const char *code, int id, long n, const int *ii);
void restart_write_ss(MPI_Comm, const char *base, const char *code, int id, long n, const Rigid *ss);
void restart_write_stream_one_node(MPI_Comm, const char *base, const char *code, int id, const void *data, StreamWriter);

void restart_read_pp(MPI_Comm, const char *base, const char *code, int id, int *n, Particle *pp);
void restart_read_ii(MPI_Comm, const char *base, const char *code, int id, int *n, int *ii);
void restart_read_ss(MPI_Comm, const char *base, const char *code, int id, int *n, Rigid *ss);
void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader, void *data);



enum {
    VCONT_TYPE_NONE,
    VCONT_TYPE_CART,
    VCONT_TYPE_RAD
};


struct TCart {};
struct TRad {};


union Trans {
    TCart cart;
    TRad  rad;
};


struct Sampler {
    int3 L;
    long nsamples;

    float3 *gridvel;
    int    *gridnum;
    float3 *totvel;
    int    *totnum;
    float3 *dtotvel;
    int    *dtotnum;
};

struct State {
    float3 cur;
    float3 olde, sume;
    float3 f;
};

struct Param {
    float3 target;
    float Kp, Ki, Kd;
};


struct PidVCont {
    Param param;
    State state;
    Sampler sampler;

    MPI_Comm comm;

    FILE *fdump;

    Trans trans;
    int type;
};

__global__ void vcont_reduce_by_warp_kernel(const float3 *gvel, const int *gnum, const uint ncells,      float3 *vel, int *num) {
    assert(blockDim.x == 32);
    int i, c;
    float3 v = make_float3(0, 0, 0);
    int n = 0;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    c = blockIdx.x;

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
static __device__ float3 transform(Coords_v, TCart, const Particle p) {
    enum {X, Y, Z};
    float3 u;
    u.x = p.v[X];
    u.y = p.v[Y];
    u.z = p.v[Z];
    return u;
}
static __device__ float3 transform(Coords_v c, TRad, const Particle p) {
    enum {X, Y, Z};
    float3 u;
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
static __device__ bool valid_cell(int3 L, int3 c) {
    return c.x < L.x && c.y < L.y && c.z < L.z;
}

static __device__ int get_cid(int3 L, int3 c) {
    return c.x + L.x * (c.y + L.y * c.z);
}

template<typename Trans>
__global__ void vcont_sample_kernel(Coords_v coords, Trans t, int3 L, const int *cellsstart, const int *cellscount, const Particle *pp,
                            float3 *gridv, int *gridn) {
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

#ifndef UDX_CONTROL_WARPSIZE_COMMON
#define UDX_CONTROL_WARPSIZE_COMMON
enum {WARPSIZE = 32};
#endif

static void ini_dump(int rank,      FILE **f) {
    *f = NULL;
    if (rank) return;
    UC(efopen(DUMP_BASE "/vcont.txt", "w",      f));
    fprintf(*f, "#vx vy vz fx fy fz\n");
    msg_print("Velocity controller: dump to " DUMP_BASE "/vcont.txt");
}

static void fin_dump(FILE *f) {
    if (f) UC(efclose(f));
}

static void reini_sampler(     Sampler *s) {
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

static void ini(MPI_Comm comm, int3 L,      PidVCont *c) {
    int rank;

    UC(ini_sampler(L, &c->sampler));
    UC(reini_sampler(&c->sampler));
    UC(ini_state(&c->state));

    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_dup(comm, &c->comm));

    UC(ini_dump(rank,      &c->fdump));

    c->type = VCONT_TYPE_NONE;
}

void vcont_ini(MPI_Comm comm, int3 L,      PidVCont **c) {
    PidVCont *vc;
    EMALLOC(1, c);
    vc = *c;
    UC(ini(comm, L,      vc));
}

static void fin_sampler(Sampler *s) {
    Dfree(s->gridvel);
    Dfree(s->gridnum);
    CC(d::FreeHost(s->totvel));
    CC(d::FreeHost(s->totnum));
}

void vcont_fin(     PidVCont *c) {
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

void vcont_set_params(float factor, float Kp, float Ki, float Kd,      PidVCont *c) {
    set_params(factor, Kp, Ki, Kd, &c->param);
}

void vcont_set_target(float3 vtarget,      PidVCont *c) {
    c->param.target = vtarget;
    c->state.olde   = vtarget;
}

void vcont_set_cart(     PidVCont *cont) {
    cont->type = VCONT_TYPE_CART;
}

void vcont_set_radial(     PidVCont *cont) {
    cont->type = VCONT_TYPE_RAD;
}

template <typename Trans>
static void sample(Trans t, const Coords *coords, const Particle *pp, const int *starts, const int *counts,      Sampler *s) {
    int3 L = s->L;
    Coords_v coordsv;

    coords_get_view(coords, &coordsv);

    dim3 block(8, 8, 1);
    dim3 grid(ceiln(L.x, block.x),
              ceiln(L.y, block.y),
              ceiln(L.z, block.z));

    KL(vcont_sample_kernel, (grid, block), (coordsv, t, L, starts, counts, pp,      s->gridvel, s->gridnum));
    s->nsamples++;
}

void vcont_sample(const Coords *coords, int, const Particle *pp, const int *starts, const int *counts,      PidVCont *c) {
    switch (c->type) {
    case VCONT_TYPE_NONE:
        break;
    case VCONT_TYPE_CART:
        sample(c->trans.cart, coords, pp, starts, counts,      &c->sampler);
        break;
    case VCONT_TYPE_RAD:
        sample(c->trans.cart, coords, pp, starts, counts,      &c->sampler);
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

    KL(vcont_reduce_by_warp_kernel, (nchunks, WARPSIZE), (s->gridvel, s->gridnum, ncells,      s->dtotvel, s->dtotnum));
    dSync();

    *v = make_double3(0, 0, 0);
    *n = 0;

    for (i = 0; i < nchunks; ++i) {
        add(s->totvel + i,      v);
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
    scal(fac,      &vcur);

    return make_float3(vcur.x, vcur.y, vcur.z);
}

static float3 update_state(const Param *p, float3 vcur, State *s) {
    float3 e, de, f;

    s->cur = vcur;
    diff(&p->target, &vcur,      &e);
    diff(&e, &s->olde,      &de);
    add(&e,      &s->sume);
    s->olde = e;

    f = make_float3(0, 0, 0);

    axpy(p->Kp, &e,            &f);
    axpy(p->Ki, &s->sume,      &f);
    axpy(p->Kd, &de,           &f);
    s->f = f;
    return f;
}

float3 vcont_adjustF(     PidVCont *c) {
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








#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>






struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);
struct DCont;
struct DContMap;

struct Config;
struct Coords;

void den_ini(int maxp,      DCont**);
void den_map_ini(DContMap**);

void den_fin(DCont*);
void den_map_fin(DContMap*);

void den_map_set_none(const Coords*, DContMap*);
void den_map_set_circle(const Coords*, float R, DContMap*);

void den_map_set_conf(const Config*, const Coords*, DContMap*);

void den_reset(int n,      DCont*d);
void den_filter_particles(int maxdensity, const DContMap*, const int *starts, const int *counts,      DCont*);
void den_download_ndead(DCont*);

const int* den_get_deathlist(DCont*);
int        den_get_ndead(DCont*);

void den_map_set_conf(const Config *cfg, const Coords *c, DContMap *m) {
    const char *type;

    UC(conf_lookup_string(cfg, "denoutflow.type", &type));
    if (same_str(type, "none")) {
        UC(den_map_set_none(c,      m));
    }
    else if (same_str(type, "circle")) {
        float R;
        UC(conf_lookup_float(cfg, "denoutflow.R", &R));
        UC(den_map_set_circle(c, R,      m));
    } else {
        ERR("Unrecognized type <%s>", type);
    }
}



#include <stdio.h>
#include <curand_kernel.h>













#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
#endif
#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif


#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())









#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif

#ifndef UDX_CONTROL_COORDS_COMMON
#define UDX_CONTROL_COORDS_COMMON
struct Coords_v {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};
#endif
struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);

#ifndef UDX_CONTROL_SHFL_COMMON
#define UDX_CONTROL_SHFL_COMMON
#define _I_ static __device__

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

#undef _I_
#define _S_ static __device__
#define _I_ static __device__

template <typename T>
_I_ T warpReduceSum(T v)
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += shfl_down(v, offset);
    return v;
}

_I_ float2 warpReduceSum(float2 val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
    }
    return val;
}

_I_ float3 warpReduceSum(float3 val)
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

#endif



__global__ void den_kill_kernel(int maxden, const int *starts, const int *counts, int n, const int *cids,      int *ndead, int *kk) {
    int i, nd, cid, s, c, pid, j;
    nd = 0;
    c = maxden;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        cid = cids[i];
        c = counts[cid];
        s = starts[cid];
    }

    nd = max(0, c - maxden);


    if (i < n) {
        for (j = 0; j < nd; ++j) {
            pid = s + j;
            kk[pid] = 1;
        }
    }


    nd = warpReduceSum(nd);
    if (threadIdx.x % warpSize == 0)
        atomicAdd(ndead, nd);
}
struct DCont {
    int *ndead_dev, ndead;
    int *kk;

};

struct DContMap {
    int n;
    int *cids;
};
static void reset_ndead(DCont *d) {
    CC(d::MemsetAsync(d->ndead_dev, 0, sizeof(int)));
    d->ndead = 0;
}

void den_ini(int maxp,      DCont **d0) {
    DCont *d;
    size_t sz;

    EMALLOC(1, d0);
    d = *d0;

    sz = maxp * sizeof(int);
    Dalloc(&d->kk, maxp);
    Dalloc(&d->ndead_dev, 1);

    CC(d::MemsetAsync(d->kk, 0, sz));
    reset_ndead(d);
}

void den_fin(DCont *d) {
    Dfree(d->kk);
    Dfree(d->ndead_dev);
    EFREE(d);
}

void den_reset(int n,      DCont *d) {
    reset_ndead(d);
    CC(d::MemsetAsync(d->kk, 0, n * sizeof(int)));
}

void den_filter_particles(int maxdensity, const DContMap *m, const int *starts, const int *counts,      DCont *d) {
    KL( den_kill_kernel, (k_cnf(m->n)), (maxdensity, starts, counts, m->n, m->cids,      d->ndead_dev, d->kk) );
}

void den_download_ndead(DCont *d) {
    CC(d::Memcpy(&d->ndead, d->ndead_dev, sizeof(int), D2H));

}


const int* den_get_deathlist(DCont *d) {return d->kk;}
int        den_get_ndead(DCont *d)     {return d->ndead;}
struct Circle {
    float R;
};

struct Plate {

};

struct None {};

static int predicate(const Coords *c, Circle p, int i, int j, int k) {
    enum {X, Y, Z};

    float3 r;
    float R;

    r = make_float3(i + (1 - xs(c)) * 0.5f,
                    j + (1 - ys(c)) * 0.5f,
                    k + (1 - zs(c)) * 0.5f);
    local2center(c, r,      &r);

    R = sqrt(r.x * r.x + r.y * r.y);
    return R < p.R && R >= p.R - 1;
}

static int predicate(const Coords*, None, int, int, int) {
    return 0;
}

template <typename T>
static void ini_map(const Coords *coords, T p, int **ids, int *nids) {
    int *ii, i, j, k, n;
    size_t sz;
    int Lx, Ly, Lz;
    Lx = xs(coords);  Ly = ys(coords);  Lz = zs(coords);

    EMALLOC(Lx * Ly * Lz, &ii);

    n = 0;
    for (k = 0; k < Lz; ++k) {
        for (j = 0; j < Ly; ++j) {
            for (i = 0; i < Lx; ++i) {
                if (predicate(coords, p, i, j, k))
                    ii[n++] = i + Lx * (j + Ly * k);
            }
        }
    }

    *nids = n;
    sz = n * sizeof(int);

    Dalloc(ids, n);
    CC(d::Memcpy(*ids, ii, sz, H2D));

    EFREE(ii);
}

void den_map_ini(DContMap **m) {
    EMALLOC(1, m);
    (*m)->cids = NULL;
}

void den_map_set_none(const Coords *c, DContMap *m) {
    None p;
    ini_map(c, p, &m->cids, &m->n);
}

void den_map_set_circle(const Coords *c, float R, DContMap *m) {
    Circle p;
    p.R = R;
    ini_map(c, p, &m->cids, &m->n);
}

void den_map_fin(DContMap *m) {
    if (m->cids) Dfree(m->cids);
    EFREE(m);
}





#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>
#include <vector_types.h>






struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);

struct Outflow;

struct Config;
struct Coords;
struct Particle;
struct float3;

void outflow_ini(int maxp,      Outflow **o);
void outflow_fin(     Outflow *o);

void outflow_set_params_circle(const Coords*, float3 c, float R, Outflow *o);
void outflow_set_params_plate(const Coords*, int dir, float r0, Outflow *o);

void outflow_set_conf(const Config *cfg, const Coords*, Outflow *);

void outflow_filter_particles(int n, const Particle *pp,      Outflow *o);

void outflow_download_ndead(Outflow *o);

int* outflow_get_deathlist(Outflow *o);
int  outflow_get_ndead(Outflow *o);


void outflow_set_conf(const Config *cfg, const Coords *coords, Outflow *o) {
    const char *type;
    UC(conf_lookup_string(cfg, "outflow.type", &type));

    if      (same_str(type, "circle")) {
        float3 center;
        float R;
        UC(conf_lookup_float(cfg, "outflow.R", &R));
        UC(conf_lookup_float3(cfg, "outflow.center", &center));

        outflow_set_params_circle(coords, center, R,      o);
    }
    else if (same_str(type, "plate")) {
        int dir;
        float r0;
        UC(conf_lookup_int(cfg, "outflow.direction", &dir));
        UC(conf_lookup_float(cfg, "outflow.position", &r0));
        outflow_set_params_plate(coords, dir, r0,      o);
    }
    else {
        ERR("Unrecognized type <%s>", type);
    }
}



#include <stdio.h>













#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
#endif
#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif

#ifndef UDX_CONTROL_MATH_COMMON
#define UDX_CONTROL_MATH_COMMON


#define _HD_ __host__ __device__

template <typename T1, typename T2>
_HD_ void scal(const T1 a,        T2 *b) {
    b->x *= a;
    b->y *= a;
    b->z *= a;
}

template <typename T1, typename T2>
_HD_ void add(const T1 *a,        T2 *b) {
    b->x += a->x;
    b->y += a->y;
    b->z += a->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void diff(const T1 *a, const T2 *b,      T3 *c) {
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ T1 dot(const T2 *a, const T3 *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void cross(const T1 *a, const T2 *b,      T3 *c) {
    c->x = a->y * b->z - a->z * b->y;
    c->y = a->z * b->x - a->x * b->z;
    c->z = a->x * b->y - a->y * b->x;
}

template <typename T1, typename T2, typename T3>
_HD_ void scalmult(const T1 *a, const T2 x,      const T3 *b) {
    b->x = a->x * x;
    b->y = a->y * x;
    b->z = a->z * x;
}


template <typename T1, typename T2, typename T3, typename T4>
_HD_ void apxb(const T1 *a, const T2 x, const T3 *b,      T4 *c) {
    c->x = a->x + x * b->x;
    c->y = a->y + x * b->y;
    c->z = a->z + x * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void axpy(const T1 a, const T2 *x,        T3 *y) {
    y->x += a * x->x;
    y->y += a * x->y;
    y->z += a * x->z;
}
#endif

#ifndef UDX_CONTROL_COORDS_COMMON
#define UDX_CONTROL_COORDS_COMMON
struct Coords_v {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};
#endif
struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);


#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())









#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif



enum {
    OFLOW_TYPE_NONE,
    OFLOW_TYPE_PLATE,
    OFLOW_TYPE_CIRCLE
};

struct OutflowParamsCircle {
    float Rsq;
    float3 c;
};


struct OutflowParamsPlate {
    float a, b, c, d;
};

union OutflowParamsU {
    OutflowParamsPlate plate;
    OutflowParamsCircle circle;
};

struct Outflow {
    int *kk;
    int *ndead_dev;
    int ndead;

    OutflowParamsU params;
    int type;
};

static __device__ int outflow_predicate(OutflowParamsPlate p, const float r[3]) {
    enum {X, Y, Z};
    float s;

    s = p.a * r[X] + p.b * r[Y] + p.c * r[Z] + p.d;

    return s > 0;
}

static __device__ int outflow_predicate(OutflowParamsCircle p, const float r[3]) {
    enum {X, Y, Z};
    float x, y, rsq;
    x = r[X] - p.c.x;
    y = r[Y] - p.c.y;

    rsq = x*x + y*y;
    return rsq > p.Rsq;
}

template <typename Par>
__global__ void outflow_filter_kernel(int n, const Particle *pp, Par params,      int *ndead, int *kk) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int dead;
    Particle p;
    if (i >= n) return;

    p = pp[i];
    dead = outflow_predicate(params, p.r);
    kk[i] = dead;
    if (dead)
        atomicAdd(ndead, 1);
}

static void reset_ndead(Outflow *o) {
    CC(d::MemsetAsync(o->ndead_dev, 0, sizeof(int)));
    o->ndead = 0;
}

void outflow_ini(int maxp,      Outflow **o0) {
    Outflow *o;
    size_t sz;

    EMALLOC(1, o0);
    o = *o0;

    sz = maxp * sizeof(int);
    Dalloc(&o->kk, maxp);
    Dalloc(&o->ndead_dev, 1);

    CC(d::MemsetAsync(o->kk, 0, sz));
    UC(reset_ndead(o));
    o->type = OFLOW_TYPE_NONE;
}

void outflow_fin(     Outflow *o) {
    Dfree(o->kk);
    Dfree(o->ndead_dev);
    EFREE(o);
}

void outflow_set_params_circle(const Coords *coords, float3 c, float R, Outflow *o) {
    enum {X, Y, Z};
    OutflowParamsCircle p;
    p.Rsq = R*R;


    global2local(coords, c,      &p.c);

    o->params.circle = p;
    o->type = OFLOW_TYPE_CIRCLE;
}

void outflow_set_params_plate(const Coords *c, int dir, float r0, Outflow *o) {
    enum {X, Y, Z};
    OutflowParamsPlate p = {0, 0, 0, 0};

    switch (dir) {
    case X:
        p.a = 1;
        p.d = xg2xl(c, r0);
        break;
    case Y:
        p.b = 1;
        p.d = yg2yl(c, r0);
        break;
    case Z:
        p.c = 1;
        p.d = zg2zl(c, r0);
        break;
    default:
        ERR("undefined direction %d\n", dir);
        break;
    };

    o->params.plate = p;
    o->type = OFLOW_TYPE_PLATE;
}

void outflow_filter_particles(int n, const Particle *pp,      Outflow *o) {
    reset_ndead(o);

    switch(o->type) {
    case OFLOW_TYPE_CIRCLE:
        KL(outflow_filter_kernel, (k_cnf(n)), (n, pp, o->params.circle,      o->ndead_dev, o->kk) );
        break;
    case OFLOW_TYPE_PLATE:
        KL(outflow_filter_kernel, (k_cnf(n)), (n, pp, o->params.plate,      o->ndead_dev, o->kk) );
        break;
    case OFLOW_TYPE_NONE:
    default:
        ERR("No type set");
        break;
    };
}

void outflow_download_ndead(Outflow *o) {
    CC(d::Memcpy(&o->ndead, o->ndead_dev, sizeof(int), D2H));

}

int* outflow_get_deathlist(Outflow *o) {return o->kk;}
int  outflow_get_ndead(Outflow *o)     {return o->ndead;}













#ifndef UDX_BOP_COMMON
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

struct Cbuf {
    enum {SIZ=256};
    char c[SIZ];
};
struct BopData;
typedef int BopStatus;

enum {BopFLOAT, BopDOUBLE, BopINT, BopFASCII, BopIASCII, _BopNTYPES};
typedef int BopType;

BopStatus bop_ini(BopData **d);
BopStatus bop_alloc(BopData *d);
BopStatus bop_fin(BopData *d);

BopStatus bop_set_n(long n, BopData *d);
BopStatus bop_set_vars(int n, const char *vars, BopData *d);
BopStatus bop_set_type(BopType type, BopData *d);
void* bop_get_data(BopData *d);

BopStatus bop_get_n(const BopData *d, long *n);
BopStatus bop_get_nvars(const BopData *d, int *n);
BopStatus bop_get_vars(const BopData *d, Cbuf *vars);
BopStatus bop_get_vars(const BopData *d, const char **vars);
BopStatus bop_get_type(const BopData *d, BopType *type);
const void* bop_get_data(const BopData *d);

BopStatus bop_summary(const BopData *d);
BopStatus bop_concatenate(const int nd, const BopData **dd, BopData *dall);

bool         bop_success(BopStatus status);
const char * bop_report_error_desc(BopStatus status);
char *       bop_report_error_mesg();



#define CBUFSIZE 256
#define CERRSIZE 1024


struct BopData {
    long n;
    int nvars;
    char vars[CBUFSIZE];
    BopType type;
    void *data;
};


extern char bop_error_msg[CERRSIZE];

enum {
    BOP_SUCCESS,
    BOP_BADALLOC,
    BOP_BADFILE,
    BOP_WRONGVAR,
    BOP_MISMATCH,
    BOP_WFORMAT,
    BOP_NULLPTR,
    BOP_WMPISIZE,
    BOP_OVERFLOW,
    _BOP_NERR
};

#define xstr(s) str(s)
#define str(s) #s

BopStatus safe_malloc(size_t sz, void **data);
BopStatus safe_open(const char *fname, const char *mode, FILE **f);

size_t get_bsize(BopType t);

void get_path(const char *full, char *path);
void get_fname_values(const char *fnbop, char *fnval);

const char * type2str(BopType t);
BopType str2type(const char *str);

void report_err(const char *fmt, ...);

enum {SERIAL = -1};
BopStatus read_header(int rank, const char *fname,      char *dfname, BopData *d);

BopStatus bop_write_header(MPI_Comm comm, const char *name, const BopData *d);
BopStatus bop_write_values(MPI_Comm comm, const char *name, const BopData *d);

BopStatus bop_read_header(MPI_Comm comm, const char *hfname, BopData *d, char *dfname);
BopStatus bop_read_values(MPI_Comm comm, const char *dfname, BopData *d);

BopStatus bop_write_header(const char *name, const BopData *d);
BopStatus bop_write_values(const char *name, const BopData *d);

BopStatus bop_read_header(const char *hfname, BopData *d, char *dfname);
BopStatus bop_read_values(const char *dfname, BopData *d);

#define UDX_BOP_COMMON
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

char bop_error_msg[1024] = {0};
static const char * err_desc[_BOP_NERR] = {
    "success",
    "bad allocation",
    "bad file descriptor",
    "wrong number of variables",
    "file types mismatch",
    "wrong format",
    "wrong mpi size",
    "null pointer",
    "overflow"
};

BopStatus bop_ini(BopData **d) {
    BopStatus s;
    BopData *b;
    s = safe_malloc(sizeof(BopData), (void **) d);
    if (s == BOP_SUCCESS) {
        b = *d;
        b->n = 0;
        b->nvars = 0;
        b->vars[0] = '\0';
        b->type = BopFLOAT;
        b->data = NULL;
    }
    return s;
}

BopStatus bop_alloc(BopData *d) {
    size_t sz, bsize;
    bsize = get_bsize(d->type);
    sz = d->n * d->nvars * bsize;

    return safe_malloc(sz, &d->data);
}

BopStatus bop_fin(BopData *d) {
    if (d->data) free(d->data);
    free(d);
    return BOP_SUCCESS;
}

BopStatus bop_set_n(long n, BopData *d) {
    if (d) {
        d->n = n;
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

BopStatus bop_set_vars(int n, const char *vars, BopData *d) {
    if (d) {
        d->nvars = n;
        strcpy(d->vars, vars);
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

BopStatus bop_set_type(BopType type, BopData *d) {
    if (d) {
        d->type = type;
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

BopStatus bop_get_n(const BopData *d, long *n) {
   if (d) {
        *n = d->n;
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

BopStatus bop_get_nvars(const BopData *d, int *n) {
   if (d) {
        *n = d->nvars;
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

static BopStatus extract_vars(int n, const char *v,      Cbuf *vars) {
    int i, ret;
    for (i = 0; i < n; ++i) {
        ret = sscanf(v, "%s", vars[i].c);
        if (ret != 1)
            return BOP_WRONGVAR;
        v = strstr(v, vars[i].c);
        v += strlen(vars[i].c);
    }
    return BOP_SUCCESS;
}

BopStatus bop_get_vars(const BopData *d, Cbuf *vars) {
   if (d) {
        return extract_vars(d->nvars, d->vars,      vars);
    }
    return BOP_NULLPTR;
}

BopStatus bop_get_vars(const BopData *d, const char **vars) {
   if (d) {
       *vars = d->vars;
       return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

BopStatus bop_get_type(const BopData *d, BopType *type) {
   if (d) {
        *type = d->type;
        return BOP_SUCCESS;
    }
    return BOP_NULLPTR;
}

void* bop_get_data(BopData *d) {
    return d->data;
}

const void* bop_get_data(const BopData *d) {
    return d->data;
}


BopStatus bop_summary(const BopData *d) {
    fprintf(stderr, "(reader) found %ld entries, %d field(s)\n", d->n, d->nvars);
    fprintf(stderr, "\tformat: %s\n", type2str(d->type));
    fprintf(stderr, "\tvars: %s\n", d->vars);
    return BOP_SUCCESS;
}

BopStatus bop_concatenate(const int nd, const BopData **dd, BopData *dall) {
    const BopData *d = dd[0];
    long n             = d->n;
    const BopType type = d->type;
    const int nvars    = d->nvars;
    size_t bsize;
    int i;
    long ni, start;
    const void *src;
    char *dst;

    for (i = 1; i < nd; ++i) {
        d = dd[i];
        n += d->n;
        if (type != d->type || nvars != d->nvars)
            return BOP_MISMATCH;
    }
    bsize = get_bsize(type);

    dall->n = n;
    dall->nvars = nvars;
    dall->type = type;
    bop_alloc(dall);

    d = dd[0];
    strcpy(dall->vars, d->vars);

    start = 0;

    for (i = 0; i < nd; ++i) {
        d = dd[i];
        ni = d->n;
        src = d->data;
        dst = (char *) dall->data + bsize * start;
        memcpy(dst, src, ni * nvars * bsize);

        start += ni * nvars;
    }

    return BOP_SUCCESS;
}

bool bop_success(BopStatus status) {
    return status == BOP_SUCCESS;
}

const char* bop_report_error_desc(BopStatus status) {
    assert(status >= 0 && status < _BOP_NERR);
    return err_desc[status];
}

char* bop_report_error_mesg() {
    return bop_error_msg;
}


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

static const char *typestr[] = {
    "float", "double", "int", "fascii", "iascii"};

BopStatus safe_malloc(size_t sz, void **data) {
    *data = malloc(sz);

    if (*data == NULL) {
        report_err("could not allocate array of %ld bytes\n", sz);
        return BOP_BADALLOC;
    }
    return BOP_SUCCESS;
}

BopStatus safe_open(const char *fname, const char *mode, FILE **f) {
    *f = fopen(fname, mode);
    if (*f == NULL) {
        report_err("could not open <%s>\n", fname);
        return BOP_BADFILE;
    }
    return BOP_SUCCESS;
}

size_t get_bsize(BopType t) {
    switch(t) {
    case BopFLOAT:
    case BopFASCII:
        return sizeof(float);
    case BopDOUBLE:
        return sizeof(double);
    case BopINT:
    case BopIASCII:
        return sizeof(int);
    };
    return 0;
}

#define SEP '/'
void get_path(const char *full, char *path) {
    int i = strlen(full);
    while (--i >= 0 && full[i] != SEP);

    if (i) memcpy(path, full, (i+1)*sizeof(char));
}

void get_fname_values(const char *fnbop, char *fnval) {
    int i = strlen(fnbop);
    const int n = i;
    while (--i >= 0 && fnbop[i] != SEP);

    memcpy(fnval, fnbop + i + 1, (n-i)*sizeof(char));

    i = strlen(fnval);
    strncpy(fnval + i - 4, ".values", 8);
}

#undef SEP

BopType str2type(const char *str) {
    if      (strcmp(str,  "float") == 0) return BopFLOAT;
    else if (strcmp(str, "double") == 0) return BopDOUBLE;
    else if (strcmp(str,    "int") == 0) return BopINT;
    else if (strcmp(str,  "ascii") == 0) return BopFASCII;
    else if (strcmp(str, "iascii") == 0) return BopIASCII;

    return BopFLOAT;
}

const char * type2str(BopType t) {
    if (t >= 0 && t < _BopNTYPES)
        return typestr[t];
    return NULL;
}

void report_err(const char *fmt, ...) {
    va_list ap;
    char msg[CERRSIZE];

    va_start(ap, fmt);
    vsnprintf(msg, CERRSIZE - 1, fmt, ap);
    va_end(ap);

    snprintf(bop_error_msg, CERRSIZE, ":%s:%d: %s",
             __FILE__, __LINE__, msg);
}

#include <stdio.h>
#include <string.h>

static int is_desc(const char *s, const char *d) {
    if (strcmp(s, d) == 0)
        return 1;
    return 0;
}

static void read_data_file(const char *data,      char *df) {
    strcpy(df, data);
}

static void read_variables(const char *data,      BopData *d) {
    strcpy(d->vars, data);
}

static void read_type(const char *data,      BopData *d) {
    d->type = str2type(data);
}

static void extract_desc_data(const char *l,      char *desc, char *data) {
    const char *c;
    char *d;
    for (c = l, d = desc; l < l + strlen(l); ++c, ++d) {
        if (*c == ':')
            break;
        *d = *c;
    }
    while (*c == ':' || *c == ' ') ++c;
    strcpy(data, c);
}

static void read_nrank(int rank, const char *data,          FILE *f,      BopData *d) {
    int i, size;
    long nloc;
    sscanf(data, "%d", &size);
    for (i = 0; i < size; ++i) {
        fscanf(f, "%ld\n", &nloc);
        if (i == rank) d->n = nloc;
    }
}

static BopStatus parse_entry(int rank, const char *l,      FILE *f, char *dfname, BopData *d) {
    char desc[CBUFSIZE] = {0}, data[CBUFSIZE] = {0};
    extract_desc_data(l,      desc, data);

    if (is_desc(desc, "DATA_FILE"))
        read_data_file(data,      dfname);
    else if (is_desc(desc, "DATA_FORMAT"))
        read_type(data,      d);
    else if (is_desc(desc, "VARIABLES"))
        read_variables(data,      d);
    else if (is_desc(desc, "NRANK"))
        read_nrank(rank, data,      f,      d);
    else {
        report_err("unprocessed desc: <%s>, data: <%s>\n", desc, data);
        return BOP_WFORMAT;
    }
    return BOP_SUCCESS;
}

static BopStatus read_n_data(const char *data,      BopData *d) {
    long n;
    if (1 != sscanf(data, "%ld", &n))
        return BOP_WFORMAT;
    d->n = n;
    return BOP_SUCCESS;
}

static int count_vars(const char *var) {
    int i;
    char buf[CBUFSIZE];

    i = 0;
    while (1 == sscanf(var, "%s", buf)) {
        var = strstr(var, buf);
        var += strlen(buf);
        ++i;
    }
    return i;
}

static bool read_entry(FILE *f, char *l) {
    int ret = fscanf(f, " %" xstr(CBUFSIZE) "[^\n]c", l);
    return EOF != ret;
}

BopStatus read_header(int rank, const char *fname,      char *dfname, BopData *d) {
    FILE *f;
    char line[CBUFSIZE];
    int l = 0;
    BopStatus s;

    s = safe_open(fname, "r", &f);
    if (s != BOP_SUCCESS) return s;

    while (read_entry(f, line)) {

        if (l == 0) s = read_n_data(line,      d);
        else        s = parse_entry(rank, line,      f, dfname, d);

        if (s != BOP_SUCCESS) return s;
        ++l;
    }

    d->nvars = count_vars(d->vars);

    fclose(f);
    return s;
}

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

template <typename T>
static void write_mpi(MPI_Comm comm, const char *fname, long n, const T *data, MPI_Datatype type) {
    MPI_Offset base, offset, len;
    MPI_Status status;
    MPI_File f;

    MPI_File_open(comm, fname , MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f);
    MPI_File_set_size(f, 0);
    MPI_File_get_position(f, &base);

    len = n * sizeof(T);
    offset = 0;
    MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
    MPI_File_write_at_all(f, base + offset, data, n, type, &status);
    MPI_File_close(&f);
}

static BopStatus write_header(MPI_Comm comm, const char *fhname, const char *fdname, const BopData *d) {
    enum {
        ROOT     = 0,
        MAX_SIZE = 2048
    };
    int rank, size, nchar;
    long nloc, ntot;
    char *buf;
    BopStatus s;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    nloc = d->n;
    ntot = 0;
    MPI_Reduce(&nloc, &ntot, 1, MPI_LONG, MPI_SUM, ROOT, comm);

    s = safe_malloc(MAX_SIZE * sizeof(char), (void**) &buf);
    if (s != BOP_SUCCESS) return s;

    nchar = 0;
    if (rank == ROOT) {
        nchar += sprintf(buf + nchar, "%ld\n", ntot);
        nchar += sprintf(buf + nchar, "DATA_FILE: %s\n", fdname);
        nchar += sprintf(buf + nchar, "DATA_FORMAT: %s\n", type2str(d->type));
        nchar += sprintf(buf + nchar, "VARIABLES: %s\n", d->vars);
        nchar += sprintf(buf + nchar, "NRANK: %d\n", size);
    }
    nchar += sprintf(buf + nchar, "%ld\n", d->n);

    if (nchar >= MAX_SIZE) {
        report_err("buffer is too small for header: %d, %d\n", nchar, MAX_SIZE);
        return BOP_OVERFLOW;
    }

    write_mpi(comm, fhname, nchar, buf, MPI_CHAR);

    free(buf);
    return s;
}

BopStatus bop_write_header(MPI_Comm comm, const char *name, const BopData *d) {
    char fnval[CBUFSIZE] = {0},
        fnval0[CBUFSIZE] = {0},
        fnhead[CBUFSIZE] = {0};

    sprintf(fnhead, "%s.bop", name);

    get_path(fnhead, fnval);
    get_fname_values(fnhead, fnval0);
    strcat(fnval, fnval0);

    return write_header(comm, fnhead, fnval0, d);
}

template <typename T>
static BopStatus write_ascii(MPI_Comm comm, const char *pattern, const char *fname, const T *data, long n, int nvars) {
    char *buf;
    long i, k, j = 0, ns = 0;
    BopStatus s;
    enum {NPN = 16};

    s = safe_malloc(NPN * n * nvars * sizeof(char), (void**) &buf);
    if (s != BOP_SUCCESS) return s;

    for (i = 0; i < n; ++i) {
        for (k = 0; k < nvars; ++k)
            ns += sprintf(buf + ns, pattern, data[j++]);
        ns += sprintf(buf + ns, "\n");
    }
    write_mpi(comm, fname, ns, buf, MPI_CHAR);
    free(buf);
    return s;
}

static BopStatus write_data(MPI_Comm comm, const char *fnval, const BopData *d) {
    long n = d->n;
    int nvars = d->nvars;

    switch(d->type) {
    case BopFLOAT:
        write_mpi(comm, fnval, n * nvars, (const float *) d->data, MPI_FLOAT);
        break;
    case BopDOUBLE:
        write_mpi(comm, fnval, n * nvars, (const double *) d->data, MPI_DOUBLE);
        break;
    case BopINT:
        write_mpi(comm, fnval, n * nvars, (const int *) d->data, MPI_INT);
        break;
    case BopFASCII:
        return write_ascii(comm, "%.6e", fnval, (const float *) d->data, n, nvars);
        break;
    case BopIASCII:
        return write_ascii(comm, "%d", fnval, (const int *) d->data, n, nvars);
        break;
    }
    return BOP_SUCCESS;
}

BopStatus bop_write_values(MPI_Comm comm, const char *name, const BopData *d) {
    char dfname[CBUFSIZE] = {0};
    sprintf(dfname, "%s.values", name);
    return write_data(comm, dfname, d);
}

BopStatus bop_read_header(MPI_Comm comm, const char *hfname, BopData *d, char *dfname) {
    BopStatus s;
    char dfname0[CBUFSIZE] = {0}, locdfname[CBUFSIZE] = {0};
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    s = read_header(rank, hfname,      dfname0, d);

    get_path(hfname, locdfname);
    strcat(locdfname, dfname0);
    strcpy(dfname, locdfname);

    return s;
}

template <typename T>
static BopStatus read_mpi(MPI_Comm comm, const char *fname, long n, MPI_Datatype type, T *data) {
    MPI_Offset base, offset, len;
    MPI_Status status;
    MPI_File f;

    MPI_File_open(comm, fname , MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
    MPI_File_get_position(f, &base);

    len = n * sizeof(T);
    offset = 0;
    MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm);
    MPI_File_read_at_all(f, base + offset, data, n, type, &status);
    MPI_File_close(&f);
    return BOP_SUCCESS;
}

template <typename T>
static BopStatus read_ascii() {
    return BOP_SUCCESS;
}

BopStatus bop_read_values(MPI_Comm comm, const char *dfname, BopData *d) {
    long n = d->n;
    int nvars = d->nvars;

    switch (d->type) {
    case BopFLOAT:
        return read_mpi(comm, dfname, n * nvars, MPI_FLOAT, (float *) d->data);
    case BopDOUBLE:
        return read_mpi(comm, dfname, n * nvars, MPI_DOUBLE, (double *) d->data);
    case BopINT:
        return read_mpi(comm, dfname, n * nvars, MPI_INT, (int *) d->data);
    case BopFASCII:
    case BopIASCII:
        fprintf(stderr, "Not implemented\n");
        return BOP_WFORMAT;
    };
    return BOP_SUCCESS;
}

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static BopStatus write_header(const char *fhname, const char *fdname, const BopData *d) {
    FILE *f;
    BopStatus s;

    s = safe_open(fhname, "w", &f);
    if (s != BOP_SUCCESS) return s;

    fprintf(f, "%ld\n", d->n);
    fprintf(f, "DATA_FILE: %s\n", fdname);
    fprintf(f, "DATA_FORMAT: %s\n", type2str(d->type));
    fprintf(f, "VARIABLES: %s\n", d->vars);

    fclose(f);
    return s;
}

BopStatus bop_write_header(const char *name, const BopData *d) {
    char fnval[CBUFSIZE] = {0},
        fnval0[CBUFSIZE] = {0},
        fnhead[CBUFSIZE] = {0};

    sprintf(fnhead, "%s.bop", name);

    get_path(fnhead, fnval);
    get_fname_values(fnhead, fnval0);
    strcat(fnval, fnval0);

    return write_header(fnhead, fnval0, d);
}

template <typename T>
static void write_ascii(const char pattern[], const T *data, const long n, const long nvars, FILE *f) {
    long i, k, j = 0;
    for (i = 0; i < n; ++i) {
        for (k = 0; k < nvars; ++k)
            fprintf(f, pattern, data[j++]);
        fprintf(f, "\n");
    }
}

static BopStatus write_data(const char *fnval, const BopData *d) {
    FILE *fd;
    size_t bsize;
    BopStatus s;

    s = safe_open(fnval, "w", &fd);
    bsize = get_bsize(d->type);

    if (s != BOP_SUCCESS) return s;

    switch(d->type) {
    case BopFLOAT:
    case BopDOUBLE:
    case BopINT:
        fwrite(d->data, bsize, d->n * d->nvars, fd);
        break;
    case BopFASCII:
        write_ascii("%.6e ", (const float*) d->data, d->n, d->nvars, fd);
        break;
    case BopIASCII:
        write_ascii("%d ", (const int*) d->data, d->n, d->nvars, fd);
        break;
    }
    fclose(fd);

    return s;
}


BopStatus bop_write_values(const char *name, const BopData *d) {
    char dfname[CBUFSIZE] = {0};
    sprintf(dfname, "%s.values", name);
    return write_data(dfname, d);
}

BopStatus bop_read_header(const char *hfname, BopData *d, char *dfname) {
    BopStatus s;
    char dfname0[CBUFSIZE] = {0}, locdfname[CBUFSIZE] = {0};
    s = read_header(SERIAL, hfname,      dfname0, d);

    get_path(hfname, locdfname);
    strcat(locdfname, dfname0);
    strcpy(dfname, locdfname);

    return s;
}

static BopStatus read_values(const char *fn, long n, int nvars, size_t bsize, void *data) {
    FILE *f;
    BopStatus s;
    s = safe_open(fn, "r", &f);
    if (s != BOP_SUCCESS) return s;

    fread(data, bsize, n * nvars, f);

    fclose(f);
    return s;
}

template <typename real>
static BopStatus read_ascii_values(const char pattern[], const char *fn, void *data) {
    char buf[CBUFSIZE] = {0}, *str;
    long i = 0, j;
    FILE *f;
    BopStatus s;
    real *d, a;

    s = safe_open(fn, "r", &f);
    if (s != BOP_SUCCESS) return s;

    d = (real*) data;

    while (fscanf(f, " %[^\n]" xstr(MAXC) "c", buf) == 1) {
        str = buf;
        j = 0;
        while (sscanf(str, pattern, &a, &j) == 1) {
            str += j;
            d[i++] = a;
        }
        memset(buf, 0,  sizeof(buf));
    }

    fclose(f);
    return s;
}


BopStatus bop_read_values(const char *dfname, BopData *d) {
    size_t bsize;
    bsize = get_bsize(d->type);

    switch (d->type) {
    case BopFLOAT:
    case BopDOUBLE:
    case BopINT:
        return read_values(dfname, d->n, d->nvars, bsize, d->data);
    case BopFASCII:
        return read_ascii_values<float>(" %f%n", dfname, d->data);
    case BopIASCII:
        return read_ascii_values<int>  (" %d%n", dfname, d->data);
    };
    return BOP_SUCCESS;
}











#include <mpi.h>
#include <stdio.h>










#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
#endif








#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif

#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif
#ifndef UDX_GRID_INTER_DEF_COMMON
#define UDX_GRID_INTER_DEF_COMMON
#include "constants.cuh"
#endif

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())










void grid_write(int3 subGrid, int3 subDomain, MPI_Comm, const char *path, int ncmp, const float **data, const char **names);

struct GridSampler;
struct GridSampleData;

struct Particle;

void grid_sampler_data_ini(GridSampleData **);
void grid_sampler_data_fin(GridSampleData  *);

void grid_sampler_data_reset(GridSampleData *);
void grid_sampler_data_push(long n, const Particle *pp, const int *cc,
                            const float *ss, GridSampleData *);


void grid_sampler_ini(bool colors, bool stress, int3 L, int3 N, GridSampler**);
void grid_sampler_fin(GridSampler*);

void grid_sampler_reset(GridSampler*);
void grid_sampler_add(const GridSampleData*, GridSampler*);
void grid_sampler_dump(MPI_Comm, const char *dir, long id, GridSampler*);
enum {
    RHO, VX, VY, VZ,
    NFIELDS_P
};

enum {
    SXX, SXY, SXZ,
    SYY, SYZ, SZZ,
    NFIELDS_S
};

enum {
    NFIELDS_C = N_COLOR
};

enum {
    TOT_NFIELDS = NFIELDS_P + NFIELDS_S + NFIELDS_C
};

static const char *names_p[NFIELDS_P] =
    {"density", "u", "v", "w"};

static const char *names_s[NFIELDS_S] =
    {"sxx", "sxy", "sxz",
     "syy", "syz", "szz"};

#define make_str(a) #a ,
static const char *names_c[NFIELDS_C] =
    {XMACRO_COLOR(make_str)};
#undef make_str

struct Grid {
    int3 L, N;
    float *p[NFIELDS_P];
    float *s[NFIELDS_S];
    float *c[NFIELDS_C];
    bool colors, stress;
};

struct GridSampler {
    Grid sdev, stdev, hst;
    long nsteps;
};

struct SampleDatum {
    long n;
    const Particle *pp;
    const float *ss;
    const int *cc;
};

enum {
    MAX_N_DATA = 10
};

struct GridSampleData {
    int n;
    SampleDatum d[MAX_N_DATA];
};

struct SamplerPart {
    float3 r, v;
};

struct SamplerStress {
    float3 s1, s2;
};
#define _S_ static __device__
#define _I_ static __device__

_I_ void sampler_fetch_part(int i, const SampleDatum *d, SamplerPart *p) {
    float2 p0, p1, p2;
    const float2 *pp = (const float2*) d->pp;
    p0 = pp[3*i+0]; p1 = pp[3*i+1]; p2 = pp[3*i+2];

    p->r = make_float3(p0.x, p0.y, p1.x);
    p->v = make_float3(p1.y, p2.x, p2.y);
}

_I_ void sampler_fetch_color(int i, const SampleDatum *d, int *c) {
    *c = d->cc[i];
}

_I_ void sampler_fetch_stress(int i, const SampleDatum *d, SamplerStress *s) {
    float2 s0, s1, s2;
    const float2 *ss = (const float2*) d->ss;
    s0 = ss[3*i+0]; s1 = ss[3*i+1]; s2 = ss[3*i+2];

    s->s1 = make_float3(s0.x, s0.y, s1.x);
    s->s2 = make_float3(s1.y, s2.x, s2.y);
}

#undef _S_
#undef _I_
#define _S_ static __device__
#define _I_ static __device__

_I_ void sampler_add_part(int i, const SamplerPart *p, const Grid *g) {
    atomicAdd(g->p[RHO] + i, 1.f);
    atomicAdd(g->p[VX]  + i, p->v.x);
    atomicAdd(g->p[VY]  + i, p->v.y);
    atomicAdd(g->p[VZ]  + i, p->v.z);
}

_I_ void sampler_add_color(int i, const int c, const Grid *g) {
#define ADD_COLOR(a)                               \
    if (ENUM_COLOR(a) == c)                        \
        atomicAdd(g->c[ENUM_COLOR(a)] + i, 1.f);

    XMACRO_COLOR(ADD_COLOR)

#undef ADD_COLOR
}

_I_ void sampler_add_stress(int i, const SamplerStress *s, const Grid *g) {
    atomicAdd(g->s[SXX] + i, s->s1.x);
    atomicAdd(g->s[SXY] + i, s->s1.y);
    atomicAdd(g->s[SXZ] + i, s->s1.z);

    atomicAdd(g->s[SYY] + i, s->s2.x);
    atomicAdd(g->s[SYZ] + i, s->s2.y);
    atomicAdd(g->s[SZZ] + i, s->s2.z);
}

#undef _S_
#undef _I_
#define _S_ static __device__

_S_ float sampler_get_spacing(int L, int N) {
    return (float) L / (float) N;
}

_S_ int sampler_get_coord(float x, int L, int N) {
    int i;
    float dx;
    dx = sampler_get_spacing(L, N);
    i = (x + L/2) / dx;
    return min(N-1, max(0, i));
}

_S_ int3 sampler_get_cell_coords(float3 r, int3 L, int3 N) {
    return make_int3(sampler_get_coord(r.x, L.x, N.x),
                     sampler_get_coord(r.y, L.y, N.y),
                     sampler_get_coord(r.z, L.z, N.z));
}

_S_ int sampler_get_grid_id(int3 gc, int3 N) {
    return gc.x + N.x * (gc.y + N.y * gc.z);
}

_S_ int sampler_get_grid_id(const SamplerPart *p, const Grid *g) {
    int3 gcoords;
    gcoords = sampler_get_cell_coords(p->r, g->L, g->N);
    return sampler_get_grid_id(gcoords, g->N);
}

__global__ void sampler_add_kernel(SampleDatum data,      Grid grid) {
    int pid, gid;
    SamplerPart p;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= data.n) return;

    sampler_fetch_part(pid, &data, &p);
    gid = sampler_get_grid_id(&p, &grid);

    sampler_add_part(gid, &p, &grid);

    if (grid.colors) {
        int c;
        sampler_fetch_color(pid, &data, &c);
        sampler_add_color(gid, c, &grid);
    }

    if (grid.stress) {
        SamplerStress s;
        sampler_fetch_stress(pid, &data, &s);
        sampler_add_stress(gid, &s, &grid);
    }
}

_S_ long sampler_get_size(const Grid *g) {
    int3 N = g->N;
    return N.x * N.y * N.z;
}

_S_ float sampler_cell_volume(const Grid *g) {
    float dx, dy, dz;
    dx = sampler_get_spacing(g->L.x, g->N.x);
    dy = sampler_get_spacing(g->L.y, g->N.y);
    dz = sampler_get_spacing(g->L.z, g->N.z);
    return dx * dy * dz;
}

__global__ void sampler_space_avg_kernel(Grid g) {
    long i;
    float inv_rho;
    float rho, vol;
    float sv, sr, ss;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= sampler_get_size(&g)) return;

    rho = g.p[RHO][i];
    vol = sampler_cell_volume(&g);

    inv_rho = fabs(rho) > 1e-6 ? 1.f / rho : 0;

    sv = inv_rho;
    sr = 1.f / vol;

    g.p[RHO][i] = sr * rho;

    g.p[VX][i] *= sv;
    g.p[VY][i] *= sv;
    g.p[VZ][i] *= sv;

    if (g.colors) {
#define scale(a) g.c[ENUM_COLOR(a)][i] *= inv_rho;
        XMACRO_COLOR(scale)
#undef scale
    }

    if (g.stress) {
        ss = sr;
        g.s[SXX][i] *= ss;
        g.s[SXY][i] *= ss;
        g.s[SXZ][i] *= ss;
        g.s[SYY][i] *= ss;
        g.s[SYZ][i] *= ss;
        g.s[SZZ][i] *= ss;
    }
}

__global__ void sampler_add_to_grid_kernel(const Grid src, Grid dst) {
    long i, j;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= sampler_get_size(&src)) return;

    for (j = 0; j < NFIELDS_P; ++j)
        dst.p[j][i] += src.p[j][i];

    if (src.colors) {
        for (j = 0; j < NFIELDS_C; ++j)
            dst.c[j][i] += src.c[j][i];
    }

    if (src.stress) {
        for (j = 0; j < NFIELDS_S; ++j)
            dst.s[j][i] += src.s[j][i];
    }
}

__global__ void sampler_time_avg_kernel(int nsteps, Grid g) {
    long i, j;
    float s;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= sampler_get_size(&g)) return;

    s = 1.0 / nsteps;

    for (j = 0; j < NFIELDS_P; ++j)
        g.p[j][i] *= s;

    if (g.colors)
        for (j = 0; j < NFIELDS_C; ++j)
            g.c[j][i] *= s;

    if (g.stress)
        for (j = 0; j < NFIELDS_S; ++j)
            g.s[j][i] *= s;
}

#undef _S_

static long get_size(const Grid *g) {
    int3 N = g->N;
    return N.x * N.y * N.z;
}

static void dev_alloc_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) Dalloc(&a[i], n);
}

static void ini_dev_grid(bool colors, bool stress, int3 L, int3 N, Grid *g) {
    long n;
    g->N = N;
    g->L = L;
    g->colors = colors;
    g->stress = stress;

    n = get_size(g);

    UC(             dev_alloc_n(NFIELDS_P, n, g->p) );
    if (colors) UC( dev_alloc_n(NFIELDS_C, n, g->c) );
    if (stress) UC( dev_alloc_n(NFIELDS_S, n, g->s) );
}

static void hst_alloc_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) EMALLOC(n, &a[i]);
}

static void ini_hst_grid(bool colors, bool stress, int3 L, int3 N, Grid *g) {
    long n;
    g->N = N;
    g->L = L;
    g->colors = colors;
    g->stress = stress;

    n = get_size(g);
    UC(             hst_alloc_n(NFIELDS_P, n, g->p) );
    if (colors) UC( hst_alloc_n(NFIELDS_C, n, g->c) );
    if (stress) UC( hst_alloc_n(NFIELDS_S, n, g->s) );
}

static void dev_free_n(int na, float *a[]) {
    for (int i = 0; i < na; ++i) Dfree(a[i]);
}

static void fin_dev_grid(Grid *g) {
    UC(                dev_free_n(NFIELDS_P, g->p) );
    if (g->colors) UC( dev_free_n(NFIELDS_C, g->c) );
    if (g->stress) UC( dev_free_n(NFIELDS_S, g->s) );
}

static void hst_free_n(int na, float *a[]) {
    for (int i = 0; i < na; ++i) EFREE(a[i]);
}

static void fin_hst_grid(Grid *g) {
    UC(                hst_free_n(NFIELDS_P, g->p) );
    if (g->colors) UC( hst_free_n(NFIELDS_C, g->c) );
    if (g->stress) UC( hst_free_n(NFIELDS_S, g->s) );
}

void grid_sampler_ini(bool colors, bool stress, int3 L, int3 N, GridSampler **s0) {
    GridSampler *s;
    EMALLOC(1, s0);
    s = *s0;
    UC(ini_dev_grid(colors, stress, L, N, &s->sdev));
    UC(ini_dev_grid(colors, stress, L, N, &s->stdev));
    UC(ini_hst_grid(colors, stress, L, N, &s->hst));
    UC(grid_sampler_reset(s));
}

void grid_sampler_fin(GridSampler *s) {
    UC(fin_dev_grid(&s->sdev));
    UC(fin_dev_grid(&s->stdev));
    UC(fin_hst_grid(&s->hst));
    EFREE(s);
}

static void reset_dev_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) DzeroA(a[i], n);
}

static void reset_dev_grid(Grid *g) {
    long n = get_size(g);
    UC(                reset_dev_n(NFIELDS_P, n, g->p) );
    if (g->colors) UC( reset_dev_n(NFIELDS_C, n, g->c) );
    if (g->stress) UC( reset_dev_n(NFIELDS_S, n, g->s) );
}

void grid_sampler_reset(GridSampler *s) {
    s->nsteps = 0;
    UC(reset_dev_grid(&s->stdev));
}

static void add_data_to_grid(const GridSampleData *data, Grid *g) {
    long i, n;
    SampleDatum d;
    for (i = 0; i < data->n; ++i) {
        d = data->d[i];
        n = d.n;
        KL(sampler_add_kernel, (k_cnf(n)), (d, *g));
    }
}

static void space_avg(Grid *g) {
    long n = get_size(g);
    KL(sampler_space_avg_kernel, (k_cnf(n)), (*g));
}

static void add_to_grid(const Grid *s, Grid *st) {
    long n = get_size(s);
    KL(sampler_add_to_grid_kernel, (k_cnf(n)), (*s, *st));
}

void grid_sampler_add(const GridSampleData *data, GridSampler *s) {
    Grid *sg, *stg;
    sg  = &s->sdev;
    stg = &s->stdev;
    UC(reset_dev_grid(sg));
    UC(add_data_to_grid(data, sg));
    UC(space_avg(sg));
    UC(add_to_grid(sg, stg));
    s->nsteps ++;
}

static void time_avg(int nsteps, Grid *g) {
    long n = get_size(g);
    KL(sampler_time_avg_kernel, (k_cnf(n)), (nsteps, *g));
}

static void download_n(int na, long n, float * const dev[], float *hst[]) {
   for (int i = 0; i < na; ++i) aD2H(hst[i], dev[i], n);
}

static void download(const Grid *dev, Grid *hst) {
    long n = get_size(dev);
    UC(                  download_n(NFIELDS_P, n, dev->p, hst->p) );
    if (dev->colors) UC( download_n(NFIELDS_C, n, dev->c, hst->c) );
    if (dev->stress) UC( download_n(NFIELDS_S, n, dev->s, hst->s) );
}

static void build_desc(int ncmp, const char * const name[], const float * const data[],
                       int *j, const char *dst_name[], const float *dst_data[]) {
    for (int i = 0; i < ncmp; ++i) {
        dst_data[*j] = data[i];
        dst_name[*j] = name[i];
        ++(*j);
    }
}

static void dump(MPI_Comm cart, const char *dir, long id, const Grid *g) {
    char path[FILENAME_MAX] = {'\0'};
    int ncmp;
    const float *data[TOT_NFIELDS];
    const char *names[TOT_NFIELDS];

    ncmp = 0;
    build_desc               (NFIELDS_P, names_p, g->p,      &ncmp, names, data);
    if (g->colors) build_desc(NFIELDS_C, names_c, g->c,      &ncmp, names, data);
    if (g->stress) build_desc(NFIELDS_S, names_s, g->s,      &ncmp, names, data);

    snprintf(path, FILENAME_MAX, DUMP_BASE "/%s/%04ld.raw", dir, id);

    UC(grid_write(g->N, g->L, cart, path, ncmp, data, names));
}

void grid_sampler_dump(MPI_Comm cart, const char *dir, long id, GridSampler *s) {
    UC(time_avg(s->nsteps, &s->stdev));
    UC(download(&s->stdev, &s->hst));
    dSync();
    UC(dump(cart, dir, id, &s->hst));
}
void grid_sampler_data_ini(GridSampleData **s) {
    EMALLOC(1, s);
    grid_sampler_data_reset(*s);
}

void grid_sampler_data_fin(GridSampleData  *s) {
    EFREE(s);
}

void grid_sampler_data_reset(GridSampleData *s) {
    s->n = 0;
}

void grid_sampler_data_push(long n, const Particle *pp, const int *cc, const float *ss, GridSampleData *s) {
    int id;
    SampleDatum *d;
    id = s->n++;

    if (s->n >= MAX_N_DATA)
        ERR("Too many data: %d/%d", s->n / MAX_N_DATA);

    d = s->d + id;
    d->n = n;
    d->pp = pp;
    d->cc = cc;
    d->ss = ss;
}






#include <stdio.h>






struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);

struct GenColor;
struct Particle;
struct Coords;
struct Config;

void inter_color_ini(GenColor**);
void inter_color_fin(GenColor*);

void inter_color_set_drop(float R, GenColor*);
void inter_color_set_uniform(GenColor*);

void inter_color_set_conf(const Config*, GenColor*);

void inter_color_apply_hst(const Coords*, const GenColor*, int n, const Particle *pp,      int *cc);
void inter_color_apply_dev(const Coords*, const GenColor*, int n, const Particle *pp,      int *cc);


void inter_color_set_conf(const Config *cfg, GenColor *gc) {
    const char *kind;
    UC(conf_lookup_string(cfg, "iccolor.kind", &kind));

    if      (same_str(kind, "uniform")) {
        inter_color_set_uniform(gc);
    }
    else if (same_str(kind, "drop")) {
        float R;
        UC(conf_lookup_float(cfg, "iccolor.R", &R));
        inter_color_set_drop(R, gc);
    }
    else if (same_str(kind, "none")) {
        return;
    }
    else {
        ERR("unrecognised kind <%s>", kind);
    }
}



#include <stdio.h>




#ifndef UDX_GRID_INTER_DEF_COMMON
#define UDX_GRID_INTER_DEF_COMMON
#endif

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())









#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif

#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif











struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);


enum {
    COLOR_NONE,
    COLOR_DROP,
    COLOR_UNIF
};

struct GenColor {
    int kind;
    float R;
};
static void set_color_unif(int n,      int* cc) {
    int i;
    for (i = 0; i < n; i++) cc[i] = BLUE_COLOR;
}
static void set_color_drop(float rad, const Coords *coords, int n, const Particle *pp,      int *cc) {
    enum {X, Y, Z};
    int lx, ly, lz;
    int i, r, g;
    bool inside;
    float x, y, z;
    float x0, y0, z0;
    Particle p;

    lx = xdomain(coords);
    ly = ydomain(coords);
    lz = zdomain(coords);

    x0 = 0.5*lx; y0 = 0.5*ly; z0 = 0.5*lz;
    enum {R = RED_COLOR, G = BLUE_COLOR};
    for (i = r = g = 0; i < n; i++) {
        p = pp[i];
        x = p.r[X]; y = p.r[Y]; z = p.r[Z];
        x = xl2xg(coords, x);
        y = yl2yg(coords, y);
        z = zl2zg(coords, z);
        x -= x0; y -= y0; z -= z0;
        inside = x*x + y*y + z*z < rad*rad;
        if (inside) {cc[i] = R; r++;}
        else        {cc[i] = G; g++;}
    }
    msg_print("color scheme: Bangladesh");
    msg_print("red/green : %d/%d", r, g);
}
void inter_color_ini(GenColor **c) {
    GenColor *gc;
    EMALLOC(1, c);
    gc = *c;
    gc->kind = COLOR_NONE;
}

void inter_color_fin(GenColor *c) {
    EFREE(c);
}

void inter_color_set_drop(float R, GenColor *c) {
    c->kind = COLOR_DROP;
    c->R = R;
}

void inter_color_set_uniform(GenColor *c) {
    c->kind = COLOR_UNIF;
}

void inter_color_apply_hst(const Coords *coords, const GenColor *gc, int n, const Particle *pp,      int *cc) {
    switch (gc->kind) {
    case COLOR_UNIF:
        set_color_unif(n,      cc);
        break;
    case COLOR_DROP:
        set_color_drop(gc->R, coords, n, pp,      cc);
        break;
    case COLOR_NONE:
        break;
    default:
        ERR("Unrecognised kind <%d>", gc->kind);
        break;
    };
}

void inter_color_apply_dev(const Coords *coords, const GenColor *gc, int n, const Particle *pp,      int *cc) {
    int *cc_hst;
    Particle *pp_hst;

    if (gc->kind == COLOR_NONE) return;

    EMALLOC(n, &cc_hst);
    EMALLOC(n, &pp_hst);

    cD2H(pp_hst, pp, n);

    UC(inter_color_apply_hst(coords, gc, n, pp_hst,      cc_hst));

    cH2D(cc, cc_hst, n);

    EFREE(pp_hst);
    EFREE(cc_hst);
}












#include <stdio.h>





#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif

namespace cc {
void check(int rc, const char *file, int line);
}


namespace cc {
void check(int rc, const char *file, int line) {
    if (rc != 0) {
        error_signal_cuda(file, line, d::emsg());
        error_report();
        error_abort();
    }
}
}



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>




static int         err_line;
static const char *err_file;
static char        err_msg [BUFSIZ];
static char        err_kind[BUFSIZ];
static int         err_status = 0;


enum {MAX_TRACE = 128};
static char stack      [ MAX_TRACE ][ BUFSIZ ];
static char back_trace [ MAX_TRACE  * BUFSIZ ];
static int  stack_sz = 0;


void error_stack_pop() {
    --stack_sz;
    assert (stack_sz >= 0);
}

static void check_stack_overflow() {
    if (stack_sz >= MAX_TRACE) {
        error_signal(__FILE__, __LINE__, "stack overflow (%d / %d)", stack_sz, MAX_TRACE);
        error_report();
        error_abort();
    }
}

void error_stack_push(const char *file, int line) {
    sprintf(stack[stack_sz], ": %s: %d:", file, line);
    ++stack_sz;

    check_stack_overflow();
}

static void stack_dump() {
    int i, nchar;
    char *bt = back_trace;

    if (stack_sz) {
        nchar = sprintf(bt, "backtrace:\n");
        bt += nchar;
    }
    for (i = stack_sz-1; i >= 0; --i) {
        nchar = sprintf(bt, "%s\n", stack[i]);
        assert(nchar >= 0);
        bt += nchar;
    }
}

static void set_err_loc(const char *file, int line) {
    err_line = line;
    err_file = file;
    memset(err_msg, 0, sizeof(err_msg));
}

void error_signal(const char *file, int line, const char *fmt, ...) {
    set_err_loc(file, line);
    err_status = 1;
    strcpy(err_kind, "udx");

    va_list ap;
    va_start(ap, fmt);
    vsprintf(err_msg, fmt, ap);
    va_end(ap);
}

void error_signal_cuda(const char *file, int line, const char *msg) {
    set_err_loc(file, line);
    err_status = 1;
    strcpy(err_kind, "cuda");
    strcpy(err_msg, msg);
}

void error_signal_mpi(const char *file, int line, const char *msg) {
    set_err_loc(file, line);
    err_status = 1;
    strcpy(err_kind, "mpi");
    strcpy(err_msg, msg);
}

bool error_get() {return err_status;}
void error_report() {
    if (err_status) {
        stack_dump();
        msg_print("%s: %d: %s error: %s\n%s",
                  err_file, err_line, err_kind, err_msg, back_trace);
    }
}

void error_abort() { exit(1); }

void error_print_stack() {
    stack_dump();
    msg_print("%s", back_trace);
}



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

void os_mkdir(const char* path);
long os_time();
void os_srand(long int seedval);
double os_drand();
void os_sleep(unsigned int seconds);
void os_malloc_stats();








static double Bytes2MBytes(size_t sz) {
    return (double) sz / (double) (1 << 20);
}

void emalloc(size_t size,      void **data) {
    *data = malloc(size);
    if (NULL == *data) {
        msg_print("standard library: %s", strerror(errno));
        os_malloc_stats();
        ERR("Failed to allocate array of size %g MB\n", Bytes2MBytes(size));
    }
}

void efree(void *ptr) {
    free(ptr);
    ptr = NULL;
}

void *ememcpy(void *dest, const void *src, size_t n) {
    if (NULL == dest) ERR("NULL == dest");
    if (NULL == src) ERR("NULL == src");
    return memcpy(dest, src, n);
}

void efopen(const char *fname, const char *mode,      FILE **pq) {
    FILE *q;
    q = fopen(fname, mode);
    if (NULL == q)
        ERR("Could not open file <%s> with mode <%s>", fname, mode);
    *pq = q;
}

void efclose(FILE *f) {
    if (fclose(f) != 0) ERR("Failed to close");
}

void efread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t nmemb0;
    nmemb0 = fread(ptr, size, nmemb, stream);
    if (nmemb != nmemb0)
        ERR("`fread` failed: nmemb0=%ld   !=    nmemb=%lds, size=%ld",
            nmemb0, nmemb, size);
}

void efwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t nmemb0;
    nmemb0 = fwrite(ptr, size, nmemb, stream);
    if (nmemb != nmemb0)
        ERR("`fwrite` failed: nmemb0=%ld   !=    nmemb=%lds, size=%ld",
            nmemb0, nmemb, size);
}

void efgets(char *s, int size, FILE *stream) {
    char *p;
    p = fgets(s, size, stream);
    if (p == NULL)
        ERR("`fgets` failed: size=%d", size);
}

bool same_str(const char *a, const char *b) {
    return 0 == strcmp(a, b);
}

const char* get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}



#include <stdio.h>
#include <mpi.h>








namespace mpicheck {
void check(int code, const char *file, int line) {
    char msg[BUFSIZ], msg0[BUFSIZ];
    int n;
    if (m::is_success(code)) return;
    m::Error_string(code,      msg0, &n);
    sprintf(msg, "%s (%d)", msg0, code);
    error_signal_mpi(file, line, msg);
    error_report();
    error_abort();
}
}



#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>



static int rank;

void msg_ini(int rnk) { rank = rnk; }

static char last_eq(const char *s, char c) {
    int n;
    n = strlen(s);
    return n > 0 && s[n - 1] == c;
}
static FILE* open(const char *path) {
    static int fst = 1;
    FILE *f;
    if (fst) {
        fst = 0;
        f = fopen(path, "w");
        if (f == NULL) {
            fprintf(stderr, "%s:%d: fail to write: '%s'\n", __FILE__, __LINE__, path);
            exit(2);
        }
    } else {
        f = fopen(path, "a");
        if (f == NULL) {
            fprintf(stderr, "%s:%d: fail to append: '%s'\n", __FILE__, __LINE__, path);
            exit(2);
        }
    }
    return f;
}

static bool is_master(int r) {return r == 0;}
static void print0(FILE *f, const char *s) {
    if (last_eq(s, '\n')) fprintf(f, "%s",   s);
    else                  fprintf(f, "%s\n", s);
}
static void print(const char *msg, FILE *f) {
    print0(f, msg);
    if (is_master(rank))
        print0(stderr, msg);
}

void msg_print(const char *fmt, ...) {
    char msg[BUFSIZ], name[FILENAME_MAX];
    va_list ap;
    FILE *f;
    snprintf(name, FILENAME_MAX - 1, ".%03d", rank);

    va_start(ap, fmt);
    vsnprintf(msg, BUFSIZ - 1, fmt, ap);
    va_end(ap);

    f = open(name);
    print(msg, f);
    if (fclose(f) != 0) {
        fprintf(stderr, "%s:%d: fail to close: '%s'\n",
                __FILE__, __LINE__, name);
        exit(2);
    }
}

#include <stdio.h>
#include <stdlib.h>

#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <malloc.h>






static void safe_mkdir(const char *path)  {
    mode_t mode;
    int rc, ok;
    mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
    rc = mkdir(path, mode);
    ok = (rc == 0 || errno == EEXIST);
    if (!ok) {
        msg_print("os_mkdir: cannot create directory ‘%s’", path);
        ERR("errno: %d\n", errno);
        exit(1);
    }
}

#define END '\0'
#define SEP '/'

void os_mkdir(const char *path0) {
    char path[FILENAME_MAX] = {'\0'};
    unsigned int len;
    char *p;

    len = snprintf(path, sizeof(path), "%s", path0);

    if (len >= sizeof(path)) ERR("Path is too long: %s", path0);
    if (len == 0) ERR("empty path");

    if (path[len-1] == SEP) path[len-1] = END;

    for (p = path; *p != END; ++p) {
        if (*p == SEP) {
            *p = END;
            UC(safe_mkdir(path));
            *p = SEP;
        }
    }
    UC(safe_mkdir(path));
}

#undef SEP
#undef END

long os_time() { return time(NULL); }
void os_srand(long int seedval) { srand48(seedval); }
double os_drand() { return drand48(); }
void os_sleep(unsigned int seconds) { sleep(seconds); }
void os_malloc_stats() { malloc_stats(); }





#include <ctype.h>



int string_nword(const char *s);

int string_nword(const char *s) {
    enum {OUT, IN};
    const char *s0;
    char c;
    int state, n;
    s0 = s;
    state = OUT;
    n = 0;
    while ((c = *s++) != '\0') {
        if (!(isalnum(c) || isspace(c)))
            ERR("illegal character '%c' in '%s'", c, s0);
        else if (isspace(c))
            state = OUT;
        else if (state == OUT) {
            state = IN;
            n++;
        }
    }
    return n;
}











void linal_inv3x3(const float m0[6],      float minv0[6]);


void linal_inv3x3(const float *m,      float *r) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    double eps = 1e-8;

    double xx, yy, zz, xy, xz, yz;
    double yz2, xz2, xy2;
    double mx, my, mz;
    double d, i;

    xx = m[XX]; yy = m[YY]; zz = m[ZZ];
    xy = m[XY]; xz = m[XZ];
    yz = m[YZ];
    yz2 = yz*yz; xz2 = xz*xz; xy2 = xy*xy;

    mx  = yy*zz-yz2;
    my  = xy*zz-xz*yz;
    mz  = xy*yz-xz*yy;
    d   = mz*xz-my*xy+mx*xx;
    if (d < eps && d > -eps)
        ERR("Matrix is singular, determinant %g", d);
    i   = 1/d;

    r[XX] =  mx*i;
    r[XY] = -my*i;
    r[XZ] =  mz*i;
    r[YY] = i*(xx*zz-xz2);
    r[YZ] = i*(xy*xz-xx*yz);
    r[ZZ] = i*(xx*yy-xy2);
}





#include <limits>
#include <stdio.h>
#include <stdint.h>







struct RNDunif;
void rnd_ini(int x, int y, int z, int c,      RNDunif **r);
void rnd_fin(RNDunif *r);
float rnd_get(RNDunif *r);
typedef uint32_t integer;
struct RNDunif { integer x, y, z, c; };

void rnd_ini(int x, int y, int z, int c,      RNDunif **pq) {
    RNDunif *q;
    EMALLOC(1, &q);
    *pq = q;
    q->x = x; q->y = y; q->z = z; q->c = c;
}

void rnd_fin(RNDunif *r) { EFREE(r); }

static integer get_int(RNDunif *r) {
    uint64_t t, a = 698769069ULL;
    r->x = 69069 * r->x + 12345;
    r->y ^= ( r->y << 13 );
    r->y ^= ( r->y >> 17 );
    r->y ^= ( r->y << 5 );
    t = a * r->z + r->c;
    r->c = ( t >> 32 );
    return r->x + r->y + ( r->z = t );
}

float rnd_get(RNDunif *r) {
    return get_int(r) / float( std::numeric_limits<integer>::max() );
}





#include <math.h>

namespace tri_hst {
double kahan_area0(double a, double b, double c);
double kahan_area(const double a[3], const double b[3], const double c[3]);
void   ac_bc_cross(const double a[3], const double b[3], const double c[3],      double r[3]);
double shewchuk_area(const double a[3], const double b[3], const double c[3]);
double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]);
void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                   double *x, double *y);
}

#define _I_
#define _S_ static
#define BEGIN namespace tri_hst {
#define END }

BEGIN

_S_ void diff(const double *a, const double *b,      double  *c) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
_S_ double vabs(double *a) {
    enum {X, Y, Z};
    double r;
    r = a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z];
    return sqrt(r);
}
_S_ void swap(double *a, double *b) {
    double t;
    t = *a; *a = *b; *b = t;
}
_S_ int less(double *a, double *b) { return (*a) < (*b); }
_S_ void  sort3(double *a, double *b, double *c) {
    if (less(c, b)) swap(c, b);
    if (less(b, a)) swap(b, a);
    if (less(c, b)) swap(c, b);
}

_I_ double kahan_area0(double a, double b, double c) {
    sort3(&c, &b, &a);
    return sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))/4;
}

_I_ double kahan_area(const double r0[3], const double r1[3], const double r2[3]) {
    double r01[3], r12[3], r20[3], a, b, c;
    diff(r0, r1,      r01);
    diff(r1, r2,      r12);
    diff(r2, r0,      r20);
    a = vabs(r01); b = vabs(r12); c = vabs(r20);
    return kahan_area0(a, b, c);
}

_S_ double orient2d_xy(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = X, Q = Y};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_S_ double orient2d_zx(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = Z, Q = X};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_S_ double orient2d_yz(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = Y, Q = Z};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_I_ double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]) {
    enum {X, Y, Z};
    double ax, ay, az, bx, by, bz, cx, cy, cz;

    ax = a[X] - d[X]; ay = a[Y] - d[Y]; az = a[Z] - d[Z];
    bx = b[X] - d[X]; by = b[Y] - d[Y]; bz = b[Z] - d[Z];
    cx = c[X] - d[X]; cy = c[Y] - d[Y]; cz = c[Z] - d[Z];

    return ax*(by*cz - bz*cy) + bx*(cy*az - cz*ay) + cx*(ay*bz - az*by);
}

_I_ void ac_bc_cross(const double a[3], const double b[3], const double c[3],      double r[3]) {
    enum {X, Y, Z};
    r[X] = orient2d_yz(a, b, c);
    r[Y] = orient2d_zx(a, b, c);
    r[Z] = orient2d_xy(a, b, c);
}

_I_ double shewchuk_area(const double a[3], const double b[3], const double c[3]) {
    double x, y, z;
    x = orient2d_yz(a, b, c);
    y = orient2d_zx(a, b, c);
    z = orient2d_xy(a, b, c);
    return sqrt(x*x + y*y + z*z)/2;
}

_I_ void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                       double *px, double *py) {
    enum {X, Y, Z};
    double bcx, bcy, bcz, bc;
    double k[3], l[3];

    ac_bc_cross(b, c, a,      k);
    ac_bc_cross(c, b, d,      l);

    bcx = b[X] - c[X];
    bcy = b[Y] - c[Y];
    bcz = b[Z] - c[Z];
    bc = sqrt(bcx*bcx + bcy*bcy + bcz*bcz);

    *py = -orient3d(a, b, c, d)*bc;
    *px = k[X]*l[X] + k[Y]*l[Y] + k[Z]*l[Z];
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END





#include <stdio.h>










struct Tform_v { float o[3], s[3]; };
struct Tform;
struct Tform_v;

void tform_ini(Tform**);
void tform_fin(Tform*);

void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3],      Tform*);
void tform_chain(const Tform *A, const Tform *B,      Tform *C);
void tform_to_grid(const float lo[3], const float hi[3], const int n[3],      Tform*);
void tform_from_grid(const float a0[3], const float b0[3], const int n[3],      Tform*);
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3],
                     Tform*);


void tform_convert(const Tform*, const float a0[3],      float a1[3]);

void tform_to_view(const Tform*,      Tform_v*);

void tform_log(Tform*);
void tform_dump(Tform*, FILE*);

struct Tform   { float o[3], s[3]; };
void tform_ini(Tform **pq) { EMALLOC(1, pq); }
void tform_fin(Tform *q)   { EFREE(q);       }

static void report(const float a0[3], const float a1[3],   const float b0[3], const float b1[3]) {
    enum {X, Y, Z};
    msg_print("a0: %g %g %g", a0[X], a0[Y], a0[Z]);
    msg_print("a1: %g %g %g", a1[X], a1[Y], a1[Z]);
    msg_print("b0: %g %g %g", b0[X], b0[Y], b0[Z]);
    msg_print("b1: %g %g %g", b1[X], b1[Y], b1[Z]);
}
enum {TFORM_OK, TFORM_ZERO};
static int os(float a0, float a1,   float b0, float b1,      float *o, float *s) {

    const float eps = 1e-8;
    float d;
    d = b0 - a0;
    if (-eps < d && d < eps) return TFORM_ZERO;
    *o = (a1*b0-a0*b1)/(b0-a0);
    *s = (b1-a1)      /(b0-a0);
    return TFORM_OK;
}
void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3],      Tform* t) {
    int r;
    enum {X, Y, Z};
    r = os(a0[X], a1[X], b0[X], b1[X],      &t->o[X], &t->s[X]);
    if (r != TFORM_OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }

    r = os(a0[Y], a1[Y], b0[Y], b1[Y],      &t->o[Y], &t->s[Y]);
    if (r != TFORM_OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }

    r = os(a0[Z], a1[Z], b0[Z], b1[Z],      &t->o[Z], &t->s[Z]);
    if (r != TFORM_OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }
}

static int smallp(const float s[3]) {
    enum {X, Y, Z};
    const float eps = 1e-12;
    int cx, cy, cz;
    cx = -eps < s[X] && s[X] < eps;
    cy = -eps < s[Y] && s[Y] < eps;
    cz = -eps < s[Z] && s[Z] < eps;
    return cx && cy && cz;
}
void tform_convert(const Tform *t, const float a0[3],      float a1[3]) {
    enum {X, Y, Z};
    const float *o, *s;
    o = t->o; s = t->s;
    if (smallp(s))
        ERR("tform_convert failed: s = [%g %g %g]\n", s[X], s[Y], s[Z]);
    a1[X] = s[X]*a0[X] + o[X];
    a1[Y] = s[Y]*a0[Y] + o[Y];
    a1[Z] = s[Z]*a0[Z] + o[Z];
}

void tform_chain(const Tform *t1, const Tform *t2,      Tform *t) {
    float a0[3] = {0, 0, 0};
    float b0[3] = {1, 1, 1};
    float a1[3], b1[3], a2[3], b2[3];

    tform_convert(t1, a0,      a1);
    tform_convert(t1, b0,      b1);

    tform_convert(t2, a1,      a2);
    tform_convert(t2, b1,      b2);

    UC(tform_vector(a0, a2, b0, b2, t));
}

void tform_log(Tform *t) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    msg_print("tform: o = [%g %g %g]", o[X], o[Y], o[Z]);
    msg_print("tform: s = [%g %g %g]", s[X], s[Y], s[Z]);
}

void tform_dump(Tform *t, FILE *f) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    fprintf(f, "%16.10e %16.10e %16.10e\n", o[X], o[Y], o[Z]);
    fprintf(f, "%16.10e %16.10e %16.10e\n", s[X], s[Y], s[Z]);
}

void tform_to_grid(const float a0[3], const float b0[3], const int n[3],      Tform *t) {
    enum {X, Y, Z};
    float a1[3], b1[3];
    a1[X] = a1[Y] = a1[Z] = -0.5;
    b1[X] = n[X] - 0.5; b1[Y] = n[Y] - 0.5; b1[Z] = n[Z] - 0.5;
    UC(tform_vector(a0, a1,   b0, b1,      t));
}
void tform_from_grid(const float a0[3], const float b0[3], const int n[3],      Tform *t) {
    enum {X, Y, Z};
    float a1[3], b1[3];
    a1[X] = a1[Y] = a1[Z] = -0.5;
    b1[X] = n[X] - 0.5; b1[Y] = n[Y] - 0.5; b1[Z] = n[Z] - 0.5;
    UC(tform_vector(a1, a0,   b1, b0,      t));
}
enum {LH_OK, LH_BAD};
static int assert_lh(const float lo[3], const float hi[3]) {
    enum {X, Y, Z};
    int c;
    c = lo[X] < hi[X] && lo[X] < hi[X] && lo[X] < hi[X];
    return c ? LH_OK : LH_BAD;
}
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3],
                     Tform *t) {
    enum {X, Y, Z};

    Tform *f2g, *g2t;
    if (assert_lh(f_lo, f_hi) == LH_BAD)
        ERR("wrong f_[lo|hi]: [%g %g %g] [%g %g %g]",
            f_lo[X], f_lo[Y], f_lo[X], f_hi[X], f_hi[Y], f_hi[Z]);

    if (assert_lh(t_lo, t_hi) == LH_BAD)
        ERR("wrong f_[lo|hi]: [%g %g %g] [%g %g %g]",
            t_lo[X], t_lo[Y], t_lo[X], t_hi[X], t_hi[Y], t_hi[Z]);

    UC(tform_ini(&f2g));
    UC(tform_ini(&g2t));

    UC(tform_from_grid(f_lo, f_hi, f_n,      f2g));
    UC(tform_to_grid  (t_lo, t_hi, t_n,      g2t));
    UC(tform_chain(f2g, g2t,      t));

    tform_fin(g2t);
    tform_fin(f2g);
}

void tform_to_view(const Tform *t, Tform_v *v) {
    enum {X, Y, Z};
    if (smallp(t->s))
        ERR("tform_to_view failed: t->s = [%g %g %g]\n", t->s[X], t->s[Y], t->s[Z]);

    v->o[X] = t->o[X];
    v->o[Y] = t->o[Y];
    v->o[Z] = t->o[Z];

    v->s[X] = t->s[X];
    v->s[Y] = t->s[Y];
    v->s[Z] = t->s[Z];
}

















#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>







struct Particle;
struct Vectors;
struct Coords;

void vectors_float_ini(int n, const float*,      Vectors**);
void vectors_postions_ini(int n, const Particle*,      Vectors**);
void vectors_postions_edge_ini  (const Coords*, int n, const Particle*,      Vectors**);
void vectors_postions_center_ini(const Coords*, int n, const Particle*,      Vectors**);
void vectors_velocities_ini(int n, const Particle*,      Vectors**);
void vectors_zero_ini(int n,      Vectors**);

void vectors_fin(Vectors*);
void vectors_get(const Vectors*, int i,      float r[3]);

struct Particle;
struct Scalars;
struct Vectors;

void scalars_float_ini(int n, const float*,      Scalars**);
void scalars_double_ini(int n, const double*,      Scalars**);
void scalars_vectors_ini(int n, const Vectors*, int dim,      Scalars**);

void scalars_zero_ini(int n,      Scalars**);
void scalars_one_ini(int n,      Scalars**);

void scalars_fin(Scalars*);
double scalars_get(const Scalars*, int i);
enum {MAGIC = 123};
struct Scalars {
    int type;
    int n;
    union {
        const float  *ff;
        const double *dd;
        const Vectors *vec;
    } D;
    int stamp;
};

static double float_get (const Scalars*, int i);
static double double_get(const Scalars*, int i);

static double vecx_get(const Scalars*, int i);
static double vecy_get(const Scalars*, int i);
static double vecz_get(const Scalars*, int i);

static double zero_get(const Scalars*, int i);
static double one_get(const Scalars*, int i);

enum {SCALAR_FLOAT, SCALAR_DOUBLE, SCALAR_VECX, SCALAR_VECY, SCALAR_VECZ, SCALAR_ZERO, SCALAR_ONE};
static double (*scalar_get_table[])(const Scalars*, int i) =
{ float_get,  double_get, vecx_get, vecy_get, vecz_get, zero_get, one_get};
void scalars_float_ini(int n, const float *rr,      Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = SCALAR_FLOAT; q->n = n; q->D.ff = rr;
    q->stamp = MAGIC;
    *pq = q;
}

void scalars_double_ini(int n, const double *rr,      Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = SCALAR_DOUBLE; q->n = n; q->D.dd = rr;
    q->stamp = MAGIC;
    *pq = q;
}

void scalars_vectors_ini(int n, const Vectors *vec, int dim,      Scalars **pq) {
    enum {X, Y, Z};
    Scalars *q;
    EMALLOC(1, &q);
    switch (dim) {
    case X: q->type = SCALAR_VECX; break;
    case Y: q->type = SCALAR_VECY; break;
    case Z: q->type = SCALAR_VECZ; break;
    default: ERR("wrong dim=%d", dim);
    }
    q->stamp = MAGIC;
    q->n = n; q->D.vec = vec;
    *pq =q;
}

void scalars_zero_ini(int n,      Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = SCALAR_ZERO; q->n = n;
    q->stamp = MAGIC;
    *pq = q;
}


void scalars_one_ini(int n,      Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = SCALAR_ONE; q->n = n;
    q->stamp = MAGIC;
    *pq = q;
}

void scalars_fin(Scalars *q) { EFREE(q); }

static double float_get(const Scalars *q, int i) { return q->D.ff[i]; }
static double double_get(const Scalars *q, int i) { return q->D.dd[i]; }
static double vecx_get(const Scalars *q, int i)   {
    enum {X};
    float r[3];
    vectors_get(q->D.vec, i,      r);
    return r[X];
}
static double vecy_get(const Scalars *q, int i)   {
    enum {X, Y};
    float r[3];
    vectors_get(q->D.vec, i,      r);
    return r[Y];
}
static double vecz_get(const Scalars *q, int i)   {
    enum {X, Y, Z};
    float r[3];
    vectors_get(q->D.vec, i,      r);
    return r[Z];
}

static double zero_get(const Scalars*, int) { return 0; }
static double one_get(const Scalars*, int) { return 1; }
double scalars_get(const Scalars *q, int i) {
    int n;
    if (q->stamp != MAGIC)
        ERR("scalar is not initilized");
    n = q->n;
    if (i >= n) ERR("i = %d    >=   n = %d", i, n);
    if (i < 0)  ERR("i = %d    < 0", i);
    return scalar_get_table[q->type](q, i);
}




#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>






#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif
struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);
struct Tform;
struct Tform_v;

void tform_ini(Tform**);
void tform_fin(Tform*);

void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3],      Tform*);
void tform_chain(const Tform *A, const Tform *B,      Tform *C);
void tform_to_grid(const float lo[3], const float hi[3], const int n[3],      Tform*);
void tform_from_grid(const float a0[3], const float b0[3], const int n[3],      Tform*);
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3],
                     Tform*);


void tform_convert(const Tform*, const float a0[3],      float a1[3]);

void tform_to_view(const Tform*,      Tform_v*);

void tform_log(Tform*);
void tform_dump(Tform*, FILE*);

struct Particle;
struct Vectors;
struct Coords;

void vectors_float_ini(int n, const float*,      Vectors**);
void vectors_postions_ini(int n, const Particle*,      Vectors**);
void vectors_postions_edge_ini  (const Coords*, int n, const Particle*,      Vectors**);
void vectors_postions_center_ini(const Coords*, int n, const Particle*,      Vectors**);
void vectors_velocities_ini(int n, const Particle*,      Vectors**);
void vectors_zero_ini(int n,      Vectors**);

void vectors_fin(Vectors*);
void vectors_get(const Vectors*, int i,      float r[3]);
struct Vectors {
    int type;
    int n;
    union {
        const float *rr;
        const Particle *pp;
    } D;
    Tform *tform;
};

static void float_get(const Vectors*, int i, float r[3]);

static void positions_get(const Vectors*, int i, float r[3]);
static void positions_edge_get(const Vectors*, int i, float r[3]);
static void positions_center_get(const Vectors*, int i, float r[3]);
static void velocities_get(const Vectors*, int i, float r[3]);
static void zero_get(const Vectors*, int i, float r[3]);

enum {VECTOR_FLOAT, VECTOR_POSITIONS, VECTOR_POSITIONS_EDGE, VECTOR_POSITIONS_CENTER, VECTOR_VELOCITIES, VECTOR_ZERO};
static void (*vector_get_table[])(const Vectors*, int i, float r[3]) =
{ float_get,  positions_get, positions_edge_get, positions_center_get, velocities_get, zero_get };

typedef  void (*Local2Global)(const Coords *coords, float a[3],      float b[3]);
void vectors_float_ini(int n, const float *rr,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_FLOAT; q->n = n; q->D.rr = rr;
    *pq = q;
}

void vectors_postions_ini(int n, const Particle *pp,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_POSITIONS; q->n = n; q->D.pp = pp;
    *pq = q;
}

static void l2edge(const Coords *coords, float a[3],      float b[3]) {
    enum {X, Y, Z};
    b[X] = xl2xg(coords, a[X]);
    b[Y] = yl2yg(coords, a[Y]);
    b[Z] = zl2zg(coords, a[Z]);
}
static void l2center(const Coords *coords, float a[3],      float b[3]) {
    enum {X, Y, Z};
    b[X] = xl2xc(coords, a[X]);
    b[Y] = yl2yc(coords, a[Y]);
    b[Z] = zl2zc(coords, a[Z]);
}

static void edge_ini_tform(const Coords *coords, Local2Global local2global, Tform **pq) {
    enum {X, Y, Z};
    Tform *q;
    float a0[3], a1[3], b0[3], b1[3];
    a0[X] = a0[Y] = a0[Z] = 0;
    b0[X] = b0[Y] = b0[Z] = 1;
    local2global(coords, a0,      a1);
    local2global(coords, b0,      b1);
    tform_ini(&q);
    UC(tform_vector(a0, a1, b0, b1,      q));
    *pq = q;
}
void vectors_postions_edge_ini(const Coords *coords, int n, const Particle *pp,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_POSITIONS_EDGE; q->n = n; q->D.pp = pp;
    edge_ini_tform(coords, l2edge,      &q->tform);
    *pq = q;
}

void vectors_postions_center_ini(const Coords *coords, int n, const Particle *pp,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_POSITIONS_EDGE; q->n = n; q->D.pp = pp;
    edge_ini_tform(coords, l2center,      &q->tform);
    *pq = q;
}

void vectors_velocities_ini(int n, const Particle *pp,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_VELOCITIES; q->n = n; q->D.pp = pp;
    *pq = q;
}

void vectors_zero_ini(int n,      Vectors **pq) {
    Vectors *q;
    EMALLOC(1, &q);
    q->type = VECTOR_ZERO; q->n = n;
    *pq = q;
}

void vectors_fin(Vectors *q) {
    if (q->type == VECTOR_POSITIONS_EDGE ||
        q->type == VECTOR_POSITIONS_CENTER)
        tform_fin(q->tform);
    EFREE(q);
}

static void float_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const float *rr;
    rr = q->D.rr;
    r[X] = rr[3*i + 0];
    r[Y] = rr[3*i + 1];
    r[Z] = rr[3*i + 2];
}
static void positions_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    r[X] = pp[i].r[X];
    r[Y] = pp[i].r[Y];
    r[Z] = pp[i].r[Z];
}

static void shift_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    UC(tform_convert(q->tform, pp[i].r,      r));
}
static void positions_edge_get(const Vectors *q, int i, float r[3]) { shift_get(q, i, r); }
static void positions_center_get(const Vectors *q, int i, float r[3]) { shift_get(q, i, r); }
static void velocities_get(const Vectors *q, int i, float r[3]) {
    enum {X, Y, Z};
    const Particle *pp;
    pp = q->D.pp;
    r[X] = pp[i].v[X];
    r[Y] = pp[i].v[Y];
    r[Z] = pp[i].v[Z];
}
static void zero_get(const Vectors*, int, float r[3]) {
    enum {X, Y, Z};
    r[X] = r[Y] = r[Z] = 0;
}
void vectors_get(const Vectors *q, int i,      float r[3]) {
    int n;
    n = q->n;
    if (i >= n) ERR("i = %d    >=   n = %d", i, n);
    if (i < 0)  ERR("i = %d    < 0", i);
    vector_get_table[q->type](q, i, r);
}








#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE



#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())









#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
#endif

struct Particle;
struct Positioncp;
struct float4;

void convert_pp2rr_current (long n, const Particle *pp, Positioncp *rr);
void convert_pp2rr_previous(long n, const Particle *pp, Positioncp *rr);
void convert_pp2f4_pos(int n, const float *pp,      float4 *zpp);

__global__ void convert_pp2rr_current_kernel (long n, const Particle *pp, Positioncp *rr) {
    enum {X, Y, Z};
    const float *s;
    float *d;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    d = rr[i].rc;
    s = pp[i].r;

    d[X] = s[X];
    d[Y] = s[Y];
    d[Z] = s[Z];
}

__global__ void convert_pp2rr_previous_kernel(long n, const Particle *pp, Positioncp *rr) {
    enum {X, Y, Z};
    const float *s;
    float *d;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    d = rr[i].rp;
    s = pp[i].r;

    d[X] = s[X];
    d[Y] = s[Y];
    d[Z] = s[Z];
}

__global__ void convert_pp2f4_pos_kernel(int n, const float *pp,      float4 *zpp) {
    enum {X, Y, Z};
    int i;
    const float *r, *v;
    float x, y, z;
    float vx, vy, vz;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    r = &pp[6*i];
    v = &pp[6*i + 3];

    x =  r[X];  y = r[Y];  z = r[Z];
    vx = v[X]; vy = v[Y]; vz = v[Z];

    zpp[2*i]     = make_float4(x,   y,  z, 0);
    zpp[2*i + 1] = make_float4(vx, vy, vz, 0);
}

void convert_pp2rr_current (long n, const Particle *pp, Positioncp *rr) {
    KL(convert_pp2rr_current_kernel, (k_cnf(n)), (n, pp, rr));
}

void convert_pp2rr_previous(long n, const Particle *pp, Positioncp *rr) {
    KL(convert_pp2rr_previous_kernel, (k_cnf(n)), (n, pp, rr));
}

void convert_pp2f4_pos(int n, const float *pp,      float4 *zpp) {
    static_assert(sizeof(Particle) == 6 * sizeof(float),
                  "sizeof(Particle) != 6 * sizeof(float)");
    KL(convert_pp2f4_pos_kernel, (k_cnf(n)), (n, pp, zpp));
}




#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif

#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif

struct Force;


float force_stat_max(int n, const Force*);
static __device__ float sumsq(Force f) {
    enum {X, Y, Z};
    float x, y, z;
    x = f.f[X]; y = f.f[Y]; z = f.f[Z];
    return x*x + y*y + z*z;
}

struct SumSq {
    __device__ float operator()(Force f) const { return sumsq(f); }
};

float force_stat_max(int n, const Force *dev) {
    using namespace thrust;
    if (!d::is_device_pointer(dev))
        ERR("`dev` is not a device pointer");
    float init, m;
    SumSq           unary;
    maximum<float> binary;
    device_ptr<const Force> beg(dev), end(dev+n);
    init = 0;
    try {
        m = transform_reduce(beg, end, unary, init, binary);
    } catch (system_error e) {
        msg_print("thrust::transform_reduce failed");
        ERR("%s", e.what());
    }
    return sqrt(m);
}















#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdlib.h>
#include <stdio.h>


void edg_ini(int md, int nv,                     int *hx);
void edg_set(int md, int i, int j, int val,      int *hx,       int *hy);
int  edg_get(int md, int i, int j,         const int *hx, const int *hy);
int  edg_valid(int md, int i, int j,         const int *hx);

#include <stdlib.h>
#include <stdio.h>


void edg_ini(int md, int nv,      int *hx) {
    int n, i;
    n = md * nv;
    for (i = 0; i < n; i ++) hx[i] = -1;
}

void edg_set(int md, int f, int x, int y,       int *hx, int *hy) {
    int d, j;
    j = f*md; d = 0;
    while (hx[j] != -1) {
        j++; d++;
        if (d >= md) ERR("invalid edg_set call: %d/%d/%d/%d/%d", md, f, x, y, j);
    }
    hx[j] = x; hy[j] = y;
}

int edg_get(int md, int i, int x, const int *hx, const int *hy) {
    int d;
    i *= md; d = 0;
    while (hx[i] != x) {
        i++; d++;
        if (d >= md) ERR("invalid edg_get call: %d/%d/%d", md, i, x);
    }
    return hy[i];
}

int edg_valid(int md, int i, int x, const int *hx) {
    int d;
    i *= md; d = 0;
    while (hx[i] != x) {
        i++; d++;
        if (d >= md) return 0;
    }
    return 1;
}




#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>




#ifndef UDX_GRID_INTER_DEF_COMMON
#define UDX_GRID_INTER_DEF_COMMON
#endif


#if   defined(DEV_CUDA)


#elif defined(DEV_CPU)

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
#else
  #error DEV_* is undefined
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif
#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif

#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())









#ifndef UDX_CONTROL_SHFL_COMMON
#define UDX_CONTROL_SHFL_COMMON
#define _I_ static __device__

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

#undef _I_
#define _S_ static __device__
#define _I_ static __device__

template <typename T>
_I_ T warpReduceSum(T v)
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += shfl_down(v, offset);
    return v;
}

_I_ float2 warpReduceSum(float2 val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
    }
    return val;
}

_I_ float3 warpReduceSum(float3 val)
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

#endif

#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
#endif

#ifndef UDX_CONTROL_PARTICLE_DEV_COMMON
#define UDX_CONTROL_PARTICLE_DEV_COMMON
static __device__ float3 fetch_pos(int i, const Particle *pp) {
    enum {X, Y, Z};
    float3 r;
    r.x = pp[i].r[X]; r.y = pp[i].r[Y]; r.z = pp[i].r[Z];
    return r;
}

static __device__ float3 fetch_vel(int i, const Particle *pp) {
    enum {X, Y, Z};
    float3 v;
    v.x = pp[i].v[X]; v.y = pp[i].v[Y]; v.z = pp[i].v[Z];
    return v;
}
#endif

struct Particle;
struct float3;
void minmax(const Particle *pp, int nv, int nobj,      float3 *lo, float3 *hi);

#ifndef UDX_CONTROL_WARPSIZE_COMMON
#define UDX_CONTROL_WARPSIZE_COMMON
enum { WARPSIZE = 32 };
#endif
enum {
    BLOCK_SIZE = 128,
    NWARPS     = BLOCK_SIZE / WARPSIZE
};

static const float minmax_minv = -100000000.;
static const float minmax_maxv =  100000000.;

static __device__ float3 minmax_minf3(float3 a, float3 b) {
    return make_float3(min(a.x, b.x),
                       min(a.y, b.y),
                       min(a.z, b.z));
}

static __device__ float3 minmax_maxf3(float3 a, float3 b) {
    return make_float3(max(a.x, b.x),
                       max(a.y, b.y),
                       max(a.z, b.z));
}

static __device__ float3 minmax_warp_reduce_min(float3 val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x = min(val.x, shfl_down(val.x, offset));
        val.y = min(val.y, shfl_down(val.y, offset));
        val.z = min(val.z, shfl_down(val.z, offset));
    }
    return val;
}

static __device__ float3 minmax_warp_reduce_max(float3 val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x = max(val.x, shfl_down(val.x, offset));
        val.y = max(val.y, shfl_down(val.y, offset));
        val.z = max(val.z, shfl_down(val.z, offset));
    }
    return val;
}


__global__ void minmax_kernel(int nv, const Particle *pp, float3 *lo, float3 *hi) {
    enum {LO, HI, D};
    int i, objid, stride, pid, wid, lid;
    float3 r, lmin, lmax;
    __shared__ float3 lohi[D][NWARPS];

    stride = blockDim.x * gridDim.x;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    lid = i % warpSize;
    wid = i / warpSize;
    objid = blockIdx.y;

    lmin = make_float3(minmax_maxv, minmax_maxv, minmax_maxv);
    lmax = make_float3(minmax_minv, minmax_minv, minmax_minv);

    for (; i < nv; i += stride) {
        pid = objid * nv + i;
        r = fetch_pos(pid, pp);
        lmin = minmax_minf3(r, lmin);
        lmax = minmax_maxf3(r, lmax);
    }

    lmin = minmax_warp_reduce_min(lmin);
    lmax = minmax_warp_reduce_max(lmax);

    if (lid == 0) {
        lohi[LO][wid] = lmin;
        lohi[HI][wid] = lmax;
    }

    __syncthreads();

    if (wid == 0) {
        if (lid < NWARPS) {
            lmin = lohi[LO][lid];
            lmax = lohi[HI][lid];
        }
        lmin = minmax_warp_reduce_min(lmin);
        lmax = minmax_warp_reduce_max(lmax);

        if (lid == 0) {
            lo[objid] = lmin;
            hi[objid] = lmax;
        }
    }
}
void minmax(const Particle *pp, int nv, int nobj,      float3 *lo, float3 *hi) {
    dim3 thrd(BLOCK_SIZE, 1);
    dim3 blck(1, nobj);

    KL(minmax_kernel, (blck, thrd), (nv, pp, lo, hi));
}








#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>







struct KahanSum;

void kahan_sum_ini(     KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input);
double kahan_sum_get(const KahanSum*);
double kahan_sum_compensation(const KahanSum*);














struct KahanSum {
    double c, y, sum;
};
void kahan_sum_ini(     KahanSum **pq) {
    KahanSum *q;
    EMALLOC(1, &q);
    q->c = q->sum = 0;
    *pq = q;
}

void kahan_sum_fin(KahanSum *q) { EFREE(q); }
void kahan_sum_add(KahanSum *q, double input) {
    double t;
    double y, c, sum;
    y = q->y; c = q->c; sum = q->sum;

    y = input - c;
    t = sum + y;
    c = (t - sum) - y;
    sum = t;

    q->y = y; q->c = c; q->sum = sum;
}

double kahan_sum_get(const KahanSum *q) { return q->sum; }
double kahan_sum_compensation(const KahanSum *q) { return q->c; }




#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>
#include <string.h>






int string_nword(const char *s);

struct KeyList;

void key_list_ini(KeyList**);
void key_list_copy(KeyList*,      KeyList**);
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
enum {MAX_NK = 100 };
enum {STAMP_GOOD = 42 };

struct KeyList {
    int nk;
    int  ww[MAX_NK];
    int  mark[MAX_NK];
    char keys[MAX_NK][FILENAME_MAX];
    int stamp;
};
static char *key_cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
void key_list_ini(KeyList **pq) {
    KeyList *q;
    EMALLOC(1, &q);
    q->nk = 0;
    q->stamp = STAMP_GOOD;
    *pq = q;
}

void key_list_copy(KeyList *a,      KeyList **pq) {
    int i, nk;
    KeyList *q;
    EMALLOC(1, &q);
    nk = a->nk;
    for (i = 0; i < nk; i++) {
        q->ww[i] = a->ww[i];
        q->mark[i] = a->mark[i];
        key_cpy(q->keys[i], a->keys[i]);
    }
    q->stamp = STAMP_GOOD;
    q->nk = nk;
    *pq = q;
}

void key_list_fin(KeyList *q) {
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    EFREE(q);
}

void key_list_push(KeyList *q, const char *k) {
    int nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    if (nk == MAX_NK)
        ERR("nk=%d == MAX_NK=%d", nk, MAX_NK);
    key_cpy(q->keys[nk], k);
    q->ww[nk] = string_nword(k);
    q->mark[nk] = 0;
    q->nk = nk + 1;
}

int key_list_has(KeyList *q, const char *k) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (same_str(q->keys[i], k)) return 1;
    return 0;
}

int key_list_offset(KeyList *q, const char *k) {
    int i, nk, offset;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = offset = 0; i < nk; i++) {
        if (same_str(q->keys[i], k)) return offset;
        offset += q->ww[i];
    }
    ERR("key `%s` is not found", k);
    return -1;
}

int key_list_width(KeyList *q, const char *k) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++) {
        if (same_str(q->keys[i], k)) return q->ww[i];
    }
    ERR("key `%s` is not found", k);
    return -1;
}

int key_list_size(KeyList *q) {
    int i, nk, size;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = size = 0; i < nk; i++)
        size += q->ww[i];
    return size;
}

const char* key_list_name(KeyList *q, int i) {
    int n;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    n = key_list_size(q);
    if (i < 0) ERR("i=%d < 0", i);
    if (i >= n) ERR("i=%d >= n=%d", i, n);
    return q->keys[i];
}

void key_list_mark(KeyList *q, const char *k) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (same_str(q->keys[i], k)) {
            q->mark[i] = 1;
            return;
        }
    ERR("key `%s` is not found", k);
}

int  key_list_marked(KeyList *q) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        if (q->mark[i] == 0) return 0;
    return 1;
}

void  key_list_unmark(KeyList *q) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    for (i = 0; i < nk; i++)
        q->mark[i] = 0;
}

void key_list_log(KeyList *q) {
    int i, nk;
    if (q->stamp != STAMP_GOOD) ERR("KeyList is not initialized");
    nk = q->nk;
    msg_print("<key_list");
    for (i = 0; i < nk; i++)
        msg_print("'%s' %d", q->keys[i], q->mark[i]);
    msg_print(">");
}






#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>




#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())





















#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
#endif
#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#ifndef UDX_ALGO_DEVICE_API_COMMON
#define UDX_ALGO_DEVICE_API_COMMON
#endif
#endif


#if   defined(DEV_CUDA)


#elif defined(DEV_CPU)

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
#else
  #error DEV_* is undefined
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif

struct Scan;

void scan_apply(const int *input, int size,      int *output,       Scan*);
void scan_ini(int size,      Scan**);
void scan_fin(Scan*);


#define _I_ static __device__

template <typename T>
_I_ T scan_shfl_down(T val, unsigned int width) {
    return __shfl_down_sync(0xFFFFFFFF, val, width);
}

template <typename T>
_I_ T scan_shfl_up(T val, unsigned int width) {
    return __shfl_up_sync(0xFFFFFFFF, val, width);
}
template <typename T>
_I_ T scan_shfl(T val, unsigned int width) {
    return __shfl_sync(0xFFFFFFFF, val, width);
}

#undef _I_
__global__ void scan_compress(const int nentries, const int4 *const in, uchar4 *const out) {
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (4 * gid >= nentries)
        return;

    const int4 entry = in[gid];

    out[gid] = make_uchar4(entry.x, entry.y, entry.z, entry.w);
}

template <int NWARP>
__global__ void scan_breduce(const uint4 *vin, unsigned int *vout, int n) {

    const int wid = threadIdx.x/32;
    const int lid = threadIdx.x%32;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    uint4 val = make_uint4(0,0,0,0);

    __shared__ unsigned int shtmp[NWARP];

    if (tid < n) val = vin[tid];

    val.x = ((val.x & 0xFF000000) >> 24) + ((val.x & 0xFF0000) >> 16) + ((val.x & 0xFF00) >> 8) + (val.x & 0xFF);
    val.y = ((val.y & 0xFF000000) >> 24) + ((val.y & 0xFF0000) >> 16) + ((val.y & 0xFF00) >> 8) + (val.y & 0xFF);
    val.z = ((val.z & 0xFF000000) >> 24) + ((val.z & 0xFF0000) >> 16) + ((val.z & 0xFF00) >> 8) + (val.z & 0xFF);
    val.w = ((val.w & 0xFF000000) >> 24) + ((val.w & 0xFF0000) >> 16) + ((val.w & 0xFF00) >> 8) + (val.w & 0xFF);
    val.x += val.y + val.z + val.w;

#pragma unroll
    for(int i = 16; i > 0; i >>= 1)
        val.x += scan_shfl_down((int)val.x, i);

    if (0 == lid)
        shtmp[wid] = val.x;

    __syncthreads();
    if (0 == wid) {
        val.x = (lid < NWARP) ? shtmp[lid] : 0;

#pragma unroll
        for(int i = 16; i > 0; i >>= 1)
            val.x += scan_shfl_down((int)val.x, i);
    }
    if (0 == threadIdx.x) vout[blockIdx.x] = val.x;
    return;
}

template <int BDIM>
__global__ void scan_bexscan(unsigned int *v, int n) {

    extern __shared__ unsigned int shtmp[];

    for(int i = threadIdx.x; i < n; i += BDIM) shtmp[i] = v[i];

    int i = threadIdx.x;
    __syncthreads();
    for(; n >= BDIM; i += BDIM, n -= BDIM) {

        __syncthreads();
        if (i > 0 && 0 == threadIdx.x)
            shtmp[i] += shtmp[i-1];

        unsigned int a = 0;

#pragma unroll
        for(int j = 1; j < BDIM; j <<= 1) {
            a = 0;

            __syncthreads();
            if (threadIdx.x >= j) a = shtmp[i] + shtmp[i-j];

            __syncthreads();
            if (threadIdx.x >= j) shtmp[i] = a;
        }
        v[i] = shtmp[i];
    }
    if (threadIdx.x < n) {

        __syncthreads();
        if (i > 0 && 0 == threadIdx.x)
            shtmp[i] += shtmp[i-1];

        unsigned int a = 0;
        for(int j = 1; j < BDIM; j <<= 1) {
            a = 0;

            __syncthreads();
            if (threadIdx.x >= j) a = shtmp[i] + shtmp[i-j];

            __syncthreads();
            if (threadIdx.x >= j) shtmp[i] = a;
        }
        v[i] = shtmp[i];
    }
    return;
}

template <int NWARP>
__global__ void scan_gexscan(const uint4 *vin, unsigned int *offs, uint4 *vout, int n) {

    const int wid = threadIdx.x/32;
    const int lid = threadIdx.x%32;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    uint4 val[4];

    __shared__ unsigned int woff[NWARP];

    if (tid < n) val[0] = vin[tid];
    else         val[0] = make_uint4(0,0,0,0);


    val[3].y = (val[0].w & 0x000000FF);
    val[3].z = (val[0].w & 0x0000FF00) >>  8;
    val[3].w = (val[0].w & 0x00FF0000) >> 16;
    val[3].x = (val[0].w & 0xFF000000) >> 24;

    val[2].y = (val[0].z & 0x000000FF);
    val[2].z = (val[0].z & 0x0000FF00) >>  8;
    val[2].w = (val[0].z & 0x00FF0000) >> 16;
    val[2].x = (val[0].z & 0xFF000000) >> 24;

    val[1].y = (val[0].y & 0x000000FF);
    val[1].z = (val[0].y & 0x0000FF00) >>  8;
    val[1].w = (val[0].y & 0x00FF0000) >> 16;
    val[1].x = (val[0].y & 0xFF000000) >> 24;

    val[0].y = (val[0].x & 0x000000FF);
    val[0].z = (val[0].x & 0x0000FF00) >>  8;
    val[0].w = (val[0].x & 0x00FF0000) >> 16;
    val[0].x = (val[0].x & 0xFF000000) >> 24;

    uint4 tu4;
    tu4.x = val[0].x + val[0].y + val[0].z + val[0].w;
    tu4.y = val[1].x + val[1].y + val[1].z + val[1].w;
    tu4.z = val[2].x + val[2].y + val[2].z + val[2].w;
    tu4.w = val[3].x + val[3].y + val[3].z + val[3].w;

    unsigned int tmp = tu4.x + tu4.y + tu4.z + tu4.w;

    tu4.w = tmp;
#pragma unroll
    for(int i = 1; i < 32; i <<= 1)
        tu4.w += (lid >= i)*scan_shfl_up((int)tu4.w, i);

    if (lid == 31)
        if (wid < NWARP-1) woff[wid+1] = tu4.w;
        else               woff[0]     = (blockIdx.x > 0) ? offs[blockIdx.x-1] : 0;

    tu4.w -= tmp;

    __syncthreads();
    if (0 == wid) {
        tmp = (lid < NWARP) ? woff[lid] : 0;
#pragma unroll
        for(int i = 1; i < NWARP; i <<= 1)
            tmp += (lid >= i)*scan_shfl_up((int)tmp, i);

        if (lid < NWARP) woff[lid] = tmp;
    }
    __syncthreads();

    if (tid >= n) return;

    uint4 lps;
    lps.x = woff[wid] + tu4.w;
    lps.y = lps.x + tu4.x;
    lps.z = lps.y + tu4.y;
    lps.w = lps.z + tu4.z;

    val[0].x = lps.x;
    val[1].x = lps.y;
    val[2].x = lps.z;
    val[3].x = lps.w;

    val[0].y += val[0].x;
    val[1].y += val[1].x;
    val[2].y += val[2].x;
    val[3].y += val[3].x;

    val[0].z += val[0].y;
    val[1].z += val[1].y;
    val[2].z += val[2].y;
    val[3].z += val[3].y;

    val[0].w += val[0].z;
    val[1].w += val[1].z;
    val[2].w += val[2].z;
    val[3].w += val[3].z;

    vout += tid*4;

#pragma unroll
    for(int i = 0; i < 4; i++)
        vout[i] = val[i];
    return;
}

#if   defined(DEV_CUDA)
struct Scan {
    int size;
    unsigned int  *tmp;
    unsigned char *compressed;
};
enum { THREADS = 128 };
static void scan0(const unsigned char *input, int size,      uint *output,       uint *tmp) {
    int nblocks = ((size / 16) + THREADS - 1 ) / THREADS;
    KL(scan_breduce<THREADS/32>, (nblocks, THREADS                ), ((uint4 *)input, tmp, size / 16));
    KL(scan_bexscan<THREADS>   , (1, THREADS, nblocks*sizeof(uint)), (tmp, nblocks));
    KL(scan_gexscan<THREADS/32>, (nblocks, THREADS                ), ((uint4 *)input, tmp, (uint4 *)output, size / 16));
}

void scan_apply(const int *input, int size,      int *output,       Scan *w) {
    if (size > w->size)
        ERR(" size = %d > w->size = %d", size, w->size);
    if (!d::is_device_pointer(input))  ERR("`input`  is not a device pointer");
    if (!d::is_device_pointer(output)) ERR("`output` is not a device pointer");

    KL(scan_compress, (k_cnf(size)), (size, (const int4*) input,      (uchar4 *) w->compressed));
    scan0(w->compressed, size,      (uint*) output,       w->tmp);
}

void scan_ini(int size,      Scan **work) {
    Scan *w;
    EMALLOC(1, work);
    w = *work;
    w->size = size;
    Dalloc(&w->tmp, 64 * 64 * 64 / THREADS);
    Dalloc(&w->compressed, 4 * size);
}

void scan_fin(     Scan *w) {
    Dfree(w->tmp);
    Dfree(w->compressed);
    EFREE(w);
}
#elif defined(DEV_CPU)
struct Scan { };
void scan_ini(int, Scan**) { }
void scan_fin(Scan*) { }

void scan_apply(const int* input, int size,      int* output,       Scan*) {
    int i, s;
    s = 0;
    for (i = 0; i < size; i++) {
        output[i] = s;
        s += input[i];
    }

}
#else
  #error DEV_* is undefined
#endif







#define UDX_DBG_CLIST_D_API_COMMON
#define UDX_DBG_CLIST_DEF_COMMON
#define UDX_DBG_CLIST_KL_COMMON
#define UDX_DBG_CLIST_TYPE_COMMON










#include <stdio.h>


struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);

struct Dbg;

struct Config;
struct Force;
struct Particle;
struct int3;
struct Coords;

#ifndef UDX_DEBUG_KIND_COMMON
#define UDX_DEBUG_KIND_COMMON
enum {
    DBG_POS,
    DBG_POS_SOFT,
    DBG_VEL,
    DBG_FORCES,
    DBG_COLORS,
    DBG_CLIST,
    DBG_NKIND_
};
#endif

void dbg_ini(Dbg**);
void dbg_fin(Dbg*);

void dbg_enable(int kind, Dbg *dbg);
void dbg_disable(int kind, Dbg *dbg);
void dbg_set_verbose(bool, Dbg *dbg);
void dbg_set_dump(bool, Dbg *dbg);

void dbg_set_conf(const Config*, Dbg*);

void dbg_check_pos(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_pos_soft(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_vel(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_forces(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp, const Force *ff);
void dbg_check_colors(const Coords *c, const Dbg *dbg, int n, const int *cc);
void dbg_check_clist(const Coords *c, const Dbg *dbg, int3 L, const int *starts, const int *counts, const Particle *pp);

static void set(const Config *cfg, const char *desc, int kind, Dbg *dbg) {
    int enabled = 0;
    UC(conf_lookup_bool(cfg, desc, &enabled));
    if (enabled) dbg_enable (kind, dbg);
    else         dbg_disable(kind, dbg);
}

void dbg_set_conf(const Config *cfg, Dbg *dbg) {
    set(cfg, "dbg.pos",      DBG_POS,      dbg);
    set(cfg, "dbg.pos_soft", DBG_POS_SOFT, dbg);
    set(cfg, "dbg.vel",      DBG_VEL,      dbg);
    set(cfg, "dbg.forces",   DBG_FORCES,   dbg);
    set(cfg, "dbg.colors",   DBG_COLORS,   dbg);
    set(cfg, "dbg.clist",    DBG_CLIST,    dbg);

    int v;
    UC(conf_lookup_bool(cfg, "dbg.verbose", &v));
    dbg_set_verbose(v, dbg);
    UC(conf_lookup_bool(cfg, "dbg.dump", &v));
    dbg_set_dump(v, dbg);
}



#include <stdio.h>
#include <assert.h>

#ifndef UDX_DBG_CLIST_DEF_COMMON
#define UDX_DBG_CLIST_DEF_COMMON
#endif

#ifndef UDX_DBG_CLIST_TYPE_COMMON
#define UDX_DBG_CLIST_TYPE_COMMON
#endif

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())






















#ifndef UDX_DBG_CLIST_KL_COMMON
#define UDX_DBG_CLIST_KL_COMMON
#endif
#ifndef UDX_DBG_CLIST_MSG_COMMON
#define UDX_DBG_CLIST_MSG_COMMON
#endif
#ifndef UDX_DBG_CLIST_D_API_COMMON
#define UDX_DBG_CLIST_D_API_COMMON
#endif
#ifndef UDX_CONTROL_COORDS_COMMON
#define UDX_CONTROL_COORDS_COMMON
struct Coords_v {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};
#endif
struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);
struct Particle;
struct Force;
struct TxtRead;

void txt_write_pp(long n, const Particle*, const char *path);
void txt_write_pp_ff(long n, const Particle*, const Force*, const char *path);

void txt_read_pp(const char *path, TxtRead **);
void txt_read_pp_ff(const char *path, TxtRead **);
void txt_read_ff(const char *path, TxtRead **);
void txt_read_fin(TxtRead*);

int txt_read_get_n(const TxtRead *);
const Particle* txt_read_get_pp(const TxtRead *);
const Force*    txt_read_get_ff(const TxtRead *);


#define ERRLIST(_) _(NONE),                     \
        _(INVALID),                             \
        _(INF_VAL),                             \
        _(NAN_VAL)

#define make_str(s) #s
#define make_enum(s) ERR_##s

enum {
    ERRLIST(make_enum),
    NERRORS
};

static const char *err_str[NERRORS] = {ERRLIST(make_str)};

#undef ERRLIST
#undef make_str
#undef make_enum

typedef int err_type;

static __device__ err_type dbg_error;

static void err_ini() {
    err_type e = ERR_NONE;
    void *devPtr;
    CC(d::GetSymbolAddress(&devPtr, &dbg_error));
    CC(d::MemcpyToSymbol(&dbg_error, &e, sizeof(err_type)));
}

static err_type err_get() {
    err_type e;
    CC(d::MemcpyFromSymbol(&e, &dbg_error, sizeof(err_type)));
    return e;
}

static const char* get_err_str(err_type e) {return err_str[e];}

static __device__ void dbg_report(err_type e) {
    if (e != ERR_NONE) atomicExch(&dbg_error, e);
}

static __device__ err_type dbg_check_float(float a) {
    if (isnan(a)) return ERR_NAN_VAL;
    if (isinf(a)) return ERR_INF_VAL;
    return ERR_NONE;
}

static __device__ err_type dbg_check_float3(const float a[3]) {
    enum {X, Y, Z};
    err_type e;
#define check_ret(A) if ((e = dbg_check_float(A)) != ERR_NONE) return e
    check_ret(a[X]);
    check_ret(a[Y]);
    check_ret(a[Z]);
#undef check_ret
    return ERR_NONE;
}
static __device__ bool dbg_valid_pos(float x, int L) {
    if (x < -0.5*L || x > 0.5*L)
        return false;
    return true;
}

static __device__ bool dbg_valid_unpacked_pos(int3 L, float x, float y, float z) {
    bool ok = true;
    ok &= dbg_valid_pos(x, L.x);
    ok &= dbg_valid_pos(y, L.y);
    ok &= dbg_valid_pos(z, L.z);

    return ok;
}

static __device__ err_type dbg_valid_pos(int3 L, const Particle p, bool verbose) {
    enum {X, Y, Z};
    float x, y, z;
    err_type e;
    x  = p.r[X];  y = p.r[Y];  z = p.r[Z];
    e = dbg_check_float3(p.r);
    if (e != ERR_NONE) return e;
    if (dbg_valid_unpacked_pos(L, x, y, z)) e = ERR_NONE;
    else                                e = ERR_INVALID;

    if (verbose && e != ERR_NONE)
        printf("DBG: pos pu: p = [%g, %g %g], [%g, %g, %g]\n",
               p.r[X], p.r[Y], p.r[Z], p.v[X], p.v[Y], p.v[Z]);

    return e;
}

__global__ void dbg_check_pos_kernel(int3 L, const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p;
    if (i >= n) return;
    p = pp[i];
    err_type e = dbg_valid_pos(L, p, verbose);
    dbg_report(e);
}

static __device__ bool dbg_valid_unpacked_pos_pu(int3 L, float x, float y, float z) {
    bool ok = true;
    ok &= dbg_valid_pos(x, L.x + 3);
    ok &= dbg_valid_pos(y, L.y + 3);
    ok &= dbg_valid_pos(z, L.z + 3);

    return ok;
}

static __device__ err_type dbg_valid_pos_pu(int3 L, const Particle p, bool verbose) {
    enum {X, Y, Z};
    float x, y, z;
    err_type e;
    x  = p.r[X];  y = p.r[Y];  z = p.r[Z];
    e = dbg_check_float3(p.r);
    if (e != ERR_NONE) return e;
    if (dbg_valid_unpacked_pos_pu(L, x, y, z)) e = ERR_NONE;
    else                                   e = ERR_INVALID;

    if (verbose && e != ERR_NONE)
        printf("DBG: pos pu: p = [%g, %g %g], [%g, %g, %g]\n",
               p.r[X], p.r[Y], p.r[Z], p.v[X], p.v[Y], p.v[Z]);

    return e;
}

__global__ void dbg_check_pos_pu_kernel(int3 L, const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p;
    if (i >= n) return;
    p = pp[i];
    err_type e = dbg_valid_pos_pu(L, p, verbose);
    dbg_report(e);
}
static __device__ bool dbg_valid_vel(float dt, float v, int L) {
    float dx = fabs(v * dt);
    if (dx > L/2)
        return false;
    return true;
}

static __device__ bool dbg_valid_vel3(float dt, int3 L, float vx, float vy, float vz) {
    bool ok = true;
    ok &= dbg_valid_vel(dt, vx, L.x);
    ok &= dbg_valid_vel(dt, vy, L.y);
    ok &= dbg_valid_vel(dt, vz, L.z);

    return ok;
}

static __device__ err_type dbg_valid_vv(float dt, int3 L, const Particle p, bool verbose) {
    enum {X, Y, Z};
    err_type e;
    const float *v = p.v;
    e = dbg_check_float3(v);
    if (e != ERR_NONE) return e;
    if ( dbg_valid_vel3(dt, L, v[X], v[Y], v[Z])) e = ERR_NONE;
    else                                       e = ERR_INVALID;

    if (verbose && e != ERR_NONE)
        printf("DBG: vel: p = [%g, %g %g], [%g, %g, %g]\n",
               p.r[X], p.r[Y], p.r[Z], p.v[X], p.v[Y], p.v[Z]);

    return e;
}

__global__ void dbg_check_vv_kernel(float dt, int3 L, const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p;
    if (i >= n) return;
    p = pp[i];
    err_type e = dbg_valid_vv(dt, L, p, verbose);
    dbg_report(e);
}
static __device__ bool dbg_valid_acc(float dt, float a, int L, bool verbose) {
    float dx = fabs(a * dt * dt);
    if (dx > 0.5*L) {
        if (verbose) printf("DBG: a = %g (L = %d)\n", a, L);
        return false;
    }
    return true;
}

static __device__ bool dbg_valid_unpacked_f(float dt, int3 L, float fx, float fy, float fz, bool verbose) {
    bool ok = true;
    ok &= dbg_valid_acc(dt, fx, L.x, verbose);
    ok &= dbg_valid_acc(dt, fy, L.y, verbose);
    ok &= dbg_valid_acc(dt, fz, L.z, verbose);

    return ok;
}

static __device__ err_type dbg_valid_f(float dt, int3 L, const Force *f, bool verbose) {
    enum {X, Y, Z};
    float fx, fy, fz;
    err_type e;
    fx = f->f[X]; fy = f->f[Y]; fz = f->f[Z];
    e = dbg_check_float3(f->f);
    if (e != ERR_NONE) return e;
    if (dbg_valid_unpacked_f(dt, L, fx, fy, fz, verbose)) e = ERR_NONE;
    else                                               e = ERR_INVALID;
    return e;
}

__global__ void dbg_check_ff_kernel(float dt, int3 L, const Force *ff, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    err_type e = dbg_valid_f(dt, L, ff + i, verbose);
    dbg_report(e);
}
static __device__ err_type dbg_valid_color(int c, bool verbose) {
    if (c != BLUE_COLOR && c != RED_COLOR) {
        if (verbose) printf("DBG: color = %d\n", c);
        return ERR_INVALID;
    }
    return ERR_NONE;
}

__global__ void dbg_check_cc_kernel(const int *cc, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    err_type e = dbg_valid_color(cc[i], verbose);
    dbg_report(e);
}
static __device__ int3 dbg_cell_coords(int3 L, int cid) {
    int3 c;
    c.x = cid % L.x;
    c.z = cid / (L.y * L.x);
    c.y = (cid - L.y * L.x * c.z) / L.x;
    return c;
}

static __device__ bool dbg_at_coord(float x, int c, int L, bool verbose, const char *dir) {
    x += L/2;
    if ((x < c) || (x > c + 1)) {
        if (verbose) printf("Particle out of its cell : %s = %f, c = %d, L = %d\n", dir, x, c, L);
        return false;
    }
    return true;
}

static __device__ bool dbg_at_coords(const float r[3], int3 coords, int3 L, bool verbose) {
    enum {X, Y, Z};
    bool ok = true;
    ok &= dbg_at_coord(r[X], coords.x, L.x, verbose, "x");
    ok &= dbg_at_coord(r[Y], coords.y, L.y, verbose, "y");
    ok &= dbg_at_coord(r[Z], coords.z, L.z, verbose, "z");
    return ok;
}

static __device__ err_type dbg_valid_particle(int3 L, int cid, const Particle *p, bool verbose) {
    err_type e;
    e = dbg_check_float3(p->r); if (e != ERR_NONE) return e;
    e = dbg_check_float3(p->v); if (e != ERR_NONE) return e;

    int3 coords = dbg_cell_coords(L, cid);
    bool valid = dbg_at_coords(p->r, coords, L, verbose);
    if (!valid) e = ERR_INVALID;

    return e;
}

static __device__ err_type dbg_valid_cell(int3 L, int cid, int s, int c, const Particle *pp, bool verbose) {
    err_type e = ERR_NONE;
    int i;
    Particle p;

    for (i = 0; i < c; ++i) {
        p = pp[s + i];
        e = dbg_valid_particle(L, cid, &p, verbose);
        if (e != ERR_NONE) return e;
    }
    return e;
}

__global__ void dbg_check_clist_kernel(int3 ncells, const int *ss, const int *cc, const Particle *pp, bool verbose = true) {
    int i, c, s;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= ncells.x * ncells.y * ncells.z) return;

    s = ss[i];
    c = cc[i];

    err_type e = dbg_valid_cell(ncells, i, s, c, pp, verbose);
    dbg_report(e);
}

struct Dbg {
    int state[DBG_NKIND_];
    bool verbose;
    bool dump;
};
void dbg_ini(Dbg **dbg) {
    Dbg *d;
    int i;
    EMALLOC(1, dbg);
    d = *dbg;
    for (i = 0; i < DBG_NKIND_; ++i) d->state[i] = 0;
    d->verbose = false;
    d->dump = false;
}
void dbg_fin(Dbg *dbg) {
    EFREE(dbg);
}

static void set(int kind, int val, Dbg *dbg) {
    if (kind < 0 || kind >= DBG_NKIND_)
        ERR("Unrecognised kind of id <%d>", kind);
    dbg->state[kind] = val;
}

void dbg_enable(int kind, Dbg *dbg) {
    UC(set(kind, 1, dbg));
}

void dbg_disable(int kind, Dbg *dbg) {
    UC(set(kind, 0, dbg));
}

void dbg_set_verbose(bool active, Dbg *dbg) {dbg->verbose = active;}
void dbg_set_dump(bool active, Dbg *dbg) {dbg->dump = active;}

static int check(const Dbg *dbg, int kind) {return dbg->state[kind];}

static bool error() {
    err_type e;
    UC(e = err_get());
    return e != ERR_NONE;
}

static void gen_name(const Coords *c, const char *base,      char *name) {
    int r;
    char stamp[FILENAME_MAX];
    coord_stamp(c,      stamp);
    r = sprintf(name, "dbg.%s.%s.txt", base, stamp);
    if (r < 0) ERR("sprintf failed");
}

static void dump_pp(const Coords *c, const char *base, int n, const Particle *dev) {
    Particle *hst;
    char name[FILENAME_MAX];
    EMALLOC(n, &hst);
    cD2H(hst, dev, n);
    UC(gen_name(c, base, name));
    UC(txt_write_pp(n, hst, name));
    EFREE(hst);
}

static void dump_pp_ff(const Coords *c, const char *base, int n, const Particle *ppdev, const Force *ffdev) {
    Particle *pphst;
    Force *ffhst;
    char name[FILENAME_MAX];

    EMALLOC(n, &pphst);
    EMALLOC(n, &ffhst);

    cD2H(pphst, ppdev, n);
    cD2H(ffhst, ffdev, n);

    UC(gen_name(c, base, name));

    UC(txt_write_pp_ff(n, pphst, ffhst, name));

    EFREE(pphst);
    EFREE(ffhst);
}

static void print() {
    err_type e;
    UC(e = err_get());
    ERR("DBG: error: %s", get_err_str(e));
}

void dbg_check_pos(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp) {
    int3 L = subdomain(c);
    if (!check(dbg, DBG_POS))
        return;
    UC(err_ini());
    KL(dbg_check_pos_kernel, (k_cnf(n)), (L, pp, n));
    if (error()) {
        if (dbg->dump)
            dump_pp(c, base, n, pp);
        UC(print());
    }
}

void dbg_check_pos_soft(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp) {
    int3 L = subdomain(c);
    if (!check(dbg, DBG_POS_SOFT))
        return;
    UC(err_ini());
    KL(dbg_check_pos_pu_kernel, (k_cnf(n)), (L, pp, n));
    if (error()) {
        if (dbg->dump)
            dump_pp(c, base, n, pp);
        UC(print());
    }
}

void dbg_check_vel(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp) {
    int3 L = subdomain(c);

    if (!check(dbg, DBG_VEL))
        return;
    UC(err_ini());
    KL(dbg_check_vv_kernel, (k_cnf(n)), (dt, L, pp, n));
    if (error()) {
        if (dbg->dump)
            dump_pp(c, base, n, pp);
        UC(print());
    }
}

void dbg_check_forces(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp, const Force *ff) {
    int3 L = subdomain(c);

    if (!check(dbg, DBG_FORCES))
        return;
    UC(err_ini());
    KL(dbg_check_ff_kernel, (k_cnf(n)), (dt, L, ff, n));
    if (error()) {
        if (dbg->dump)
            dump_pp_ff(c, base, n, pp, ff);
        UC(print());
    }
}

void dbg_check_colors(const Coords*, const Dbg *dbg, int n, const int *cc) {
    if (!check(dbg, DBG_COLORS))
        return;
    UC(err_ini());
    KL(dbg_check_cc_kernel, (k_cnf(n)), (cc, n));
    if (error()) {
        UC(print());
    }
}

void dbg_check_clist(const Coords*, const Dbg *dbg, int3 L, const int *starts, const int *counts, const Particle *pp) {
    if (!check(dbg, DBG_CLIST))
        return;
    int n = L.x * L.y * L.z;
    UC(err_ini());
    KL(dbg_check_clist_kernel, (k_cnf(n)), (L, starts, counts, pp));
    if (error()) {
        UC(print());
    }
}





#include <stdio.h>
#include <assert.h>





#ifndef UDX_DBG_CLIST_MSG_COMMON
#define UDX_DBG_CLIST_MSG_COMMON
#endif





#ifndef UDX_DBG_CLIST_DEF_COMMON
#define UDX_DBG_CLIST_DEF_COMMON
#endif
#ifndef UDX_DBG_CLIST_TYPE_COMMON
#define UDX_DBG_CLIST_TYPE_COMMON
#endif

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())









struct Scan;

void scan_apply(const int *input, int size,      int *output,       Scan*);
void scan_ini(int size,      Scan**);
void scan_fin(Scan*);

#ifndef UDX_DBG_CLIST_D_API_COMMON
#define UDX_DBG_CLIST_D_API_COMMON
#endif







#ifndef UDX_DBG_CLIST_KL_COMMON
#define UDX_DBG_CLIST_KL_COMMON
#endif

#ifndef UDX_CLIST_PARTLIST_COMMON
#define UDX_CLIST_PARTLIST_COMMON
enum {
    ALIVE = 0,
    DEAD  = 1
};




struct PartList {
    const Particle *pp;
    const int *deathlist;
};
#endif
static __device__ bool is_dead(int i, const PartList lp) {
    if (lp.deathlist)
        return lp.deathlist[i] == DEAD;
    return false;
}

static __device__ bool is_alive(int i, const PartList lp) {
    return !is_dead(i, lp);
}

#ifndef UDX_CONTROL_PARTICLE_DEV_COMMON
#define UDX_CONTROL_PARTICLE_DEV_COMMON
static __device__ float3 fetch_pos(int i, const Particle *pp) {
    enum {X, Y, Z};
    float3 r;
    r.x = pp[i].r[X]; r.y = pp[i].r[Y]; r.z = pp[i].r[Z];
    return r;
}

static __device__ float3 fetch_vel(int i, const Particle *pp) {
    enum {X, Y, Z};
    float3 v;
    v.x = pp[i].v[X]; v.y = pp[i].v[Y]; v.z = pp[i].v[Z];
    return v;
}
#endif

struct Clist;
struct ClistMap;

void clist_ini(int LX, int LY, int LZ,      Clist **c);
void clist_fin(     Clist *c);

void clist_ini_map(int maxp, int nA, const Clist *c,      ClistMap **m);
void clist_fin_map(ClistMap *m);

const int* clists_get_ss(const Clist *c);
const int* clists_get_cc(const Clist *c);
int3       clists_get_dim(const Clist *c);
int        clists_get_n(const Clist *c);
const uint* clist_get_ids(const ClistMap *m);

void clist_ini_counts(Clist *c);
void clist_subindex(bool project, int aid, int n, const PartList lp,      Clist *c, ClistMap *m);
void clist_build_map(const int nn[],      Clist *c, ClistMap *m);


void clist_subindex_local(int n, const PartList lp,      Clist *c, ClistMap *m);
void clist_subindex_remote(int n, const PartList lp,      Clist *c, ClistMap *m);

void clist_gather_pp(const Particle *pplo, const Particle *ppre, const ClistMap *m, long nout,      Particle *ppout);
void clist_gather_ii(const int *iilo, const int *iire, const ClistMap *m, long nout,      int *iiout);


void clist_build(int nlo, int nout, const Particle *pplo,      Particle *ppout, Clist *c, ClistMap *m);
#ifndef UDX_CLIST_DEV_COMMON
#define UDX_CLIST_DEV_COMMON
#define _I_ static __device__

enum { CLIST_NMAX = (256*256*256) };

_I_ uint clist_encode_id(int s, int pid)
{

    assert(pid < CLIST_NMAX);
    return CLIST_NMAX * s + pid;
}

_I_ void clist_decode_id(uint c,      int *s, int *pid)
{

    *pid = c % CLIST_NMAX;
    *s   = c / CLIST_NMAX;
}

#undef _I_
#endif

struct Clist {
    int3 dims;
    int ncells;
    int *starts, *counts;
};

struct ClistMap {
    int nA;
    uchar4 **ee;
    uint *ii;
    Scan *scan;
    long maxp;
};
enum {
    CLIST_VALID   = 0,
    CLIST_INVALID = 255
};

static __device__ bool clist_inside(float3 r, int3 L) {
    return
        (r.x >= -L.x/2) && (r.x < L.x/2) &&
        (r.y >= -L.y/2) && (r.y < L.y/2) &&
        (r.z >= -L.z/2) && (r.z < L.z/2)  ;
}

static __device__ int clist_project_cid(int i, const int L) {
    return i < 0 ? 0 : i >= L ? L - 1 : i;
}

static __device__ uchar4 clist_get_entry(const bool project, float3 r, int3 L) {
    uchar4 e;
    int ix, iy, iz;


    ix = (int) ((double) r.x + L.x/2);
    iy = (int) ((double) r.y + L.y/2);
    iz = (int) ((double) r.z + L.z/2);

    if (project) {
        ix = clist_project_cid(ix, L.x);
        iy = clist_project_cid(iy, L.y);
        iz = clist_project_cid(iz, L.z);
    }

    if (project || clist_inside(r, L)) e.w =   CLIST_VALID;
    else                         e.w = CLIST_INVALID;

    e.x = ix; e.y = iy; e.z = iz;
    return e;
}

static __device__ int clist_get_cid(int3 L, uchar4 e) {
    return e.x + L.x * (e.y + L.y * e.z);
}


__global__ void clist_subindex_kernel(bool project, int3 L, int n, const PartList lp,        int *counts,      uchar4 *ee) {
    int i, cid;
    float3 r;
    uchar4 e;
    bool dead;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    dead = is_dead(i, lp);
    r = fetch_pos(i, lp.pp);

    if (dead) {
        e.x = e.y = e.z = 0;
        e.w = CLIST_INVALID;
    } else {
        e = clist_get_entry(project, r, L);
        cid = clist_get_cid(L, e);
    }

    if (e.w != CLIST_INVALID)
        e.w = atomicAdd(counts + cid, 1);

    ee[i] = e;
}

__global__ void clist_get_ids_kernel(int array_id, int3 L, int n, const int *starts, const uchar4 *ee,      uint *ii) {
    int i, cid, id, start;
    uchar4 e;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    e = ee[i];
    cid = clist_get_cid(L, e);

    if (e.w != CLIST_INVALID) {
        start = starts[cid];
        id = start + e.w;
        ii[id] = clist_encode_id(array_id, i);
    }
}

template <typename T, int N>
__device__ void clist_fetch(const Sarray<const T*, N> src, uint i,      T *d) {
    int src_id, array_id;
    clist_decode_id(i,      &array_id, &src_id);
    *d = src.d[array_id][src_id];
}

template <typename T, int N>
__global__ void clist_gather_kernel(const Sarray<const T*, N> src, const uint *ii, int n,      T *dd) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    uint code = ii[i];
    clist_fetch(src, code,      dd + i);
}
void clist_ini(int LX, int LY, int LZ,      Clist **cl) {
    Clist *c;
    EMALLOC(1, cl);
    c = *cl;
    c->dims.x = LX;
    c->dims.y = LY;
    c->dims.z = LZ;
    c->ncells = LX * LY * LZ;

    int n = (c->ncells + 16);
    Dalloc(&c->starts, n);
    Dalloc(&c->counts, n);
}

void clist_ini_map(int maxp, int nA, const Clist *c,      ClistMap **map) {
    long i, size;
    EMALLOC(1, map);
    ClistMap *m = *map;

    UC(scan_ini(c->ncells + 16,      &m->scan));

    m->maxp = maxp;

    m->nA = nA;
    EMALLOC(nA, &m->ee);

    for (i = 0; i < nA; ++i) Dalloc(&m->ee[i], maxp);

    size = nA * maxp;
    Dalloc(&m->ii, size);
}
void clist_fin(     Clist *c) {
    Dfree(c->starts);
    Dfree(c->counts);
    EFREE(c);
}

void clist_fin_map(ClistMap *m) {
    UC(scan_fin(     m->scan));

    for (int i = 0; i < m->nA; ++i) Dfree(m->ee[i]);
    EFREE(m->ee);
    Dfree(m->ii);
    EFREE(m);
}
enum {LOCAL, REMOTE};

const int* clists_get_ss(const Clist *c) {return c->starts;}
const int* clists_get_cc(const Clist *c) {return c->counts;}
int3       clists_get_dim(const Clist *c)  {return c->dims;}
int        clists_get_n(const Clist *c)  {return c->ncells;}
const uint* clist_get_ids(const ClistMap *m) {return m->ii;}

void clist_ini_counts(     Clist *c) {
    if (c->ncells) CC(d::MemsetAsync(c->counts, 0, (c->ncells + 16) * sizeof(int)));
}

static void check_input_size(int id, long n, const ClistMap *m) {
    long cap = m->maxp;
    if (n > cap)
        ERR("Too many input particles for array %d (%ld / %ld)", id, n, cap);
}

static void comp_subindices(bool project, int n, const PartList lp, int3 dims,      int *cc, uchar4 *ee) {
    if (n) KL(clist_subindex_kernel, (k_cnf(n)), (project, dims, n, lp,        cc,      ee));
}

void clist_subindex(bool project, int aid, int n, const PartList lp,      Clist *c, ClistMap *m) {
    UC(check_input_size(aid, n, m));
    UC(comp_subindices(project, n, lp, c->dims,      c->counts, m->ee[aid]));
}

void clist_subindex_local(int n, const PartList lp,      Clist *c, ClistMap *m) {
    UC(clist_subindex(false, LOCAL, n, lp,      c, m));
}

void clist_subindex_remote(int n, const PartList lp,      Clist *c, ClistMap *m) {
    UC(clist_subindex(false, REMOTE, n, lp,      c, m));
}

void clist_build_map(const int nn[],      Clist *c, ClistMap *m) {
    int nc, *cc, *ss, n, i;
    const uchar4 *ee;
    uint *ii = m->ii;
    int3 dims = c->dims;
    nc = c->ncells;
    cc = c->counts;
    ss = c->starts;

    scan_apply(cc, nc + 16,      ss,       m->scan);

    for (i = 0; i < m->nA; ++i) {
        n = nn[i];
        ee = m->ee[i];
        UC(check_input_size(i, n, m));
        if (n) KL(clist_get_ids_kernel, (k_cnf(n)), (i, dims, n, ss, ee,      ii));
    }
}

static void check_map_capacity(long n, const ClistMap *m) {
    long cap = m->maxp * m->nA;
    if (n > cap)
        ERR("Too many particles for this cell list (%ld / %ld)", n, cap);
}

void clist_gather_pp(const Particle *pplo, const Particle *ppre, const ClistMap *m, long nout,      Particle *ppout) {
    Sarray <const Particle*, 2> src = {pplo, ppre};
    UC(check_map_capacity(nout, m));
    if (nout)
        KL(clist_gather_kernel, (k_cnf(nout)), (src, m->ii, nout,      ppout));
}

void clist_gather_ii(const int *iilo, const int *iire, const ClistMap *m, long nout,      int *iiout) {
    Sarray <const int*, 2> src = {iilo, iire};
    UC(check_map_capacity(nout, m));
    if (nout)
        KL(clist_gather_kernel, (k_cnf(nout)), (src, m->ii, nout,      iiout));
}

void clist_build(int nlo, int nout, const Particle *pplo,      Particle *ppout, Clist *c, ClistMap *m) {
    const int nn[] = {nlo, 0};
    PartList lp;
    lp.pp = pplo;
    lp.deathlist = NULL;
    UC(clist_ini_counts(     c));
    UC(clist_subindex_local (nlo, lp,      c, m));
    UC(clist_build_map(nn,      c, m));
    UC(clist_gather_pp(pplo, NULL, m, nout, ppout));
}






#include <assert.h>
#include <stdio.h>



#ifndef UDX_DBG_CLIST_D_API_COMMON
#define UDX_DBG_CLIST_D_API_COMMON
#endif


#if   defined(DEV_CUDA)


#elif defined(DEV_CPU)

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
#else
  #error DEV_* is undefined
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
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
#else
  #error DEV_* is undefined
#endif







struct RNDunif;
void rnd_ini(int x, int y, int z, int c,      RNDunif **r);
void rnd_fin(RNDunif *r);
float rnd_get(RNDunif *r);
namespace rnd {




static __const__ int N = 18;

static __const__ float gold   = 0.6180339887498948482;
static __const__ float silver = 0.4142135623730950488;
static __const__ float bronze = 0.00008877875787352212838853023;
static __const__ float tin    = 0.00004602357186447026756768986;
static __const__ float sqrt2 = 1.41421356237309514547;















__inline__ __device__ float rem( float r ) {
    return r - floorf( r );
}


__inline__ __device__ float FMA( float x, float y, float z ) {
    return __fmaf_rz( x, y, z );
}






template<int N> __inline__ __device__ float __logistic_core( float x )
{
    float x2 = x * x;
    float r = FMA( FMA( 8.0, x2, -8.0 ), x2, 1.0 );
    return __logistic_core < N - 2 > ( r );
}

template<> __inline__ __device__ float __logistic_core<1>( float x ) {
    return FMA( 2.0 * x, x, -1.0 );
}

template<> __inline__ __device__ float __logistic_core<0>( float x ) {
    return x;
}

inline __device__ float mean0var1ii( float seed, int u, int v )
{
    float p = rem( ( ( u & 0x3FF ) * gold ) + u * bronze + ( ( v & 0x3FF ) * silver ) + v * tin );
    float l = __logistic_core<N>( seed - p );
    return l * sqrt2;
}

inline __device__ float mean0var1uu( float seed, uint u, uint v )
{

    float p = rem( ( ( u & 0x3FFU ) * gold ) + u * bronze + ( ( v & 0x3FFU ) * silver ) + v * tin );

    float l = __logistic_core<N>( seed - p );

    return l * sqrt2;
}

}
#ifndef UDX_CONTROL_MATH_COMMON
#define UDX_CONTROL_MATH_COMMON


#define _HD_ __host__ __device__

template <typename T1, typename T2>
_HD_ void scal(const T1 a,        T2 *b) {
    b->x *= a;
    b->y *= a;
    b->z *= a;
}

template <typename T1, typename T2>
_HD_ void add(const T1 *a,        T2 *b) {
    b->x += a->x;
    b->y += a->y;
    b->z += a->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void diff(const T1 *a, const T2 *b,      T3 *c) {
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ T1 dot(const T2 *a, const T3 *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void cross(const T1 *a, const T2 *b,      T3 *c) {
    c->x = a->y * b->z - a->z * b->y;
    c->y = a->z * b->x - a->x * b->z;
    c->z = a->x * b->y - a->y * b->x;
}

template <typename T1, typename T2, typename T3>
_HD_ void scalmult(const T1 *a, const T2 x,      const T3 *b) {
    b->x = a->x * x;
    b->y = a->y * x;
    b->z = a->z * x;
}


template <typename T1, typename T2, typename T3, typename T4>
_HD_ void apxb(const T1 *a, const T2 x, const T3 *b,      T4 *c) {
    c->x = a->x + x * b->x;
    c->y = a->y + x * b->y;
    c->z = a->z + x * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void axpy(const T1 a, const T2 *x,        T3 *y) {
    y->x += a * x->x;
    y->y += a * x->y;
    y->z += a * x->z;
}
#endif

#ifndef UDX_DBG_CLIST_DEF_COMMON
#define UDX_DBG_CLIST_DEF_COMMON
#endif
#ifndef UDX_DBG_CLIST_MSG_COMMON
#define UDX_DBG_CLIST_MSG_COMMON
#endif








#ifndef UDX_SHARED_FRAG_COMMON
#define UDX_SHARED_FRAG_COMMON
enum {frag_bulk = 26};

struct int3;

namespace frag_hst {

int i2dx(int i);
int i2dy(int i);
int i2dz(int i);
void i2d3(int i,      int d[3]);

int d2i(int x, int y, int z);
int d32i(const int d[3]);

int ncell(int3 L, int i);

int antid2i(int x, int y, int z);
int anti(int i);

void estimates(int3 L, int nfrags, float maxd,      int *cap);

}
#endif

#ifndef UDX_DBG_CLIST_KL_COMMON
#define UDX_DBG_CLIST_KL_COMMON
#endif
#ifndef UDX_DBG_CLIST_TYPE_COMMON
#define UDX_DBG_CLIST_TYPE_COMMON
#endif

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )


#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())









#include "fragment.cuh"

struct Dbg;

struct Config;
struct Force;
struct Particle;
struct int3;
struct Coords;

#ifndef UDX_DEBUG_KIND_COMMON
#define UDX_DEBUG_KIND_COMMON
enum {
    DBG_POS,
    DBG_POS_SOFT,
    DBG_VEL,
    DBG_FORCES,
    DBG_COLORS,
    DBG_CLIST,
    DBG_NKIND_
};
#endif

void dbg_ini(Dbg**);
void dbg_fin(Dbg*);

void dbg_enable(int kind, Dbg *dbg);
void dbg_disable(int kind, Dbg *dbg);
void dbg_set_verbose(bool, Dbg *dbg);
void dbg_set_dump(bool, Dbg *dbg);

void dbg_set_conf(const Config*, Dbg*);

void dbg_check_pos(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_pos_soft(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_vel(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_forces(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp, const Force *ff);
void dbg_check_colors(const Coords *c, const Dbg *dbg, int n, const int *cc);
void dbg_check_clist(const Coords *c, const Dbg *dbg, int3 L, const int *starts, const int *counts, const Particle *pp);


enum {
    MAX_PAR = (N_COLOR * (N_COLOR + 1)) / 2
};


struct PairDPD {
    float a, g, s, spow;
};

struct PairDPDC {
    int ncolors;
    float a[MAX_PAR], g[MAX_PAR], s[MAX_PAR], spow;
};


struct PairDPDCM {
    int ncolors;
    float a[N_COLOR], g[N_COLOR], s[N_COLOR], spow;
};

struct PairDPDLJ {
    float a, g, s, spow;
    float ljs, lje;
};

struct PairAdhesion {
    float k1, k2;
};

struct PairPa {
    float x, y, z;
    float vx, vy, vz;

    int color;
};

struct PairFo {
    float x, y, z;
};

struct PairSFo {
    float x, y, z;

    float sxx, syy, szz, sxy, sxz, syz;
};
#define _S_ static __device__
#define _I_ static __device__

_S_ float cap(float x, float lo, float hi) {
    if      (x > hi) return hi;
    else if (x < lo) return lo;
    else             return x;
}

static const float EPS = 1e-6;
enum {NORM_OK, NORM_BIG, NORM_SMALL};
_S_ int norm(       float3 *pos,      float *pr, float *pinvr) {

    float x, y, z, invr, r, r2;
    x = pos->x; y = pos->y; z = pos->z;

    r2 = x*x + y*y + z*z;
    if      (r2 >= 1 )   return NORM_BIG;
    else if (r2 < EPS) {
        *pr = pos->x = pos->y = pos->z = 0;
        return NORM_SMALL;
    } else {
        invr = rsqrtf(r2);
        r = r2 * invr;
        x *= invr; y *= invr; z *= invr;
        pos->x = x; pos->y = y; pos->z = z; *pr = r; *pinvr = invr;
        return NORM_OK;
    }
}

_S_ float ker_wrf(float s, float x) {
    return powf(x, s);
}

_S_ float magn_dpd(float a, float g, float s, float kpow,
                   float rnd, float r, float ev) {
    float wr, wc;
    float rm, f0;

    rm = max(1 - r, 0.0f);
    wc = rm;
    wr = ker_wrf(kpow, rm);

    f0  = (-g * wr * ev + s * rnd) * wr;
    f0 +=                        a * wc;
    return f0;
}

_S_ float magn_lj(float s, float e, float invr) {
    float t2, t4, t6, f;
    t2 = s * s * invr * invr;
    t4 = t2 * t2;
    t6 = t4 * t2;
    f = e * 24 * invr * t6 * (2 * t6 - 1);
    f = cap(f, 0, 1e4);
    return f;
}

_S_ float magn_adhesion(float k1, float k2, float r, float ev) {
    float spring, visc;
    spring = - k1 * r;
    visc   = - k2 * ev;
    return spring + visc;
}

_S_ float force_magn(const PairDPD *p, float rnd, float ev, float r, float) {
    return magn_dpd(p->a, p->g, p->s, p->spow, rnd, r, ev);
}

_S_ float force_magn(const PairDPDLJ *p, float rnd, float ev, float r, float invr) {
    float f;
    f  = magn_dpd(p->a, p->g, p->s, p->spow, rnd, r, ev);
    f += magn_lj(p->ljs, p->lje, invr);
    return f;
}

_S_ float force_magn(const PairAdhesion *p, float, float ev, float r, float) {
    return magn_adhesion(p->k1, p->k2, r, ev);
}

_S_ void magn2fo(float f0, float3 er, float3 dr,      PairFo *f) {
    f->x = f0 * er.x;
    f->y = f0 * er.y;
    f->z = f0 * er.z;
}

_S_ void magn2fo(float f0, float3 er, float3 dr,      PairSFo *f) {
    f->x = f0 * er.x;
    f->y = f0 * er.y;
    f->z = f0 * er.z;

    f->sxx = 0.5f * f->x * dr.x;
    f->sxy = 0.5f * f->x * dr.y;
    f->sxz = 0.5f * f->x * dr.z;
    f->syy = 0.5f * f->y * dr.y;
    f->syz = 0.5f * f->y * dr.z;
    f->szz = 0.5f * f->z * dr.z;
}

_S_ void make_zero(PairFo *f) {
    f->x = f->y = f->z = 0;
}

_S_ void make_zero(PairSFo *f) {
    f->x = f->y = f->z = 0;
    f->sxx = f->sxy = f->sxz = 0;
    f->syy = f->syz = f->szz = 0;
}

template <typename Param, typename Fo>
_I_ void pair_force(const Param *p, PairPa a, PairPa b, float rnd,      Fo *f)
{
    float r, invr, ev, f0;
    float3 dr, er, dv;
    int vnstat;

    dr.x = a.x - b.x;
    dr.y = a.y - b.y;
    dr.z = a.z - b.z;

    dv.x = a.vx - b.vx;
    dv.y = a.vy - b.vy;
    dv.z = a.vz - b.vz;

    er = dr;

    vnstat = norm(       &er,       &r, &invr);
    if (vnstat == NORM_BIG) {
        make_zero(f);
        return;
    }

    ev = dot<float>(&er, &dv);

    f0 = force_magn(p, rnd, ev, r, invr);
    magn2fo(f0, er, dr,      f);
}

_S_ int colors2pid(int ca, int cb) {
    int c0, c1;
    c0 = ca < cb ? ca : cb;
    c1 = ca < cb ? cb : ca;
    return c1 * (c1+1) / 2 + c0;
}

template <typename Fo>
_I_ void pair_force(const PairDPDC *pc, PairPa a, PairPa b, float rnd,      Fo *f) {
    PairDPD p;
    int pid;
    pid = colors2pid(a.color, b.color);
    p.a = pc->a[pid];
    p.g = pc->g[pid];
    p.s = pc->s[pid];
    p.spow = pc->spow;
    pair_force(&p, a, b, rnd,      f);
}


template <typename Fo>
_I_ void pair_force(const PairDPDCM *pc, PairPa a, PairPa b, float rnd,      Fo *f) {
    PairDPD p;
    int pid = a.color;
    p.a = pc->a[pid];
    p.g = pc->g[pid];
    p.s = pc->s[pid];
    p.spow = pc->spow;
    pair_force(&p, a, b, rnd,      f);
}


_I_ void pair_add(const PairFo *b,      PairFo *a)
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;
}

_I_ void pair_add(const PairSFo *b,      PairSFo *a)
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;

    a->sxx += b->sxx;
    a->sxy += b->sxy;
    a->sxz += b->sxz;
    a->syy += b->syy;
    a->syz += b->syz;
    a->szz += b->szz;
}

#undef _S_
#undef _I_
struct PairParams;
struct Config;

struct PairDPD;
struct PairDPDC;
struct PairDPDCM;
struct PairDPDLJ;
struct PairAdhesion;

void pair_ini(PairParams **);
void pair_fin(PairParams *);

void pair_set_dpd(int ncol, const float a[], const float g[], float spow, PairParams *p);
void pair_set_lj(float sigma, float eps, PairParams *p);
void pair_set_adhesion(float k1, float k2, PairParams *p);

void pair_set_conf(const Config *, const char *name_space, PairParams *);

void pair_compute_dpd_sigma(float kBT, float dt, PairParams *p);

void pair_get_view_dpd(const PairParams*, PairDPD*);
void pair_get_view_dpd_color(const PairParams*, PairDPDC*);
void pair_get_view_dpd_mirrored(const PairParams*, PairDPDCM*);
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
void pair_get_view_adhesion(const PairParams*, PairAdhesion*);

#ifndef UDX_CLIST_PARTLIST_COMMON
#define UDX_CLIST_PARTLIST_COMMON
enum {
    ALIVE = 0,
    DEAD  = 1
};




struct PartList {
    const Particle *pp;
    const int *deathlist;
};
#endif

struct Clist;
struct ClistMap;

void clist_ini(int LX, int LY, int LZ,      Clist **c);
void clist_fin(     Clist *c);

void clist_ini_map(int maxp, int nA, const Clist *c,      ClistMap **m);
void clist_fin_map(ClistMap *m);

const int* clists_get_ss(const Clist *c);
const int* clists_get_cc(const Clist *c);
int3       clists_get_dim(const Clist *c);
int        clists_get_n(const Clist *c);
const uint* clist_get_ids(const ClistMap *m);

void clist_ini_counts(Clist *c);
void clist_subindex(bool project, int aid, int n, const PartList lp,      Clist *c, ClistMap *m);
void clist_build_map(const int nn[],      Clist *c, ClistMap *m);


void clist_subindex_local(int n, const PartList lp,      Clist *c, ClistMap *m);
void clist_subindex_remote(int n, const PartList lp,      Clist *c, ClistMap *m);

void clist_gather_pp(const Particle *pplo, const Particle *ppre, const ClistMap *m, long nout,      Particle *ppout);
void clist_gather_ii(const int *iilo, const int *iire, const ClistMap *m, long nout,      int *iiout);


void clist_build(int nlo, int nout, const Particle *pplo,      Particle *ppout, Clist *c, ClistMap *m);
#ifndef UDX_CLIST_DEV_COMMON
#define UDX_CLIST_DEV_COMMON
#define _I_ static __device__

enum { CLIST_NMAX = (256*256*256) };

_I_ uint clist_encode_id(int s, int pid)
{

    assert(pid < CLIST_NMAX);
    return CLIST_NMAX * s + pid;
}

_I_ void clist_decode_id(uint c,      int *s, int *pid)
{

    *pid = c % CLIST_NMAX;
    *s   = c / CLIST_NMAX;
}

#undef _I_
#endif

struct Contact;
struct int3;
struct PairParams;

void cnt_ini(int maxp, int rank, int3 L, int nobj,      Contact **c);
void cnt_fin(Contact *c);

void cnt_build_cells(int nw, const PairParams **prms, const PaWrap *pw,      Contact *c);
void cnt_bulk(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw);
void cnt_halo(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw, Pap26 all_pp, Fop26 all_ff, const int *all_counts);

template <typename Par>
static __device__ void cnt_pair(Par params, PairPa a, PairPa b, float rnd,
                            float *fx, float *fy, float *fz) {
    PairFo f;
    pair_force(&params, a, b, rnd,      &f);
    *fx = f.x; *fy = f.y; *fz = f.z;
}

enum {CNT_FULL, CNT_EMPTY};
struct CntMap {
    int  org0, org1, org2;
    int cnt0, cnt1, cnt2;
};


static __device__ int cnt_endp(const CntMap m, int i) { return i >= m.cnt2; }

static __device__ int cnt_m2id(const CntMap m, int i) {

    int m1, m2;
    int id;
    m1 = (int)(i >= m.cnt0);
    m2 = (int)(i >= m.cnt1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}
static __device__ int cnt_tex2map(int3 L, int zplane, float x, float y, float z, const int *starts,      CntMap *m) {
    int cnt0, cnt1, cnt2, org0;
    int org1, org2;
    int xcenter, xstart, xcount;
    int ycenter, zcenter, zmy;
    bool zvalid;
    int count0, count1, count2;
    int cid0, cid1, cid2;

    xcenter = L.x / 2 + (int)floorf(x);
    xstart = max(0, xcenter - 1);
    xcount = min(L.x, xcenter + 2) - xstart;

    if (xcenter - 1 >= L.x || xcenter + 2 <= 0) return CNT_EMPTY;

    ycenter = L.y / 2 + (int)floorf(y);

    zcenter = L.z / 2 + (int)floorf(z);
    zmy = zcenter - 1 + zplane;
    zvalid = zmy >= 0 && zmy < L.z;

    count0 = count1 = count2 = 0;

    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < L.y) {
        cid0 = xstart + L.x * (ycenter - 1 + L.y * zmy);
        org0 = starts[cid0];
        count0 = starts[cid0 + xcount] - org0;
    }

    if (zvalid && ycenter >= 0 && ycenter < L.y) {
        cid1 = xstart + L.x * (ycenter + L.y * zmy);
        org1 = starts[cid1];
        count1 = starts[cid1 + xcount] - org1;
    }

    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < L.y) {
        cid2 = xstart + L.x * (ycenter + 1 + L.y * zmy);
        org2 = starts[cid2];
        count2 = starts[cid2 + xcount] - org2;
    }

    cnt0 = count0;
    cnt1 = count0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->cnt0 = cnt0; m->cnt1 = cnt1; m->cnt2 = cnt2;
    return CNT_FULL;
}
static __device__ int cnt_r2map(int3 L, int zplane, int n, float x, float y, float z, const int *starts,      CntMap *m) {

    int cnt0, cnt1, cnt2, org0;
    int org1, org2;
    int xcenter, xstart, xcount;
    int ycenter, zcenter, zmy;
    bool zvalid;
    int count0, count1, count2;
    int cid0, cid1, cid2;

    xcenter = min(L.x - 1, max(0, L.x / 2 + (int)floorf(x)));
    xstart = max(0, xcenter - 1);
    xcount = min(L.x, xcenter + 2) - xstart;
    if (xcenter - 1 >= L.x || xcenter + 2 <= 0) return CNT_EMPTY;
    ycenter = min(L.y - 1, max(0, L.y / 2 + (int)floorf(y)));
    zcenter = min(L.z - 1, max(0, L.z / 2 + (int)floorf(z)));
    zmy = zcenter - 1 + zplane;
    zvalid = zmy >= 0 && zmy < L.z;
    count0 = count1 = count2 = 0;
    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < L.y) {
        cid0 = xstart + L.x * (ycenter - 1 + L.y * zmy);
        org0 = starts[cid0];
        count0 = starts[cid0 + xcount] - org0;
    }
    if (zvalid && ycenter >= 0 && ycenter < L.y) {
        cid1 = xstart + L.x * (ycenter + L.y * zmy);
        org1 = starts[cid1];
        count1 = starts[cid1 + xcount] - org1;
    }

    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < L.y) {
        cid2 = xstart + L.x * (ycenter + 1 + L.y * zmy);
        org2 = starts[cid2];
        count2 = starts[cid2 + xcount] - org2;
    }

    cnt0 = count0;
    cnt1 = count0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->cnt0 = cnt0; m->cnt1 = cnt1; m->cnt2 = cnt2;
    return CNT_FULL;
}
static __device__ void cnt_f22p(float2 a, float2 b, float2 c, PairPa *p) {
    p->x  = a.x; p->y  = a.y; p->z  = b.x;
    p->vx = b.y; p->vy = c.x; p->vz = c.y;
}

static __device__ void cnt_fetch(int i, const float2 *pp, PairPa *p) {
    float2 a, b, c;
    a = __ldg(pp + 3 * i + 0);
    b = __ldg(pp + 3 * i + 1);
    c = __ldg(pp + 3 * i + 2);
    cnt_f22p(a, b, c, p);
}

template <typename Par>
__global__ void cnt_bulk_kernel(bool self, Par params, int3 L, int n_dst, const float2 *pp_dst,
                     const int *starts, const uint *ids_src, const float2 *pp_src,
                     float seed,      float *ff_dst, float *ff_src) {
    CntMap m;

    float fx, fy, fz, rnd;
    PairPa a, b;
    int i, gid, aid, zplane;
    float xforce, yforce, zforce;
    int mapstatus;
    int slot, objid, bid, sentry;
    uint code;

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    aid = gid / 3;
    zplane = gid % 3;

    if (aid >= n_dst) return;

    cnt_fetch(aid, pp_dst, &a);
    mapstatus = cnt_r2map(L, zplane, n_dst, a.x, a.y, a.z, starts,      &m);

    if (mapstatus == CNT_EMPTY) return;
    xforce = yforce = zforce = 0;
    for (i = 0; !cnt_endp(m, i); ++i) {
        slot = cnt_m2id(m, i);
        code = ids_src[slot];
        clist_decode_id(code,      &objid, &bid);

        if (self && aid <= bid) continue;

        cnt_fetch(bid, pp_src, &b);

        rnd = rnd::mean0var1ii(seed, aid, bid);

        cnt_pair(params, a, b, rnd,      &fx, &fy, &fz);

        xforce += fx;
        yforce += fy;
        zforce += fz;

        sentry = 3 * bid;
        atomicAdd(ff_src + sentry,     -fx);
        atomicAdd(ff_src + sentry + 1, -fy);
        atomicAdd(ff_src + sentry + 2, -fz);
    }

    atomicAdd(ff_dst + 3 * aid + 0, xforce);
    atomicAdd(ff_dst + 3 * aid + 1, yforce);
    atomicAdd(ff_dst + 3 * aid + 2, zforce);
}
template <typename Par>
static __device__
void interactions_one_p(Par params, int3 L, float seed, PairPa a, int aid,
                        const int *starts, const uint *ids_src, const float2 *pp_src,
                             float *ff_src, float *fA) {
    enum {X, Y, Z};
    CntMap m;
    int mapstatus;
    PairPa b;
    float fx, fy, fz;
    float xforce = 0, yforce = 0, zforce = 0;
    int zplane;
    int i, slot, oid, bid, sentry;
    uint code;
    float rnd;

    for (zplane = 0; zplane < 3; ++zplane) {
        mapstatus = cnt_tex2map(L, zplane, a.x, a.y, a.z, starts,      &m);
        if (mapstatus == CNT_EMPTY) continue;
        for (i = 0; !cnt_endp(m, i); ++i) {
            slot = cnt_m2id(m, i);
            code = ids_src[slot];
            clist_decode_id(code,      &oid, &bid);

            cnt_fetch(bid, pp_src,      &b);
            rnd = rnd::mean0var1ii(seed, aid, bid);
            cnt_pair(params, a, b, rnd,      &fx, &fy, &fz);
            xforce += fx;
            yforce += fy;
            zforce += fz;

            sentry = 3 * bid;
            atomicAdd(ff_src + sentry,     -fx);
            atomicAdd(ff_src + sentry + 1, -fy);
            atomicAdd(ff_src + sentry + 2, -fz);
        }
    }
    fA[X] += xforce; fA[Y] += yforce; fA[Z] += zforce;
}

template <typename Par>
__global__ void cnt_halo_kernel(Par params, int3 L, float seed,
                     const int27 starts, int n_dst, Pap26 hpp,
                     const int *cellstarts, const uint *ids_src, const float2 *pp_src,
                          Fop26 hff, float *ff_src) {
    int aid, start;
    int fid;
    PairPa a;
    float *fA;

    aid = threadIdx.x + blockDim.x * blockIdx.x;
    if (aid >= n_dst) return;

    fid = frag_dev::frag_get_fid(starts.d, aid);
    start = starts.d[fid];
    cnt_fetch(aid - start, (const float2*)hpp.d[fid], &a);
    fA = hff.d[fid][aid - start].f;
    interactions_one_p(params, L, seed, a, aid, cellstarts, ids_src, pp_src,
                            ff_src, fA);
}
struct Contact {
    int nobj;
    Clist    **cells;
    ClistMap **cmap;
    RNDunif *rgen;
    int3 L;
};
static int get_id_inter(int a, int b) {
    int i, j;
    i = a > b ? a : b;
    j = a > b ? b : a;
    return j + i*(i+1)/2;

}

void cnt_bulk(const Contact *c, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw) {
    int i, j, self, id_prm;
    float rnd;
    const PairParams *prm;
    PairDPDLJ pv;
    const int *starts;
    const uint *ids;
    PaWrap pwi, pwj;
    FoWrap fwi, fwj;
    if (nw == 0) return;

    for (i = 0; i < nw; ++i) {
        pwi = pw[i];
        fwi = fw[i];
        for (j = 0; j <= i; ++j) {
            self = (i == j);
            pwj = pw[j];
            fwj = fw[j];

            starts = clists_get_ss(c->cells[j]);
            ids    = clist_get_ids(c->cmap[j]);

            id_prm = get_id_inter(i, j);

            prm = prms[id_prm];
            if (!prm) continue;
            UC(pair_get_view_dpd_lj(prm, &pv));

            rnd = rnd_get(c->rgen);
            KL(cnt_bulk_kernel, (k_cnf(3 * pwi.n)),
               (self, pv, c->L,
                pwi.n, (const float2*) pwi.pp,
                starts, ids, (const float2*) pwj.pp,
                rnd,      (float*) fwi.ff, (float*) fwj.ff));
        }
    }
}

static int scan(int n, const int cc[], int ss[]) {
    int i, s;
    ss[0] = 0;
    for (i = s = 0; i < n; ++i)
        ss[i + 1] = (s += cc[i]);
    return ss[n];
}

static void halo_one_type(int idh, const Contact *c, int nw, const PairParams **prms,
                          PaWrap *pw, FoWrap *fw,
                          Pap26 PP, Fop26 FF, const int *counts) {
    int i, n, id_prm;
    const PairParams *prm;
    const int *cellstarts;
    const uint *ids;
    int27 starts;
    PairDPDLJ pv;
    PaWrap pwi;
    FoWrap fwi;

    n = scan(26, counts, starts.d);

    for (i = 0; i < nw; ++i) {
        pwi = pw[i];
        fwi = fw[i];
        cellstarts = clists_get_ss(c->cells[i]);
        ids = clist_get_ids(c->cmap[i]);

        id_prm = get_id_inter(idh, i);
        prm = prms[id_prm];
        if (!prm) continue;

        UC(pair_get_view_dpd_lj(prm, &pv));

        KL(cnt_halo_kernel, (k_cnf(n)),
           (pv, c->L, rnd_get(c->rgen), starts,
            n, PP, cellstarts, ids, (const float2*) pwi.pp,
                 FF, (float*) fwi.ff));
    }
}

void cnt_halo(const Contact *cnt, int nw, const PairParams **prms,
              PaWrap *pw, FoWrap *fw, Pap26 all_pp, Fop26 all_ff, const int *all_counts) {
    enum {NFRAGS = 26};
    Pap26 pp;
    Fop26 ff;
    int i, j, start[NFRAGS] = {0};
    const int *counts;

    for (i = 0; i < nw; i++) {
        counts = all_counts + NFRAGS * i;
        for (j = 0; j < NFRAGS; ++j) {
            pp.d[j] = all_pp.d[j] + start[j];
            ff.d[j] = all_ff.d[j] + start[j];
            start[j] += counts[j];
        }
        halo_one_type(i, cnt, nw, prms, pw, fw, pp, ff, counts);
    }
}
enum {
    CID_SUBINDEX,
    CN_SUBINDICES
};

void cnt_ini(int maxp, int rank, int3 L, int nobj,      Contact **cnt) {
    EMALLOC(1, cnt);
    Contact *c = *cnt;

    c->L = L;
    c->nobj = nobj;

    EMALLOC(nobj, &c->cells);
    EMALLOC(nobj, &c->cmap);

    for (int i = 0; i < nobj; ++i) {
        clist_ini(L.x, L.y, L.z,      &c->cells[i]);
        clist_ini_map(maxp, CN_SUBINDICES, c->cells[i],      &c->cmap[i]);
    }
    UC(rnd_ini(7119 - rank, 187 + rank, 18278, 15674,      &c->rgen));
}

void cnt_fin(Contact *c) {
    for (int i = 0; i < c->nobj; ++i) {
        clist_fin(     c->cells[i]);
        clist_fin_map(     c->cmap[i]);
    }
    UC(rnd_fin(c->rgen));
    EFREE(c->cells);
    EFREE(c->cmap);
    EFREE(c);
}

static void build_cells(const PaWrap *w, Clist *c, ClistMap *m) {
    static const bool project = true;
    PartList lp;
    const int cc[] = {w->n};

    lp.pp = w->pp;
    lp.deathlist = NULL;

    clist_ini_counts(c);
    clist_subindex(project, CID_SUBINDEX, w->n, lp,      c, m);
    clist_build_map(cc,      c, m);
}

static bool has_work(int i, int, const PairParams **prms) {
    int j, k;
    for (j = 0; j <= i; ++j) {
        k = get_id_inter(i, j);
        if (prms[k]) return true;
    }
    return false;
}

void cnt_build_cells(int nw, const PairParams **prms, const PaWrap *pw,      Contact *c) {
    for (int i = 0; i < nw; ++i) {
        if (!has_work(i, nw, prms)) continue;
        build_cells(&pw[i], c->cells[i], c->cmap[i]);
    }
}
