#include "macros.cuh"













#include "config.cuh"

#include <stdlib.h>
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
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();

struct Particle;
struct Force;
struct BForce;
struct float3;
struct Config;
struct Coords;

void bforce_ini(BForce **p);
void bforce_fin(BForce *p);

void bforce_set_none(     BForce *p);
void bforce_set_cste(float3 f,      BForce *p);
void bforce_set_dp(float a,      BForce *p);
void bforce_set_shear(float a,      BForce *p);
void bforce_set_rol(float a,      BForce *p);
void bforce_set_rad(float a,      BForce *p);

void bforce_set_conf(const Config *cfg,      BForce *bf);

void bforce_adjust(float3 f,      BForce *fpar);
void bforce_apply(const Coords *c, float mass, const BForce *bf, int n, const Particle *pp,      Force *ff);

#include <stdio.h>



#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
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

#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
#include "device.cuh"
#endif





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
#ifndef UDX_SCHEME_KL_COMMON
#define UDX_SCHEME_KL_COMMON
#include "launch.cuh"
#endif
#ifndef UDX_SCHEME_CC_COMMON
#define UDX_SCHEME_CC_COMMON


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




#endif














#include <mpi.h>
#include <stdio.h>
#include <math.h>

void msg_ini(int rank);
void msg_print(const char *fmt, ...);





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

struct Coords;
struct Particle;
struct Rigid;

#ifndef UDX_SIM_RESTART_COMMON
#define UDX_SIM_RESTART_COMMON
enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};
#endif

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






#include <mpi.h>
#include <math.h>




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
void msg_ini(int rank);
void msg_print(const char *fmt, ...);





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

struct Force;


float force_stat_max(int n, const Force*);
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

struct TimeStep;
struct TimeStepAccel;
struct Force;
struct Config;

void time_step_accel_ini(     TimeStepAccel**);
void time_step_accel_fin(TimeStepAccel*);

void time_step_accel_push(TimeStepAccel*, float m, int n, Force*);
void time_step_accel_reset(TimeStepAccel*);


void time_step_ini(const Config*,      TimeStep**);
void time_step_fin(TimeStep*);

float time_step_dt(TimeStep*, MPI_Comm, TimeStepAccel*);
float time_step_dt0(TimeStep*);
void time_step_log(TimeStep*);










#include <stdio.h>
#include <assert.h>




#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
#include "device.cuh"
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

#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
#endif
void msg_ini(int rank);
void msg_print(const char *fmt, ...);
#ifndef UDX_SCHEME_CC_COMMON
#define UDX_SCHEME_CC_COMMON


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




#endif

#ifndef UDX_SCHEME_KL_COMMON
#define UDX_SCHEME_KL_COMMON
#include "launch.cuh"
#endif
#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif














struct Force;
struct Particle;

void scheme_move_apply(float dt, float mass, int n, const Force*, Particle*);
void scheme_move_clear_vel(int n, Particle*);

#include <stdio.h>
#include <mpi.h>

void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
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

struct Particle;
struct Config;
struct Restrain;

struct SchemeQQ {
    Particle *o, *r;
    int on, rn;
};

void scheme_restrain_ini(Restrain**);
void scheme_restrain_fin(Restrain*);

void scheme_restrain_set_red(Restrain*);
void scheme_restrain_set_rbc(Restrain*);
void scheme_restrain_set_none(Restrain*);
void scheme_restrain_set_freq(int freq, Restrain*);

void scheme_restrain_set_conf(const Config*, Restrain*);

void scheme_restrain_apply(MPI_Comm comm, const int *cc, long it,      Restrain *r, SchemeQQ qq);

#include <mpi.h>
#include <stdio.h>




#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif













#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
#endif

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
#ifndef UDX_SCHEME_CC_COMMON
#define UDX_SCHEME_CC_COMMON


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




#endif

#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}















#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <vector_types.h>



#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
#endif














#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
#include "device.cuh"
#endif
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






void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void os_mkdir(const char* path);
long os_time();
void os_srand(long int seedval);
double os_drand();
void os_sleep(unsigned int seconds);
void os_malloc_stats();


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





#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
struct Dbg;

struct Config;
struct Force;
struct Particle;
struct int3;
struct Coords;

enum {
    DBG_POS,
    DBG_POS_SOFT,
    DBG_VEL,
    DBG_FORCES,
    DBG_COLORS,
    DBG_CLIST,
    DBG_NKIND_
};

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

void msg_ini(int rank);
void msg_print(const char *fmt, ...);
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
struct Particle;
struct Positioncp;
struct float4;

void convert_pp2rr_current (long n, const Particle *pp, Positioncp *rr);
void convert_pp2rr_previous(long n, const Particle *pp, Positioncp *rr);
void convert_pp2f4_pos(int n, const float *pp,      float4 *zpp);

struct Coords;
struct Config;

void coords_ini(MPI_Comm cart, int Lx, int Ly, int Lz, Coords **c);
void coords_fin(Coords *c);

void coords_ini_conf(MPI_Comm cart, const Config *cfg, Coords **c);

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

void parray_push_pp(const Particle *pp, PaArray *a);
void parray_push_cc(const int *cc, PaArray *a);

bool parray_is_colored(const PaArray *a);

void parray_get_view(const PaArray *a, PaArray_v *v);
void parray_get_view(const PaArray *a, PaCArray_v *v);
#endif
#ifndef UDX_SIM_OBJECTS_FARRAY_COMMON
#define UDX_SIM_OBJECTS_FARRAY_COMMON
struct FoArray {
    bool stress;
    float *ff;
    float *ss;
};

struct FoArray_v;
struct FoSArray_v;

struct Force;

void farray_push_ff(Force *ff, FoArray *a);
void farray_push_ss(float *ss, FoArray *a);

void farray_clear(FoArray *);

bool farray_has_stress(const FoArray *a);

void farray_get_view(const FoArray *a, FoArray_v *v);
void farray_get_view(const FoArray *a, FoSArray_v *v);
#endif
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
#endif
enum {
    ALIVE = 0,
    DEAD  = 1
};




struct PartList {
    const Particle *pp;
    const int *deathlist;
};
struct Triangles {
    int4 *tt;
    int nt;
};
struct Triangles;
struct MeshRead;

void triangles_ini(MeshRead*,      Triangles**);
void triangles_fin(Triangles*);
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

struct Particle;
struct float3;
void minmax(const Particle *pp, int nv, int nobj,      float3 *lo, float3 *hi);

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
struct float3;
void io_com_dump(MPI_Comm, const Coords*, const char *name, long id, int n, const int *ii, const float3 *rr);
struct int4;
struct MeshRead;

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
#ifndef UDX_SIM_MESH_IO_COMMON
#define UDX_SIM_MESH_IO_COMMON
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

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory,      MeshWrite**);
void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead*, const char *directory,      MeshWrite**);

void mesh_write_particles(MeshWrite*, MPI_Comm, const Coords*, int nm, const Particle*, int id);
void mesh_write_vectors(MeshWrite*, MPI_Comm, int nm, Vectors *pos, Vectors *vel, int id);

void mesh_write_fin(MeshWrite*);
#endif
#ifndef UDX_SIM_RESTART_COMMON
#define UDX_SIM_RESTART_COMMON
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
#endif
struct Coords;
struct Rigid;

void io_rig_dump(MPI_Comm comm, const Coords *c, const char *name, long id, int ns, const Rigid *ss);

struct IoBop;

void io_bop_ini(MPI_Comm comm, int maxp, IoBop **t);
void io_bop_fin(IoBop *t);

void io_bop_parts       (MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const char *name, int id, IoBop *t);
void io_bop_parts_forces(MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const Force *ff, const char *name, int id, IoBop *t);
void io_bop_stresses    (MPI_Comm cart, long n, const float *ss, const char *name, int id, IoBop *t);
void io_bop_ids         (MPI_Comm cart, long n, const int   *ii, const char *name, int id, IoBop *t);
void io_bop_colors      (MPI_Comm cart, long n, const int   *cc, const char *name, int id, IoBop *t);

struct Config;
struct RbcParams;
struct RbcParams_v {
    float gammaC, gammaT, kBT;
    float kb, phi;
    float ks, x0, mpow;
    float ka, kd, kv;
    float totArea, totVolume;
};

void rbc_params_ini(RbcParams **);
void rbc_params_fin(RbcParams *);

void rbc_params_set_fluct(float gammaC, float gammaT, float kBT, RbcParams *);
void rbc_params_set_bending(float kb, float phi, RbcParams *);
void rbc_params_set_spring(float ks, float x0, float mpow , RbcParams *);
void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *);
void rbc_params_set_tot_area_volume(float totArea, float totVolume, RbcParams *);

double rbc_params_get_tot_volume(const RbcParams *);

void rbc_params_set_conf(const Config *c, const char *name, RbcParams *);

RbcParams_v rbc_params_get_view(const RbcParams *);
struct AreaVolume;
struct Particle;

struct RbcQuants {
    int n, nc;
    int nt, nv;
    int md;
    Particle *pp, *pp_hst;
    bool ids;
    int *ii;
    AreaVolume *area_volume;
};

struct Coords;
struct MeshRead;

void rbc_ini(long maxnc, bool ids, const MeshRead*, RbcQuants*);
void rbc_fin(RbcQuants*);

void rbc_gen_mesh(const Coords*, MPI_Comm, const MeshRead*, const char *ic,      RbcQuants*);
void rbc_gen_freeze(MPI_Comm,      RbcQuants*);
void rbc_strt_quants(MPI_Comm, const MeshRead*, const char *base, const char *name, const int id, RbcQuants*);
void rbc_strt_dump(MPI_Comm, const char *base, const char *name, int id, const RbcQuants *q);

struct RbcParams;
struct RbcForce;
struct Force;
struct Config;
struct MeshRead;
struct RbcQuants;

void rbc_force_ini(const MeshRead*, RbcForce**);
void rbc_force_fin(RbcForce*);

void rbc_force_set_stressful(int nt, float totArea,      RbcForce*);
void rbc_force_set_stressfree(const char *fname,      RbcForce*);

void rbc_force_set_rnd0(RbcForce*);
void rbc_force_set_rnd1(int seed, RbcForce*);
void rbc_force_set_bending(const MeshRead*, const char*, RbcForce*);

void rbc_force_set_conf(const MeshRead*, const Config *cfg, const char *name, RbcForce*);

void rbc_force_apply(RbcForce*, const RbcParams*, float dt, const RbcQuants*,      Force*);
void rbc_force_stat(     float *pArea, float *pVolume);
struct AreaVolume;
struct Particle;
struct int4;

void area_volume_ini(int nv, int nt, const int4 *tt, int max_cell, AreaVolume**);
void area_volume_fin(AreaVolume*);

void area_volume_compute(AreaVolume*, int nm, const Particle*,      float **av);
void area_volume_host(AreaVolume*,      float **av);
struct RbcStretch;
struct Force;

void rbc_stretch_ini(const char *path, int nv,      RbcStretch**);
void rbc_stretch_fin(RbcStretch*);
void rbc_stretch_apply(int nm, const RbcStretch*,      Force*);
struct RbcCom;
struct float3;
struct Particle;

void rbc_com_ini(int nv, int nm_max,      RbcCom**);
void rbc_com_fin(     RbcCom*);
void rbc_com_apply(RbcCom*, int nm, const Particle*,      float3 **rr, float3 **vv);

struct RigQuants {
    int n, ns, nps;
    Particle *pp_hst, *pp;
    Rigid    *ss_hst, *ss;
    float   *rr0_hst, *rr0;


    int nt, nv;
    int4 *dtt;
    float *dvv;
    Particle *i_pp_hst, *i_pp;

    Rigid *ss_dmp;

    long maxp;
};

struct Coords;
struct RigPinInfo;
struct MeshRead;

void rig_ini(long maxs, long maxp, const MeshRead*, RigQuants*);
void rig_fin(RigQuants *q);

void rig_gen_mesh(const Coords*, MPI_Comm, const MeshRead*, const char *ic,      RigQuants *);
void rig_gen_freeze(const Coords*, bool empty_pp, int numdensity, float rig_mass, const RigPinInfo*,
                    MPI_Comm, const MeshRead*, Particle *opp, int *on, RigQuants *q);
void rig_strt_quants(MPI_Comm, const MeshRead*, const char *base, const char *name, const int id, RigQuants *q);

void rig_strt_dump_templ(MPI_Comm, const char *base, const char *name, const RigQuants *q);
void rig_strt_dump(MPI_Comm, const char *base, const char *name, const int id, const RigQuants *q);
struct RigPinInfo;
struct int3;
struct Config;

enum {NOT_PERIODIC = -1};

void rig_pininfo_ini(RigPinInfo **);
void rig_pininfo_fin(RigPinInfo *);
void rig_pininfo_set(int3 com, int3 axis, RigPinInfo *);
void rig_pininfo_set_pdir(int pdir, RigPinInfo *);
void rig_pininfo_set_conf(const Config *cfg, const char *ns, RigPinInfo *);

int rig_pininfo_get_pdir(const RigPinInfo *);

struct Particle;
struct Rigid;
struct Force;

void rig_reinit_ft(const int ns,      Rigid *ss);
void rig_update(const RigPinInfo *pi, float dt, int n, const Force *ff, const float *rr0, int ns,      Particle *pp, Rigid *ss);
void rig_generate(int ns, const Rigid *ss, int nps, const float *rr0,      Particle *pp);
void rig_update_mesh(float dt, int ns, const Rigid *ss, int nv, const float *vv,      Particle *pp);


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

struct WallQuants {
    float4 *pp;
    int n;
    int3 L;
};

struct WallTicket;
struct WallRepulsePrm;


void wall_ini_quants(int3 L, WallQuants*);
void wall_ini_ticket(int3 L, WallTicket**);

void wall_fin_quants(WallQuants*);
void wall_fin_ticket(WallTicket*);

void wall_gen_quants(MPI_Comm, int numdensity, int maxn, const Sdf*,          int *o_n, Particle *o_pp,      WallQuants*);
void wall_gen_ticket(const WallQuants*, WallTicket*);

void wall_strt_quants(MPI_Comm, const char *base, int maxn, WallQuants*);
void wall_strt_dump_templ(MPI_Comm, const char *base, const WallQuants*);

void wall_force(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                const WallTicket*, int n, const PaArray*, const FoArray*);

void wall_force_adhesion(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                         const WallTicket*, int n, const PaArray*, const FoArray*);

void wall_repulse(const Sdf*, const WallRepulsePrm*, long n, const PaArray*, const FoArray*);

void wall_repulse_prm_ini(float lambda, WallRepulsePrm**);
void wall_repulse_prm_ini_conf(const Config*, const char *ns, WallRepulsePrm**);
void wall_repulse_prm_fin(WallRepulsePrm*);
#endif

struct Force;
struct Particle;

void scheme_move_apply(float dt, float mass, int n, const Force*, Particle*);
void scheme_move_clear_vel(int n, Particle*);
struct Particle;
struct Force;
struct BForce;
struct float3;
struct Config;
struct Coords;

void bforce_ini(BForce **p);
void bforce_fin(BForce *p);

void bforce_set_none(     BForce *p);
void bforce_set_cste(float3 f,      BForce *p);
void bforce_set_dp(float a,      BForce *p);
void bforce_set_shear(float a,      BForce *p);
void bforce_set_rol(float a,      BForce *p);
void bforce_set_rad(float a,      BForce *p);

void bforce_set_conf(const Config *cfg,      BForce *bf);

void bforce_adjust(float3 f,      BForce *fpar);
void bforce_apply(const Coords *c, float mass, const BForce *bf, int n, const Particle *pp,      Force *ff);
struct TimeStep;
struct TimeStepAccel;
struct Force;
struct Config;

void time_step_accel_ini(     TimeStepAccel**);
void time_step_accel_fin(TimeStepAccel*);

void time_step_accel_push(TimeStepAccel*, float m, int n, Force*);
void time_step_accel_reset(TimeStepAccel*);


void time_step_ini(const Config*,      TimeStep**);
void time_step_fin(TimeStep*);

float time_step_dt(TimeStep*, MPI_Comm, TimeStepAccel*);
float time_step_dt0(TimeStep*);
void time_step_log(TimeStep*);

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

struct DRbcPack;
struct DRbcComm;
struct DRbcUnpack;

void drbc_pack_ini(bool ids, int3 L, int maxc, int nv, DRbcPack **p);
void drbc_comm_ini(MPI_Comm comm,      DRbcComm **c);
void drbc_unpack_ini(bool ids, int3 L, int maxc, int nv, DRbcUnpack **u);

void drbc_pack_fin(DRbcPack *p);
void drbc_comm_fin(DRbcComm *c);
void drbc_unpack_fin(DRbcUnpack *u);

void drbc_build_map(int nc, int nv, const Particle *pp, DRbcPack *p);
void drbc_pack(const RbcQuants *q,      DRbcPack *p);
void drbc_download(DRbcPack *p);

void drbc_post_recv(DRbcComm *c, DRbcUnpack *u);
void drbc_post_send(DRbcPack *p, DRbcComm *c);
void drbc_wait_recv(DRbcComm *c, DRbcUnpack *u);
void drbc_wait_send(DRbcComm *c);

void drbc_unpack_bulk(const DRbcPack *p,      RbcQuants *q);
void drbc_unpack_halo(const DRbcUnpack *u,      RbcQuants *q);
struct DRigPack;
struct DRigComm;
struct DRigUnpack;

void drig_pack_ini(int3 L, int maxns, int nv, DRigPack **p);
void drig_comm_ini(MPI_Comm comm,      DRigComm **c);
void drig_unpack_ini(int3 L, int maxns, int nv, DRigUnpack **u);

void drig_pack_fin(DRigPack *p);
void drig_comm_fin(DRigComm *c);
void drig_unpack_fin(DRigUnpack *u);

void drig_build_map(int ns, const Rigid *ss,      DRigPack *p);
void drig_pack(int ns, int nv, const Rigid *ss, const Particle *ipp,      DRigPack *p);
void drig_download(DRigPack *p);

void drig_post_recv(DRigComm *c, DRigUnpack *u);
void drig_post_send(DRigPack *p, DRigComm *c);
void drig_wait_recv(DRigComm *c, DRigUnpack *u);
void drig_wait_send(DRigComm *c);

void drig_unpack_bulk(const DRigPack *p,      RigQuants *q);
void drig_unpack_halo(const DRigUnpack *u,      RigQuants *q);
struct EMeshPack;
struct EMeshComm;
struct EMeshUnpack;
struct EMeshPackM;
struct EMeshCommM;
struct EMeshUnpackM;



void emesh_pack_ini(int3 L, int nv, int max_mesh_num, EMeshPack **p);
void emesh_comm_ini(MPI_Comm comm,      EMeshComm **c);
void emesh_unpack_ini(int3 L, int nv, int max_mesh_num, EMeshUnpack **u);

void emesh_pack_fin(EMeshPack *p);
void emesh_comm_fin(EMeshComm *c);
void emesh_unpack_fin(EMeshUnpack *u);

void emesh_build_map(int nm, int nv, const Particle *pp,      EMeshPack *p);

void emesh_pack_pp(int nv, const Particle *pp,      EMeshPack *p);
void emesh_pack_rrcp(int nv, const Positioncp *rr,      EMeshPack *p);
void emesh_download(EMeshPack *p);

void emesh_post_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_post_send(EMeshPack *p, EMeshComm *c);
void emesh_wait_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_wait_send(EMeshComm *c);

void emesh_unpack_pp(int nv, const EMeshUnpack *u,      int *nmhalo, Particle *pp);
void emesh_unpack_rrcp(int nv, const EMeshUnpack *u,      int *nmhalo, Positioncp *rr);

void emesh_get_num_frag_mesh(const EMeshUnpack *u,      int cc[NFRAGS]);



void emesh_packm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshPackM **p);
void emesh_commm_ini(MPI_Comm comm,      EMeshCommM **c);
void emesh_unpackm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshUnpackM **u);

void emesh_packm_fin(EMeshPackM *p);
void emesh_commm_fin(EMeshCommM *c);
void emesh_unpackm_fin(EMeshUnpackM *u);

void emesh_packM(int nt, const int counts[NFRAGS], const Momentum *mm,      EMeshPackM *p);
void emesh_downloadM(const int counts[NFRAGS], EMeshPackM *p);

void emesh_post_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_post_send(EMeshPackM *p, EMeshCommM *c);
void emesh_wait_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_wait_send(EMeshCommM *c);

void emesh_upload(EMeshUnpackM *u);
void emesh_unpack_mom(int nt, const EMeshPack *p, const EMeshUnpackM *u,      Momentum *mm);

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

void mesh_bounce_ini(int maxpp,      MeshBB **d);
void mesh_bounce_fin(     MeshBB *d);

void mesh_bounce_reini(int n,      MeshBB *d);
void mesh_bounce_find_collisions(float dt, int nm, MeshInfo meshinfo, const Positioncp *i_rr, int3 L,
                                 const int *starts, const int *counts, const Particle *pp, const Particle *pp0,      MeshBB *d);
void mesh_bounce_select_collisions(int n,      MeshBB *d);
void mesh_bounce_bounce(float dt, float mass, int n, const MeshBB *d, MeshInfo meshinfo,
                        const Positioncp *i_rr, const Particle *pp0,      Particle *pp, Momentum *mm);

void mesh_bounce_collect_rig_momentum(int ns, MeshInfo meshinfo, const Particle *pp, const Momentum *mm,      Rigid *ss);
void mesh_bounce_collect_rbc_momentum(float mass, int nc, MeshInfo mi, const Momentum *mm,      Particle *pp);
struct float3;
struct Particle;
struct int4;
struct Triangles;

int collision_inside_1p(int spdir, const float *r, const float *vv, const int4 *tt, const int nt);

void collision_label_ini(long n, int lab_out,      int *labels);
void collision_label(int pdir, int n, const Particle *pp, const Triangles *tri,
                     int nv, int nm, const Particle *i_pp,
                     const float3 *minext, const float3 *maxext,
                     int lab_in,      int *labels);

struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);

void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*);
void sdf_get_view(const Sdf*,      Sdf_v*);

void sdf_bulk_wall(const Sdf*,        int *s_n, Particle *s_pp,       int *w_n, Particle *w_pp);
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv,      int *stay);
double sdf_compute_volume(MPI_Comm, int3 L, const Sdf*, long nsamples);

void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n,      Particle*);

#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON

struct OptFlu {
    bool colors, ids, ss, push;
};


struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};


struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};


struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};


struct OptParams {
    int3 L;
    float kBT;
    int numdensity;
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
#endif

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

void objects_ini(const Config*, const Opt*, MPI_Comm, const Coords*, int maxp, Objects**);
void objects_fin(Objects*);

void objects_save_mesh(Objects*);
void objects_clear_vel(Objects*);
void objects_advance(float dt, Objects*);
void objects_distribute(Objects*);
void objects_update_dpd_prms(float dt, float kBT, Objects*);

void objects_clear_forces(Objects*);
void objects_body_forces(const BForce*, Objects *o);

void objects_mesh_dump(Objects*);
void objects_diag_dump(float t, Objects*);
void objects_part_dump(long id, Objects*, IoBop*);

void objects_strt_templ(const char *base, Objects*);
void objects_strt_dump(const char *base, long id, Objects*);

bool objects_have_bounce(const Objects*);

void objects_get_particles_all(Objects*, PFarrays*);
void objects_get_particles_mbr(Objects*, PFarrays*);
void objects_get_accel(const Objects*, TimeStepAccel*);

void objects_get_params_fsi(const Objects*, const PairParams*[]);
void objects_get_params_adhesion(const Objects*, const PairParams*[]);
void objects_get_params_repulsion(const Objects*, const WallRepulsePrm*[]);
void objects_get_params_cnt(const Objects*, const PairParams***);

void objects_gen_mesh(Objects*);
void objects_remove_from_wall(const Sdf *sdf, Objects *o);
void objects_gen_freeze(PFarray*, Objects*);

void objects_restart(Objects*);

void objects_activate_mbr_bounce(Objects*);
void objects_bounce(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, Objects *obj);
void objects_recolor_flu(Objects*, PFarray *flu);
double objects_mbr_tot_volume(const Objects*);

void objects_check_size(const Objects*);
void objects_check_vel(const Objects*, const Dbg*, float dt);
void objects_check_forces(const Objects*, const Dbg*, float dt);


struct MbrDistr {
    DRbcPack *p;
    DRbcComm *c;
    DRbcUnpack *u;
};


struct RigDistr {
    DRigPack *p;
    DRigComm *c;
    DRigUnpack *u;
};


struct MeshExch {
    EMeshPack *p;
    EMeshComm *c;
    EMeshUnpack *u;
};


struct MeshMomExch {
    EMeshPackM *p;
    EMeshCommM *c;
    EMeshUnpackM *u;
};


struct Colorer {
    Particle *pp_mesh;
    float3 *lo, *hi;
};


struct BounceBackData {
    Positioncp *rr_cp;
    MeshMomExch *e;
    Momentum *mm;
};


struct Obj {
    char name[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    float mass;
    MeshRead   *mesh;
    MeshWrite  *mesh_write;
    MeshExch   *mesh_exch;
    BounceBackData *bbdata;

    PairParams *fsi;
    PairParams *adhesion;
    WallRepulsePrm *wall_rep_prm;
};


struct Mbr : Obj {
    RbcQuants q;
    MbrDistr d;
    Force *ff;
    Force *ff_fast;
    RbcForce   *force;
    RbcParams  *params;
    RbcCom     *com;
    RbcStretch *stretch;
    Triangles  *tri;

    bool active_bounce;
    Colorer  *colorer;
};


struct Rig : Obj {
    RigQuants q;
    Force *ff;

    RigPinInfo *pininfo;
    RigDistr d;
};

struct Dump {
    Particle *pp;
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
    bool active;
    PairParams **cnt_params;
};
static void gen_name_mesh_dir(const char *name, char *mesh_dir) {
    strcpy(mesh_dir, "ply/");
    strcat(mesh_dir, name);
}

static void ini_mbr_distr(bool ids, int nv, MPI_Comm comm, int3 L,      MbrDistr *d) {
    UC(drbc_pack_ini(ids, L, MAX_CELL_NUM, nv,      &d->p));
    UC(drbc_comm_ini(comm,      &d->c));
    UC(drbc_unpack_ini(ids, L, MAX_CELL_NUM, nv,      &d->u));
}

static void ini_rig_distr(int nv, MPI_Comm comm, int3 L,      RigDistr *d) {
    UC(drig_pack_ini(L, MAX_SOLIDS, nv,      &d->p));
    UC(drig_comm_ini(comm,      &d->c));
    UC(drig_unpack_ini(L,MAX_SOLIDS, nv,      &d->u));
}

static void ini_mesh_exch(int3 L, int nv, int max_m, MPI_Comm comm,      MeshExch **me) {
    MeshExch *e;
    EMALLOC(1, me);
    e = *me;
    UC(emesh_pack_ini(L, nv, max_m,      &e->p));
    UC(emesh_comm_ini(comm,      &e->c));
    UC(emesh_unpack_ini(L, nv, max_m,      &e->u));
}

static void ini_mesh_mom_exch(int nt, int max_m, MPI_Comm comm,      MeshMomExch **mme) {
    MeshMomExch *e;
    EMALLOC(1, mme);
    e = *mme;
    UC(emesh_packm_ini(nt, max_m,      &e->p));
    UC(emesh_commm_ini(comm,      &e->c));
    UC(emesh_unpackm_ini(nt, max_m,      &e->u));
}

static void ini_bbdata(int nt, int nv, int max_m, MPI_Comm cart,      BounceBackData **bbdata) {
    BounceBackData *bb;
    EMALLOC(1, bbdata);
    bb = *bbdata;
    UC(ini_mesh_mom_exch(nt, max_m, cart, &bb->e));
    Dalloc(&bb->mm,    max_m * nt);
    Dalloc(&bb->rr_cp, max_m * nv);
}

static void ini_colorer(int nv, int max_m,      Colorer **col) {
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
    UC(mesh_write_ini_from_mesh(cart, opt->shifttype, obj->mesh, mesh_dir,      &obj->mesh_write));

    obj->mass = opt->mass;

    UC(ini_params(cfg, obj->name, "fsi", &obj->fsi));
    UC(ini_params(cfg, obj->name, "adhesion", &obj->adhesion));
    UC(ini_repulsion_params(cfg, obj->name, &obj->wall_rep_prm));

    nt = mesh_read_get_nt(obj->mesh);
    nv = mesh_read_get_nv(obj->mesh);

    obj->mesh_exch = NULL;
    obj->bbdata    = NULL;

    if (opt->bounce) {
        UC(ini_bbdata(nt, nv, max_m, cart,      &obj->bbdata));
        UC(ini_mesh_exch(L, nv, max_m, cart,      &obj->mesh_exch));
    }
}

static void ini_mbr(const Config *cfg, const OptMbr *opt, MPI_Comm cart, int3 L,
                    bool recolor,      Mbr **membrane) {
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

    UC(triangles_ini(m->mesh,      &m->tri));
    UC(rbc_ini(max_m, opt->ids, m->mesh, &m->q));
    UC(ini_mbr_distr(opt->ids, nv, cart, L,      &m->d));

    if (opt->dump_com) UC(rbc_com_ini(nv, max_m,      &m->com));
    if (opt->stretch)  UC(rbc_stretch_ini(opt->stretch_file, nv,      &m->stretch));

    UC(rbc_params_ini(&m->params));
    UC(rbc_params_set_conf(cfg, m->name, m->params));

    UC(rbc_force_ini(m->mesh,      &m->force));
    UC(rbc_force_set_conf(m->mesh, cfg, m->name, m->force));

    if (recolor) {
        if (!m->mesh_exch)
            UC(ini_mesh_exch(L, nv, max_m, cart,      &m->mesh_exch));
        UC(ini_colorer(nv, max_m,      &m->colorer));
    }
}

static void ini_rig(const Config *cfg, const OptRig *opt, MPI_Comm cart, int maxp, int3 L,      Rig **rigid) {
    Rig *r;
    long max_m = MAX_SOLIDS;
    int nv;
    EMALLOC(1, rigid);
    r = *rigid;

    UC(ini_obj_common(cfg, opt, cart, max_m, L, r));

    UC(rig_ini(max_m, maxp, r->mesh, &r->q));

    Dalloc(&r->ff, maxp);

    nv = r->q.nv;

    UC(ini_rig_distr(nv, cart, L,      &r->d));

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
        UC(mesh_bounce_ini(maxp,      &obj->bb));

    UC(ini_dump(maxp, &obj->dump));
    UC(ini_cnt_params(cfg, obj->nmbr, obj->nrig, obj->mbr, obj->rig, &obj->cnt_params));
}
static void fin_mbr_distr(     MbrDistr *d) {
    UC(drbc_pack_fin(     d->p));
    UC(drbc_comm_fin(     d->c));
    UC(drbc_unpack_fin(     d->u));
}

static void fin_rig_distr(     RigDistr *d) {
    UC(drig_pack_fin(     d->p));
    UC(drig_comm_fin(     d->c));
    UC(drig_unpack_fin(     d->u));
}

static void fin_mesh_exch(     MeshExch *e) {
    UC(emesh_pack_fin(        e->p));
    UC(emesh_comm_fin(        e->c));
    UC(emesh_unpack_fin(      e->u));
    EFREE(e);
}

static void fin_mesh_mom_exch(     MeshMomExch *e) {
    UC(emesh_packm_fin(        e->p));
    UC(emesh_commm_fin(        e->c));
    UC(emesh_unpackm_fin(      e->u));
    EFREE(e);
}

static void fin_bbdata(     BounceBackData *bb) {
    UC(fin_mesh_mom_exch(     bb->e));
    Dfree(bb->mm);
    Dfree(bb->rr_cp);
    EFREE(bb);
}

static void fin_colorer(     Colorer *c) {
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

    UC(fin_mbr_distr(     &m->d));

    Dfree(m->ff);
    Dfree(m->ff_fast);
    UC(triangles_fin(m->tri));

    if (m->com)     UC(rbc_com_fin(     m->com));
    if (m->stretch) UC(rbc_stretch_fin(     m->stretch));

    UC(rbc_params_fin(m->params));

    if (m->colorer)   UC(fin_colorer(     m->colorer));
    if (m->mesh_exch) UC(fin_mesh_exch(     m->mesh_exch));

    UC(fin_obj_common(m));

    EFREE(m);
}

static void fin_rig(Rig *r) {
    UC(rig_fin(&r->q));
    Dfree(r->ff);

    UC(fin_rig_distr(     &r->d));
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
    if (obj->bb)  UC(mesh_bounce_fin(     obj->bb));
    UC(fin_cnt_params(obj->nmbr, obj->nrig, obj->cnt_params));
    UC(fin_dump(obj->dump));
    UC(coords_fin(obj->coords));
    MC(m::Comm_free(&obj->cart));
    EFREE(obj);
}
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
static void clear_forces(int n, Force* ff) {
    if (n) DzeroA(ff, n);
}

static void clear_mbr_forces(Mbr *m) {
    UC(clear_forces(m->q.n, m->ff));
}

static void clear_rig_forces(Rig *r) {
    UC(clear_forces(r->q.n, r->ff));
    UC(rig_reinit_ft(r->q.ns,      r->q.ss));
}

void objects_clear_forces(Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(clear_mbr_forces(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(clear_rig_forces(obj->rig[i]));
}


static void internal_forces_mbr(float dt, const OptMbr *opt, Mbr *m) {
    UC(rbc_force_apply(m->force, m->params, dt, &m->q,      m->ff_fast));
    if (opt->stretch) UC(rbc_stretch_apply(m->q.nc, m->stretch,      m->ff_fast));
}

static void bforce_mbr(const Coords *c, const BForce *bf, Mbr *m) {
    UC(bforce_apply(c, m->mass, bf, m->q.n, m->q.pp,      m->ff));
}

static void bforce_rig(const Coords *c, const BForce *bf, Rig *r) {
    UC(bforce_apply(c, r->mass, bf, r->q.n, r->q.pp,      r->ff));
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
    UC(rig_update(r->pininfo, dt, q->n, r->ff, q->rr0, q->ns,      q->pp, q->ss));
    UC(rig_update_mesh(dt, q->ns, q->ss, q->nv, q->dvv,      q->i_pp));
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

    UC(drbc_build_map(q->nc, q->nv, q->pp,      d->p));
    UC(drbc_pack(q,      d->p));
    UC(drbc_download(    d->p));

    UC(drbc_post_send(d->p, d->c));
    UC(drbc_post_recv(d->c, d->u));

    UC(drbc_unpack_bulk(d->p,      q));

    UC(drbc_wait_send(d->c));
    UC(drbc_wait_recv(d->c, d->u));

    UC(drbc_unpack_halo(d->u,      q));
    dSync();
}

static void distribute_rig(Rig *r) {
    RigQuants *q = &r->q;
    RigDistr  *d = &r->d;
    int nv = q->nv;

    UC(drig_build_map(q->ns, q->ss,      d->p));
    UC(drig_pack(q->ns, nv, q->ss, q->i_pp,      d->p));
    UC(drig_download(    d->p));

    UC(drig_post_send(d->p, d->c));
    UC(drig_post_recv(d->c, d->u));

    UC(drig_unpack_bulk(d->p,      q));

    UC(drig_wait_send(d->c));
    UC(drig_wait_recv(d->c, d->u));

    UC(drig_unpack_halo(d->u,      q));

    q->n = q->ns * q->nps;
    UC(rig_generate(q->ns, q->ss, q->nps, q->rr0,      q->pp));
    dSync();
}

void objects_distribute (Objects *obj) {
    int i;
    if (!obj->active) return;
    for (i = 0; i < obj->nmbr; ++i) UC(distribute_mbr(obj->mbr[i]));
    for (i = 0; i < obj->nrig; ++i) UC(distribute_rig(obj->rig[i]));
}

static void update_dpd_prms(float dt, float kBT, Obj *o) {
    if (o->fsi) UC(pair_compute_dpd_sigma(kBT, dt,      o->fsi));
}

void objects_update_dpd_prms(float dt, float kBT, Objects *obj) {
    int i, n, ninter;
    for (i = 0; i < obj->nmbr; ++i) update_dpd_prms(dt, kBT, obj->mbr[i]);
    for (i = 0; i < obj->nrig; ++i) update_dpd_prms(dt, kBT, obj->rig[i]);

    n = obj->nmbr + obj->nrig;
    ninter = n*(n+1)/2;
    for (i = 0; i < ninter; ++i) {
        if (obj->cnt_params[i])
            UC(pair_compute_dpd_sigma(kBT, dt,      obj->cnt_params[i]));
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

    UC(emesh_build_map(q->nc, q->nv, q->pp,      e->p));
    UC(emesh_pack_rrcp(q->nv, m->bbdata->rr_cp,      e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));
}

static void mesh_pack_and_send_rig(Rig *r) {
    const RigQuants *q = &r->q;
    MeshExch  *e = r->mesh_exch;

    UC(emesh_build_map(q->ns, q->nv, q->i_pp,      e->p));
    UC(emesh_pack_rrcp(q->nv, r->bbdata->rr_cp,      e->p));
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


    UC(emesh_unpack_rrcp(q->nv, e->u,      &nmhalo, m->bbdata->rr_cp + q->n));
    return nmhalo;
}

static int mesh_recv_unpack_rig(Rig *r) {
    MeshExch *e = r->mesh_exch;
    RigQuants *q = &r->q;
    int nmhalo;
    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));


    UC(emesh_unpack_rrcp(q->nv, e->u,      &nmhalo, r->bbdata->rr_cp + q->ns * q->nv));
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
                                   flu_pp, flu_pp0,      bb));
    UC(mesh_bounce_select_collisions(nflu,      bb));
}

static void find_and_select_collisions_mbr(float dt, long nflu, const Clist *cflu, const Particle *flu_pp, const Particle *flu_pp0, int nm, Mbr *m, MeshBB *bb) {
    MeshInfo mi = mesh_info_mbr(m);
    UC(mesh_bounce_find_collisions(dt, nm, mi, m->bbdata->rr_cp,
                                   clists_get_dim(cflu),
                                   clists_get_ss(cflu),
                                   clists_get_cc(cflu),
                                   flu_pp, flu_pp0,      bb));
    UC(mesh_bounce_select_collisions(nflu,      bb));
}

static void bounce_mbr(float dt, float flu_mass, const MeshBB *bb, long n, const Particle *pp0, Particle *pp, Mbr *m) {
    MeshInfo mi = mesh_info_mbr(m);
    UC(mesh_bounce_bounce(dt, flu_mass, n, bb, mi, m->bbdata->rr_cp, pp0,      pp, m->bbdata->mm));
}

static void bounce_rig(float dt, float flu_mass, const MeshBB *bb, long n, const Particle *pp0, Particle *pp, Rig *r) {
    MeshInfo mi = mesh_info_rig(r);
    UC(mesh_bounce_bounce(dt, flu_mass, n, bb, mi, r->bbdata->rr_cp, pp0,      pp, r->bbdata->mm));
}

static void mom_pack_and_send_mbr(Mbr *m) {
    const RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshMomExch *e = bb->e;
    int counts[NFRAGS];

    UC(emesh_get_num_frag_mesh(m->mesh_exch->u,      counts));
    UC(emesh_packM(q->nt, counts, bb->mm + q->n,      e->p));
    UC(emesh_downloadM(counts,      e->p));

    UC(emesh_post_recv(e->c, e->u));
    UC(emesh_post_send(e->p, e->c));
}

static void mom_pack_and_send_rig(Rig *r) {
    const RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshMomExch *e = bb->e;
    int counts[NFRAGS];

    UC(emesh_get_num_frag_mesh(r->mesh_exch->u,      counts));
    UC(emesh_packM(q->nt, counts, bb->mm + q->ns * q->nt,      e->p));
    UC(emesh_downloadM(counts,      e->p));

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
    UC(emesh_unpack_mom(q->nt, meshe->p, e->u,      bb->mm));
}

static void mom_recv_unpack_rig(Rig *r) {
    const RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshMomExch *e = bb->e;
    MeshExch *meshe = r->mesh_exch;

    UC(emesh_wait_recv(e->c, e->u));
    UC(emesh_wait_send(e->c));

    UC(emesh_upload(e->u));
    UC(emesh_unpack_mom(q->nt, meshe->p, e->u,      bb->mm));
}

static void collect_mom_mbr(float dt, Mbr *m) {
    RbcQuants *q = &m->q;
    BounceBackData *bb = m->bbdata;
    MeshInfo mi = mesh_info_mbr(m);

    UC(mesh_bounce_collect_rbc_momentum(m->mass, q->nc, mi, bb->mm,      q->pp));
}

static void collect_mom_rig(Rig *r) {
    RigQuants *q = &r->q;
    BounceBackData *bb = r->bbdata;
    MeshInfo mi = mesh_info_rig(r);

    UC(mesh_bounce_collect_rig_momentum(q->ns, mi, q->i_pp, bb->mm,      q->ss));
}


static void bounce_mbr(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, MeshBB *bb, Mbr *m) {
    int nmhalo;
    if (!m->bbdata || !m->active_bounce) return;

    mesh_pack_and_send_mbr(m);
    nmhalo = mesh_recv_unpack_mbr(m);

    UC(mesh_bounce_reini(n,      bb));
    clear_momentum_mbr(nmhalo, m);

    find_and_select_collisions_mbr(dt, n, flu_cells, flu_pp, flu_pp0, nmhalo + m->q.nc, m, bb);

    bounce_mbr(dt, flu_mass, bb, n, flu_pp0,      flu_pp, m);

    mom_pack_and_send_mbr(m);
    mom_recv_unpack_mbr(m);
    collect_mom_mbr(dt, m);
}

static void bounce_rig(float dt, float flu_mass, const Clist *flu_cells, long n, const Particle *flu_pp0, Particle *flu_pp, MeshBB *bb, Rig *r) {
    int nmhalo;
    if (!r->bbdata) return;

    mesh_pack_and_send_rig(r);
    nmhalo = mesh_recv_unpack_rig(r);

    UC(mesh_bounce_reini(n,      bb));
    clear_momentum_rig(nmhalo, r);

    find_and_select_collisions_rig(dt, n, flu_cells, flu_pp, flu_pp0, nmhalo + r->q.ns, r, bb);

    bounce_rig(dt, flu_mass, bb, n, flu_pp0,      flu_pp, r);

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
static void reini_colors(PFarray *flu) {
    UC(collision_label_ini(flu->n, BLUE_COLOR,      (int*) flu->p.cc));
}

static int exchange_mesh_mbr(Mbr *m, Particle *pp) {
    int nm, nv, nmhalo;
    const RbcQuants *q = &m->q;
    MeshExch *e = m->mesh_exch;
    nm = q->nc;
    nv = q->nv;

    UC(emesh_build_map(nm, nv, q->pp,      e->p));
    UC(emesh_pack_pp(nv, q->pp,      e->p));
    UC(emesh_download(e->p));

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));

    if (nm*nv) aD2D(pp, q->pp, nm*nv);

    UC(emesh_wait_send(e->c));
    UC(emesh_wait_recv(e->c, e->u));

    UC(emesh_unpack_pp(nv, e->u,      &nmhalo, pp + nm * nv));
    return nm + nmhalo;
}

static void recolor_flu_from_mbr(Mbr *m, PFarray *flu) {
    int nm, nv;
    Colorer *c = m->colorer;
    if (c == NULL)
        ERR("colorer was not initialised");

    nv = m->q.nv;
    nm = exchange_mesh_mbr(m, c->pp_mesh);

    UC(minmax(c->pp_mesh, nv, nm,      c->lo, c->hi));

    UC(collision_label(NOT_PERIODIC, flu->n, (const Particle*) flu->p.pp, m->tri, nv, nm,
                       c->pp_mesh, c->lo, c->hi, RED_COLOR,      (int*) flu->p.cc));
}

void objects_recolor_flu(Objects *obj, PFarray *flu) {
    int i;
    if (!obj->active) return;
    if (!obj->nmbr) return;
    if (flu->n == 0) return;

    UC(reini_colors(flu));
    for (i = 0; i < obj->nmbr; ++i) UC(recolor_flu_from_mbr(obj->mbr[i], flu));
}
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
        UC(rbc_com_apply(m->com, nc, q->pp,      &rr, &vv));
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
static void gen_mesh_mbr(Coords *coords, MPI_Comm cart, Mbr *m) {
    MeshRead *mesh = m->mesh;
    UC(rbc_gen_mesh(coords, cart, mesh, m->ic_file,      &m->q));
}

static void gen_mesh_rig(Coords *coords, MPI_Comm cart, Rig *r) {
    MeshRead *mesh = r->mesh;
    UC(rig_gen_mesh(coords, cart, mesh, r->ic_file,      &r->q));
}

void objects_gen_mesh(Objects *o) {
    int i;
    for (i = 0; i < o->nmbr; ++i) gen_mesh_mbr(o->coords, o->cart, o->mbr[i]);
    for (i = 0; i < o->nrig; ++i) gen_mesh_rig(o->coords, o->cart, o->rig[i]);
}

template <typename T>
static void remove(T *data, int nv, int *stay, int nc) {
    int c;
    for (c = 0; c < nc; c++)
        cA2A(data + nv*c, data + nv * stay[c], nv);
}

static void remove_mbr(const Sdf *sdf, Mbr *m) {
    int nc0, stay[MAX_CELL_NUM];
    RbcQuants *q = &m->q;
    q->nc = sdf_who_stays(sdf, q->n, q->pp, nc0 = q->nc, q->nv,      stay);
    q->n = q->nc * q->nv;
    remove(q->pp, q->nv, stay, q->nc);
    msg_print("%s: %d/%d survived", m->name, q->nc, nc0);
}

static void remove_rig(const Sdf *sdf, Rig *r) {
    int nip, ns0, stay[MAX_SOLIDS];
    RigQuants *q = &r->q;
    nip = q->ns * q->nv;
    q->ns = sdf_who_stays(sdf, nip, q->i_pp, ns0 = q->ns, q->nv,      stay);
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
    UC(rbc_gen_freeze(cart,      &m->q));
}

static void gen_freeze_rig(const Coords *coords, MPI_Comm cart, bool empty_pp, int numdensity, PFarray *flu, Rig *r) {
    const MeshRead *mesh = r->mesh;
    int n;
    Particle *pp = (Particle*) flu->p.pp;
    n = flu->n;

    UC(rig_gen_freeze(coords, empty_pp, numdensity, r->mass, r->pininfo,
                      cart, mesh, pp, &n,      &r->q));

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





#include <stdio.h>
#include <mpi.h>
#include <vector_types.h>



#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
#endif

void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();


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

void parray_push_pp(const Particle *pp, PaArray *a);
void parray_push_cc(const int *cc, PaArray *a);

bool parray_is_colored(const PaArray *a);

void parray_get_view(const PaArray *a, PaArray_v *v);
void parray_get_view(const PaArray *a, PaCArray_v *v);
#endif
#ifndef UDX_SIM_OBJECTS_FARRAY_COMMON
#define UDX_SIM_OBJECTS_FARRAY_COMMON
struct FoArray {
    bool stress;
    float *ff;
    float *ss;
};

struct FoArray_v;
struct FoSArray_v;

struct Force;

void farray_push_ff(Force *ff, FoArray *a);
void farray_push_ss(float *ss, FoArray *a);

void farray_clear(FoArray *);

bool farray_has_stress(const FoArray *a);

void farray_get_view(const FoArray *a, FoArray_v *v);
void farray_get_view(const FoArray *a, FoSArray_v *v);
#endif
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
#endif

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

struct Contact;
struct int3;
struct PairParams;

void cnt_ini(int maxp, int rank, int3 L, int nobj,      Contact **c);
void cnt_fin(Contact *c);

void cnt_build_cells(int nw, const PairParams **prms, const PaWrap *pw,      Contact *c);
void cnt_bulk(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw);
void cnt_halo(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw, Pap26 all_pp, Fop26 all_ff, const int *all_counts);
struct RNDunif;
struct Fsi;
struct int3;

void fsi_ini(int rank, int3 L,      Fsi **fsi);
void fsi_fin(Fsi *fsi);

void fsi_bind_solvent(PaArray pa, Force *ff, int n, const int *starts,      Fsi *fsi);
void fsi_bulk(Fsi *fsi, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw);
void fsi_halo(Fsi *fsi, int nw, const PairParams **prms, Pap26 all_pp, Fop26 all_ff, const int *all_counts);

struct EObjPack;
struct EObjComm;
struct EObjUnpack;

struct EObjPackF;
struct EObjCommF;
struct EObjUnpackF;

void eobj_pack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjPack **p);
void eobj_comm_ini(MPI_Comm cart,      EObjComm **c);
void eobj_unpack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjUnpack **u);

void eobj_pack_fin(EObjPack *p);
void eobj_comm_fin(EObjComm *c);
void eobj_unpack_fin(EObjUnpack *u);

void eobj_build_map(int nw, const PaWrap *ww,      EObjPack *p);

void eobj_pack(int nw, const PaWrap *ww,      EObjPack *p);
void eobj_download(int nw, EObjPack *p);

void eobj_post_recv(EObjComm *c, EObjUnpack *u);
void eobj_post_send(EObjPack *p, EObjComm *c);
void eobj_wait_recv(EObjComm *c, EObjUnpack *u);
void eobj_wait_send(EObjComm *c);

int26 eobj_get_counts(EObjUnpack *u);
void  eobj_get_all_counts(int nw, EObjUnpack *u, int *all_counts);
Pap26 eobj_upload_shift(EObjUnpack *u);
Fop26 eobj_reini_ff(const EObjUnpack *u, EObjPackF *pf);


void eobj_packf_ini(int3 L, int maxd, int maxpsolid, EObjPackF **p);
void eobj_commf_ini(MPI_Comm cart,      EObjCommF **c);
void eobj_unpackf_ini(int3 L, int maxd, int maxpsolid, EObjUnpackF **u);

void eobj_packf_fin(EObjPackF *p);
void eobj_commf_fin(EObjCommF *c);
void eobj_unpackf_fin(EObjUnpackF *u);

void eobj_download_ff(EObjPackF *p);

void eobj_post_recv_ff(EObjCommF *c, EObjUnpackF *u);
void eobj_post_send_ff(EObjPackF *p, EObjCommF *c);
void eobj_wait_recv_ff(EObjCommF *c, EObjUnpackF *u);
void eobj_wait_send_ff(EObjCommF *c);

void eobj_unpack_ff(EObjUnpackF *u, const EObjPack *p, int nw,      FoWrap *ww);

#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON

struct OptFlu {
    bool colors, ids, ss, push;
};


struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};


struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};


struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};


struct OptParams {
    int3 L;
    float kBT;
    int numdensity;
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
#endif

struct ObjInter;

struct Config;
struct Opt;
struct PFarray;
struct PFarrays;

void obj_inter_ini(const Config*, const Opt*, MPI_Comm, float dt, int maxp,      ObjInter**);
void obj_inter_fin(ObjInter*);

void obj_inter_forces(ObjInter*, const PairParams **fsi_prms, const PairParams **cnt_prms, PFarray *flu, const int *flu_start, PFarrays *obj);

struct ObjExch {
    EObjPack *p;
    EObjComm *c;
    EObjUnpack *u;

    EObjPackF *pf;
    EObjCommF *cf;
    EObjUnpackF *uf;
};


struct ObjInter {
    ObjExch *e;
    Contact *cnt;
    Fsi     *fsi;
};

static void ini_obj_exch(MPI_Comm comm, int3 L,      ObjExch **oe) {
    ObjExch *e;
    int maxpsolid = MAX_PSOLID_NUM;
    EMALLOC(1, oe);
    e = *oe;

    UC(eobj_pack_ini   (L, MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid,      &e->p));
    UC(eobj_comm_ini   (comm,                                              &e->c));
    UC(eobj_unpack_ini (L, MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid,      &e->u));

    UC(eobj_packf_ini  (L, MAX_OBJ_DENSITY, maxpsolid,      &e->pf));
    UC(eobj_commf_ini  (comm,                               &e->cf));
    UC(eobj_unpackf_ini(L, MAX_OBJ_DENSITY, maxpsolid,      &e->uf));
}

void obj_inter_ini(const Config *cfg, const Opt *opt, MPI_Comm cart, float dt, int maxp,      ObjInter **oi) {
    int rank;
    ObjInter *o;
    int3 L = opt->params.L;
    EMALLOC(1, oi);
    o = *oi;

    MC(m::Comm_rank(cart, &rank));
    UC(ini_obj_exch(cart, L, &o->e));

    o->cnt = NULL;
    o->fsi = NULL;

    int nobj = opt->nmbr + opt->nrig;
    UC(cnt_ini(maxp, rank, L, nobj,      &o->cnt));
    UC(fsi_ini(rank, L,      &o->fsi));
}


static void fin_obj_exch(     ObjExch *e) {
    UC(eobj_pack_fin   (     e->p));
    UC(eobj_comm_fin   (     e->c));
    UC(eobj_unpack_fin (     e->u));

    UC(eobj_packf_fin  (     e->pf));
    UC(eobj_commf_fin  (     e->cf));
    UC(eobj_unpackf_fin(     e->uf));
    EFREE(e);
}

void obj_inter_fin(ObjInter *o) {
    UC(fin_obj_exch(o->e));
    if (o->cnt) UC(cnt_fin(o->cnt));
    if (o->fsi) UC(fsi_fin(o->fsi));
    EFREE(o);
}
static void fill_wrappers(PFarrays *obj, int *nwrappers, PaWrap *pw, FoWrap *fw) {
    long i, nw, n;
    PaArray p;
    FoArray f;

    (*nwrappers) = nw = pfarrays_size(obj);

    if (nw > MAX_OBJ_TYPES)
        ERR("Too many objects in wrappers : %d/%d", nw, MAX_OBJ_TYPES);

    for (i = 0; i < nw; ++i, ++pw, ++fw) {
        UC(pfarrays_get(i, obj, &n, &p, &f));
        pw->n = fw->n = n;
        pw->pp = (const Particle*) p.pp;
        fw->ff = (Force*) f.ff;
    }
}

static void bind_solvent(PFarray *flu, const int *starts, Fsi *fsi) {
    UC(fsi_bind_solvent(flu->p, (Force*) flu->f.ff, flu->n, starts,      fsi));
}

static void forces_cnt(ObjInter *oi, int nw, const PairParams *prms[], PaWrap *pw, FoWrap *fw) {
    cnt_build_cells(  nw, prms, pw,      oi->cnt);
    cnt_bulk(oi->cnt, nw, prms, pw, fw);
}

static void forces_fsi(ObjInter *oi, int nw, const PairParams *prms[], PaWrap *pw, FoWrap *fw) {
    fsi_bulk(oi->fsi, nw, prms, pw, fw);
}

static bool has_fsi_work(int nw, const PairParams **prms) {
    for (int i = 0; i < nw; ++i)
        if (prms[i]) return true;
    return false;
}

static bool has_work(const ObjInter *o, int nw, const PairParams **fsi_prms) {
    return o->cnt || has_fsi_work(nw, fsi_prms);
}

void obj_inter_forces(ObjInter *oi, const PairParams **fsi_prms, const PairParams **cnt_prms, PFarray *flu, const int *flu_start, PFarrays *obj) {
    PaWrap pw[MAX_OBJ_TYPES];
    FoWrap fw[MAX_OBJ_TYPES];
    int nw = 0;
    ObjExch *e = oi->e;
    int all_counts[26 * MAX_OBJ_TYPES] = {0};

    UC(fill_wrappers(obj,      &nw, pw, fw));
    if (!nw) return;
    if (!has_work(oi, nw, fsi_prms)) return;



    UC(eobj_build_map(nw, pw,      e->p));
    UC(eobj_pack(nw, pw,      e->p));
    UC(eobj_download(nw,      e->p));

    UC(eobj_post_send(e->p, e->c));
    UC(eobj_post_recv(e->c, e->u));



    if (oi->fsi) UC(bind_solvent(flu, flu_start, oi->fsi));

    if (oi->fsi) UC(forces_fsi(oi, nw, fsi_prms, pw, fw));
    if (oi->cnt) UC(forces_cnt(oi, nw, cnt_prms, pw, fw));



    UC(eobj_wait_send(e->c));
    UC(eobj_wait_recv(e->c, e->u));

    eobj_get_all_counts(nw, e->u,      all_counts);
    Pap26 hpp = eobj_upload_shift(e->u);
    Fop26 hff = eobj_reini_ff(e->u, e->pf);

    if (oi->fsi) UC(fsi_halo(oi->fsi, nw, fsi_prms,         hpp, hff, all_counts));
    if (oi->cnt) UC(cnt_halo(oi->cnt, nw, cnt_prms, pw, fw, hpp, hff, all_counts));



    UC(eobj_download_ff(e->pf));

    UC(eobj_post_send_ff(e->pf, e->cf));
    UC(eobj_post_recv_ff(e->cf, e->uf));

    UC(eobj_wait_send_ff(e->cf));
    UC(eobj_wait_recv_ff(e->cf, e->uf));

    UC(eobj_unpack_ff(e->uf, e->p, nw,      fw));
}






#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
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





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

#ifndef UDX_SIM_MESH_IO_COMMON
#define UDX_SIM_MESH_IO_COMMON
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

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory,      MeshWrite**);
void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead*, const char *directory,      MeshWrite**);

void mesh_write_particles(MeshWrite*, MPI_Comm, const Coords*, int nm, const Particle*, int id);
void mesh_write_vectors(MeshWrite*, MPI_Comm, int nm, Vectors *pos, Vectors *vel, int id);

void mesh_write_fin(MeshWrite*);
#endif

#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON

struct OptFlu {
    bool colors, ids, ss, push;
};


struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};


struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};


struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};


struct OptParams {
    int3 L;
    float kBT;
    int numdensity;
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
#endif

static int str2shifttype(const char *type) {
    if      (same_str(type, "edge"  )) return MESH_WRITE_EDGE;
    else if (same_str(type, "center")) return MESH_WRITE_CENTER;
    else
        ERR("Unrecognised rbc shift type <%s>", type);
    return -1;
}


static int get_shifttype_ns(const Config *c, const char *ns, const char *d) {
    const char *type;
    UC(conf_lookup_string_ns(c, ns, d, &type));
    return str2shifttype(type);
}

static void lookup_bool(const Config *c, const char *desc, bool *res) {
    int b;
    UC(conf_lookup_bool(c, desc, &b));
    *res = b;
}

static void lookup_bool_ns(const Config *c, const char *ns, const char *desc, bool *res) {
    int b;
    UC(conf_lookup_bool_ns(c, ns, desc, &b));
    *res = b;
}

static void lookup_string(const Config *c, const char *desc, char *res) {
    const char *s;
    UC(conf_lookup_string(c, desc, &s));
    strcpy(res, s);
}

static void lookup_string_ns(const Config *c, const char *ns, const char *desc, char *res) {
    const char *s;
    UC(conf_lookup_string_ns(c, ns, desc, &s));
    strcpy(res, s);
}

static void read_flu(const Config *c, OptFlu *o) {
    UC(lookup_bool(c, "flu.colors", &o->colors));
    UC(lookup_bool(c, "flu.ids", &o->ids));
    UC(lookup_bool(c, "flu.stresses", &o->ss));
    UC(lookup_bool(c, "flu.push", &o->push));
}

static void read_obj_common(const Config *c, bool restart, const char *ns, OptObj *o) {
    o->shifttype = get_shifttype_ns(c, ns, "shifttype");
    UC(lookup_bool_ns(c, ns, "push",   &o->push));
    UC(lookup_bool_ns(c, ns, "bounce", &o->bounce));
    UC(conf_lookup_float_ns(c, ns, "mass", &o->mass));

    UC(lookup_string_ns(c, ns, "templ_file", o->templ_file));

    if (restart) o->ic_file[0] = '\0';
    else         UC(lookup_string_ns(c, ns, "ic_file", o->ic_file));

    strcpy(o->name, ns);
}

static void read_mbr(const Config *c, bool restart, const char *ns, OptMbr *o) {
    UC(read_obj_common(c, restart, ns, o));
    UC(lookup_bool_ns(c, ns, "ids", &o->ids));
    UC(lookup_bool_ns(c, ns, "stretch", &o->stretch));

    UC(lookup_bool(c, "dump.rbc_com", &o->dump_com));
    UC(conf_lookup_int_ns(c, ns, "substeps", &o->substeps));

    if (o->stretch)
        UC(lookup_string_ns(c, ns, "stretch_file", o->stretch_file));
}

static void read_mbr_array(const Config *c, bool restart, int *nmbr, OptMbr *oo) {
    int i, n;
    const char *ss[MAX_MBR_TYPES];
    UC(conf_lookup_vstring(c, "membranes", MAX_MBR_TYPES, &n, ss));
    for (i = 0; i < n; ++i)
        UC(read_mbr(c, restart, ss[i], &oo[i]));
    *nmbr = n;
}

static void read_rig(const Config *c, bool restart, const char *ns, OptRig *o) {
    UC(read_obj_common(c, restart, ns, o));
    UC(lookup_bool_ns(c, ns, "empty_pp", &o->empty_pp));
}

static void read_rig_array(const Config *c, bool restart, int *nrig, OptRig *oo) {
    int i, n;
    const char *ss[MAX_RIG_TYPES];
    UC(conf_lookup_vstring(c, "rigids", MAX_RIG_TYPES, &n, ss));
    for (i = 0; i < n; ++i)
        UC(read_rig(c, restart, ss[i], &oo[i]));
    *nrig = n;
}

static void read_wall(const Config *c, OptWall *o) {
    UC(lookup_bool(c, "wall.active", &o->active));
}

static void read_params(const Config *c, OptParams *p) {
    UC(conf_lookup_int3 (c, "glb.L",          &p->L         ));
    UC(conf_lookup_float(c, "glb.kBT",        &p->kBT       ));
    UC(conf_lookup_int  (c, "glb.numdensity", &p->numdensity));
}

static void read_dump(const Config *c, OptDump *o) {
    UC(lookup_string(c, "dump.strt_base_dump", o->strt_base_dump));
    UC(lookup_string(c, "dump.strt_base_read", o->strt_base_read));

    UC(lookup_bool(c, "dump.parts", &o->parts));
    UC(conf_lookup_float(c, "dump.freq_parts", &o->freq_parts));

    UC(lookup_bool(c, "dump.mesh", &o->mesh));
    UC(conf_lookup_float(c, "dump.freq_mesh", &o->freq_mesh));

    UC(lookup_bool(c, "dump.field", &o->field));
    UC(conf_lookup_float(c, "dump.freq_field", &o->freq_field));

    UC(lookup_bool(c, "dump.strt", &o->strt));
    UC(conf_lookup_float(c, "dump.freq_strt", &o->freq_strt));

    UC(lookup_bool(c, "dump.forces", &o->forces));
    UC(conf_lookup_float(c, "dump.freq_diag", &o->freq_diag));
}

static void read_common(const Config *c, Opt *o) {
    UC(conf_lookup_int(c, "sampler.n_per_dump", &o->sampler_npdump));
    UC(conf_lookup_int3(c, "sampler.grid_ref", &o->sampler_grid_ref));

    UC(lookup_bool(c, "outflow.active", &o->outflow));
    UC(lookup_bool(c, "inflow.active", &o->inflow));
    UC(lookup_bool(c, "denoutflow.active", &o->denoutflow));
    UC(lookup_bool(c, "vcon.active", &o->vcon));

    UC(conf_lookup_int(c, "flu.recolor_freq", &o->recolor_freq));

    UC(lookup_bool(c, "glb.restart", &o->restart));
}

void opt_read(const Config *c, Opt *o) {
    UC(read_common(c, o));
    UC(read_params(c, &o->params));
    UC(read_dump(c, &o->dump));
    UC(read_flu(c, &o->flu));

    UC(read_mbr_array(c, o->restart, &o->nmbr, o->mbr));
    UC(read_rig_array(c, o->restart, &o->nrig, o->rig));

    UC(read_wall(c, &o->wall));
}

static void check_mbr(const OptMbr *m) {
    enum {
        MIN_SUBSTEPS = 1,
        MAX_SUBSTEPS = 1000
    };
    if (m->dump_com && !m->ids)
        ERR("%s: Need ids activated to dump com!", m->name);

    if (m->substeps < MIN_SUBSTEPS || m->substeps > MAX_SUBSTEPS)
        ERR("%s: substeps out of range: %d (min: %d, max: %d)",
            m->name, m->substeps, MIN_SUBSTEPS, MAX_SUBSTEPS);
}

void opt_check(const Opt *o) {
    for (int i = 0; i < o->nmbr; ++i)
        check_mbr(&o->mbr[i]);
}

static long maxp_estimate(const OptParams *p) {
    int3 L = p->L;
    int flu, obj, estimate;
    flu = L.x * L.y * L.z * p->numdensity;
    obj = MAX_OBJ_TYPES * MAX_PSOLID_NUM * MAX_SOLIDS;
    estimate = flu > obj ? flu : obj;
    return SAFETY_FACTOR_MAXP * estimate;
}

long opt_estimate_maxp(const Opt *o) {
    return maxp_estimate(&o->params);
}






#include <stdio.h>
#include <mpi.h>
#include <vector_types.h>

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
#endif






void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
void msg_ini(int rank);
void msg_print(const char *fmt, ...);

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

void parray_push_pp(const Particle *pp, PaArray *a);
void parray_push_cc(const int *cc, PaArray *a);

bool parray_is_colored(const PaArray *a);

void parray_get_view(const PaArray *a, PaArray_v *v);
void parray_get_view(const PaArray *a, PaCArray_v *v);
#endif
#ifndef UDX_SIM_OBJECTS_FARRAY_COMMON
#define UDX_SIM_OBJECTS_FARRAY_COMMON
struct FoArray {
    bool stress;
    float *ff;
    float *ss;
};

struct FoArray_v;
struct FoSArray_v;

struct Force;

void farray_push_ff(Force *ff, FoArray *a);
void farray_push_ss(float *ss, FoArray *a);

void farray_clear(FoArray *);

bool farray_has_stress(const FoArray *a);

void farray_get_view(const FoArray *a, FoArray_v *v);
void farray_get_view(const FoArray *a, FoSArray_v *v);
#endif
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
#endif

struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);

void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*);
void sdf_get_view(const Sdf*,      Sdf_v*);

void sdf_bulk_wall(const Sdf*,        int *s_n, Particle *s_pp,       int *w_n, Particle *w_pp);
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv,      int *stay);
double sdf_compute_volume(MPI_Comm, int3 L, const Sdf*, long nsamples);

void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n,      Particle*);
struct Wvel;
struct WvelStep;

struct WvelCste_v;
struct WvelShear_v;
struct WvelHS_v;

struct float3;
struct Config;

enum {
    WALL_VEL_V_CSTE,
    WALL_VEL_V_SHEAR,
    WALL_VEL_V_HS,
};

void wvel_ini(Wvel **wv);
void wvel_fin(Wvel *wv);

void wvel_step_ini(WvelStep **wv);
void wvel_step_fin(WvelStep *wv);

void wvel_set_cste(float3 u, Wvel *vw);
void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw);
void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, Wvel *vw);
void wvel_set_hs(float u, float h, Wvel *vw);
void wvel_set_timestep(float dt, Wvel *vw);

void wvel_set_conf(const Config *cfg, Wvel *vw);

void wvel_get_step(float t, const Wvel *wv,      WvelStep *);

int wvel_get_type(const WvelStep *w);

void wvel_get_view(const WvelStep *w,      WvelCste_v *v);
void wvel_get_view(const WvelStep *w,      WvelShear_v *v);
void wvel_get_view(const WvelStep *w,      WvelHS_v *v);
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

struct WallQuants {
    float4 *pp;
    int n;
    int3 L;
};

struct WallTicket;
struct WallRepulsePrm;


void wall_ini_quants(int3 L, WallQuants*);
void wall_ini_ticket(int3 L, WallTicket**);

void wall_fin_quants(WallQuants*);
void wall_fin_ticket(WallTicket*);

void wall_gen_quants(MPI_Comm, int numdensity, int maxn, const Sdf*,          int *o_n, Particle *o_pp,      WallQuants*);
void wall_gen_ticket(const WallQuants*, WallTicket*);

void wall_strt_quants(MPI_Comm, const char *base, int maxn, WallQuants*);
void wall_strt_dump_templ(MPI_Comm, const char *base, const WallQuants*);

void wall_force(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                const WallTicket*, int n, const PaArray*, const FoArray*);

void wall_force_adhesion(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                         const WallTicket*, int n, const PaArray*, const FoArray*);

void wall_repulse(const Sdf*, const WallRepulsePrm*, long n, const PaArray*, const FoArray*);

void wall_repulse_prm_ini(float lambda, WallRepulsePrm**);
void wall_repulse_prm_ini_conf(const Config*, const char *ns, WallRepulsePrm**);
void wall_repulse_prm_fin(WallRepulsePrm*);
#endif

#ifndef UDX_SIM_OBJECTS_OPT_COMMON
#define UDX_SIM_OBJECTS_OPT_COMMON

struct OptFlu {
    bool colors, ids, ss, push;
};


struct OptObj {
    bool push, bounce;
    int shifttype;
    float mass;
    char templ_file[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    char name[FILENAME_MAX];
};


struct OptMbr : OptObj {
    bool ids, stretch, dump_com;
    int substeps;
    char stretch_file[FILENAME_MAX];
};


struct OptRig : OptObj {
    bool empty_pp;
};

struct OptWall {
    bool active;
};


struct OptParams {
    int3 L;
    float kBT;
    int numdensity;
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
#endif

struct Wall;

struct OptParams;
struct Config;
struct PairParams;
struct Coords;

void wall_ini(const Config*, int3 L, Wall**);
void wall_fin(Wall*);

void wall_gen(MPI_Comm, const Coords*, OptParams, bool dump_sdf,
                     int *n, Particle *pp,      Wall*);

void wall_restart(MPI_Comm, const Coords*, OptParams, bool dump_sdf,
                  const char *base,      Wall*);

void wall_dump_templ(const Wall*, MPI_Comm, const char *base);

void wall_interact(const Coords*, const PairParams*[], Wall*, PFarrays*);
void wall_adhesion(const Coords*, const PairParams*[], Wall*, PFarrays*);
void wall_bounce(const Wall*, const Coords*, float dt, PFarrays*);
void wall_repulse(const Wall*, const WallRepulsePrm *[], PFarrays*);

void wall_update_vel(float t, Wall *);

void wall_get_sdf_ptr(const Wall*, const Sdf**);
double wall_compute_volume(const Wall*, MPI_Comm, int3 L);
struct Wall {
    Sdf *sdf;
    WallQuants q;
    WallTicket *t;
    Wvel *vel;
    WvelStep *velstep;
};
void wall_ini(const Config *cfg, int3 L, Wall **wall) {
    Wall *w;
    EMALLOC(1, wall);
    w = *wall;
    UC(sdf_ini(L, &w->sdf));
    UC(wall_ini_quants(L, &w->q));
    UC(wall_ini_ticket(L, &w->t));
    UC(wvel_ini(&w->vel));
    UC(wvel_set_conf(cfg, w->vel));
    UC(wvel_step_ini(&w->velstep));
}

void wall_fin(Wall *w) {
    UC(sdf_fin(w->sdf));
    UC(wall_fin_quants(&w->q));
    UC(wall_fin_ticket(w->t));
    UC(wvel_fin(w->vel));
    UC(wvel_step_fin(w->velstep));
    EFREE(w);
}
static long get_max_parts(OptParams p) {
    int3 L = p.L;
    return p.numdensity *
        (L.x + 2 * XWM) *
        (L.y + 2 * YWM) *
        (L.z + 2 * ZWM);
}

void wall_gen(MPI_Comm cart, const Coords *coords, OptParams op, bool dump_sdf,
                     int *n, Particle *pp,      Wall *w) {
    long maxn, nold, nnew;
    if (!w) return;

    maxn = get_max_parts(op);
    nold = *n;
    UC(sdf_gen(coords, cart, dump_sdf,      w->sdf));
    MC(m::Barrier(cart));
    UC(wall_gen_quants(cart, op.numdensity, maxn, w->sdf,      n, pp, &w->q));
    if (w->q.n) UC(wall_gen_ticket(&w->q, w->t));
    nnew = *n;
    msg_print("solvent particles survived: %d/%d", nnew, nold);
}

void wall_restart(MPI_Comm cart, const Coords *coords, OptParams op, bool dump_sdf,
                  const char *base,      Wall *w) {
    long maxn;
    if (!w) return;

    maxn = get_max_parts(op);
    UC(sdf_gen(coords, cart, dump_sdf,      w->sdf));
    MC(m::Barrier(cart));
    UC(wall_strt_quants(cart, base, maxn, &w->q));
    UC(wall_gen_ticket(&w->q, w->t));
}

void wall_dump_templ(const Wall *w, MPI_Comm cart, const char *base) {
    UC(wall_strt_dump_templ(cart, base, &w->q));
}

void wall_get_sdf_ptr(const Wall *w, const Sdf **s) {*s = w->sdf;}

double wall_compute_volume(const Wall *w, MPI_Comm comm, int3 L) {
    enum {NSAMPLES = 100000};
    return sdf_compute_volume(comm, L, w->sdf, NSAMPLES);
}

void wall_interact(const Coords *coords, const PairParams *prms[], Wall *w, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;
    const PairParams *prm;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        prm = prms[i];
        if (!n)   continue;
        if (!prm) continue;
        UC(wall_force(prm, w->velstep, coords, w->sdf, &w->q, w->t, n, &p,      &f));
    }
}

void wall_adhesion(const Coords *coords, const PairParams* params[], Wall *w, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;
    const PairParams *par;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        par = params[i];

        if (!par) continue;
        if (!n)   continue;

        UC(wall_force_adhesion(par, w->velstep, coords, w->sdf, &w->q, w->t, n, &p,      &f));
    }
}

void wall_bounce(const Wall *w, const Coords *coords, float dt, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        if (n) UC(sdf_bounce(dt, w->velstep, coords, w->sdf, n,      (Particle*) p.pp));
    }
}

void wall_repulse(const Wall *w, const WallRepulsePrm *params[], PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;
    const WallRepulsePrm *wrp;

    na = pfarrays_size(aa);
    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        wrp = params[i];

        if (!n)   continue;
        if (!wrp) continue;

        UC(wall_repulse(w->sdf, wrp, n, &p,      &f));
    }
}

void wall_update_vel(float t, Wall *w) {
    UC(wvel_get_step(t, w->vel,      w->velstep));
}






















#include <mpi.h>
#include <string.h>
#include <vector_types.h>





#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
#include "device.cuh"
#endif

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

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
#endif
void os_mkdir(const char* path);
long os_time();
void os_srand(long int seedval);
double os_drand();
void os_sleep(unsigned int seconds);
void os_malloc_stats();





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
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









#include <stdio.h>
#include <vector_types.h>






void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
void msg_ini(int rank);
void msg_print(const char *fmt, ...);
struct int4;
struct MeshRead;

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


#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <mpi.h>









void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
void msg_ini(int rank);
void msg_print(const char *fmt, ...);
void os_mkdir(const char* path);
long os_time();
void os_srand(long int seedval);
double os_drand();
void os_sleep(unsigned int seconds);
void os_malloc_stats();

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

#undef PATTERN









#include <string.h>
#include <ctype.h>
#include <mpi.h>





void os_mkdir(const char* path);
long os_time();
void os_srand(long int seedval);
double os_drand();
void os_sleep(unsigned int seconds);
void os_malloc_stats();
void msg_ini(int rank);
void msg_print(const char *fmt, ...);





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
int string_nword(const char *s);


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










#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector_types.h>

struct Coords;
void coords_ini(MPI_Comm cart, int Lx, int Ly, int Lz, Coords **c);
void coords_fin(Coords *c);
int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);
int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);



void error_stack_push(const char *file, int line);
void error_stack_pop();
void error_signal(const char *file, int line, const char *fmt, ...);
bool error_get();
void error_report();
void error_abort();






#include <mpi.h>
#include <stdio.h>
#include <vector_types.h>




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





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();

#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

void xmf_write(int3 domainSize, int3 gridSize, const char *path, int ncomp, const char **names);
void raw_write(int3 N, MPI_Comm, const char *path, int ncomp, const float **data, const char **names);















#ifndef UDX_IO_COM_COMMON
#define UDX_IO_COM_COMMON

#include <stdio.h>
#include <vector_types.h>
#include <mpi.h>




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

#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
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

void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
void os_mkdir(const char* path);
long os_time();
void os_srand(long int seedval);
double os_drand();
void os_sleep(unsigned int seconds);
void os_malloc_stats();





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();

struct WriteFile;

int write_file_open(MPI_Comm, const char *path,      WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*);
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*);

int write_shift_indices(MPI_Comm, int,      int*);
int write_reduce(MPI_Comm, int,      int*);

struct Coords;
struct float3;
#endif





#include <stdio.h>

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#ifndef UDX_IO_DIAG_COMMON_CUH
#define UDX_IO_DIAG_COMMON_CUH
#include "types.cuh"
#endif
#endif

void msg_ini(int rank);
void msg_print(const char *fmt, ...);
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();

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









#include <stdio.h>
#include <string.h>
#include <vector_types.h>
#include <mpi.h>




void msg_ini(int rank);
void msg_print(const char *fmt, ...);
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
#include "diagnostic.cuh"





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
struct int4;
struct MeshRead;

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

struct DiagMesh;
struct MeshRead;

void diag_mesh_ini(const char *path, MeshRead*, DiagMesh**);
void diag_mesh_fin(DiagMesh*);
void diag_mesh_apply(DiagMesh*, MPI_Comm, float time, int nc, Particle*);
struct DiagMesh {
    char path[FILENAME_MAX];
    int nv, nt;
    int4 *tt;
};
void diag_mesh_ini(const char *path, MeshRead *mesh, DiagMesh **pq) {
    FILE *f;
    DiagMesh *q;
    EMALLOC(1, &q);
    *pq = q;
    q->nv = mesh_read_get_nv(mesh);
    q->nt = mesh_read_get_nt(mesh);
    EMALLOC(q->nt, &q->tt);
    EMEMCPY(q->nt, mesh_read_get_tri(mesh), q->tt);
    snprintf(q->path, FILENAME_MAX, "%s/%s", DUMP_BASE, path);
    UC(efopen(q->path, "w",      &f));
    UC(efclose(f));
    msg_print("DiagMesh: %s", q->path);
}

void diag_mesh_fin(DiagMesh *q) {
    EFREE(q->tt);
    EFREE(q);
}

void diag_mesh_apply(DiagMesh*, MPI_Comm, float, int, Particle*) {

}





#include <stdio.h>
#include <string.h>
#include <mpi.h>




void msg_ini(int rank);
void msg_print(const char *fmt, ...);
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
#include "diagnostic.cuh"





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

struct Particle;
struct DiagPart;

void diag_part_ini(const char *path,      DiagPart**);
void diag_part_fin(DiagPart*);
void diag_part_apply(DiagPart*, MPI_Comm, float time, int n, const Particle*);
enum {STAMP_GOOD = 123456};
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector_types.h>




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

void os_mkdir(const char* path);
long os_time();
void os_srand(long int seedval);
double os_drand();
void os_sleep(unsigned int seconds);
void os_malloc_stats();





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
void msg_ini(int rank);
void msg_print(const char *fmt, ...);

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








#include <stdio.h>
#include <string.h>
#include <mpi.h>

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#ifndef UDX_IO_TYPE_COMMON
#define UDX_IO_TYPE_COMMON
#include "types.cuh"
#endif
#endif








void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
void os_mkdir(const char* path);
long os_time();
void os_srand(long int seedval);
double os_drand();
void os_sleep(unsigned int seconds);
void os_malloc_stats();
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

struct WriteFile;

int write_file_open(MPI_Comm, const char *path,      WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*);
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*);

int write_shift_indices(MPI_Comm, int,      int*);
int write_reduce(MPI_Comm, int,      int*);

struct Coords;
struct Rigid;






#include <stdio.h>
#include <string.h>
#include <mpi.h>





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

#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#endif
#ifndef UDX_SIM_OBJECTS_DEF_COMMON
#define UDX_SIM_OBJECTS_DEF_COMMON
#include "constants.cuh"
#endif
void msg_ini(int rank);
void msg_print(const char *fmt, ...);





void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);


void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
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

struct Coords;
struct Particle;
struct Rigid;

#ifndef UDX_SIM_RESTART_COMMON
#define UDX_SIM_RESTART_COMMON
enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};
#endif

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






























#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
