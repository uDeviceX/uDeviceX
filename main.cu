#include "config.cuh"

#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <vector_types.h>


#if !defined(KL_RELEASE)    && !defined(KL_NONE)   &&   \
    !defined(KL_SYNC)
#define KL_RELEASE
#endif


#if !defined(CC_RELEASE) && !defined(CC_SYNC)
  #define CC_RELEASE
#endif


#if !defined(DEV_CUDA) && !defined(DEV_CPU)
  #define DEV_CUDA
#endif


#if !defined(CUDA_ALLOC_RELEASE) && !defined(CUDA_ALLOC_DEBUG)
  #define CUDA_ALLOC_RELEASE
#endif


#if !defined(RBC_DOUBLE) && !defined(RBC_FLOAT)
  #define RBC_DOUBLE
#endif


#if !defined(RBC_SPRING_IGNORE) && !defined(RBC_SPRING_FAIL)
  #define RBC_SPRING_IGNORE
#endif

#ifndef DUMP_BASE
#define DUMP_BASE "."
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
void os_mkdir(const char* path);
long os_time();
void os_srand(long int seedval);
double os_drand();
void os_sleep(unsigned int seconds);
void os_malloc_stats();

namespace d {
typedef int Stream_t;
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);

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
}

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

    XWM = 6,
    YWM = 6,
    ZWM = 6,

    SAFETY_FACTOR_MAXP = 3,


    MAX_PSOLID_NUM = 12000,


    MAX_SOLIDS = 100,


    MAX_MBR_TYPES = 2,
    MAX_RIG_TYPES = 2,
    MAX_OBJ_TYPES = MAX_MBR_TYPES + MAX_RIG_TYPES,


    MAX_OBJ_DENSITY = 30,


    MAX_CELL_NUM = 1500,


    MAX_TEXO_SIZE = 100000000,


    HSAFETY_FACTOR = 10,


    ODSTR_FACTOR = 3,
};


#define RBC_SPRING_CAP (0.99)
void msg_ini(int rank);
void msg_print(const char *fmt, ...);
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

template <typename T, int N>
struct Sarray {
    T d[N];
};

struct Particle {
    float r[3], v[3];
};

struct Rigid {
    float Iinv[6],
        mass,
        com[3],
        v[3], om[3],
        e0[3], e1[3], e2[3],
        fo[3], to[3],
        id;
};

struct Force {
    float f[3];
};

struct Momentum {
    float P[3], L[3];
};

struct Positioncp {
    float rc[3];
    float rp[3];
};

typedef Sarray<int, 26>       int26;
typedef Sarray<int, 27>       int27;

typedef Sarray<int*, 26>     intp26;
typedef Sarray<Particle*, 26> Pap26;
typedef Sarray<Force*,    26> Fop26;
typedef Sarray<Momentum*, 26> Mop26;
typedef Sarray<Positioncp*, 26> Pocpp26;


struct PaWrap {
    int n;
    const Particle *pp;
};


struct FoWrap {
    int n;
    Force *ff;
};

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
enum {
    ALIVE = 0,
    DEAD  = 1
};




struct PartList {
    const Particle *pp;
    const int *deathlist;
};

struct Particle;
struct DiagPart;

void diag_part_ini(const char *path,      DiagPart**);
void diag_part_fin(DiagPart*);
void diag_part_apply(DiagPart*, MPI_Comm, float time, int n, const Particle*);
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

struct FluQuants {
    Particle *pp, *pp0;
    int       n;
    Clist     *cells;
    ClistMap *mcells;
    Particle *pp_hst;



    bool ids, colors;

    int *ii, *ii0;
    int *ii_hst;

    int *cc, *cc0;
    int *cc_hst;

    int maxp;
};

struct Coords;
struct GenColor;

void flu_ini(bool colors, bool ids, int3 L, int maxp, FluQuants *q);
void flu_fin(FluQuants *q);

void flu_gen_quants(const Coords*, int numdensity, const GenColor *gc, FluQuants *q);
void flu_gen_ids(MPI_Comm, const int n, FluQuants *q);

void flu_strt_quants(MPI_Comm, const char *base, const int id, FluQuants *q);
void flu_strt_dump(MPI_Comm, const char *base, const int id, const FluQuants *q);

void flu_txt_dump(const Coords*, const FluQuants *q);



void flu_build_cells(     FluQuants *q);

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

struct DFluPack;
struct DFluComm;
struct DFluUnpack;
struct DFluStatus;

void dflu_pack_ini(bool colors, bool ids, int3 L, int maxdensity, DFluPack **p);
void dflu_comm_ini(MPI_Comm comm,      DFluComm **c);
void dflu_unpack_ini(bool colors, bool ids, int3 L, int maxdensity, DFluUnpack **u);

void dflu_pack_fin(DFluPack *p);
void dflu_comm_fin(DFluComm *c);
void dflu_unpack_fin(DFluUnpack *u);


void dflu_build_map(int n, const PartList lp, DFluPack *p);


void dflu_pack(const FluQuants *q,      DFluPack *p);
void dflu_download(DFluPack *p,      DFluStatus *s);


void dflu_post_recv(DFluComm *c, DFluUnpack *u);
void dflu_post_send(DFluPack *p, DFluComm   *c);
void dflu_wait_recv(DFluComm *c, DFluUnpack *u);
void dflu_wait_send(DFluComm *c);


void dflu_unpack(     DFluUnpack *u);



void dflu_bulk(PartList lp,      FluQuants *q);
void dflu_halo(const DFluUnpack *u,      FluQuants *q);
void dflu_gather(int ndead, const DFluPack *p, const DFluUnpack *u,      FluQuants *q);
struct DFluStatus;
struct Config;


void dflu_status_ini(DFluStatus **s);
void dflu_status_fin(DFluStatus  *s);


int  dflu_status_nullp(DFluStatus *s);
int  dflu_status_success(DFluStatus *s);
void dflu_status_log(DFluStatus *s);


void dflu_status_exceed(int frag, int counter, int capacity,      DFluStatus *s);




namespace flu {

enum { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct LFrag {
    PaArray parray;
    const int *ii;
    int n;
};

struct RFrag {
    PaArray parray;
    const int *start;
    int dx, dy, dz, xcells, ycells, zcells;
    int type;
};

struct RndFrag {
    float seed;
    int mask;
};

typedef Sarray<LFrag,   26> LFrag26;
typedef Sarray<RFrag,   26> RFrag26;
typedef Sarray<RndFrag, 26> RndFrag26;

}
struct EFluPack;
struct EFluComm;
struct EFluUnpack;
struct PaArray;

void eflu_pack_ini(bool colors, int3 L, int maxd, EFluPack **p);
void eflu_comm_ini(MPI_Comm comm,      EFluComm **c);
void eflu_unpack_ini(bool colors, int3 L, int maxd, EFluUnpack **u);

void eflu_pack_fin(EFluPack *p);
void eflu_comm_fin(EFluComm *c);
void eflu_unpack_fin(EFluUnpack *u);

void eflu_compute_map(const int *start, const int *count,      EFluPack *p);
void eflu_download_cell_starts(     EFluPack *p);

void eflu_pack(const PaArray *parray,      EFluPack *p);
void eflu_download_data(EFluPack *p);

void eflu_post_recv(EFluComm *c, EFluUnpack *u);
void eflu_post_send(EFluPack *p, EFluComm *c);
void eflu_wait_recv(EFluComm *c, EFluUnpack *u);
void eflu_wait_send(EFluComm *c);

void eflu_unpack(EFluUnpack *u);

using flu::LFrag26;
using flu::RFrag26;

void eflu_get_local_frags(const EFluPack *p,      LFrag26 *lfrags);
void eflu_get_remote_frags(const EFluUnpack *u,      RFrag26 *rfrags);


struct PairParams;
struct FluForcesBulk;
struct FluForcesHalo;
struct PaArray;
struct FoArray;

void fluforces_bulk_ini(int3 L, int maxp,      FluForcesBulk **b);
void fluforces_bulk_fin(     FluForcesBulk *b);

void fluforces_bulk_prepare(int n, const PaArray *a,      FluForcesBulk *b);
void fluforces_bulk_apply(const PairParams*, int n, const FluForcesBulk *b, const int *start, const int *count,
                               const FoArray *ff);

void fluforces_halo_ini(MPI_Comm cart, int3 L,      FluForcesHalo **hd);
void fluforces_halo_fin(     FluForcesHalo *h);

void fluforces_halo_prepare(flu::LFrag26 lfrags, flu::RFrag26 rfrags,      FluForcesHalo *h);
void fluforces_halo_apply(const PairParams*, const FluForcesHalo *h,      const FoArray *farray);

struct IoBop;

void io_bop_ini(MPI_Comm comm, int maxp, IoBop **t);
void io_bop_fin(IoBop *t);

void io_bop_parts       (MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const char *name, int id, IoBop *t);
void io_bop_parts_forces(MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const Force *ff, const char *name, int id, IoBop *t);
void io_bop_stresses    (MPI_Comm cart, long n, const float *ss, const char *name, int id, IoBop *t);
void io_bop_ids         (MPI_Comm cart, long n, const int   *ii, const char *name, int id, IoBop *t);
void io_bop_colors      (MPI_Comm cart, long n, const int   *cc, const char *name, int id, IoBop *t);

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
struct Force;
struct Particle;

void scheme_move_apply(float dt, float mass, int n, const Force*, Particle*);
void scheme_move_clear_vel(int n, Particle*);
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
struct TimeLine;

void time_line_ini(float start,      TimeLine**);
void time_line_fin(TimeLine*);

void time_line_advance(float dt, TimeLine*);
int  time_line_cross(const TimeLine*, float interval);

void time_line_strt_dump(MPI_Comm, const char *base, int id, const TimeLine*);
void time_line_strt_read(const char *base, int id, TimeLine*);

float time_line_get_current  (const TimeLine*);
long  time_line_get_iteration(const TimeLine*);
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

struct int3;
struct Particle;
struct Coords;

struct Rnd;

void color_linear_flux(const Coords*, int3 L, int dir, int color, int n, const Particle *pp,     int *cc);

void color_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc);
void decolor_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc);

void tracers_ini(const int n);

void tracers_fin();

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

struct ObjInter;

struct Config;
struct Opt;
struct PFarray;
struct PFarrays;

void obj_inter_ini(const Config*, const Opt*, MPI_Comm, float dt, int maxp,      ObjInter**);
void obj_inter_fin(ObjInter*);

void obj_inter_forces(ObjInter*, const PairParams **fsi_prms, const PairParams **cnt_prms, PFarray *flu, const int *flu_start, PFarrays *obj);

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

struct Sim;
struct Config;

void sim_ini(const Config*, MPI_Comm,      Sim**);
void sim_fin(Sim*);

void sim_gen(Sim*);
void sim_strt(Sim*);

#define _I_ static
#define _S_ static


struct FluDistr {
    DFluPack *p;
    DFluComm *c;
    DFluUnpack *u;
    DFluStatus *s;
};


struct FluExch {
    EFluPack *p;
    EFluComm *c;
    EFluUnpack *u;
};


struct Recolorer {
    bool flux_active;
    int flux_dir;
};


struct Tracers {
    bool active;
    int freq;
    float R;
    float iniP;
    float delP;
};


struct Flu {
    FluQuants q;
    PairParams *params;

    FluDistr d;
    FluExch e;

    FluForcesBulk *bulk;
    FluForcesHalo *halo;

    Force *ff;
    Force *ff_hst;

    float *ss;
    float *ss_hst;

    float mass;
};


struct Vcon {
    PidVCont *vcont;
    int log_freq;
    int adjust_freq;
    int sample_freq;
};



struct Time {
    TimeLine *t;
    float eq, end;
    float mbr_bb;
    TimeStep *step;
    TimeStepAccel *accel;
};

struct Sampler {
    GridSampler *s;
    GridSampleData *d;
};

struct Dump {

    Particle *pp;

    IoBop *bop;
    DiagPart *diagpart;
    Sampler field_sampler;

    int id_field, id_bop, id_strt;
};

struct Sim {

    Flu flu;
    Wall *wall;
    Objects *obj;

    BForce *bforce;


    Time time;
    Coords *coords;
    ObjInter *objinter;
    Recolorer recolorer;
    Tracers tracers;
    Vcon vcon;
    Restrain *restrain;
    Dbg *dbg;
    MPI_Comm cart;


    Outflow *outflow;
    Inflow *inflow;
    DCont    *denoutflow;
    DContMap *mapoutflow;


    bool equilibrating;
    bool active_mbr_bb;

    Opt opt;
    Dump dump;


    GenColor *gen_color;
};
_I_ bool active_walls(const Sim *s) {
    return !s->equilibrating && s->opt.wall.active;
}

_I_ bool is_sampling_time(const Sim *s) {
    const Opt *opt = &s->opt;
    const float freq = opt->dump.freq_field / opt->sampler_npdump;
    const TimeLine *time = s->time.t;

    return time_line_cross(time, freq);
}

_I_ bool is_mbr_bb_activation_time(Sim *s) {
    float t, tbb;
    t = time_line_get_current(s->time.t);
    tbb = s->time.mbr_bb;

    if (t > tbb && !s->active_mbr_bb) {
        s->active_mbr_bb = true;
        return true;
    }
    return false;
}

_I_ void utils_compute_hematocrit(const Sim *s) {
    const double SMALL = 1e-3;
    const Opt *opt = &s->opt;
    double Vdomain, Vrbc, Ht;
    if (opt->wall.active) {
        Vdomain = wall_compute_volume(s->wall, s->cart, s->opt.params.L);
    }
    else {
        const Coords *c = s->coords;
        Vdomain = xdomain(c) * ydomain(c) * zdomain(c);
    }
    if (Vdomain < SMALL)
        ERR("Vdomain = %g", Vdomain);

    Vrbc = objects_mbr_tot_volume(s->obj);
    Ht = Vrbc / Vdomain;

    msg_print("Geometry volume: %g", Vdomain);
    msg_print("Hematocrit: %g", Ht);
}

_I_ void utils_dump_history(const Config *cfg, const char *fname) {
    FILE *f;
    UC(efopen(fname, "w", &f));
    UC(conf_write_history(cfg, f));
    UC(efclose(f));
}

_I_ float utils_get_dt0(Sim *s) {
    TimeStep *time_step;
    time_step = s->time.step;
    return time_step_dt0(time_step);
}

_S_ void push_accel(const Sim *s, TimeStepAccel *aa) {
    const Flu *flu = &s->flu;
    if (flu->q.n) UC(time_step_accel_push(aa, flu->mass, flu->q.n, flu->ff));
    UC(objects_get_accel(s->obj, aa));
}

_I_ float utils_get_dt(Sim *s, TimeLine *time) {

    float dt;
    const Opt *opt = &s->opt;
    if (s->equilibrating)
        dt = time_step_dt0(s->time.step);
    else {
        UC(time_step_accel_reset(s->time.accel));
        UC(push_accel(s,      s->time.accel));
        dt = time_step_dt(s->time.step, s->cart, s->time.accel);


        if (time_line_cross(time, opt->dump.freq_diag))
            time_step_log(s->time.step);
    }
    return dt;
}

_I_ void utils_get_pf_flu(Sim *s, PFarray *flu) {
    Flu *f = &s->flu;
    flu->n = f->q.n;
    UC(parray_push_pp(f->q.pp, &flu->p));
    if (s->opt.flu.colors)
        parray_push_cc(f->q.cc, &flu->p);

    UC(farray_push_ff(f->ff, &flu->f));
}

_I_ void utils_set_n_flu_pf(const PFarray *flu, Sim *s) {
    Flu *f = &s->flu;
    f->q.n = flu->n;
}
_S_ void body_force(const BForce *bforce, Sim *s) {
    Flu *flu = &s->flu;
    const Opt *opt = &s->opt;

    if (opt->flu.push)
        UC(bforce_apply(s->coords, flu->mass, bforce, flu->q.n, flu->q.pp,      flu->ff));
    UC(objects_body_forces(bforce, s->obj));
}

_S_ void clear_forces(int n, Force* ff) {
    if (n) DzeroA(ff, n);
}

_S_ void clear_stresses(int n, float* ss) {
    if (n) DzeroA(ss, 6*n);
}

_S_ void forces_wall(bool fluss, Sim *s) {
    PaArray po;
    FoArray fo;
    PFarrays *pf;
    Flu *flu = &s->flu;
    Wall *w =  s->wall;
    const PairParams *dpd_prms[1 + MAX_OBJ_TYPES];
    const Opt *opt = &s->opt;

    if (!w) return;

    parray_push_pp(flu->q.pp, &po);
    if (opt->flu.colors)
        parray_push_cc(flu->q.cc, &po);

    farray_push_ff(flu->ff, &fo);
    if (fluss)
        farray_push_ss(flu->ss, &fo);
    dpd_prms[0] = flu->params;

    UC(pfarrays_ini(&pf));
    UC(pfarrays_push(pf, flu->q.n, po, fo));

    UC(objects_get_particles_all(s->obj, pf));
    UC(objects_get_params_fsi(s->obj, dpd_prms + 1));
    UC(wall_interact(s->coords, dpd_prms, w, pf));

    {
        const WallRepulsePrm *repulsion_prms[MAX_OBJ_TYPES];
        UC(pfarrays_clear(pf));
        UC(objects_get_params_repulsion(s->obj, repulsion_prms));
        UC(objects_get_particles_all(s->obj, pf));
        UC(wall_repulse(w, repulsion_prms, pf));
    }

    {
        const PairParams *adhesion_prms[MAX_OBJ_TYPES];
        UC(pfarrays_clear(pf));
        UC(objects_get_params_adhesion(s->obj, adhesion_prms));
        UC(objects_get_particles_all(s->obj, pf));
        UC(wall_adhesion(s->coords, adhesion_prms, w, pf));
    }

    UC(pfarrays_fin(pf));
}

_S_ void forces_dpd(bool stress, Flu *f) {
    const int *count = clists_get_cc(f->q.cells);
    const int *start = clists_get_ss(f->q.cells);
    PaArray parray;
    FoArray farray;
    flu::LFrag26 lfrags;
    flu::RFrag26 rfrags;

    FluExch *e = &f->e;

    parray_push_pp(f->q.pp,      &parray);
    if (f->q.colors)
        parray_push_cc(f->q.cc,      &parray);

    farray_push_ff(f->ff,      &farray);
    if (stress)
        farray_push_ss(f->ss,      &farray);

    UC(eflu_compute_map(start, count,      e->p));
    UC(eflu_download_cell_starts(     e->p));
    UC(eflu_pack(&parray,      e->p));
    UC(eflu_download_data(     e->p));

    UC(eflu_post_recv(e->c, e->u));
    UC(eflu_post_send(e->p, e->c));

    UC(fluforces_bulk_prepare(f->q.n, &parray,      f->bulk));
    UC(fluforces_bulk_apply(f->params, f->q.n, f->bulk, start, count,      &farray));

    UC(eflu_wait_recv(e->c, e->u));
    UC(eflu_wait_send(e->c));

    UC(eflu_unpack(e->u));

    UC(eflu_get_local_frags(e->p,      &lfrags));
    UC(eflu_get_remote_frags(e->u,      &rfrags));

    UC(fluforces_halo_prepare(lfrags, rfrags,      f->halo));
    UC(fluforces_halo_apply(f->params, f->halo,      &farray));
}

_S_ void forces_objects(Sim *sim) {
    PFarrays *obj;
    PFarray flu;
    const PairParams *fsi_prms[MAX_OBJ_TYPES];
    const PairParams **cnt_prms;

    UC(pfarrays_ini(&obj));
    UC(objects_get_particles_all(sim->obj, obj));
    UC(objects_get_params_fsi(sim->obj, fsi_prms));
    UC(objects_get_params_cnt(sim->obj, &cnt_prms));
    UC(utils_get_pf_flu(sim, &flu));

    UC(obj_inter_forces(sim->objinter, fsi_prms, cnt_prms, &flu, clists_get_ss(sim->flu.q.cells), obj));

    UC(pfarrays_fin(obj));
}

_I_ void forces(float dt, TimeLine *time, const BForce *bforce, Sim *s) {
    Flu *flu = &s->flu;
    bool fluss, tfluss;
    Opt *opt = &s->opt;

    tfluss = time_line_cross(time, opt->dump.freq_parts) ||
        is_sampling_time(s);
    fluss = opt->flu.ss && tfluss;

    UC(clear_forces(flu->q.n, flu->ff));
    if (fluss) UC(clear_stresses(flu->q.n, flu->ss));
    UC(objects_clear_forces(s->obj));

    UC(forces_dpd(fluss, flu));
    if (active_walls(s)) UC(forces_wall(fluss, s));


    UC(forces_objects(s));

    if (!s->equilibrating) UC(body_force(bforce, s));

    dSync();
}

_S_ void ini_flu_exch(const Opt *opt, MPI_Comm comm, int3 L,      FluExch *e) {
    int maxd = HSAFETY_FACTOR * opt->params.numdensity;

    UC(eflu_pack_ini(opt->flu.colors, L, maxd,      &e->p));
    UC(eflu_comm_ini(comm,      &e->c));
    UC(eflu_unpack_ini(opt->flu.colors, L, maxd,      &e->u));
}

_S_ void ini_flu_distr(const Opt *opt, MPI_Comm comm, int3 L,      FluDistr *d) {
    float maxdensity = ODSTR_FACTOR * opt->params.numdensity;
    UC(dflu_pack_ini(opt->flu.colors, opt->flu.ids, L, maxdensity,      &d->p));
    UC(dflu_comm_ini(comm,      &d->c));
    UC(dflu_unpack_ini(opt->flu.colors, opt->flu.ids, L, maxdensity,      &d->u));
    UC(dflu_status_ini(     &d->s));
}

_S_ void ini_vcon(MPI_Comm comm, int3 L, const Config *cfg,      Vcon *c) {
    UC(conf_lookup_int(cfg, "vcon.log_freq", &c->log_freq));
    UC(conf_lookup_int(cfg, "vcon.adjust_freq", &c->adjust_freq));
    UC(conf_lookup_int(cfg, "vcon.sample_freq", &c->sample_freq));

    UC(vcont_ini(comm, L,      &c->vcont));
    UC(vcont_set_conf(cfg,      c->vcont));
}

_S_ void ini_outflow(const Coords *coords, int maxp, const Config *cfg, Outflow **o) {
    UC(outflow_ini(maxp,      o));
    UC(outflow_set_conf(cfg, coords, *o));
}

_S_ void ini_denoutflow(const Coords *c, int maxp, const Config *cfg, DCont **d, DContMap **m) {
    UC(den_ini(maxp,      d));
    UC(den_map_ini(     m));
    UC(den_map_set_conf(cfg, c, *m));
}

_S_ void ini_tracers(const int maxp) {
    UC(tracers_ini(maxp));
}

_S_ void ini_inflow(const Coords *coords, int3 L, const Config *cfg, Inflow **i) {

    int2 nc;
    nc.x = L.y;
    nc.y = L.z/2;
    UC(inflow_ini(nc,      i));
    UC(inflow_ini_params_conf(coords, cfg, *i));
    UC(inflow_ini_velocity(*i));
}

_S_ void read_pair_params(const Config *cfg, const char *ns, PairParams **par) {
    PairParams *p;
    UC(pair_ini(par));
    p = *par;
    UC(pair_set_conf(cfg, ns, p));
}

_S_ void ini_pair_params(const Config *cfg, const char *name, const char *pair, PairParams **par) {
    const char *ns;
    UC(conf_lookup_string_ns(cfg, name, pair, &ns));
    UC(read_pair_params(cfg, ns, par));
}

_S_ void ini_flu(const Config *cfg, const Opt *opt, MPI_Comm cart, int maxp,      Flu *f) {
    int3 L = opt->params.L;

    UC(flu_ini(opt->flu.colors, opt->flu.ids, L, maxp, &f->q));
    UC(fluforces_bulk_ini(L, maxp,      &f->bulk));
    UC(fluforces_halo_ini(cart, L,      &f->halo));

    UC(ini_flu_distr(opt, cart, L,      &f->d));
    UC(ini_flu_exch(opt, cart, L,      &f->e));

    UC(Dalloc(&f->ff, maxp));
    EMALLOC(maxp,      &f->ff_hst);

    if (opt->flu.ss) {
        UC(Dalloc(&f->ss, 6*maxp));
        EMALLOC(6*maxp,      &f->ss_hst);
    }

    UC(conf_lookup_float(cfg, "flu.mass", &f->mass));
    UC(ini_pair_params(cfg, "flu", "self", &f->params));
}

_S_ void read_tracer_opt(const Config *c, Tracers *o, Opt *opt) {
    int b;
    UC(conf_lookup_bool(c, "tracers.active", &b));
    o->active = b;
    opt->tracers = b;
    if (b) {
        UC(conf_lookup_int(c, "tracers.freq", &o->freq));
        UC(conf_lookup_float(c, "tracers.radius", &o->R));
        UC(conf_lookup_float(c, "tracers.createprob", &o->iniP));
        UC(conf_lookup_float(c, "tracers.deleteprob", &o->delP));
    }
}

_S_ void read_recolor_opt(const Config *c, Recolorer *o) {
    int b;
    UC(conf_lookup_bool(c, "recolor.active", &b));
    o->flux_active = b;
    UC(conf_lookup_int(c, "recolor.dir", &o->flux_dir));
}

_S_ void coords_log(const Coords *c) {
    msg_print("domain: %d %d %d", xdomain(c), ydomain(c), zdomain(c));
    msg_print("subdomain: [%d:%d][%d:%d][%d:%d]",
              xlo(c), xhi(c), ylo(c), yhi(c), zlo(c), zhi(c));
}

_S_ int gsize(int L, int r) {
    return r >= 0 ? L * r : L / r;
}

_S_ int3 grid_size(int3 L, int3 r) {
    int3 N;
    N.x = gsize(L.x, r.x);
    N.y = gsize(L.y, r.y);
    N.z = gsize(L.z, r.z);
    return N;
}

_S_ void ini_sampler(const Coords *c, const Opt *opt, Sampler *s) {
    int3 N, L;
    bool stress, colors;
    stress = opt->flu.ss;
    colors = opt->flu.colors;
    L = subdomain(c);
    N = grid_size(L, opt->sampler_grid_ref);

    UC(grid_sampler_data_ini(&s->d));
    UC(grid_sampler_ini(colors, stress, L, N, &s->s));
}

_S_ void ini_dump(int maxp, MPI_Comm cart, const Coords *c, const Opt *opt, Dump *d) {
    enum {NPARRAY = 3};
    EMALLOC(NPARRAY * maxp, &d->pp);

    UC(diag_part_ini("diag.txt", &d->diagpart));
    if (opt->dump.parts) UC(io_bop_ini(cart, maxp, &d->bop));
    if (opt->dump.field) {
        UC(ini_sampler(c, opt, &d->field_sampler));
        UC(os_mkdir(DUMP_BASE "/raw"));
    }

    d->id_bop = d->id_strt = 0;
}

_S_ void ini_time(const Config *cfg,      Time *t) {
    const float t0 = 0;
    UC(conf_lookup_float(cfg, "time.end",    &t->end));
    UC(conf_lookup_float(cfg, "time.wall",   &t->eq));
    UC(conf_lookup_float(cfg, "time.mbr_bb", &t->mbr_bb));
    UC(time_line_ini(t0, &t->t));
    UC(time_step_ini(cfg, &t->step));
    UC(time_step_accel_ini(&t->accel));
}

_S_ void ini_common(const Config *cfg, int3 L, MPI_Comm cart,      Sim *s) {
    MC(m::Comm_dup(cart, &s->cart));
    UC(coords_ini(s->cart, L.x, L.y, L.z,      &s->coords));
    UC(coords_log(s->coords));

    UC(dbg_ini(&s->dbg));
    UC(dbg_set_conf(cfg, s->dbg));

    UC(ini_time(cfg, &s->time));
}

_S_ void ini_optional_features(const Config *cfg, const Opt *opt, Sim *s) {
    int maxp = opt_estimate_maxp(opt);
    int3 L = opt->params.L;

    if (opt->tracers)    UC(ini_tracers(maxp));
    if (opt->vcon)       UC(ini_vcon(s->cart, L, cfg,      &s->vcon));
    if (opt->outflow)    UC(ini_outflow(s->coords, maxp, cfg,      &s->outflow));
    if (opt->inflow)     UC(ini_inflow (s->coords, L, cfg,       &s->inflow ));
    if (opt->denoutflow) UC(ini_denoutflow(s->coords, maxp, cfg,      &s->denoutflow, &s->mapoutflow));
}

void sim_ini(const Config *cfg, MPI_Comm cart,      Sim **sim) {
    float dt;
    Sim *s;
    int maxp;
    Opt *opt;
    int3 L;

    EMALLOC(1, sim);
    s = *sim;
    opt = &s->opt;

    UC(opt_read(cfg, opt));
    UC(read_recolor_opt(cfg, &s->recolorer));
    UC(read_tracer_opt(cfg, &s->tracers, opt));
    UC(opt_check(opt));
    L = opt->params.L;

    UC(ini_common(cfg, L, cart,      s));

    dt = time_step_dt0(s->time.step);
    time_line_advance(dt, s->time.t);

    maxp = opt_estimate_maxp(opt);

    UC(ini_dump(maxp, s->cart, s->coords, opt,      &s->dump));

    UC(bforce_ini(&s->bforce));
    UC(bforce_set_conf(cfg, s->bforce));

    UC(ini_flu(cfg, opt, s->cart, maxp,      &s->flu));
    if (opt->wall.active) UC(wall_ini(cfg, L, &s->wall));

    UC(objects_ini(cfg, opt, s->cart, s->coords, maxp, &s->obj));

    UC(obj_inter_ini(cfg, opt, s->cart, dt, maxp,      &s->objinter));

    UC(ini_optional_features(cfg, opt,      s));

    UC(scheme_restrain_ini(&s->restrain));
    UC(scheme_restrain_set_conf(cfg, s->restrain));

    UC(inter_color_ini(&s->gen_color));
    UC(inter_color_set_conf(cfg, s->gen_color));

    if (m::is_master(s->cart))
        UC(utils_dump_history(cfg, "conf.history.cfg"));

    s->active_mbr_bb = false;

    MC(m::Barrier(s->cart));
}
_S_ void fin_flu_exch(     FluExch *e) {
    UC(eflu_pack_fin(     e->p));
    UC(eflu_comm_fin(     e->c));
    UC(eflu_unpack_fin(     e->u));
}

_S_ void fin_flu_distr(     FluDistr *d) {
    UC(dflu_pack_fin(     d->p));
    UC(dflu_comm_fin(     d->c));
    UC(dflu_unpack_fin(     d->u));
}

_S_ void fin_outflow(Outflow *o) {
    UC(outflow_fin(     o));
}

_S_ void fin_denoutflow(DCont *d, DContMap *m) {
    UC(den_fin(d));
    UC(den_map_fin(m));
}

_S_ void fin_inflow(Inflow *i) {
    UC(inflow_fin(     i));
}


_S_ void fin_flu(const Opt *opt, Flu *f) {
    UC(flu_fin(&f->q));
    UC(fluforces_bulk_fin(     f->bulk));
    UC(fluforces_halo_fin(     f->halo));

    UC(fin_flu_distr(     &f->d));
    UC(fin_flu_exch(     &f->e));

    UC(Dfree(f->ff));
    EFREE(f->ff_hst);

    if (opt->flu.ss) {
        UC(Dfree(f->ss));
        EFREE(f->ss_hst);
    }
}

_S_ void fin_vcon(Vcon *c) {
    UC(vcont_fin(c->vcont));
}

_S_ void fin_pair_params(Sim *s) {
    UC(pair_fin(s->flu.params));
}

_S_ void fin_sampler(Sampler *s) {
    UC(grid_sampler_data_fin(s->d));
    UC(grid_sampler_fin(s->s));
}

_S_ void fin_dump(const Opt *opt, Dump *d) {
    UC(diag_part_fin(d->diagpart));
    if (opt->dump.parts) UC(io_bop_fin(d->bop));
    if (opt->dump.field) UC(fin_sampler(&d->field_sampler));
    EFREE(d->pp);
}

_S_ void fin_time(Time *t) {
    UC(time_step_fin(t->step));
    UC(time_step_accel_fin(t->accel));
}

_S_ void fin_optional_features(const Opt *opt, Sim *s) {
    if (opt->tracers)    UC(tracers_fin());
    if (opt->vcon)       UC(fin_vcon(     &s->vcon));
    if (opt->outflow)    UC(fin_outflow(     s->outflow));
    if (opt->inflow)     UC(fin_inflow (     s->inflow ));
    if (opt->denoutflow) UC(fin_denoutflow(     s->denoutflow, s->mapoutflow));
}

void sim_fin(Sim *s) {
    const Opt *opt = &s->opt;

    UC(fin_optional_features(opt, s));

    UC(fin_flu(opt, &s->flu));
    if (opt->wall.active) UC(wall_fin(s->wall));

    UC(obj_inter_fin(s->objinter));
    UC(objects_fin(s->obj));

    UC(bforce_fin(s->bforce));

    UC(fin_dump(opt, &s->dump));
    UC(scheme_restrain_fin(s->restrain));
    UC(coords_fin(     s->coords));
    UC(fin_time(&s->time));
    UC(fin_pair_params(s));
    UC(inter_color_fin(s->gen_color));
    UC(dbg_fin(s->dbg));

    MC(m::Comm_free(&s->cart));
    EFREE(s);
}
_S_ void check_size(long n, long nmax) {
    if (n < 0 || n > nmax)
        ERR("wrong size: %ld / %ld", n, nmax);
}

_I_ void check_sizes(Sim *s) {
    UC(check_size(s->flu.q.n , s->flu.q.maxp));
    UC(objects_check_size(s->obj));
}

_I_ void check_pos_soft(Sim *s) {
    UC(dbg_check_pos_soft(s->coords, "flu", s->dbg, s->flu.q.n, s->flu.q.pp));
}

_I_ void check_vel(float dt, Sim *s) {
    const Coords *c = s->coords;
    UC(dbg_check_vel(dt, c, "flu", s->dbg, s->flu.q.n, s->flu.q.pp));
    UC(objects_check_vel(s->obj, s->dbg, dt));
}

_I_ void check_forces(float dt, Sim *s) {
    const Coords *c = s->coords;
    UC(dbg_check_forces(dt, c, "flu.ff", s->dbg, s->flu.q.n, s->flu.q.pp, s->flu.ff));
    UC(objects_check_forces(s->obj, s->dbg, dt));
}
_S_ long download_all_pp(Sim *s) {
    long i, np = 0;
    Flu *flu = &s->flu;
    PFarrays *pf;
    PFarray p;
    UC(pfarrays_ini(&pf));
    UC(objects_get_particles_all(s->obj, pf));

    if (flu->q.n) {
        aD2H(s->dump.pp + np, flu->q.pp, flu->q.n);
        np += flu->q.n;
    }
    for (i = 0; i < pfarrays_size(pf); ++i) {
        pfarrays_get(i, pf, &p);
        aD2H(s->dump.pp + np, (const Particle*) p.p.pp, p.n);
        np += p.n;
    }

    UC(pfarrays_fin(pf));
    dSync();
    return np;
}

_S_ void dump_diag(float t, Sim *s) {
    if (t < 0) ERR("time = %g < 0", t);
    long n;
    n = download_all_pp(s);
    UC(diag_part_apply(s->dump.diagpart, s->cart, t, n, s->dump.pp));
    UC(objects_diag_dump(t, s->obj));
}

_S_ void dump_parts(Sim *s) {
    const Flu *flu = &s->flu;
    const Opt *opt = &s->opt;
    IoBop *bop = s->dump.bop;
    int id_bop = s->dump.id_bop;
    cD2H(flu->q.pp_hst, flu->q.pp, flu->q.n);
    if (opt->flu.ids) {
        cD2H(flu->q.ii_hst, flu->q.ii, flu->q.n);
        io_bop_ids(s->cart, flu->q.n, flu->q.ii_hst, "id_solvent", id_bop, bop);
    }
    if (opt->flu.colors) {
        cD2H(flu->q.cc_hst, flu->q.cc, flu->q.n);
        io_bop_colors(s->cart, flu->q.n, flu->q.cc_hst, "colors_solvent", id_bop, bop);
    }
    if (opt->flu.ss) {
        cD2H(flu->ss_hst, flu->ss, 6 * flu->q.n);
        io_bop_stresses(s->cart, flu->q.n, flu->ss_hst, "stress_solvent", id_bop, bop);
    }

    if (opt->dump.forces) {
        cD2H(flu->ff_hst, flu->ff, flu->q.n);
        io_bop_parts_forces(s->cart, s->coords, flu->q.n, flu->q.pp_hst, flu->ff_hst, "solvent", id_bop, bop);
    } else {
        io_bop_parts(s->cart, s->coords, flu->q.n, flu->q.pp_hst, "solvent", id_bop, bop);
    }

    UC(objects_part_dump(id_bop, s->obj, bop));
    s->dump.id_bop = ++id_bop;
}

_S_ void dump_mesh(Sim *s) {
    UC(objects_mesh_dump(s->obj));
}

_S_ void dump_field(Sim *s) {
    Dump *d = &s->dump;
    UC(grid_sampler_dump(s->cart, "raw", d->id_field, d->field_sampler.s));
    UC(grid_sampler_reset(d->field_sampler.s));
    d->id_field ++;
}

_I_ void dump_strt_templ(Sim *s) {
    const Opt *opt = &s->opt;
    const char *base = opt->dump.strt_base_dump;
    if (opt->dump.strt) {
        if (opt->wall.active) UC(wall_dump_templ(s->wall, s->cart, base));
        UC(objects_strt_templ(base, s->obj));
    }
}

_S_ void dump_strt0(int id, Sim *s) {
    Flu *flu = &s->flu;
    const Opt *opt = &s->opt;
    const char *base = opt->dump.strt_base_dump;
    UC(flu_strt_dump(s->cart, base, id, &flu->q));
    UC(objects_strt_dump(base, id, s->obj));
    if (opt->vcon)   vcont_strt_dump(s->cart, base, id, s->vcon.vcont);
}

_S_ void dump_strt(Sim *s) {
    dump_strt0(s->dump.id_strt++, s);
}

_I_ void dump_strt_final(Sim *s) {
    dump_strt0(RESTART_FINAL, s);
}

_I_ void dump(TimeLine *time, Sim *s) {
    const OptDump *o = &s->opt.dump;
    if (time_line_cross(time, o->freq_diag))
        dump_diag(time_line_get_current(time), s);
    if (o->parts && time_line_cross(time, o->freq_parts))
        dump_parts(s);
    if (o->mesh && time_line_cross(time, o->freq_mesh))
        dump_mesh(s);
    if (o->field && time_line_cross(time, o->freq_field))
        dump_field(s);
    if (o->strt  && time_line_cross(time, o->freq_strt))
        dump_strt(s);
}


_S_ bool valid_step(long id, const int freq) {
    return (freq != 0) && (id % freq == 0);
}

_I_ void sample_vcont(const Coords *coords, long id, const Flu *f,      Vcon *c) {
    if (valid_step(id, c->sample_freq)) {
        const FluQuants *q = &f->q;
        vcont_sample(coords, q->n, q->pp, clists_get_ss(q->cells), clists_get_cc(q->cells),      c->vcont);
    }
}

_I_ void adjust_bforce(long id,      Vcon *c, BForce *bforce) {
    if (valid_step(id, c->adjust_freq)) {
        float3 f;
        f = vcont_adjustF(     c->vcont);
        bforce_adjust(f,      bforce);
    }
}

_I_ void log_vcont(long id, const Vcon *c) {
    if (valid_step(id, c->log_freq))
        vcont_log(c->vcont);
}



_I_ void recolor_flux(const Coords *c, const Recolorer *opt, Flu *f) {
    int3 L = subdomain(c);
    if (opt->flux_active)
        UC(color_linear_flux(c, L, opt->flux_dir, RED_COLOR, f->q.n, f->q.pp,      f->q.cc));
}

_I_ void recolor_tracers(const Coords *c, const Tracers *opt, const int it, Flu *f) {
    bool cond;
    cond = opt->active && it % opt->freq == 0;
    if (cond) {
        UC(  color_tracers(c, RED_COLOR,  opt->R, opt->iniP, f->q.n, f->q.pp,      f->q.cc));
        UC(decolor_tracers(c, BLUE_COLOR, opt->R, opt->delP, f->q.n, f->q.pp,      f->q.cc));
    }
}


_I_ void apply_inflow(float kBT, int numdensity, float dt, Inflow *i, Flu *f) {
    if (f->q.colors)
        UC(inflow_create_pp_cc(kBT, numdensity, dt, RED_COLOR, i, &f->q.n, f->q.pp, f->q.cc));
    else
        UC(inflow_create_pp(kBT, numdensity, dt, i, &f->q.n, f->q.pp));
}

_I_ void mark_outflow(const Flu *f, Outflow *o) {
    UC(outflow_filter_particles(f->q.n, f->q.pp,      o));
    UC(outflow_download_ndead(o));
}

_I_ void mark_outflowden(OptParams params, const Flu *f, const DContMap *m,      DCont *d) {
    const int *ss, *cc;
    int n;
    n = f->q.n;
    ss = clists_get_ss(f->q.cells);
    cc = clists_get_cc(f->q.cells);

    UC(den_reset(n,      d));
    UC(den_filter_particles(params.numdensity, m, ss, cc,      d));
    UC(den_download_ndead(     d));
}

_S_ void sample(Sim *s) {
    Flu *flu = &s->flu;
    Sampler *sam = &s->dump.field_sampler;

    UC(grid_sampler_data_reset(sam->d));
    UC(grid_sampler_data_push(flu->q.n, flu->q.pp, flu->q.cc, flu->ss, sam->d));
    UC(grid_sampler_add(sam->d, sam->s));
}

_I_ void field_sample(Sim *s) {
    if (s->opt.dump.field && is_sampling_time(s))
        sample(s);
}

_I_ void colors_from_rbc(Sim *s) {
    PFarray flu;
    const Opt *opt = &s->opt;

    if (!opt->flu.colors) return;

    utils_get_pf_flu(s, &flu);
    UC(objects_recolor_flu(s->obj, &flu));
}

_I_ void recolor_from_rbc(long it, Sim *s) {
    bool cond;
    const Opt *opt = &s->opt;
    cond = opt->flu.colors && opt->recolor_freq && (it % opt->recolor_freq == 0);
    if (cond) {
        msg_print("recolor inside rbc");
        UC(colors_from_rbc(s));
    }
}

_I_ void activate_mbr_bb(Sim *s) {
    UC(colors_from_rbc(s));
    UC(objects_activate_mbr_bounce(s->obj));
}
_I_ void clear_vel(Sim *s) {
    Flu *flu = &s->flu;
    UC(scheme_move_clear_vel(flu->q.n, flu->q.pp));
    UC(objects_clear_vel(s->obj));
}

_I_ void update_solvent(float dt,      Flu *f) {
    UC(scheme_move_apply(dt, f->mass, f->q.n, f->ff, f->q.pp));
}

_I_ void store_solvent(Flu *f) {
    FluQuants *q = &f->q;
    long n = q->n;
    if (n)
        aD2D(q->pp0, q->pp, n);
}

_I_ void restrain(long it, Sim *s) {
    SchemeQQ qq;
    PFarrays *pf;
    PFarray p;
    UC(pfarrays_ini(&pf));
    UC(objects_get_particles_mbr(s->obj, pf));

    if (pfarrays_size(pf) > 0) {
        UC(pfarrays_get(0, pf, &p));
        qq.r  = (Particle*) p.p.pp;
        qq.rn = p.n;
    } else {
        qq.r  = NULL;
        qq.rn = 0;
    }
    qq.o = s->flu.q.pp;
    qq.on = s->flu.q.n;

    UC(scheme_restrain_apply(s->cart, s->flu.q.cc, it,      s->restrain, qq));

    UC(pfarrays_fin(pf));
}

_I_ void bounce_wall(float dt, Sim *s) {
    Flu *f = &s->flu;
    PaArray pa;
    FoArray fo;
    PFarrays *pf;
    UC(pfarrays_ini(&pf));
    UC(parray_push_pp(f->q.pp, &pa));
    UC(pfarrays_push(pf, f->q.n, pa, fo));

    UC(objects_get_particles_mbr(s->obj, pf));

    UC(wall_bounce(s->wall, s->coords, dt, pf));
    UC(pfarrays_fin(pf));
}

_I_ void bounce_objects(float dt, Sim *s) {
    Flu *flu = &s->flu;
    FluQuants *q = &flu->q;
    UC(objects_bounce(dt, flu->mass, q->cells, q->n, q->pp0, q->pp, s->obj));
}

_S_ void flu_update_dpd_prms(float dt, float kBT, Sim *s) {
    UC(pair_compute_dpd_sigma(kBT, dt,      s->flu.params));
}

_I_ void update_params(float dt, Sim *s) {
    float kBT;
    const Opt *opt = &s->opt;
    kBT = opt->params.kBT;
    UC(flu_update_dpd_prms(dt, kBT, s));
    UC(objects_update_dpd_prms(dt, kBT, s->obj));
}



_S_ void log_and_fail(Coords *c, DFluStatus *s, FluQuants *q) {
    unsigned int time;
    time = 10;
    UC(dflu_status_log(s));
    UC(flu_txt_dump(c, q));
    msg_print("sleep for %d seconds", time);
    os_sleep(time);
    ERR("dflu_download failed");
}

_I_ void distribute_flu(Sim *s) {
    PartList lp;
    FluQuants *q = &s->flu.q;
    FluDistr *d = &s->flu.d;
    int ndead;
    const Opt *opt = &s->opt;

    lp.pp = q->pp;

    if (opt->denoutflow) {
        lp.deathlist = den_get_deathlist(s->denoutflow);
        ndead        = den_get_ndead(s->denoutflow);
    }
    else if (opt->outflow) {
        lp.deathlist = outflow_get_deathlist(s->outflow);
        ndead        = outflow_get_ndead    (s->outflow);
    } else {
        lp.deathlist = NULL;
        ndead = 0;
    }
    UC(dflu_build_map(q->n, lp,      d->p));
    UC(dflu_pack(q,      d->p));

    UC(dflu_download(     d->p, d->s));
    if (!dflu_status_success(d->s))
        UC(log_and_fail(s->coords, d->s, q));

    UC(dflu_post_send(d->p, d->c));
    UC(dflu_post_recv(d->c, d->u));

    UC(dflu_bulk(lp,      q));

    UC(dflu_wait_send(d->c));
    UC(dflu_wait_recv(d->c, d->u));

    UC(dflu_unpack(     d->u));

    UC(dflu_halo(d->u,      q));
    UC(dflu_gather(ndead, d->p, d->u,      q));

    dSync();
}
_S_ void pre_run(Sim *s) {
    s->equilibrating = false;

    UC(dump_strt_templ(s));
    UC(utils_compute_hematocrit(s));
}

_S_ void step(TimeLine *time, float dt, float tstart, Sim *s) {
    long it;
    Flu *flu = &s->flu;
    Wall *wall = s->wall;
    BForce *bforce = s->bforce;
    const Opt *opt = &s->opt;

    UC(update_params(dt, s));

    if (active_walls(s))
        wall_update_vel(time_line_get_current(time) - tstart, wall);

    UC(check_sizes(s));
    UC(check_pos_soft(s));

    UC(distribute_flu(s));
    UC(objects_distribute(s->obj));

    if (objects_have_bounce(s->obj)) {
        if (is_mbr_bb_activation_time(s))
            activate_mbr_bb(s);

        UC(store_solvent(&s->flu));
        UC(objects_save_mesh(s->obj));
    }

    UC(check_sizes(s));
    UC(forces(dt, time, bforce, s));
    UC(check_forces(dt, s));

    it = time_line_get_iteration(time);
    UC(field_sample(s));
    UC(dump(time, s));

    UC(restrain(it,      s));
    UC(update_solvent(dt,      flu));
    UC(objects_advance(dt, s->obj));
    UC(recolor_from_rbc(it, s));

    UC(check_vel(dt, s));
    if (opt->vcon && !s->equilibrating) {
        sample_vcont(s->coords, it, flu,      &s->vcon);
        adjust_bforce(it,      &s->vcon, bforce);
        log_vcont(it, &s->vcon);
    }

    if (active_walls(s)) bounce_wall(dt, s);

    UC(bounce_objects(dt, s));

    UC(check_pos_soft(s));
    UC(check_vel(dt, s));

    if (!s->equilibrating) {
        if (opt->inflow)     UC(apply_inflow(s->opt.params.kBT, s->opt.params.numdensity, dt, s->inflow,      flu));
        if (opt->outflow)    UC(mark_outflow(flu,      s->outflow));
        if (opt->denoutflow) UC(mark_outflowden(s->opt.params, flu, s->mapoutflow,      s->denoutflow));
        if (opt->flu.colors) UC(recolor_flux(s->coords, &s->recolorer, flu));
        if (opt->flu.colors) UC(recolor_tracers(s->coords, &s->tracers, it, flu));
    }
}


_I_ void run(float ts, float te, Sim *s) {
    float dt;

    dt = utils_get_dt0(s);
    while (time_line_get_current(s->time.t) < te) {
        UC(step(s->time.t, dt, ts, s));
        time_line_advance(dt, s->time.t);
        dt = utils_get_dt(s, s->time.t);
    }
    UC(distribute_flu(     s));
}
_S_ void gen_flu(Sim *s) {
    Flu *flu = &s->flu;
    UC(flu_gen_quants(s->coords, s->opt.params.numdensity, s->gen_color, &flu->q));
    UC(flu_build_cells(&flu->q));
    if (s->opt.flu.ids) flu_gen_ids(s->cart, flu->q.n, &flu->q);
}

_S_ void gen_wall(Sim *s) {
    Flu *flu = &s->flu;
    Wall *w = s->wall;
    bool dump_sdf = s->opt.dump.field;

    if (!s->opt.wall.active) return;

    UC(wall_gen(s->cart, s->coords, s->opt.params, dump_sdf,
                &flu->q.n, flu->q.pp, w));
}

_S_ void freeze(Sim *s) {
    Flu *flu = &s->flu;
    const Sdf *sdf = NULL;
    PFarray pfflu;
    const Opt *opt = &s->opt;

    UC(objects_gen_mesh(s->obj));
    dSync();

    UC(gen_wall(s));

    if (s->opt.wall.active) UC(wall_get_sdf_ptr(s->wall, &sdf));

    UC(objects_remove_from_wall(sdf, s->obj));
    dSync();

    UC(utils_get_pf_flu(s, &pfflu));
    UC(objects_gen_freeze(&pfflu, s->obj));
    UC(utils_set_n_flu_pf(&pfflu, s));
    dSync();

    UC(clear_vel(s));

    if (opt->flu.colors) {
        Particle *pp = flu->q.pp;
        int n = flu->q.n;
        int *cc = flu->q.cc;
        inter_color_apply_dev(s->coords, s->gen_color, n, pp,       cc);
    }
}

void sim_gen(Sim *s) {
    const Opt *opt = &s->opt;
    float tstart = 0;
    s->equilibrating = true;

    UC(gen_flu(s));

    MC(m::Barrier(s->cart));

    run(tstart, s->time.eq, s);

    freeze(     s);
    dSync();

    if (opt->flu.colors) UC(colors_from_rbc(s));

    tstart = s->time.eq;
    pre_run(s);
    run(tstart, s->time.end, s);


    if (opt->dump.strt) dump_strt_final(s);
}

_S_ void gen_from_restart(Sim *s) {
    Flu *flu = &s->flu;
    const Opt *opt = &s->opt;
    const char *base = opt->dump.strt_base_read;
    bool dump_sdf = opt->dump.field;

    UC(flu_strt_quants(s->cart, base, RESTART_BEGIN, &flu->q));
    UC(objects_restart(s->obj));
    if (opt->wall.active) wall_restart(s->cart, s->coords, opt->params, dump_sdf, base, s->wall);
    if (opt->vcon) vcont_strt_read(base, RESTART_BEGIN, s->vcon.vcont);
}

void sim_strt(Sim *s) {
    const Opt *opt = &s->opt;

    gen_from_restart(s);

    MC(m::Barrier(s->cart));

    pre_run(s);
    run(s->time.eq, s->time.end, s);
    if (opt->dump.strt) UC(dump_strt_final(s));
}

#undef _I_
#undef _S_
