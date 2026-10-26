// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/rbc/force/rnd/imp.cpp */
#include <stdlib.h>
#include <stdio.h>
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
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
#ifndef UDX_FORCE_DEVICE_API_COMMON
#define UDX_FORCE_DEVICE_API_COMMON
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
#endif
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

/* begin src/rbc/force/rnd/api/type.h */
namespace rbc_api {
typedef struct Generator_st *Generator_t;
} /* namespace */
/* end src/rbc/force/rnd/api/type.h */
/* begin src/rbc/force/rnd/api/imp.h */
namespace rnd_api {
typedef struct Generator_st *Generator_t;
int CreateGenerator(Generator_t *g);
int SetPseudoRandomGeneratorSeed(Generator_t g,  long seed);
int DestroyGenerator(Generator_t g);
int GenerateNormal(Generator_t g, float *outputPtr, size_t n);
}
/* end src/rbc/force/rnd/api/imp.h */
/* begin src/rbc/force/rnd/imp.h */
struct RbcRnd;

// tag::enum[]
#ifndef UDX_RBC_FORCE_SEED_TIME
#define UDX_RBC_FORCE_SEED_TIME
enum {SEED_TIME = -1}; /* magic seed: ini from "time" */
#endif
// end::enum[]

// tag::mem[]
void rbc_rnd_ini(int n, long seed, RbcRnd**);
void rbc_rnd_fin(RbcRnd*);
// end::mem[]

// tag::int[]
void rbc_rnd_gen(RbcRnd*, int n, /**/ float**); // <1>
float rbc_rnd_get_hst(const RbcRnd*, int i);    // <2>
// end::int[]
/* end src/rbc/force/rnd/imp.h */

/* begin src/rbc/force/rnd/imp/type.h */
struct RbcRnd { /* data */
    float *r;
    rnd_api::Generator_t g;
    int max;
};
/* end src/rbc/force/rnd/imp/type.h */
/* begin src/rbc/force/rnd/imp/cu.h */
#define CU(x)                                                           \
    do {                                                                \
        int _curc = (x);                                                \
        if ( _curc != 0)                                                \
            ERR("curand error status: %d", _curc);                      \
    } while(0)
/* end src/rbc/force/rnd/imp/cu.h */
/* begin src/rbc/force/rnd/imp/seed.h */
static long decode_seed_time() {
    long t;
    t = os_time();
    msg_print("seed time: %ld", t);
    return t;
}
static int decode_seed(long seed) {
    if (seed == SEED_TIME) return decode_seed_time();
    else return seed;
}
/* end src/rbc/force/rnd/imp/seed.h */
/* begin src/rbc/force/rnd/imp/main.h */
static void ini0(RbcRnd *d, int n, long seed) {
    Dalloc(&d->r, n);
    CU(rnd_api::CreateGenerator(&d->g));
    seed = decode_seed(seed);
    CU(rnd_api::SetPseudoRandomGeneratorSeed(d->g,  seed));
    d->max = n;
}
void rbc_rnd_ini(int n, long seed, RbcRnd **pq) {
    RbcRnd *q;
    EMALLOC(1, &q);
    UC(ini0(q, n, seed));
    *pq = q;
}

static void fin0(RbcRnd *d) {
    Dfree(d->r);
    CU(rnd_api::DestroyGenerator(d->g));
}
void rbc_rnd_fin(RbcRnd *d) {
    fin0(d);
    EFREE(d);
}

static void assert_n(int n, int max, const char *s) {
    if (n > max)
        ERR("%s: n = %d > max = %d", s, n , max);
}

void rbc_rnd_gen(RbcRnd *d, int n, float **pq) {
    assert_n(n, d->max, "rbc::rnd::gen");
    rnd_api::GenerateNormal(d->g, d->r, n);
    *pq = d->r;
}

float rbc_rnd_get_hst(const RbcRnd *d, int i) {
    float x;
    float *r;
    assert_n(i, d->max, "rbc::rnd::get_hst");
    r = d->r;
    cD2H(&x, &r[i], 1);
    return x;
}
/* end src/rbc/force/rnd/imp/main.h */
/* end src/rbc/force/rnd/imp.cpp */

#line 1 "src/rbc/imp.cpp"
