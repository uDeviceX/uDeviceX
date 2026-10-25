// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/rbc/force/rnd/api/imp.cpp */
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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

namespace rnd_api {

#if   defined(DEV_CUDA)
/* begin src/rbc/force/rnd/api/imp/cuda.h */
struct Generator_st { curandGenerator_t g; };

static int CreateGenerator0(curandGenerator_t *g) {
    return curandCreateGenerator(g, CURAND_RNG_PSEUDO_DEFAULT);
}
int CreateGenerator(Generator_t *pg) {
    int rc;
    Generator_st *s;
    EMALLOC(1, &s);
    rc = CreateGenerator0(&s->g);
    *pg = s;
    return rc;
}

int DestroyGenerator(Generator_t s) {
    int rc;
    rc = curandDestroyGenerator(s->g);
    EFREE(s);
    return rc;
}

int SetPseudoRandomGeneratorSeed(Generator_t s,  long seed) {
    return curandSetPseudoRandomGeneratorSeed(s->g, seed);
}

int GenerateNormal(Generator_t s, float *outputPtr, size_t n) {
    float mean, std;
    mean = 0; std = 1;
    return curandGenerateNormal(s->g, outputPtr, n, mean, std);
}
/* end src/rbc/force/rnd/api/imp/cuda.h */
#elif defined(DEV_CPU)
/* begin src/rbc/force/rnd/api/imp/gaussrand.h */
/* Gaussian rnd with mean = 0 and std = 1
   Knuth and due originally to Marsaglia */
double gaussrand() {
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if (phase == 0) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    phase = 1 - phase;
    return X;
}
/* end src/rbc/force/rnd/api/imp/gaussrand.h */
/* begin src/rbc/force/rnd/api/imp/cpu.h */
int CreateGenerator(Generator_t*) { return 0; }
int DestroyGenerator(Generator_t) { return 0; }
int SetPseudoRandomGeneratorSeed(Generator_t,  long seed) { srand(seed); return 0; }
int GenerateNormal(Generator_t, float *outputPtr, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) outputPtr[i] = gaussrand();
    return 0;
}
/* end src/rbc/force/rnd/api/imp/cpu.h */
#else
  #error DEV_* is undefined
#endif

} /* namespace */
/* end src/rbc/force/rnd/api/imp.cpp */

#line 1 "src/rbc/force/rnd/imp.cpp"
