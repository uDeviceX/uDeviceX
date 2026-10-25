// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/algo/scalars/imp.cpp */
#include <stdio.h>

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
/* begin src/algo/scalars/imp/type.h */
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

enum {FLOAT, DOUBLE, VECX, VECY, VECZ, ZERO, ONE};
static double (*get[])(const Scalars*, int i) =
{ float_get,  double_get, vecx_get, vecy_get, vecz_get, zero_get, one_get};
/* end src/algo/scalars/imp/type.h */
/* begin src/algo/scalars/imp/main.h */
void scalars_float_ini(int n, const float *rr, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = FLOAT; q->n = n; q->D.ff = rr;
    q->stamp = MAGIC;
    *pq = q;
}

void scalars_double_ini(int n, const double *rr, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = DOUBLE; q->n = n; q->D.dd = rr;
    q->stamp = MAGIC;
    *pq = q;
}

void scalars_vectors_ini(int n, const Vectors *vec, int dim, /**/ Scalars **pq) {
    enum {X, Y, Z};
    Scalars *q;
    EMALLOC(1, &q);
    switch (dim) {
    case X: q->type = VECX; break;
    case Y: q->type = VECY; break;
    case Z: q->type = VECZ; break;
    default: ERR("wrong dim=%d", dim);
    }
    q->stamp = MAGIC;
    q->n = n; q->D.vec = vec;
    *pq =q;
}

void scalars_zero_ini(int n, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = ZERO; q->n = n;
    q->stamp = MAGIC;
    *pq = q;
}


void scalars_one_ini(int n, /**/ Scalars **pq) {
    Scalars *q;
    EMALLOC(1, &q);
    q->type = ONE; q->n = n;
    q->stamp = MAGIC;
    *pq = q;
}

void scalars_fin(Scalars *q) { EFREE(q); }

static double float_get(const Scalars *q, int i) { return q->D.ff[i]; }
static double double_get(const Scalars *q, int i) { return q->D.dd[i]; }
static double vecx_get(const Scalars *q, int i)   {
    enum {X};
    float r[3];
    vectors_get(q->D.vec, i, /**/ r);
    return r[X];
}
static double vecy_get(const Scalars *q, int i)   {
    enum {X, Y};
    float r[3];
    vectors_get(q->D.vec, i, /**/ r);
    return r[Y];
}
static double vecz_get(const Scalars *q, int i)   {
    enum {X, Y, Z};
    float r[3];
    vectors_get(q->D.vec, i, /**/ r);
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
    return get[q->type](q, i);
}
/* end src/algo/scalars/imp/main.h */
/* end src/algo/scalars/imp.cpp */

#line 1 "src/algo/scan/imp.cu"
