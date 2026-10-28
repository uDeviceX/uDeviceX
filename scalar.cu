#include <stdio.h>

struct Vectors;
void vectors_get(const Vectors*, int i, float r[3]);

void emalloc(size_t, void**);
void efree(void*);
void error_stack_push(const char *file, int line);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *file, int line, const char *fmt, ...);

#define UC(F) do { \
        error_stack_push(__FILE__, __LINE__); \
        F; \
        if (error_get()) { error_report(); error_abort(); } \
        error_stack_pop(); \
    } while (0)
#define ERR(fmt, ...) do { \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report(); error_abort(); \
    } while (0)
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))

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
