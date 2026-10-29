#define DEV_CUDA
#define CC_RELEASE

#include "macros.cuh"
#include "device.cuh"

#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <curand.h>

void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
void emalloc(size_t, void **);
void efree(void *);
long os_time();
void msg_print(const char *, ...);

namespace cc { void check(int, const char *, int); }
#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)
#define CU(x) do { \
    int _curc = (x); \
    if (_curc != 0) ERR("curand error status: %d", _curc); \
} while (0)

enum {SEED_TIME = -1};

namespace rnd_api {
struct Generator_st { curandGenerator_t g; };
typedef Generator_st *Generator_t;

int CreateGenerator(Generator_t *out) {
    Generator_st *s;
    EMALLOC(1, &s);
    const int rc = curandCreateGenerator(&s->g, CURAND_RNG_PSEUDO_DEFAULT);
    *out = s;
    return rc;
}

int DestroyGenerator(Generator_t s) {
    const int rc = curandDestroyGenerator(s->g);
    EFREE(s);
    return rc;
}

int SetPseudoRandomGeneratorSeed(Generator_t s, long seed) {
    return curandSetPseudoRandomGeneratorSeed(s->g, seed);
}

int GenerateNormal(Generator_t s, float *out, size_t n) {
    return curandGenerateNormal(s->g, out, n, 0, 1);
}
}

struct RbcRnd {
    float *r;
    rnd_api::Generator_t g;
    int max;
};

static long decode_seed(long seed) {
    if (seed != SEED_TIME) return seed;
    const long t = os_time();
    msg_print("seed time: %ld", t);
    return t;
}

static void initialize(RbcRnd *d, int n, long seed) {
    Dalloc(&d->r, n);
    CU(rnd_api::CreateGenerator(&d->g));
    CU(rnd_api::SetPseudoRandomGeneratorSeed(d->g, decode_seed(seed)));
    d->max = n;
}

void rbc_rnd_ini(int n, long seed, RbcRnd **out) {
    RbcRnd *q;
    EMALLOC(1, &q);
    UC(initialize(q, n, seed));
    *out = q;
}

void rbc_rnd_fin(RbcRnd *d) {
    Dfree(d->r);
    CU(rnd_api::DestroyGenerator(d->g));
    EFREE(d);
}

static void check_size(int n, int max, const char *name) {
    if (n > max) ERR("%s: n = %d > max = %d", name, n, max);
}

void rbc_rnd_gen(RbcRnd *d, int n, float **out) {
    check_size(n, d->max, "rbc::rnd::gen");
    rnd_api::GenerateNormal(d->g, d->r, n);
    *out = d->r;
}

float rbc_rnd_get_hst(const RbcRnd *d, int i) {
    float x;
    check_size(i, d->max, "rbc::rnd::get_hst");
    cD2H(&x, &d->r[i], 1);
    return x;
}
