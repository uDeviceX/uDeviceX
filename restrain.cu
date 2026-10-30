#include "memory.cuh"
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include "config.cuh"
#include "macros.cuh"
#include "types.cuh"
#include "constants.cuh"
#include "device.cuh"
#include "launch.cuh"

#include <mpi.h>
void msg_print(const char *, ...);

void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);

#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)
namespace cc {
void check(int, const char *, int);
}

#define MC(ans) do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)
namespace mpicheck {
void check(int, const char *, int);
}

namespace m {
int Allreduce(const void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
}

struct SchemeQQ {
    Particle *o;
    Particle *r;
    int on;
    int rn;
};

#include "warp.cuh"


#ifndef UDX_SIM_SCHEME_DEVICE_API_COMMON
#define UDX_SIM_SCHEME_DEVICE_API_COMMON
#include "device.cuh"
#endif



namespace restrain_dev {
struct MapColor {
    const int *cc;
    int color;
};

struct MapGrey {};
static __device__ int valid_p(const MapGrey, int) { return 1; }
static __device__ int valid_p(const MapColor m, int i) {
    return m.cc[i] == m.color;
}

template <typename Map>
__global__ void sum(Map m, int n, const Particle *pp,      int *ntot, float3 *vtot) {
    enum {X, Y, Z};
    int i, good, ngood;
    float3 v;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    good = (i < n) && valid_p(m, i);

    if (good) {
        const Particle p = pp[i];
        v.x = p.v[X];
        v.y = p.v[Y];
        v.z = p.v[Z];
    } else {
        v.x = v.y = v.z = 0;
    }

    v     = warpReduceSum(v);
    ngood = warpReduceSum(good);

    if ((threadIdx.x % warpSize == 0) && ngood > 0) {
        atomicAdd(&vtot->x, v.x);
        atomicAdd(&vtot->y, v.y);
        atomicAdd(&vtot->z, v.z);
        atomicAdd(ntot, ngood);
    }
}

template <typename Map>
__global__ void shift(Map m, float3 v, int n,      Particle *pp) {
    enum {X, Y, Z};
    int i, good;
    i = threadIdx.x + blockDim.x * blockIdx.x;

    good = (i < n) && valid_p(m, i);

    if (good) {
        Particle p = pp[i];
        p.v[X] -= v.x;
        p.v[Y] -= v.y;
        p.v[Z] -= v.z;
        pp[i] = p;
    }
}
}

enum {
    RSTR_NONE,
    RSTR_COL,
    RSTR_RBC
};

struct Restrain {
    int kind;
    int freq;


    int *n;
    float3 *v;
};
void scheme_restrain_ini(Restrain **rstr) {
    Restrain *r;
    EMALLOC(1, rstr);
    r = *rstr;

    CC(d::alloc_pinned((void**) &r->n, sizeof(int)));
    CC(d::alloc_pinned((void**) &r->v, sizeof(float3)));
}

void scheme_restrain_fin(Restrain *r) {
    CC(d::FreeHost(r->n));
    CC(d::FreeHost(r->v));
    EFREE(r);
}

void scheme_restrain_set_red(Restrain *r) {
    r->kind = RSTR_COL;
}

void scheme_restrain_set_rbc(Restrain *r) {
    r->kind = RSTR_RBC;
}

void scheme_restrain_set_none(Restrain *r) {
    r->kind = RSTR_NONE;
}

void scheme_restrain_set_freq(int freq, Restrain *r) {
    r->freq = freq;
}

static const char* kind2tag(int i) {
    static const char *tags[] = {"NONE", "RED", "RBC"};
    return tags[i];
}

static void print_stats(const char *tag, int n, float3 v) {
    msg_print("restrain %s: n = %d [% .3e % .3e % .3e]", tag, n, v.x, v.y, v.z);
}

static void report(const Restrain *r, long it) {
    int n, freq, cond, kind;
    float3 v;
    kind = r->kind;
    freq = r->freq;
    cond = freq > 0 && it % freq == 0 && kind != RSTR_NONE;
    n = *(r->n);
    v = *(r->v);
    if (cond)
        print_stats(kind2tag(kind), n, v);
}

static void reini(Restrain *r) {
    *(r->n) = 0;
    *(r->v) = make_float3(0, 0, 0);
}

static int all_reduce(MPI_Comm comm, int n) {
    int ntot = 0;
    MC(m::Allreduce(&n, &ntot, 1, MPI_INT, MPI_SUM, comm));
    return ntot;
}

static float3 all_reduce(MPI_Comm comm, float3 v) {
    float3 vtot = make_float3(0, 0, 0);
    MC(m::Allreduce(&v.x, &vtot.x, 3, MPI_FLOAT, MPI_SUM, comm));
    return vtot;
}

template <typename Map>
static void apply(MPI_Comm comm, Map m, int np,      Particle *pp, Restrain *r) {
    int n;
    float3 v;
    float scale;

    reini(      r);
    KL(restrain_dev::sum, (k_cnf(np)), (m, np, pp,      r->n, r->v));
    dSync();

    n = *(r->n);
    v = *(r->v);

    n = all_reduce(comm, n);
    v = all_reduce(comm, v);

    scale = n ? 1.0 / n : 1;
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;

    KL(restrain_dev::shift, (k_cnf(np)), (m, v, np,      pp));

    *(r->v) = v;
    *(r->n) = n;
}

void scheme_restrain_apply(MPI_Comm comm, const int *cc, long it,      Restrain *r, SchemeQQ qq) {
    int kind;
    kind = r->kind;
    switch (kind) {
    case RSTR_NONE:
        break;
    case RSTR_COL:
        restrain_dev::MapColor mc;
        mc.cc = cc;
        mc.color = RED_COLOR;
        apply(comm, mc, qq.on,      qq.o, r);
        report(r, it);
        break;
    case RSTR_RBC:
        restrain_dev::MapGrey mg;
        apply(comm, mg, qq.rn,      qq.r, r);
        report(r, it);
        break;
    default:
        ERR("Unknown kind <%d>", kind);
    };
}

