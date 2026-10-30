#include "memory.cuh"
#include "macros.cuh"
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include <stdio.h>
#include "config.cuh"
#include "constants.cuh"
#include "types.cuh"
#include "device.cuh"
#include "launch.cuh"

namespace cc { void check(int code, const char *file, int line); }
void error_stack_push(const char *file, int line);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *file, int line, const char *fmt, ...);
bool same_str(const char *, const char *);
#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)
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

#include "warp.cuh"

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

#include "vecmath.cuh"

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

