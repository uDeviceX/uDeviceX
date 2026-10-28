#include "macros.cuh"
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <vector_types.h>
#include "config.cuh"
#include "constants.cuh"
#include "types.cuh"
#include "device.cuh"
#include "launch.cuh"

namespace cc { void check(int code, const char *file, int line); }
void emalloc(size_t, void **);
void efree(void *);
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
bool same_str(const char *, const char *);
#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)
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

