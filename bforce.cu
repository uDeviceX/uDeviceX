#include "memory.cuh"
#define DEV_CUDA
#define CC_RELEASE

#include "macros.cuh"
#include "config.cuh"
#include "device.cuh"
#include "launch.cuh"
#include "types.cuh"
#include <vector_types.h>

struct Config;
struct Coords;
void coords_get_view(const Coords *, struct Coords_v *);
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);

struct Coords_v {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};

static __device__ int xdomain(const Coords_v c) { return c.Lx * c.xd; }
static __device__ int ydomain(const Coords_v c) { return c.Ly * c.yd; }
static __device__ float xl2xc(const Coords_v c, float x) {
    return x - 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
}
static __device__ float yl2yc(const Coords_v c, float y) {
    return y - 0.5f * c.Ly * (c.yd - 2 * c.yc - 1);
}
static __device__ float xl2xg(const Coords_v c, float x) {
    return (c.xc + 0.5f) * c.Lx + x;
}
static __device__ float yl2yg(const Coords_v c, float y) {
    return (c.yc + 0.5f) * c.Ly + y;
}
struct BForce_cste_v {
    float3 a;
};


struct BForce_dp_v {
    float a;
};


struct BForce_shear_v {
    float a;
};


struct BForce_rol_v {
    float a;
};


struct BForce_rad_v {
    float a;
};

enum {
    BODY_FORCE_V_NONE,
    BODY_FORCE_V_CSTE,
    BODY_FORCE_V_DP,
    BODY_FORCE_V_SHEAR,
    BODY_FORCE_V_ROL,
    BODY_FORCE_V_RAD
};

union BForceParam_v {
    BForce_cste_v cste;
    BForce_dp_v dp;
    BForce_shear_v shear;
    BForce_rol_v rol;
    BForce_rad_v rad;
};

struct BForce_v {
    BForceParam_v p;
    int type;
};

static __device__ float3 get_gp(Coords_v, BForce_cste_v par, Particle) {
    return par.a;
}


static __device__ float3 get_gp(Coords_v c, BForce_dp_v par, Particle p) {
    enum {X, Y, Z};
    float d, f;
    d = yl2yc(c, p.r[Y]);
    f = d > 0 ? par.a : -par.a;
    return make_float3(f, 0, 0);
}


static __device__ float3 get_gp(Coords_v c, BForce_shear_v par, Particle p) {
    enum {X, Y, Z};
    float d, f;
    d = yl2yc(c, p.r[Y]);
    f = d * par.a;
    return make_float3(f, 0, 0);
}


static __device__ float3 get_gp(Coords_v c, BForce_rol_v par, Particle p) {
    enum {X, Y, Z};
    float x, y, lx, ly;
    float3 f;
    const float PI = 3.141592653589793;

    lx = xdomain(c);
    ly = ydomain(c);

    x = xl2xg(c, p.r[X]);
    y = yl2yg(c, p.r[Y]);

    x *= 2*PI / lx;
    y *= 2*PI / ly;

    f.x =  2*sin(x)*cos(y) * par.a;
    f.y = -2*cos(x)*sin(y) * par.a;
    f.z = 0;
    return f;
}


static __device__ float3 get_gp(Coords_v c, BForce_rad_v par, Particle p) {
    enum {X, Y, Z};
    float x, y, fact;
    float3 f;

    x = xl2xc(c, p.r[X]);
    y = yl2yc(c, p.r[Y]);

    fact = par.a / (x*x + y*y);

    f.x = fact * x;
    f.y = fact * y;
    f.z = 0;
    return f;
}


template <typename P>
__global__ void force(Coords_v c, P par, float mass, int n, const Particle *pp,      Force *ff) {
    enum {X, Y, Z};
    int i;
    Particle p;
    float3 gp;
    float *f;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    p = pp[i];
    f = ff[i].f;
    gp = get_gp(c, par, p);

    f[X] += mass * gp.x;
    f[Y] += mass * gp.y;
    f[Z] += mass * gp.z;
}



struct BForce_cste {
    float3 a;
};


struct BForce_dp {
    float a;
};


struct BForce_shear {
    float a;
};


struct BForce_rol {
    float a;
};


struct BForce_rad {
    float a;
};

enum {
    BODY_FORCE_NONE,
    BODY_FORCE_CSTE,
    BODY_FORCE_DP,
    BODY_FORCE_SHEAR,
    BODY_FORCE_ROL,
    BODY_FORCE_RAD
};

union BForceParam {
    BForce_cste cste;
    BForce_dp dp;
    BForce_shear shear;
    BForce_rol rol;
    BForce_rad rad;
};

struct BForce {
    BForceParam p;
    int type;
};
void bforce_ini(BForce **p) { EMALLOC(1, p); }
void bforce_fin(BForce *p)  { EFREE(p);      }

void bforce_set_none(     BForce *p) {
    p->type = BODY_FORCE_NONE;
}

void bforce_set_cste(float3 f,      BForce *p) {
    p->type = BODY_FORCE_CSTE;
    p->p.cste.a = f;
}

void bforce_set_dp(float a,      BForce *p) {
    p->type = BODY_FORCE_DP;
    p->p.dp.a = a;
}

void bforce_set_shear(float a,      BForce *p) {
    p->type = BODY_FORCE_SHEAR;
    p->p.shear.a = a;
}

void bforce_set_rol(float a,      BForce *p) {
    p->type = BODY_FORCE_ROL;
    p->p.rol.a = a;
}

void bforce_set_rad(float a,      BForce *p) {
    p->type = BODY_FORCE_RAD;
    p->p.rad.a = a;
}
void bforce_adjust(float3 f,      BForce *bforce) {
    int type;
    BForceParam *p;
    type = bforce->type;
    p    = &bforce->p;

    switch (type) {
    case BODY_FORCE_NONE:
        break;
    case BODY_FORCE_CSTE:
        p->cste.a = f;
        break;
    case BODY_FORCE_RAD:

        p->rad.a = f.x;
        break;
    case BODY_FORCE_DP:
    case BODY_FORCE_SHEAR:
    case BODY_FORCE_ROL:
    default:
        ERR("not implemented");
        break;
    };
}

static void get_view(BForce_cste bf, BForce_v *v) {
    v->type = BODY_FORCE_V_CSTE;
    v->p.cste.a = bf.a;
}

static void get_view(BForce_dp bf, BForce_v *v) {
    v->type = BODY_FORCE_V_DP;
    v->p.dp.a = bf.a;
}

static void get_view(BForce_shear bf, BForce_v *v) {
    v->type = BODY_FORCE_V_SHEAR;
    v->p.shear.a = bf.a;
}

static void get_view(BForce_rol bf, BForce_v *v) {
    v->type = BODY_FORCE_V_ROL;
    v->p.rol.a = bf.a;
}

static void get_view(BForce_rad bf, BForce_v *v) {
    v->type = BODY_FORCE_V_RAD;
    v->p.rad.a = bf.a;
}

static void bforce_get_view(const BForce *bforce,      BForce_v *view) {
    int type;
    BForceParam p;
    type = bforce->type;
    p    = bforce->p;

    switch (type) {
    case BODY_FORCE_NONE:
        view->type = BODY_FORCE_V_NONE;
        break;
    case BODY_FORCE_CSTE:
        get_view(p.cste,      view);
        break;
    case BODY_FORCE_DP:
        get_view(p.dp,      view);
        break;
    case BODY_FORCE_SHEAR:
        get_view(p.shear,      view);
        break;
    case BODY_FORCE_ROL:
        get_view(p.rol,      view);
        break;
    case BODY_FORCE_RAD:
        get_view(p.rad,      view);
        break;
    default:
        ERR("wrong type");
        break;
    };
}

void bforce_apply(const Coords *c, float mass, const BForce *bf, int n, const Particle *pp,      Force *ff) {
    int type;
    BForce_v view;
    BForceParam_v p;
    Coords_v coordsv;

    coords_get_view(c, &coordsv);

    view.type = BODY_FORCE_V_NONE;
    bforce_get_view(bf,      &view);

    type = view.type;
    p    = view.p;

    switch (type) {
    case BODY_FORCE_V_NONE:
        break;
    case BODY_FORCE_V_CSTE:
        KL(force, (k_cnf(n)), (coordsv, p.cste, mass, n, pp,      ff));
        break;
    case BODY_FORCE_V_DP:
        KL(force, (k_cnf(n)), (coordsv, p.dp, mass, n, pp,      ff));
        break;
    case BODY_FORCE_V_SHEAR:
        KL(force, (k_cnf(n)), (coordsv, p.shear, mass, n, pp,      ff));
        break;
    case BODY_FORCE_V_ROL:
        KL(force, (k_cnf(n)), (coordsv, p.rol, mass, n, pp,      ff));
        break;
    case BODY_FORCE_V_RAD:
        KL(force, (k_cnf(n)), (coordsv, p.rad, mass, n, pp,      ff));
        break;
    default:
        ERR("wrong type <%d>", type);
        break;
    };
}
