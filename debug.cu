#include "macros.cuh"
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "constants.cuh"
#include "types.cuh"
#include "device.cuh"
#include "launch.cuh"

namespace cc { void check(int code, const char *file, int line); }
void emalloc(size_t, void **);
void efree(void *);
void error_stack_push(const char *file, int line);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *file, int line, const char *fmt, ...);
#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)

enum { DBG_POS, DBG_POS_SOFT, DBG_VEL, DBG_FORCES, DBG_COLORS, DBG_CLIST, DBG_NKIND_ };
struct Config; struct Coords; struct Dbg;
void conf_lookup_bool(const Config *, const char *, int *);
int3 subdomain(const Coords *);
void coord_stamp(const Coords *, char *);
void txt_write_pp(long, const Particle *, const char *);
void txt_write_pp_ff(long, const Particle *, const Force *, const char *);
#define ERRLIST(_) _(NONE),                     \
        _(INVALID),                             \
        _(INF_VAL),                             \
        _(NAN_VAL)

#define make_str(s) #s
#define make_enum(s) ERR_##s

enum {
    ERRLIST(make_enum),
    NERRORS
};

static const char *err_str[NERRORS] = {ERRLIST(make_str)};

#undef ERRLIST
#undef make_str
#undef make_enum

typedef int err_type;

static __device__ err_type dbg_error;

static void err_ini() {
    err_type e = ERR_NONE;
    void *devPtr;
    CC(d::GetSymbolAddress(&devPtr, &dbg_error));
    CC(d::MemcpyToSymbol(&dbg_error, &e, sizeof(err_type)));
}

static err_type err_get() {
    err_type e;
    CC(d::MemcpyFromSymbol(&e, &dbg_error, sizeof(err_type)));
    return e;
}

static const char* get_err_str(err_type e) {return err_str[e];}

static __device__ void dbg_report(err_type e) {
    if (e != ERR_NONE) atomicExch(&dbg_error, e);
}

static __device__ err_type dbg_check_float(float a) {
    if (isnan(a)) return ERR_NAN_VAL;
    if (isinf(a)) return ERR_INF_VAL;
    return ERR_NONE;
}

static __device__ err_type dbg_check_float3(const float a[3]) {
    enum {X, Y, Z};
    err_type e;
#define check_ret(A) if ((e = dbg_check_float(A)) != ERR_NONE) return e
    check_ret(a[X]);
    check_ret(a[Y]);
    check_ret(a[Z]);
#undef check_ret
    return ERR_NONE;
}
static __device__ bool dbg_valid_pos(float x, int L) {
    if (x < -0.5*L || x > 0.5*L)
        return false;
    return true;
}

static __device__ bool dbg_valid_unpacked_pos(int3 L, float x, float y, float z) {
    bool ok = true;
    ok &= dbg_valid_pos(x, L.x);
    ok &= dbg_valid_pos(y, L.y);
    ok &= dbg_valid_pos(z, L.z);

    return ok;
}

static __device__ err_type dbg_valid_pos(int3 L, const Particle p, bool verbose) {
    enum {X, Y, Z};
    float x, y, z;
    err_type e;
    x  = p.r[X];  y = p.r[Y];  z = p.r[Z];
    e = dbg_check_float3(p.r);
    if (e != ERR_NONE) return e;
    if (dbg_valid_unpacked_pos(L, x, y, z)) e = ERR_NONE;
    else                                e = ERR_INVALID;

    if (verbose && e != ERR_NONE)
        printf("DBG: pos pu: p = [%g, %g %g], [%g, %g, %g]\n",
               p.r[X], p.r[Y], p.r[Z], p.v[X], p.v[Y], p.v[Z]);

    return e;
}

__global__ void dbg_check_pos_kernel(int3 L, const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p;
    if (i >= n) return;
    p = pp[i];
    err_type e = dbg_valid_pos(L, p, verbose);
    dbg_report(e);
}

static __device__ bool dbg_valid_unpacked_pos_pu(int3 L, float x, float y, float z) {
    bool ok = true;
    ok &= dbg_valid_pos(x, L.x + 3);
    ok &= dbg_valid_pos(y, L.y + 3);
    ok &= dbg_valid_pos(z, L.z + 3);

    return ok;
}

static __device__ err_type dbg_valid_pos_pu(int3 L, const Particle p, bool verbose) {
    enum {X, Y, Z};
    float x, y, z;
    err_type e;
    x  = p.r[X];  y = p.r[Y];  z = p.r[Z];
    e = dbg_check_float3(p.r);
    if (e != ERR_NONE) return e;
    if (dbg_valid_unpacked_pos_pu(L, x, y, z)) e = ERR_NONE;
    else                                   e = ERR_INVALID;

    if (verbose && e != ERR_NONE)
        printf("DBG: pos pu: p = [%g, %g %g], [%g, %g, %g]\n",
               p.r[X], p.r[Y], p.r[Z], p.v[X], p.v[Y], p.v[Z]);

    return e;
}

__global__ void dbg_check_pos_pu_kernel(int3 L, const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p;
    if (i >= n) return;
    p = pp[i];
    err_type e = dbg_valid_pos_pu(L, p, verbose);
    dbg_report(e);
}
static __device__ bool dbg_valid_vel(float dt, float v, int L) {
    float dx = fabs(v * dt);
    if (dx > L/2)
        return false;
    return true;
}

static __device__ bool dbg_valid_vel3(float dt, int3 L, float vx, float vy, float vz) {
    bool ok = true;
    ok &= dbg_valid_vel(dt, vx, L.x);
    ok &= dbg_valid_vel(dt, vy, L.y);
    ok &= dbg_valid_vel(dt, vz, L.z);

    return ok;
}

static __device__ err_type dbg_valid_vv(float dt, int3 L, const Particle p, bool verbose) {
    enum {X, Y, Z};
    err_type e;
    const float *v = p.v;
    e = dbg_check_float3(v);
    if (e != ERR_NONE) return e;
    if ( dbg_valid_vel3(dt, L, v[X], v[Y], v[Z])) e = ERR_NONE;
    else                                       e = ERR_INVALID;

    if (verbose && e != ERR_NONE)
        printf("DBG: vel: p = [%g, %g %g], [%g, %g, %g]\n",
               p.r[X], p.r[Y], p.r[Z], p.v[X], p.v[Y], p.v[Z]);

    return e;
}

__global__ void dbg_check_vv_kernel(float dt, int3 L, const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p;
    if (i >= n) return;
    p = pp[i];
    err_type e = dbg_valid_vv(dt, L, p, verbose);
    dbg_report(e);
}
static __device__ bool dbg_valid_acc(float dt, float a, int L, bool verbose) {
    float dx = fabs(a * dt * dt);
    if (dx > 0.5*L) {
        if (verbose) printf("DBG: a = %g (L = %d)\n", a, L);
        return false;
    }
    return true;
}

static __device__ bool dbg_valid_unpacked_f(float dt, int3 L, float fx, float fy, float fz, bool verbose) {
    bool ok = true;
    ok &= dbg_valid_acc(dt, fx, L.x, verbose);
    ok &= dbg_valid_acc(dt, fy, L.y, verbose);
    ok &= dbg_valid_acc(dt, fz, L.z, verbose);

    return ok;
}

static __device__ err_type dbg_valid_f(float dt, int3 L, const Force *f, bool verbose) {
    enum {X, Y, Z};
    float fx, fy, fz;
    err_type e;
    fx = f->f[X]; fy = f->f[Y]; fz = f->f[Z];
    e = dbg_check_float3(f->f);
    if (e != ERR_NONE) return e;
    if (dbg_valid_unpacked_f(dt, L, fx, fy, fz, verbose)) e = ERR_NONE;
    else                                               e = ERR_INVALID;
    return e;
}

__global__ void dbg_check_ff_kernel(float dt, int3 L, const Force *ff, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    err_type e = dbg_valid_f(dt, L, ff + i, verbose);
    dbg_report(e);
}
static __device__ err_type dbg_valid_color(int c, bool verbose) {
    if (c != BLUE_COLOR && c != RED_COLOR) {
        if (verbose) printf("DBG: color = %d\n", c);
        return ERR_INVALID;
    }
    return ERR_NONE;
}

__global__ void dbg_check_cc_kernel(const int *cc, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    err_type e = dbg_valid_color(cc[i], verbose);
    dbg_report(e);
}
static __device__ int3 dbg_cell_coords(int3 L, int cid) {
    int3 c;
    c.x = cid % L.x;
    c.z = cid / (L.y * L.x);
    c.y = (cid - L.y * L.x * c.z) / L.x;
    return c;
}

static __device__ bool dbg_at_coord(float x, int c, int L, bool verbose, const char *dir) {
    x += L/2;
    if ((x < c) || (x > c + 1)) {
        if (verbose) printf("Particle out of its cell : %s = %f, c = %d, L = %d\n", dir, x, c, L);
        return false;
    }
    return true;
}

static __device__ bool dbg_at_coords(const float r[3], int3 coords, int3 L, bool verbose) {
    enum {X, Y, Z};
    bool ok = true;
    ok &= dbg_at_coord(r[X], coords.x, L.x, verbose, "x");
    ok &= dbg_at_coord(r[Y], coords.y, L.y, verbose, "y");
    ok &= dbg_at_coord(r[Z], coords.z, L.z, verbose, "z");
    return ok;
}

static __device__ err_type dbg_valid_particle(int3 L, int cid, const Particle *p, bool verbose) {
    err_type e;
    e = dbg_check_float3(p->r); if (e != ERR_NONE) return e;
    e = dbg_check_float3(p->v); if (e != ERR_NONE) return e;

    int3 coords = dbg_cell_coords(L, cid);
    bool valid = dbg_at_coords(p->r, coords, L, verbose);
    if (!valid) e = ERR_INVALID;

    return e;
}

static __device__ err_type dbg_valid_cell(int3 L, int cid, int s, int c, const Particle *pp, bool verbose) {
    err_type e = ERR_NONE;
    int i;
    Particle p;

    for (i = 0; i < c; ++i) {
        p = pp[s + i];
        e = dbg_valid_particle(L, cid, &p, verbose);
        if (e != ERR_NONE) return e;
    }
    return e;
}

__global__ void dbg_check_clist_kernel(int3 ncells, const int *ss, const int *cc, const Particle *pp, bool verbose = true) {
    int i, c, s;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= ncells.x * ncells.y * ncells.z) return;

    s = ss[i];
    c = cc[i];

    err_type e = dbg_valid_cell(ncells, i, s, c, pp, verbose);
    dbg_report(e);
}

struct Dbg {
    int state[DBG_NKIND_];
    bool verbose;
    bool dump;
};
void dbg_ini(Dbg **dbg) {
    Dbg *d;
    int i;
    EMALLOC(1, dbg);
    d = *dbg;
    for (i = 0; i < DBG_NKIND_; ++i) d->state[i] = 0;
    d->verbose = false;
    d->dump = false;
}
void dbg_fin(Dbg *dbg) {
    EFREE(dbg);
}

static void set(int kind, int val, Dbg *dbg) {
    if (kind < 0 || kind >= DBG_NKIND_)
        ERR("Unrecognised kind of id <%d>", kind);
    dbg->state[kind] = val;
}

static void set_conf(const Config *cfg, const char *desc, int kind, Dbg *dbg) {
    int enabled = 0;
    UC(conf_lookup_bool(cfg, desc, &enabled));
    set(kind, enabled, dbg);
}

void dbg_enable(int kind, Dbg *dbg) {
    UC(set(kind, 1, dbg));
}

void dbg_disable(int kind, Dbg *dbg) {
    UC(set(kind, 0, dbg));
}

void dbg_set_verbose(bool active, Dbg *dbg) {dbg->verbose = active;}
void dbg_set_dump(bool active, Dbg *dbg) {dbg->dump = active;}

void dbg_set_conf(const Config *cfg, Dbg *dbg) {
    set_conf(cfg, "dbg.pos",      DBG_POS,      dbg);
    set_conf(cfg, "dbg.pos_soft", DBG_POS_SOFT, dbg);
    set_conf(cfg, "dbg.vel",      DBG_VEL,      dbg);
    set_conf(cfg, "dbg.forces",   DBG_FORCES,   dbg);
    set_conf(cfg, "dbg.colors",   DBG_COLORS,   dbg);
    set_conf(cfg, "dbg.clist",    DBG_CLIST,    dbg);

    int v;
    UC(conf_lookup_bool(cfg, "dbg.verbose", &v));
    dbg_set_verbose(v, dbg);
    UC(conf_lookup_bool(cfg, "dbg.dump", &v));
    dbg_set_dump(v, dbg);
}

static int check(const Dbg *dbg, int kind) {return dbg->state[kind];}

static bool error() {
    err_type e;
    UC(e = err_get());
    return e != ERR_NONE;
}

static void gen_name(const Coords *c, const char *base,      char *name) {
    int r;
    char stamp[FILENAME_MAX];
    coord_stamp(c,      stamp);
    r = sprintf(name, "dbg.%s.%s.txt", base, stamp);
    if (r < 0) ERR("sprintf failed");
}

static void dump_pp(const Coords *c, const char *base, int n, const Particle *dev) {
    Particle *hst;
    char name[FILENAME_MAX];
    EMALLOC(n, &hst);
    cD2H(hst, dev, n);
    UC(gen_name(c, base, name));
    UC(txt_write_pp(n, hst, name));
    EFREE(hst);
}

static void dump_pp_ff(const Coords *c, const char *base, int n, const Particle *ppdev, const Force *ffdev) {
    Particle *pphst;
    Force *ffhst;
    char name[FILENAME_MAX];

    EMALLOC(n, &pphst);
    EMALLOC(n, &ffhst);

    cD2H(pphst, ppdev, n);
    cD2H(ffhst, ffdev, n);

    UC(gen_name(c, base, name));

    UC(txt_write_pp_ff(n, pphst, ffhst, name));

    EFREE(pphst);
    EFREE(ffhst);
}

static void print() {
    err_type e;
    UC(e = err_get());
    ERR("DBG: error: %s", get_err_str(e));
}

void dbg_check_pos(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp) {
    int3 L = subdomain(c);
    if (!check(dbg, DBG_POS))
        return;
    UC(err_ini());
    KL(dbg_check_pos_kernel, (k_cnf(n)), (L, pp, n));
    if (error()) {
        if (dbg->dump)
            dump_pp(c, base, n, pp);
        UC(print());
    }
}

void dbg_check_pos_soft(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp) {
    int3 L = subdomain(c);
    if (!check(dbg, DBG_POS_SOFT))
        return;
    UC(err_ini());
    KL(dbg_check_pos_pu_kernel, (k_cnf(n)), (L, pp, n));
    if (error()) {
        if (dbg->dump)
            dump_pp(c, base, n, pp);
        UC(print());
    }
}

void dbg_check_vel(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp) {
    int3 L = subdomain(c);

    if (!check(dbg, DBG_VEL))
        return;
    UC(err_ini());
    KL(dbg_check_vv_kernel, (k_cnf(n)), (dt, L, pp, n));
    if (error()) {
        if (dbg->dump)
            dump_pp(c, base, n, pp);
        UC(print());
    }
}

void dbg_check_forces(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp, const Force *ff) {
    int3 L = subdomain(c);

    if (!check(dbg, DBG_FORCES))
        return;
    UC(err_ini());
    KL(dbg_check_ff_kernel, (k_cnf(n)), (dt, L, ff, n));
    if (error()) {
        if (dbg->dump)
            dump_pp_ff(c, base, n, pp, ff);
        UC(print());
    }
}

void dbg_check_colors(const Coords*, const Dbg *dbg, int n, const int *cc) {
    if (!check(dbg, DBG_COLORS))
        return;
    UC(err_ini());
    KL(dbg_check_cc_kernel, (k_cnf(n)), (cc, n));
    if (error()) {
        UC(print());
    }
}

void dbg_check_clist(const Coords*, const Dbg *dbg, int3 L, const int *starts, const int *counts, const Particle *pp) {
    if (!check(dbg, DBG_CLIST))
        return;
    int n = L.x * L.y * L.z;
    UC(err_ini());
    KL(dbg_check_clist_kernel, (k_cnf(n)), (L, starts, counts, pp));
    if (error()) {
        UC(print());
    }
}
