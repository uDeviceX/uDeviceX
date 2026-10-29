#include "macros.cuh"
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>
#include <mpi.h>
#include <vector_types.h>
#include "config.cuh"
#include "constants.cuh"
#include "types.cuh"
#include "device.cuh"
#include "launch.cuh"

namespace cc { void check(int code, const char *file, int line); }
namespace mpicheck { void check(int code, const char *file, int line); }
namespace m {
int Comm_rank(MPI_Comm, int *);
int Comm_dup(MPI_Comm, MPI_Comm *);
int Comm_free(MPI_Comm *);
int Allreduce(const void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
}
void emalloc(size_t, void **);
void efree(void *);
void efopen(const char *, const char *, FILE **);
void efclose(FILE *);
void msg_print(const char *, ...);
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
bool same_str(const char *, const char *);
#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)
#define MC(ans) do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <stdio.h>
#include <mpi.h>
#include <vector_types.h>

#ifndef UDX_CONTROL_UTILS_COMMON
#define UDX_CONTROL_UTILS_COMMON




#endif
#ifndef UDX_CONTROL_ERROR_COMMON
#define UDX_CONTROL_ERROR_COMMON

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

struct PidVCont;
struct Coords;
struct Config;
struct float3;
struct int3;
struct Particle;

void vcont_ini(MPI_Comm comm, int3 L,      PidVCont **c);
void vcont_fin(     PidVCont *cont);

void vcont_set_params(float factor, float Kp, float Ki, float Kd,      PidVCont *c);
void vcont_set_target(float3 vtarget,      PidVCont *c);
void vcont_set_cart(     PidVCont *c);
void vcont_set_radial(     PidVCont *c);

void vcont_set_conf(const Config *cfg,      PidVCont *c);

void vcont_strt_dump(MPI_Comm, const char *base, int id, const PidVCont*);
void vcont_strt_read(          const char *base, int id, PidVCont*);

void   vcont_sample(const Coords *coords, int n, const Particle *pp, const int *starts, const int *counts,      PidVCont *c);
float3 vcont_adjustF(     PidVCont *c);
void   vcont_log(const PidVCont *c);

void vcont_set_conf(const Config *cfg,      PidVCont *c) {
    const char *type;
    float3 U;
    float factor, Kp, Ki, Kd;

    UC(conf_lookup_string(cfg, "vcon.type", &type));
    UC(conf_lookup_float3(cfg, "vcon.U", &U));
    UC(conf_lookup_float(cfg, "vcon.factor", &factor));
    UC(conf_lookup_float(cfg, "vcon.Kp", &Kp));
    UC(conf_lookup_float(cfg, "vcon.Ki", &Ki));
    UC(conf_lookup_float(cfg, "vcon.Kd", &Kd));

    UC(vcont_set_params(factor, Kp, Ki, Kd,      c));
    UC(vcont_set_target(U,      c));

    if      (same_str(type, "cart"))
        UC(vcont_set_cart(     c));
    else if (same_str(type, "rad"))
        UC(vcont_set_radial(     c));
    else
        ERR("Unrecognised type <%s>", type);

}



#include <stdio.h>
#include <mpi.h>
#include <assert.h>




#ifndef UDX_CONTROL_ERROR_COMMON
#define UDX_CONTROL_ERROR_COMMON

#endif


#ifndef UDX_CONTROL_TYPE_COMMON
#define UDX_CONTROL_TYPE_COMMON
#endif





namespace scheme_move_dev {
static __device__ void update0(float dt, float m, const float *f, float *r, float *v) {
    enum {X, Y, Z};
    v[X] += f[X]/m*dt;
    v[Y] += f[Y]/m*dt;
    v[Z] += f[Z]/m*dt;

    r[X] += v[X]*dt;
    r[Y] += v[Y]*dt;
    r[Z] += v[Z]*dt;
}

__global__ void update(float dt, float mass, Particle *pp, const Force *ff, int n) {
    float *r, *v;
    const float *f;
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    r = pp[pid].r;
    v = pp[pid].v;
    f = ff[pid].f;
    update0(dt, mass, f, r, v);
}

__global__ void clear_vel(int n, Particle *pp) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    for (int c = 0; c < 3; ++c) pp[pid].v[c] = 0;
}
}

void scheme_move_apply(float dt, float mass, int n, const Force *ff, Particle *pp) {
    KL(scheme_move_dev::update, (k_cnf(n)), (dt, mass, pp, ff, n));
}

void scheme_move_clear_vel(int n, Particle *pp) {
    KL(scheme_move_dev::clear_vel, (k_cnf(n)), (n, pp));
}









#ifndef UDX_CONTROL_D_API_COMMON
#define UDX_CONTROL_D_API_COMMON
#endif







#ifndef UDX_CONTROL_KL_COMMON
#define UDX_CONTROL_KL_COMMON
#endif



#include "vecmath.cuh"
#ifndef UDX_CONTROL_COORDS_COMMON
#define UDX_CONTROL_COORDS_COMMON
struct Coords_v {
    int xc, yc, zc;
    int xd, yd, zd;
    int Lx, Ly, Lz;
};
#endif

static __device__ float xl2xc(const Coords_v c, float xl) {
    return xl - 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
}

static __device__ float yl2yc(const Coords_v c, float yl) {
    return yl - 0.5f * c.Ly * (c.yd - 2 * c.yc - 1);
}


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

#define UDX_CONTROL_SHFL_COMMON
#include "warp.cuh"

struct Coords;
struct Particle;
struct Rigid;

enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};

typedef void (*StreamWriter)(const void*, FILE*);
typedef void (*StreamReader)(FILE*, void*);

void restart_write_pp(MPI_Comm, const char *base, const char *code, int id, long n, const Particle *pp);
void restart_write_ii(MPI_Comm, const char *base, const char *code, int id, long n, const int *ii);
void restart_write_ss(MPI_Comm, const char *base, const char *code, int id, long n, const Rigid *ss);
void restart_write_stream_one_node(MPI_Comm, const char *base, const char *code, int id, const void *data, StreamWriter);

void restart_read_pp(MPI_Comm, const char *base, const char *code, int id, int *n, Particle *pp);
void restart_read_ii(MPI_Comm, const char *base, const char *code, int id, int *n, int *ii);
void restart_read_ss(MPI_Comm, const char *base, const char *code, int id, int *n, Rigid *ss);
void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader, void *data);



enum {
    VCONT_TYPE_NONE,
    VCONT_TYPE_CART,
    VCONT_TYPE_RAD
};


struct TCart {};
struct TRad {};


union Trans {
    TCart cart;
    TRad  rad;
};


struct Sampler {
    int3 L;
    long nsamples;

    float3 *gridvel;
    int    *gridnum;
    float3 *totvel;
    int    *totnum;
    float3 *dtotvel;
    int    *dtotnum;
};

struct State {
    float3 cur;
    float3 olde, sume;
    float3 f;
};

struct Param {
    float3 target;
    float Kp, Ki, Kd;
};


struct PidVCont {
    Param param;
    State state;
    Sampler sampler;

    MPI_Comm comm;

    FILE *fdump;

    Trans trans;
    int type;
};

__global__ void vcont_reduce_by_warp_kernel(const float3 *gvel, const int *gnum, const uint ncells,      float3 *vel, int *num) {
    assert(blockDim.x == 32);
    int i, c;
    float3 v = make_float3(0, 0, 0);
    int n = 0;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    c = blockIdx.x;

    if (i < ncells) {
        v = gvel[i];
        n = gnum[i];
    }

    v = warpReduceSum(v);
    n = warpReduceSum(n);

    if ((threadIdx.x % warpSize) == 0) {
        vel[c] = v;
        num[c] = n;
    }
}
static __device__ float3 transform(Coords_v, TCart, const Particle p) {
    enum {X, Y, Z};
    float3 u;
    u.x = p.v[X];
    u.y = p.v[Y];
    u.z = p.v[Z];
    return u;
}
static __device__ float3 transform(Coords_v c, TRad, const Particle p) {
    enum {X, Y, Z};
    float3 u;
    float x, y, r, rinv;
    float cost, sint;
    x = xl2xc(c, p.r[X]);
    y = yl2yc(c, p.r[Y]);

    r = sqrt(x*x + y*y);
    rinv = 1 / r;
    cost = rinv * x;
    sint = rinv * y;

    u.x = r * (  cost * p.v[X] + sint * p.v[Y]);
    u.y = r * (- sint * p.v[X] + cost * p.v[Y]);
    u.z = p.v[Z];
    return u;
}
static __device__ bool valid_cell(int3 L, int3 c) {
    return c.x < L.x && c.y < L.y && c.z < L.z;
}

static __device__ int get_cid(int3 L, int3 c) {
    return c.x + L.x * (c.y + L.y * c.z);
}

template<typename Trans>
__global__ void vcont_sample_kernel(Coords_v coords, Trans t, int3 L, const int *cellsstart, const int *cellscount, const Particle *pp,
                            float3 *gridv, int *gridn) {
    Particle p;
    float3 u;
    int pid, cid, num;
    const int3 c = make_int3(threadIdx.x + blockIdx.x * blockDim.x,
                             threadIdx.y + blockIdx.y * blockDim.y,
                             threadIdx.z + blockIdx.z * blockDim.z);

    if (valid_cell(L, c)) {
        cid = get_cid(L, c);
        num = cellscount[cid];

        for (pid = cellsstart[cid]; pid < cellsstart[cid] + num; pid++) {
            p = pp[pid];
            u = transform(coords, t, p);

            gridv[cid].x += u.x;
            gridv[cid].y += u.y;
            gridv[cid].z += u.z;
        }
        gridn[cid] += num;
    }
}

#ifndef UDX_CONTROL_WARPSIZE_COMMON
#define UDX_CONTROL_WARPSIZE_COMMON
enum {WARPSIZE = 32};
#endif

static void ini_dump(int rank,      FILE **f) {
    *f = NULL;
    if (rank) return;
    UC(efopen(DUMP_BASE "/vcont.txt", "w",      f));
    fprintf(*f, "#vx vy vz fx fy fz\n");
    msg_print("Velocity controller: dump to " DUMP_BASE "/vcont.txt");
}

static void fin_dump(FILE *f) {
    if (f) UC(efclose(f));
}

static void reini_sampler(     Sampler *s) {
    int3 L = s->L;
    int ncells = L.x * L.y * L.z;

    if (ncells) {
        CC(d::MemsetAsync(s->gridvel, 0, ncells * sizeof(float3)));
        CC(d::MemsetAsync(s->gridnum, 0, ncells * sizeof(int)));
    }

    s->nsamples = 0;
}

static void ini_sampler(int3 L, Sampler *s) {
    int ncells, nchunks;
    s->L = L;
    ncells = L.x * L.y * L.z;

    Dalloc(&s->gridvel, ncells);
    Dalloc(&s->gridnum, ncells);

    nchunks = ceiln(ncells, WARPSIZE);

    CC(d::alloc_pinned((void **) &s->totvel, nchunks * sizeof(float3)));
    CC(d::alloc_pinned((void **) &s->totnum, nchunks * sizeof(int)));
    CC(d::HostGetDevicePointer((void **) &s->dtotvel, s->totvel, 0));
    CC(d::HostGetDevicePointer((void **) &s->dtotnum, s->totnum, 0));
}

static void ini_state(State *s) {
    s->f = s->sume = s->cur = make_float3(0, 0, 0);
}

static void ini(MPI_Comm comm, int3 L,      PidVCont *c) {
    int rank;

    UC(ini_sampler(L, &c->sampler));
    UC(reini_sampler(&c->sampler));
    UC(ini_state(&c->state));

    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_dup(comm, &c->comm));

    UC(ini_dump(rank,      &c->fdump));

    c->type = VCONT_TYPE_NONE;
}

void vcont_ini(MPI_Comm comm, int3 L,      PidVCont **c) {
    PidVCont *vc;
    EMALLOC(1, c);
    vc = *c;
    UC(ini(comm, L,      vc));
}

static void fin_sampler(Sampler *s) {
    Dfree(s->gridvel);
    Dfree(s->gridnum);
    CC(d::FreeHost(s->totvel));
    CC(d::FreeHost(s->totnum));
}

void vcont_fin(     PidVCont *c) {
    UC(fin_sampler(&c->sampler));
    MC(m::Comm_free(&c->comm));
    UC(fin_dump(c->fdump));
    EFREE(c);
}

static void set_params(float f, float Kp, float Ki, float Kd, Param *p) {
    p->Kp = f * Kp;
    p->Ki = f * Ki;
    p->Kd = f * Kd;
}

void vcont_set_params(float factor, float Kp, float Ki, float Kd,      PidVCont *c) {
    set_params(factor, Kp, Ki, Kd, &c->param);
}

void vcont_set_target(float3 vtarget,      PidVCont *c) {
    c->param.target = vtarget;
    c->state.olde   = vtarget;
}

void vcont_set_cart(     PidVCont *cont) {
    cont->type = VCONT_TYPE_CART;
}

void vcont_set_radial(     PidVCont *cont) {
    cont->type = VCONT_TYPE_RAD;
}

template <typename Trans>
static void sample(Trans t, const Coords *coords, const Particle *pp, const int *starts, const int *counts,      Sampler *s) {
    int3 L = s->L;
    Coords_v coordsv;

    coords_get_view(coords, &coordsv);

    dim3 block(8, 8, 1);
    dim3 grid(ceiln(L.x, block.x),
              ceiln(L.y, block.y),
              ceiln(L.z, block.z));

    KL(vcont_sample_kernel, (grid, block), (coordsv, t, L, starts, counts, pp,      s->gridvel, s->gridnum));
    s->nsamples++;
}

void vcont_sample(const Coords *coords, int, const Particle *pp, const int *starts, const int *counts,      PidVCont *c) {
    switch (c->type) {
    case VCONT_TYPE_NONE:
        break;
    case VCONT_TYPE_CART:
        sample(c->trans.cart, coords, pp, starts, counts,      &c->sampler);
        break;
    case VCONT_TYPE_RAD:
        sample(c->trans.cart, coords, pp, starts, counts,      &c->sampler);
        break;
    default:
        ERR("Unknown type");
        break;
    };
}

static void reduce_local(Sampler *s, double3 *v, long *n) {
    int3 L = s->L;
    int ncells, nchunks, i;
    ncells = L.x * L.y * L.z;
    nchunks = ceiln(ncells, WARPSIZE);

    KL(vcont_reduce_by_warp_kernel, (nchunks, WARPSIZE), (s->gridvel, s->gridnum, ncells,      s->dtotvel, s->dtotnum));
    dSync();

    *v = make_double3(0, 0, 0);
    *n = 0;

    for (i = 0; i < nchunks; ++i) {
        add(s->totvel + i,      v);
        *n += s->totnum[i];
    }
}

static float3 average_samples(MPI_Comm comm, Sampler *s) {
    long ncur;
    double3 vcur;
    double fac;

    UC(reduce_local(s, &vcur, &ncur));

    MC(m::Allreduce(MPI_IN_PLACE, &vcur.x, 3, MPI_DOUBLE, MPI_SUM, comm));
    MC(m::Allreduce(MPI_IN_PLACE, &ncur,   1, MPI_LONG,   MPI_SUM, comm));

    fac = ncur ? (1.0 / ncur) : 1.0;
    scal(fac,      &vcur);

    return make_float3(vcur.x, vcur.y, vcur.z);
}

static float3 update_state(const Param *p, float3 vcur, State *s) {
    float3 e, de, f;

    s->cur = vcur;
    diff(&p->target, &vcur,      &e);
    diff(&e, &s->olde,      &de);
    add(&e,      &s->sume);
    s->olde = e;

    f = make_float3(0, 0, 0);

    axpy(p->Kp, &e,            &f);
    axpy(p->Ki, &s->sume,      &f);
    axpy(p->Kd, &de,           &f);
    s->f = f;
    return f;
}

float3 vcont_adjustF(     PidVCont *c) {
    float3 vcur;
    vcur = average_samples(c->comm, &c->sampler);
    UC(reini_sampler(&c->sampler));

    return update_state(&c->param, vcur, &c->state);
}

void vcont_log(const PidVCont *c) {
    if (c->fdump == NULL) return;
    float3 v = c->state.cur;
    float3 f = c->state.f;
    fprintf(c->fdump, "%.3e %.3e %.3e %.3e %.3e %.3e\n",
            v.x, v.y, v.z, f.x, f.y, f.z);
    fflush(c->fdump);
}
#define CODE "vcon"
#define PAT "%g %g %g\n"

static void writef3(float3 v, FILE *f) {
    fprintf(f, PAT, v.x, v.y, v.z);
}

static void write_state(const State *s, FILE *f) {
    writef3(s->cur, f);
    writef3(s->olde, f);
    writef3(s->sume, f);
    writef3(s->f, f);
}

static void stream_write_state(const void *data, FILE *f) {
    const PidVCont *c = (const PidVCont*) data;
    write_state(&c->state, f);
}

void vcont_strt_dump(MPI_Comm comm, const char *base, int id, const PidVCont *c) {
    StreamWriter sw = stream_write_state;
    restart_write_stream_one_node(comm, base, CODE, id, (void*) c, sw);
}

static void readf3(FILE *f, float3 *v) {
    if (3 != fscanf(f, PAT, &v->x, &v->y, &v->z))
        ERR("wrong format\n");
}

static void read_state(FILE *f, State *s) {
    UC(readf3(f, &s->cur));
    UC(readf3(f, &s->olde));
    UC(readf3(f, &s->sume));
    UC(readf3(f, &s->f));
}

static void stream_read_state(FILE *f, void *data) {
    PidVCont *c = (PidVCont*) data;
    UC(read_state(f, &c->state));
}

void vcont_strt_read(const char *base, int id, PidVCont *c) {
    StreamReader sr = stream_read_state;
    restart_read_stream_one_node(base, CODE, id, sr, (void*) c);
}

#undef CODE
#undef PAT

