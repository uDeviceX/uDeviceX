#include "memory.cuh"
#define DEV_CUDA
#define CC_RELEASE
#include "macros.cuh"
#include "config.cuh"
#include "constants.cuh"
#include "types.cuh"
#include <mpi.h>
#include <vector_types.h>
#include <stdio.h>
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
void msg_print(const char *, ...);
#ifndef WALLOBJ_ARRAY_TYPES
#define WALLOBJ_ARRAY_TYPES
struct PaArray {
    bool colors;
    const float *pp;
    const int *cc;
};

struct PaArray_v;
struct PaCArray_v;

struct Particle;

void parray_push_pp(const Particle *pp, PaArray *a);
void parray_push_cc(const int *cc, PaArray *a);

bool parray_is_colored(const PaArray *a);

void parray_get_view(const PaArray *a, PaArray_v *v);
void parray_get_view(const PaArray *a, PaCArray_v *v);
#endif
#ifndef UDX_SIM_OBJECTS_FARRAY_COMMON
#define UDX_SIM_OBJECTS_FARRAY_COMMON
struct FoArray {
    bool stress;
    float *ff;
    float *ss;
};

struct FoArray_v;
struct FoSArray_v;

struct Force;

void farray_push_ff(Force *ff, FoArray *a);
void farray_push_ss(float *ss, FoArray *a);

void farray_clear(FoArray *);

bool farray_has_stress(const FoArray *a);

void farray_get_view(const FoArray *a, FoArray_v *v);
void farray_get_view(const FoArray *a, FoSArray_v *v);
#endif
#ifndef UDX_SIM_OBJECTS_PFARRAYS_COMMON
#define UDX_SIM_OBJECTS_PFARRAYS_COMMON
struct PFarray {
    PaArray p;
    FoArray f;
    long n;
};

struct PFarrays;

struct PaArray;
struct FoArray;

void pfarrays_ini(PFarrays**);
void pfarrays_fin(PFarrays*);

void pfarrays_clear(PFarrays*);
void pfarrays_push(PFarrays*, long n, PaArray, FoArray);
void pfarrays_push(PFarrays*, PFarray);
int  pfarrays_size(const PFarrays*);
void pfarrays_get(int i, const PFarrays*, long *n, PaArray*, FoArray*);
void pfarrays_get(int i, const PFarrays*, PFarray*);
#endif
namespace m { int Barrier(MPI_Comm); }
namespace mpicheck { void check(int, const char *, int); }
#define MC(ans) do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)
struct Sdf;
struct Sdf_v;
struct WvelStep;
struct Coords;
struct Particle;
struct int3;

void sdf_ini(int3 L, Sdf**);
void sdf_fin(Sdf*);

void sdf_gen(const Coords*, MPI_Comm cart, bool dump, Sdf*);
void sdf_get_view(const Sdf*,      Sdf_v*);

void sdf_bulk_wall(const Sdf*,        int *s_n, Particle *s_pp,       int *w_n, Particle *w_pp);
int  sdf_who_stays(const Sdf*, int n, const Particle*, int nc, int nv,      int *stay);
double sdf_compute_volume(MPI_Comm, int3 L, const Sdf*, long nsamples);

void sdf_bounce(float dt, const WvelStep*, const Coords*, const Sdf*, int n,      Particle*);
struct Wvel;
struct WvelStep;

struct WvelCste_v;
struct WvelShear_v;
struct WvelHS_v;

struct float3;
struct Config;

enum {
    WALL_VEL_V_CSTE,
    WALL_VEL_V_SHEAR,
    WALL_VEL_V_HS,
};

void wvel_ini(Wvel **wv);
void wvel_fin(Wvel *wv);

void wvel_step_ini(WvelStep **wv);
void wvel_step_fin(WvelStep *wv);

void wvel_set_cste(float3 u, Wvel *vw);
void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw);
void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, Wvel *vw);
void wvel_set_hs(float u, float h, Wvel *vw);
void wvel_set_timestep(float dt, Wvel *vw);

void wvel_set_conf(const Config *cfg, Wvel *vw);

void wvel_get_step(float t, const Wvel *wv,      WvelStep *);

int wvel_get_type(const WvelStep *w);

void wvel_get_view(const WvelStep *w,      WvelCste_v *v);
void wvel_get_view(const WvelStep *w,      WvelShear_v *v);
void wvel_get_view(const WvelStep *w,      WvelHS_v *v);
#ifndef UDX_SIM_WALLS_COMMON
#define UDX_SIM_WALLS_COMMON
struct PairParams;
struct PaArray;
struct FoArray;
struct WvelStep;
struct Coords;
struct Sdf;
struct int3;
struct float4;
struct Config;

struct WallQuants {
    float4 *pp;
    int n;
    int3 L;
};

struct WallTicket;
struct WallRepulsePrm;


void wall_ini_quants(int3 L, WallQuants*);
void wall_ini_ticket(int3 L, WallTicket**);

void wall_fin_quants(WallQuants*);
void wall_fin_ticket(WallTicket*);

void wall_gen_quants(MPI_Comm, int numdensity, int maxn, const Sdf*,          int *o_n, Particle *o_pp,      WallQuants*);
void wall_gen_ticket(const WallQuants*, WallTicket*);

void wall_strt_quants(MPI_Comm, const char *base, int maxn, WallQuants*);
void wall_strt_dump_templ(MPI_Comm, const char *base, const WallQuants*);

void wall_force(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                const WallTicket*, int n, const PaArray*, const FoArray*);

void wall_force_adhesion(const PairParams*, const WvelStep *, const Coords*, const Sdf*, const WallQuants*,
                         const WallTicket*, int n, const PaArray*, const FoArray*);

void wall_repulse(const Sdf*, const WallRepulsePrm*, long n, const PaArray*, const FoArray*);

void wall_repulse_prm_ini(float lambda, WallRepulsePrm**);
void wall_repulse_prm_ini_conf(const Config*, const char *ns, WallRepulsePrm**);
void wall_repulse_prm_fin(WallRepulsePrm*);
#endif
struct OptParams {
    int3 L;
    float kBT;
    int numdensity;
};
struct Wall;

struct OptParams;
struct Config;
struct PairParams;
struct Coords;

void wall_ini(const Config*, int3 L, Wall**);
void wall_fin(Wall*);

void wall_gen(MPI_Comm, const Coords*, OptParams, bool dump_sdf,
                     int *n, Particle *pp,      Wall*);

void wall_restart(MPI_Comm, const Coords*, OptParams, bool dump_sdf,
                  const char *base,      Wall*);

void wall_dump_templ(const Wall*, MPI_Comm, const char *base);

void wall_interact(const Coords*, const PairParams*[], Wall*, PFarrays*);
void wall_adhesion(const Coords*, const PairParams*[], Wall*, PFarrays*);
void wall_bounce(const Wall*, const Coords*, float dt, PFarrays*);
void wall_repulse(const Wall*, const WallRepulsePrm *[], PFarrays*);

void wall_update_vel(float t, Wall *);

void wall_get_sdf_ptr(const Wall*, const Sdf**);
double wall_compute_volume(const Wall*, MPI_Comm, int3 L);
struct Wall {
    Sdf *sdf;
    WallQuants q;
    WallTicket *t;
    Wvel *vel;
    WvelStep *velstep;
};
void wall_ini(const Config *cfg, int3 L, Wall **wall) {
    Wall *w;
    EMALLOC(1, wall);
    w = *wall;
    UC(sdf_ini(L, &w->sdf));
    UC(wall_ini_quants(L, &w->q));
    UC(wall_ini_ticket(L, &w->t));
    UC(wvel_ini(&w->vel));
    UC(wvel_set_conf(cfg, w->vel));
    UC(wvel_step_ini(&w->velstep));
}

void wall_fin(Wall *w) {
    UC(sdf_fin(w->sdf));
    UC(wall_fin_quants(&w->q));
    UC(wall_fin_ticket(w->t));
    UC(wvel_fin(w->vel));
    UC(wvel_step_fin(w->velstep));
    EFREE(w);
}
static long get_max_parts(OptParams p) {
    int3 L = p.L;
    return p.numdensity *
        (L.x + 2 * XWM) *
        (L.y + 2 * YWM) *
        (L.z + 2 * ZWM);
}

void wall_gen(MPI_Comm cart, const Coords *coords, OptParams op, bool dump_sdf,
                     int *n, Particle *pp,      Wall *w) {
    long maxn, nold, nnew;
    if (!w) return;

    maxn = get_max_parts(op);
    nold = *n;
    UC(sdf_gen(coords, cart, dump_sdf,      w->sdf));
    MC(m::Barrier(cart));
    UC(wall_gen_quants(cart, op.numdensity, maxn, w->sdf,      n, pp, &w->q));
    if (w->q.n) UC(wall_gen_ticket(&w->q, w->t));
    nnew = *n;
    msg_print("solvent particles survived: %d/%d", nnew, nold);
}

void wall_restart(MPI_Comm cart, const Coords *coords, OptParams op, bool dump_sdf,
                  const char *base,      Wall *w) {
    long maxn;
    if (!w) return;

    maxn = get_max_parts(op);
    UC(sdf_gen(coords, cart, dump_sdf,      w->sdf));
    MC(m::Barrier(cart));
    UC(wall_strt_quants(cart, base, maxn, &w->q));
    UC(wall_gen_ticket(&w->q, w->t));
}

void wall_dump_templ(const Wall *w, MPI_Comm cart, const char *base) {
    UC(wall_strt_dump_templ(cart, base, &w->q));
}

void wall_get_sdf_ptr(const Wall *w, const Sdf **s) {*s = w->sdf;}

double wall_compute_volume(const Wall *w, MPI_Comm comm, int3 L) {
    enum {NSAMPLES = 100000};
    return sdf_compute_volume(comm, L, w->sdf, NSAMPLES);
}

void wall_interact(const Coords *coords, const PairParams *prms[], Wall *w, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;
    const PairParams *prm;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        prm = prms[i];
        if (!n)   continue;
        if (!prm) continue;
        UC(wall_force(prm, w->velstep, coords, w->sdf, &w->q, w->t, n, &p,      &f));
    }
}

void wall_adhesion(const Coords *coords, const PairParams* params[], Wall *w, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;
    const PairParams *par;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        par = params[i];

        if (!par) continue;
        if (!n)   continue;

        UC(wall_force_adhesion(par, w->velstep, coords, w->sdf, &w->q, w->t, n, &p,      &f));
    }
}

void wall_bounce(const Wall *w, const Coords *coords, float dt, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        if (n) UC(sdf_bounce(dt, w->velstep, coords, w->sdf, n,      (Particle*) p.pp));
    }
}

void wall_repulse(const Wall *w, const WallRepulsePrm *params[], PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;
    const WallRepulsePrm *wrp;

    na = pfarrays_size(aa);
    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        wrp = params[i];

        if (!n)   continue;
        if (!wrp) continue;

        UC(wall_repulse(w->sdf, wrp, n, &p,      &f));
    }
}

void wall_update_vel(float t, Wall *w) {
    UC(wvel_get_step(t, w->vel,      w->velstep));
}
