#include "memory.cuh"
#include "macros.cuh"

#include "config.cuh"

#include <stdio.h>
#include <assert.h>

#ifndef UDX_FORCE_TYPE_COMMON
#define UDX_FORCE_TYPE_COMMON
#include "types.cuh"
#endif
#ifndef UDX_FORCE_DEF_COMMON
#define UDX_FORCE_DEF_COMMON
#include "constants.cuh"
#endif

#ifndef UDX_FORCE_DEVICE_API_COMMON
#define UDX_FORCE_DEVICE_API_COMMON
#include "device.cuh"
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
#define atomicAdd(...) (0)

#define atomicExch(...)
#define __float2half_rn(...) (0)
#define __fmaf_rz(...) (0.0)
#define __ldg(p) (*(p))
#define __popc(...)
#define __shfl(...) (0.0)
#define __shfl_down(...) (0.0)
#define __shfl_up(...) (0)
#define __shfl_xor(...) (0)

#define __syncthreads(...)

#define tex3D(...)

#define  Ifetch(t, i)  (0);
#define F4fetch(t, i)  (make_float4(0, 0, 0, 0))
#define F2fetch(t, i)  (make_float2(0, 0))
#define Ffetch(t, i)   (0.0)
#define Tfetch(T, t, i) (T{})

#define Ttex3D(T, to, i, j, k) (T{})
namespace d {
static uint3 threadIdx, blockIdx, gridDim;
static dim3  blockDim;
static int   warpSize = 1;
}

#define threadIdx d::threadIdx
#define blockDim  d::blockDim
#define gridDim   d::gridDim
#define blockIdx  d::blockIdx
#define warpSize  d::warpSize
#else
  #error DEV_* is undefined
#endif

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
#ifndef UDX_FORCE_KL_COMMON
#define UDX_FORCE_KL_COMMON
#include "launch.cuh"
#endif
#include "warp.cuh"

#ifndef UDX_FORCE_CC_COMMON
#define UDX_FORCE_CC_COMMON

#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}

#endif
#include "texo.cuh"

struct RNDunif;
void rnd_ini(int x, int y, int z, int c,      RNDunif **r);
void rnd_fin(RNDunif *r);
float rnd_get(RNDunif *r);
#include "randomdev.cuh"

#include "vecmath.cuh"
#define UDX_FORCE_MATH_COMMON

#ifndef UDX_FORCE_PAIR_TYPE_COMMON
#define UDX_FORCE_PAIR_TYPE_COMMON

enum {
    MAX_PAR = (N_COLOR * (N_COLOR + 1)) / 2
};

struct PairDPD {
    float a, g, s, spow;
};

struct PairDPDC {
    int ncolors;
    float a[MAX_PAR], g[MAX_PAR], s[MAX_PAR], spow;
};

struct PairDPDCM {
    int ncolors;
    float a[N_COLOR], g[N_COLOR], s[N_COLOR], spow;
};

struct PairDPDLJ {
    float a, g, s, spow;
    float ljs, lje;
};

struct PairAdhesion {
    float k1, k2;
};

struct PairPa {
    float x, y, z;
    float vx, vy, vz;

    int color;
};

struct PairFo {
    float x, y, z;
};

struct PairSFo {
    float x, y, z;

    float sxx, syy, szz, sxy, sxz, syz;
};
#endif
#ifndef UDX_FORCE_PAIR_DEV_COMMON
#define UDX_FORCE_PAIR_DEV_COMMON
#define _S_ static __device__
#define _I_ static __device__

static const float EPS = 1e-6;
enum {NORM_OK, NORM_BIG, NORM_SMALL};
_S_ int norm(       float3 *pos,      float *pr, float *pinvr) {

    float x, y, z, invr, r, r2;
    x = pos->x; y = pos->y; z = pos->z;

    r2 = x*x + y*y + z*z;
    if      (r2 >= 1 )   return NORM_BIG;
    else if (r2 < EPS) {
        *pr = pos->x = pos->y = pos->z = 0;
        return NORM_SMALL;
    } else {
        invr = rsqrtf(r2);
        r = r2 * invr;
        x *= invr; y *= invr; z *= invr;
        pos->x = x; pos->y = y; pos->z = z; *pr = r; *pinvr = invr;
        return NORM_OK;
    }
}

_S_ float ker_wrf(float s, float x) {
    return powf(x, s);
}

_S_ float magn_dpd(float a, float g, float s, float kpow,
                   float rnd, float r, float ev) {
    float wr, wc;
    float rm, f0;

    rm = max(1 - r, 0.0f);
    wc = rm;
    wr = ker_wrf(kpow, rm);

    f0  = (-g * wr * ev + s * rnd) * wr;
    f0 +=                        a * wc;
    return f0;
}

_S_ float force_magn(const PairDPD *p, float rnd, float ev, float r, float) {
    return magn_dpd(p->a, p->g, p->s, p->spow, rnd, r, ev);
}

_S_ void magn2fo(float f0, float3 er, float3 dr,      PairFo *f) {
    f->x = f0 * er.x;
    f->y = f0 * er.y;
    f->z = f0 * er.z;
}

_S_ void magn2fo(float f0, float3 er, float3 dr,      PairSFo *f) {
    f->x = f0 * er.x;
    f->y = f0 * er.y;
    f->z = f0 * er.z;

    f->sxx = 0.5f * f->x * dr.x;
    f->sxy = 0.5f * f->x * dr.y;
    f->sxz = 0.5f * f->x * dr.z;
    f->syy = 0.5f * f->y * dr.y;
    f->syz = 0.5f * f->y * dr.z;
    f->szz = 0.5f * f->z * dr.z;
}

_S_ void make_zero(PairFo *f) {
    f->x = f->y = f->z = 0;
}

_S_ void make_zero(PairSFo *f) {
    f->x = f->y = f->z = 0;
    f->sxx = f->sxy = f->sxz = 0;
    f->syy = f->syz = f->szz = 0;
}

template <typename Param, typename Fo>
_I_ void pair_force(const Param *p, PairPa a, PairPa b, float rnd,      Fo *f)
{
    float r, invr, ev, f0;
    float3 dr, er, dv;
    int vnstat;

    dr.x = a.x - b.x;
    dr.y = a.y - b.y;
    dr.z = a.z - b.z;

    dv.x = a.vx - b.vx;
    dv.y = a.vy - b.vy;
    dv.z = a.vz - b.vz;

    er = dr;

    vnstat = norm(       &er,       &r, &invr);
    if (vnstat == NORM_BIG) {
        make_zero(f);
        return;
    }

    ev = dot<float>(&er, &dv);

    f0 = force_magn(p, rnd, ev, r, invr);
    magn2fo(f0, er, dr,      f);
}

_S_ int colors2pid(int ca, int cb) {
    int c0, c1;
    c0 = ca < cb ? ca : cb;
    c1 = ca < cb ? cb : ca;
    return c1 * (c1+1) / 2 + c0;
}

template <typename Fo>
_I_ void pair_force(const PairDPDC *pc, PairPa a, PairPa b, float rnd,      Fo *f) {
    PairDPD p;
    int pid;
    pid = colors2pid(a.color, b.color);
    p.a = pc->a[pid];
    p.g = pc->g[pid];
    p.s = pc->s[pid];
    p.spow = pc->spow;
    pair_force(&p, a, b, rnd,      f);
}

template <typename Fo>
_I_ void pair_force(const PairDPDCM *pc, PairPa a, PairPa b, float rnd,      Fo *f) {
    PairDPD p;
    int pid = a.color;
    p.a = pc->a[pid];
    p.g = pc->g[pid];
    p.s = pc->s[pid];
    p.spow = pc->spow;
    pair_force(&p, a, b, rnd,      f);
}

_I_ void pair_add(const PairFo *b,      PairFo *a)
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;
}

_I_ void pair_add(const PairSFo *b,      PairSFo *a)
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;

    a->sxx += b->sxx;
    a->sxy += b->sxy;
    a->sxz += b->sxz;
    a->syy += b->syy;
    a->syz += b->syz;
    a->szz += b->szz;
}

#undef _S_
#undef _I_
#endif
struct PairParams;
struct Config;

struct PairDPD;
struct PairDPDC;
struct PairDPDCM;
struct PairDPDLJ;
struct PairAdhesion;

void pair_ini(PairParams **);
void pair_fin(PairParams *);

void pair_set_dpd(int ncol, const float a[], const float g[], float spow, PairParams *p);
void pair_set_lj(float sigma, float eps, PairParams *p);
void pair_set_adhesion(float k1, float k2, PairParams *p);

void pair_set_conf(const Config *, const char *name_space, PairParams *);

void pair_compute_dpd_sigma(float kBT, float dt, PairParams *p);

void pair_get_view_dpd(const PairParams*, PairDPD*);
void pair_get_view_dpd_color(const PairParams*, PairDPDC*);
void pair_get_view_dpd_mirrored(const PairParams*, PairDPDCM*);
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
void pair_get_view_adhesion(const PairParams*, PairAdhesion*);

struct FoArray_v {
    float *ff;
};

struct FoSArray_v {
    float *ff;
    float *ss;
};
#define _I_ static __device__
#define _S_ static __device__

template <int S, typename Fo, typename FA>
_S_ void farray_atomic_add_force(const Fo *f, int i,      FA a) {
    enum {X, Y, Z};
    float *af = a.ff + 3*i;
    atomicAdd(&af[X], S * f->x);
    atomicAdd(&af[Y], S * f->y);
    atomicAdd(&af[Z], S * f->z);
}

_S_ void farray_atomic_add_stress(const PairSFo *f, int i,      FoSArray_v a) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    float *s = a.ss + 6*i;
    atomicAdd(&s[XX], f->sxx);
    atomicAdd(&s[XY], f->sxy);
    atomicAdd(&s[XZ], f->sxz);
    atomicAdd(&s[YY], f->syy);
    atomicAdd(&s[YZ], f->syz);
    atomicAdd(&s[ZZ], f->szz);
}

template <int S>
_I_ void farray_atomic_add(const PairFo *f, int i,      FoArray_v a)
{
    farray_atomic_add_force<S>(f, i,      a);
}

template <int S>
_I_ void farray_atomic_add(const PairSFo *f, int i,      FoSArray_v a)
{
    farray_atomic_add_force<S>(f, i,      a);
    farray_atomic_add_stress(f, i,      a);
}

_I_ PairFo farray_fo0(FoArray_v)
{
    PairFo f;
    f.x = f.y = f.z = 0;
    return f;
}

_I_ PairSFo farray_fo0(FoSArray_v)
{
    PairSFo f;
    f.x = f.y = f.z = 0;
    f.sxx = f.sxy = f.sxz = 0;
    f.syy = f.syz = f.szz = 0;
    return f;
}

#undef _I_
#undef _S_
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

struct PairParams;
struct Force;
struct float4;
struct int3;
struct RNDunif;
struct FoArray;

#ifndef UDX_FORCE_BPA_COMMON
#define UDX_FORCE_BPA_COMMON
struct BPaArray {
    bool colors;
    const float4 *pp;
    const int *cc;
};
#endif

void flocal_push_pp(const float4 *pp, BPaArray *a);
void flocal_push_cc(const int *cc, BPaArray *a);

void flocal_apply(const PairParams*, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd,
                       const FoArray *ff);

struct TBPaArray_v {
    Texo<float4> pp;
};

struct TBPaCArray_v {
    Texo<float4> pp;
    Texo<int> cc;
};

#define _S_ static __device__
#define _I_ static __device__

template <typename Parray>
_S_ void fbulk_bpa_fetch_p(Parray a, int i, PairPa *p) {
    float4 r, v;
    r = a.pp[2*i + 0];
    v = a.pp[2*i + 1];

    p->x  = r.x;  p->y  = r.y;  p->z  = r.z;
    p->vx = v.x;  p->vy = v.y;  p->vz = v.z;
}

template <typename Parray>
_S_ void fbulk_tbpa_fetch_p(Parray a, int i, PairPa *p) {
    float4 r, v;
    r = texo_fetch(a.pp, 2*i + 0);
    v = texo_fetch(a.pp, 2*i + 1);

    p->x  = r.x;  p->y  = r.y;  p->z  = r.z;
    p->vx = v.x;  p->vy = v.y;  p->vz = v.z;
}

_I_ void fbulk_fetch(TBPaArray_v a, int i, PairPa *p) {
    fbulk_tbpa_fetch_p(a, i, p);
}

_I_ void fbulk_fetch(TBPaCArray_v a, int i, PairPa *p) {
    fbulk_tbpa_fetch_p(a, i, p);
    p->color = texo_fetch(a.cc, i);
}

#undef _S_
#undef _I_
#define _S_ static __device__

_S_ bool fbulk_cutoff_range(PairPa pa, PairPa pb) {
    float x, y, z;
    x = pa.x - pb.x;
    y = pa.y - pb.y;
    z = pa.z - pb.z;
    return x*x + y*y + z*z <= 1.f;
}

_S_ int3 fbulk_get_cid(int3 L, const PairPa *pa) {
    int3 c;
    c.x = pa->x + L.x / 2;
    c.y = pa->y + L.y / 2;
    c.z = pa->z + L.z / 2;
    return c;
}

_S_ bool fbulk_valid_c(int c, int hi) {
    return (c >= 0) && (c < hi);
}

template <typename Par, typename Parray, typename Farray, typename Fo>
_S_ void fbulk_loop_pp(const Par *params, int ia, PairPa pa, Parray parray, int start, int end, float seed,      Fo *fa, Farray farray) {
    enum {X, Y, Z};
    int ib;
    PairPa pb;
    Fo f;
    float rnd;

    for (ib = start; ib < end; ++ib) {
        if (ib >= ia) continue;

        fbulk_fetch(parray, ib, &pb);

        if (!fbulk_cutoff_range(pa, pb)) continue;

        rnd = rnd::mean0var1ii(seed, ia, ib);
        pair_force(params, pa, pb, rnd,      &f);

        pair_add(&f,      fa);

        farray_atomic_add<-1>(&f, ib,      farray);
    }
}

template <typename Par, typename PArray, typename Farray, typename Fo>
_S_ void fbulk_one_row(const Par *params, int3 L, int dz, int dy, int ia, int3 ca, PairPa pa, PArray parray, const int *start, float seed,
                                    Fo *fa, Farray farray) {
    int3 cb;
    int enddx, startx, endx, cid0, bs, be;
    cb.z = ca.z + dz;
    cb.y = ca.y + dy;
    if (!fbulk_valid_c(cb.z, L.z)) return;
    if (!fbulk_valid_c(cb.y, L.y)) return;

    enddx = (dz == 0 && dy == 0) ? 0 : 1;

    startx =     max(    0, ca.x - 1    );
    endx   = 1 + min(L.x-1, ca.x + enddx);

    cid0 = L.x * (cb.y + L.y * cb.z);

    bs = start[cid0 + startx];
    be = start[cid0 + endx];

    fbulk_loop_pp(params, ia, pa, parray, bs, be, seed,      fa, farray);
}

template <typename Par, typename Parray, typename Farray>
__global__ void fbulk_apply_kernel(Par params, int3 L, int n, Parray parray, const int *start, float seed,      Farray farray) {
    enum {X, Y, Z};
    int ia;
    int3 ca;
    PairPa pa;
    auto fa = farray_fo0(farray);

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;

    fbulk_fetch(parray, ia, &pa);
    ca = fbulk_get_cid(L, &pa);

#define ONE_ROW(dz, dy) fbulk_one_row (&params, L, dz, dy, ia, ca, pa, parray, start, seed,      &fa, farray)

    ONE_ROW(-1, -1);
    ONE_ROW(-1,  0);
    ONE_ROW(-1,  1);
    ONE_ROW( 0, -1);
    ONE_ROW( 0,  0);

#undef ONE_ROW

    farray_atomic_add<1>(&fa, ia, farray);
}

#undef _S_

void flocal_push_pp(const float4 *pp, BPaArray *a) {
    a->pp = pp;
    a->colors = false;
}

void flocal_push_cc(const int *cc, BPaArray *a) {
    a->cc = cc;
    a->colors = true;
}

static void tbarray_get_view(int n, BPaArray a, TBPaArray_v *v) {
    texo_setup(2*n, (float4*) a.pp, &v->pp);
}

static void tbarray_get_view(int n, BPaArray a, TBPaCArray_v *v) {
    texo_setup(2*n, (float4*) a.pp, &v->pp);
    texo_setup(n,      (int*) a.cc, &v->cc);
}

static void tbarray_fin_view(TBPaArray_v *v) {
    texo_destroy(&v->pp);
}

static void tbarray_fin_view(TBPaCArray_v *v) {
    texo_destroy(&v->pp);
    texo_destroy(&v->cc);
}

template <typename Par, typename Parray, typename Farray>
static void interactions(Par params, int3 L, int n, Parray parray, const int *start, RNDunif *rnd,      Farray farray) {
    enum {THR=128};
    float seed;
    if (n <= 0) return;
    seed = rnd_get(rnd);

    KL(fbulk_apply_kernel,
       (ceiln((n), THR), THR),
       (params, L, n, parray, start, seed,      farray));
}

template <typename Par, typename Parray>
static void dispatch_stress(Par params, int3 L, int n, Parray parray, const int *start, RNDunif *rnd,      const FoArray *farray) {
    if (farray_has_stress(farray)) {
        FoSArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, n, parray, start, rnd,      farray_v);
    }
    else {
        FoArray_v farray_v;
        farray_get_view(farray, &farray_v);
        interactions(params, L, n, parray, start, rnd,      farray_v);
    }
}

static void apply_color(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd,      const FoArray *farray) {
    PairDPDC pv;
    TBPaCArray_v parray_v;

    pair_get_view_dpd_color(params, &pv);
    tbarray_get_view(n, parray, &parray_v);

    dispatch_stress(pv, L, n, parray_v, start, rnd,      farray);

    tbarray_fin_view(&parray_v);
}

static void apply_grey(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd,      const FoArray *farray) {
    PairDPD pv;
    TBPaArray_v parray_v;
    pair_get_view_dpd(params, &pv);
    tbarray_get_view(n, parray, &parray_v);

    dispatch_stress(pv, L, n, parray_v, start, rnd,      farray);

    tbarray_fin_view(&parray_v);
}

void flocal_apply(const PairParams *params, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd,      const FoArray *farray) {
    if (parray.colors)
        apply_color(params, L, n, parray, start, rnd,      farray);
    else
        apply_grey(params, L, n, parray, start, rnd,      farray);
}

#include <mpi.h>
#include <assert.h>
#include <vector_types.h>

#include <cuda_fp16.h>

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

struct RNDunif;
void rnd_ini(int x, int y, int z, int c,      RNDunif **r);
void rnd_fin(RNDunif *r);
float rnd_get(RNDunif *r);

#ifndef UDX_FORCE_TYPE_COMMON
#define UDX_FORCE_TYPE_COMMON
#include "types.cuh"
#endif

#ifndef UDX_FORCE_PARRAY_COMMON
#define UDX_FORCE_PARRAY_COMMON
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

#ifndef UDX_FORCE_DEVICE_API_COMMON
#define UDX_FORCE_DEVICE_API_COMMON
#include "device.cuh"
#endif
#ifndef UDX_FORCE_KL_COMMON
#define UDX_FORCE_KL_COMMON
#include "launch.cuh"
#endif
#ifndef UDX_FORCE_CC_COMMON
#define UDX_FORCE_CC_COMMON

#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}

#endif

#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}

void msg_ini(int rank);
void msg_print(const char *fmt, ...);
namespace m {
int Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Barrier(MPI_Comm comm);
int Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int Cancel(MPI_Request *request);
int Cart_rank(MPI_Comm comm, const int coords[], int *rank);
int Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart);
int Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
int Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]);
int Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int Comm_free(MPI_Comm *comm);
int Comm_rank(MPI_Comm comm, int *rank);
int Comm_size( MPI_Comm comm, int *size );
int Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
int File_close(MPI_File *fh);
int File_get_position(MPI_File fh, MPI_Offset *offset);
int File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);
int File_seek(MPI_File fh, MPI_Offset offset, int whence);
int File_set_size(MPI_File fh, MPI_Offset size);
int File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
int Finalize(void);
int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
           int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
            const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm);
int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
              int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
               const int displs[], MPI_Datatype recvtype, MPI_Comm comm);
int Init(int *argc, char ***argv);
int Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
int Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
int Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
int Type_commit(MPI_Datatype *type);
int Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);
int Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int Error_string(int errorcode, char *string, int *resultlen);

bool is_master(MPI_Comm comm);
int is_success(int errorcode);
int is_pending(int errorcode);
int is_err_in_status(int errorcode);
int status2errcode(MPI_Status *status);
}
#ifndef UDX_FORCE_FRAG_COMMON
#define UDX_FORCE_FRAG_COMMON
enum {frag_bulk = 26};
#endif

struct int3;

namespace frag_hst {

int i2dx(int i);
int i2dy(int i);
int i2dz(int i);
void i2d3(int i,      int d[3]);

int d2i(int x, int y, int z);
int d32i(const int d[3]);

int ncell(int3 L, int i);

int antid2i(int x, int y, int z);
int anti(int i);

void estimates(int3 L, int nfrags, float maxd,      int *cap);

}
struct Particle;
struct Positioncp;
struct float4;

void convert_pp2rr_current (long n, const Particle *pp, Positioncp *rr);
void convert_pp2rr_previous(long n, const Particle *pp, Positioncp *rr);
void convert_pp2f4_pos(int n, const float *pp,      float4 *zpp);

namespace flu {

enum { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct LFrag {
    PaArray parray;
    const int *ii;
    int n;
};

struct RFrag {
    PaArray parray;
    const int *start;
    int dx, dy, dz, xcells, ycells, zcells;
    int type;
};

struct RndFrag {
    float seed;
    int mask;
};

typedef Sarray<LFrag,   26> LFrag26;
typedef Sarray<RFrag,   26> RFrag26;
typedef Sarray<RndFrag, 26> RndFrag26;

}
struct PairParams;
struct Force;
struct float4;
struct int3;
struct RNDunif;
struct FoArray;

#ifndef UDX_FORCE_BPA_COMMON
#define UDX_FORCE_BPA_COMMON
struct BPaArray {
    bool colors;
    const float4 *pp;
    const int *cc;
};
#endif

void flocal_push_pp(const float4 *pp, BPaArray *a);
void flocal_push_cc(const int *cc, BPaArray *a);

void flocal_apply(const PairParams*, int3 L, int n, BPaArray parray, const int *start, RNDunif *rnd,
                       const FoArray *ff);

struct PairParams;
struct int3;
struct FoArray;

void fhalo_apply(const PairParams*, int3 L, const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd,      const FoArray *farray);

struct PairParams;
struct FluForcesBulk;
struct FluForcesHalo;
struct PaArray;
struct FoArray;

void fluforces_bulk_ini(int3 L, int maxp,      FluForcesBulk **b);
void fluforces_bulk_fin(     FluForcesBulk *b);

void fluforces_bulk_prepare(int n, const PaArray *a,      FluForcesBulk *b);
void fluforces_bulk_apply(const PairParams*, int n, const FluForcesBulk *b, const int *start, const int *count,
                               const FoArray *ff);

void fluforces_halo_ini(MPI_Comm cart, int3 L,      FluForcesHalo **hd);
void fluforces_halo_fin(     FluForcesHalo *h);

void fluforces_halo_prepare(flu::LFrag26 lfrags, flu::RFrag26 rfrags,      FluForcesHalo *h);
void fluforces_halo_apply(const PairParams*, const FluForcesHalo *h,      const FoArray *farray);

struct FluForcesBulk {
    float4  *zipped_pp;
    RNDunif *rnd;
    const int *colors;

    int3 L;
};

struct FluForcesHalo {
    RNDunif *trunks[26];
    bool masks[26];

    flu::LFrag26 lfrags;
    flu::RFrag26 rfrags;
    flu::RndFrag26 rndfrags;

    int3 L;
};
void fluforces_bulk_ini(int3 L, int maxp,      FluForcesBulk **bd) {
    FluForcesBulk *b;
    EMALLOC(1, bd);
    b = *bd;
    Dalloc(&b->zipped_pp, 2 * maxp);
    UC(rnd_ini(0, 0, 0, 0,      &b->rnd));
    b->colors = NULL;
    b->L = L;
}

static int is_plus(const int d[3]) {
    enum {X, Y, Z};
    return (d[X] + d[Y] + d[Z] > 0) ||
        ((d[X] + d[Y] + d[Z] == 0) && (d[X] > 0)) ||
        ((d[X] == 0) && (d[Y] > 0)) ||
        (d[Y] == 0 && d[Z] > 0);
}

static int imin(int a, int b) {return a < b ? a : b;}
static int imax(int a, int b) {return a < b ? b : a;}

static void get_interrank_infos(MPI_Comm cart, int fid,      RNDunif* trunks[], bool masks[]) {
    int coordsneighbor[3], c, indx[3], rank, dstrank;
    int seed, seed_base, seed_offset;
    int dims[3], periods[3], coords[3];
    int d[3];
    frag_hst::i2d3(fid, d);

    MC(m::Cart_get(cart, 3, dims, periods, coords));
    MC(m::Cart_rank(cart, coords,      &rank));

    for (c = 0; c < 3; ++c)
        coordsneighbor[c] = (coords[c] + d[c] + dims[c]) % dims[c];

    MC(m::Cart_rank(cart, coordsneighbor,      &dstrank));

    for (c = 0; c < 3; ++c)
        indx[c] = imin(coords[c], coordsneighbor[c]) * dims[c] +
            imax(coords[c], coordsneighbor[c]);

    seed_base = indx[0] + dims[0] * dims[0] * (indx[1] + dims[1] * dims[1] * indx[2]);

    {
        int mysign = 2 * is_plus(d) - 1;
        int v[3] = {1 + mysign * d[0], 1 + mysign * d[1], 1 + mysign * d[2]};
        seed_offset = v[0] + 3 * (v[1] + 3 * v[2]);
    }

    seed = seed_base + seed_offset;

    UC(rnd_ini(390 + seed, seed + 615, 12309, 23094,      &trunks[fid]));

    if (dstrank != rank)
        masks[fid] = imin(dstrank, rank) == rank;
    else {
        int alter_ego = frag_hst::anti(fid);
        masks[fid] = imin(fid, alter_ego) == fid;
    }
}

void fluforces_halo_ini(MPI_Comm cart, int3 L,      FluForcesHalo **hd) {
    FluForcesHalo *h;
    EMALLOC(1, hd);
    h = *hd;

    h->L = L;

    for (int i = 0; i < 26; ++i)
        get_interrank_infos(cart, i,      h->trunks, h->masks);
}
void fluforces_bulk_fin(     FluForcesBulk *b) {
    CC(d::Free(b->zipped_pp));
    UC(rnd_fin(b->rnd));
    EFREE(b);
}

void fluforces_halo_fin(     FluForcesHalo *h) {
    for (int i = 0; i < 26; ++i)
        UC(rnd_fin(h->trunks[i]));
    EFREE(h);
}
void fluforces_bulk_prepare(int n, const PaArray *a,      FluForcesBulk *b) {
    if (n == 0) return;
    convert_pp2f4_pos(n, a->pp,      b->zipped_pp);
    if (a->colors)
        b->colors = a->cc;
    else
        b->colors = NULL;
}

void fluforces_bulk_apply(const PairParams *par, int n, const FluForcesBulk *b, const int *start, const int *count,
                               const FoArray *farray) {
    BPaArray a;
    if (n == 0) return;

    flocal_push_pp(b->zipped_pp, &a);
    if (b->colors)
        flocal_push_cc(b->colors, &a);

    UC(flocal_apply(par, b->L, n, a, start, b->rnd,      farray));
}

void fluforces_halo_prepare(flu::LFrag26 lfrags, flu::RFrag26 rfrags,      FluForcesHalo *h) {
    h->lfrags = lfrags;
    h->rfrags = rfrags;

    for (int i = 0; i < 26; ++i) {
        h->rndfrags.d[i] = {
            .seed = rnd_get(h->trunks[i]),
            .mask = h->masks[i]
        };
    }
}

void fluforces_halo_apply(const PairParams *par, const FluForcesHalo *h,      const FoArray *farray) {
    fhalo_apply(par, h->L, h->lfrags, h->rfrags, h->rndfrags, farray);
}

#include <stdio.h>
#include <assert.h>

#ifndef UDX_FORCE_TYPE_COMMON
#define UDX_FORCE_TYPE_COMMON
#include "types.cuh"
#endif
bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void msg_ini(int rank);
void msg_print(const char *fmt, ...);

#ifndef UDX_FORCE_DEVICE_API_COMMON
#define UDX_FORCE_DEVICE_API_COMMON
#include "device.cuh"
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
#define atomicAdd(...) (0)

#define atomicExch(...)
#define __float2half_rn(...) (0)
#define __fmaf_rz(...) (0.0)
#define __ldg(p) (*(p))
#define __popc(...)
#define __shfl(...) (0.0)
#define __shfl_down(...) (0.0)
#define __shfl_up(...) (0)
#define __shfl_xor(...) (0)

#define __syncthreads(...)

#define tex3D(...)

#define  Ifetch(t, i)  (0);
#define F4fetch(t, i)  (make_float4(0, 0, 0, 0))
#define F2fetch(t, i)  (make_float2(0, 0))
#define Ffetch(t, i)   (0.0)
#define Tfetch(T, t, i) (T{})

#define Ttex3D(T, to, i, j, k) (T{})
namespace d {
static uint3 threadIdx, blockIdx, gridDim;
static dim3  blockDim;
static int   warpSize = 1;
}

#define threadIdx d::threadIdx
#define blockDim  d::blockDim
#define gridDim   d::gridDim
#define blockIdx  d::blockIdx
#define warpSize  d::warpSize
#else
  #error DEV_* is undefined
#endif

#if   defined(DEV_CUDA)

#elif defined(DEV_CPU)

#ifdef __device__
  #undef  __device__
  #define __device__
#endif

#ifdef __global__
  #undef  __global__
  #define __global__
#endif

#ifdef __host__
  #undef  __host__
  #define __host__
#endif

#ifdef __noinline__
  #undef  __noinline__
  #define __noinline__
#endif

#ifdef __forceinline__
  #undef  __forceinline__
  #define __forceinline__
#endif

#ifdef __constant__
  #undef  __constant__
  #define __constant__
#endif

#ifdef __shared__
  #undef  __shared__
  #define __shared__
#endif

#ifdef __managed__
  #undef  __managed__
  #define __managed__
#endif

#ifdef __restrict__
  #undef  __restrict__
  #define __restrict__
#endif
#else
  #error DEV_* is undefined
#endif
#ifndef UDX_FORCE_CC_COMMON
#define UDX_FORCE_CC_COMMON

#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}

#endif
#ifndef UDX_FORCE_KL_COMMON
#define UDX_FORCE_KL_COMMON
#include "launch.cuh"
#endif
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <curand.h>

void os_mkdir(const char* path);
long os_time();
void os_srand(long int seedval);
double os_drand();
void os_sleep(unsigned int seconds);
void os_malloc_stats();
void msg_ini(int rank);
void msg_print(const char *fmt, ...);
#ifndef UDX_FORCE_DEVICE_API_COMMON
#define UDX_FORCE_DEVICE_API_COMMON
#include "device.cuh"
#endif

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}

#include <stdio.h>
#include <assert.h>
#include <stddef.h>

#define CONTAINER_OF(ptr, type, member) \
    ((type*)((char*)(ptr) - offsetof(type, member)))

#ifndef UDX_FORCE_DEF_COMMON
#define UDX_FORCE_DEF_COMMON
#include "constants.cuh"
#endif

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void msg_ini(int rank);
void msg_print(const char *fmt, ...);
bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

struct RbcParams;
struct Kantor;
struct Force;
struct MeshRead;
struct RbcQuants;

void kantor_ini(const MeshRead*, Kantor**);
void kantor_fin(Kantor*);

void kantor_apply(Kantor*, const RbcParams*, const RbcQuants*,      Force*);
struct RbcParams;
struct Juelicher;
struct Force;
struct MeshRead;
struct RbcQuants;

void juelicher_ini(const MeshRead*, Juelicher**);
void juelicher_fin(Juelicher*);

void juelicher_apply(Juelicher*, const RbcParams*, const RbcQuants*,      Force*);

struct RbcParams;
struct Bending;
struct Force;
struct MeshRead;
struct RbcQuants;

void bending_kantor_ini(const MeshRead*, Bending**);
void bending_juelicher_ini(const MeshRead*, Bending**);
void bending_none_ini(const MeshRead*, Bending**);

void bending_fin(Bending*);

void bending_apply(Bending*, const RbcParams*, const RbcQuants*,      Force*);
struct Bending {
    struct Bending_vtable *vtable;
};

struct Bending_vtable {
    void (*apply)(Bending*, const RbcParams*, const RbcQuants*,      Force*);
    void (*fin)(Bending*);
};

void bending_apply(Bending *q, const RbcParams *params, const RbcQuants *quants,      Force *ff) {
    q->vtable->apply(q, params, quants,      ff);
}

void bending_fin(Bending *q) {
    q->vtable->fin(q);
}

struct BendingKantor {
    Bending bending;
    Kantor *kantor;
};
static void method_kantor_apply(Bending *q, const RbcParams *params, const RbcQuants *quants,      Force *ff) {
    BendingKantor *b = CONTAINER_OF(q, BendingKantor, bending);
    UC(kantor_apply(b->kantor, params, quants, ff));
}
static void method_kantor_fin(Bending *q) {
    BendingKantor *b = CONTAINER_OF(q, BendingKantor, bending);
    UC(kantor_fin(b->kantor));
    EFREE(q);
}
static Bending_vtable BendingKantor_vtable = { method_kantor_apply, method_kantor_fin};
void bending_kantor_ini(const MeshRead *cell,      Bending **pq) {
    BendingKantor *q;
    EMALLOC(1, &q);
    UC(kantor_ini(cell, &q->kantor));
    q->bending.vtable = &BendingKantor_vtable;
    *pq = &q->bending;
}

struct BendingJuelicher {
    Bending bending;
    Juelicher *juelicher;
};
static void method_juelicher_apply(Bending *q, const RbcParams *params, const RbcQuants *quants,      Force *ff) {
    BendingJuelicher *b = CONTAINER_OF(q, BendingJuelicher, bending);
    UC(juelicher_apply(b->juelicher, params, quants, ff));
}
static void method_juelicher_fin(Bending *q) {
    BendingJuelicher *b = CONTAINER_OF(q, BendingJuelicher, bending);
    UC(juelicher_fin(b->juelicher));
    EFREE(q);
}
static Bending_vtable BendingJuelicher_vtable = { method_juelicher_apply, method_juelicher_fin};
void bending_juelicher_ini(const MeshRead *cell,      Bending **pq) {
    BendingJuelicher *q;
    EMALLOC(1, &q);
    UC(juelicher_ini(cell, &q->juelicher));
    q->bending.vtable = &BendingJuelicher_vtable;
    *pq = &q->bending;
}

struct BendingNone { Bending bending; };
static void method_none_apply(Bending*, const RbcParams*, const RbcQuants*,      Force*) { }
static void method_none_fin(Bending*) { }
static Bending_vtable BendingNone_vtable = { method_none_apply, method_none_fin};
void bending_none_ini(const MeshRead*,      Bending **pq) {
    BendingNone *q;
    EMALLOC(1, &q);
    q->bending.vtable = &BendingNone_vtable;
    *pq = &q->bending;
}

#include <stdio.h>
#include <string.h>

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
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
struct int4;
struct MeshRead;

void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);

struct RbcParams;
struct RbcForce;
struct Force;
struct Config;
struct MeshRead;
struct RbcQuants;

void rbc_force_ini(const MeshRead*, RbcForce**);
void rbc_force_fin(RbcForce*);

void rbc_force_set_stressful(int nt, float totArea,      RbcForce*);
void rbc_force_set_stressfree(const char *fname,      RbcForce*);

void rbc_force_set_rnd0(RbcForce*);
void rbc_force_set_rnd1(int seed, RbcForce*);
void rbc_force_set_bending(const MeshRead*, const char*, RbcForce*);

void rbc_force_set_conf(const MeshRead*, const Config *cfg, const char *name, RbcForce*);

void rbc_force_apply(RbcForce*, const RbcParams*, float dt, const RbcQuants*,      Force*);
void rbc_force_stat(     float *pArea, float *pVolume);

void rbc_force_set_conf(const MeshRead *cell, const Config *cfg, const char *name, RbcForce *q) {
    int stress_free, rnd;
    const char *bending;
    UC(conf_lookup_bool_ns(cfg, name, "stress_free", &stress_free));
    UC(conf_lookup_bool_ns(cfg, name, "rnd", &rnd));
    UC(conf_lookup_string_ns(cfg, name, "bending", &bending));
    UC(rbc_force_set_bending(cell, bending,      q));

    if (stress_free) {
        const char *fname;
        UC(conf_lookup_string_ns(cfg, name, "stress_free_file", &fname));
        UC(rbc_force_set_stressfree(fname,      q));
    }
    else {
        float Atot = 0;
        int nt;
        nt = mesh_read_get_nt(cell);
        UC(conf_lookup_float_ns(cfg, name, "totArea", &Atot));
        UC(rbc_force_set_stressful(nt, Atot,      q));
    }

    if (rnd) {
        int seed;
        UC(conf_lookup_int_ns(cfg, name, "seed", &seed));
        UC(rbc_force_set_rnd1(seed, q));
    }
    else {
        UC(rbc_force_set_rnd0(q));
    }
}

#include <stdio.h>
#include <assert.h>

#ifndef UDX_FORCE_DEF_COMMON
#define UDX_FORCE_DEF_COMMON
#include "constants.cuh"
#endif
#ifndef UDX_FORCE_TYPE_COMMON
#define UDX_FORCE_TYPE_COMMON
#include "types.cuh"
#endif

  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace tri_dev {
  #define END }

BEGIN

_S_ void diff(const double *a, const double *b,      double  *c) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
_S_ double vabs(double *a) {
    enum {X, Y, Z};
    double r;
    r = a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z];
    return sqrt(r);
}
_S_ void swap(double *a, double *b) {
    double t;
    t = *a; *a = *b; *b = t;
}
_S_ int less(double *a, double *b) { return (*a) < (*b); }
_S_ void  sort3(double *a, double *b, double *c) {
    if (less(c, b)) swap(c, b);
    if (less(b, a)) swap(b, a);
    if (less(c, b)) swap(c, b);
}

_I_ double kahan_area0(double a, double b, double c) {
    sort3(&c, &b, &a);
    return sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))/4;
}

_I_ double kahan_area(const double r0[3], const double r1[3], const double r2[3]) {
    double r01[3], r12[3], r20[3], a, b, c;
    diff(r0, r1,      r01);
    diff(r1, r2,      r12);
    diff(r2, r0,      r20);
    a = vabs(r01); b = vabs(r12); c = vabs(r20);
    return kahan_area0(a, b, c);
}

_S_ double orient2d_xy(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = X, Q = Y};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_S_ double orient2d_zx(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = Z, Q = X};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_S_ double orient2d_yz(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = Y, Q = Z};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_I_ double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]) {
    enum {X, Y, Z};
    double ax, ay, az, bx, by, bz, cx, cy, cz;

    ax = a[X] - d[X]; ay = a[Y] - d[Y]; az = a[Z] - d[Z];
    bx = b[X] - d[X]; by = b[Y] - d[Y]; bz = b[Z] - d[Z];
    cx = c[X] - d[X]; cy = c[Y] - d[Y]; cz = c[Z] - d[Z];

    return ax*(by*cz - bz*cy) + bx*(cy*az - cz*ay) + cx*(ay*bz - az*by);
}

_I_ void ac_bc_cross(const double a[3], const double b[3], const double c[3],      double r[3]) {
    enum {X, Y, Z};
    r[X] = orient2d_yz(a, b, c);
    r[Y] = orient2d_zx(a, b, c);
    r[Z] = orient2d_xy(a, b, c);
}

_I_ void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                       double *px, double *py) {
    enum {X, Y, Z};
    double bcx, bcy, bcz, bc;
    double k[3], l[3];

    ac_bc_cross(b, c, a,      k);
    ac_bc_cross(c, b, d,      l);

    bcx = b[X] - c[X];
    bcy = b[Y] - c[Y];
    bcz = b[Z] - c[Z];
    bc = sqrt(bcx*bcx + bcy*bcy + bcz*bcz);

    *py = -orient3d(a, b, c, d)*bc;
    *px = k[X]*l[X] + k[Y]*l[Y] + k[Z]*l[Z];
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
void msg_ini(int rank);
void msg_print(const char *fmt, ...);
bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

#if   defined(DEV_CUDA)

#elif defined(DEV_CPU)

#ifdef __device__
  #undef  __device__
  #define __device__
#endif

#ifdef __global__
  #undef  __global__
  #define __global__
#endif

#ifdef __host__
  #undef  __host__
  #define __host__
#endif

#ifdef __noinline__
  #undef  __noinline__
  #define __noinline__
#endif

#ifdef __forceinline__
  #undef  __forceinline__
  #define __forceinline__
#endif

#ifdef __constant__
  #undef  __constant__
  #define __constant__
#endif

#ifdef __shared__
  #undef  __shared__
  #define __shared__
#endif

#ifdef __managed__
  #undef  __managed__
  #define __managed__
#endif

#ifdef __restrict__
  #undef  __restrict__
  #define __restrict__
#endif
#else
  #error DEV_* is undefined
#endif
#ifndef UDX_FORCE_DEVICE_API_COMMON
#define UDX_FORCE_DEVICE_API_COMMON
#include "device.cuh"
 #endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
#define atomicAdd(...) (0)

#define atomicExch(...)
#define __float2half_rn(...) (0)
#define __fmaf_rz(...) (0.0)
#define __ldg(p) (*(p))
#define __popc(...)
#define __shfl(...) (0.0)
#define __shfl_down(...) (0.0)
#define __shfl_up(...) (0)
#define __shfl_xor(...) (0)

#define __syncthreads(...)

#define tex3D(...)

#define  Ifetch(t, i)  (0);
#define F4fetch(t, i)  (make_float4(0, 0, 0, 0))
#define F2fetch(t, i)  (make_float2(0, 0))
#define Ffetch(t, i)   (0.0)
#define Tfetch(T, t, i) (T{})

#define Ttex3D(T, to, i, j, k) (T{})
namespace d {
static uint3 threadIdx, blockIdx, gridDim;
static dim3  blockDim;
static int   warpSize = 1;
}

#define threadIdx d::threadIdx
#define blockDim  d::blockDim
#define gridDim   d::gridDim
#define blockIdx  d::blockIdx
#define warpSize  d::warpSize
#else
  #error DEV_* is undefined
#endif

#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}

#ifndef UDX_FORCE_KL_COMMON
#define UDX_FORCE_KL_COMMON
#include "launch.cuh"
#endif

struct Config;
struct RbcParams;
struct RbcParams_v {
    float gammaC, gammaT, kBT;
    float kb, phi;
    float ks, x0, mpow;
    float ka, kd, kv;
    float totArea, totVolume;
};

void rbc_params_ini(RbcParams **);
void rbc_params_fin(RbcParams *);

void rbc_params_set_fluct(float gammaC, float gammaT, float kBT, RbcParams *);
void rbc_params_set_bending(float kb, float phi, RbcParams *);
void rbc_params_set_spring(float ks, float x0, float mpow , RbcParams *);
void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *);
void rbc_params_set_tot_area_volume(float totArea, float totVolume, RbcParams *);

double rbc_params_get_tot_volume(const RbcParams *);

void rbc_params_set_conf(const Config *c, const char *name, RbcParams *);

RbcParams_v rbc_params_get_view(const RbcParams *);
struct Adj;
struct Adj_v;
struct AdjMap;
struct int4;

void adj_ini(int md, int nt, int nv, const int4 *tt,      Adj**);
void adj_fin(Adj*);
int  adj_get_map(int i, const Adj*,      AdjMap *m);
int  adj_get_max(const Adj*);
void adj_get_anti(const Adj*,      int *anti);

void adj_view_ini(const Adj*, Adj_v**);
void adj_view_fin(Adj_v*);
struct RbcShape;
struct MeshRead;
struct Adj;

void rbc_shape_ini(const Adj*, const float *rr,      RbcShape**);
void rbc_shape_fin(     RbcShape*);

void rbc_shape_edg (RbcShape*,      float**);
void rbc_shape_area(RbcShape*,      float**);

void rbc_shape_total_area(RbcShape*,      float*);
void rbc_shape_total_volume(RbcShape*,      float*);
struct RbcParams;
struct Bending;
struct Force;
struct MeshRead;
struct RbcQuants;

void bending_kantor_ini(const MeshRead*, Bending**);
void bending_juelicher_ini(const MeshRead*, Bending**);
void bending_none_ini(const MeshRead*, Bending**);

void bending_fin(Bending*);

void bending_apply(Bending*, const RbcParams*, const RbcQuants*,      Force*);

struct RbcRnd;

#ifndef UDX_RBC_FORCE_SEED_TIME
#define UDX_RBC_FORCE_SEED_TIME
enum {SEED_TIME = -1};
#endif

void rbc_rnd_ini(int n, long seed, RbcRnd**);
void rbc_rnd_fin(RbcRnd*);

void rbc_rnd_gen(RbcRnd*, int n,      float**);
float rbc_rnd_get_hst(const RbcRnd*, int i);
struct AreaVolume;
struct Particle;

struct RbcQuants {
    int n, nc;
    int nt, nv;
    int md;
    Particle *pp, *pp_hst;
    bool ids;
    int *ii;
    AreaVolume *area_volume;
};

struct AdjMap {
    int i0, i1, i2, i3, i4;
    int rbc;
};
struct Adj_v {
    int *adj0;
    int *adj1;
    int nv, md;
};

static __device__ int fetch(const int *a, int i) { return a[i]; }
static __device__ int adj_get_map0(int md, int nv, int i, const int *adj0, const int *adj1,      AdjMap *m) {
    int i0, i1, i2, i3, i4;
    int rbc, offset, j, k;
    i0 = i / md;
    j  = i % md;

    k   = (i0 % nv) * md;
    i1 = fetch(adj0, k + j);
    if (i1 == -1) return 0;

    i2 = fetch(adj0, k + ((j + 1) % md));
    if (i2 == -1) {
        i2 = fetch(adj0, k    );
        i3 = fetch(adj0, k + 1);
    } else {
        i3 = fetch(adj0, k + ((j + 2) % md));
        if (i3 == -1) i3 = fetch(adj0, k);
    }
    i4 = fetch(adj1, k + j);

    rbc = i0 / nv;
    offset = rbc * nv;
    i1 += offset; i2 += offset; i3 += offset; i4 += offset;

    assert(rbc < MAX_CELL_NUM);

    m->rbc = rbc;
    m->i0 = i0; m->i1 = i1; m->i2 = i2; m->i3 = i3; m->i4 = i4;
    return 1;
}

static __device__ int adj_get_map(int i, Adj_v *adj,      AdjMap *m) {
    int md, nv;
    md = adj->md;
    nv = adj->nv;
    const int *adj0 = adj->adj0;
    const int *adj1 = adj->adj1;
    return adj_get_map0(md, nv, i, adj0, adj1, m);
}
struct AreaVolume;
struct Particle;
struct int4;

void area_volume_ini(int nv, int nt, const int4 *tt, int max_cell, AreaVolume**);
void area_volume_fin(AreaVolume*);

void area_volume_compute(AreaVolume*, int nm, const Particle*,      float **av);
void area_volume_host(AreaVolume*,      float **av);

enum {
    RBC_SFUL,
    RBC_SFREE
};

struct StressFree_v {
    float *ll;
    float *aa;
};

struct StressFul_v {
    float l0;
    float a0;
};

union SFreeInfo {
    StressFul_v sful;
    StressFree_v sfree;
};

enum {
    RBC_RND0,
    RBC_RND1,
};

struct Rnd0_v {};
struct Rnd1_v {
    int *anti;
    float *rr;
};

union RndInfo {
    Rnd0_v rnd0;
    Rnd1_v rnd1;
};

struct RbcForce {
    RbcRnd *rnd;
    Bending *bending;

    Adj   *adj;
    Adj_v *adj_v;

    int stype;
    SFreeInfo sinfo;
    int rtype;
    RndInfo rinfo;
};

struct RbcForceStressInfo { double l0, a0; };
struct RbcForceRnd0Info {};
struct RbcForceRnd1Info { double r; };
static __device__ RbcForceStressInfo rbc_force_fetch_stress_info(int, StressFul_v sv) {
    RbcForceStressInfo si;
    si.l0 = sv.l0;
    si.a0 = sv.a0;
    return si;
}

static __device__ RbcForceStressInfo rbc_force_fetch_stress_info(int i, StressFree_v sv) {
    RbcForceStressInfo si;
    si.l0 = sv.ll[i];
    si.a0 = sv.aa[i];
    return si;
}

static __device__ RbcForceRnd0Info rbc_force_fetch_rnd_info(int, int, Rnd0_v rv) {
    RbcForceRnd0Info ri;
    return ri;
}

static __device__ RbcForceRnd1Info rbc_force_fetch_rnd_info(int i0, int j, Rnd1_v rv) {

    RbcForceRnd1Info ri;
    int i1;
    i1 = rv.anti[i0];
    if (i1 > i0) j = j - i0 + i1;
    ri.r = rv.rr[j];
    return ri;
}
static __device__ double3 rbc_force_fvolume(const RbcParams_v *par, double3 r2, double3 r3, double v0, double v) {
    double f0;
    double3 f;
    cross(&r3, &r2,      &f);
    f0 = par->kv * (v - v0) / (6 * v0);
    scal(f0,        &f);
    return f;
}

static __device__ double3 rbc_force_farea(const RbcParams_v *par, double3 x21, double3 x31, double3 x32,   double a0, double A0, double A) {
    double3 nn, f;
    double a, f0, fa, fA;
    cross(&x21, &x31,      &nn);
    cross(&nn, &x32,       &f);
    a = 0.5 * sqrt(dot<double>(&nn, &nn));

    fA = - par->ka * (A - A0) / (4 * A0 * a);
    fa = - par->kd * (a - a0) / (4 * a0 * a);
    f0 = fA + fa;
    scal(f0,        &f);
    return f;
}

enum {RBC_FORCE_SPRING_OK, RBC_FORCE_SPRING_LONG};
static __device__ double rbc_force_sq(double x) { return x * x; }
static __device__ double rbc_force_wlc0(double r) { return (4*rbc_force_sq(r)-9*r+6)/(4*rbc_force_sq(r-1)); }
static __device__ double rbc_force_wlc(double lmax, double ks, double r) { return ks/lmax*rbc_force_wlc0(r/lmax); }
static __device__ double3 rbc_force_fspring(const RbcParams_v *par, double3 x21, double l0,      int *pstatus) {
  #define wlc_r(r) (rbc_force_wlc(lmax, ks, r))
    double m, r, fwlc, fpow, lmax, ks, x0;
    *pstatus = RBC_FORCE_SPRING_OK;
    ks = par->ks; m = par->mpow; x0 = par->x0;
    r = sqrtf(dot<double>(&x21, &x21));
    lmax = l0 / x0;
    if (r >= lmax) {
        *pstatus = RBC_FORCE_SPRING_LONG;
        r = RBC_SPRING_CAP * lmax;
    }
    fwlc =   wlc_r(r);
    fpow = - wlc_r(l0) * powf(l0, m + 1) / powf(r, m + 1);
    scal(fwlc + fpow,        &x21);
    return x21;
  #undef wlc_r
}

static __device__ void rbc_force_report_tri(double3 r1, double3 r2, double3 r3) {
    printf("bad triangle: [%g %g %g] [%g %g %g] [%g %g %g]\n",
           r1.x, r1.y, r1.z,   r2.x, r2.y, r2.z,   r3.x, r3.y, r3.z);
    assert(0);
}
static __device__ double3 rbc_force_ftri(const RbcParams_v *par, double3 r1, double3 r2, double3 r3,
                             RbcForceStressInfo si, double area, double volume) {
    int spring_status;
    double3 f, fv, fs;
    double3 x21, x32, x31;

    diff(&r2, &r1,      &x21);
    diff(&r3, &r2,      &x32);
    diff(&r3, &r1,      &x31);

    f = rbc_force_farea(par, x21, x31, x32, si.a0, par->totArea, area);
    fv = rbc_force_fvolume(par, r2, r3, par->totVolume, volume);
    add(&fv,        &f);

    fs = rbc_force_fspring(par, x21, si.l0, &spring_status);
#ifdef RBC_SPRING_FAIL
    if (spring_status != RBC_FORCE_SPRING_OK) rbc_force_report_tri(r1, r2, r3);
#endif
    add(&fs,        &f);
    return f;
}

static __device__ double3 rbc_force_fvisc(const RbcParams_v *par, double3 r1, double3 r2, double3 u1, double3 u2) {
    const double gammaC = par->gammaC, gammaT = par->gammaT;
    double3 du, dr, f = make_double3(0, 0, 0);
    diff(&u2, &u1,      &du);
    diff(&r1, &r2,      &dr);

    double fac = dot<double>(&du, &dr) / dot<double>(&dr, &dr);

    axpy(gammaT      , &du,      &f);
    axpy(gammaC * fac, &dr,      &f);

    return f;
}

static __device__ double3 rbc_force_frnd(double, const RbcParams_v*, double3, double3, RbcForceRnd0Info) {
    double3 f;
    f.x = f.y = f.z = 0;
    return f;
}

static __device__ double  rbc_force_frnd0(double dt, const RbcParams_v *par, double rnd) {
    double f, g, T;
    g = par->gammaC; T = par->kBT;
    f  = sqrtf(2*g*T/dt)*rnd;
    return f;
}

static __device__ double3 rbc_force_frnd(double dt, const RbcParams_v *par, double3 r1, double3 r2, RbcForceRnd1Info rnd) {
    double3 dr;
    double r, f0;
    diff(&r1, &r2,      &dr);
    r = sqrtf(dot<double>(&dr, &dr));
    f0 = rbc_force_frnd0(dt, par, rnd.r);
    scal(f0/r,        &dr);
    return dr;
}
struct RbcForcePart { double3 r, v; };
typedef double3 Pos;

static __device__ Pos rbc_force_fetch_pos(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pos r;
    r.x = pp[i].r[X];
    r.y = pp[i].r[Y];
    r.z = pp[i].r[Z];
    return r;
}

static __device__ RbcForcePart rbc_force_fetch_part(const Particle *pp, int i) {
    enum {X, Y, Z};
    RbcForcePart p;
    const float *r, *v;
    r = pp[i].r; v = pp[i].v;
    p.r.x = r[X]; p.r.y = r[Y]; p.r.z = r[Z];
    p.v.x = v[X]; p.v.y = v[Y]; p.v.z = v[Z];
    return p;
}

template <typename RndInfo>
static __device__ void rbc_force_adj_tris(double dt,
                                const RbcParams_v *par, const Particle *pp,  const RbcForcePart p0, const float *av,
                                const RbcForceStressInfo si, RndInfo ri,
                                AdjMap *m,        double f[3]) {
    enum {X, Y, Z};
    double3 f0;
    int i1, i2, rbc;
    double area, volume;
    i1 = m->i1; i2 = m->i2; rbc = m->rbc;

    const RbcForcePart p1 = rbc_force_fetch_part(pp, i1);
    const Pos  r2 = rbc_force_fetch_pos(pp,  i2);

    area = av[2*rbc]; volume = av[2 * rbc + 1];
    f0 = rbc_force_ftri(par, p0.r, p1.r, r2, si, area, volume);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;

    f0 = rbc_force_fvisc(par, p0.r, p1.r, p0.v, p1.v);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;

    f0 = rbc_force_frnd(dt, par, p0.r, p1.r, ri);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;
}

template <typename Stress_v, typename Rnd_v>
__global__ void rbc_force_kernel(double dt,
                      RbcParams_v par, int md, int nv, int nc, const Particle *pp,
                      Adj_v adj,
                      Stress_v sv, Rnd_v rv,
                      const float *av,      float *ff) {
    enum {X, Y, Z};
    int i, pid, valid;
    double f[3];
    AdjMap m;
    RbcForceStressInfo si;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    pid = i / md;

    if (pid >= nc * nv) return;
    valid = adj_get_map(i, &adj,      &m);
    if (!valid) return;
    si = rbc_force_fetch_stress_info(i % (md * nv), sv);
    auto ri = rbc_force_fetch_rnd_info(i % (md * nv), i, rv);

    const RbcForcePart p0 = rbc_force_fetch_part(pp, m.i0);

    f[X] = f[Y] = f[Z] = 0;
    rbc_force_adj_tris(dt, &par, pp, p0, av, si, ri, &m,        f);

    atomicAdd(&ff[3 * pid + 0], f[X]);
    atomicAdd(&ff[3 * pid + 1], f[Y]);
    atomicAdd(&ff[3 * pid + 2], f[Z]);
}
static bool is_stress_free(const RbcForce *f) {
    return f->stype == RBC_SFREE;
}

static bool is_rnd(const RbcForce *f) {
    return f->rtype == RBC_RND1;
}

void rbc_force_ini(const MeshRead *cell, RbcForce **pq) {
    RbcForce *q;
    int md, nt, nv;
    const int4 *tt;
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    md = mesh_read_get_md(cell);
    tt = mesh_read_get_tri(cell);

    UC(adj_ini(md, nt, nv, tt,      &q->adj));
    UC(adj_view_ini(q->adj,      &q->adj_v));

    *pq = q;
}

static void fin_rnd(RbcRnd *rnd) {
    rbc_rnd_fin(rnd);
}

static void fin_stress(RbcForce *f) {
    if (is_stress_free(f)) {
        StressFree_v v = f->sinfo.sfree;
        Dfree(v.ll);
        Dfree(v.aa);
    }
}

void rbc_force_fin(RbcForce *q) {
    if (is_rnd(q)) {
        Rnd1_v v = q->rinfo.rnd1;
        Dfree(v.anti);
        UC(fin_rnd(q->rnd));
    }
    UC(fin_stress(q));
    UC(adj_fin(q->adj));
    UC(adj_view_fin(q->adj_v));
    UC(bending_fin(q->bending));
    EFREE(q);
}

void rbc_force_set_stressful(int nt, float totArea,      RbcForce *f) {
    StressFul_v v;
    float a0 = totArea / nt;

    v.a0 = a0;
    v.l0 = sqrt(a0 * 4.0 / sqrt(3.0));

    f->stype = RBC_SFUL;
    f->sinfo.sful = v;
}

void rbc_force_set_stressfree(const char *fname,      RbcForce *f) {
    StressFree_v v;
    MeshRead *cell;
    const Adj *adj;
    RbcShape *shape;
    const float *rr;
    float *ll_hst, *aa_hst;
    int n;

    adj = f->adj;
    UC(mesh_read_ini_off(fname, &cell));
    rr = mesh_read_get_vert(cell);

    UC(rbc_shape_ini(adj, rr,      &shape));

    n = adj_get_max(adj);

    rbc_shape_edg(shape, &ll_hst);
    rbc_shape_area(shape, &aa_hst);

    Dalloc(&v.ll, n);
    Dalloc(&v.aa, n);

    cH2D(v.ll, ll_hst, n);
    cH2D(v.aa, aa_hst, n);

    UC(rbc_shape_fin(shape));
    UC(mesh_read_fin(cell));

    f->stype = RBC_SFREE;
    f->sinfo.sfree = v;
}

void rbc_force_set_rnd0(RbcForce *f) {
    Rnd0_v rnd0;
    f->rtype = RBC_RND0;
    f->rinfo.rnd0 = rnd0;
}

void rbc_force_set_rnd1(int seed, RbcForce *f) {
    Rnd1_v rnd1;
    int n;
    int *hst;
    Adj *adj = f->adj;

    n = adj_get_max(adj);
    Dalloc(&rnd1.anti, n);

    EMALLOC(n, &hst);
    adj_get_anti(adj,      hst);
    cH2D(rnd1.anti, hst, n);
    EFREE(hst);

    UC(rbc_rnd_ini(n * MAX_CELL_NUM, seed, &f->rnd));

    f->rtype = RBC_RND1;
    f->rinfo.rnd1 = rnd1;
}

void rbc_force_set_bending(const MeshRead *cell, const char *type, RbcForce *q) {
    if (same_str(type, "kantor"))
        UC(bending_kantor_ini(cell,      &q->bending));
    else if (same_str(type, "juelicher"))
        UC(bending_juelicher_ini(cell,      &q->bending));
    else if (same_str(type, "none"))
        UC(bending_none_ini(cell,      &q->bending));
    else
        ERR("unknown bending type '%s'", type);
    msg_print("bending: %s", type);
}
static void get_stress_view(const RbcForce *f,      StressFul_v *v) {
    *v = f->sinfo.sful;
}

static void get_stress_view(const RbcForce *f,      StressFree_v *v) {
    *v = f->sinfo.sfree;
}

static void get_rnd_view(const RbcForce *f,      Rnd0_v *v) {
    *v = f->rinfo.rnd0;
}

static void get_rnd_view(const RbcForce *f,      Rnd1_v *v) {
    *v = f->rinfo.rnd1;
}

template <typename Stress_v, typename Rnd_v>
static void apply(float dt, RbcParams_v parv, int nc, int nv, int md,
                  const Particle *pp,
                  const Adj_v *adj_v, Stress_v sv, Rnd_v rv,
                  float *av,      Force *ff) {
    if (!d::is_device_pointer(ff))  ERR("`ff` is not a device pointer");
    KL(rbc_force_kernel, (k_cnf(nc*nv*md)), (dt, parv, md, nv, nc, pp,
                                                 *adj_v, sv, rv, av,      (float*)ff));
}

template <typename Stress_v>
static void dispatch_rnd(float dt, RbcParams_v parv, int nc, int nv, int md,
                         const Particle *pp, RbcRnd *rnd,
                         const Adj_v *adj_v, RbcForce *t, Stress_v sv,
                         float *av,      Force *ff){
    if (is_rnd(t)) {
        Rnd1_v rv;
        get_rnd_view(t, &rv);
        rbc_rnd_gen(rnd, nc * md * nv,      &rv.rr);
        apply(dt, parv, nc, nv, md, pp, adj_v, sv, rv, av,      ff);
    }
    else {
        Rnd0_v rv;
        get_rnd_view(t, &rv);
        apply(dt, parv, nc, nv, md, pp, adj_v, sv, rv, av,      ff);
    }
}

void rbc_force_apply(RbcForce *t, const RbcParams *par, float dt, const RbcQuants *q,      Force *ff) {
    RbcParams_v parv;
    float *av;
    if (q->nc <= 0) return;
    if (!d::is_device_pointer(q->pp))
        ERR("`q->pp` is not a device pointer");

    parv = rbc_params_get_view(par);
    UC(area_volume_compute(q->area_volume, q->nc, q->pp,      &av));
    UC(bending_apply(t->bending, par, q,        ff));

    if (is_stress_free(t)) {
        StressFree_v si;
        get_stress_view(t, &si);
        dispatch_rnd(dt, parv, q->nc, q->nv, q->md, q->pp, t->rnd,
                     t->adj_v, t, si, av,      ff);
    }
    else {
        StressFul_v si;
        get_stress_view(t, &si);
        dispatch_rnd(dt, parv, q->nc, q->nv, q->md, q->pp, t->rnd,
                     t->adj_v, t, si, av,      ff);
    }
}
void rbc_force_stat(float *pArea, float *pVolume) {
    *pArea = 1.0; *pVolume = 1.0;
}

struct HeRead;
struct He;
int he_read_int4_ini(int, int, const int4 *, HeRead **);
int he_read_fin(HeRead *);
int he_ini(HeRead *, He **);
int he_fin(He *);
int he_ne(He *);
int he_nxt(He *, int);
int he_flp(He *, int);
int he_ver(He *, int);
int he_hdg_edg(He *, int);

#define UDX_RBC_BENDING_COMMON

#include <stdio.h>
#include <assert.h>

struct RbcParams;
struct Juelicher;
struct Force;
struct MeshRead;
struct RbcQuants;

void juelicher_ini(const MeshRead*, Juelicher**);
void juelicher_fin(Juelicher*);

void juelicher_apply(Juelicher*, const RbcParams*, const RbcQuants*,      Force*);
struct Juelicher {
    int4 *tri, *dih;

    float *area, *lentheta, *theta;
    float *lentheta_tot, *area_tot, *curva_mean_area_tot;
    float *f, *fad;

    int ne;
};

#ifndef UDX_RBC_KANTOR0_COMMON
#define UDX_RBC_KANTOR0_COMMON
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace force_kantor0_dev {
  #define END }

BEGIN

_S_ double rsqrt0(double x) { return rsqrt(x); }
#define PRINT(fmt, ...) printf((fmt), ##__VA_ARGS__)
#define EXIT() assert(0)

_S_ double max(double a, double b) { return a > b ? a : b; }

enum {ANGLE_OK, ANGLE_BAD_A, ANGLE_BAD_B};
_S_ int angle(double3 a, double3 b,      double *pcos, double *pover_sin,
              double *pover_a, double *pover_b) {
    double  cost, over_sin, over_a, over_b;
    double aa, bb, ab;
    *pcos = *pover_sin = *pover_a = *pover_b = 0.0;

    aa = dot<double>(&a, &a);
    bb = dot<double>(&b, &b);
    ab = dot<double>(&a, &b);

    if (aa == 0) return ANGLE_BAD_A;
    if (bb == 0) return ANGLE_BAD_B;

    over_a = rsqrt0(aa);
    over_b = rsqrt0(bb);
    cost = ab * over_a * over_b;
    over_sin = rsqrt0(max(1.0 - cost*cost, 1e-6));

    *pcos = cost; *pover_sin = over_sin; *pover_a = over_a; *pover_b = over_b;
    return ANGLE_OK;
}

_S_ void dih_write(double3 a, double3 b, double3 c, double3 d) {
    PRINT("%g %g %g\n", a.x, a.y, a.z);
    PRINT("%g %g %g\n", b.x, b.y, b.z);
    PRINT("%g %g %g\n", c.x, c.y, c.z);
    PRINT("%g %g %g\n", d.x, d.y, d.z);
}

template <int update>
_S_ double3 dih0(double phi, double kb,
                 double3 a, double3 b, double3 c, double3 d) {
    int status;
    double over_k, over_n, cosTheta, over_sin,
        beta, b11, b12, sint0kb, cost0kb;
    double3 ab, ac, cd, bc, bd, da, k, n, k_n;
    diff(&a, &b,      &ab);
    diff(&a, &c,      &ac);
    diff(&c, &d,      &cd);
    diff(&b, &d,      &bd);
    diff(&b, &c,      &bc);
    diff(&d, &a,      &da);

    cross(&ab, &ac,      &k);
    cross(&bd, &bc,      &n);

    status = angle(k, n,
                   &cosTheta, &over_sin,
                   &over_k, &over_n);
    if (status != ANGLE_OK) {
        PRINT("bad dihedral\n");
        dih_write(a, b, c, d);
        EXIT();
    }
    diff(&k, &n,      &k_n);
    over_sin = copysign(over_sin, dot<double>(&k_n, &da));

    sint0kb = sin(phi) * kb;
    cost0kb = cos(phi) * kb;
    beta = cost0kb - cosTheta * sint0kb * over_sin;

    b11 = -beta *  cosTheta * over_k * over_k;
    b12 =  beta *             over_k * over_n;

    if (update == 1) {
        double3 r32, f1, f;
        diff(&c, &b,      &r32);
        cross(&k, &r32,      &f);
        cross(&n, &r32,      &f1);
        scal(b11,      &f);
        axpy(b12, &f1,      &f);
        return f;
    }
    else if (update == 2) {
        double3 f, f1, f2, f3;
        double b22 = -beta * cosTheta * over_n * over_n;

        cross(&k, &ac,      &f);
        cross(&k, &cd,      &f1);
        cross(&n, &ac,      &f2);
        cross(&n, &cd,      &f3);

        scal(b11,      &f);
        add(&f2,      &f1);
        axpy(b12, &f1,      &f);
        axpy(b22, &f3,      &f);
        return f;
    }
    else {
        double3 f;
        f.x = f.y = f.z = 0;
        assert(0);
        return f;
    }
}

_I_ double3 dih_a(double phi, double kb, double3 a, double3 b, double3 c, double3 d) {
    return dih0<1>(phi, kb, a, b, c, d);
}

_I_ double3 dih_b(double phi, double kb, double3 a, double3 b, double3 c, double3 d) {
    return dih0<2>(phi, kb, a, b, c, d);
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
#endif

enum {X, Y, Z};
static __device__  int small(double s) {
    const double eps = 1e-12;
    if      (s >  eps) return 0;
    else if (s < -eps) return 0;
    else               return 1;
}
static __device__ void juelicher_append(double x, int i, float *a) { atomicAdd(&a[i], x); }
static __device__ double juelicher_tri_area(const double a[3], const double b[3], const double c[3]) { return tri_dev::kahan_area(a, b, c); }
static __device__ double juelicher_tri_dih(const double a[3], const double b[3], const double c[3], const double d[3]) {
    double x, y;
    tri_dev::dihedral_xy(a, b, c, d,      &x, &y);
    return -atan2(y, x);
}
static __device__ void juelicher_vec_scalar(const double a[3], double s,      double b[3]) {
    b[X] = s*a[X]; b[Y] = s*a[Y]; b[Z] = s*a[Z];
}
static __device__ void juelicher_vec_minus(const double a[3], const double b[3],      double c[3]) {
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
static __device__ double juelicher_vec_dot(const double a[3], const double b[3]) {
   return a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z];
}
static __device__ void juelicher_vec_cross(const double a[3], const double b[3],      double c[3]) {
   c[X] = a[Y]*b[Z]-b[Y]*a[Z];
   c[Y] = b[X]*a[Z]-a[X]*b[Z];
   c[Z] = a[X]*b[Y]-b[X]*a[Y];
}
static __device__ double juelicher_vec_abs(const double a[3]) { return sqrt(juelicher_vec_dot(a, a)); }

static __device__ void juelicher_vec_copy(const double a[3], double b[3]) {
    b[X] = a[X]; b[Y] = a[Y]; b[Z] = a[Z];
}

static __device__ double juelicher_edg_abs(const double a[3], const double b[3]) {
    double u[3];
    juelicher_vec_minus(b, a, u);
    return juelicher_vec_abs(u);
}

static __device__ double juelicher_tri_cot(const double a[3], const double b[3], const double c[3]) {
    double x, y, u[3], v[3];
    y = 2 * juelicher_tri_area(a, b, c);
    juelicher_vec_minus(a, b, u);
    juelicher_vec_minus(c, b, v);
    x = juelicher_vec_dot(u, v);
    return x/y;
}

static __device__ void juelicher_vec_linear_combination(double al, const double a[3], double be, const double b[3],
                                 double c[3]) {
    c[X] = al*a[X] + be*b[X];
    c[Y] = al*a[Y] + be*b[Y];
    c[Z] = al*a[Z] + be*b[Z];
}

static __device__ void juelicher_vec_negative(const double a[3],      double b[3]) {
    b[X] = -a[X]; b[Y] = -a[Y]; b[Z] = -a[Z];
}
static __device__ void juelicher_vec_norm(const double a[3],      double b[3]) {
    double s;
    s = juelicher_vec_abs(a);
    if (!small(s)) juelicher_vec_scalar(a, 1/s,      b);
    else juelicher_vec_copy(a, b);
}

static __device__ void juelicher_dedg_abs(double a[3], double b[3],      double da[3], double db[3]) {
    double u[3], e[3];
    juelicher_vec_minus(b, a,   u);
    juelicher_vec_norm(u,   e);
    juelicher_vec_copy(e,     db);
    juelicher_vec_negative(e, da);
}

static __device__ void juelicher_vec_scalar_append(const double a[3], double s, int i,
                                              float *f) {
    juelicher_append(s*a[X], 3*i,     f);
    juelicher_append(s*a[Y], 3*i + 1, f);
    juelicher_append(s*a[Z], 3*i + 2, f);
}

static __device__ void juelicher_tri_normal(const double a[3], const double b[3], const double c[3],      double e[3]) {
    double u[3], v[3], n[3];
    juelicher_vec_minus(b, a,   u);
    juelicher_vec_minus(c, a,   v);
    juelicher_vec_cross(u, v,   n);
    juelicher_vec_norm(n,   e);
}

static __device__ void juelicher_ddih_angle(const double a[3], const double b[3], const double c[3], const double d[3],
                                  double da[3], double db[3], double dc[3], double dd[3]) {
    double n[3], k[3];
    double e, An, Ak, cn, bn, bk, ck;

    juelicher_tri_normal(a, b, c,   n);
    juelicher_tri_normal(c, b, d,   k);

    e = juelicher_edg_abs(c, b);

    An = juelicher_tri_area(a, b, c);
    Ak = juelicher_tri_area(c, b, d);

    juelicher_vec_scalar(n, e/(2*An),   da);
    juelicher_vec_scalar(k, e/(2*Ak),   dd);

    cn = juelicher_tri_cot(b, c, a);
    bn = juelicher_tri_cot(a, b, c);

    bk = juelicher_tri_cot(c, b, d);
    ck = juelicher_tri_cot(d, c, b);

    juelicher_vec_linear_combination(-bn/e, n, -bk/e, k,    dc);
    juelicher_vec_linear_combination(-cn/e, n, -ck/e, k,    db);
}

static __device__ void juelicher_tri_edg(const double a[3], const double b[3], const double c[3],
                                    double ab[3], double bc[3], double ca[3]) {
    juelicher_vec_minus(b, a,   ab);
    juelicher_vec_minus(c, b,   bc);
    juelicher_vec_minus(a, c,   ca);
}

static __device__ void juelicher_dtri_area(double a[3], double b[3], double c[3],       double da[3], double db[3], double dc[3]) {
    double n[3], n2[3], ab[3], bc[3], ca[3];
    juelicher_tri_normal(a, b, c,   n);
    juelicher_vec_scalar(n, 0.5,   n2);
    juelicher_tri_edg(a, b, c,   ab, bc, ca);
    juelicher_vec_cross(n2, bc,   da);
    juelicher_vec_cross(n2, ca,   db);
    juelicher_vec_cross(n2, ab,   dc);
}

static __device__ void juelicher_get(const Particle *pp, int i,      double r[3]) {
    r[X] = pp[i].r[X];
    r[Y] = pp[i].r[Y];
    r[Z] = pp[i].r[Z];
}
static __device__ void juelicher_get2(const Particle *pp, int i, int j,
                                 double a[3], double b[3]) {
    juelicher_get(pp, i,      a);
    juelicher_get(pp, j,      b);
}
static __device__ void juelicher_get3(const Particle *pp, int i, int j, int k,
                                 double a[3], double b[3], double c[3]) {
    juelicher_get(pp, i,      a);
    juelicher_get(pp, j,      b);
    juelicher_get(pp, k,      c);
}
static __device__ void juelicher_get4(const Particle *pp, int i, int j, int k, int l,
                                 double a[3], double b[3], double c[3], double d[3]) {
    juelicher_get(pp, i,      a);
    juelicher_get(pp, j,      b);
    juelicher_get(pp, k,      c);
    juelicher_get(pp, l,      d);
}

__global__ void juelicher_sum_kernel(int nv, const float *from,      float *to) {
    int i, c;
    float s;
    i  = threadIdx.x + blockIdx.x * blockDim.x;
    c  = blockIdx.y;
    if (i < nv) s = from[c*nv + i]; else s = 0;
    s = warpReduceSum(s);
    if ((threadIdx.x % warpSize) == 0) atomicAdd(&to[c], s);
}

static __device__ void juelicher_compute_area0(const Particle *pp, int4 tri,      float *area) {
    double area0;
    double a[3], b[3], c[3];
    int i, j, k;
    i = tri.x; j = tri.y; k = tri.z;
    juelicher_get3(pp, i, j, k,      a, b, c);
    area0 = juelicher_tri_area(a, b, c)/3;

    juelicher_append(area0, i, area);
    juelicher_append(area0, j, area);
    juelicher_append(area0, k, area);
}

__global__ void juelicher_compute_area_kernel(int nv, int nt, int nc,
                             const Particle *pp, const int4 *tri,
                                  float *area) {
    int i;
    int t, c;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nt*nc) return;

    c = i / nt;
    t = i % nt;

    pp   += nv*c;
    area += nv*c;
    tri  += t;

    juelicher_compute_area0(pp, *tri,      area);
}

static __device__ void juelicher_compute_theta_len0(const Particle *pp, int4 dih,
                                               float *theta, float *lentheta) {
    int i, j, k, l;
    double a[3], b[3], c[3], d[3], u[3];
    double len0, theta0, lentheta0;

    i = dih.x; j = dih.y; k = dih.z; l = dih.w;
    juelicher_get4(pp, i, j, k, l,      a, b, c, d);

    *theta = theta0 = juelicher_tri_dih(a, b, c, d);
    juelicher_vec_minus(c, b, u);
    len0 = juelicher_vec_abs(u);
    lentheta0 = len0*theta0;

    juelicher_append(lentheta0, j,  lentheta);
    juelicher_append(lentheta0, k,  lentheta);
}

__global__ void juelicher_compute_theta_len_kernel(int nv, int ne, int nc,
                                  const Particle *pp, const int4 *dih,
                                       float *theta, float *lentheta) {
    int i;
    int e, c;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= ne*nc) return;

    c = i / ne;
    e = i % ne;

    pp       += nv*c;
    theta    += ne*c + e;
    lentheta += nv*c;
    dih      += e;

    juelicher_compute_theta_len0(pp, *dih,      theta, lentheta);
}

__global__ void juelicher_compute_mean_curv_kernel(int nc, float H0, float kad,
                                  const float *lentheta, const float *area,
                                       float *mean) {
    int i;
    float pi;
    pi = 3.141592653589793;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nc) return;
    mean[i] = (lentheta[i]/4 - H0 * area[i])*(4*kad*pi/area[i]);
}

static __device__ void juelicher_force_edg0(float H0, float curva_mean_area_tot,
                                  const Particle *pp, const int4 dih,
                                  float theta, const float *lentheta, const float *area,
                                       float *f, float *fad) {
    int j, k;
    double b[3], c[3], db[3], dc[3];
    double coef;

    j = dih.y; k = dih.z;
    juelicher_get2(pp, j, k,      b, c);
    juelicher_dedg_abs(b, c, db, dc);

    coef = - ( (lentheta[j]/area[j]/4 - H0) + (lentheta[k]/area[k]/4 - H0) ) * theta;
    juelicher_vec_scalar_append(db, coef, j, f);
    juelicher_vec_scalar_append(dc, coef, k, f);
    coef = -curva_mean_area_tot/4 * theta;
    juelicher_vec_scalar_append(db, coef, j, fad);
    juelicher_vec_scalar_append(dc, coef, k, fad);
}

__global__ void juelicher_force_edg_kernel(int nv, int ne, int nc, float H0,
                          const Particle *pp, const int4 *dih,
                          const float *curva_mean_area_tot,
                          const float *theta, const float *lentheta, const float *area,
                               float *f, float *fad) {
    int i;
    int e, c;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= ne*nc) return;

    c = i / ne;
    e = i % ne;

    pp       += nv*c;
    dih      += e;

    curva_mean_area_tot += c;
    theta    += ne*c + e;
    lentheta += nv*c;
    area     += nv*c;

    f        += 3*nv*c;
    fad      += 3*nv*c;

    juelicher_force_edg0(H0, *curva_mean_area_tot,
               pp, *dih,
               *theta, lentheta, area,      f, fad);
}

static __device__ void juelicher_force_lentheta0(float H0, float curva_mean_area_tot,
                                       const Particle *pp, const int4 dih,
                                       const float *lentheta, const float *area,
                                            float *f, float *fad) {
    int i, j, k, l;
    double len0, coef;
    double a[3], b[3], c[3], d[3], u[3];
    double da[3], db[3], dc[3], dd[3];

    i = dih.x; j = dih.y; k = dih.z; l = dih.w;
    juelicher_get4(pp, i, j, k, l,      a, b, c, d);
    juelicher_ddih_angle(a, b, c, d,      da, db, dc, dd);
    juelicher_vec_minus(c, b, u);
    len0 = juelicher_vec_abs(u);

    coef =  -(  (lentheta[j]/area[j]/4 - H0) + (lentheta[k]/area[k]/4 - H0) ) * len0 ;

    juelicher_vec_scalar_append(da, coef, i, f);
    juelicher_vec_scalar_append(db, coef, j, f);
    juelicher_vec_scalar_append(dc, coef, k, f);
    juelicher_vec_scalar_append(dd, coef, l, f);

    coef = -curva_mean_area_tot/2.0*len0;
    juelicher_vec_scalar_append(da, coef, i, fad);
    juelicher_vec_scalar_append(db, coef, j, fad);
    juelicher_vec_scalar_append(dc, coef, k, fad);
    juelicher_vec_scalar_append(dd, coef, l, fad);
}

__global__ void juelicher_force_lentheta_kernel(int nv, int ne, int nc, float H0,
                               const Particle *pp, const int4 *dih,
                               const float *curva_mean_area_tot,
                               const float *lentheta, const float *area,
                                    float *f, float *fad) {
    int i;
    int e, c;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= ne*nc) return;

    c = i / ne;
    e = i % ne;

    pp       += nv*c;
    dih      += e;

    curva_mean_area_tot += c;
    lentheta += nv*c;
    area     += nv*c;

    f        += 3*nv*c;
    fad      += 3*nv*c;

    juelicher_force_lentheta0(H0, *curva_mean_area_tot,
                    pp, *dih,
                    lentheta, area,      f, fad);
}

static __device__ void juelicher_force_area0(float H0, const Particle *pp, const int4 tri,
                                   const float *lentheta, const float *area,
                                        float *f) {
    int i, j, k;
    double a[3], b[3], c[3];
    double da[3], db[3], dc[3];
    double coef1, coef2, coef;

    i = tri.x; j = tri.y; k = tri.z;
    juelicher_get3(pp, i, j, k,      a, b, c);

    juelicher_dtri_area(a, b, c, da, db, dc);

    coef1 = 1.0/3.0;

    coef2 = lentheta[i]*lentheta[i]/8.0/area[i]/area[i] - 2.0*H0*H0;
    coef = coef1 * coef2;
    juelicher_vec_scalar_append(da, coef, i, f);

    coef2 = lentheta[j]*lentheta[j]/8.0/area[j]/area[j] - 2.0*H0*H0;
    coef = coef1 * coef2;
    juelicher_vec_scalar_append(db, coef, j, f);

    coef2 = lentheta[k]*lentheta[k]/8.0/area[k]/area[k] - 2.0*H0*H0;
    coef = coef1 * coef2;
    juelicher_vec_scalar_append(dc, coef, k, f);
}

__global__ void juelicher_force_area_kernel(int nv, int nt, int nc, float H0,
                           const Particle *pp, const int4 *tri,
                           const float *lentheta, const float *area,
                                float *f) {
    int i;
    int t, c;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nt*nc) return;

    c = i / nt;
    t = i % nt;

    pp       += nv*c;
    tri      += t;

    lentheta += nv*c;
    area     += nv*c;

    f        += 3*nv*c;

    juelicher_force_area0(H0, pp, *tri,
                lentheta, area,      f);
}

__global__ void juelicher_accumulate_kernel(int n,
                           const float *ff0, const float *ff1,
                                Force *ff) {
    enum {X, Y, Z};
    int i;
    Force *f;
    const float *f0, *f1;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    f0 = &ff0[3*i];
    f1 = &ff1[3*i];
    f = &ff[i];

    f->f[X] += f0[X] + f1[X];
    f->f[Y] += f0[Y] + f1[Y];
    f->f[Z] += f0[Z] + f1[Z];
}
void juelicher_ini(const MeshRead *cell, Juelicher **pq) {
#   define  nxt(h)     he_nxt(he, (h))
#   define  flp(h)     he_flp(he, (h))
#   define  ver(h)     he_ver(he, (h))
#   define  hdg_edg(e) he_hdg_edg(he, (e))
    Juelicher *q;
    int nt, nv, ne, e, i, j, k, l, h, n, nn, nnf;
    const int4 *tri;
    HeRead *he_read;
    He *he;
    int4 *dih;

    EMALLOC(1, &q);
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    tri = mesh_read_get_tri(cell);

    he_read_int4_ini(nv, nt, tri,      &he_read);
    he_ini(he_read,      &he);

    ne = he_ne(he);
    EMALLOC(ne, &dih);
    Dalloc(&q->dih, ne);
    Dalloc(&q->tri, nt);

    Dalloc(&q->theta,    MAX_CELL_NUM*ne);
    Dalloc(&q->area,     MAX_CELL_NUM*nv);
    Dalloc(&q->lentheta, MAX_CELL_NUM*nv);

    Dalloc(&q->lentheta_tot, MAX_CELL_NUM);
    Dalloc(&q->area_tot,     MAX_CELL_NUM);
    Dalloc(&q->curva_mean_area_tot,     MAX_CELL_NUM);
    Dalloc(&q->f,   3*MAX_CELL_NUM*nv);
    Dalloc(&q->fad, 3*MAX_CELL_NUM*nv);

    for (e = 0; e < ne; e++) {
        h = hdg_edg(e); n = nxt(h); nn = nxt(nxt(h));
        j = ver(h); k = ver(n); i = ver(nn);
        nnf = nxt(nxt(flp(h)));
        l = ver(nnf);
        dih[e].x = i; dih[e].y = j; dih[e].z = k; dih[e].w = l;
    }

    q->ne = ne;
    cH2D(q->dih, dih, ne);
    cH2D(q->tri, tri,  nt);

    he_read_fin(he_read);
    he_fin(he);
    EFREE(dih);

    *pq = q;
}

void juelicher_fin(Juelicher *q) {
    Dfree(q->dih);
    Dfree(q->tri);
    Dfree(q->area);
    Dfree(q->theta);
    Dfree(q->lentheta);
    Dfree(q->lentheta_tot);
    Dfree(q->area_tot);
    Dfree(q->curva_mean_area_tot);
    Dfree(q->f);
    Dfree(q->fad);
    EFREE(q);
}

static void sum(int nv, int nc, const float *from,      float *to) {
    dim3 thrd(128, 1);
    dim3 blck(ceiln(nv, thrd.x), nc);
    Dzero(to, nc);
    KL(juelicher_sum_kernel, (blck, thrd), (nv, from,      to));
}

void juelicher_apply(Juelicher *q, const RbcParams*, const RbcQuants *quants,      Force *ff) {
    int nc, ne, nt, nv;

    int4 *tri, *dih;
    float *area, *lentheta, *theta;
    float *lentheta_tot, *area_tot, *curva_mean_area_tot;
    float *f, *fad;
    const Particle *pp;
    float pi, kb, H0, kad;
    pi = 3.141592653589793;

    H0 = -1.0/2.0;
    kb = 1.0;
    kad = 2*kb/pi;

    tri = q->tri; dih = q->dih;
    area = q->area; lentheta = q->lentheta; theta = q->theta;
    lentheta_tot = q->lentheta_tot;
    area_tot = q->area_tot;
    curva_mean_area_tot = q->curva_mean_area_tot;
    f = q->f;
    fad = q->fad;
    pp = quants->pp;

    nv = quants->nv; nc = quants->nc; nt = quants->nt;
    ne = q->ne;

    if (!d::is_device_pointer(pp))
        ERR("`q->pp` is not a device pointer");
    if (quants->nc <= 0) return;

    Dzero(area, nv*nc);
    KL(juelicher_compute_area_kernel, (k_cnf(nt*nc)), (nv, nt, nc, pp, tri,      area));
    dSync();
    sum(nv, nc, area,      area_tot); dSync();

    Dzero(theta, ne*nc);
    Dzero(lentheta, nv*nc);
    KL(juelicher_compute_theta_len_kernel, (k_cnf(ne*nc)), (nv, ne, nc, pp, dih,      theta, lentheta));
    dSync();

    sum(nv, nc, lentheta,      lentheta_tot); dSync();
    KL(juelicher_compute_mean_curv_kernel, (k_cnf(nc)), (nc, H0, kad, lentheta_tot, area_tot,      curva_mean_area_tot));
    dSync();

    Dzero(f, nv*nc);
    Dzero(fad, nv*nc);
    KL(juelicher_force_edg_kernel, (k_cnf(ne*nc)),
       (nv, ne, nc, H0, pp, dih, curva_mean_area_tot, theta, lentheta, area,      f, fad));
    dSync();

    KL(juelicher_force_lentheta_kernel, (k_cnf(ne*nc)),
       (nv, ne, nc, H0, pp, dih, curva_mean_area_tot, lentheta, area,      f, fad));
    dSync();

    KL(juelicher_force_area_kernel, (k_cnf(ne*nc)),
       (nv, nt, nc, H0, pp, tri, lentheta, area,      f));
    dSync();

    KL(juelicher_accumulate_kernel, (k_cnf(nv*nc)),
       (nv*nc, f, fad,      ff));

}

#include <stdio.h>
#include <assert.h>

struct RbcParams;
struct Kantor;
struct Force;
struct MeshRead;
struct RbcQuants;

void kantor_ini(const MeshRead*, Kantor**);
void kantor_fin(Kantor*);

void kantor_apply(Kantor*, const RbcParams*, const RbcQuants*,      Force*);
struct Kantor {
    Adj   *adj;
    Adj_v *adj_v;
};

#ifndef UDX_RBC_KANTOR0_COMMON
#define UDX_RBC_KANTOR0_COMMON
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace force_kantor0_dev {
  #define END }

BEGIN

_S_ double rsqrt0(double x) { return rsqrt(x); }
#define PRINT(fmt, ...) printf((fmt), ##__VA_ARGS__)
#define EXIT() assert(0)

_S_ double max(double a, double b) { return a > b ? a : b; }

enum {ANGLE_OK, ANGLE_BAD_A, ANGLE_BAD_B};
_S_ int angle(double3 a, double3 b,      double *pcos, double *pover_sin,
              double *pover_a, double *pover_b) {
    double  cost, over_sin, over_a, over_b;
    double aa, bb, ab;
    *pcos = *pover_sin = *pover_a = *pover_b = 0.0;

    aa = dot<double>(&a, &a);
    bb = dot<double>(&b, &b);
    ab = dot<double>(&a, &b);

    if (aa == 0) return ANGLE_BAD_A;
    if (bb == 0) return ANGLE_BAD_B;

    over_a = rsqrt0(aa);
    over_b = rsqrt0(bb);
    cost = ab * over_a * over_b;
    over_sin = rsqrt0(max(1.0 - cost*cost, 1e-6));

    *pcos = cost; *pover_sin = over_sin; *pover_a = over_a; *pover_b = over_b;
    return ANGLE_OK;
}

_S_ void dih_write(double3 a, double3 b, double3 c, double3 d) {
    PRINT("%g %g %g\n", a.x, a.y, a.z);
    PRINT("%g %g %g\n", b.x, b.y, b.z);
    PRINT("%g %g %g\n", c.x, c.y, c.z);
    PRINT("%g %g %g\n", d.x, d.y, d.z);
}

template <int update>
_S_ double3 dih0(double phi, double kb,
                 double3 a, double3 b, double3 c, double3 d) {
    int status;
    double over_k, over_n, cosTheta, over_sin,
        beta, b11, b12, sint0kb, cost0kb;
    double3 ab, ac, cd, bc, bd, da, k, n, k_n;
    diff(&a, &b,      &ab);
    diff(&a, &c,      &ac);
    diff(&c, &d,      &cd);
    diff(&b, &d,      &bd);
    diff(&b, &c,      &bc);
    diff(&d, &a,      &da);

    cross(&ab, &ac,      &k);
    cross(&bd, &bc,      &n);

    status = angle(k, n,
                   &cosTheta, &over_sin,
                   &over_k, &over_n);
    if (status != ANGLE_OK) {
        PRINT("bad dihedral\n");
        dih_write(a, b, c, d);
        EXIT();
    }
    diff(&k, &n,      &k_n);
    over_sin = copysign(over_sin, dot<double>(&k_n, &da));

    sint0kb = sin(phi) * kb;
    cost0kb = cos(phi) * kb;
    beta = cost0kb - cosTheta * sint0kb * over_sin;

    b11 = -beta *  cosTheta * over_k * over_k;
    b12 =  beta *             over_k * over_n;

    if (update == 1) {
        double3 r32, f1, f;
        diff(&c, &b,      &r32);
        cross(&k, &r32,      &f);
        cross(&n, &r32,      &f1);
        scal(b11,      &f);
        axpy(b12, &f1,      &f);
        return f;
    }
    else if (update == 2) {
        double3 f, f1, f2, f3;
        double b22 = -beta * cosTheta * over_n * over_n;

        cross(&k, &ac,      &f);
        cross(&k, &cd,      &f1);
        cross(&n, &ac,      &f2);
        cross(&n, &cd,      &f3);

        scal(b11,      &f);
        add(&f2,      &f1);
        axpy(b12, &f1,      &f);
        axpy(b22, &f3,      &f);
        return f;
    }
    else {
        double3 f;
        f.x = f.y = f.z = 0;
        assert(0);
        return f;
    }
}

_I_ double3 dih_a(double phi, double kb, double3 a, double3 b, double3 c, double3 d) {
    return dih0<1>(phi, kb, a, b, c, d);
}

_I_ double3 dih_b(double phi, double kb, double3 a, double3 b, double3 c, double3 d) {
    return dih0<2>(phi, kb, a, b, c, d);
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
#endif

struct KantorPart { double3 r, v; };
typedef double3 Pos;

static __device__ Pos kantor_fetch_pos(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pos r;
    r.x = pp[i].r[X];
    r.y = pp[i].r[Y];
    r.z = pp[i].r[Z];
    return r;
}

static __device__ KantorPart kantor_fetch_part(const Particle *pp, int i) {
    enum {X, Y, Z};
    KantorPart p;
    const float *r, *v;
    r = pp[i].r; v = pp[i].v;
    p.r.x = r[X]; p.r.y = r[Y]; p.r.z = r[Z];
    p.v.x = v[X]; p.v.y = v[Y]; p.v.z = v[Z];
    return p;
}

static __device__ void kantor_adj_dihedrals(const RbcParams_v *par, const Particle *pp, double3 r0,
                                     AdjMap *m,        double f[3]) {
    enum {X, Y, Z};
    Pos r1, r2, r3, r4;
    double3 f0;
    double phi, kb;
    r1 = kantor_fetch_pos(pp, m->i1);
    r2 = kantor_fetch_pos(pp, m->i2);
    r3 = kantor_fetch_pos(pp, m->i3);
    r4 = kantor_fetch_pos(pp, m->i4);

    phi = par->phi / 180.0 * M_PI;
    kb  = par->kb;

    f0 = force_kantor0_dev::dih_a(phi, kb, r0, r2, r1, r4);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;

    f0 = force_kantor0_dev::dih_b(phi, kb, r1, r0, r2, r3);
    f[X] += f0.x; f[Y] += f0.y; f[Z] += f0.z;
}

__global__ void kantor_force_kernel(RbcParams_v par, int md, int nv, int nc, const Particle *pp,
                      Adj_v adj,      float *ff) {
    enum {X, Y, Z};
    int i, pid, valid;
    double f[3];
    AdjMap m;

    i = threadIdx.x + blockDim.x * blockIdx.x;
    pid = i / md;

    if (pid >= nc * nv) return;
    valid = adj_get_map(i, &adj,      &m);
    if (!valid) return;

    const KantorPart p0 = kantor_fetch_part(pp, m.i0);

    f[X] = f[Y] = f[Z] = 0;
    kantor_adj_dihedrals(&par, pp, p0.r, &m,                 f);

    atomicAdd(&ff[3 * pid + 0], f[X]);
    atomicAdd(&ff[3 * pid + 1], f[Y]);
    atomicAdd(&ff[3 * pid + 2], f[Z]);
}
void kantor_ini(const MeshRead *cell, Kantor **pq) {
    Kantor *q;
    int md, nt, nv;
    const int4 *tt;
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    md = mesh_read_get_md(cell);
    tt = mesh_read_get_tri(cell);

    UC(adj_ini(md, nt, nv, tt,      &q->adj));
    UC(adj_view_ini(q->adj,      &q->adj_v));

    *pq = q;
}

void kantor_fin(Kantor *q) {
    UC(adj_fin(q->adj));
    UC(adj_view_fin(q->adj_v));
    EFREE(q);
}
void kantor_apply(Kantor *t, const RbcParams *par, const RbcQuants *q,      Force *ff) {
    int nc, nv, md;
    RbcParams_v parv;
    if (q->nc <= 0) return;
    if (!d::is_device_pointer(q->pp))
        ERR("`q->pp` is not a device pointer");
    if (!d::is_device_pointer(ff))
        ERR("`ff` is not a device pointer");
    md = q->md; nv = q->nv; nc = q->nc;
    parv = rbc_params_get_view(par);
    KL(kantor_force_kernel, (k_cnf(nc*nv*md)),
       (parv, md, nv, nc, q->pp, *t->adj_v,      (float*)ff));
}

#include <stdio.h>
#include <math.h>
#include <string.h>

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
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

#ifndef UDX_FORCE_DEF_COMMON
#define UDX_FORCE_DEF_COMMON
#include "constants.cuh"
#endif
#ifndef UDX_FORCE_PAIR_TYPE_COMMON
#define UDX_FORCE_PAIR_TYPE_COMMON

enum {
    MAX_PAR = (N_COLOR * (N_COLOR + 1)) / 2
};

struct PairDPD {
    float a, g, s, spow;
};

struct PairDPDC {
    int ncolors;
    float a[MAX_PAR], g[MAX_PAR], s[MAX_PAR], spow;
};

struct PairDPDCM {
    int ncolors;
    float a[N_COLOR], g[N_COLOR], s[N_COLOR], spow;
};

struct PairDPDLJ {
    float a, g, s, spow;
    float ljs, lje;
};

struct PairAdhesion {
    float k1, k2;
};

struct PairPa {
    float x, y, z;
    float vx, vy, vz;

    int color;
};

struct PairFo {
    float x, y, z;
};

struct PairSFo {
    float x, y, z;

    float sxx, syy, szz, sxy, sxz, syz;
};
#endif
struct PairParams;
struct Config;

struct PairDPD;
struct PairDPDC;
struct PairDPDCM;
struct PairDPDLJ;
struct PairAdhesion;

void pair_ini(PairParams **);
void pair_fin(PairParams *);

void pair_set_dpd(int ncol, const float a[], const float g[], float spow, PairParams *p);
void pair_set_lj(float sigma, float eps, PairParams *p);
void pair_set_adhesion(float k1, float k2, PairParams *p);

void pair_set_conf(const Config *, const char *name_space, PairParams *);

void pair_compute_dpd_sigma(float kBT, float dt, PairParams *p);

void pair_get_view_dpd(const PairParams*, PairDPD*);
void pair_get_view_dpd_color(const PairParams*, PairDPDC*);
void pair_get_view_dpd_mirrored(const PairParams*, PairDPDCM*);
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
void pair_get_view_adhesion(const PairParams*, PairAdhesion*);

static int get_ncol(int npar) {
    int d = lround( sqrt(1 + 8 * npar) );
    return (d - 1) / 2;
}

void pair_set_conf(const Config *cfg, const char *ns, PairParams *par) {
    int dpd, lj, adhesion;

    UC(conf_lookup_bool_ns(cfg, ns, "dpd",      &dpd));
    UC(conf_lookup_bool_ns(cfg, ns, "lj",       &lj));
    UC(conf_lookup_bool_ns(cfg, ns, "adhesion", &adhesion));

    if (dpd) {
        int na, ng, nc;
        float a[MAX_PAR], g[MAX_PAR], spow;

        UC(conf_lookup_vfloat_ns(cfg, ns, "a", MAX_PAR, &na, a));
        UC(conf_lookup_vfloat_ns(cfg, ns, "g", MAX_PAR, &ng, g));
        UC(conf_lookup_float_ns(cfg, ns, "spow", &spow));

        if (na != ng)
            ERR("%s.a and %s.g must have the same length: %d / %d", ns, ns, na, ng);

        nc = get_ncol(na);

        UC(pair_set_dpd(nc, a, g, spow,      par));
    }

    if (lj) {
        float s, e;

        UC(conf_lookup_float_ns(cfg, ns, "ljs", &s));
        UC(conf_lookup_float_ns(cfg, ns, "lje", &e));

        UC(pair_set_lj(s, e,      par));
    }

    if (adhesion) {
        float k1, k2;

        UC(conf_lookup_float_ns(cfg, ns, "k1", &k1));
        UC(conf_lookup_float_ns(cfg, ns, "k2", &k2));

        UC(pair_set_adhesion(k1, k2,      par));
    }
}

#include <stdio.h>
#include <string.h>
#include <math.h>

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

struct PairParams {
    bool dpd;
    float a[MAX_PAR];
    float g[MAX_PAR];
    float s[MAX_PAR];
    int ncolors;
    float spow;

    bool lj;
    float ljs;
    float lje;

    bool adhesion;
    float k1;
    float k2;
};

void pair_ini(PairParams **par) {
    PairParams *p;
    EMALLOC(1, par);
    p = *par;

    p->ncolors = 0;
    p->dpd = p->lj = p->adhesion = false;
}

void pair_fin(PairParams *p) {
    EFREE(p);
}

void pair_set_lj(float sigma, float eps, PairParams *p) {
    p->lj = true;
    p->ljs = sigma;
    p->lje = eps;
}

static int get_npar(int ncol) { return (ncol * (ncol+1)) / 2; }

void pair_set_dpd(int ncol, const float a[], const float g[], float spow, PairParams *p) {
    int npar = get_npar(ncol);
    size_t sz = sizeof(float) * npar;

    if (ncol <= 0)      ERR("need at least one color (given %d)\n", ncol);
    if (ncol > N_COLOR) ERR("too many colors (given %d)\n", ncol);

    p->dpd = true;
    p->ncolors = ncol;
    memcpy(p->a, a, sz);
    memcpy(p->g, g, sz);
    p->spow = spow;
}

void pair_set_adhesion(float k1, float k2, PairParams *p) {
    p->adhesion = true;
    p->k1 = k1;
    p->k2 = k2;
}

static void check_dpd(const PairParams *p) {
    if (!p->dpd) ERR("Must have dpd parameters");
}

static void check_lj(const PairParams *p) {
    if (!p->lj) ERR("Must have lj parameters");
}

static void check_adhesion(const PairParams *p) {
    if (!p->adhesion) ERR("Must have adhesion parameters");
}

void pair_compute_dpd_sigma(float kBT, float dt, PairParams *p) {
    int i, npar, ncol;
    UC(check_dpd(p));
    ncol = p->ncolors;
    npar = get_npar(ncol);

    for (i = 0; i < npar; ++i)
        p->s[i] = sqrt(2 * kBT * p->g[i] / dt);
}

void pair_get_view_dpd(const PairParams *p, PairDPD *v) {
    enum {PID=0};
    UC(check_dpd(p));
    v->a = p->a[PID];
    v->g = p->g[PID];
    v->s = p->s[PID];
    v->spow = p->spow;
}

void pair_get_view_dpd_color(const PairParams *p, PairDPDC *v) {
    size_t sz;
    int ncol, npar;
    UC(check_dpd(p));
    ncol = v->ncolors = p->ncolors;
    npar = get_npar(ncol);
    sz = npar * sizeof(float);
    memcpy(v->a, p->a, sz);
    memcpy(v->g, p->g, sz);
    memcpy(v->s, p->s, sz);
    v->spow = p->spow;
}

void pair_get_view_dpd_mirrored(const PairParams *p, PairDPDCM *v) {
    int i, j, nc;
    UC(check_dpd(p));
    nc = v->ncolors = p->ncolors;
    for (i = 0; i < nc; ++i) {
        j = (i+1) * (i+2) / 2 - 1;
        v->a[i] = p->a[j];
        v->g[i] = p->g[j];
        v->s[i] = p->s[j];
    }
    v->spow = p->spow;
}

void pair_get_view_dpd_lj(const PairParams *p, PairDPDLJ *v) {
    enum {PID=0};
    UC(check_dpd(p));
    UC(check_lj(p));
    v->a = p->a[PID];
    v->g = p->g[PID];
    v->s = p->s[PID];
    v->ljs = p->ljs;
    v->lje = p->lje;
    v->spow = p->spow;
}

void pair_get_view_adhesion(const PairParams *p, PairAdhesion *v) {
    UC(check_adhesion(p));
    v->k1 = p->k1;
    v->k2 = p->k2;
}

#include <stdio.h>
#include <assert.h>

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
#ifndef UDX_FORCE_DEVICE_API_COMMON
#define UDX_FORCE_DEVICE_API_COMMON
#include "device.cuh"
#endif

#if   defined(DEV_CUDA)

#elif defined(DEV_CPU)

#ifdef __device__
  #undef  __device__
  #define __device__
#endif

#ifdef __global__
  #undef  __global__
  #define __global__
#endif

#ifdef __host__
  #undef  __host__
  #define __host__
#endif

#ifdef __noinline__
  #undef  __noinline__
  #define __noinline__
#endif

#ifdef __forceinline__
  #undef  __forceinline__
  #define __forceinline__
#endif

#ifdef __constant__
  #undef  __constant__
  #define __constant__
#endif

#ifdef __shared__
  #undef  __shared__
  #define __shared__
#endif

#ifdef __managed__
  #undef  __managed__
  #define __managed__
#endif

#ifdef __restrict__
  #undef  __restrict__
  #define __restrict__
#endif
#else
  #error DEV_* is undefined
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
#define atomicAdd(...) (0)

#define atomicExch(...)
#define __float2half_rn(...) (0)
#define __fmaf_rz(...) (0.0)
#define __ldg(p) (*(p))
#define __popc(...)
#define __shfl(...) (0.0)
#define __shfl_down(...) (0.0)
#define __shfl_up(...) (0)
#define __shfl_xor(...) (0)

#define __syncthreads(...)

#define tex3D(...)

#define  Ifetch(t, i)  (0);
#define F4fetch(t, i)  (make_float4(0, 0, 0, 0))
#define F2fetch(t, i)  (make_float2(0, 0))
#define Ffetch(t, i)   (0.0)
#define Tfetch(T, t, i) (T{})

#define Ttex3D(T, to, i, j, k) (T{})
namespace d {
static uint3 threadIdx, blockIdx, gridDim;
static dim3  blockDim;
static int   warpSize = 1;
}

#define threadIdx d::threadIdx
#define blockDim  d::blockDim
#define gridDim   d::gridDim
#define blockIdx  d::blockIdx
#define warpSize  d::warpSize
#else
  #error DEV_* is undefined
#endif

struct RNDunif;
void rnd_ini(int x, int y, int z, int c,      RNDunif **r);
void rnd_fin(RNDunif *r);
float rnd_get(RNDunif *r);

#ifndef UDX_FORCE_DEF_COMMON
#define UDX_FORCE_DEF_COMMON
#include "constants.cuh"
#endif
void msg_ini(int rank);
void msg_print(const char *fmt, ...);
bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}

#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
#ifndef UDX_FORCE_FRAG_COMMON
#define UDX_FORCE_FRAG_COMMON
enum {frag_bulk = 26};
#endif

struct int3;

namespace frag_hst {

int i2dx(int i);
int i2dy(int i);
int i2dz(int i);
void i2d3(int i,      int d[3]);

int d2i(int x, int y, int z);
int d32i(const int d[3]);

int ncell(int3 L, int i);

int antid2i(int x, int y, int z);
int anti(int i);

void estimates(int3 L, int nfrags, float maxd,      int *cap);

}
#include "fragment.cuh"

#ifndef UDX_FORCE_KL_COMMON
#define UDX_FORCE_KL_COMMON
#include "launch.cuh"
#endif
#ifndef UDX_FORCE_TYPE_COMMON
#define UDX_FORCE_TYPE_COMMON
#include "types.cuh"
#endif

struct Dbg;

struct Config;
struct Force;
struct Particle;
struct int3;
struct Coords;

enum {
    DBG_POS,
    DBG_POS_SOFT,
    DBG_VEL,
    DBG_FORCES,
    DBG_COLORS,
    DBG_CLIST,
    DBG_NKIND_
};

void dbg_ini(Dbg**);
void dbg_fin(Dbg*);

void dbg_enable(int kind, Dbg *dbg);
void dbg_disable(int kind, Dbg *dbg);
void dbg_set_verbose(bool, Dbg *dbg);
void dbg_set_dump(bool, Dbg *dbg);

void dbg_set_conf(const Config*, Dbg*);

void dbg_check_pos(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_pos_soft(const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_vel(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_forces(float dt, const Coords *c, const char *base, const Dbg *dbg, int n, const Particle *pp, const Force *ff);
void dbg_check_colors(const Coords *c, const Dbg *dbg, int n, const int *cc);
void dbg_check_clist(const Coords *c, const Dbg *dbg, int3 L, const int *starts, const int *counts, const Particle *pp);

#ifndef UDX_FORCE_PAIR_TYPE_COMMON
#define UDX_FORCE_PAIR_TYPE_COMMON

enum {
    MAX_PAR = (N_COLOR * (N_COLOR + 1)) / 2
};

struct PairDPD {
    float a, g, s, spow;
};

struct PairDPDC {
    int ncolors;
    float a[MAX_PAR], g[MAX_PAR], s[MAX_PAR], spow;
};

struct PairDPDCM {
    int ncolors;
    float a[N_COLOR], g[N_COLOR], s[N_COLOR], spow;
};

struct PairDPDLJ {
    float a, g, s, spow;
    float ljs, lje;
};

struct PairAdhesion {
    float k1, k2;
};

struct PairPa {
    float x, y, z;
    float vx, vy, vz;

    int color;
};

struct PairFo {
    float x, y, z;
};

struct PairSFo {
    float x, y, z;

    float sxx, syy, szz, sxy, sxz, syz;
};
#endif
#ifndef UDX_FORCE_PAIR_DEV_COMMON
#define UDX_FORCE_PAIR_DEV_COMMON
#define _S_ static __device__
#define _I_ static __device__

_S_ float cap(float x, float lo, float hi) {
    if      (x > hi) return hi;
    else if (x < lo) return lo;
    else             return x;
}

static const float EPS = 1e-6;
enum {NORM_OK, NORM_BIG, NORM_SMALL};
_S_ int norm(       float3 *pos,      float *pr, float *pinvr) {

    float x, y, z, invr, r, r2;
    x = pos->x; y = pos->y; z = pos->z;

    r2 = x*x + y*y + z*z;
    if      (r2 >= 1 )   return NORM_BIG;
    else if (r2 < EPS) {
        *pr = pos->x = pos->y = pos->z = 0;
        return NORM_SMALL;
    } else {
        invr = rsqrtf(r2);
        r = r2 * invr;
        x *= invr; y *= invr; z *= invr;
        pos->x = x; pos->y = y; pos->z = z; *pr = r; *pinvr = invr;
        return NORM_OK;
    }
}

_S_ float ker_wrf(float s, float x) {
    return powf(x, s);
}

_S_ float magn_dpd(float a, float g, float s, float kpow,
                   float rnd, float r, float ev) {
    float wr, wc;
    float rm, f0;

    rm = max(1 - r, 0.0f);
    wc = rm;
    wr = ker_wrf(kpow, rm);

    f0  = (-g * wr * ev + s * rnd) * wr;
    f0 +=                        a * wc;
    return f0;
}

_S_ float magn_lj(float s, float e, float invr) {
    float t2, t4, t6, f;
    t2 = s * s * invr * invr;
    t4 = t2 * t2;
    t6 = t4 * t2;
    f = e * 24 * invr * t6 * (2 * t6 - 1);
    f = cap(f, 0, 1e4);
    return f;
}

_S_ float magn_adhesion(float k1, float k2, float r, float ev) {
    float spring, visc;
    spring = - k1 * r;
    visc   = - k2 * ev;
    return spring + visc;
}

_S_ float force_magn(const PairDPD *p, float rnd, float ev, float r, float) {
    return magn_dpd(p->a, p->g, p->s, p->spow, rnd, r, ev);
}

_S_ float force_magn(const PairDPDLJ *p, float rnd, float ev, float r, float invr) {
    float f;
    f  = magn_dpd(p->a, p->g, p->s, p->spow, rnd, r, ev);
    f += magn_lj(p->ljs, p->lje, invr);
    return f;
}

_S_ float force_magn(const PairAdhesion *p, float, float ev, float r, float) {
    return magn_adhesion(p->k1, p->k2, r, ev);
}

_S_ void magn2fo(float f0, float3 er, float3 dr,      PairFo *f) {
    f->x = f0 * er.x;
    f->y = f0 * er.y;
    f->z = f0 * er.z;
}

_S_ void magn2fo(float f0, float3 er, float3 dr,      PairSFo *f) {
    f->x = f0 * er.x;
    f->y = f0 * er.y;
    f->z = f0 * er.z;

    f->sxx = 0.5f * f->x * dr.x;
    f->sxy = 0.5f * f->x * dr.y;
    f->sxz = 0.5f * f->x * dr.z;
    f->syy = 0.5f * f->y * dr.y;
    f->syz = 0.5f * f->y * dr.z;
    f->szz = 0.5f * f->z * dr.z;
}

_S_ void make_zero(PairFo *f) {
    f->x = f->y = f->z = 0;
}

_S_ void make_zero(PairSFo *f) {
    f->x = f->y = f->z = 0;
    f->sxx = f->sxy = f->sxz = 0;
    f->syy = f->syz = f->szz = 0;
}

template <typename Param, typename Fo>
_I_ void pair_force(const Param *p, PairPa a, PairPa b, float rnd,      Fo *f)
{
    float r, invr, ev, f0;
    float3 dr, er, dv;
    int vnstat;

    dr.x = a.x - b.x;
    dr.y = a.y - b.y;
    dr.z = a.z - b.z;

    dv.x = a.vx - b.vx;
    dv.y = a.vy - b.vy;
    dv.z = a.vz - b.vz;

    er = dr;

    vnstat = norm(       &er,       &r, &invr);
    if (vnstat == NORM_BIG) {
        make_zero(f);
        return;
    }

    ev = dot<float>(&er, &dv);

    f0 = force_magn(p, rnd, ev, r, invr);
    magn2fo(f0, er, dr,      f);
}

_S_ int colors2pid(int ca, int cb) {
    int c0, c1;
    c0 = ca < cb ? ca : cb;
    c1 = ca < cb ? cb : ca;
    return c1 * (c1+1) / 2 + c0;
}

template <typename Fo>
_I_ void pair_force(const PairDPDC *pc, PairPa a, PairPa b, float rnd,      Fo *f) {
    PairDPD p;
    int pid;
    pid = colors2pid(a.color, b.color);
    p.a = pc->a[pid];
    p.g = pc->g[pid];
    p.s = pc->s[pid];
    p.spow = pc->spow;
    pair_force(&p, a, b, rnd,      f);
}

template <typename Fo>
_I_ void pair_force(const PairDPDCM *pc, PairPa a, PairPa b, float rnd,      Fo *f) {
    PairDPD p;
    int pid = a.color;
    p.a = pc->a[pid];
    p.g = pc->g[pid];
    p.s = pc->s[pid];
    p.spow = pc->spow;
    pair_force(&p, a, b, rnd,      f);
}

_I_ void pair_add(const PairFo *b,      PairFo *a)
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;
}

_I_ void pair_add(const PairSFo *b,      PairSFo *a)
{
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;

    a->sxx += b->sxx;
    a->sxy += b->sxy;
    a->sxz += b->sxz;
    a->syy += b->syy;
    a->syz += b->syz;
    a->szz += b->szz;
}

#undef _S_
#undef _I_
#endif
struct PairParams;
struct Config;

struct PairDPD;
struct PairDPDC;
struct PairDPDCM;
struct PairDPDLJ;
struct PairAdhesion;

void pair_ini(PairParams **);
void pair_fin(PairParams *);

void pair_set_dpd(int ncol, const float a[], const float g[], float spow, PairParams *p);
void pair_set_lj(float sigma, float eps, PairParams *p);
void pair_set_adhesion(float k1, float k2, PairParams *p);

void pair_set_conf(const Config *, const char *name_space, PairParams *);

void pair_compute_dpd_sigma(float kBT, float dt, PairParams *p);

void pair_get_view_dpd(const PairParams*, PairDPD*);
void pair_get_view_dpd_color(const PairParams*, PairDPDC*);
void pair_get_view_dpd_mirrored(const PairParams*, PairDPDCM*);
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
void pair_get_view_adhesion(const PairParams*, PairAdhesion*);

struct PaArray_v {
    const float *pp;
};

struct PaCArray_v {
    const float *pp;
    const int *cc;
};
#ifndef UDX_FORCE_PARRAY_COMMON
#define UDX_FORCE_PARRAY_COMMON
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
#define _S_ static __device__
#define _I_ static __device__

template<typename PA>
_S_ void parray_get_pp(PA a, int i,      PairPa *p) {
    const float2 *pp = (const float2 *) a.pp;
    float2 s0, s1, s2;
    s0 = __ldg(pp + 3*i + 0);
    s1 = __ldg(pp + 3*i + 1);
    s2 = __ldg(pp + 3*i + 2);

    p->x = s0.x;
    p->y = s0.y;
    p->z = s1.x;

    p->vx = s1.y;
    p->vy = s2.x;
    p->vz = s2.y;
}

_I_ void parray_get(PaArray_v a, int i,      PairPa *p)
{
    parray_get_pp(a, i,      p);
}

_I_ void parray_get(PaCArray_v a, int i,      PairPa *p)
{
    parray_get_pp(a, i,      p);
    p->color = a.cc[i];
}

#undef _S_
#undef _I_

struct RNDunif;
struct Fsi;
struct int3;

void fsi_ini(int rank, int3 L,      Fsi **fsi);
void fsi_fin(Fsi *fsi);

void fsi_bind_solvent(PaArray pa, Force *ff, int n, const int *starts,      Fsi *fsi);
void fsi_bulk(Fsi *fsi, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw);
void fsi_halo(Fsi *fsi, int nw, const PairParams **prms, Pap26 all_pp, Fop26 all_ff, const int *all_counts);

typedef PairPa Pa;
struct FsiFo { float *x, *y, *z; };
static __device__ float fsi_random(uint lid, uint rid, float seed) {
    return rnd::mean0var1uu(seed, lid, rid);
}

static __device__ FsiFo fsi_ff2f(float *ff, int i) {
    FsiFo f;
    ff += 3*i;
    f.x = ff++; f.y = ff++; f.z = ff++;
    return f;
}

template <typename Par>
static __device__ void fsi_pair0(Par params, const Pa a, const Pa b, float rnd,      float *fx, float *fy, float *fz) {
    PairFo f;
    pair_force(&params, a, b, rnd,      &f);
    *fx = f.x; *fy = f.y; *fz = f.z;
}

template <typename Par>
static __device__ void fsi_pair(Par params, const Pa a, const Pa b, float rnd,
                            float *fx, float *fy, float *fz, FsiFo f) {

    float x, y, z;
    fsi_pair0(params, a, b, rnd,      &x, &y, &z);
    *fx += x; *fy += y; *fz += z;
    atomicAdd(f.x, -x); atomicAdd(f.y, -y); atomicAdd(f.z, -z);
}
struct FsiMap {
    int  org0, org1, org2;
    uint cnt0, cnt1, cnt2;
};

static __device__ int fsi_endp(const FsiMap m, uint i) { return i >= m.cnt2; }

static __device__ uint fsi_m2id(const FsiMap m, uint i) {

    int m1, m2;
    uint id;
    m1 = (int)(i >= m.cnt0);
    m2 = (int)(i >= m.cnt1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}
static __device__ int fsi_r2map(int3 L, const int *start, int zplane, int n, float x, float y, float z,      FsiMap *m) {

    int cnt0, cnt1, cnt2, org0;
    int org1, org2;
    int xoffset, yoffset, zoffset;
    int ncells;

    xoffset = L.x / 2;
    yoffset = L.y / 2;
    zoffset = L.z / 2;
    ncells = L.x * L.y * L.z;

    const int xcenter = xoffset + (int)floorf(x);
    const int xstart = max(0, xcenter - 1);
    const int xcount = min(L.x, xcenter + 2) - xstart;

    if (xcenter - 1 >= L.x || xcenter + 2 <= 0) return 0;

    const int ycenter = yoffset + (int)floorf(y);

    const int zcenter = zoffset + (int)floorf(z);
    const int zmy = zcenter - 1 + zplane;
    const bool zvalid = zmy >= 0 && zmy < L.z;

    int count0 = 0, count1 = 0, count2 = 0;

    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < L.y) {
        const int cid0 = xstart + L.x * (ycenter - 1 + L.y * zmy);
        org0 = start[cid0];
        count0 = ((cid0 + xcount == ncells)
                  ? n
                  : start[cid0 + xcount]) -
            org0;
    }

    if (zvalid && ycenter >= 0 && ycenter < L.y) {
        const int cid1 = xstart + L.x * (ycenter + L.y * zmy);
        org1 = start[cid1];
        count1 = ((cid1 + xcount == ncells)
                  ? n
                  : start[cid1 + xcount]) -
            org1;
    }

    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < L.y) {
        const int cid2 = xstart + L.x * (ycenter + 1 + L.y * zmy);
        org2 = start[cid2];
        count2 = ((cid2 + xcount == ncells)
                  ? n
                  : start[cid2 + xcount]) -
            org2;
    }

    cnt0 = count0;
    cnt1 = count0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->cnt0 = cnt0; m->cnt1 = cnt1; m->cnt2 = cnt2;
    return 1;
}
static __device__ int fsi_tex2map(int3 L, const int *start, int zplane, int n1, float x, float y, float z,      FsiMap *m) {

    int xstart, xcount;
    int zmy;
    int xcenter, ycenter, zcenter;
    bool zvalid;
    int count0, count1, count2;
    int cid0, cid1, cid2;
    int cnt0, cnt1, cnt2;
    int org0, org1, org2;
    int xoffset, yoffset, zoffset;
    int ncells;

    xoffset = L.x / 2;
    yoffset = L.y / 2;
    zoffset = L.z / 2;
    ncells = L.x * L.y * L.z;

    xcenter = xoffset + (int)floorf(x);
    xstart = max(0, xcenter - 1);
    xcount = min(L.x, xcenter + 2) - xstart;

    if (xcenter - 1 >= L.x || xcenter + 2 <= 0) return 0;

    ycenter = yoffset + (int)floorf(y);

    zcenter = zoffset + (int)floorf(z);
    zmy = zcenter - 1 + zplane;
    zvalid = zmy >= 0 && zmy < L.z;

    count0 = count1 = count2 = 0;

    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < L.y) {
        cid0 = xstart + L.x * (ycenter - 1 + L.y * zmy);
        org0 = start[cid0];
        count0 = ((cid0 + xcount == ncells)
                  ? n1
                  : start[cid0 + xcount]) -
            org0;
    }

    if (zvalid && ycenter >= 0 && ycenter < L.y) {
        cid1 = xstart + L.x * (ycenter + L.y * zmy);
        org1 = start[cid1];
        count1 = ((cid1 + xcount == ncells)
                  ? n1
                  : start[cid1 + xcount]) -
            org1;
    }

    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < L.y) {
        cid2 = xstart + L.x * (ycenter + 1 + L.y * zmy);
        org2 = start[cid2];
        count2 = ((cid2 + xcount == ncells)
                  ? n1
                  : start[cid2 + xcount]) -
            org2;
    }

    cnt0 = count0;
    cnt1 = count0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->cnt0 = cnt0; m->cnt1 = cnt1; m->cnt2 = cnt2;
    return 1;
}
static __device__ void fsi_pp2p(const Particle *pp, int i,      Pa *p) {
    enum {X, Y, Z};
    Particle a = pp[i];
    p->x  = a.r[X]; p->y  = a.r[Y]; p->z  = a.r[Z];
    p->vx = a.v[X]; p->vy = a.v[Y]; p->vz = a.v[Z];
}

static __device__ int fsi_p2map(int3 L, const int *start, int zplane, int n, const Pa p,      FsiMap *m) {

    return fsi_r2map(L, start, zplane, n, p.x, p.y, p.z, m);
}

template <typename Par, typename Parray>
static __device__ void fsi_bulk0(Par params, const Pa a, Parray parray, int lid, const FsiMap m, float seed,
                             float *fx, float *fy, float *fz, float *ff) {

    Pa b;
    FsiFo f;
    int i, rid;
    *fx = *fy = *fz = 0;
    for (i = 0; !fsi_endp(m, i); ++i) {
        rid = fsi_m2id(m, i);
        parray_get(parray, rid,      &b);
        f = fsi_ff2f(ff, rid);
        fsi_pair(params, a, b, fsi_random(lid, rid, seed),      fx, fy, fz,   f);
    }
}

template <typename Par, typename Parray>
static __device__ void bulk1(Par params, const Pa a, Parray parray,
                             const FsiFo f, int i, const FsiMap m, float seed,      float *ff) {
    float fx, fy, fz;
    fsi_bulk0(params, a, parray, i, m, seed,      &fx, &fy, &fz, ff);
    atomicAdd(f.x, fx);
    atomicAdd(f.y, fy);
    atomicAdd(f.z, fz);
}

template <typename Par, typename Parray>
static __device__ void bulk2(Par params, int3 L, const int *start, const Particle *ppA, Parray parray, int i, int zplane, int n, float seed,
                                  float *ff, float *ff0) {
    Pa p;
    FsiFo f;
    FsiMap m;
    fsi_pp2p(ppA, i,      &p);
    f = fsi_ff2f(ff, i);
    if (!fsi_p2map(L, start, zplane, n, p,      &m)) return;
    bulk1(params, p, parray, f, i, m, seed,      ff0);
}

template <typename Par, typename Parray>
__global__ void fsi_bulk_kernel(Par params, int3 L, const int *start, const Particle *ppA, Parray parray, int n0, int n1, float seed, float *ff, float *ff0) {
    int gid, i, zplane;
    gid    = threadIdx.x + blockDim.x * blockIdx.x;
    i      = gid / 3;
    zplane = gid % 3;
    if (i >= n0) return;
    bulk2(params, L, start, ppA, parray, i, zplane, n1, seed, ff, ff0);
}
static __device__ Pa warp2p(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pa p;
    pp += i;

     p.x = pp->r[X];  p.y = pp->r[Y];  p.z = pp->r[Z];
    p.vx = pp->v[X]; p.vy = pp->v[Y]; p.vz = pp->v[Z];
    return p;
}

template <typename Par, typename Parray>
static __device__ void halo0(Par params, int3 L, const int *start, Pa A, int aid, Parray parray, int nb, float seed,      float *fA, float *ffB) {
    enum {X, Y, Z};

    Pa B;
    FsiFo f;

    FsiMap m;
    int zplane;
    int i, bid;
    float fx, fy, fz;

    fx = fy = fz = 0;
    for (zplane = 0; zplane < 3; ++zplane) {
        if (!fsi_tex2map(L, start, zplane, nb, A.x, A.y, A.z,      &m)) continue;
        for (i = 0; !fsi_endp(m, i); ++i) {
            bid = fsi_m2id(m, i);
            parray_get(parray, bid,      &B);
            f = fsi_ff2f(ffB, bid);
            fsi_pair(params, A, B, fsi_random(aid, bid, seed),      &fx, &fy, &fz,   f);
        }
    }
    fA[X] += fx; fA[Y] += fy; fA[Z] += fz;
}

template <typename Par, typename Parray>
static __device__ void halo1(Par params, int3 L, const int *cellsstart, int27 starts, Pap26 pp, Fop26 ff, int aid, Parray parray, int nb, float seed,      float *ffB) {
    int fid;
    int start;
    Pa A;
    float *fA;

    fid = frag_dev::frag_get_fid(starts.d, aid);
    start = starts.d[fid];

    A = warp2p(pp.d[fid], aid - start);

    fA = ff.d[fid][aid-start].f;

    halo0(params, L, cellsstart, A, aid, parray, nb, seed,      fA, ffB);
}

template <typename Par, typename Parray>
__global__ void fsi_halo_kernel(Par params, int3 L, const int *cellsstart, int27 starts, Pap26 pp, Fop26 ff, Parray parray, int na, int nb, float seed,      float *ffB) {
    int aid;
    aid = threadIdx.x + blockDim.x * blockIdx.x;
    if (aid >= na) return;
    halo1(params, L, cellsstart, starts, pp, ff, aid, parray, nb, seed,      ffB);
}

struct RNDunif;

struct SolventWrap {
    PaArray pa;
    Force *ff;
    int n;
    const int *starts;
};

struct Fsi {
    SolventWrap *wo;
    RNDunif     *rgen;
    int3 L;
};
void fsi_ini(int rank, int3 L,      Fsi **fsi) {
    Fsi *f;
    EMALLOC(1, fsi);
    f = *fsi;

    f->L = L;
    UC(rnd_ini(1908 - rank, 1409 + rank, 290, 12968,      &f->rgen));
    EMALLOC(1, &f->wo);
}

void fsi_fin(Fsi *fsi) {
    UC(rnd_fin(fsi->rgen));
    EFREE(fsi->wo);
    EFREE(fsi);
}

void fsi_bind_solvent(PaArray pa, Force *ff, int n, const int *starts,      Fsi *fsi) {
    SolventWrap *w = fsi->wo;
    w->pa = pa;
    w->n  = n;
    w->ff = ff;
    w->starts = starts;
}

static void bulk_one_wrap(const PairParams *params, PaWrap *pw, FoWrap *fw, Fsi *fsi) {
    int n0, n1;
    float rnd;
    const Particle *ppA = pw->pp;
    SolventWrap *wo = fsi->wo;
    const PaArray *parray = &wo->pa;

    rnd = rnd_get(fsi->rgen);
    n0  = pw->n;
    n1  = wo->n;

    if (parray_is_colored(parray)) {
        PairDPDCM pv;
        PaCArray_v paview;

        pair_get_view_dpd_mirrored(params, &pv);
        parray_get_view(parray, &paview);

        KL(fsi_bulk_kernel, (k_cnf(3*n0)), (pv, fsi->L, wo->starts, ppA, paview,
                                          n0, n1,
                                          rnd, (float*)fw->ff, (float*)wo->ff));
    }
    else {
        PairDPD pv;
        PaArray_v paview;

        pair_get_view_dpd(params, &pv);
        parray_get_view(parray, &paview);

        KL(fsi_bulk_kernel, (k_cnf(3*n0)), (pv, fsi->L, wo->starts, ppA, paview,
                                          n0, n1,
                                          rnd, (float*)fw->ff, (float*)wo->ff));
    }
}

void fsi_bulk(Fsi *fsi, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw) {
    if (nw == 0)
        return;

    for (int i = 0; i < nw; i++)
        if (prms[i]) bulk_one_wrap(prms[i], &pw[i], &fw[i], fsi);
}
static void fsi_halo_one_type(const PairParams *params, Fsi *fsi, Pap26 PP, Fop26 FF, const int counts[26]) {
    enum {NFRAGS = 26};
    int i, n, s;
    int27 starts;
    SolventWrap *wo = fsi->wo;
    const PaArray *parray = &wo->pa;
    float rnd = rnd_get(fsi->rgen);

    starts.d[0] = 0;
    for (i = s = 0; i < NFRAGS; ++i) starts.d[i + 1] = (s += counts[i]);
    n = starts.d[NFRAGS];

    if (parray_is_colored(parray)) {
        PairDPDCM pv;
        PaCArray_v pav;

        pair_get_view_dpd_mirrored(params, &pv);
        parray_get_view(parray, &pav);

        KL(fsi_halo_kernel, (k_cnf(n)), (pv, fsi->L, wo->starts, starts, PP, FF, pav, n, wo->n, rnd,      (float*)wo->ff));
    }
    else {
        PairDPD pv;
        PaArray_v pav;

        pair_get_view_dpd(params, &pv);
        parray_get_view(parray, &pav);

        KL(fsi_halo_kernel, (k_cnf(n)), (pv, fsi->L, wo->starts, starts, PP, FF, pav, n, wo->n, rnd,      (float*)wo->ff));
    }
}

void fsi_halo(Fsi *fsi, int nw, const PairParams **prms, Pap26 all_pp, Fop26 all_ff, const int *all_counts) {
    enum {NFRAGS = 26};
    Pap26 pp;
    Fop26 ff;
    int i, j, start[NFRAGS];
    const int *counts;

    memset(start, 0, sizeof(start));

    for (i = 0; i < nw; i++) {
        counts = all_counts + NFRAGS * i;
        for (j = 0; j < NFRAGS; ++j) {
            pp.d[j] = all_pp.d[j] + start[j];
            ff.d[j] = all_ff.d[j] + start[j];
            start[j] += counts[j];
        }
        if (prms[i]) fsi_halo_one_type(prms[i], fsi, pp, ff, counts);
    }
}
