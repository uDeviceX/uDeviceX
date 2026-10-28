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
#include "fragment.cuh"

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
#define ERR(fmt, ...) do { error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); error_report(); error_abort(); } while (0)
#define UC(F) do { error_stack_push(__FILE__, __LINE__); F; if (error_get()) { error_report(); error_abort(); } error_stack_pop(); } while (0)
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define ceiln(m, n) (((m) + (n) - 1)/(n))
#define k_cnf(n) ceiln((n), 128), 128
#define Dalloc(D, n) CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))
#define Dfree(D) CC(d::Free(D))

template <typename T1, typename T2, typename T3>
__device__ T1 contact_dot(const T2 *a, const T3 *b) { return a->x*b->x + a->y*b->y + a->z*b->z; }
namespace rnd {
static __const__ int N = 18;
static __const__ float gold=0.6180339887498948482, silver=0.4142135623730950488;
static __const__ float bronze=0.00008877875787352212838853023, tin=0.00004602357186447026756768986;
static __const__ float sqrt2=1.41421356237309514547;
__inline__ __device__ float rem(float r) { return r-floorf(r); }
__inline__ __device__ float FMA(float x,float y,float z) { return __fmaf_rz(x,y,z); }
template<int M> __inline__ __device__ float logistic(float x) { float x2=x*x; return logistic<M-2>(FMA(FMA(8.0,x2,-8.0),x2,1.0)); }
template<> __inline__ __device__ float logistic<1>(float x) { return FMA(2.0*x,x,-1.0); }
template<> __inline__ __device__ float logistic<0>(float x) { return x; }
inline __device__ float mean0var1ii(float seed,int u,int v) { float p=rem((u&0x3FF)*gold+u*bronze+(v&0x3FF)*silver+v*tin); return logistic<N>(seed-p)*sqrt2; }
}
struct RNDunif;
void rnd_ini(int,int,int,int,RNDunif**);
void rnd_fin(RNDunif*);
float rnd_get(RNDunif*);
struct Scan;
struct PartList;
struct Clist; struct ClistMap;
void clist_ini(int,int,int,Clist**); void clist_fin(Clist*);
void clist_ini_map(int,int,const Clist*,ClistMap**); void clist_fin_map(ClistMap*);
const int* clists_get_ss(const Clist*); const uint* clist_get_ids(const ClistMap*);
void clist_ini_counts(Clist*); void clist_subindex(bool,int,int,const PartList,Clist*,ClistMap*); void clist_build_map(const int[],Clist*,ClistMap*);
struct PairParams; struct PairDPDLJ; struct Config;
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
void pair_force_dummy();
int get_id_inter(int,int); int contact_scan(int,const int[],int[]);
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

    ev = contact_dot<float>(&er, &dv);

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

#ifndef UDX_CLIST_PARTLIST_COMMON
#define UDX_CLIST_PARTLIST_COMMON
enum {
    ALIVE = 0,
    DEAD  = 1
};




struct PartList {
    const Particle *pp;
    const int *deathlist;
};
#endif

struct Clist;
struct ClistMap;

void clist_ini(int LX, int LY, int LZ,      Clist **c);
void clist_fin(     Clist *c);

void clist_ini_map(int maxp, int nA, const Clist *c,      ClistMap **m);
void clist_fin_map(ClistMap *m);

const int* clists_get_ss(const Clist *c);
const int* clists_get_cc(const Clist *c);
int3       clists_get_dim(const Clist *c);
int        clists_get_n(const Clist *c);
const uint* clist_get_ids(const ClistMap *m);

void clist_ini_counts(Clist *c);
void clist_subindex(bool project, int aid, int n, const PartList lp,      Clist *c, ClistMap *m);
void clist_build_map(const int nn[],      Clist *c, ClistMap *m);


void clist_subindex_local(int n, const PartList lp,      Clist *c, ClistMap *m);
void clist_subindex_remote(int n, const PartList lp,      Clist *c, ClistMap *m);

void clist_gather_pp(const Particle *pplo, const Particle *ppre, const ClistMap *m, long nout,      Particle *ppout);
void clist_gather_ii(const int *iilo, const int *iire, const ClistMap *m, long nout,      int *iiout);


void clist_build(int nlo, int nout, const Particle *pplo,      Particle *ppout, Clist *c, ClistMap *m);
#ifndef UDX_CLIST_DEV_COMMON
#define UDX_CLIST_DEV_COMMON
#define _I_ static __device__

enum { CLIST_NMAX = (256*256*256) };

_I_ uint clist_encode_id(int s, int pid)
{

    assert(pid < CLIST_NMAX);
    return CLIST_NMAX * s + pid;
}

_I_ void contact_decode_id(uint c,      int *s, int *pid)
{

    *pid = c % CLIST_NMAX;
    *s   = c / CLIST_NMAX;
}

#undef _I_
#endif

struct Contact;
struct int3;
struct PairParams;

void cnt_ini(int maxp, int rank, int3 L, int nobj,      Contact **c);
void cnt_fin(Contact *c);

void cnt_build_cells(int nw, const PairParams **prms, const PaWrap *pw,      Contact *c);
void cnt_bulk(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw);
void cnt_halo(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw, Pap26 all_pp, Fop26 all_ff, const int *all_counts);
int get_id_inter(int a, int b);
int contact_scan(int n, const int cc[], int ss[]);

template <typename Par>
static __device__ void cnt_pair(Par params, PairPa a, PairPa b, float rnd,
                            float *fx, float *fy, float *fz) {
    PairFo f;
    pair_force(&params, a, b, rnd,      &f);
    *fx = f.x; *fy = f.y; *fz = f.z;
}

enum {CNT_FULL, CNT_EMPTY};
struct CntMap {
    int  org0, org1, org2;
    int cnt0, cnt1, cnt2;
};


static __device__ int cnt_endp(const CntMap m, int i) { return i >= m.cnt2; }

static __device__ int cnt_m2id(const CntMap m, int i) {

    int m1, m2;
    int id;
    m1 = (int)(i >= m.cnt0);
    m2 = (int)(i >= m.cnt1);
    id = i + (m2 ? m.org2 : m1 ? m.org1 : m.org0);
    return id;
}
static __device__ int cnt_tex2map(int3 L, int zplane, float x, float y, float z, const int *starts,      CntMap *m) {
    int cnt0, cnt1, cnt2, org0;
    int org1, org2;
    int xcenter, xstart, xcount;
    int ycenter, zcenter, zmy;
    bool zvalid;
    int count0, count1, count2;
    int cid0, cid1, cid2;

    xcenter = L.x / 2 + (int)floorf(x);
    xstart = max(0, xcenter - 1);
    xcount = min(L.x, xcenter + 2) - xstart;

    if (xcenter - 1 >= L.x || xcenter + 2 <= 0) return CNT_EMPTY;

    ycenter = L.y / 2 + (int)floorf(y);

    zcenter = L.z / 2 + (int)floorf(z);
    zmy = zcenter - 1 + zplane;
    zvalid = zmy >= 0 && zmy < L.z;

    count0 = count1 = count2 = 0;

    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < L.y) {
        cid0 = xstart + L.x * (ycenter - 1 + L.y * zmy);
        org0 = starts[cid0];
        count0 = starts[cid0 + xcount] - org0;
    }

    if (zvalid && ycenter >= 0 && ycenter < L.y) {
        cid1 = xstart + L.x * (ycenter + L.y * zmy);
        org1 = starts[cid1];
        count1 = starts[cid1 + xcount] - org1;
    }

    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < L.y) {
        cid2 = xstart + L.x * (ycenter + 1 + L.y * zmy);
        org2 = starts[cid2];
        count2 = starts[cid2 + xcount] - org2;
    }

    cnt0 = count0;
    cnt1 = count0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->cnt0 = cnt0; m->cnt1 = cnt1; m->cnt2 = cnt2;
    return CNT_FULL;
}
static __device__ int cnt_r2map(int3 L, int zplane, int n, float x, float y, float z, const int *starts,      CntMap *m) {

    int cnt0, cnt1, cnt2, org0;
    int org1, org2;
    int xcenter, xstart, xcount;
    int ycenter, zcenter, zmy;
    bool zvalid;
    int count0, count1, count2;
    int cid0, cid1, cid2;

    xcenter = min(L.x - 1, max(0, L.x / 2 + (int)floorf(x)));
    xstart = max(0, xcenter - 1);
    xcount = min(L.x, xcenter + 2) - xstart;
    if (xcenter - 1 >= L.x || xcenter + 2 <= 0) return CNT_EMPTY;
    ycenter = min(L.y - 1, max(0, L.y / 2 + (int)floorf(y)));
    zcenter = min(L.z - 1, max(0, L.z / 2 + (int)floorf(z)));
    zmy = zcenter - 1 + zplane;
    zvalid = zmy >= 0 && zmy < L.z;
    count0 = count1 = count2 = 0;
    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < L.y) {
        cid0 = xstart + L.x * (ycenter - 1 + L.y * zmy);
        org0 = starts[cid0];
        count0 = starts[cid0 + xcount] - org0;
    }
    if (zvalid && ycenter >= 0 && ycenter < L.y) {
        cid1 = xstart + L.x * (ycenter + L.y * zmy);
        org1 = starts[cid1];
        count1 = starts[cid1 + xcount] - org1;
    }

    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < L.y) {
        cid2 = xstart + L.x * (ycenter + 1 + L.y * zmy);
        org2 = starts[cid2];
        count2 = starts[cid2 + xcount] - org2;
    }

    cnt0 = count0;
    cnt1 = count0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->cnt0 = cnt0; m->cnt1 = cnt1; m->cnt2 = cnt2;
    return CNT_FULL;
}
static __device__ void cnt_f22p(float2 a, float2 b, float2 c, PairPa *p) {
    p->x  = a.x; p->y  = a.y; p->z  = b.x;
    p->vx = b.y; p->vy = c.x; p->vz = c.y;
}

static __device__ void cnt_fetch(int i, const float2 *pp, PairPa *p) {
    float2 a, b, c;
    a = __ldg(pp + 3 * i + 0);
    b = __ldg(pp + 3 * i + 1);
    c = __ldg(pp + 3 * i + 2);
    cnt_f22p(a, b, c, p);
}

template <typename Par>
__global__ void cnt_bulk_kernel(bool self, Par params, int3 L, int n_dst, const float2 *pp_dst,
                     const int *starts, const uint *ids_src, const float2 *pp_src,
                     float seed,      float *ff_dst, float *ff_src) {
    CntMap m;

    float fx, fy, fz, rnd;
    PairPa a, b;
    int i, gid, aid, zplane;
    float xforce, yforce, zforce;
    int mapstatus;
    int slot, objid, bid, sentry;
    uint code;

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    aid = gid / 3;
    zplane = gid % 3;

    if (aid >= n_dst) return;

    cnt_fetch(aid, pp_dst, &a);
    mapstatus = cnt_r2map(L, zplane, n_dst, a.x, a.y, a.z, starts,      &m);

    if (mapstatus == CNT_EMPTY) return;
    xforce = yforce = zforce = 0;
    for (i = 0; !cnt_endp(m, i); ++i) {
        slot = cnt_m2id(m, i);
        code = ids_src[slot];
        contact_decode_id(code,      &objid, &bid);

        if (self && aid <= bid) continue;

        cnt_fetch(bid, pp_src, &b);

        rnd = rnd::mean0var1ii(seed, aid, bid);

        cnt_pair(params, a, b, rnd,      &fx, &fy, &fz);

        xforce += fx;
        yforce += fy;
        zforce += fz;

        sentry = 3 * bid;
        atomicAdd(ff_src + sentry,     -fx);
        atomicAdd(ff_src + sentry + 1, -fy);
        atomicAdd(ff_src + sentry + 2, -fz);
    }

    atomicAdd(ff_dst + 3 * aid + 0, xforce);
    atomicAdd(ff_dst + 3 * aid + 1, yforce);
    atomicAdd(ff_dst + 3 * aid + 2, zforce);
}
template <typename Par>
static __device__
void interactions_one_p(Par params, int3 L, float seed, PairPa a, int aid,
                        const int *starts, const uint *ids_src, const float2 *pp_src,
                             float *ff_src, float *fA) {
    enum {X, Y, Z};
    CntMap m;
    int mapstatus;
    PairPa b;
    float fx, fy, fz;
    float xforce = 0, yforce = 0, zforce = 0;
    int zplane;
    int i, slot, oid, bid, sentry;
    uint code;
    float rnd;

    for (zplane = 0; zplane < 3; ++zplane) {
        mapstatus = cnt_tex2map(L, zplane, a.x, a.y, a.z, starts,      &m);
        if (mapstatus == CNT_EMPTY) continue;
        for (i = 0; !cnt_endp(m, i); ++i) {
            slot = cnt_m2id(m, i);
            code = ids_src[slot];
            contact_decode_id(code,      &oid, &bid);

            cnt_fetch(bid, pp_src,      &b);
            rnd = rnd::mean0var1ii(seed, aid, bid);
            cnt_pair(params, a, b, rnd,      &fx, &fy, &fz);
            xforce += fx;
            yforce += fy;
            zforce += fz;

            sentry = 3 * bid;
            atomicAdd(ff_src + sentry,     -fx);
            atomicAdd(ff_src + sentry + 1, -fy);
            atomicAdd(ff_src + sentry + 2, -fz);
        }
    }
    fA[X] += xforce; fA[Y] += yforce; fA[Z] += zforce;
}

template <typename Par>
__global__ void cnt_halo_kernel(Par params, int3 L, float seed,
                     const int27 starts, int n_dst, Pap26 hpp,
                     const int *cellstarts, const uint *ids_src, const float2 *pp_src,
                          Fop26 hff, float *ff_src) {
    int aid, start;
    int fid;
    PairPa a;
    float *fA;

    aid = threadIdx.x + blockDim.x * blockIdx.x;
    if (aid >= n_dst) return;

    fid = frag_dev::frag_get_fid(starts.d, aid);
    start = starts.d[fid];
    cnt_fetch(aid - start, (const float2*)hpp.d[fid], &a);
    fA = hff.d[fid][aid - start].f;
    interactions_one_p(params, L, seed, a, aid, cellstarts, ids_src, pp_src,
                            ff_src, fA);
}
struct Contact {
    int nobj;
    Clist    **cells;
    ClistMap **cmap;
    RNDunif *rgen;
    int3 L;
};
void cnt_bulk(const Contact *c, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw) {
    int i, j, self, id_prm;
    float rnd;
    const PairParams *prm;
    PairDPDLJ pv;
    const int *starts;
    const uint *ids;
    PaWrap pwi, pwj;
    FoWrap fwi, fwj;
    if (nw == 0) return;

    for (i = 0; i < nw; ++i) {
        pwi = pw[i];
        fwi = fw[i];
        for (j = 0; j <= i; ++j) {
            self = (i == j);
            pwj = pw[j];
            fwj = fw[j];

            starts = clists_get_ss(c->cells[j]);
            ids    = clist_get_ids(c->cmap[j]);

            id_prm = get_id_inter(i, j);

            prm = prms[id_prm];
            if (!prm) continue;
            UC(pair_get_view_dpd_lj(prm, &pv));

            rnd = rnd_get(c->rgen);
            KL(cnt_bulk_kernel, (k_cnf(3 * pwi.n)),
               (self, pv, c->L,
                pwi.n, (const float2*) pwi.pp,
                starts, ids, (const float2*) pwj.pp,
                rnd,      (float*) fwi.ff, (float*) fwj.ff));
        }
    }
}

static void halo_one_type(int idh, const Contact *c, int nw, const PairParams **prms,
                          PaWrap *pw, FoWrap *fw,
                          Pap26 PP, Fop26 FF, const int *counts) {
    int i, n, id_prm;
    const PairParams *prm;
    const int *cellstarts;
    const uint *ids;
    int27 starts;
    PairDPDLJ pv;
    PaWrap pwi;
    FoWrap fwi;

    n = contact_scan(26, counts, starts.d);

    for (i = 0; i < nw; ++i) {
        pwi = pw[i];
        fwi = fw[i];
        cellstarts = clists_get_ss(c->cells[i]);
        ids = clist_get_ids(c->cmap[i]);

        id_prm = get_id_inter(idh, i);
        prm = prms[id_prm];
        if (!prm) continue;

        UC(pair_get_view_dpd_lj(prm, &pv));

        KL(cnt_halo_kernel, (k_cnf(n)),
           (pv, c->L, rnd_get(c->rgen), starts,
            n, PP, cellstarts, ids, (const float2*) pwi.pp,
                 FF, (float*) fwi.ff));
    }
}

void cnt_halo(const Contact *cnt, int nw, const PairParams **prms,
              PaWrap *pw, FoWrap *fw, Pap26 all_pp, Fop26 all_ff, const int *all_counts) {
    enum {NFRAGS = 26};
    Pap26 pp;
    Fop26 ff;
    int i, j, start[NFRAGS] = {0};
    const int *counts;

    for (i = 0; i < nw; i++) {
        counts = all_counts + NFRAGS * i;
        for (j = 0; j < NFRAGS; ++j) {
            pp.d[j] = all_pp.d[j] + start[j];
            ff.d[j] = all_ff.d[j] + start[j];
            start[j] += counts[j];
        }
        halo_one_type(i, cnt, nw, prms, pw, fw, pp, ff, counts);
    }
}
enum {
    CID_SUBINDEX,
    CN_SUBINDICES
};

void cnt_ini(int maxp, int rank, int3 L, int nobj,      Contact **cnt) {
    EMALLOC(1, cnt);
    Contact *c = *cnt;

    c->L = L;
    c->nobj = nobj;

    EMALLOC(nobj, &c->cells);
    EMALLOC(nobj, &c->cmap);

    for (int i = 0; i < nobj; ++i) {
        clist_ini(L.x, L.y, L.z,      &c->cells[i]);
        clist_ini_map(maxp, CN_SUBINDICES, c->cells[i],      &c->cmap[i]);
    }
    UC(rnd_ini(7119 - rank, 187 + rank, 18278, 15674,      &c->rgen));
}

void cnt_fin(Contact *c) {
    for (int i = 0; i < c->nobj; ++i) {
        clist_fin(     c->cells[i]);
        clist_fin_map(     c->cmap[i]);
    }
    UC(rnd_fin(c->rgen));
    EFREE(c->cells);
    EFREE(c->cmap);
    EFREE(c);
}

static void build_cells(const PaWrap *w, Clist *c, ClistMap *m) {
    static const bool project = true;
    PartList lp;
    const int cc[] = {w->n};

    lp.pp = w->pp;
    lp.deathlist = NULL;

    clist_ini_counts(c);
    clist_subindex(project, CID_SUBINDEX, w->n, lp,      c, m);
    clist_build_map(cc,      c, m);
}

static bool has_work(int i, int, const PairParams **prms) {
    int j, k;
    for (j = 0; j <= i; ++j) {
        k = get_id_inter(i, j);
        if (prms[k]) return true;
    }
    return false;
}

void cnt_build_cells(int nw, const PairParams **prms, const PaWrap *pw,      Contact *c) {
    for (int i = 0; i < nw; ++i) {
        if (!has_work(i, nw, prms)) continue;
        build_cells(&pw[i], c->cells[i], c->cmap[i]);
    }
}
