#include "macros.cuh"
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <assert.h>
#include <stdio.h>
#include <vector_types.h>
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
#define ERR(fmt, ...) do { \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report(); error_abort(); \
    } while (0)
#define UC(F) do { \
        error_stack_push(__FILE__, __LINE__); F; \
        if (error_get()) { error_report(); error_abort(); } \
        error_stack_pop(); \
    } while (0)

enum { ALIVE = 0, DEAD = 1 };
struct PartList { const Particle *pp; const int *deathlist; };
static __device__ bool is_dead(int i, const PartList lp) {
    return lp.deathlist ? lp.deathlist[i] == DEAD : false;
}
static __device__ float3 fetch_pos(int i, const Particle *pp) {
    return make_float3(pp[i].r[0], pp[i].r[1], pp[i].r[2]);
}
struct Scan;
void scan_apply(const int *, int, int *, Scan *);
void scan_ini(int, Scan **);
void scan_fin(Scan *);

#ifndef UDX_CLIST_DEV_COMMON
#define UDX_CLIST_DEV_COMMON
#define _I_ static __device__

enum { CLIST_NMAX = (256*256*256) };

_I_ uint clist_encode_id(int s, int pid)
{

    assert(pid < CLIST_NMAX);
    return CLIST_NMAX * s + pid;
}

_I_ void clist_decode_id(uint c,      int *s, int *pid)
{

    *pid = c % CLIST_NMAX;
    *s   = c / CLIST_NMAX;
}

#undef _I_
#endif

struct Clist {
    int3 dims;
    int ncells;
    int *starts, *counts;
};

struct ClistMap {
    int nA;
    uchar4 **ee;
    uint *ii;
    Scan *scan;
    long maxp;
};
enum {
    CLIST_VALID   = 0,
    CLIST_INVALID = 255
};

static __device__ bool clist_inside(float3 r, int3 L) {
    return
        (r.x >= -L.x/2) && (r.x < L.x/2) &&
        (r.y >= -L.y/2) && (r.y < L.y/2) &&
        (r.z >= -L.z/2) && (r.z < L.z/2)  ;
}

static __device__ int clist_project_cid(int i, const int L) {
    return i < 0 ? 0 : i >= L ? L - 1 : i;
}

static __device__ uchar4 clist_get_entry(const bool project, float3 r, int3 L) {
    uchar4 e;
    int ix, iy, iz;


    ix = (int) ((double) r.x + L.x/2);
    iy = (int) ((double) r.y + L.y/2);
    iz = (int) ((double) r.z + L.z/2);

    if (project) {
        ix = clist_project_cid(ix, L.x);
        iy = clist_project_cid(iy, L.y);
        iz = clist_project_cid(iz, L.z);
    }

    if (project || clist_inside(r, L)) e.w =   CLIST_VALID;
    else                         e.w = CLIST_INVALID;

    e.x = ix; e.y = iy; e.z = iz;
    return e;
}

static __device__ int clist_get_cid(int3 L, uchar4 e) {
    return e.x + L.x * (e.y + L.y * e.z);
}


__global__ void clist_subindex_kernel(bool project, int3 L, int n, const PartList lp,        int *counts,      uchar4 *ee) {
    int i, cid;
    float3 r;
    uchar4 e;
    bool dead;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    dead = is_dead(i, lp);
    r = fetch_pos(i, lp.pp);

    if (dead) {
        e.x = e.y = e.z = 0;
        e.w = CLIST_INVALID;
    } else {
        e = clist_get_entry(project, r, L);
        cid = clist_get_cid(L, e);
    }

    if (e.w != CLIST_INVALID)
        e.w = atomicAdd(counts + cid, 1);

    ee[i] = e;
}

__global__ void clist_get_ids_kernel(int array_id, int3 L, int n, const int *starts, const uchar4 *ee,      uint *ii) {
    int i, cid, id, start;
    uchar4 e;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    e = ee[i];
    cid = clist_get_cid(L, e);

    if (e.w != CLIST_INVALID) {
        start = starts[cid];
        id = start + e.w;
        ii[id] = clist_encode_id(array_id, i);
    }
}

template <typename T, int N>
__device__ void clist_fetch(const Sarray<const T*, N> src, uint i,      T *d) {
    int src_id, array_id;
    clist_decode_id(i,      &array_id, &src_id);
    *d = src.d[array_id][src_id];
}

template <typename T, int N>
__global__ void clist_gather_kernel(const Sarray<const T*, N> src, const uint *ii, int n,      T *dd) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    uint code = ii[i];
    clist_fetch(src, code,      dd + i);
}
void clist_ini(int LX, int LY, int LZ,      Clist **cl) {
    Clist *c;
    EMALLOC(1, cl);
    c = *cl;
    c->dims.x = LX;
    c->dims.y = LY;
    c->dims.z = LZ;
    c->ncells = LX * LY * LZ;

    int n = (c->ncells + 16);
    Dalloc(&c->starts, n);
    Dalloc(&c->counts, n);
}

void clist_ini_map(int maxp, int nA, const Clist *c,      ClistMap **map) {
    long i, size;
    EMALLOC(1, map);
    ClistMap *m = *map;

    UC(scan_ini(c->ncells + 16,      &m->scan));

    m->maxp = maxp;

    m->nA = nA;
    EMALLOC(nA, &m->ee);

    for (i = 0; i < nA; ++i) Dalloc(&m->ee[i], maxp);

    size = nA * maxp;
    Dalloc(&m->ii, size);
}
void clist_fin(     Clist *c) {
    Dfree(c->starts);
    Dfree(c->counts);
    EFREE(c);
}

void clist_fin_map(ClistMap *m) {
    UC(scan_fin(     m->scan));

    for (int i = 0; i < m->nA; ++i) Dfree(m->ee[i]);
    EFREE(m->ee);
    Dfree(m->ii);
    EFREE(m);
}
enum {LOCAL, REMOTE};

const int* clists_get_ss(const Clist *c) {return c->starts;}
const int* clists_get_cc(const Clist *c) {return c->counts;}
int3       clists_get_dim(const Clist *c)  {return c->dims;}
int        clists_get_n(const Clist *c)  {return c->ncells;}
const uint* clist_get_ids(const ClistMap *m) {return m->ii;}

void clist_ini_counts(     Clist *c) {
    if (c->ncells) CC(d::MemsetAsync(c->counts, 0, (c->ncells + 16) * sizeof(int)));
}

static void check_input_size(int id, long n, const ClistMap *m) {
    long cap = m->maxp;
    if (n > cap)
        ERR("Too many input particles for array %d (%ld / %ld)", id, n, cap);
}

static void comp_subindices(bool project, int n, const PartList lp, int3 dims,      int *cc, uchar4 *ee) {
    if (n) KL(clist_subindex_kernel, (k_cnf(n)), (project, dims, n, lp,        cc,      ee));
}

void clist_subindex(bool project, int aid, int n, const PartList lp,      Clist *c, ClistMap *m) {
    UC(check_input_size(aid, n, m));
    UC(comp_subindices(project, n, lp, c->dims,      c->counts, m->ee[aid]));
}

void clist_subindex_local(int n, const PartList lp,      Clist *c, ClistMap *m) {
    UC(clist_subindex(false, LOCAL, n, lp,      c, m));
}

void clist_subindex_remote(int n, const PartList lp,      Clist *c, ClistMap *m) {
    UC(clist_subindex(false, REMOTE, n, lp,      c, m));
}

void clist_build_map(const int nn[],      Clist *c, ClistMap *m) {
    int nc, *cc, *ss, n, i;
    const uchar4 *ee;
    uint *ii = m->ii;
    int3 dims = c->dims;
    nc = c->ncells;
    cc = c->counts;
    ss = c->starts;

    scan_apply(cc, nc + 16,      ss,       m->scan);

    for (i = 0; i < m->nA; ++i) {
        n = nn[i];
        ee = m->ee[i];
        UC(check_input_size(i, n, m));
        if (n) KL(clist_get_ids_kernel, (k_cnf(n)), (i, dims, n, ss, ee,      ii));
    }
}

static void check_map_capacity(long n, const ClistMap *m) {
    long cap = m->maxp * m->nA;
    if (n > cap)
        ERR("Too many particles for this cell list (%ld / %ld)", n, cap);
}

void clist_gather_pp(const Particle *pplo, const Particle *ppre, const ClistMap *m, long nout,      Particle *ppout) {
    Sarray <const Particle*, 2> src = {pplo, ppre};
    UC(check_map_capacity(nout, m));
    if (nout)
        KL(clist_gather_kernel, (k_cnf(nout)), (src, m->ii, nout,      ppout));
}

void clist_gather_ii(const int *iilo, const int *iire, const ClistMap *m, long nout,      int *iiout) {
    Sarray <const int*, 2> src = {iilo, iire};
    UC(check_map_capacity(nout, m));
    if (nout)
        KL(clist_gather_kernel, (k_cnf(nout)), (src, m->ii, nout,      iiout));
}

void clist_build(int nlo, int nout, const Particle *pplo,      Particle *ppout, Clist *c, ClistMap *m) {
    const int nn[] = {nlo, 0};
    PartList lp;
    lp.pp = pplo;
    lp.deathlist = NULL;
    UC(clist_ini_counts(     c));
    UC(clist_subindex_local (nlo, lp,      c, m));
    UC(clist_build_map(nn,      c, m));
    UC(clist_gather_pp(pplo, NULL, m, nout, ppout));
}

