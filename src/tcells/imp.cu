#include <cstdio>
#include <conf.h>
#include "inc/conf.h"
#include "common.h"
#include "m.h"
#include "cc.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "scan/int.h"
#include "tcells/int.h"

#include "kl.h"

namespace tcells {
namespace sub {

enum { NCELLS = XS * YS * ZS };


enum {X, Y, Z};

#define BBOX_MARGIN 0.1f
#define _HD_  __host__ __device__

template <typename T> _HD_ T min3(T a, T b, T c) {return min(a, min(b, c));}
template <typename T> _HD_ T max3(T a, T b, T c) {return max(a, max(b, c));}

static __host__ __device__ void loadr(const Particle *pp, int i, /**/ float r[3]) {
    Particle p = pp[i];
    r[X] = p.r[X];
    r[Y] = p.r[Y];
    r[Z] = p.r[Z];
}

static __host__ __device__ void loadt(const int *tt, int i, /**/ int t[3]) {
    t[0] = tt[3*i + 0];
    t[1] = tt[3*i + 1];
    t[2] = tt[3*i + 2];
}

static _HD_ void tbbox(const float *A, const float *B, const float *C, float *bb) {
    bb[2*X + 0] = min3(A[X], B[X], C[X]) - BBOX_MARGIN;
    bb[2*X + 1] = max3(A[X], B[X], C[X]) + BBOX_MARGIN;
    bb[2*Y + 0] = min3(A[Y], B[Y], C[Y]) - BBOX_MARGIN;
    bb[2*Y + 1] = max3(A[Y], B[Y], C[Y]) + BBOX_MARGIN;
    bb[2*Z + 0] = min3(A[Z], B[Z], C[Z]) - BBOX_MARGIN;
    bb[2*Z + 1] = max3(A[Z], B[Z], C[Z]) + BBOX_MARGIN;
}
    
static void countt(const int nt, const int *tt, const int nv, const Particle *pp, const int ns, /**/ int *counts) {
    memset(counts, 0, NCELLS * sizeof(int));
    float A[3], B[3], C[3], bbox[6];
    int t[3];
        
    for (int is = 0; is < ns; ++is)
    for (int it = 0; it < nt; ++it) {
        loadt(tt, it, /**/ t);

        const int base = nv * is;

        loadr(pp, base + t[0], /**/ A);
        loadr(pp, base + t[1], /**/ B);
        loadr(pp, base + t[2], /**/ C);
            
        tbbox(A, B, C, /**/ bbox);

        const int startx = max(int (bbox[2*X + 0] + XS/2), 0);
        const int starty = max(int (bbox[2*Y + 0] + YS/2), 0);
        const int startz = max(int (bbox[2*Z + 0] + ZS/2), 0);

        const int endx = min(int (bbox[2*X + 1] + XS/2) + 1, XS);
        const int endy = min(int (bbox[2*Y + 1] + YS/2) + 1, YS);
        const int endz = min(int (bbox[2*Z + 1] + ZS/2) + 1, ZS);

        for (int iz = startz; iz < endz; ++iz)
        for (int iy = starty; iy < endy; ++iy)
        for (int ix = startx; ix < endx; ++ix) {
            const int cid = ix + XS * (iy + YS * iz);
            ++counts[cid];
        }
    }
}

static void fill_ids(const int nt, const int *tt, const int nv, const Particle *pp, const int ns, const int *starts, /**/ int *counts, int *ids) {
    memset(counts, 0, NCELLS * sizeof(int));
    float A[3], B[3], C[3], bbox[6];
    int t[3];

    tbbox(A, B, C, /**/ bbox);
        
    for (int is = 0; is < ns; ++is)
    for (int it = 0; it < nt; ++it) {
        const int id = is * nt + it;

        loadt(tt, it, /**/ t);

        const int base = nv * is;

        loadr(pp, base + t[0], /**/ A);
        loadr(pp, base + t[1], /**/ B);
        loadr(pp, base + t[2], /**/ C);
            
        tbbox(A, B, C, /**/ bbox);

        const int startx = max(int (bbox[2*X + 0] + XS/2), 0);
        const int starty = max(int (bbox[2*Y + 0] + YS/2), 0);
        const int startz = max(int (bbox[2*Z + 0] + ZS/2), 0);

        const int endx = min(int (bbox[2*X + 1] + XS/2) + 1, XS);
        const int endy = min(int (bbox[2*Y + 1] + YS/2) + 1, YS);
        const int endz = min(int (bbox[2*Z + 1] + ZS/2) + 1, ZS);

        for (int iz = startz; iz < endz; ++iz)
        for (int iy = starty; iy < endy; ++iy)
        for (int ix = startx; ix < endx; ++ix) {
            const int cid = ix + XS * (iy + YS * iz);
            const int subindex = counts[cid]++;
            const int start = starts[cid];

            ids[start + subindex] = id;
        }
    }
}

static void exscan(const int *counts, int *starts) {
    starts[0] = 0;
    for (int i = 1; i < NCELLS; ++i)
    starts[i] = starts[i-1] + counts[i-1];
}
    
void build_hst(const Mesh m, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids) {
    countt(m.nt, m.tt, m.nv, i_pp, ns, /**/ counts);

    exscan(counts, starts);

    fill_ids(m.nt, m.tt, m.nv, i_pp, ns, starts, /**/ counts, ids);
}

namespace tckernels
{
__global__ void countt(const int nt, const int *tt, const int nv, const Particle *pp, const int ns, /**/ int *counts) {
    const int thid = threadIdx.x + blockIdx.x * blockDim.x;
    float A[3], B[3], C[3], bbox[6];
    int t[3];
        
    if (thid >= nt * ns) return;
        
    const int tid = thid % nt;
    const int sid = thid / nt;

    loadt(tt, tid, /**/ t);

    const int base = nv * sid;

    loadr(pp, base + t[0], /**/ A);
    loadr(pp, base + t[1], /**/ B);
    loadr(pp, base + t[2], /**/ C);

    tbbox(A, B, C, /**/ bbox);

    const int startx = max(int (bbox[2*X + 0] + XS/2), 0);
    const int starty = max(int (bbox[2*Y + 0] + YS/2), 0);
    const int startz = max(int (bbox[2*Z + 0] + ZS/2), 0);

    const int endx = min(int (bbox[2*X + 1] + XS/2) + 1, XS);
    const int endy = min(int (bbox[2*Y + 1] + YS/2) + 1, YS);
    const int endz = min(int (bbox[2*Z + 1] + ZS/2) + 1, ZS);

    for (int iz = startz; iz < endz; ++iz)
    for (int iy = starty; iy < endy; ++iy)
    for (int ix = startx; ix < endx; ++ix) {
        const int cid = ix + XS * (iy + YS * iz);
        atomicAdd(counts + cid, 1);
    }
}

__global__ void fill_ids(const int nt, const int *tt, const int nv, const Particle *pp, const int ns, const int *starts, /**/ int *counts, int *ids) {
    const int thid = threadIdx.x + blockIdx.x * blockDim.x;
    float A[3], B[3], C[3], bbox[6];
    int t[3];
    
    if (thid >= nt * ns) return;
        
    const int tid = thid % nt;
    const int sid = thid / nt;
            
    loadt(tt, tid, /**/ t);

    const int base = nv * sid;

    loadr(pp, base + t[0], /**/ A);
    loadr(pp, base + t[1], /**/ B);
    loadr(pp, base + t[2], /**/ C);

    tbbox(A, B, C, /**/ bbox);

    const int startx = max(int (bbox[2*X + 0] + XS/2), 0);
    const int starty = max(int (bbox[2*Y + 0] + YS/2), 0);
    const int startz = max(int (bbox[2*Z + 0] + ZS/2), 0);

    const int endx = min(int (bbox[2*X + 1] + XS/2) + 1, XS);
    const int endy = min(int (bbox[2*Y + 1] + YS/2) + 1, YS);
    const int endz = min(int (bbox[2*Z + 1] + ZS/2) + 1, ZS);

    for (int iz = startz; iz < endz; ++iz)
    for (int iy = starty; iy < endy; ++iy)
    for (int ix = startx; ix < endx; ++ix) {
        const int cid = ix + XS * (iy + YS * iz);
        const int subindex = atomicAdd(counts + cid, 1);
        const int start = starts[cid];

        ids[start + subindex] = thid;
    }
}
}

void build_dev(const Mesh m, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids, /*w*/ scan::Work *w) {
    CC(cudaMemsetAsync(counts, 0, NCELLS * sizeof(int)));
    CC(cudaMemsetAsync(starts, 0, NCELLS * sizeof(int)));

    if (ns == 0) return;
    
    KL(tckernels::countt, (k_cnf(ns*m.nt)), (m.nt, m.tt, m.nv, i_pp, ns, /**/ counts));
    scan::scan(counts, NCELLS, /**/ starts, /*w*/ w);
    CC(cudaMemsetAsync(counts, 0, NCELLS * sizeof(int)));
    KL(tckernels::fill_ids, (k_cnf(ns*m.nt)), (m.nt, m.tt, m.nv, i_pp, ns, starts, /**/ counts, ids));
}

} // sub
} // tcells
