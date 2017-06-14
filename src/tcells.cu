#include <mpi.h>
#include <conf.h>
#include "m.h"
#include "common.h"
#include "tcells.h"

static const int NCELLS = XS * YS * ZS;

enum {X, Y, Z};

#define BBOX_MARGIN 0.1f
#define _HD_  __host__ __device__

template <typename T> _HD_ T min2(T a, T b) {return a < b ? a : b;}
template <typename T> _HD_ T max2(T a, T b) {return a < b ? b : a;}
template <typename T> _HD_ T min3(T a, T b, T c) {return min2(a, min2(b, c));}
template <typename T> _HD_ T max3(T a, T b, T c) {return max2(a, max2(b, c));}
    
static _HD_ void tbbox(const float *A, const float *B, const float *C, float *bb)
{
    bb[2*X + 0] = min3(A[X], B[X], C[X]) - BBOX_MARGIN;
    bb[2*X + 1] = max3(A[X], B[X], C[X]) + BBOX_MARGIN;
    bb[2*Y + 0] = min3(A[Y], B[Y], C[Y]) - BBOX_MARGIN;
    bb[2*Y + 1] = max3(A[Y], B[Y], C[Y]) + BBOX_MARGIN;
    bb[2*Z + 0] = min3(A[Z], B[Z], C[Z]) - BBOX_MARGIN;
    bb[2*Z + 1] = max3(A[Z], B[Z], C[Z]) + BBOX_MARGIN;
}
    
static void countt(const int nt, const int *tt, const int nv, const Particle *pp, const int ns, /**/ int *counts)
{
    memset(counts, 0, NCELLS * sizeof(int));
        
    for (int is = 0; is < ns; ++is)
    for (int it = 0; it < nt; ++it)
    {
        const int t1 = tt[3*it + 0];
        const int t2 = tt[3*it + 1];
        const int t3 = tt[3*it + 2];

        const int base = nv * is;
        
#define loadr(i) {pp[base + i].r[X], pp[base + i].r[Y], pp[base + i].r[Z]}       
        const float A[3] = loadr(t1);
        const float B[3] = loadr(t2);
        const float C[3] = loadr(t3);
#undef loadr
        float bbox[6]; tbbox(A, B, C, /**/ bbox);

        const int startx = max2(int (bbox[2*X + 0] + XS/2), 0);
        const int starty = max2(int (bbox[2*Y + 0] + YS/2), 0);
        const int startz = max2(int (bbox[2*Z + 0] + ZS/2), 0);

        const int endx = min2(int (bbox[2*X + 1] + XS/2) + 1, XS);
        const int endy = min2(int (bbox[2*Y + 1] + YS/2) + 1, YS);
        const int endz = min2(int (bbox[2*Z + 1] + ZS/2) + 1, ZS);

        for (int iz = startz; iz < endz; ++iz)
        for (int iy = starty; iy < endy; ++iy)
        for (int ix = startx; ix < endx; ++ix)
        {
            const int cid = ix + XS * (iy + YS * iz);
            ++counts[cid];
        }
    }
}

static void fill_ids(const int nt, const int *tt, const int nv, const Particle *pp, const int ns, const int *starts, /**/ int *counts, int *ids)
{
    memset(counts, 0, NCELLS * sizeof(int));
        
    for (int is = 0; is < ns; ++is)
    for (int it = 0; it < nt; ++it)
    {
        const int id = is * nt + it;
            
        const int t1 = tt[3*it + 0];
        const int t2 = tt[3*it + 1];
        const int t3 = tt[3*it + 2];
            
        const int base = nv * is;
        
#define loadr(i) {pp[base + i].r[X], pp[base + i].r[Y], pp[base + i].r[Z]}       
        const float A[3] = loadr(t1);
        const float B[3] = loadr(t2);
        const float C[3] = loadr(t3);
#undef loadr
        float bbox[6]; tbbox(A, B, C, /**/ bbox);

        const int startx = max2(int (bbox[2*X + 0] + XS/2), 0);
        const int starty = max2(int (bbox[2*Y + 0] + YS/2), 0);
        const int startz = max2(int (bbox[2*Z + 0] + ZS/2), 0);

        const int endx = min2(int (bbox[2*X + 1] + XS/2) + 1, XS);
        const int endy = min2(int (bbox[2*Y + 1] + YS/2) + 1, YS);
        const int endz = min2(int (bbox[2*Z + 1] + ZS/2) + 1, ZS);

        for (int iz = startz; iz < endz; ++iz)
        for (int iy = starty; iy < endy; ++iy)
        for (int ix = startx; ix < endx; ++ix)
        {
            const int cid = ix + XS * (iy + YS * iz);
            const int subindex = counts[cid]++;
            const int start = starts[cid];

            ids[start + subindex] = id;
        }
    }
}

static void exscan(const int *counts, int *starts)
{
    starts[0] = 0;
    for (int i = 1; i < NCELLS; ++i)
    starts[i] = starts[i-1] + counts[i-1];
}
    
void build_tcells_hst(const Mesh m, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids)
{
    countt(m.nt, m.tt, m.nv, i_pp, ns, /**/ counts);

    exscan(counts, starts);

    fill_ids(m.nt, m.tt, m.nv, i_pp, ns, starts, /**/ counts, ids);
}

namespace tckernels
{
__global__ void countt(const int nt, const int *tt, const int nv, const Particle *pp, const int ns, /**/ int *counts)
{
    const int thid = threadIdx.x + blockIdx.x * blockDim.x;

    if (thid >= nt * ns) return;
        
    const int tid = thid % nt;
    const int sid = thid / nt;

    const int t1 = tt[3*tid + 0];
    const int t2 = tt[3*tid + 1];
    const int t3 = tt[3*tid + 2];

    const int base = nv * sid;
        
#define loadr(i) {pp[base + i].r[X], pp[base + i].r[Y], pp[base + i].r[Z]}       
    const float A[3] = loadr(t1);
    const float B[3] = loadr(t2);
    const float C[3] = loadr(t3);
#undef loadr
    float bbox[6]; tbbox(A, B, C, /**/ bbox);

    const int startx = max2(int (bbox[2*X + 0] + XS/2), 0);
    const int starty = max2(int (bbox[2*Y + 0] + YS/2), 0);
    const int startz = max2(int (bbox[2*Z + 0] + ZS/2), 0);

    const int endx = min2(int (bbox[2*X + 1] + XS/2) + 1, XS);
    const int endy = min2(int (bbox[2*Y + 1] + YS/2) + 1, YS);
    const int endz = min2(int (bbox[2*Z + 1] + ZS/2) + 1, ZS);

    for (int iz = startz; iz < endz; ++iz)
    for (int iy = starty; iy < endy; ++iy)
    for (int ix = startx; ix < endx; ++ix)
    {
        const int cid = ix + XS * (iy + YS * iz);
        atomicAdd(counts + cid, 1);
    }
}

__global__ void fill_ids(const int nt, const int *tt, const int nv, const Particle *pp, const int ns, const int *starts, /**/ int *counts, int *ids)
{
    const int thid = threadIdx.x + blockIdx.x * blockDim.x;

    if (thid >= nt * ns) return;
        
    const int tid = thid % nt;
    const int sid = thid / nt;
            
    const int t1 = tt[3*tid + 0];
    const int t2 = tt[3*tid + 1];
    const int t3 = tt[3*tid + 2];

    const int base = nv * sid;
        
#define loadr(i) {pp[base + i].r[X], pp[base + i].r[Y], pp[base + i].r[Z]}       
    const float A[3] = loadr(t1);
    const float B[3] = loadr(t2);
    const float C[3] = loadr(t3);
#undef loadr
    float bbox[6]; tbbox(A, B, C, /**/ bbox);

    const int startx = max2(int (bbox[2*X + 0] + XS/2), 0);
    const int starty = max2(int (bbox[2*Y + 0] + YS/2), 0);
    const int startz = max2(int (bbox[2*Z + 0] + ZS/2), 0);

    const int endx = min2(int (bbox[2*X + 1] + XS/2) + 1, XS);
    const int endy = min2(int (bbox[2*Y + 1] + YS/2) + 1, YS);
    const int endz = min2(int (bbox[2*Z + 1] + ZS/2) + 1, ZS);

    for (int iz = startz; iz < endz; ++iz)
    for (int iy = starty; iy < endy; ++iy)
    for (int ix = startx; ix < endx; ++ix)
    {
        const int cid = ix + XS * (iy + YS * iz);
        const int subindex = atomicAdd(counts + cid, 1);
        const int start = starts[cid];

        ids[start + subindex] = thid;
    }
}
}

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

void build_tcells_dev(const Mesh m, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids)
{
    CC(cudaMemset(counts, 0, NCELLS * sizeof(int)));

    if (ns == 0)
    {
        CC(cudaMemset(starts, 0, NCELLS * sizeof(int)));
        return;
    }
    
    tckernels::countt <<< k_cnf(ns*m.nt) >>> (m.nt, m.tt, m.nv, i_pp, ns, /**/ counts);

    thrust::exclusive_scan(thrust::device, counts, counts + NCELLS, starts);

    CC(cudaMemset(counts, 0, NCELLS * sizeof(int)));
    
    tckernels::fill_ids <<< k_cnf(ns*m.nt) >>> (m.nt, m.tt, m.nv, i_pp, ns, starts, /**/ counts, ids);
}
