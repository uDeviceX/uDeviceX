#include <mpi.h>
#include "common.h"
#include <conf.h>
#include "collision.h"

#include "mrescue.h"

namespace mrescue
{
#define _DH_ __device__ __host__
    
int *tags_hst, *tags_dev;
    
static _DH_ int min2(int a, int b) {return a < b ? a : b;}
static _DH_ int max2(int a, int b) {return a < b ? b : a;}

void ini(int n)
{
    tags_hst = new int[n];
    CC(cudaMalloc(&tags_dev, n*sizeof(int)));
}

void fin()
{
    delete[] tags_hst;
    CC(cudaFree(tags_dev));
}

static _DH_ void project_t(const float *a, const float *b, const float *c,
                           const float *va, const float *vb, const float *vc,
                           const float *r, /**/ float *rp, float *vp, float *n)
{
    const float ab[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    const float ac[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
    const float ar[3] = {r[0]-a[0], r[1]-a[1], r[2]-a[2]};

    n[0] = ab[1]*ac[2] - ab[2]*ac[1];
    n[1] = ab[2]*ac[0] - ab[0]*ac[2];
    n[2] = ab[0]*ac[1] - ab[1]*ac[0];
        
#define dot(x, y) (x[0]*y[0] + x[1]*y[1] + x[2]*y[2])
    {
        const float s = 1.f / sqrt(dot(n,n));
        n[0] *= s; n[1] *= s; n[2] *= s;
    }
        
    const float arn = dot(ar, n);
    const float g[3] = {r[0] - arn * n[0] - a[0],
                        r[1] - arn * n[1] - a[1],
                        r[2] - arn * n[2] - a[2]};
        
    float u, v;
    {
        const float ga1 = dot(g, ab);
        const float ga2 = dot(g, ac);
        const float a11 = dot(ab, ab);
        const float a12 = dot(ab, ac);
        const float a22 = dot(ac, ac);

        const float fac = 1.f / (a11*a22 - a12*a12);
            
        u = (ga1 * a22 - ga2 * a12) * fac;
        v = (ga2 * a11 - ga1 * a12) * fac;
    }
        
#undef dot
        
    // project (u,v) onto unit triangle

    if ( (v > u - 1) && (v < u + 1) && (v > 1 - u) )
    {
        const float a_ = 0.5f * (u + v - 1);
        u -= a_;
        v -= a_;
    }
    else
    {
        u = max2(min2(1.f, u), 0.f);
        v = max2(min2(v, 1.f-u), 0.f);
    }
        
    // compute projected point
    const float wa = 1 - u - v;

    rp[0] = wa + a[0] + u * b[0] + v * c[0];
    rp[1] = wa + a[1] + u * b[1] + v * c[1];
    rp[2] = wa + a[2] + u * b[2] + v * c[2];

    vp[0] = wa + va[0] + u * vb[0] + v * vc[0];
    vp[1] = wa + va[1] + u * vb[1] + v * vc[1];
    vp[2] = wa + va[2] + u * vb[2] + v * vc[2];
}

#include <curand.h>
#include <curand_kernel.h>
    
static _DH_ void rescue_1p(const Particle *vv, const int *tt, const int nt, const int sid, const int nv,
                           const int *tcstarts, const int *tccounts, const int *tcids, unsigned long seed, /**/ Particle *p)
{        
    float dr2b = 1000.f, rpb[3] = {0}, vpb[3] = {0}, nb[3] = {0};

    // check around me if there is triangles and select the closest one
        
    const int xcid = min2(max2(0, int (p->r[0] + XS/2)), XS-1);
    const int ycid = min2(max2(0, int (p->r[1] + YS/2)), YS-1);
    const int zcid = min2(max2(0, int (p->r[2] + ZS/2)), ZS-1);
    const int cid = xcid + XS * (ycid + YS * zcid);
        
    const int start = tcstarts[cid];
    const int count = tccounts[cid];
        
    for (int i = start; i < start + count; ++i)
    {
        const int btid = tcids[i];
        const int tid  = btid % nt;
        const int mid  = btid / nt;

        const int t1 = mid * nv + tt[3*tid + 0];
        const int t2 = mid * nv + tt[3*tid + 1];
        const int t3 = mid * nv + tt[3*tid + 2];
            
        const Particle pa = vv[t1];
        const Particle pb = vv[t2];
        const Particle pc = vv[t3];
                        
        float rp[3], n[3], vp[3];
        project_t(pa.r, pb.r, pc.r, pa.v, pb.v, pc.v, p->r, /**/ rp, vp, n);

        const float dr[3] = {p->r[0] - rp[0], p->r[1] - rp[1], p->r[2] - rp[2]};
        const float dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];

        if (dr2 < dr2b)
        {
            dr2b = dr2;
            rpb[0] = rp[0]; rpb[1] = rp[1]; rpb[2] = rp[2];
            vpb[0] = vp[0]; vpb[1] = vp[1]; vpb[2] = vp[2];
            nb[0] = n[0]; nb[1] = n[1]; nb[2] = n[2];
        }
    }

    // otherwise pick one randomly

#if (defined (__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    curandState_t crstate;
    curand_init (seed, threadIdx.x + blockIdx.x * blockDim.x, 0, &crstate );
#endif
        
    if (dr2b == 1000.f)
    {
#if (defined (__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
        const int tid = curand(&crstate) % nt;
#else
        const int tid = rand() % nt;
#endif

        const int t1 = sid * nv + tt[3*tid + 0];
        const int t2 = sid * nv + tt[3*tid + 1];
        const int t3 = sid * nv + tt[3*tid + 2];

        const Particle pa = vv[t1];
        const Particle pb = vv[t2];
        const Particle pc = vv[t3];
            
        rpb[0] = (pa.r[0] + pb.r[0] + pc.r[0]) * 0.333333f;
        rpb[1] = (pa.r[1] + pb.r[1] + pc.r[1]) * 0.333333f;
        rpb[2] = (pa.r[2] + pb.r[2] + pc.r[2]) * 0.333333f;

        vpb[0] = (pa.v[0] + pb.v[0] + pc.v[0]) * 0.333333f;
        vpb[1] = (pa.v[1] + pb.v[1] + pc.v[1]) * 0.333333f;
        vpb[2] = (pa.v[2] + pb.v[2] + pc.v[2]) * 0.333333f;
            
        {
            const float ab[3] = {pb.r[0]-pa.r[0], pb.r[1]-pa.r[1], pb.r[2]-pa.r[2]};
            const float ac[3] = {pc.r[0]-pa.r[0], pc.r[1]-pa.r[1], pc.r[2]-pa.r[2]};
            
            nb[0] = ab[1]*ac[2] - ab[2]*ac[1];
            nb[1] = ab[2]*ac[0] - ab[0]*ac[2];
            nb[2] = ab[0]*ac[1] - ab[1]*ac[0];
            
            const float s = 1.f / sqrt(nb[0]*nb[0] + nb[1]*nb[1] + nb[2]*nb[2]);
            nb[0] *= s; nb[1] *= s; nb[2] *= s;
        }
    }

    // new particle position
#define eps 1e-1
        
#if (defined (__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    const float u = curand_uniform(&crstate) * eps;
#else
    const float u = drand48() * eps;
#endif
    p->r[0] = rpb[0] + u * nb[0];
    p->r[1] = rpb[1] + u * nb[1];
    p->r[2] = rpb[2] + u * nb[2];

    p->v[0] = vpb[0];
    p->v[1] = vpb[1];
    p->v[2] = vpb[2];
#undef eps
}
    
void rescue_hst(const Mesh m, const Particle *i_pp, const int ns, const int n,
                const int *tcstarts, const int *tccounts, const int *tcids, /**/ Particle *pp)
{
    collision::inside_hst(pp, n, m, i_pp, ns, /**/ tags_hst);

    for (int i = 0; i < n; ++i)
    {
        const int tag = tags_hst[i];
            
        if (tag != -1)
        rescue_1p(i_pp, m.tt, m.nt, tag, m.nv, tcstarts, tccounts, tcids, rand(), /**/ pp + i);
    }
}

static __global__ void rescue_dev_k(const Particle *vv, const int *tt, const int nt, const int nv,
                                    const int *tcstarts, const int *tccounts, const int *tcids, const int *tags, const int n,
                                    unsigned long seed, /**/ Particle *pp)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
        
    const int tag = tags[i];
    if (tag == -1) return;

    Particle p = pp[i];
    rescue_1p(vv, tt, nt, tag, nv, tcstarts, tccounts, tcids, seed, /**/ &p);
    pp[i] = p;
}
    
void rescue_dev(const Mesh m, const Particle *i_pp, const int ns, const int n,
                const int *tcstarts, const int *tccounts, const int *tcids, /**/ Particle *pp)
{
    if (ns == 0 || n == 0) return;
        
    collision::inside_dev(pp, n, m, i_pp, ns, /**/ tags_dev);
    rescue_dev_k <<< k_cnf(n) >>> (i_pp, m.tt, m.nt, m.nv, tcstarts, tccounts, tcids, tags_dev, n, rand(), /**/ pp);
}
}
