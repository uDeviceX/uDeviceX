#include "../.conf.h"
#include "../common.h"

#include "roots.h"
#include "mbounce.h"

namespace mbounce
{
    enum {X, Y, Z};
    
    enum BBState
    {
        BB_SUCCESS,   /* succesfully bounced                       */
        BB_NOCROSS,   /* did not cross the plane                   */
        BB_WTRIANGLE, /* [w]rong triangle                          */
        BB_HFAIL,     /* no time solution h                        */
        BB_HNEXT      /* h triangle is not the first to be crossed */
    };

#define _DH_ __device__ __host__

#define BBOX_MARGIN 0.1f

#define debug_output
    
    template <typename T>  T min2(T a, T b) {return a < b ? a : b;}
    template <typename T>  T max2(T a, T b) {return a < b ? b : a;}

    static _DH_ void rvprev(const float *r1, const float *v1, const float *f0, /**/ float *r0, float *v0)
    {
#ifdef FORWARD_EULER
        for (int c = 0; c < 3; ++c)
        {
            v0[c] = v1[c] - f0[c] * dt;
            r0[c] = r1[c] - v0[c] * dt;
        }
#else // velocity-verlet
        for (int c = 0; c < 3; ++c)
        {
            r0[c] = r1[c] - v1[c] * dt;
            //v0[c] = v1[c] - f0[c] * dt;

            // BB assumes r0 + v0 dt = r1 for now
            v0[c] = v1[c];
        }
#endif
    }
    
    static _DH_ bool cubic_root(float a, float b, float c, float d, /**/ float *h)
    {
        #define valid(t) ((t) >= 0 && (t) <= dt)
        #define eps 1e-8
        
        if (fabs(a) > eps) // cubic
        {
            typedef double real;
            const real b_ = b /= a;
            const real c_ = c /= a;
            const real d_ = d /= a;
            
            real h1, h2, h3;
            int nsol = roots::cubic(b_, c_, d_, &h1, &h2, &h3);

            if (valid(h1))             {*h = h1; return true;}
            if (nsol > 1 && valid(h2)) {*h = h2; return true;}
            if (nsol > 2 && valid(h3)) {*h = h3; return true;}
        }
        else if(fabs(b) > eps) // quadratic
        {
            float h1, h2;
            if (!roots::quadratic(b, c, d, &h1, &h2)) return false;
            if (valid(h1)) {*h = h1; return true;}
            if (valid(h2)) {*h = h2; return true;}
        }
        else if (fabs(c) > eps) // linear
        {
            const float h1 = -d/c;
            if (valid(h1)) {*h = h1; return true;}
        }
        
        return false;
    }
    
    /* see Fedosov PhD Thesis */
    static _DH_ BBState intersect_triangle(const float *s10, const float *s20, const float *s30,
                                           const float *vs1, const float *vs2, const float *vs3,
                                           const Particle *p0,  /*io*/ float *h, /**/ float *rw, float *vw)
    {
#define diff(a, b) {a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z]}
#define cross(a, b) {a[Y] * b[Z] - a[Z] * b[Y], a[Z] * b[X] - a[X] * b[Z], a[X] * b[Y] - a[Y] * b[X]}
#define dot(a, b) (a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z])
#define apxb(a, x, b) {a[X] + (float) x * b[X], a[Y] + (float) x * b[Y], a[Z] + (float) x * b[Z]} 
        
        const float *r0 = p0->r;
        const float *v0 = p0->v;
    
        const float a1[3] = diff(s20, s10);
        const float a2[3] = diff(s30, s10);
    
        const float at1[3] = diff(vs2, vs1);
        const float at2[3] = diff(vs3, vs1);

        // n(t) = n + t*nt + t^2 * ntt
        const float n0[3] = cross(a1, a2);
        const float ntt[3] = cross(at1, at2);
        const float nt[3] = {a1[Y] * at2[Z] - a1[Z] * at2[Y]  +  at1[Y] * a2[Z] - at1[Z] * a2[Y],
                             a1[Z] * at2[X] - a1[X] * at2[Z]  +  at1[Z] * a2[X] - at1[X] * a2[Z],
                             a1[X] * at2[Y] - a1[Y] * at2[X]  +  at1[X] * a2[Y] - at1[Y] * a2[X]};
    
        const float dr0[3] = diff(r0, s10);
        
        // check intersection with plane
        {
            const float r1[3] = apxb(r0, dt, v0);
            const float s11[3] = apxb(s10, dt, vs1);

            const float n1[3] = {n0[X] + (float) dt * (nt[X] + (float) dt * ntt[X]),
                                 n0[Y] + (float) dt * (nt[Y] + (float) dt * ntt[Y]),
                                 n0[Z] + (float) dt * (nt[Z] + (float) dt * ntt[Z])};
            
            const float dr1[3] = diff(r1, s11);

            const float b0 = dot(dr0, n0);
            const float b1 = dot(dr1, n1);

            if (b0 * b1 > 0)
            return BB_NOCROSS;
        }

        // find intersection time with plane

        const float dv[3] = diff(v0, vs1);
        
        const float a = dot(ntt, dv);
        const float b = dot(ntt, dr0) + dot(nt, dv);
        const float c = dot(nt, dr0) + dot(n0, dv);
        const float d = dot(n0, dr0);

        float hl;
        
        if (!cubic_root(a, b, c, d, &hl))
        {
            //printf("%g %g %g %g\n", a, b, c, d);
            return BB_HFAIL;
        }

        if (hl < *h)
        *h = hl;
        else
        return BB_HNEXT;
        
        rw[X] = r0[X] + hl * v0[X];
        rw[Y] = r0[Y] + hl * v0[Y];
        rw[Z] = r0[Z] + hl * v0[Z];

        // check if inside triangle

        {
            const float g[3] = {rw[X] - s10[X] - hl * vs1[X],
                                rw[Y] - s10[Y] - hl * vs1[Y],
                                rw[Z] - s10[Z] - hl * vs1[Z]};

            const float a1_[3] = apxb(a1, hl, at1);
            const float a2_[3] = apxb(a2, hl, at2);
            
            const float ga1 = dot(g, a1_);
            const float ga2 = dot(g, a2_);
            const float a11 = dot(a1_, a1_);
            const float a12 = dot(a1_, a2_);
            const float a22 = dot(a2_, a2_);

            const float fac = 1.f / (a11*a22 - a12*a12);
            
            const float u = (ga1 * a22 - ga2 * a12) * fac;
            const float v = (ga2 * a11 - ga1 * a12) * fac;

            if (!((u >= 0) && (v >= 0) && (u+v <= 1)))
            return BB_WTRIANGLE;

            // linear interpolation of velocity vw
            const float w = 1 - u - v;
            vw[X] = w * vs1[X] + u * vs2[X] + v * vs3[X];
            vw[Y] = w * vs1[Y] + u * vs2[Y] + v * vs3[Y];
            vw[Z] = w * vs1[Z] + u * vs2[Z] + v * vs3[Z];
        }
        return BB_SUCCESS;
    }

    static _DH_ void lin_mom_solid(const float *v1, const float *vn, /**/ float *dP)
    {
        for (int c = 0; c < 3; ++c)
        dP[c] = -(vn[c] - v1[c]) / dt;
    }

    static _DH_ void ang_mom_solid(const float *com, const float *rw, const float *v0, const float *vn, /**/ float *dL)
    {
        const float dr[3] = {rw[X] - com[X], rw[Y] - com[Y], rw[Z] - com[Z]};
        
        dL[X] = -(dr[Y] * vn[Z] - dr[Z] * vn[Y] - dr[Y] * v0[Z] + dr[Z] * v0[Y]) / dt;
        dL[Y] = -(dr[Z] * vn[X] - dr[X] * vn[Z] - dr[Z] * v0[X] + dr[X] * v0[Z]) / dt;
        dL[Z] = -(dr[X] * vn[Y] - dr[Y] * vn[X] - dr[X] * v0[Y] + dr[Y] * v0[X]) / dt;
    }

#ifdef debug_output
    int bbstates_hst[5], dstep = 0;
    __device__ int bbstates_dev[5];
#endif


    
    static _DH_ bool find_better_intersection(const int *tt, const int it, const Particle *i_pp, const Particle *p0, /* io */ float *h, /**/ float *rw, float *vw)
    {
        // load data
        const int t1 = tt[3*it + 0];
        const int t2 = tt[3*it + 1];
        const int t3 = tt[3*it + 2];

#define revert_r(P) do {                        \
            P.r[X] -= dt * P.v[X];              \
            P.r[Y] -= dt * P.v[Y];              \
            P.r[Z] -= dt * P.v[Z];              \
        } while(0)

        Particle pA = i_pp[t1]; revert_r(pA);
        Particle pB = i_pp[t2]; revert_r(pB);
        Particle pC = i_pp[t3]; revert_r(pC);

#undef revert_r
        
        const BBState bbstate = intersect_triangle(pA.r, pB.r, pC.r, pA.v, pB.v, pC.v, p0, /* io */ h, /**/ rw, vw);

#ifdef debug_output
#if (defined (__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
        atomicAdd(bbstates_dev + bbstate, 1);
#else
        bbstates_hst[bbstate] ++;
#endif
#endif
        return bbstate == BB_SUCCESS;
    }
    
    static _DH_ void bounce_back(const Particle *p0, const float *rw, const float *vw, const float h, /**/ Particle *pn)
    {
        pn->v[X] = 2 * vw[X] - p0->v[X];
        pn->v[Y] = 2 * vw[Y] - p0->v[Y];
        pn->v[Z] = 2 * vw[Z] - p0->v[Z];

        pn->r[X] = rw[X] + (dt-h) * pn->v[X];
        pn->r[Y] = rw[Y] + (dt-h) * pn->v[Y];
        pn->r[Z] = rw[Z] + (dt-h) * pn->v[Z];
    }
    
    /* One node, no periodicity for now */
    void bounce_tcells_hst(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                           const int n, /**/ Particle *pp, Solid *ss)
    {
#ifdef debug_output
        if (dstep % steps_per_dump == 0)
        for (int c = 0; c < 4; ++c) bbstates_hst[c] = 0;
#endif
        
        for (int i = 0; i < n; ++i)
        {
            const Particle p1 = pp[i];
            
            Particle p0; rvprev(p1.r, p1.v, ff[i].f, /**/ p0.r, p0.v);

            const int xcid_ = int (p1.r[X] + XS/2);
            const int ycid_ = int (p1.r[Y] + YS/2);
            const int zcid_ = int (p1.r[Z] + ZS/2);

            float h = 2*dt; // must be higher than any valid result
            float rw[3], vw[3];

            int sid = -1;
            
            for (int zcid = max2(zcid_-1, 0); zcid <= min2(zcid_ + 1, ZS - 1); ++zcid)
            for (int ycid = max2(ycid_-1, 0); ycid <= min2(ycid_ + 1, YS - 1); ++ycid)
            for (int xcid = max2(xcid_-1, 0); xcid <= min2(xcid_ + 1, XS - 1); ++xcid)
            {
                const int cid = xcid + XS * (ycid + YS * zcid);
                const int start = tcellstarts[cid];
                const int count = tcellcounts[cid];
                
                for (int j = start; j < start + count; ++j)
                {
                    const int tid = tids[j];
                    const int it  = tid % m.nt;
                    const int mid = tid / m.nt;
                    
                    if (find_better_intersection(m.tt, it, i_pp, &p0, /*io*/ &h, /**/ rw, vw))
                    sid = mid;
                }
            }

            if (sid != -1)
            {
                Particle pn;
                bounce_back(&p0, rw, vw, h, /**/ &pn);

                float dP[3], dL[3];
                lin_mom_solid(p1.v, pn.v, /**/ dP);
                ang_mom_solid(ss[sid].com, rw, p0.v, pn.v, /**/ dL);
                
                pp[i] = pn;

                ss[sid].fo[X] += dP[X];
                ss[sid].fo[Y] += dP[Y];
                ss[sid].fo[Z] += dP[Z];

                ss[sid].to[X] += dL[X];
                ss[sid].to[Y] += dL[Y];
                ss[sid].to[Z] += dL[Z];
            }
        }

#ifdef debug_output
        if ((++dstep) % steps_per_dump == 0)
        printf("%d success, %d nocross, %d wrong triangle, %d hfailed\n",
               bbstates_hst[0], bbstates_hst[1], bbstates_hst[2], bbstates_hst[3]);
#endif
    }
}
