#include "sbounce.h"
#include <cassert>

#include "bbshapes.impl.h"

namespace solidbounce {

    #define POINTWISE_BB_MOMENTUM
    //#define debug_output
    
    using namespace bbshapes;
    enum {X, Y, Z};
    
    _DH_ void rvprev(const float *r1, const float *v1, const float *f0, /**/ float *r0, float *v0)
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
    
    _DH_ bool inside_prev(const float *r, const float *vcm, const float *om0)
    {
        const float rl[3] = {float(r[X] + dt * (vcm[X] + om0[Y] * r[Z] - om0[Z] * r[Y])),
                             float(r[Y] + dt * (vcm[Y] + om0[Z] * r[X] - om0[X] * r[Z])),
                             float(r[Z] + dt * (vcm[Z] + om0[X] * r[Y] - om0[Y] * r[X]))};
        
        return shape::inside(rl);
    }

    
    _DH_ void collision_point(const float *r0, const float *v0, const float h, /**/ float *rcol)
    {
        for (int c = 0; c < 3; ++c)
        rcol[c] = r0[c] + h * v0[c];
    }

    _DH_ void vsolid(const float *vcm, const float *om, const float *r, /**/ float *vs)
    {
        vs[X] = vcm[X] + om[Y]*r[Z] - om[Z]*r[Y];
        vs[Y] = vcm[Y] + om[Z]*r[X] - om[X]*r[Z];
        vs[Z] = vcm[Z] + om[X]*r[Y] - om[Y]*r[X];
    }

    _DH_ void bounce_particle(const float *vs, const float *rcol, const float *v0, const float h, /**/ float *rn, float *vn)
    {
        assert(h >= 0);
        assert(h <= dt);
        
        for (int c = 0; c < 3; ++c)
        {
            vn[c] = 2 * vs[c] - v0[c];
            rn[c] = rcol[c] + (dt - h) * vn[c];
        }
    }

    _DH_ void rescue_particle(const float *vcm, const float *om, /**/ float *r, float *v)
    {
        shape::rescue(/**/ r);
        vsolid(vcm, om, r, /**/ v);

        assert(!shape::inside(r));
    }

    _DH_ void lin_mom_solid(const float *v1, const float *vn, /**/ float *dP)
    {
        for (int c = 0; c < 3; ++c)
        dP[c] = -(vn[c] - v1[c]) / dt;
    }

    _DH_ void ang_mom_solid(const float *com, const float *r1, const float *rn, const float *v1, const float *vn, /**/ float *dL)
    {
        const float drn[3] = {rn[X] - com[X], rn[Y] - com[Y], rn[Z] - com[Z]};
        const float dr1[3] = {r1[X] - com[X], r1[Y] - com[Y], r1[Z] - com[Z]};
        
        dL[X] = -(drn[Y] * vn[Z] - drn[Z] * vn[Y] - dr1[Y] * v1[Z] + dr1[Z] * v1[Y]) / dt;
        dL[Y] = -(drn[Z] * vn[X] - drn[X] * vn[Z] - dr1[Z] * v1[X] + dr1[X] * v1[Z]) / dt;
        dL[Z] = -(drn[X] * vn[Y] - drn[Y] * vn[X] - dr1[X] * v1[Y] + dr1[Y] * v1[X]) / dt;
    }

#ifdef debug_output
    int nrescued, nbounced, still_in, failed, step = 0;
    __device__ int bbinfosdev[5];
#endif

    enum BBState
    {
        BB_SUCCESS,
        BB_RESCUED,
        BB_FAILED,
        BB_INSIDE,
        BB_NOBOUNCE
    };
    
    _DH_ BBState bb_part_local(const float *fp, const float *vcm, const float *om, /*o*/ Particle *p1, float *rw, float *vw, /*w*/ Particle *p0)
    {
        float h;
        
        if (!shape::inside(p1->r))
        return BB_NOBOUNCE;

        /* previous position and velocity                        */
        /* this step should be dependant on the time scheme only */
        
        rvprev(p1->r, p1->v, fp, /**/ p0->r, p0->v);

        /* rescue particles which were already in the solid   */
        /* put them back on the surface with surface velocity */

        if (inside_prev(p0->r, vcm, om))
        {
            rescue_particle(vcm, om, /**/ p1->r, p1->v);
            return BB_RESCUED;
        }
        
        /* find collision point */
        
        if (!shape::intersect(p0->r, p0->v, vcm, om, /**/ &h))
        return BB_FAILED;
        
        assert(h >= 0 );
        assert(h <= dt);
        
        collision_point(p0->r, p0->v, h, /**/ rw);
        
        /* handle collision for particle */
        
        vsolid(vcm, om, rw, /**/ vw);

        bounce_particle(vw, rw, p0->v, h, /**/ p1->r, p1->v);

        if (shape::inside(p1->r))
        return BB_INSIDE;

        return BB_SUCCESS;
    }
    
    _DH_ void r2local (const float *e0, const float *e1, const float *e2, const float *com, const float *rg, /**/ float *rl)
    {
        float x = rg[X] - com[X];
        float y = rg[Y] - com[Y];
        float z = rg[Z] - com[Z];
        
        rl[X] = x*e0[X] + y*e0[Y] + z*e0[Z];
        rl[Y] = x*e1[X] + y*e1[Y] + z*e1[Z];
        rl[Z] = x*e2[X] + y*e2[Y] + z*e2[Z];
    }

    _DH_ void r2global(const float *e0, const float *e1, const float *e2, const float *com, const float *rl, /**/ float *rg)
    {
        rg[X] = com[X] + rl[X] * e0[X] + rl[Y] * e1[X] + rl[Z] * e2[X];
        rg[Y] = com[Y] + rl[X] * e0[Y] + rl[Y] * e1[Y] + rl[Z] * e2[Y];
        rg[Z] = com[Z] + rl[X] * e0[Z] + rl[Y] * e1[Z] + rl[Z] * e2[Z];
    }

    _DH_ void v2local (const float *e0, const float *e1, const float *e2, const float *vg, /**/ float *vl)
    {
        vl[X] = vg[X]*e0[X] + vg[Y]*e0[Y] + vg[Z]*e0[Z];
        vl[Y] = vg[X]*e1[X] + vg[Y]*e1[Y] + vg[Z]*e1[Z];
        vl[Z] = vg[X]*e2[X] + vg[Y]*e2[Y] + vg[Z]*e2[Z];
    }

    _DH_ void v2global(const float *e0, const float *e1, const float *e2, const float *vl, /**/ float *vg)
    {
        vg[X] = vl[X] * e0[X] + vl[Y] * e1[X] + vl[Z] * e2[X];
        vg[Y] = vl[X] * e0[Y] + vl[Y] * e1[Y] + vl[Z] * e2[Y];
        vg[Z] = vl[X] * e0[Z] + vl[Y] * e1[Z] + vl[Z] * e2[Z];
    }
    
    void bounce_1s(const Force *ff, const int np, /**/ Particle *pp, Solid *shst)
    {
        Particle p0l, p1, pn, pnl;
        float dP[3], dL[3], vcml[3], oml[3], fl[3], rwl[3], vwl[3];

        for (int ip = 0; ip < np; ++ip)
        {
            p1 = pp[ip];
            pn = p1;

            r2local(shst->e0, shst->e1, shst->e2, shst->com, pn.r, /**/ pnl.r);
            v2local(shst->e0, shst->e1, shst->e2,            pn.v, /**/ pnl.v);
                
            v2local(shst->e0, shst->e1, shst->e2,  shst->v, /**/ vcml);
            v2local(shst->e0, shst->e1, shst->e2, shst->om, /**/  oml);
                
            v2local(shst->e0, shst->e1, shst->e2, ff[ip].f, /**/ fl);
            
            BBState bbstate = bb_part_local(fl, vcml, oml, /*o*/ &pnl, rwl, vwl, /*w*/ &p0l);
            
            r2global(shst->e0, shst->e1, shst->e2, shst->com, pnl.r, /**/ pn.r);
            v2global(shst->e0, shst->e1, shst->e2,            pnl.v, /**/ pn.v); 

#ifdef debug_output
            switch(bbstate)
            {
            case BB_SUCCESS: ++nbounced; break;
            case BB_RESCUED: ++nrescued; break;
            case BB_FAILED:  ++failed;   break;
            case BB_INSIDE:  ++still_in; break;
            }
#endif
            
            /* transfer momentum */
            
            dP[X] = dP[Y] = dP[Z] = 0;
            dL[X] = dL[Y] = dL[Z] = 0;

#if defined(POINTWISE_BB_MOMENTUM)
            if (bbstate == BB_SUCCESS)
            {
                float rw[3], v0[3];
                
                r2global(shst->e0, shst->e1, shst->e2, shst->com, rwl, /**/ rw);
                v2global(shst->e0, shst->e1, shst->e2,          p0l.v, /**/ v0); 
                
                lin_mom_solid(v0, pn.v, /**/ dP);
                ang_mom_solid(shst->com, rw, rw, v0, pn.v, /**/ dL);
            }
#endif
                
            for (int d = 0; d < 3; ++d)
            {
                shst->fo[d] += dP[d];
                shst->to[d] += dL[d];
            }

            pp[ip] = pn;
        }
    }

    void bounce(const Force *ff, const int np, const int ns, /**/ Particle *pp, Solid *shst)
    {
#ifdef debug_output
        if (step % steps_per_dump == 0)
        nbounced = nrescued = still_in = failed = 0;
#endif

        for (int j = 0; j < ns; ++j)
        bounce_1s(ff, np, /**/ pp, shst + j);
        
#ifdef debug_output
        if ((++step) % steps_per_dump == 0)
        printf("%d rescued, %d boounced, %d still in, %d failed\n", nrescued, nbounced, still_in, failed);
#endif
    }

    __device__ void warpReduceSumf3(float *x)
    {
        for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        {
            x[X] += __shfl_down(x[X], offset);
            x[Y] += __shfl_down(x[Y], offset);
            x[Z] += __shfl_down(x[Z], offset);
        }
    }

    __global__ void bounce_kernel(const Force *ff, const int np, /**/ Particle *pp, Solid *sdev)
    {
        const int pid = threadIdx.x + blockDim.x * blockIdx.x;

        float dP[3] = {0.f, 0.f, 0.f};
        float dL[3] = {0.f, 0.f, 0.f};

        if (pid < np)
        {
            Particle p0l, p1, pn, pnl;
            float vcml[3], oml[3], fl[3], rwl[3], vwl[3];
            
            p1 = pp[pid];
            pn = p1;

            r2local(sdev->e0, sdev->e1, sdev->e2, sdev->com, pn.r, /**/ pnl.r);
            v2local(sdev->e0, sdev->e1, sdev->e2,            pn.v, /**/ pnl.v);
                
            v2local(sdev->e0, sdev->e1, sdev->e2,  sdev->v, /**/ vcml);
            v2local(sdev->e0, sdev->e1, sdev->e2, sdev->om, /**/  oml);
                
            v2local(sdev->e0, sdev->e1, sdev->e2, ff[pid].f, /**/ fl);
                
            BBState bbstate = bb_part_local(fl, vcml, oml, /*o*/ &pnl, rwl, vwl, /*w*/ &p0l);

#ifdef debug_output
            if (bbstate != BB_NOBOUNCE) atomicAdd(bbinfosdev + bbstate, 1);
#endif
            
            r2global(sdev->e0, sdev->e1, sdev->e2, sdev->com, pnl.r, /**/ pn.r);
            v2global(sdev->e0, sdev->e1, sdev->e2,            pnl.v, /**/ pn.v); 
                
            /* transfer momentum */

#if defined(POINTWISE_BB_MOMENTUM)
            if (bbstate == BB_SUCCESS)
            {
                float rw[3], v0[3];
                
                r2global(sdev->e0, sdev->e1, sdev->e2, sdev->com, rwl, /**/ rw);
                v2global(sdev->e0, sdev->e1, sdev->e2,          p0l.v, /**/ v0); 
                                
                lin_mom_solid(v0, pn.v, /**/ dP);
                ang_mom_solid(sdev->com, rw, rw, v0, pn.v, /**/ dL);
            }
#endif       
            pp[pid] = pn;
        }

        /* momentum reduction */
        
        warpReduceSumf3(dP);
        warpReduceSumf3(dL);

        const float normdP = fmaxf(fmaxf(fabsf(dP[X]), fabsf(dP[Y])), fabsf(dP[Z]));
        const float normdL = fmaxf(fmaxf(fabsf(dL[X]), fabsf(dL[Y])), fabsf(dL[Z]));

        const bool warp_contribute = (normdP > 1e-12) && (normdL > 1e-12);
        
        if (warp_contribute && ((threadIdx.x & (warpSize - 1)) == 0))
        {
            atomicAdd(sdev->fo + X, dP[X]);
            atomicAdd(sdev->fo + Y, dP[Y]);
            atomicAdd(sdev->fo + Z, dP[Z]);

            atomicAdd(sdev->to + X, dL[X]);
            atomicAdd(sdev->to + Y, dL[Y]);
            atomicAdd(sdev->to + Z, dL[Z]);
        }
    }

    void bounce_nohost(const Force *ff, const int np, const int ns, /**/ Particle *pp, Solid *sdev)
    {
#ifdef debug_output
        if (step % steps_per_dump == 0)
        {
            const int zeros[5] = {0, 0, 0, 0, 0};
            CC(cudaMemcpyToSymbol(bbinfosdev, zeros, 5*sizeof(int)));
        }
#endif

        for (int j = 0; j < ns; ++j)
        bounce_kernel <<<k_cnf(np)>>> (ff, np, /**/ pp, sdev + j);

#ifdef debug_output
        if ((++step) % steps_per_dump == 0)
        {
            int bbinfos[5];
            CC(cudaMemcpyFromSymbol(bbinfos, bbinfosdev, 5*sizeof(int)));
            
            printf("%d rescued, %d boounced, %d still in, %d failed\n", bbinfos[BB_RESCUED], bbinfos[BB_SUCCESS], bbinfos[BB_INSIDE], bbinfos[BB_FAILED]);
        }
#endif
    }
}
