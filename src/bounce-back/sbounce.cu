#include "sbounce.h"
#include "../../last-bit/last-bit.h"
#include <cassert>

namespace solidbounce {

    enum {X, Y, Z};
        
    // from forward Euler
    _DH_ void rvprev(const float *r1, const float *v1, const float *f0, /**/ float *r0, float *v0)
    {
        for (int c = 0; c < 3; ++c)
        {
            v0[c] = v1[c] - f0[c] * dt;
            r0[c] = r1[c] - v0[c] * dt;
        }
    }
    
    /*
      return true if a root h lies in [0, dt] (output h), false otherwise
      if 2 roots in [0, dt], smallest root h0 is returned (first collision)
    */
    _DH_ bool robust_quadratic_roots(const float a, const float b, const float c, /**/ float *h)
    {
        float D, h0, h1;
        int sgnb;

        sgnb = b > 0 ? 1 : -1;
        D = b*b - 4*a*c;

        if (D < 0) return false;
        
        h0 = (-b - sgnb * sqrt(D)) / (2 * a);
        h1 = c / (a * h0);
        
        if (h0 > h1)
        {
            float htmp = h1;
            h1 = h0; h0 = htmp;
        }

        if (h0 >= 0 && h0 <= dt) {*h = h0; return true;}
        if (h1 >= 0 && h1 <= dt) {*h = h1; return true;}
        
        return false;
    }

#if defined(rsph)

#define shape sphere

    namespace sphere
    {
        #define rsph_bb rsph

        _DH_ bool inside(const float *r)
        {
            return r[X] * r[X] + r[Y] * r[Y] + r[Z] * r[Z] < rsph_bb * rsph_bb;
        }

        _DH_ bool intersect(const float *r0, const float *v0, const float *om0, /**/ float *h)
        {
            float r0x = r0[X], r0y = r0[Y], r0z = r0[Z];
            float v0x = v0[X], v0y = v0[Y], v0z = v0[Z];
                        
            const float a = v0x*v0x + v0y*v0y + v0z*v0z;
            
            const float b = 2 * (r0x * v0x + r0y * v0y + r0z * v0z);
            const float c = r0x * r0x + r0y * r0y + r0z * r0z - rsph_bb * rsph_bb;
        
            return robust_quadratic_roots(a, b, c, h);
        }

        _DH_ void rescue(float *r)
        {
            float scale = (rsph_bb + 1e-6) / sqrt(r[X] * r[X] + r[Y] * r[Y] + r[Z] * r[Z]);

            r[X] *= scale;
            r[Y] *= scale;
            r[Z] *= scale;
        }
    }
    
#elif defined(rcyl)

#define shape cylinder
    
    namespace cylinder
    {   
        #define rcyl_bb rcyl

        _DH_ bool inside(const float *r)
        {
            return r[X] * r[X] + r[Y] * r[Y] < rcyl_bb * rcyl_bb;
        }

        /* output h between 0 and dt */
        _DH_ bool intersect(const float *r0, const float *v0, const float *om0, /**/ float *h)
        {
            float r0x = r0[X], r0y = r0[Y];
            float v0x = v0[X], v0y = v0[Y];

            const float a = v0x * v0x + v0y * v0y;
            
            const float b = 2 * (r0x * v0x + r0y * v0y);
                        
            const float c = r0x * r0x + r0y * r0y - rcyl_bb * rcyl_bb;

            return robust_quadratic_roots(a, b, c, h);
        }

        _DH_ void rescue(float *r)
        {
            float scale = (rcyl_bb + 1e-6) / sqrt(r[X] * r[X] + r[Y] * r[Y]);

            r[X] *= scale;
            r[Y] *= scale;
        }
    }

#elif defined(a2_ellipse)

#define shape ellipse // "extruded" ellipse x^2/2^ + y^2/b^2 = 1
    
    namespace ellipse
    {
#define rcyl_bb rcyl

#define a2_bb a2_ellipse 
#define b2_bb b2_ellipse

        _DH_ bool inside(const float *r)
        {
            const float x = r[X];
            const float y = r[Y];
            
            return x*x / a2_bb + y*y / b2_bb < 1;
        }
        
        /* output h between 0 and dt */
        // for now: assume vcm = 0
        _DH_ bool intersect(const float *r0, const float *v0, const float *om0, /**/ float *h)
        {
            const float r0x = r0[X], r0y = r0[Y];
            const float v0x = v0[X], v0y = v0[Y];

            const float om0z = -om0[Z];
            
            const float v0x_ = v0x - om0z * (r0y + dt * v0y);
            const float v0y_ = v0y + om0z * (r0x + dt * v0x);

            const float r0x_ = r0x + dt * om0z * (r0y + dt * v0y);
            const float r0y_ = r0y - dt * om0z * (r0x + dt * v0x);
            
            const float a = v0x_*v0x_ / a2_bb + v0y_*v0y_ / b2_bb;
            
            const float b = 2 * (r0x_ * v0x_ / a2_bb + r0y_ * v0y_ / b2_bb);
                        
            const float c = r0x_ * r0x_ / a2_bb + r0y_ * r0y_ / b2_bb - 1;

            return robust_quadratic_roots(a, b, c, h);
        }

        _DH_ void rescue(float *r) // cheap rescue
        {
            float scale = (1 + 1e-6) / sqrt(r[X] * r[X] / a2_bb + r[Y] * r[Y] / b2_bb);

            r[X] *= scale;
            r[Y] *= scale;
        }
    }
    
#else

#define shape none
    namespace none
    {
        _DH_ bool inside(const float *r)
        {
            printf("solidbounce: not implemented\n");
            exit(1);

            return false;
        }

        _DH_ bool intersect(const float *r0, const float *v0, const float *om0, /**/ float *h)
        {
            printf("solidbounce: not implemented\n");
            exit(1);

            return false;
        }

        _DH_ void rescue(float *r)
        {
            printf("solidbounce: not implemented\n");
            exit(1);
        }
    }
    
#endif

    _DH_ bool inside_prev(const float *r, const float *om0)
    {
        float rl[3] = {r[X] + dt * (om0[Y] * r[Z] - om0[Z] * r[Y]),
                       r[Y] + dt * (om0[Z] * r[X] - om0[X] * r[Z]),
                       r[Z] + dt * (om0[X] * r[Y] - om0[Y] * r[X])};
        
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
        //vsolid(vcm, om, r, /**/ v);

        assert(!shape::inside(r));
    }

    _DH_ void lin_mom_solid(const float *v1, const float *vn, /**/ float *dF)
    {
        for (int c = 0; c < 3; ++c)
        dF[c] = -(vn[c] - v1[c]) / dt;
    }

    _DH_ void ang_mom_solid(const float *r1, const float *rn, const float *v1, const float *vn, /**/ float *dL)
    {
        dL[X] = -(rn[Y] * vn[Z] - rn[Z] * vn[Y] - r1[Y] * v1[Z] + r1[Z] * v1[Y]) / dt;
        dL[Y] = -(rn[Z] * vn[X] - rn[X] * vn[Z] - r1[Z] * v1[X] + r1[X] * v1[Z]) / dt;
        dL[Z] = -(rn[X] * vn[Y] - rn[Y] * vn[X] - r1[X] * v1[Y] + r1[Y] * v1[X]) / dt;
    }

#define debug_output
#ifdef debug_output
    int nrescued, nbounced, still_in, failed, step = 0;
    FILE * fdebug;
#endif
    
    __host__ void bounce_part_local(const float *fp, const float *vcm, const float *om, /*o*/ Particle *p1, /*w*/ Particle *p0)
    {
        float rcol[3] = {0, 0, 0}, vs[3] = {0, 0, 0};
        float h;
        
        if (!shape::inside(p1->r))
        return;

        /* previous position and velocity                        */
        /* this step should be dependant on the time scheme only */
        
        rvprev(p1->r, p1->v, fp, /**/ p0->r, p0->v);

        /* rescue particles which were already in the solid   */
        /* put them back on the surface with surface velocity */

        if (inside_prev(p0->r, om))
        {
            rescue_particle(vcm, om, /**/ p1->r, p1->v);
#ifdef debug_output
            ++nrescued;
#endif
            return;
        }
        
        /* find collision point */
        
        if (!shape::intersect(p0->r, p0->v, om, /**/ &h))
        {
            // particle will be rescued at next timestep
#ifdef debug_output
            ++failed;
#endif
            return;
        }
        
        assert(h >= 0 );
        assert(h <= dt);
        
        collision_point(p0->r, p0->v, h, /**/ rcol);
        
        /* handle collision for particle */
        
        vsolid(vcm, om, rcol, /**/ vs);

#ifdef debug_output

        #define db(...) fprintf (fdebug, __VA_ARGS__)
        
        db("%+.10e %+.10e %+.10e %+.10e %+.10e %+.10e ", p0->r[X], p0->r[Y], p0->r[Z], p0->v[X], p0->v[Y], p0->v[Z]);
        db("%+.10e %+.10e %+.10e %+.10e %+.10e %+.10e ", p1->r[X], p1->r[Y], p1->r[Z], p1->v[X], p1->v[Y], p1->v[Z]);
        db("%+.10e %+.10e %+.10e ", vs[X], vs[Y], vs[Z]);
        db("%+.10e %+.10e %+.10e ", rcol[X], rcol[Y], rcol[Z]);
#endif        

        bounce_particle(vs, rcol, p0->v, h, /**/ p1->r, p1->v);

#ifdef debug_output
        db("%+.10e %+.10e %+.10e %+.10e %+.10e %+.10e ", p1->r[X], p1->r[Y], p1->r[Z], p1->v[X], p1->v[Y], p1->v[Z]);
        
        if (shape::inside(p1->r))
        {
            ++still_in;
            db(":inside:\n");
        }
        else
        {
            ++nbounced;
            db(":success:\n");
        }
#endif
    }

    _DH_ void bb_part_local(const float *fp, const float *vcm, const float *om, /*o*/ Particle *p1, /*w*/ Particle *p0)
    {
        float rcol[3] = {0, 0, 0}, vs[3] = {0, 0, 0};
        float h;
        
        if (!shape::inside(p1->r))
        return;

        /* previous position and velocity                        */
        /* this step should be dependant on the time scheme only */
        
        rvprev(p1->r, p1->v, fp, /**/ p0->r, p0->v);

        /* rescue particles which were already in the solid   */
        /* put them back on the surface with surface velocity */

        if (inside_prev(p0->r, om))
        {
            rescue_particle(vcm, om, /**/ p1->r, p1->v);
            return;
        }
        
        /* find collision point */
        
        if (!shape::intersect(p0->r, p0->v, om, /**/ &h))
        {
            return;
        }
        
        assert(h >= 0 );
        assert(h <= dt);
        
        collision_point(p0->r, p0->v, h, /**/ rcol);
        
        /* handle collision for particle */
        
        vsolid(vcm, om, rcol, /**/ vs);

        bounce_particle(vs, rcol, p0->v, h, /**/ p1->r, p1->v);
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
    
    void bounce(const Force *ff, const int np, /**/ Particle *pp, Solid *shst)
    {
        Particle p0l, p1, pn, pnl;
        float dF[3], dL[3], vcml[3], oml[3], fl[3];

#ifdef debug_output
        fdebug = fopen("debug.txt", "a");

        if (step % 100 == 0)
        nbounced = nrescued = still_in = failed = 0;
#endif
        
        for (int ip = 0; ip < np; ++ip)
        {
            p1 = pp[ip];
            pn = p1;

            lastbit::Preserver up(pp[ip].v[X]);
                
            r2local(shst->e0, shst->e1, shst->e2, shst->com, pn.r, /**/ pnl.r);
            v2local(shst->e0, shst->e1, shst->e2,            pn.v, /**/ pnl.v);
                
            v2local(shst->e0, shst->e1, shst->e2,  shst->v, /**/ vcml);
            v2local(shst->e0, shst->e1, shst->e2, shst->om, /**/  oml);
                
            v2local(shst->e0, shst->e1, shst->e2, ff[ip].f, /**/ fl);
            
            bounce_part_local(fl, vcml, oml, /*o*/ &pnl, /*w*/ &p0l);
                
            r2global(shst->e0, shst->e1, shst->e2, shst->com, pnl.r, /**/ pn.r);
            v2global(shst->e0, shst->e1, shst->e2,            pnl.v, /**/ pn.v); 
            
            /* transfer momentum */
            
            dF[X] = dF[Y] = dF[Z] = 0;
            dL[X] = dL[Y] = dL[Z] = 0;
                
            lin_mom_solid(p1.v, pn.v, /**/ dF);
                
            ang_mom_solid(p1.r, pn.r, p1.v, pn.v, /**/ dL);
                
            for (int d = 0; d < 3; ++d)
            {
                shst->fo[d] += dF[d];
                shst->to[d] += dL[d];
            }

            pp[ip] = pn;
        }
#ifdef debug_output
        if ((++step) % 100 == 0)
        printf("%d rescued, %d boounced, %d still in, %d failed\n\n", nrescued, nbounced, still_in, failed);

        fclose(fdebug);
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

        float dF[3], dL[3];

        if (pid < np)
        {
            Particle p0l, p1, pn, pnl;
            float vcml[3], oml[3], fl[3];
            
            p1 = pp[pid];
            pn = p1;
            
            lastbit::Preserver up(pp[pid].v[X]);
            
            r2local(sdev->e0, sdev->e1, sdev->e2, sdev->com, pn.r, /**/ pnl.r);
            v2local(sdev->e0, sdev->e1, sdev->e2,            pn.v, /**/ pnl.v);
        
            v2local(sdev->e0, sdev->e1, sdev->e2,  sdev->v, /**/ vcml);
            v2local(sdev->e0, sdev->e1, sdev->e2, sdev->om, /**/  oml);
        
            v2local(sdev->e0, sdev->e1, sdev->e2, ff[pid].f, /**/ fl);
        
            bb_part_local(fl, vcml, oml, /*o*/ &pnl, /*w*/ &p0l);
        
            r2global(sdev->e0, sdev->e1, sdev->e2, sdev->com, pnl.r, /**/ pn.r);
            v2global(sdev->e0, sdev->e1, sdev->e2,            pnl.v, /**/ pn.v); 
        
            /* transfer momentum */
        
            dF[X] = dF[Y] = dF[Z] = 0;
            dL[X] = dL[Y] = dL[Z] = 0;
        
            lin_mom_solid(p1.v, pn.v, /**/ dF);
        
            ang_mom_solid(p1.r, pn.r, p1.v, pn.v, /**/ dL);
            
            pp[pid] = pn;
        }

        /* momentum reduction */
        
        warpReduceSumf3(dF);
        warpReduceSumf3(dL);

        if ((threadIdx.x & (warpSize - 1)) == 0)
        {
            atomicAdd(sdev->fo + X, dF[X]);
            atomicAdd(sdev->fo + Y, dF[Y]);
            atomicAdd(sdev->fo + Z, dF[Z]);

            atomicAdd(sdev->to + X, dL[X]);
            atomicAdd(sdev->to + Y, dL[Y]);
            atomicAdd(sdev->to + Z, dL[Z]);
        }
    }

    void bounce_nohost(const Force *ff, const int np, /**/ Particle *pp, Solid *sdev)
    {
        bounce_kernel <<<k_cnf(np)>>> (ff, np, /**/ pp, sdev);
    }
}
