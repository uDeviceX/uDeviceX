#include "sbounce.h"
#include "../../last-bit/last-bit.h"
#include <cassert>

namespace solidbounce {

    enum {X, Y, Z};
        
    // from forward Euler
    void rvprev(float *r1, float *v1, float *f0, /**/ float *r0, float *v0)
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
    bool robust_quadratic_roots(float a, float b, float c, /**/ float *h)
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

        bool inside(float *r)
        {
            return r[X] * r[X] + r[Y] * r[Y] + r[Z] * r[Z] < rsph_bb * rsph_bb;
        }

        bool intersect(float *r0, float *v0, float *om0, /**/ float *h)
        {
            float r0x = r0[X], r0y = r0[Y], r0z = r0[Z];
            float v0x = v0[X], v0y = v0[Y], v0z = v0[Z];
                        
            const float a = v0x*v0x + v0y*v0y + v0z*v0z;
            
            const float b = 2 * (r0x * v0x + r0y * v0y + r0z * v0z);
            const float c = r0x * r0x + r0y * r0y + r0z * r0z - rsph_bb * rsph_bb;
        
            return robust_quadratic_roots(a, b, c, h);
        }

        void rescue(float *r)
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

        bool inside(float *r)
        {
            return r[X] * r[X] + r[Y] * r[Y] < rcyl_bb * rcyl_bb;
        }

        /* output h between 0 and dt */
        bool intersect(float *r0, float *v0, float *om0, /**/ float *h)
        {
            float r0x = r0[X], r0y = r0[Y];
            float v0x = v0[X], v0y = v0[Y];

            const float a = v0x * v0x + v0y * v0y;
            
            const float b = 2 * (r0x * v0x + r0y * v0y);
                        
            const float c = r0x * r0x + r0y * r0y - rcyl_bb * rcyl_bb;

            return robust_quadratic_roots(a, b, c, h);
        }

        void rescue(float *r)
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

        bool inside(float *r)
        {
            const float x = r[X];
            const float y = r[Y];
            
            return x*x / a2_bb + y*y / b2_bb < 1;
        }
        
        /* output h between 0 and dt */
        // for now: assume vcm = 0
        bool intersect(float *r0, float *v0, float *om0, /**/ float *h)
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

        void rescue(float *r) // cheap rescue
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
        bool inside(float *r)
        {
            printf("solidbounce: not implemented\n");
            exit(1);

            return false;
        }

        bool intersect(float *r0, float *v0, float *om0, /**/ float *h)
        {
            printf("solidbounce: not implemented\n");
            exit(1);

            return false;
        }

        void rescue(float *r)
        {
            printf("solidbounce: not implemented\n");
            exit(1);
        }
    }
    
#endif

    bool inside_prev(float *r, float *om0)
    {
        float rl[3] = {r[X] + dt * (om0[Y] * r[Z] - om0[Z] * r[Y]),
                       r[Y] + dt * (om0[Z] * r[X] - om0[X] * r[Z]),
                       r[Z] + dt * (om0[X] * r[Y] - om0[Y] * r[X])};
        
        return shape::inside(rl);
    }

    
    void collision_point(float *r0, float *v0, float h, /**/ float *rcol)
    {
        for (int c = 0; c < 3; ++c)
        rcol[c] = r0[c] + h * v0[c];
    }

    void vsolid(float *vcm, float *om, float *r, /**/ float *vs)
    {
        vs[X] = vcm[X] + om[Y]*r[Z] - om[Z]*r[Y];
        vs[Y] = vcm[Y] + om[Z]*r[X] - om[X]*r[Z];
        vs[Z] = vcm[Z] + om[X]*r[Y] - om[Y]*r[X];
    }

    void bounce_particle(float *vs, float *rcol, float *v0, float h, /**/ float *rn, float *vn)
    {
        assert(h >= 0);
        assert(h <= dt);
        
        for (int c = 0; c < 3; ++c)
        {
            vn[c] = 2 * vs[c] - v0[c];
            rn[c] = rcol[c] + (dt - h) * vn[c];
        }
    }

    void rescue_particle(float *vcm, float *om, /**/ float *r, float *v)
    {
        shape::rescue(/**/ r);
        //vsolid(vcm, om, r, /**/ v);

        assert(!shape::inside(r));
    }

    void lin_mom_solid(float *v1, float *vn, /**/ float *dF)
    {
        for (int c = 0; c < 3; ++c)
        dF[c] = -(vn[c] - v1[c]) / dt;
    }

    void ang_mom_solid(float *r1, float *rn, float *v1, float *vn, /**/ float *dL)
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
    
    void bounce_part_local(float *fp, float *vcm, float *om, /*o*/ Particle *p1, /*w*/ Particle *p0)
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

    void r2local(float *e0, float *e1, float *e2, float *com, float *rg, /**/ float *rl)
    {
        float x = rg[X] - com[X];
        float y = rg[Y] - com[Y];
        float z = rg[Z] - com[Z];
        
        rl[X] = x*e0[X] + y*e0[Y] + z*e0[Z];
        rl[Y] = x*e1[X] + y*e1[Y] + z*e1[Z];
        rl[Z] = x*e2[X] + y*e2[Y] + z*e2[Z];
    }

    void r2global(float *e0, float *e1, float *e2, float *com, float *rl, /**/ float *rg)
    {
        rg[X] = com[X] + rl[X] * e0[X] + rl[Y] * e1[X] + rl[Z] * e2[X];
        rg[Y] = com[Y] + rl[X] * e0[Y] + rl[Y] * e1[Y] + rl[Z] * e2[Y];
        rg[Z] = com[Z] + rl[X] * e0[Z] + rl[Y] * e1[Z] + rl[Z] * e2[Z];
    }

    void v2local(float *e0, float *e1, float *e2, float *vg, /**/ float *vl)
    {
        vl[X] = vg[X]*e0[X] + vg[Y]*e0[Y] + vg[Z]*e0[Z];
        vl[Y] = vg[X]*e1[X] + vg[Y]*e1[Y] + vg[Z]*e1[Z];
        vl[Z] = vg[X]*e2[X] + vg[Y]*e2[Y] + vg[Z]*e2[Z];
    }

    void v2global(float *e0, float *e1, float *e2, float *vl, /**/ float *vg)
    {
        vg[X] = vl[X] * e0[X] + vl[Y] * e1[X] + vl[Z] * e2[X];
        vg[Y] = vl[X] * e0[Y] + vl[Y] * e1[Y] + vl[Z] * e2[Y];
        vg[Z] = vl[X] * e0[Z] + vl[Y] * e1[Z] + vl[Z] * e2[Z];
    }
    
    void bounce(float *e0, float *e1, float *e2, Force *ff, int np, float *cm, float *vcm, float *om, /**/ Particle *pp, float *r_fo, float *r_to)
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
                
            r2local(e0, e1, e2, cm, pn.r, /**/ pnl.r);
            v2local(e0, e1, e2,     pn.v, /**/ pnl.v);
                
            v2local(e0, e1, e2, vcm, /**/ vcml);
            v2local(e0, e1, e2,  om, /**/  oml);
                
            v2local(e0, e1, e2, ff[ip].f, /**/ fl);
            
            bounce_part_local(fl, vcml, oml, /*o*/ &pnl, /*w*/ &p0l);
                
            r2global(e0, e1, e2, cm, pnl.r, /**/ pn.r);
            v2global(e0, e1, e2,     pnl.v, /**/ pn.v); 
                
            /* transfer momentum */
                
            dF[X] = dF[Y] = dF[Z] = 0;
            dL[X] = dL[Y] = dL[Z] = 0;
                
            lin_mom_solid(p1.v, pn.v, /**/ dF);
                
            ang_mom_solid(p1.r, pn.r, p1.v, pn.v, /**/ dL);
                
            for (int d = 0; d < 3; ++d)
            {
                r_fo[d] += dF[d];
                r_to[d] += dL[d];
            }

            pp[ip] = pn;
        }
#ifdef debug_output
        if ((++step) % 100 == 0)
        printf("%d rescued, %d boounced, %d still in, %d failed\n\n", nrescued, nbounced, still_in, failed);

        fclose(fdebug);
#endif
    }
}
