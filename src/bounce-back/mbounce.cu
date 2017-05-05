#ifndef dt
#include "../.conf.h"
#endif
#include "../common.h"

#include <cassert>

#include "sbounce.h"
#include "bbshapes.impl.h"

namespace mbounce
{
    enum {X, Y, Z};
#define BBOX_MARGIN 0.1f
    
    static void rvprev(const float *r1, const float *v1, const float *f0, /**/ float *r0, float *v0)
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

    bool cubic_root(const float a, const float b, const float c, const float d, float *h)
    {
        // TODO
        return false;
    }
    
    /* see Fedosov PhD Thesis */
    static bool intersect_triangle(const float *s10, const float *s20, const float *s30,
                                   const float *vs1, const float *vs2, const float *vs3,
                                   const Particle *p0,  /**/ float *h, float *rw)
    {
#define diff(a, b) {a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z]}
#define cross(a, b) {a[Y] * b[Z] - a[Z] * b[Y], a[Z] * b[X] - a[X] * b[Z], a[X] * b[Y] - a[Y] * b[X]}
#define dot(a, b) a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z];
#define apxb(a, x, b) {a[X] + x * b[X], a[Y] + x * b[Y], a[Z] + x * b[Z]} 
        
        const float *r0 = p0.r;
        const float *v0 = p0.v;
    
        const float a1[3] = diff(s20, s10);
        const float a2[3] = diff(s30, s10);
    
        const float a1t[3] = diff(vs2, vs1);
        const float a2t[3] = diff(vs3, vs1);

        // n(t) = n + t*nt + t^2 * ntt
        const float n[3] = cross(a1, a2);
        const float ntt[3] = cross(at1, at2);
        const float nt[3] = {a1[Y] * at2[Z] - a1[Z] * at2[Y]  +  a2[Y] * at1[Z] - a2[Z] * at1[Y],
                             a1[Z] * at2[X] - a1[X] * at2[Z]  +  a2[Z] * at1[X] - a2[X] * at1[Z],
                             a1[X] * at2[Y] - a1[Y] * at2[X]  +  a2[X] * at1[Y] - a2[Y] * at1[X]};
    
        const float dr0[3] = diff(r0, s10);
        
        // check intersection with plane
        {
            const float r1[3] = apxb(r0, dt, v0);
            const float s11[3] = apxb(s10, dt, vs1);

            const float n1[3] = {n[X] + dt * (nt[X] + dt * ntt[X]),
                                 n[Y] + dt * (nt[Y] + dt * ntt[Y]),
                                 n[Z] + dt * (nt[Z] + dt * ntt[Z])};
            
            const float dr1[3] = diff(r1, s11);

            const float b0 = dot(dr0, n0);
            const float b1 = dot(dr1, n1);

            if (b0 * b1 > 0)
            return;
        }

        // find intersection with plane

        const float dv[3] = diff(v0, a1t);
        
        const float a = dot(ntt, dv);
        const float b = dot(ntt, dr0) + dot(nt, dv);
        const float c = dot(nt, dr0) + dot(n, dv);
        const float d = dot(n, dr0);
        
        if (!cubic_root(a, b, c, d, &h))
        return false;

        rw[X] = r0[X] + h * v0[X];
        rw[Y] = r0[Y] + h * v0[Y];
        rw[Z] = r0[Z] + h * v0[Z];

        // check if inside triangle

        {
            const float g[3] = {rw[X] - s10[X] - h * vs1[X],
                                rw[Y] - s10[Y] - h * vs1[Y],
                                rw[Z] - s10[Z] - h * vs1[Z]};

            const a1_[3] = apxb(a1, h, a1t);
            const a2_[3] = apxb(a2, h, a2t);
            
            const float ga1 = dot(g, a1_);
            const float ga2 = dot(g, a2_);
            const float a11 = dot(a1_, a1_);
            const float a12 = dot(a1_, a2_);
            const float a22 = dot(a2_, a2_);

            const fac = 1.f / (a11*a22 - a12*a12);
            
            const float u = (ga1 * a22 - ga2 * a12) * fac;
            const float v  = (ga2 * a11 - ga1 * a12) * fac;

            if (!((u >= 0) && (v >= 0) && (u+v <= 1)))
            return false;
        }

        return true;
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

    static void get_vl_solid(const float *rl, const float *om, /**/ float *vw)
    {
        vw[X] = om[Y] * rl[Z] + om[Z] * rl[Y];
        vw[Y] = om[Z] * rl[X] + om[X] * rl[Z];
        vw[Z] = om[X] * rl[Y] + om[Y] * rl[X];
    }
    
    static void bounce_1s1p(const float *f, const Mesh m, Particle *p, Solid *s)
    {
        float fl[3], v1[3], v2[3], v3[3], h, rw[3], vw[3];
        Particle p0l;
        
        for (int it = 0; it < m.nt; ++it)
        {            
            r2local(s->e0, s->e1, s->e2, s->com, p->r, /**/ pl.r);
            v2local(s->e0, s->e1, s->e2, p->v, /**/ pl.v);
            v2local(s->e0, s->e1, s->e2,    f, /**/ fl);
            
            const int t1 = tt[3*it + 0];
            const int t2 = tt[3*it + 1];
            const int t3 = tt[3*it + 2];

            const float a1[3] = {vv[3*t1+0], vv[3*t1+1], vv[3*t1+2]};
            const float a2[3] = {vv[3*t2+0], vv[3*t2+1], vv[3*t2+2]};
            const float a3[3] = {vv[3*t3+0], vv[3*t3+1], vv[3*t3+2]};

            get_vl_solid(a1, s->om, /**/ v1);
            get_vl_solid(a2, s->om, /**/ v2);
            get_vl_solid(a3, s->om, /**/ v3);

            if (intersect_triangle(a1, a2, a3, v1, v2, v3, p0l, /**/ &h, rw))
            {
                get_vl_solid(rw, s->om, /**/ vw);

                // TODO
                
                break;
            }       
        }
    }
    
    static bool in_bbox(const float *r, const float *bbox, const float tol)
    {
        return (r[X] >= bbox[2*X + 0] - tol) && (r[X] < bbox[2*X + 1] + tol) &&
            (r[Y] >= bbox[2*Y + 0] - tol) && (r[Y] < bbox[2*Y + 1] + tol) &&
            (r[Z] >= bbox[2*Z + 0] - tol) && (r[Z] < bbox[2*Z + 1] + tol);
    }
    
    static void bounce_1s(const Force *ff, const int np, const Mesh m, const float *bbox, /**/ Particle *pp, Solid *shst)
    {
        for (int i = 0; i < np; ++i)
        {
            Particle p = pp[i];
            if (in_bbox(p.r, bbox, BBOX_MARGIN))
            {
                const Force f = ff[i];
                bounce_1s1p(f.f, m, /**/ &p, shst);
                pp[i] = pp;
            }
        }
    }

    void bounce_hst(const Force *ff, const int np, const int ns, const Mesh m, /**/ Particle *pp, Solid *shst)
    {
        float bbox[6];
        
        for (int j = 0; j < ns; ++j)
        {
            Solid *s = shst + j;
            
            mesh::bbox(m.vv, m.nt, s->e1, s->e2, s->e3, /**/ bbox);

            bounce_1s(ff, np, m, bbox, /**/ pp, s);
        }
    }
}
