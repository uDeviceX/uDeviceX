namespace solidbounce {

#define X 0
#define Y 1
#define Z 2
    
    const float eps = 1e-8;
    
    // from forward Euler
    void rprev(float *r1, float *v0, /**/ float *r0)
    {
        for (int c = 0; c < 3; ++c)
        r0[c] = r1[c] - dt * v0[c];
    }
    
    void vprev(float *v1, float *f0, /**/ float *v0)
    {
        for (int c = 0; c < 3; ++c)
        v0[c] = v1[c] - dt * f0[c];
    }

    /*
      return true if a root t lies in [0, 1] (output t), false otherwise
      smallest root t is returned
    */
    bool robust_quadratic_roots(float a, float b, float c, /**/ float *t)
    {
        float D, t0, t1;
        int sgnb;

        sgnb = b > 0 ? 1 : -1;
        D = b*b - 4*a*c;

        if (D < 0) return false;
        
        t0 = (-b - sgnb * sqrt(D)) / (2 * a);
        t1 = c / (a * t0);
        
        if (t0 > t1)
        std::swap(t0, t1);

        if (t0 >= 0 && t0 <= 1) {*t = t0; return true;}
        if (t1 >= 0 && t1 <= 1) {*t = t1; return true;}

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

        bool intersect(float *r0, float *r1, /**/ float *t)
        {
#ifndef NDEBUG
            assert(inside(r1));
            assert(!inside(r0));
#endif

            float r0x = r0[X], r0y = r0[Y], r0z = r0[Z];
            float r1x = r1[X], r1y = r1[Y], r1z = r1[Z];
            
            const float inv_r = 1.f / rsph_bb;
            
            r0x *= inv_r; r0y *= inv_r; r0z *= inv_r;
            r1x *= inv_r; r1y *= inv_r; r1z *= inv_r;
            
            const float a = pow(r1x - r0x, 2) + pow(r1y - r0y, 2) + pow(r1z - r0z, 2);
            
            if (a < eps)
            return false;
            
            const float b =
                2 * r0z * (r1z - r0z) +
                2 * r0y * (r1y - r0y) +
                2 * r0x * (r1x - r0x);
            
            const float c = r0x * r0x + r0y * r0y + r0z * r0z - 1.f;
        
            return robust_quadratic_roots(a, b, c, t);
        }

        void rescue(float *r)
        {
            float scale = (rsph_bb + 1e-6) / sqrt(r[X] * r[X] + r[Y] * r[Y] + r[Z] * r[Z]);

            r[X] *= scale;
            r[Y] *= scale;
            r[Z] *= scale;
        }
    }
#endif

#if defined(rcyl)

#define shape cylinder
    
    namespace cylinder
    {
        #define rcyl_bb rcyl

        bool inside(float *r)
        {
            return r[X] * r[X] + r[Y] * r[Y] < rcyl_bb * rcyl_bb;
        }

        /* output t between 0 and 1 */
        bool intersect(float *r0, float *r1, /**/ float *t)
        {
#ifndef NDEBUG
            assert(inside(r1));
            assert(!inside(r0));
#endif
            
            float r0x = r0[X], r0y = r0[Y];
            float r1x = r1[X], r1y = r1[Y];
            
            const float inv_r = 1.f / rcyl_bb;
            
            r0x *= inv_r; r0y *= inv_r;
            r1x *= inv_r; r1y *= inv_r;
            
            const float a = pow(r1x - r0x, 2) + pow(r1y - r0y, 2);
            
            if (a < eps)
            return false;
            
            const float b =
                2 * r0x * (r1x - r0x) +
                2 * r0y * (r1y - r0y);
                
            
            const float c = r0x * r0x + r0y * r0y - 1.f;

            return robust_quadratic_roots(a, b, c, t);
        }

        void rescue(float *r)
        {
            float scale = (rcyl_bb + 1e-6) / sqrt(r[X] * r[X] + r[Y] * r[Y]);

            r[X] *= scale;
            r[Y] *= scale;
        }
    }
#endif
    
    void collision_point(float *r0, float *v0, float h, /**/ float *rcol)
    {
        for (int c = 0; c < 3; ++c)
        rcol[c] = r0[c] + h * v0[c];
    }

    void vsolid(float *cm, float *vcm, float *omega, float *r, /**/ float *vs)
    {
        float dr[3] = {r[X] - cm[X],
                       r[Y] - cm[Y],
                       r[Z] - cm[Z]};

        vs[X] = vcm[X] + omega[Y]*dr[Z] - omega[Z]*dr[Y];
        vs[Y] = vcm[Y] + omega[Z]*dr[X] - omega[X]*dr[Z];
        vs[Z] = vcm[Z] + omega[X]*dr[Y] - omega[Y]*dr[X];
    }

    void bounce_particle(float *vs, float *rcol, float *v0, float h, /**/ float *r1, float *v1)
    {
        for (int c = 0; c < 3; ++c)
        {
            v1[c] = 2 * vs[c] - v0[c];
            r1[c] = rcol[c] + (dt - h) * v1[c];
        }
    }

    void lin_mom_solid(float *v0, float *v1, /**/ float *fo)
    {
        for (int c = 0; c < 3; ++c)
        fo[c] -= (v1[c] - v0[c]) / dt;
    }

    void ang_mom_solid(float *r0, float *r1, float *v0, float *v1, /**/ float *to)
    {
        to[X] -= (r1[Y] * v1[Z] - r1[Z] * v1[Y] - r0[Y] * v0[Z] + r0[Z] * v0[Y]) / dt;
        to[Y] -= (r1[Z] * v1[X] - r1[X] * v1[Z] - r0[Z] * v0[X] + r0[X] * v0[Z]) / dt;
        to[Z] -= (r1[X] * v1[Y] - r1[Y] * v1[X] - r0[X] * v0[Y] + r0[Y] * v0[X]) / dt;
    }

    void bounce_part(float *fp, float *cm, float *vcm, float *om, /*o*/ Particle *p1, float *r_fo, float *r_to, /*w*/ Particle *p0)
    {
        float rcol[3], vs[3];
        float t;

        if (!shape::inside(p1->r))
        return;
        
        lastbit::Preserver up(p1->v[X]);
        
        /* previous position and velocity */

        vprev(p1->v, fp,    /**/ p0->v);
        rprev(p1->r, p0->v, /**/ p0->r);

        if (shape::inside(p0->r))
        {
            shape::rescue(p1->r);
#ifndef NDEBUG
            assert(!shape::inside(p1->r));
#endif
            //printf("solidbounce: rescued one particle\n");
            return;
        }
        
        /* find collision point */
        
        if (!shape::intersect(p0->r, p1->r, /**/ &t))
        {
            // particle will be rescued at next timestep
            return;
        }
        
        t = t * dt;
        
        collision_point(p0->r, p0->v, t, /**/ rcol);
        
        /* handle collision for particle */
        
        vsolid(cm, vcm, om, rcol, /**/ vs);
        
        bounce_particle(vs, rcol, p0->v, t, /**/ p1->r, p1->v);
        
        /* transfer momentum */
        
        lin_mom_solid(p0->v, p1->v, /**/ r_fo);
        
        ang_mom_solid(p0->r, p1->r, p0->v, p1->v, /**/ r_fo);
    }
    
    void bounce(Force *ff, int np, float *cm, float *vcm, float *om, /**/ Particle *pp, float *r_fo, float *r_to)
    {
        Particle p0, p1;
        
        for (int ip = 0; ip < np; ++ip)
        {
            p1 = pp[ip];

            bounce_part(ff[ip].f, cm, vcm, om, /*o*/ &p1, r_fo, r_to, /*w*/ &p0);
            
            pp[ip] = p1;
        }
    }
}
