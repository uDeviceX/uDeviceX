namespace solidbounce {

#define X 0
#define Y 1
#define Z 2
    
    const float eps = 1e-8;
    
    // from forward Euler
    void rvprev(float *r1, float *v1, float *f0, /**/ float *r0, float *v0)
    {
        for (int c = 0; c < 3; ++c)
        {
            r0[c] = r1[c] - v1[c] * dt;
            //v0[c] = v1[c] - f0[c] * dt;
            v0[c] = v1[c];
        }
    }
    
    /*
      return true if a root h lies in [0, dt] (output h), false otherwise
      if 2 roots, smallest root h0 is returned (first collision)
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
        std::swap(h0, h1);

        if (h0 >= 0 && h0 <= dt) {*h = h0; return true;}
        if (h1 >= 0 && h1 <= dt) {*h = h1; return true;}

        //printf("failed : h0 = %.6e, h1 = %.6e (dt = %.6e)\n", h0, h1, dt);
        
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

        bool intersect(float *r0, float *v0, /**/ float *h)
        {
            float r0x = r0[X], r0y = r0[Y], r0z = r0[Z];
            float v0x = v0[X], v0y = v0[Y], v0z = v0[Z];
                        
            const float a = v0x*v0x + v0y*v0y + v0z*v0z;
            
            // if (a < eps)
            // return false;
            
            const float b = 2 * (r0x * v0x + r0y * v0y + r0z * v0z);
            const float c = r0x * r0x + r0y * r0y + r0z * r0z - rsp_bb * rsp_bb;
        
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

        /* output h between 0 and dt */
        bool intersect(float *r0, float *v0, /**/ float *h)
        {
            float r0x = r0[X], r0y = r0[Y];
            float v0x = v0[X], v0y = v0[Y];

            const float a = v0x * v0x + v0y * v0y;
            
            // if (a < eps)
            // return false;
            
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

    void bounce_particle(float *vs, float *rcol, float *v0, float h, /**/ float *rn, float *vn)
    {
        for (int c = 0; c < 3; ++c)
        {
            vn[c] = 2 * vs[c] - v0[c];
            //rn[c] = rcol[c] + (dt - h) * vn[c];
            rn[c] = rcol[c] - (dt - h) * v0[c];
        }
    }

    void rescue_particle(float *cm, float *vcm, float *om, /**/ float *r, float *v)
    {
        shape::rescue(/**/ r);
        //vsolid(cm, vcm, om, r, /**/ v);

#ifndef NDEBUG
        assert(!shape::inside(r));
#endif
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

    int nrescued, nbounced, still_in, failed, step = 0;
    
    void bounce_part(float *fp, float *cm, float *vcm, float *om, /*o*/ Particle *p1, /*w*/ Particle *p0)
    {
        float rcol[3], vs[3];
        float h;

        if (!shape::inside(p1->r))
        return;
        
        lastbit::Preserver up(p1->v[X]);
        
        /* previous position and velocity                        */
        /* this step should be dependant on the time scheme only */
        
        rvprev(p1->r, p1->v, fp, /**/ p0->r, p0->v);

        /* rescue particles which were already in the solid   */
        /* put them back on the surface with surface velocity */

        if (shape::inside(p0->r))
        {
            rescue_particle(cm, vcm, om, /**/ p1->r, p1->v);
            
            ++nrescued;
            return;
        }
        
        /* find collision point */
        
        if (!shape::intersect(p0->r, p0->v, /**/ &h))
        {
            // particle will be rescued at next timestep
            ++failed;
            return;
        }
        
        collision_point(p0->r, p0->v, h, /**/ rcol);
        
        /* handle collision for particle */
        
        vsolid(cm, vcm, om, rcol, /**/ vs);
        
        bounce_particle(vs, rcol, p0->v, h, /**/ p1->r, p1->v);

        if (shape::inside(p1->r))
        ++still_in;
        
        ++nbounced;
    }
    
    void bounce(Force *ff, int np, float *cm, float *vcm, float *om, /**/ Particle *pp, float *r_fo, float *r_to)
    {
        Particle p0, p1, pn;
        float dF[3], dL[3];

        if (step % 100 == 0)
        nbounced = nrescued = still_in = failed = 0;
        
        for (int ip = 0; ip < np; ++ip)
        {
            p1 = pp[ip];
            pn = p1;
            
            bounce_part(ff[ip].f, cm, vcm, om, /*o*/ &pn, /*w*/ &p0);

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

        if ((++step) % 100 == 0)
        printf("%d rescued, %d boounced, %d still in, %d failed\n", nrescued, nbounced, still_in, failed);
    }
}
