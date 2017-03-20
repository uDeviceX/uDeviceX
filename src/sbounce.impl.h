namespace solidbounce {

#define X 0
#define Y 1
#define Z 2

    const float eps = 1e-8;
    
    void rprev(float * r1, float * vp, /**/ float * r0)
    {
        for (int c = 0; c < 3; ++c)
        r0[c] = r1[c] - dt * vp[c];
    }
    
    void vprev(float * vc, float * fp, /**/ float * vp)
    {
        for (int c = 0; c < 3; ++c)
        vp[c] = vc[c] - dt * fp[c];
    }

    namespace sphere
    {
        // for now : stationary sphere
        // TODO add this in test arguments

        #define rsph_bb rsph

        bool intersect(float * r0, float * r1, /**/ float * t)
        {
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
            
            const float D = b*b - 4*a*c;
            
            if (D < 0) return false;
            
            const float sqrtD = sqrt(D);
            
            const float t0 = (-b - sqrtD) / (2 * a);
            if (t0 > 0 && t0 < 1) {*t = t0; return true;}
            
            const float t1 = (-b + sqrtD) / (2 * a);
            if (t1 > 0 && t1 < 1) {*t = t1; return true;}
            
            return false;
        }
    }

    namespace cylinder
    {
        #define rcyl_bb rcyl

        bool intersect(float * r0, float * r1, /**/ float * t)
        {
            float r0x = r0[X], r0z = r0[Z];
            float r1x = r1[X], r1z = r1[Z];
            
            const float inv_r = 1.f / rcyl_bb;
            
            r0x *= inv_r; r0z *= inv_r;
            r1x *= inv_r; r1z *= inv_r;
            
            const float a = pow(r1x - r0x, 2) + pow(r1z - r0z, 2);
            
            if (a < eps)
            return false;
            
            const float b =
                2 * r0z * (r1z - r0z) +
                2 * r0x * (r1x - r0x);
            
            const float c = r0x * r0x + r0z * r0z - 1.f;
            
            const float D = b*b - 4*a*c;
            
            if (D < 0) return false;
            
            const float sqrtD = sqrt(D);
            
            const float t0 = (-b - sqrtD) / (2 * a);
            if (t0 > 0 && t0 < 1) {*t = t0; return true;}
            
            const float t1 = (-b + sqrtD) / (2 * a);
            if (t1 > 0 && t1 < 1) {*t = t1; return true;}
            
            return false;
        }
    }

    void colpoint(float * r0, float * vp, float h, /**/ float * rcol)
    {
        for (int c = 0; c < 3; ++c)
        rcol[c] = r0[c] + h * vp[c];
    }

    void vsolid(float * cm, float * vcm, float * omega, float * r, /**/ float * vs)
    {
        float dr[3] = {r[X] - cm[X],
                       r[Y] - cm[Y],
                       r[Z] - cm[Z]};

        vs[X] = vcm[X] + omega[Y]*dr[Z] - omega[Z]*dr[Y];
        vs[Y] = vcm[Y] + omega[Z]*dr[X] - omega[X]*dr[Z];
        vs[Z] = vcm[Z] + omega[X]*dr[Y] - omega[Y]*dr[X];
    }

    void bounce_particle(float * vs, float * rcol, float * v0, float h, /**/ float * r1, float * v1)
    {
        for (int c = 0; c < 3; ++c)
        {
            v1[c] = 2 * vs[c] - v0[c];
            r1[c] = rcol[c] + h * v1[c];
        }
    }

    void lin_mom_solid(float * v0, float * v1, /**/ float * fo)
    {
        for (int c = 0; c < 3; ++c)
        fo[c] -= (v1[c] - v0[c]) / dt;
    }

    void ang_mom_solid(float * r0, float * r1, float * v0, float * v1, /**/ float * to)
    {
        to[X] -= (r1[Y] * v1[Z] - r1[Z] * v1[Y] - r0[Y] * v0[Z] + r0[Z] * v0[Y]) / dt;
        to[Y] -= (r1[Z] * v1[X] - r1[X] * v1[Z] - r0[Z] * v0[X] + r0[X] * v0[Z]) / dt;
        to[Z] -= (r1[X] * v1[Y] - r1[Y] * v1[X] - r0[X] * v0[Y] + r0[Y] * v0[X]) / dt;
    }
    
    void bounce(Force *ff, int np, float * cm, float * vcm, float * om, /**/ Particle *pp, float * r_fo, float * r_to)
    {
        Particle p0, p1;
        float rcol[3], vs[3];
        float t;
        
        for (int ip = 0; ip < np; ++ip)
        {
            p1 = pp[ip];
            
            float * fp = ff[ip].f;
            
            /* previous position and velocity */

            vprev(p1.v, fp,   /**/ p0.v);
            rprev(p1.r, p0.v, /**/ p0.r);

            /* find collision point */
            
            if (!sphere::intersect(p0.r, p1.r, /**/ &t))
            continue;

            const float h = t * dt;
            
            colpoint(p0.r, p0.v, h, /**/ rcol);
            
            /* handle collision for particle */

            vsolid(cm, vcm, om, rcol, /**/ vs);
            
            bounce_particle(vs, rcol, p0.v, h, /**/ p1.r, p1.v);

            /* transfer momentum */

            lin_mom_solid(p0.v, p1.v, /**/ r_fo);

            ang_mom_solid(p0.r, p1.r, p0.v, p1.v, /**/ r_fo);

            pp[ip] = p1;
        }
    }
}
