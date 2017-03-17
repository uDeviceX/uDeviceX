namespace solidbounce {

    // for now : stationary sphere
    // TODO add this in test arguments
#define rsp_bb rsph
    
    enum {X, Y, Z};

    constexpr float eps = 1e-8;
    
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

    void colpoint(float * r0, float * vp, float h, /**/ float * r1ol)
    {
        for (int c = 0; c < 3; ++c)
        rcol[c] = r0[c] + h * vp[c];
    }

    void bounce_particle(float * vs, float * rcol, float * v0, float h, /**/ float * r1, float * v1)
    {
        for (int c = 0; c < 3; ++c)
        {
            v1[c] = 2 * vs[d] - v0[c];
            r1[c] = rcol[c] + h * v1[c];
        }
    }
    
    void bounce(Particle *pp, Force *ff, int n, float * r_fo, float * r_to)
    {
        Particle p0, p1;
        float rcol[3], vs[3];
        float t;
        
        for (int ip = 0; ip < n; ++ip)
        {
            p1 = pp[i];
            
            float * vc = pp[i].v;
            float * fp = ff[i].f;
            
            /* previous position and velocity */

            vprev(p1.v, fp,   /**/ p0.v);
            rprev(p1.r, p0.v, /**/ p0.r);

            /* find collision point */
            
            if (!intersect(p0.r, p1.r, /**/ &t))
            continue;

            const float h = t * dt;
            
            colpoint(p0.r, p0.v, h, /**/ rcol);
            
            /* handle collision for particle */

            // TODO compute solid velocity
            vs[X] = vs[Y] = vs[Z] = 0;
            
            bounce_particle(vs, rcol, p0.v, h, /**/ p1.r, p1.v);

            /* transfer momentum */

            // TODO
        }
    }
}
