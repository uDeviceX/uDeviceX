namespace solidbounce {

#define X 0
#define Y 1
#define Z 2

    void rprev(float * rc, float * vp, /**/ float * rp)
    {
        for (int c = 0; c < 3; ++c)
        rp[c] = rc[c] - dt * vp[c];
    }
    
    void vprev(float * vc, float * fp, /**/ float * vp)
    {
        for (int c = 0; c < 3; ++c)
        vp[c] = vc[c] - dt * fp[c];
    }

    bool has_crossed(float * rp, float * rc)
    {
        return !solid::inside(rp[X], rp[Y], rp[Z]) &&
            solid::inside(rc[X], rc[Y], rc[Z]);
    }
    
    void bounce(Particle *pp, Force *ff, int n)
    {
        float rp[3], vp[3];
        
        for (int ip = 0; ip < n; ++ip)
        {
            float * rc = pp[i].r;
            
            float * vc = pp[i].v;
            float * fp = ff[i].f;
            
            /* previous position and velocity */

            vprev(vc, fp, /**/ vp);
            rprev(rc, vp, /**/ rp);

            if (!has_crossed(rp, rc))
            continue;

            /* find collision point */

            // TODO
            
            /* handle collision */

            // TODO
        }
    }
}
