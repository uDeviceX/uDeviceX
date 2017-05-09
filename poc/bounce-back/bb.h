
enum BBState
{
    BB_SUCCESS,   /* succesfully bounced            */
    BB_NOCROSS,   /* did not cross the plane        */
    BB_WTRIANGLE, /* [w]rong triangle               */
    BB_HFAIL      /* no time solution h             */
};

static bool cubic_root(float a, float b, float c, float d, const float dt, /**/ float *h)
{
#define valid(t) ((t) >= 0 && (t) <= dt)
        
    if (fabs(a) > 1e-8) // cubic
    {
        const float sc = 1.f / a;
        b *= sc; c *= sc; d *= sc;
            
        float h1, h2, h3;
        int nsol = roots::cubic(b, c, d, &h1, &h2, &h3);

        if (valid(h1))             {*h = h1; return true;}
        if (nsol > 1 && valid(h2)) {*h = h2; return true;}
        if (nsol > 2 && valid(h3)) {*h = h3; return true;}
        return false;
    }
    else // quadratic
    {
        float h1, h2;
        if (!roots::quadratic(b, c, d, &h1, &h2)) return false;
        if (valid(h1)) {*h = h1; return true;}
        if (valid(h2)) {*h = h2; return true;}
        return false;
    }
}

BBState intersect_triangle(const float *s10, const float *s20, const float *s30,
                           const float *vs1, const float *vs2, const float *vs3,
                           const Particle *p0, const float dt, /**/ float *h, float *rw)
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
        
    if (!cubic_root(a, b, c, d, dt, h))
    return BB_HFAIL;

    rw[X] = r0[X] + *h * v0[X];
    rw[Y] = r0[Y] + *h * v0[Y];
    rw[Z] = r0[Z] + *h * v0[Z];

    // check if inside triangle

    {
        const float g[3] = {rw[X] - s10[X] - *h * vs1[X],
                            rw[Y] - s10[Y] - *h * vs1[Y],
                            rw[Z] - s10[Z] - *h * vs1[Z]};

        const float a1_[3] = apxb(a1, *h, at1);
        const float a2_[3] = apxb(a2, *h, at2);
            
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
    }

    return BB_SUCCESS;
}
