namespace mbounce {
namespace sub {

#define _DH_ __device__ __host__
enum {X, Y, Z};

_DH_ void rvprev(const float *r1, const float *v1, const float *f0, /**/ float *r0, float *v0) {
#ifdef FORWARD_EULER
    for (int c = 0; c < 3; ++c) {
        v0[c] = v1[c] - f0[c] * dt;
        r0[c] = r1[c] - v0[c] * dt;
    }
#else // velocity-verlet
    for (int c = 0; c < 3; ++c) {
        r0[c] = r1[c] - v1[c] * dt;
        //v0[c] = v1[c] - f0[c] * dt;

        // BB assumes r0 + v0 dt = r1 for now
        v0[c] = v1[c];
    }
#endif
}

template <typename real>
static _DH_ bool valid(real t) {return (t >= 0 && t <= dt);}

template <typename real>
static _DH_ bool cubic_root(real a, real b, real c, real d, /**/ real *h) {
    const real eps = 1e-6;
        
    if (fabs(a) > eps) { // cubic
        const real b_ = b /= a;
        const real c_ = c /= a;
        const real d_ = d /= a;
            
        real h1, h2, h3;
        int nsol = roots::cubic(b_, c_, d_, &h1, &h2, &h3);

        if (valid(h1))             {*h = h1; return true;}
        if (nsol > 1 && valid(h2)) {*h = h2; return true;}
        if (nsol > 2 && valid(h3)) {*h = h3; return true;}
    }
    else if(fabs(b) > eps) { // quadratic
        real h1, h2;
        if (!roots::quadratic(b, c, d, &h1, &h2)) return false;
        if (valid(h1)) {*h = h1; return true;}
        if (valid(h2)) {*h = h2; return true;}
    }
    else if (fabs(c) > eps) { // linear
        const real h1 = -d/c;
        if (valid(h1)) {*h = h1; return true;}
    }
    
    return false;
}
    
/* see Fedosov PhD Thesis */
static _DH_ BBState intersect_triangle(const float *s10, const float *s20, const float *s30,
                                       const float *vs1, const float *vs2, const float *vs3,
                                       const Particle *p0,  /*io*/ float *h, /**/ float *rw, float *vw) {
    typedef double real;
        
#define diff(a, b) {a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z]}
#define cross(a, b) {a[Y] * b[Z] - a[Z] * b[Y], a[Z] * b[X] - a[X] * b[Z], a[X] * b[Y] - a[Y] * b[X]}
#define dot(a, b) (a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z])
#define apxb(a, x, b) {a[X] + (real) x * b[X], a[Y] + (real) x * b[Y], a[Z] + (real) x * b[Z]} 
        
    const float *r0 = p0->r;
    const float *v0 = p0->v;
    
    const real a1[3] = diff(s20, s10);
    const real a2[3] = diff(s30, s10);
    
    const real at1[3] = diff(vs2, vs1);
    const real at2[3] = diff(vs3, vs1);

    // n(t) = n + t*nt + t^2 * ntt
    const real n0[3] = cross(a1, a2);
    const real ntt[3] = cross(at1, at2);
    const real nt[3] = {a1[Y] * at2[Z] - a1[Z] * at2[Y]  +  at1[Y] * a2[Z] - at1[Z] * a2[Y],
                        a1[Z] * at2[X] - a1[X] * at2[Z]  +  at1[Z] * a2[X] - at1[X] * a2[Z],
                        a1[X] * at2[Y] - a1[Y] * at2[X]  +  at1[X] * a2[Y] - at1[Y] * a2[X]};
    
    const real dr0[3] = diff(r0, s10);
        
    // check intersection with plane
    {
        const real r1[3] = apxb(r0, dt, v0);
        const real s11[3] = apxb(s10, dt, vs1);

        const real n1[3] = {n0[X] + (real) dt * (nt[X] + (real) dt * ntt[X]),
                            n0[Y] + (real) dt * (nt[Y] + (real) dt * ntt[Y]),
                            n0[Z] + (real) dt * (nt[Z] + (real) dt * ntt[Z])};
            
        const real dr1[3] = diff(r1, s11);

        const real b0 = dot(dr0, n0);
        const real b1 = dot(dr1, n1);

        if (b0 * b1 > 0)
            return BB_NOCROSS;
    }

    // find intersection time with plane
    real hl;

    {
        const real dv[3] = diff(v0, vs1);
        
        const real a = dot(ntt, dv);
        const real b = dot(ntt, dr0) + dot(nt, dv);
        const real c = dot(nt, dr0) + dot(n0, dv);
        const real d = dot(n0, dr0);        
        
        if (!cubic_root(a, b, c, d, &hl)) {
            // printf("failed : %g %g %g %g\n", a, b, c, d);
            return BB_HFAIL;
        }
    }

    if (hl > *h)
        return BB_HNEXT;

    const real rwl[3] = {r0[X] + hl * v0[X],
                         r0[Y] + hl * v0[Y],
                         r0[Z] + hl * v0[Z]};

    // check if inside triangle
    const real g[3] = {rwl[X] - s10[X] - hl * vs1[X],
                       rwl[Y] - s10[Y] - hl * vs1[Y],
                       rwl[Z] - s10[Z] - hl * vs1[Z]};

    const real a1_[3] = apxb(a1, hl, at1);
    const real a2_[3] = apxb(a2, hl, at2);
            
    const real ga1 = dot(g, a1_);
    const real ga2 = dot(g, a2_);
    const real a11 = dot(a1_, a1_);
    const real a12 = dot(a1_, a2_);
    const real a22 = dot(a2_, a2_);

    const real fac = 1.f / (a11*a22 - a12*a12);
            
    const real u = (ga1 * a22 - ga2 * a12) * fac;
    const real v = (ga2 * a11 - ga1 * a12) * fac;

    if ((u < 0) || (v < 0) || (u+v > 1))
        return BB_WTRIANGLE;

    *h = hl;
        
    rw[X] = rwl[X];
    rw[Y] = rwl[Y];
    rw[Z] = rwl[Z];

    // linear interpolation of velocity vw
    const real w = 1 - u - v;
    vw[X] = w * vs1[X] + u * vs2[X] + v * vs3[X];
    vw[Y] = w * vs1[Y] + u * vs2[Y] + v * vs3[Y];
    vw[Z] = w * vs1[Z] + u * vs2[Z] + v * vs3[Z];
    
    return BB_SUCCESS;

#undef diff
#undef cross
#undef dot
#undef apxb
}

_DH_ void lin_mom_solid(const float *v1, const float *vn, /**/ float *dP) {
    for (int c = 0; c < 3; ++c)
        dP[c] = -(vn[c] - v1[c]) / dt;
}

_DH_ void ang_mom_solid(const float *com, const float *rw, const float *v0, const float *vn, /**/ float *dL) {
    const float dr[3] = {rw[X] - com[X], rw[Y] - com[Y], rw[Z] - com[Z]};
        
    dL[X] = -(dr[Y] * vn[Z] - dr[Z] * vn[Y] - dr[Y] * v0[Z] + dr[Z] * v0[Y]) / dt;
    dL[Y] = -(dr[Z] * vn[X] - dr[X] * vn[Z] - dr[Z] * v0[X] + dr[X] * v0[Z]) / dt;
    dL[Z] = -(dr[X] * vn[Y] - dr[Y] * vn[X] - dr[X] * v0[Y] + dr[Y] * v0[X]) / dt;
}

static _DH_ void revert_r(Particle *p) {
    p->r[X] -= dt * p->v[X];
    p->r[Y] -= dt * p->v[Y];
    p->r[Z] -= dt * p->v[Z];
}

_DH_ bool find_better_intersection(const int *tt, const int it, const Particle *i_pp, const Particle *p0, /* io */ float *h, /**/ float *rw, float *vw) {
    // load data
    const int t1 = tt[3*it + 0];
    const int t2 = tt[3*it + 1];
    const int t3 = tt[3*it + 2];

    Particle pA = i_pp[t1];
    Particle pB = i_pp[t2];
    Particle pC = i_pp[t3];

    revert_r(&pA);
    revert_r(&pB);
    revert_r(&pC);
    
    const BBState bbstate = intersect_triangle(pA.r, pB.r, pC.r, pA.v, pB.v, pC.v, p0, /* io */ h, /**/ rw, vw);

    dbg::log_states(bbstate);
    
    return bbstate == BB_SUCCESS;
}

_DH_ void bounce_back(const Particle *p0, const float *rw, const float *vw, const float h, /**/ Particle *pn) {
    pn->v[X] = 2 * vw[X] - p0->v[X];
    pn->v[Y] = 2 * vw[Y] - p0->v[Y];
    pn->v[Z] = 2 * vw[Z] - p0->v[Z];

    pn->r[X] = rw[X] + (dt-h) * pn->v[X];
    pn->r[Y] = rw[Y] + (dt-h) * pn->v[Y];
    pn->r[Z] = rw[Z] + (dt-h) * pn->v[Z];
}

_DH_ void lin_mom_change(const float v0[3], const float v1[3], /**/ float dP[3]) {
    dP[X] = -(v1[X] - v0[X]);
    dP[Y] = -(v1[Y] - v0[Y]);
    dP[Z] = -(v1[Z] - v0[Z]);
}

_DH_ void ang_mom_change(const float r[3], const float v0[3], const float v1[3], /**/ float dL[3]) {
    dL[X] = -(r[Y] * v1[Z] - r[Z] * v1[Y]  -  r[Y] * v0[Z] + r[Z] - v0[Y]);
    dL[X] = -(r[Z] * v1[X] - r[X] * v1[Z]  -  r[Z] * v0[X] + r[X] - v0[Z]);
    dL[X] = -(r[X] * v1[Y] - r[Y] * v1[X]  -  r[X] * v0[Y] + r[Y] - v0[X]);
}



#undef _DH_

} // sub
} // mbounce
