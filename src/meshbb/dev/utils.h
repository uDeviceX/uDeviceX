namespace dev {

enum {X, Y, Z};

// TODO belongs to scheme/ ?
#ifdef FORWARD_EULER
__device__ void rvprev(const float *r1, const float *v1, const float *f0, /**/ float *r0, float *v0) {
    for (int c = 0; c < 3; ++c) {
        v0[c] = v1[c] - f0[c] * dt;
        r0[c] = r1[c] - v0[c] * dt;
    }
}
#else // velocity-verlet
__device__ void rvprev(const float *r1, const float *v1, const float *, /**/ float *r0, float *v0) {
    for (int c = 0; c < 3; ++c) {
        r0[c] = r1[c] - v1[c] * dt;
        //v0[c] = v1[c] - f0[c] * dt;

        // BB assumes r0 + v0 dt = r1 for now
        v0[c] = v1[c];
    }
}
#endif

__device__ void bounce_back(const Particle *p0, const float *rw, const float *vw, const float h, /**/ Particle *pn) {
    pn->v[X] = 2 * vw[X] - p0->v[X];
    pn->v[Y] = 2 * vw[Y] - p0->v[Y];
    pn->v[Z] = 2 * vw[Z] - p0->v[Z];

    pn->r[X] = rw[X] + (dt-h) * pn->v[X];
    pn->r[Y] = rw[Y] + (dt-h) * pn->v[Y];
    pn->r[Z] = rw[Z] + (dt-h) * pn->v[Z];
}

__device__ void lin_mom_change(const float v0[3], const float v1[3], /**/ float dP[3]) {
    dP[X] = -(v1[X] - v0[X]);
    dP[Y] = -(v1[Y] - v0[Y]);
    dP[Z] = -(v1[Z] - v0[Z]);
}

__device__ void ang_mom_change(const float r[3], const float v0[3], const float v1[3], /**/ float dL[3]) {
    dL[X] = -(r[Y] * v1[Z] - r[Z] * v1[Y]  -  r[Y] * v0[Z] + r[Z] - v0[Y]);
    dL[Y] = -(r[Z] * v1[X] - r[X] * v1[Z]  -  r[Z] * v0[X] + r[X] - v0[Z]);
    dL[Z] = -(r[X] * v1[Y] - r[Y] * v1[X]  -  r[X] * v0[Y] + r[Y] - v0[X]);
}

/* shift origin from 0 to R for ang momentum */
__device__ void mom_shift_ref(const float R[3], /**/ Momentum *m) {
    m->L[X] -= R[Y] * m->P[Z] - R[Z] * m->P[Y];
    m->L[Y] -= R[Z] * m->P[X] - R[X] * m->P[Z];
    m->L[Z] -= R[X] * m->P[Y] - R[Y] * m->P[X];
}

static __device__ bool nz(float a) {return fabs(a) > 1e-6f;}

__device__ bool nonzero(const Momentum *m) {
    return nz(m->P[X]) && nz(m->P[Y]) && nz(m->P[Z]) &&
        nz(m->L[X]) && nz(m->L[Y]) && nz(m->L[Z]);
}

enum {XX, XY, XZ, YY, YZ, ZZ};
/* see /poc/bounce-back/inertia.cpp */

static __device__ float Moment(const int d, const float A[3], const float B[3], const float C[3]) {
    return 0.25f * (A[d] * A[d] +
                    A[d] * B[d] +
                    B[d] * B[d] +
                    (A[d] + B[d]) * C[d] +
                    C[d] * C[d]);
}

static __device__ void compute_I(const float A[3], const float B[3], const float C[3], /**/ float I[6]) {
    
    const float Mxx = Moment(X, A, B, C);
    const float Myy = Moment(Y, A, B, C);
    const float Mzz = Moment(Z, A, B, C);

    const float D1 = C[X] + B[X] + 2 * A[X];
    const float D2 = C[X] + 2 * B[X] + A[X];
    const float D3 = 2 * C[X] + B[X] + A[X];

    const float D4 = C[Y] + B[Y] + 2 * A[Y];
    const float D5 = C[Y] + 2 * B[Y] + A[Y];
    const float D6 = 2 * C[Y] + B[Y] + A[Y];
    
    I[XX] = Myy + Mzz;
    I[XY] = -0.125f * (D1 * A[Y] + D2 * B[Y] + D3 * C[Y]);
    I[XY] = -0.125f * (D1 * A[Z] + D2 * B[Z] + D3 * C[Z]);
    I[YY] = Mzz + Mxx;
    I[YZ] = -0.125f * (D4 * A[Z] + D5 * B[Z] + D6 * C[Z]);
    I[ZZ] = Myy + Mxx;
}

static __device__ void inverse(const float A[6], /**/ float I[6]) {

    /* minors */
    const float mx = A[YY] * A[ZZ] - A[YZ] * A[YZ];
    const float my = A[XY] * A[ZZ] - A[XZ] * A[YZ];
    const float mz = A[XY] * A[YZ] - A[XZ] * A[YY];

    /* inverse determinant */
    float idet = mx * A[XX] - my * A[XY] + mz * A[XZ];
    assert( fabs(idet) > 1e-8f );
    idet = 1.f / idet;    
    
    I[XX] =  idet * mx;
    I[XY] = -idet * my;
    I[XZ] =  idet * mz;
    I[YY] =  idet * (A[XX] * A[ZZ] - A[XZ] * A[XZ]);
    I[YZ] =  idet * (A[XY] * A[XZ] - A[XX] * A[YZ]);
    I[ZZ] =  idet * (A[XX] * A[YY] - A[XY] * A[XY]);
}

static __device__ void v2f(const float r[3], const float om[3], const float v[3], /**/ float f[3]) {
    const float fac = rbc_mass / dt;
    f[X] = fac * (v[X] + r[Y] * om[Z] - r[Z] * om[Y]);
    f[Y] = fac * (v[Y] + r[Z] * om[X] - r[X] * om[Z]);
    f[Z] = fac * (v[Z] + r[X] * om[Y] - r[Y] * om[X]);
}

__device__ void M2f(const Momentum m, const float a[3], const float b[3], const float c[3],
              /**/ float fa[3], float fb[3], float fc[3]) {

    float I[6] = {0}, Iinv[6], om[3], v[3];
    const float fac = 1.f / (3.f * rbc_mass);
    
    compute_I(a, b, c, /**/ I);
    inverse(I, /**/ Iinv);

    /* angular velocity to be added (w.r.t. origin) */
    om[X] = fac * (I[XX] * m.L[X] + I[XY] * m.L[Y] + I[XZ] * m.L[Z]);
    om[Y] = fac * (I[XY] * m.L[X] + I[YY] * m.L[Y] + I[YZ] * m.L[Z]);
    om[Z] = fac * (I[XZ] * m.L[X] + I[YZ] * m.L[Y] + I[ZZ] * m.L[Z]);

    /* linear velocity to be added */
    v[X] =  fac * (m.P[X]);
    v[Y] =  fac * (m.P[Y]);
    v[Z] =  fac * (m.P[Z]);

    v2f(a, om, v, /**/ fa);
    v2f(b, om, v, /**/ fb);
    v2f(c, om, v, /**/ fc);
}

} // dev
