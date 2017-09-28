namespace dev {

enum {X, Y, Z};

// TODO belongs to scheme/ ?
// BB assumes r0 + v0 dt = r1 for now
#ifdef FORWARD_EULER
__device__ void rvprev(const real3_t *r1, const real3_t *v1, const float *f0, /**/ real3_t *r0, real3_t *v0) {
    enum {X, Y, Z};
    v0->x = v1->x - f0[X] * dt;
    v0->y = v1->y - f0[Y] * dt;
    v0->z = v1->z - f0[Z] * dt;
    
    r0->x = r1->x - v0->x * dt;
    r0->y = r1->y - v0->y * dt;
    r0->z = r1->z - v0->z * dt;
}
#else // velocity-verlet
__device__ void rvprev(const real3_t *r1, const real3_t *v1, const float *, /**/ real3_t *r0, real3_t *v0) {
    r0->x = r1->x - v1->x * dt;    
    r0->y = r1->y - v1->y * dt;
    r0->z = r1->z - v1->z * dt;    

    *v0 = *v1;
}
#endif

__device__ void bounce_back(const rPa *p0, const real3_t *rw, const real3_t *vw, const real_t h, /**/ rPa *pn) {
    pn->v.x = 2 * vw->x - p0->v.x;
    pn->v.y = 2 * vw->y - p0->v.y;
    pn->v.z = 2 * vw->z - p0->v.z;

    pn->r.x = rw->x + (dt-h) * pn->v.x;
    pn->r.y = rw->y + (dt-h) * pn->v.y;
    pn->r.z = rw->z + (dt-h) * pn->v.z;
}

__device__ void lin_mom_change(const real3_t v0, const real3_t v1, /**/ float dP[3]) {
    dP[X] = -(v1.x - v0.x);
    dP[Y] = -(v1.y - v0.y);
    dP[Z] = -(v1.z - v0.z);
}

__device__ void ang_mom_change(const real3_t r, const real3_t v0, const real3_t v1, /**/ float dL[3]) {
    dL[X] = -(r.y * v1.z - r.z * v1.y  -  r.y * v0.z + r.z - v0.y);
    dL[Y] = -(r.z * v1.x - r.x * v1.z  -  r.z * v0.x + r.x - v0.z);
    dL[Z] = -(r.x * v1.y - r.y * v1.x  -  r.x * v0.y + r.y - v0.x);
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
