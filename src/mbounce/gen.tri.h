namespace mbounce {
namespace sub {

#define __host__ __device__ __device__ __host__

enum {XX, XY, XZ, YY, YZ, ZZ};
/* see /poc/bounce-back/inertia.cpp */

static __host__ __device__ float Moment(const int d, const float A[3], const float B[3], const float C[3]) {
    return 0.25f * (A[d] * A[d] +
                    A[d] * B[d] +
                    B[d] * B[d] +
                    (A[d] + B[d]) * C[d] +
                    C[d] * C[d]);
}

static __host__ __device__ void compute_I(const float A[3], const float B[3], const float C[3], /**/ float I[6]) {
    
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

static __host__ __device__ void inverse(const float A[6], /**/ float I[6]) {

    /* minors */
    const float mx = A[YY] * A[ZZ] - A[YZ] * A[YZ];
    const float my = A[XY] * A[ZZ] - A[XZ] * A[YZ];
    const float mz = A[XY] * A[YZ] - A[XZ] * A[YY];

    /* inverse determinant */
    float idet = mx * A[XX] - my * A[XY] + mz * A[XZ];
    idet = 1.f / idet;    
    
    I[XX] =  idet * mx;
    I[XY] = -idet * my;
    I[XZ] =  idet * mz;
    I[YY] =  idet * (A[XX] * A[ZZ] - A[XZ] * A[XZ]);
    I[YZ] =  idet * (A[XY] * A[XZ] - A[XX] * A[YZ]);
    I[ZZ] =  idet * (A[XX] * A[YY] - A[XY] * A[XY]);
}

static __host__ __device__ void v2f(const float r[3], const float om[3], const float v[3], /**/ float f[3]) {
    const float fac = rbc_mass / dt;
    f[X] = fac * (v[X] + r[Y] * om[Z] - r[Z] * om[Y]);
    f[Y] = fac * (v[Y] + r[Z] * om[X] - r[X] * om[Z]);
    f[Z] = fac * (v[Z] + r[X] * om[Y] - r[Y] * om[X]);
}

__host__ __device__ void M2f(const Momentum m, const float a[3], const float b[3], const float c[3],
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

} // sub
} // mbounce
