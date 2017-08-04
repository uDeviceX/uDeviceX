namespace mbounce {
namespace sub {

#define _DH_ __device__ __host__

enum {X, Y, Z};
enum {XX, XY, XZ, YY, YZ, ZZ};
/* see /poc/bounce-back/inertia.cpp */

_DH_ float Moment(const int d, const float A[3], const float B[3], const float C[3]) {
    return 0.25f * (A[d] * A[d] +
                    A[d] * B[d] +
                    B[d] * B[d] +
                    (A[d] + B[d]) * C[d] +
                    C[d] * C[d]);
}

_DH_ void compute_I(const float A[3], const float B[3], const float C[3], /**/ float I[6]) {
    
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

_DH_ void inverse(const float A[6], /**/ float I[6]) {

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

#undef _DH_

} // sub
} // mbounce
