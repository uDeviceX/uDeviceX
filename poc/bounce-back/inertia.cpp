/* 
   compute inertia tensor of a triangle 
   see trianglemom.mac
*/

#include <cstdio>

enum {X, Y, Z};
enum {XX, XY, XZ, YY, YZ, ZZ};

float Moment(const int d, const float A[3], const float B[3], const float C[3]) {
    return 0.25f * (A[d] * A[d] +
                    A[d] * B[d] +
                    B[d] * B[d] +
                    (A[d] + B[d]) * C[d] +
                    C[d] * C[d]);
}

void compute_I(const float A[3], const float B[3], const float C[3], /**/ float I[6]) {

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

void inverse(const float A[6], /**/ float I[6]) {

    /* minors */
    const float mx = A[YY] * A[ZZ] - A[YZ] * A[YZ];
    const float my = A[XY] * A[ZZ] - A[XZ] * A[YZ];
    const float mz = A[XY] * A[YZ] - A[XZ] * A[YY];

    /* inverse determinant */
    float idet = mx * A[XX] - my * A[XY] + mz * A[XZ];
    idet = 1.f / idet;

    I[XX] =  idet * mx;
    I[XY] = -idet * my;
    I[XZ] =  idet * mx;
    I[YY] =  idet * (A[XX] * A[ZZ] - A[XZ] * A[XZ]);
    I[YZ] =  idet * (A[XY] * A[XZ] - A[XX] * A[YZ]);
    I[ZZ] =  idet * (A[XX] * A[YY] - A[XY] * A[XY]);
}

void verify(const float A[6], const float I[6]) {
    const float xx = A[XX] * I[XX] + A[XY] * I[XY] + A[XZ] * I[XZ];
    const float yy = A[XY] * I[XY] + A[YY] * I[YY] + A[YZ] * I[YZ];
    const float zz = A[XZ] * I[XZ] + A[YZ] * I[YZ] + A[ZZ] * I[ZZ];

    const float xy = A[XX] * I[XY] + A[XY] * I[YY] + A[XZ] * I[YZ];
    const float xz = A[XX] * I[XZ] + A[XY] * I[YZ] + A[XZ] * I[ZZ];
    const float yz = A[XY] * I[XZ] + A[YY] * I[YZ] + A[YZ] * I[ZZ];
    
    printf("%6e %6e %6e\n"
           "%6e %6e %6e\n"
           "%6e %6e %6e\n\n",
           xx, xy, xz,
           xy, yy, yz,
           xz, yz, zz);     
}

void print(const float I[6]) {
    printf("%6e %6e %6e\n"
           "%6e %6e %6e\n"
           "%6e %6e %6e\n\n",
           I[XX], I[XY], I[XZ],
           I[XY], I[YY], I[YZ],
           I[XZ], I[YZ], I[ZZ]); 
}

int main() {

    float A[] = {0, 0, 0};
    float B[] = {1, 0, 0};
    float C[] = {0, 1, 0};

    float I[6], Iinv[6];

    compute_I(A, B, C, /**/ I);
    print(I);

    inverse(I, /**/ Iinv);
    print(Iinv);
    verify(I, Iinv);
    
    return 0;
}

