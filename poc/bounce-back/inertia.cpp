/* 
   compute inertia tensor of a triangle 
   see trianglemom.mac
*/

#include <cstdio>

enum {X, Y, Z};
enum {XX, XY, XZ, YY, YZ, ZZ};

void compute_I(const float A[3], const float B[3], const float C[3], /**/ float I[6]) {

    const float D1 = C[X] + B[X] + 2 * A[X];
    const float D2 = C[X] + 2 * B[X] + A[X];
    const float D3 = 2 * C[X] + B[X] + A[X];

    I[XX] = 0.25f * (A[X] * A[X] +
                     A[X] * B[X] +
                     B[X] * B[X] +
                     (A[X] + B[X]) * C[X] +
                     C[X] * C[X]);
    
    I[YY] = 0.25f * (A[Y] * A[Y] +
                     A[Y] * B[Y] +
                     B[Y] * B[Y] +
                     (A[Y] + B[Y]) * C[Y] +
                     C[Y] * C[Y]);
 
    I[ZZ] = 0.25f * (A[Z] * A[Z] +
                     A[Z] * B[Z] +
                     B[Z] * B[Z] +
                     (A[Z] + B[Z]) * C[Z] +
                     C[Z] * C[Z]);

    I[XY] = 0.125f * (D1 * A[Y] + D2 * B[Y] + D3 * C[Y]);
    I[XZ] = 0.125f * (D1 * A[Z] + D2 * B[Z] + D3 * C[Z]);
    
    I[YZ] = 0.125f * ( (C[Y] + B[Y] + 2 * A[Y]) * A[Z] +
                       (C[Y] + 2 * B[Y] + A[Y]) * B[Z] +
                       (2 * C[Y] + B[Y] + A[Y]) * C[Z] );
    
}

void print(const float I[6]) {
    printf("%6e %6e %6e\n"
           "%6e %6e %6e\n"
           "%6e %6e %6e\n",
           I[XX], I[XY], I[XZ],
           I[XY], I[YY], I[YZ],
           I[XZ], I[YZ], I[ZZ]); 
}

int main() {

    float A[] = {0, 0, 0};
    float B[] = {1, 0, 0};
    float C[] = {0, 1, 0};

    float I[6];

    compute_I(A, B, C, /**/ I);

    print(I);
    
    return 0;
}

