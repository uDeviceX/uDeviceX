/* 
   compute inertia tensor of a triangle 
   see trianglemom.mac
*/

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

    I[XY] = 0.125f * (D1 * A[Y] + D2 * B[Y] + D3 * C[Y]);
    I[XY] = 0.125f * (D1 * A[Z] + D2 * B[Z] + D3 * C[Z]);

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
}


int main() {
    return 0;
}

