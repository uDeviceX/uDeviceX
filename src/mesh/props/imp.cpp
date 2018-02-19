#include <vector_types.h>
#include <string.h>

#include "inc/type.h"
#include "mesh/props.h"

enum {X, Y, Z};
enum {XX, XY, XZ, YY, YZ, ZZ};
    
#define Ax A[X]
#define Ay A[Y]
#define Az A[Z]

#define Bx B[X]
#define By B[Y]
#define Bz B[Z]

#define Cx C[X]
#define Cy C[Y]
#define Cz C[Z]

/* see http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf */
    
static void M_0(const float *A, const float *B, const float *C, /**/ float *res) {
    *res =  (+Ax * (By * Cz -Bz * Cy)
             -Ay * (Bx * Cz -Bz * Cx)
             +Az * (Bx * Cy -By * Cx)) / 6.f;
}

static void M_1(const float *A, const float *B, const float *C, /**/ float *res) {
    const float common = (+Ax * (By * Cz -Bz * Cy)
                          -Ay * (Bx * Cz -Bz * Cx)
                          +Az * (Bx * Cy -By * Cx)) / 24.f;

    res[0] = common * (Ax + Bx + Cx);
    res[1] = common * (Ay + By + Cy);
    res[2] = common * (Az + Bz + Cz);
}

static void M_2(const float *A, const float *B, const float *C, /**/ float *res) {
    /* see /poc/mesh/moments.mac */
    const float D1 = (+Ax * (By*Cz - Bz*Cy)
                      -Ay * (Bx*Cz - Bz*Cx)
                      +Az * (Bx*Cy - By*Cx)) / 60.f;
    const float D2 = By+Ay;
    const float D3 = Bz+Az;
    const float D4 = 2*Bx+Ax;
    const float D5 = 2*Cx+Bx+Ax;
    const float D6 = 0.5f * D1;
        
    res[XX] = (Ax*Ax + Ax*Bx + Bx*Bx + (Bx+Ax)*Cx + Cx*Cx) * D1;
    res[YY] = (Ay*Ay + Ay*By + By*By +    D2 * Cy + Cy*Cy) * D1;
    res[ZZ] = (Az*Az + Az*Bz + Bz*Bz +    D3 * Cz + Cz*Cz) * D1;
 
    res[XY] = (2 * Ax*Ay + Ay*Bx +        D4*By + D2*Cx +           D5*Cy) * D6;
    res[XZ] = (2 * Ax*Az + Az*Bx +        D4*Bz + D3*Cx +           D5*Cz) * D6;
    res[YZ] = (2 * Ay*Az + Az*By + (2*By+Ay)*Bz + D3*Cy + (2*Cy+By+Ay)*Cz) * D6;
}

#undef Ax
#undef Ay
#undef Az

#undef Bx
#undef By
#undef Bz

#undef Cx
#undef Cy
#undef Cz

#define load_t(vv, tid) {vv[3*tid + 0], vv[3*tid + 1], vv[3*tid + 2]}

float mesh_volume(int nt, const int4 *tt, const float *vv) {
    float Vtot = 0;
        
    for (int it = 0; it < nt; ++it) {
        int4 t = tt[it];

        const float A[3] = load_t(vv, t.x);
        const float B[3] = load_t(vv, t.y);
        const float C[3] = load_t(vv, t.z);

        float V = 0;            
        M_0(A, B, C, /**/ &V);

        Vtot += V;
    }
    return Vtot;
}

void mesh_center_of_mass(int nt, const int4 *tt, const float *vv, /**/ float *com) {
    float Vtot = 0, M1tot[3] = {0};
        
    for (int it = 0; it < nt; ++it) {
        int4 t = tt[it];
        
        const float A[3] = load_t(vv, t.x);
        const float B[3] = load_t(vv, t.y);
        const float C[3] = load_t(vv, t.z);

        float V = 0, M1[3];
            
        M_0(A, B, C, /**/ &V);
        M_1(A, B, C, /**/ M1);

        Vtot += V;
        M1tot[X] += M1[X];
        M1tot[Y] += M1[Y];
        M1tot[Z] += M1[Z];
    }

    com[X] = M1tot[X] / Vtot;
    com[Y] = M1tot[Y] / Vtot;
    com[Z] = M1tot[Z] / Vtot;
}

static void shift(const float s[3], /**/ float a[3]) {
    a[X] -= s[X]; a[Y] -= s[Y]; a[Z] -= s[Z];
}
    
void mesh_inertia_tensor(int nt, const int4 *tt, const float *vv, const float *com, const float density, /**/ float *I) {
    memset(I, 0, 6 * sizeof(float));
        
    for (int it = 0; it < nt; ++it) {
        int4 t = tt[it];
        
        float A[3] = load_t(vv, t.x);
        float B[3] = load_t(vv, t.y);
        float C[3] = load_t(vv, t.z);

        shift(com, /**/ A);
        shift(com, /**/ B);
        shift(com, /**/ C);

        float M2[6];
        M_2(A, B, C, /**/ M2);

        I[XX] += M2[YY] + M2[ZZ];
        I[YY] += M2[XX] + M2[ZZ];
        I[ZZ] += M2[XX] + M2[YY];

        I[XY] -= M2[XY];
        I[XZ] -= M2[XZ];
        I[YZ] -= M2[YZ];
    }

    for (int c = 0; c < 6; ++c)
        I[c] *= density;
}

#undef load_t
