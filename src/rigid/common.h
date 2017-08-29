namespace rig {

enum {X, Y, Z};
enum {XX, XY, XZ, YY, YZ, ZZ};
enum {YX = XY, ZX = XZ, ZY = YZ};

#define _HD_ __host__ __device__

void update_r_hst(const float *rr0, const int n, const float *com, const float *e0, const float *e1, const float *e2, /**/ Particle *pp) {
    for (int ip = 0; ip < n; ++ip) {
        float *r0 = pp[ip].r;
        const float* ro = &rr0[3*ip];
        float x = ro[X], y = ro[Y], z = ro[Z];
        r0[X] = x*e0[X] + y*e1[X] + z*e2[X];
        r0[Y] = x*e0[Y] + y*e1[Y] + z*e2[Y];
        r0[Z] = x*e0[Z] + y*e1[Z] + z*e2[Z];

        r0[X] += com[X]; r0[Y] += com[Y]; r0[Z] += com[Z];
    }
}

} // rig
