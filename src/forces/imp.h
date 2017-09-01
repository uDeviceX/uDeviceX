namespace forces {

static __device__ float wrf(const int s, float x) {
    if (s == 0) return x;
    if (s == 1) return sqrtf(x);
    if (s == 2) return sqrtf(sqrtf(x));
    if (s == 3) return sqrtf(sqrtf(sqrtf(x)));
    return powf(x, 1.f/s);
}

inline __device__ float cap(float x, float lo, float hi) {
    if      (x > hi) return hi;
    else if (x < lo) return lo;
    else             return x;
}


inline __device__ bool norm(/*io*/ float *px, float *py, float *pz, /**/ float *pr, float *pinvr) {
    /* noralize vector r = [x, y, z]  ; return 1 if |r| > 1 */
    float x, y, z, invr, r;
    float r2;
    x = *px; y = *py; z = *pz;

    r2 = x*x + y*y + z*z;
    if (r2 >= 1) return 1;
    invr = rsqrtf(r2);
    r = r2 * invr;
    x *= invr; y *= invr; z *= invr;

    *px = x; *py = y; *pz = z; *pr = r; *pinvr = invr;
    return 0;
}

inline __device__ void dpd00(int typed, int types,
                             float x, float y, float z,
                             float vx, float vy, float vz,
                             float rnd, float *fx, float *fy, float *fz) {
    /* [tab]les */
    float gamma_tbl[] = {gammadpd_solv, gammadpd_solid, gammadpd_wall, gammadpd_rbc};
    float a_tbl[] = {aij_solv, aij_solid, aij_wall, aij_rbc};

    float r2, invr, r;
    float wc, wr; /* conservative and random kernels */
    float rm; /* 1 minus r */
    float ev; /* (e dot v) */
    float gamma, sigma;
    float f;
    float t2, t4, t6, lj;
    float a;

    r2 = x * x + y * y + z * z;
    if (r2 >= 1) {
        *fx = *fy = *fz = 0;
        return;
    }
    invr = rsqrtf(r2);
    r = r2 * invr;
    x *= invr; y *= invr; z *= invr;

    rm = max(1 - r, 0.0f);
    wc = rm;
    wr = wrf(-S_LEVEL, rm);
    ev = x*vx + y*vy + z*vz;

    gamma = 0.5 * (gamma_tbl[typed] + gamma_tbl[types]);
    a     = 0.5 * (a_tbl[typed] + a_tbl[types]);
    sigma = sqrt(2*gamma*kBT / dt);

    f  = (-gamma * wr * ev + sigma * rnd) * wr;
    f +=                                a * wc;

    bool ss = (typed == SOLID_TYPE) && (types == SOLID_TYPE);
    bool sw = (typed == SOLID_TYPE) && (types ==  WALL_TYPE);
    if (ss || sw) {
        /*hack*/ const float ljsi = ss ? ljsigma : 2 * ljsigma;
        t2 = ljsi * ljsi * invr * invr;
        t4 = t2 * t2;
        t6 = t4 * t2;
        lj = ljepsilon * 24 * invr * t6 * (2 * t6 - 1);
        lj = cap(lj, 0, 1e4);
        f += lj;
    }

    *fx = f*x; *fy = f*y; *fz = f*z;
}

inline __device__ void dpd0(int typed, int types,
                            float xd, float yd, float zd,
                            float xs, float ys, float zs,
                            float vxd, float vyd, float vzd,
                            float vxs, float vys, float vzs,
                            float rnd, float *fx, float *fy, float *fz) {
    /* to relative coordinates */
    float dx, dy, dz;
    float dvx, dvy, dvz;
    dx = xd - xs; dy = yd - ys; dz = zd - zs;
    dvx = vxd - vxs; dvy = vyd - vys; dvz = vzd - vzs;
    dpd00(typed, types, dx, dy, dz, dvx, dvy, dvz, rnd, /**/ fx, fy, fz);
}

inline __device__ float3 dpd(int t1, int t2,
                             float3 r1, float3 r2,
                             float3 v1, float3 v2,
                             float rnd) {
    /* unpack float3 */
    float fx, fy, fz;
    dpd0(t1, t2,
         r1.x, r1.y, r1.z,
         r2.x, r2.y, r2.z,
         v1.x, v1.y, v1.z,
         v2.x, v2.y, v2.z,
         rnd,
         &fx, &fy, &fz);

    return make_float3(fx, fy, fz);
}

}
