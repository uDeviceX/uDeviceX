namespace forces {

static __device__ float wrf(const int s, float x) {
    if (s == 0) return x;
    if (s == 1) return sqrtf(x);
    if (s == 2) return sqrtf(sqrtf(x));
    if (s == 3) return sqrtf(sqrtf(sqrtf(x)));
    return powf(x, 1.f/s);
}

inline __device__ void dpd00(int typed, int types,
                             float x, float y, float z,
                             float vx, float vy, float vz,
                             float rnd, float *fx, float *fy, float *fz) {
    float gammadpd[] = {gammadpd_solv, gammadpd_solid, gammadpd_wall, gammadpd_rbc};
    float aij[] = {aij_solv, aij_solid, aij_wall, aij_rbc};

    float rij2, invrij, rij;
    float argwr, wr, rdotv, gammadpd_pair, sigmaf_pair;
    float f;
    float invr2, t2, t4, t6, lj;
    float aij_pair;

    rij2 = x * x + y * y + z * z;
    invrij = rsqrtf(rij2);
    rij = rij2 * invrij;

    if (rij2 >= 1) {
        *fx = *fy = *fz = 0;
        return;
    }

    argwr = max(1.f - rij, 0.f);
    wr = wrf(-S_LEVEL, argwr);

    x *= invrij;
    y *= invrij;
    z *= invrij;

    rdotv = x * vx + y * vy + z * vz;

    gammadpd_pair = 0.5 * (gammadpd[typed] + gammadpd[types]);
    sigmaf_pair = sqrt(2*gammadpd_pair*kBT / dt);
    f = (-gammadpd_pair * wr * rdotv + sigmaf_pair * rnd) * wr;

    bool ss = (typed == SOLID_TYPE) && (types == SOLID_TYPE);
    bool sw = (typed == SOLID_TYPE) && (types ==  WALL_TYPE);

    if (ss || sw) {
        /*hack*/ const float ljsi = ss ? ljsigma : 2 * ljsigma;
        invr2 = invrij * invrij;
        t2 = ljsi * ljsi * invr2;
        t4 = t2 * t2;
        t6 = t4 * t2;
        lj = min(1e4f, max(0.f, ljepsilon * 24.f * invrij * t6 * (2.f * t6 - 1.f)));
        f += lj;
    } 

    aij_pair = 0.5 * (aij[typed] + aij[types]);
    f += aij_pair * argwr;

    *fx = f * x;
    *fy = f * y;
    *fz = f * z;
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
