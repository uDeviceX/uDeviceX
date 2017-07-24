namespace forces {

template<int s>
inline __device__ float viscosity_function(float x) { return sqrtf(viscosity_function<s - 1>(x)); }
template<> inline __device__ float viscosity_function<1>(float x) { return sqrtf(x); }
template<> inline __device__ float viscosity_function<0>(float x) { return x;        }

inline __device__ void dpd00(int typed, int types,
                             float x, float y, float z,
                             float vx, float vy, float vz,
                             float rnd, float *fx, float *fy, float *fz) {
    float gammadpd[] = {gammadpd_solv, gammadpd_solid, gammadpd_wall, gammadpd_rbc};
    float aij[] = {aij_solv, aij_solid, aij_wall, aij_rbc};

    float rij2 = x * x + y * y + z * z;
    float invrij = rsqrtf(rij2);
    float rij = rij2 * invrij;

    if (rij2 >= 1) {
        *fx = *fy = *fz = 0;
        return;
    }

    float argwr = 1.f - rij;
    float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

    x *= invrij;
    y *= invrij;
    z *= invrij;

    float rdotv = x * vx + y * vy + z * vz;

    float gammadpd_pair = 0.5 * (gammadpd[typed] + gammadpd[types]);
    float sigmaf_pair = sqrt(2*gammadpd_pair*kBT / dt);
    float f = (-gammadpd_pair * wr * rdotv + sigmaf_pair * rnd) * wr;

    bool ss = (typed == SOLID_TYPE) && (types == SOLID_TYPE);
    bool sw = (typed == SOLID_TYPE) && (types ==  WALL_TYPE);

    if (ss || sw) {
        /*hack*/ const float ljsi = ss ? ljsigma : 2 * ljsigma;
        float invr2 = invrij * invrij;
        float t2 = ljsi * ljsi * invr2;
        float t4 = t2 * t2;
        float t6 = t4 * t2;
        float lj = min(1e4f, max(0.f, ljepsilon * 24.f * invrij * t6 * (2.f * t6 - 1.f)));
        f += lj;
    } 

    float aij_pair = 0.5 * (aij[typed] + aij[types]);
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
