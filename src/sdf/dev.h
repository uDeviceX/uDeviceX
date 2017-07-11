/* 3D texture object binded to  cuda array */
template <typename T>
struct tex3Dca {
    cudaTextureObject_t to;

    __device__ __forceinline__
    const T fetch(const float i, const float j, const float k) const {
        return tex3D<T>(to, i, j, k);
    }
    
    void setup(cudaArray *ca) {
        cudaResourceDesc resD;
        cudaTextureDesc  texD;

        memset(&resD, 0, sizeof(resD));
        resD.resType = cudaResourceTypeArray;
        resD.res.array.array = ca;
        
        memset(&texD, 0, sizeof(texD));
        texD.normalizedCoords = 0;
        texD.filterMode = cudaFilterModePoint;
        texD.mipmapFilterMode = cudaFilterModePoint;
        texD.addressMode[0] = cudaAddressModeWrap;
        texD.addressMode[1] = cudaAddressModeWrap;
        texD.addressMode[2] = cudaAddressModeWrap;

        CC(cudaCreateTextureObject(&to, &resD, &texD, NULL));
    }

    void destroy() {CC(cudaDestroyTextureObject(to));}
};

__device__ float sdf(const tex3Dca<float> texsdf, float x, float y, float z) {
    int c;
    float t;
    float s000, s001, s010, s100, s101, s011, s110, s111;
    float s00x, s01x, s10x, s11x;
    float s0yx, s1yx;
    float szyx;

    float tc[3], lmbd[3], r[3] = {x, y, z};
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM}; /* margin */
    int T[3] = {XTE, YTE, ZTE}; /* texture */

    for (c = 0; c < 3; ++c) {
        t = T[c] * (r[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]) - 0.5;
        lmbd[c] = t - (int)t;
        tc[c] = (int)t + 0.5;
    }
#define tex0(ix, iy, iz) (texsdf.fetch(tc[0] + ix, tc[1] + iy, tc[2] + iz))
    s000 = tex0(0, 0, 0), s001 = tex0(1, 0, 0), s010 = tex0(0, 1, 0);
    s011 = tex0(1, 1, 0), s100 = tex0(0, 0, 1), s101 = tex0(1, 0, 1);
    s110 = tex0(0, 1, 1), s111 = tex0(1, 1, 1);
#undef tex0

#define wavrg(A, B, p) A*(1-p) + p*B /* weighted average */
    s00x = wavrg(s000, s001, lmbd[0]);
    s01x = wavrg(s010, s011, lmbd[0]);
    s10x = wavrg(s100, s101, lmbd[0]);
    s11x = wavrg(s110, s111, lmbd[0]);

    s0yx = wavrg(s00x, s01x, lmbd[1]);

    s1yx = wavrg(s10x, s11x, lmbd[1]);
    szyx = wavrg(s0yx, s1yx, lmbd[2]);
#undef wavrg
    return szyx;
}

/* within the rescaled texel width error */
__device__ float cheap_sdf(const tex3Dca<float> texsdf, float x, float y, float z)  {
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};

    float tc[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
      tc[c] = (int)(T[c] * (r[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]));

#define tex0(ix, iy, iz) (texsdf.fetch(tc[0] + ix, tc[1] + iy, tc[2] + iz))
    return tex0(0, 0, 0);
#undef  tex0
}

__device__ float3 ugrad_sdf(const tex3Dca<float> texsdf, float x, float y, float z) {
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};

    float tc[3], fcts[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
      tc[c] = (int)(T[c] * (r[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]));
    for (int c = 0; c < 3; ++c)
      fcts[c] = T[c] / (2 * M[c] + L[c]);

#define tex0(ix, iy, iz) (texsdf.fetch(tc[0] + ix, tc[1] + iy, tc[2] + iz))
    float myval = tex0(0, 0, 0);
    float gx = fcts[0] * (tex0(1, 0, 0) - myval);
    float gy = fcts[1] * (tex0(0, 1, 0) - myval);
    float gz = fcts[2] * (tex0(0, 0, 1) - myval);
#undef tex0

    return make_float3(gx, gy, gz);
}

__device__ float3 grad_sdf(const tex3Dca<float> texsdf, float x, float y, float z) {
    float gx, gy, gz;
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};
    float tc[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
      tc[c] = T[c] * (r[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]) - 0.5;

#define tex0(ix, iy, iz) (texsdf.fetch(tc[0] + ix, tc[1] + iy, tc[2] + iz))
    gx = tex0(1, 0, 0) - tex0(-1,  0,  0);
    gy = tex0(0, 1, 0) - tex0( 0, -1,  0);
    gz = tex0(0, 0, 1) - tex0( 0,  0, -1);
#undef tex0

    float ggmag = sqrt(gx*gx + gy*gy + gz*gz);
    if (ggmag > 1e-6) { gx /= ggmag; gy /= ggmag; gz /= ggmag; }
    return make_float3(gx, gy, gz);
}

__global__ void fill_keys(const tex3Dca<float> texsdf, const Particle *const pp, const int n,
                          int *const key) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle p = pp[pid];
    float sdf0 = sdf(texsdf, p.r[X], p.r[Y], p.r[Z]);
    key[pid] = (int)(sdf0 >= 0) + (int)(sdf0 > 2);
}

__device__ void handle_collision(const tex3Dca<float> texsdf, float currsdf,
                                 float &x, float &y, float &z,
                                 float &vx, float &vy, float &vz) {
    float x0 = x - vx*dt, y0 = y - vy*dt, z0 = z - vz*dt;
    if (sdf(texsdf, x0, y0, z0) >= 0) { /* this is the worst case - 0 position
                                           was bad already we need to search
                                           and rescue the particle */
        float3 dsdf = grad_sdf(texsdf, x, y, z); float sdf0 = currsdf;
        x -= sdf0 * dsdf.x; y -= sdf0 * dsdf.y; z -= sdf0 * dsdf.z;
        for (int l = 8; l >= 1; --l) {
            if (sdf(texsdf, x, y, z) < 0) {
                /* we are confused anyway! use particle position as wall
                   position */
                k_wvel::bounce_vel(x, y, z, &vx, &vy, &vz); return;
            }
            float jump = 1.1f * sdf0 / (1 << l);
            x -= jump * dsdf.x; y -= jump * dsdf.y; z -= jump * dsdf.z;
        }
    }

    /*
      Bounce back (stage I)

      Find wall position (sdf(wall) = 0): make two steps of Newton's
      method for the equation phi(t) = 0, where phi(t) = sdf(rr(t))
      and rr(t) = [x + vx*t, y + vy*t, z + vz*t]. We are going back
      and `t' is in [-dt, 0].

      dphi = v . grad(sdf). Newton step is t_new = t_old - phi/dphi

      Give up if dsdf is small. Cap `t' to [-dt, 0].
    */
#define rr(t) make_float3(x + vx*t, y + vy*t, z + vz*t)
#define small(phi) (fabs(phi) < 1e-6)
    float3 r, dsdf; float phi, dphi, t = 0;
    r = rr(t); phi = currsdf;
    dsdf = ugrad_sdf(texsdf, r.x, r.y, r.z);
    dphi = vx*dsdf.x + vy*dsdf.y + vz*dsdf.z; if (small(dphi)) goto giveup;
    t -= phi/dphi;                            if (t < -dt) t = -dt; if (t > 0) t = 0;

    r = rr(t); phi = sdf(texsdf, r.x, r.y, r.z);
    dsdf = ugrad_sdf(texsdf, r.x, r.y, r.z);
    dphi = vx*dsdf.x + vy*dsdf.y + vz*dsdf.z; if (small(dphi)) goto giveup;
    t -= phi/dphi;                            if (t < -dt) t = -dt; if (t > 0) t = 0;
#undef rr
#undef small
 giveup:
    /* Bounce back (stage II)
       change particle position and velocity
    */
    float xw = x + t*vx, yw = y + t*vy, zw = z + t*vz; /* wall */
    x += 2*t*vx; y += 2*t*vy; z += 2*t*vz; /* bouncing relatively to
                                              wall */
    k_wvel::bounce_vel(xw, yw, zw, &vx, &vy, &vz);
    if (sdf(texsdf, x, y, z) >= 0) {x = x0; y = y0; z = z0;}
}

__global__ void bounce(const tex3Dca<float> texsdf, float2 *const pp, int n) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float2 data0 = pp[pid * 3];
    float2 data1 = pp[pid * 3 + 1];
    float s = cheap_sdf(texsdf, data0.x, data0.y, data1.x);
    if (s >= -1.7320 * XSIZE_WALLCELLS / XTE) {
      float currsdf = sdf(texsdf, data0.x, data0.y, data1.x);
      float2 data2 = pp[pid * 3 + 2];
      if (currsdf >= 0) {
	handle_collision(texsdf, currsdf, data0.x, data0.y, data1.x, data1.y, data2.x, data2.y);
	pp[3 * pid] = data0;
	pp[3 * pid + 1] = data1;
	pp[3 * pid + 2] = data2;
      }
    }
}

