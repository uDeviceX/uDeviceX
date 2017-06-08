namespace k_sdf {
texture<float, 3, cudaReadModeElementType> texSDF;

__device__ float sdf(float x, float y, float z) {
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XWM, YWM, ZWM};
    int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};

    float tc[3], lmbd[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c) {
        float t =
            TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) / (L[c] + 2 * MARGIN[c]) - 0.5;

        lmbd[c] = t - (int)t;
        tc[c] = (int)t + 0.5;
    }
#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    float s000 = tex0(0, 0, 0), s001 = tex0(1, 0, 0), s010 = tex0(0, 1, 0);
    float s011 = tex0(1, 1, 0), s100 = tex0(0, 0, 1), s101 = tex0(1, 0, 1);
    float s110 = tex0(0, 1, 1), s111 = tex0(1, 1, 1);
#undef tex0

#define wavrg(A, B, p) A*(1-p) + p*B /* weighted average */
    float s00x = wavrg(s000, s001, lmbd[0]);
    float s01x = wavrg(s010, s011, lmbd[0]);
    float s10x = wavrg(s100, s101, lmbd[0]);
    float s11x = wavrg(s110, s111, lmbd[0]);

    float s0yx = wavrg(s00x, s01x, lmbd[1]);

    float s1yx = wavrg(s10x, s11x, lmbd[1]);
    float szyx = wavrg(s0yx, s1yx, lmbd[2]);
#undef wavrg
    return szyx;
}

/* within the rescaled texel width error */
__device__ float cheap_sdf(float x, float y, float z)  {
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XWM, YWM, ZWM};
    int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};

    float tc[3], r[3] = {x, y, z};;
    for (int c = 0; c < 3; ++c)
    tc[c] = (int)(TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) /
                  (L[c] + 2 * MARGIN[c]));
#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    return tex0(0, 0, 0);
#undef  tex0
}

__device__ float3 ugrad_sdf(float x, float y, float z) {
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XWM, YWM, ZWM};
    int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};

    float tc[3], fcts[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
    tc[c] = (int)(TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) /
                  (L[c] + 2 * MARGIN[c]));
    for (int c = 0; c < 3; ++c) fcts[c] = TEXSIZES[c] / (2 * MARGIN[c] + L[c]);

#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    float myval = tex0(0, 0, 0);
    float gx = fcts[0] * (tex0(1, 0, 0) - myval);
    float gy = fcts[1] * (tex0(0, 1, 0) - myval);
    float gz = fcts[2] * (tex0(0, 0, 1) - myval);
#undef tex0

    return make_float3(gx, gy, gz);
}

__device__ float3 grad_sdf(float x, float y, float z) {
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XWM, YWM, ZWM};
    int TEXSIZES[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};

    float tc[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
    tc[c] =
        TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) / (L[c] + 2 * MARGIN[c]) - 0.5;

    float gx, gy, gz;
#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    gx = tex0(1, 0, 0) - tex0(-1,  0,  0);
    gy = tex0(0, 1, 0) - tex0( 0, -1,  0);
    gz = tex0(0, 0, 1) - tex0( 0,  0, -1);
#undef tex0

    float ggmag = sqrt(gx*gx + gy*gy + gz*gz);

    if (ggmag > 1e-6) {
        gx /= ggmag; gy /= ggmag; gz /= ggmag;
    }
    return make_float3(gx, gy, gz);
}

__global__ void fill_keys(const Particle *const pp, const int n,
                          int *const key) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle p = pp[pid];
    float sdf0 = sdf(p.r[0], p.r[1], p.r[2]);
    key[pid] = (int)(sdf0 >= 0) + (int)(sdf0 > 2);
}

__global__ void strip_solid4(Particle *const src, const int n, float4 *dst) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle p = src[pid];
    dst[pid] = make_float4(p.r[0], p.r[1], p.r[2], 0);
}

__device__ void handle_collision(float currsdf,
                                 float &x, float &y, float &z,
                                 float &vx, float &vy, float &vz) {
    float x0 = x - vx*dt, y0 = y - vy*dt, z0 = z - vz*dt;
    if (sdf(x0, y0, z0) >= 0) { /* this is the worst case - 0 position
                                   was bad already we need to search
                                   and rescue the particle */
        float3 dsdf = grad_sdf(x, y, z); float sdf0 = currsdf;
        x -= sdf0 * dsdf.x; y -= sdf0 * dsdf.y; z -= sdf0 * dsdf.z;
        for (int l = 8; l >= 1; --l) {
            if (sdf(x, y, z) < 0) {
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
    dsdf = ugrad_sdf(r.x, r.y, r.z);
    dphi = vx*dsdf.x + vy*dsdf.y + vz*dsdf.z; if (small(dphi)) goto giveup;
    t -= phi/dphi;                            if (t < -dt) t = -dt; if (t > 0) t = 0;

    r = rr(t); phi = sdf(r.x, r.y, r.z);
    dsdf = ugrad_sdf(r.x, r.y, r.z);
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
    if (sdf(x, y, z) >= 0) {x = x0; y = y0; z = z0;}
}

__global__ void bounce(float2 *const pp, int nparticles) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= nparticles) return;
    float2 data0 = pp[pid * 3];
    float2 data1 = pp[pid * 3 + 1];
    if (pid < nparticles) {
        float mycheapsdf = cheap_sdf(data0.x, data0.y, data1.x);

        if (mycheapsdf >=
            -1.7320f * ((float)XSIZE_WALLCELLS / (float)XTEXTURESIZE)) {
            float currsdf = sdf(data0.x, data0.y, data1.x);

            float2 data2 = pp[pid * 3 + 2];

            float3 v0 = make_float3(data1.y, data2.x, data2.y);

            if (currsdf >= 0) {
                handle_collision(currsdf, data0.x, data0.y, data1.x, data1.y, data2.x,
                                 data2.y);

                pp[3 * pid] = data0;
                pp[3 * pid + 1] = data1;
                pp[3 * pid + 2] = data2;
            }
        }
    }
}
} /* namespace k_sdf */

