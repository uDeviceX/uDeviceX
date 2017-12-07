static __device__ void p2rv(const Particle *pp, int i, /**/ float3 *r, float3 *v) {
    enum {X, Y, Z};
    Particle p = pp[i];
    
    r->x = p.r[X];
    r->y = p.r[Y];
    r->z = p.r[Z];

    v->x = p.v[X];
    v->y = p.v[Y];
    v->z = p.v[Z];
}

static __device__ void rv2p(float3 r, float3 v, int i, /**/ Particle *pp) {
    enum {X, Y, Z};
    Particle p = {r.x, r.y, r.z, v.x, v.y, v.z};
    pp[i] = p;
}

static __device__ float3 ugrad_sdf(const sdf::tex3Dca texsdf, const float3 *pos) {
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};
    int tc[3];
    float fcts[3], r[3] = {pos->x, pos->y, pos->z};

    for (int c = 0; c < 3; ++c)
        tc[c] = T[c] * (r[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]);

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

static __device__ float3 grad_sdf(const sdf::tex3Dca texsdf, const float3 *pos) {
    float gx, gy, gz;
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};
    float tc[3], r[3] = {pos->x, pos->y, pos->z};
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

static __device__ bool is_small(float f) {return fabs(f) < 1e-6f;}

static __device__ void main0(const sdf::tex3Dca texsdf, float currsdf, /* io */ float3 *r, float3 *v) {
    float3 r0, rc, rw, dsdf;
    float sdf0, jump, phi, dphi, t;
    int l;
    r0 = *r;
    axpy(-dt, v, /**/ &r0);

    if (sdf::sub::dev::sdf(texsdf, r0.x, r0.y, r0.z) >= 0) {
        dsdf = grad_sdf(texsdf, r);
        sdf0 = currsdf;

        axpy(-sdf0, &dsdf, /**/ r);
        
        for (l = 8; l >= 1; --l) {
            if (sdf::sub::dev::sdf(texsdf, r->x, r->y, r->z) < 0) {
                k_wvel::bounce_vel(r->x, r->y, r->z, &v->x, &v->y, &v->z);
                return;
            }
            jump = 1.1f * sdf0 / (1 << l);
            axpy(-jump, &dsdf, /**/ r);
        }
    }

    t = 0;

    for (l = 0; l < 2; ++l) {
        rc = *r;
        axpy(t, v, /**/ &rc);
        phi = currsdf;
        dsdf = ugrad_sdf(texsdf, &rc);
        dphi = dot<float> (v, &dsdf);

        if (is_small(dphi))
            break;
        
        t -= phi/dphi;

        if (t < -dt)
            t = -dt;
        if (t > 0)
            t = 0;
    }

    rw = *r;
    axpy(t, v, /**/ &rw);
    k_wvel::bounce_vel(rw.x, rw.y, rw.z, &v->x, &v->y, &v->z);

    *r = rw;
    axpy(-t, v, /**/ r);
    
    if (sdf::sub::dev::sdf(texsdf, r->x, r->y, r->z) >= 0)
        *r = r0;    
}

__global__ void main(const sdf::tex3Dca texsdf, int n, /**/ Particle *pp) {
    float s, currsdf;
    float3 r, v;
    int i;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    p2rv(pp, i, /**/ &r, &v);

    s = sdf::sub::dev::cheap_sdf(texsdf, r.x, r.y, r.z);

    if (s >= -1.7320 * XSIZE_WALLCELLS / XTE) {
        currsdf = sdf::sub::dev::sdf(texsdf, r.x, r.y, r.z);
        if (currsdf >= 0) {
            main0(texsdf, currsdf, /*io*/ &r, &v);
            rv2p(r, v, i, /**/ pp);
        }
    }
}
