static __device__ float cap(float x, float lo, float hi) {
    if      (x > hi) return hi;
    else if (x < lo) return lo;
    else             return x;
}

static const float EPS = 1e-6;
enum {NORM_OK, NORM_BIG, NORM_SMALL};
static __device__ int norm(/*io*/ float3 *pos, /**/ float *pr, float *pinvr) {
    /* normalize r = [x, y, z], sets |r| and 1/|r| if not big */
    float x, y, z, invr, r, r2;
    x = pos->x; y = pos->y; z = pos->z;

    r2 = x*x + y*y + z*z;
    if      (r2 >= 1 )   return NORM_BIG;
    else if (r2 < EPS) {
        *pr = pos->x = pos->y = pos->z = 0;
        return NORM_SMALL;
    } else {
        invr = rsqrtf(r2);
        r = r2 * invr;
        x *= invr; y *= invr; z *= invr;
        pos->x = x; pos->y = y; pos->z = z; *pr = r; *pinvr = invr;
        return NORM_OK;
    }
}

static __device__ float ker_wrf(const int s, float x) {
    if (s == 0) return x;
    if (s == 1) return sqrtf(x);
    if (s == 2) return sqrtf(sqrtf(x));
    if (s == 3) return sqrtf(sqrtf(sqrtf(x)));
    return powf(x, 1.f/s);
}

static __device__ float magn_dpd(float a, float g, float s, float rnd,
                                 float r, float ev) {
    float wr, wc;
    float rm, f0;

    rm = max(1 - r, 0.0f);
    wc = rm;
    wr = ker_wrf(S_LEVEL, rm);
    
    f0  = (-g * wr * ev + s * rnd) * wr;
    f0 +=                        a * wc;
    return f0;
}

static __device__ float magn_lj(float s, float e, float invr) {
    float t2, t4, t6, f;
    t2 = s * s * invr * invr;
    t4 = t2 * t2;
    t6 = t4 * t2;
    f = e * 24 * invr * t6 * (2 * t6 - 1);
    f = cap(f, 0, 1e4);
    return f;
}

static __device__ float force_magn(PairDPD p, float rnd, float ev, float r, float) {
    return magn_dpd(p.a, p.g, p.s, rnd, r, ev);
}

static __device__ float force_magn(PairDPDLJ p, float rnd, float ev, float r, float invr) {
    float f;
    f  = magn_dpd(p.a, p.g, p.s, rnd, r, ev);
    f += magn_lj(p.ljs, p.lje, invr);
    return f;
}

template <typename Param>
static __device__ void pair_force(Param p, PairPa a, PairPa b, float rnd, /**/ PairFo *f) {
    float r, invr, ev, f0;
    float3 dr, dv;
    int vnstat; /* vector normalization status */

    dr.x = a.x - b.x;
    dr.y = a.y - b.y;
    dr.z = a.z - b.z;

    dv.x = a.vx - b.vx;
    dv.y = a.vy - b.vy;
    dv.z = a.vz - b.vz;
    
    vnstat = norm(/*io*/ &dr, /*o*/ &r, &invr);
    if (vnstat == NORM_BIG) {
        f->x = f->y = f->z = 0;
        return;
    }

    ev = dot<float>(&dr, &dv);
    
    f0 = force_magn(p, rnd, ev, r, invr);
    
    f->x = f0 * dr.x;
    f->y = f0 * dr.y;
    f->z = f0 * dr.z;
}

static __device__ int colors2pid(int ca, int cb) {
    int c0, c1;
    c0 = ca < cb ? ca : cb;
    c1 = ca < cb ? cb : ca;

    return c1 * (c1+1) / 2 + c0;
}

static __device__ void pair_force(PairDPDC pc, PairPa a, PairPa b, float rnd, /**/ PairFo *f) {
    PairDPD p;
    int pid;
    pid = colors2pid(a.color, b.color);
    p.a = pc.a[pid];
    p.g = pc.g[pid];
    p.s = pc.s[pid];
    pair_force(p, a, b, rnd, /**/ f);
}

/* mirrored: parameters from particle "a" only */
static __device__ void pair_force(PairDPDCM pc, PairPa a, PairPa b, float rnd, /**/ PairFo *f) {
    PairDPD p;
    int pid = a.color;
    p.a = pc.a[pid];
    p.g = pc.g[pid];
    p.s = pc.s[pid];
    pair_force(p, a, b, rnd, /**/ f);
}
