namespace forces {
struct DPDparam { float gamma, a, rnd; };

enum {LJ_NONE, LJ_ONE, LJ_TWO};

static __device__ float wrf(const int s, float x) {
    if (s == 0) return x;
    if (s == 1) return sqrtf(x);
    if (s == 2) return sqrtf(sqrtf(x));
    if (s == 3) return sqrtf(sqrtf(sqrtf(x)));
    return powf(x, 1.f/s);
}

static __device__ float cap(float x, float lo, float hi) {
    if      (x > hi) return hi;
    else if (x < lo) return lo;
    else             return x;
}

static const float EPS = 1e-6;
enum {OK, BIG, SMALL};
static __device__ bool norm(/*io*/ float *px, float *py, float *pz, /**/ float *pr, float *pinvr) {
    /* noralize vector r = [x, y, z], sets |r| and 1/|r| if not big */
    float x, y, z, invr, r;
    float r2;
    x = *px; y = *py; z = *pz;

    r2 = x*x + y*y + z*z;
    if      (r2 >= 1 )   return BIG;
    else if (r2 < EPS) {
        *pr = *px = *py = *pz = 0;
        return SMALL;
    } else {
        invr = rsqrtf(r2);
        r = r2 * invr;
        x *= invr; y *= invr; z *= invr;
        *px = x; *py = y; *pz = z; *pr = r; *pinvr = invr;
        return OK;
    }
}

static __device__ float lj(float invr, float ljsi) {
    float t2, t4, t6, f;
    t2 = ljsi * ljsi * invr * invr;
    t4 = t2 * t2;
    t6 = t4 * t2;
    f = ljepsilon * 24 * invr * t6 * (2 * t6 - 1);
    f = cap(f, 0, 1e4);
    return f;
}

static __device__ void dpd00(int typed, int types,
                             float x, float y, float z,
                             float vx, float vy, float vz,
                             float rnd, float *fx, float *fy, float *fz) {
    /* [tab]les */
    const float gamma_tbl[] = {gammadpd_solv, gammadpd_solid, gammadpd_wall, gammadpd_rbc};
    const float a_tbl[] = {aij_solv, aij_solid, aij_wall, aij_rbc};

    float invr, r;
    float wc, wr; /* conservative and random kernels */
    float rm; /* 1 minus r */
    float ev; /* (e dot v) */
    float gamma, sigma;
    float f;
    float a;
    int vnstat; /* vector normalization status */

    vnstat = norm(/*io*/ &x, &y, &z, /*o*/ &r, &invr);
    if (vnstat == BIG) {
        *fx = *fy = *fz = 0;
        return;
    }

    rm = max(1 - r, 0.0f);
    wc = rm;
    wr = wrf(-S_LEVEL, rm);
    ev = x*vx + y*vy + z*vz;

    gamma = 0.5f * (gamma_tbl[typed] + gamma_tbl[types]);
    a     = 0.5f * (a_tbl[typed] + a_tbl[types]);
    sigma = sqrtf(2*gamma*kBT / dt);
    f  = (-gamma * wr * ev + sigma * rnd) * wr;
    f +=                                a * wc;

    bool ss = (typed == SOLID_KIND) && (types == SOLID_KIND);
    bool sw = (typed == SOLID_KIND) && (types ==  WALL_KIND);
    if (vnstat == OK && (ss || sw) ) {
        /*hack*/ const float ljsi = ss ? ljsigma : 2 * ljsigma;
        f += lj(invr, ljsi);
    }

    *fx = f*x; *fy = f*y; *fz = f*z;
}

static __device__ void dpd0(int typed, int types,
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

static __device__ void gen0(Pa *A, Pa *B, DPDparam p, int ljkind, /**/ float *fx, float *fy, float *fz) {
    dpd0(A->kind, B->kind,
         A->x,  A->y,  A->z,  B->x,  B->y,  B->z,
         A->vx, A->vy, A->vz, B->vx, B->vy, B->vz,
         p.rnd,
         fx, fy, fz);
}

static __device__ void gen1(Pa *A, Pa *B, DPDparam p, int ljkind, /**/ float *fx, float *fy, float *fz) {
    gen0(A, B, p, ljkind, /**/ fx, fy, fz);
}

static __device__ void gen2(Pa *A, Pa *B, int ca, int cb, int ljkind, float rnd,
                            /**/ float *fx, float *fy, float *fz) {
    DPDparam p;
    if (ca == RED_COLOR || cb == RED_COLOR) {
        p.gamma = gammadpd_solv;
        p.a     = aij_solv;
    } else {
        p.gamma = gammadpd_rbc;
        p.a     = aij_rbc;
    }
    p.rnd = rnd;
    gen1(A, B, p, ljkind, /**/ fx, fy, fz);
}

static __device__ void gen3(Pa *A, Pa *B, float rnd, /**/ float *fx, float *fy, float *fz) {
    int ljkind; /* call LJ? */
    int ka, kb;
    int ca, cb; /* corrected colors */
    int O, S, W;
    O = SOLVENT_KIND; S = SOLID_KIND; W = WALL_KIND;
    ka = A->kind; kb = B->kind;
    ca = A->color; cb = B->color;
    ljkind = LJ_NONE;

    if        (ka == O && kb == O) { /* OO */
        /* no correction */
    } else if (ka == O && kb == S) { /* OS */
        cb = ca;
    } else if (ka == 0 && kb == W) { /* OW */
        cb = ca;
    } else if (ka == S && kb == S) { /* SS */
        ca = cb = RED_COLOR;   ljkind = LJ_TWO;
    } else if (ka == S && kb == W) { /* SW */
        ca = cb = RED_COLOR;   ljkind = LJ_ONE;
    } else {
        //        assert(0);
    }
    gen2(A, B, ca, cb, ljkind, rnd, /**/ fx, fy, fz);
}

static __device__ void gen(Pa A, Pa B, float rnd, /**/ float *fx, float *fy, float *fz) {
    /* force A.kind <= B.kind */
    bool Flip;
    Pa *pA, *pB;
    if (A.kind > B.kind) {
        Flip = true;
        pA = &B; pB = &A;
    } else {
        Flip = false;
        pA = &A; pB = &B;
    }
    gen3(pA, pB, rnd, /**/ fx, fy, fz);
    if (Flip) {
        *fx = -(*fx);
        *fy = -(*fy);
        *fz = -(*fz);
    }
}
} /* namespace */
