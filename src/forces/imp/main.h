struct DPDparam { float gamma, a, rnd; };
enum {LJ_NONE, LJ_ONE, LJ_TWO}; /* lj hack */
static __device__ float wrf(const int s, float x) {
    if (s == 0) return x;
    if (s == 1) return sqrtf(x);
    if (s == 2) return sqrtf(sqrtf(x));
    if (s == 3) return sqrtf(sqrtf(sqrtf(x)));
    return powf(x, 1.f/s);
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

static __device__ void force0(float x, float y, float z,
                              float vx, float vy, float vz,
                              DPDparam p, int ljkind, /**/ Fo *f) {
    float invr, r;
    float wc, wr; /* conservative and random kernels */
    float rm; /* 1 minus r */
    float ev; /* (e dot v) */
    float gamma, sigma;
    float f0;
    float a;
    int vnstat; /* vector normalization status */

    vnstat = norm(/*io*/ &x, &y, &z, /*o*/ &r, &invr);
    if (vnstat == NORM_BIG) {
        f->x = f->y = f->z = 0;
        return;
    }

    rm = max(1 - r, 0.0f);
    wc = rm;
    wr = wrf(S_LEVEL, rm);
    ev = x*vx + y*vy + z*vz;

    gamma = p.gamma;
    a     = p.a;
    sigma = sqrtf(2*gamma*kBT / dt);
    f0  = (-gamma * wr * ev + sigma * p.rnd) * wr;
    f0 +=                                  a * wc;

    if (vnstat == NORM_OK && ljkind != LJ_NONE) {
        const float ljsi = LJ_ONE ? ljsigma : 2 * ljsigma;
        f0 += lj(invr, ljsi);
    }

    f->x = f0*x;
    f->y = f0*y;
    f->z = f0*z;
}

static __device__ void force1(Pa *A, Pa *B, DPDparam p, int ljkind, /**/ Fo *f) {
    force0(A->x -  B->x,    A->y -  B->y,  A->z -  B->z,
           A->vx - B->vx,   A->vy - B->vy, A->vz - B->vz,
           p, ljkind,
           f);
}

static __device__ void fill_g(float *gg) {
    enum {B = BLUE_COLOR, R = RED_COLOR, S = SOLID_COLOR, W = WALL_COLOR};
  #define g0(x, y) gg[(x) * N_COLOR + (y)]
    g0(B,B) = gdpd_b; g0(R,R) = gdpd_r; g0(S,S) = gdpd_b;

    g0(B,R) = g0(R,B) = gdpd_br;
    g0(B,S) = g0(S,B) = gdpd_bs;
    g0(B,W) = g0(W,B) = gdpd_bw;

    g0(R,S) = g0(S,R) = gdpd_rs;
    g0(R,W) = g0(W,R) = gdpd_rw;

    g0(S,W) = g0(W,S) = gdpd_sw;
  #undef g0
}

static __device__ void fill_a(float *aa) {
    enum {B = BLUE_COLOR, R = RED_COLOR, S = SOLID_COLOR, W = WALL_COLOR};
  #define a0(x, y) aa[(x) * N_COLOR + (y)]
    a0(B,B) = adpd_b; a0(R,R) = adpd_r; a0(S,S) = adpd_b;

    a0(B,R) = a0(R,B) = adpd_br;
    a0(B,S) = a0(S,B) = adpd_bs;
    a0(B,W) = a0(W,B) = adpd_bw;

    a0(R,S) = a0(S,R) = adpd_rs;
    a0(R,W) = a0(W,R) = adpd_rw;

    a0(S,W) = a0(W,S) = adpd_sw;
  #undef a0
}

static __device__ void color2par(int ca, int cb, /**/ DPDparam *p) {
  #define a0(x, y) aa[(x) * N_COLOR + (y)]
  #define g0(x, y) gg[(x) * N_COLOR + (y)]    
    enum {B = BLUE_COLOR};
    float gg[N_COLOR*N_COLOR], aa[N_COLOR*N_COLOR];
    fill_g(/**/ gg); fill_a(/**/ aa);
    if         (!multi_solvent) {
        p->gamma = g0(B,B);
        p->a     = g0(B,B);
    } else {
        p->gamma = g0(ca,cb);
        p->a     = a0(ca,cb);
    }
  #undef g0
  #undef a0
}

static __device__ void force2(Pa *A, Pa *B, int ca, int cb, int ljkind, float rnd,
                              /**/ Fo *f) {
    DPDparam p;
    color2par(ca, cb, &p); p.rnd = rnd;
    force1(A, B, p, ljkind, /**/ f);
}

static __device__ int kind2color(int k, int c) {
    enum {O = SOLVENT_KIND, S = SOLID_KIND, W = WALL_KIND};
    if (k == SOLID_KIND) c = SOLID_COLOR;
    if (k == WALL_KIND ) c = WALL_COLOR;
    return c;
}

static __device__ void force(Pa A, Pa B, float rnd, /**/ Fo *f) {
    enum {O = SOLVENT_KIND, S = SOLID_KIND, W = WALL_KIND};
    int ljkind;
    int ka, kb;
    int ca, cb;
    ka = A.kind; kb = B.kind;
    ca = A.color; cb = B.color;
    ljkind = LJ_NONE;
    if (ka == S && kb == S) ljkind = LJ_TWO;
    ca = kind2color(ka, ca);
    cb = kind2color(kb, cb);
    force2(&A, &B, ca, cb, ljkind, rnd, /**/ f);
}
