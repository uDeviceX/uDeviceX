static __device__ float3 fvolume(float3 r2, float3 r3, float v) {
    float kv, v0, f0;
    float3 f, n;
    kv = RBCkv; v0 = RBCtotVolume;

    cross(&r3, &r2, /**/ &n);
    f0 = kv * (v - v0) / (6 * v0);
    axpy(f0, &n, /**/ &f);
    return f;
}

static __device__ float3 farea(float3 x21, float3 x31, float3 x32,   float a0, float A) {
    float3 nn, nnx32, f;
    float A0, a, f0, fa, fA, ka, kA;
    A0 = RBCtotArea; ka = RBCkd; kA = RBCka;

    cross(&x21, &x31, /**/ &nn); /* normal */
    cross(&nn, &x32, /**/ &nnx32);
    a = 0.5 * sqrtf(dot<float>(&nn, &nn));

    fA = - kA * (A - A0) / (4 * A0 * a);
    fa = - ka * (a - a0) / (4 * a0 * a);
    f0 = fA + fa;
    axpy(f0, &nnx32, /**/ &f);
    return f;
}

static __device__ float3 fspring(float3 x21, float x0, float A0) {
    float  r, xx, IbforceI_wcl, kp, IbforceI_pow, l0, lmax, kbToverp;
    float3 f;
    
    r = sqrtf(dot<float>(&x21, &x21));
    l0 = sqrt(A0 * 4.0 / sqrt(3.0));
    lmax = l0 / x0;
    xx = r / lmax;

    kbToverp = RBCkbT / RBCp;
    IbforceI_wcl =
            kbToverp * (0.25f / ((1.0f - xx) * (1.0f - xx)) - 0.25f + xx) /
            r;

    kp =
            (RBCkbT * x0 * (4 * x0 * x0 - 9 * x0 + 6) * l0 * l0) /
            (4 * RBCp * (x0 - 1) * (x0 - 1));
    IbforceI_pow = -kp / powf(r, RBCmpow) / r;
    axpy(IbforceI_wcl + IbforceI_pow, &x21, /**/ &f); /* wcl and pow forces */
    return f;
}

static __device__ float3 tri0(float3 r1, float3 r2, float3 r3,
                              float x0, float A0,
                              float area, float volume) {
    float3 fv, fa, fs;
    float3 x21, x32, x31, f = make_float3(0, 0, 0);

    diff(&r2, &r1, /**/ &x21);
    diff(&r3, &r2, /**/ &x32);
    diff(&r3, &r1, /**/ &x31);

    fa = farea(x21, x31, x32,   A0, area);
    add(&fa, /**/ &f);

    fv = fvolume(r2, r3, volume);
    add(&fv, /**/ &f);

    fs = fspring(x21,  x0, A0);
    add(&fs, /**/ &f);

    return f;
}

static __device__ float3 visc(float3 r1, float3 r2, float3 u1, float3 u2) {
    const float gammaC = RBCgammaC, gammaT = RBCgammaT;
    float3 du, dr, f = make_float3(0, 0, 0);
    diff(&u2, &u1, /**/ &du);
    diff(&r1, &r2, /**/ &dr);

    const float fac = dot<float>(&du, &dr) / dot<float>(&dr, &dr);

    axpy(gammaT      , &du, /**/ &f);
    axpy(gammaC * fac, &dr, /**/ &f);

    return f;
}

/* forces from one dihedral */
template <int update> __device__ float3 dih0(float3 r1, float3 r2, float3 r3, float3 r4) {
    float overIksiI, overIdzeI, cosTheta, IsinThetaI2, sinTheta_1,
        beta, b11, b12, phi, sint0kb, cost0kb;

    float3 r12, r13, r34, r24, r41, ksi, dze, ksimdze;
    diff(&r1, &r2, /**/ &r12);
    diff(&r1, &r3, /**/ &r13);
    diff(&r3, &r4, /**/ &r34);
    diff(&r2, &r4, /**/ &r24);
    diff(&r4, &r1, /**/ &r41);

    cross(&r12, &r13, /**/ &ksi);
    cross(&r34, &r24, /**/ &dze);

    overIksiI = rsqrtf(dot<float>(&ksi, &ksi));
    overIdzeI = rsqrtf(dot<float>(&dze, &dze));

    cosTheta = dot<float>(&ksi, &dze) * overIksiI * overIdzeI;
    IsinThetaI2 = 1.0f - cosTheta * cosTheta;

    diff(&ksi, &dze, /**/ &ksimdze);

    sinTheta_1 = copysignf
        (rsqrtf(max(IsinThetaI2, 1.0e-6f)),
         dot<float>(&ksimdze, &r41)); // ">" because the normals look inside

    phi = RBCphi / 180.0 * M_PI;
    sint0kb = sin(phi) * RBCkb;
    cost0kb = cos(phi) * RBCkb;
    beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

    b11 = -beta *  cosTheta * overIksiI * overIksiI;
    b12 =  beta * overIksiI * overIdzeI;

    if (update == 1) {
        float3 r32, f1, f;
        diff(&r3, &r2, /**/ &r32);
        cross(&ksi, &r32, /**/ &f);
        cross(&dze, &r32, /**/ &f1);
        scal(b11, /**/ &f);
        axpy(b12, &f1, /**/ &f);
        return f;
    }
    else if (update == 2) {
        float3 f, f1, f2, f3;
        float b22 = -beta * cosTheta * overIdzeI * overIdzeI;

        cross(&ksi, &r13, /**/ &f);
        cross(&ksi, &r34, /**/ &f1);
        cross(&dze, &r13, /**/ &f2);
        cross(&dze, &r34, /**/ &f3);

        scal(b11, /**/ &f);
        add(&f2, /**/ &f1);
        axpy(b12, &f1, /**/ &f);
        axpy(b22, &f3, /**/ &f);

        return f;
    }
    else
        return make_float3(0, 0, 0);
}
