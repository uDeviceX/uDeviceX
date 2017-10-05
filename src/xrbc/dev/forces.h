namespace dev {

/* forces from one triangle */  
__device__ float3 tri(const float3 r1, const float3 r2, const float3 r3, const float area, const float volume) {
    float Ak, A0, n_2, coefArea, coeffVol,
        r, xx, IbforceI_wcl, kp, IbforceI_pow, ka0, kv0, x0, l0, lmax,
        kbToverp;

    float3 x21, x32, x31, nn, f = make_float3(0, 0, 0);

    diff(&r2, &r1, /**/ &x21);
    diff(&r3, &r2, /**/ &x32);
    diff(&r3, &r1, /**/ &x31);

    cross(&x21, &x31, /**/ &nn); /* normal */

    Ak = 0.5 * sqrtf(dot<float>(&nn, &nn));

    A0 = RBCtotArea / (2.0 * RBCnv - 4.);
    n_2 = 1.0 / Ak;
    ka0 = RBCka / RBCtotArea;
    coefArea = -0.25f * (ka0 * (area - RBCtotArea) * n_2)
        - RBCkd * (Ak - A0) / (4. * A0 * Ak);

    kv0 = RBCkv / (6.0 * RBCtotVolume);
    coeffVol = kv0 * (volume - RBCtotVolume);

    float3 nnx32, r3r2;
    cross(&nn, &x32, /**/ &nnx32);
    axpy(coefArea, &nnx32, /**/ &f); /* area force */
    cross(&r3, &r2, /**/ &r3r2);
    axpy(coeffVol, &r3r2, /**/ &f); /* vol force */

    r = sqrtf(dot<float>(&x21, &x21));
    r = r < 0.0001f ? 0.0001f : r;
    l0 = sqrt(A0 * 4.0 / sqrt(3.0));
    lmax = l0 / RBCx0;
    xx = r / lmax;

    kbToverp = RBCkbT / RBCp;
    IbforceI_wcl =
            kbToverp * (0.25f / ((1.0f - xx) * (1.0f - xx)) - 0.25f + xx) /
            r;

    x0 = RBCx0;
    kp =
            (RBCkbT * x0 * (4 * x0 * x0 - 9 * x0 + 6) * l0 * l0) /
            (4 * RBCp * (x0 - 1) * (x0 - 1));
    IbforceI_pow = -kp / powf(r, RBCmpow) / r;

    axpy(IbforceI_wcl + IbforceI_pow, &x21, /**/ &f); /* wcl and pow forces */
    return f;
}

__device__ float3 visc(float3 r1, float3 r2,
                       float3 u1, float3 u2) {
    const float gammaC = RBCgammaC, gammaT = 3.0 * RBCgammaC;
    float3 du, dr, f = make_float3(0, 0, 0);
    diff(&u2, &u1, /**/ &du);
    diff(&r1, &r2, /**/ &dr);

    const float fac = dot<float>(&du, &dr) / dot<float>(&dr, &dr); 
    
    axpy(gammaT      , &du, /**/ &f);
    axpy(gammaC * fac, &dr, /**/ &f);
    
    return f;
}

/* forces from one dihedral */
template <int update>
__device__ float3 dihedral(float3 r1, float3 r2, float3 r3,
                           float3 r4) {
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

__device__ float area0(const float3 v0, const float3 r1, const float3 r2) {
    float3 x1, x2, n;
    diff(&r1, &v0, /**/ &x1);
    diff(&r2, &v0, /**/ &x2);
    cross(&x1, &x2, /**/ &n);
    return 0.5f * sqrtf(dot<float>(&n, &n));
}
__device__ float volume0(float3 v0, float3 r1, float3 r2) {
    return                                      \
        0.1666666667f *
        ((v0.x*r1.y-v0.y*r1.x)*r2.z +
         (v0.z*r1.x-v0.x*r1.z)*r2.y +
         (v0.y*r1.z-v0.z*r1.y)*r2.x);
}

} // dev
