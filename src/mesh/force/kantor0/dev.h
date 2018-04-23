#ifdef FORCE_KANTOR0_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace force_kantor0_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace force_kantor0_dev {
  #define END }
#endif

BEGIN

#ifdef FORCE_KANTOR0_HOST
_S_ double rsqrt0(double x) { return pow(x, -0.5); }
#define PRINT(fmt, ...) msg_print((fmt), ##__VA_ARGS__)
#define EXIT() ERR("assert")
#else
_S_ double rsqrt0(double x) { return rsqrt(x); }
#define PRINT(fmt, ...) printf((fmt), ##__VA_ARGS__)
#define EXIT() assert(0)
#endif

_S_ double max(double a, double b) { return a > b ? a : b; }

enum {ANGLE_OK, ANGLE_BAD_A, ANGLE_BAD_B};
_S_ int angle(double3 a, double3 b, /**/ double *pcos, double *pover_sin,
              double *pover_a, double *pover_b) {
    double  cost, over_sin, over_a, over_b;
    double aa, bb, ab;
    *pcos = *pover_sin = *pover_a = *pover_b = 0.0;

    aa = dot<double>(&a, &a);
    bb = dot<double>(&b, &b);
    ab = dot<double>(&a, &b);

    if (aa == 0) return ANGLE_BAD_A;
    if (bb == 0) return ANGLE_BAD_B;

    over_a = rsqrt0(aa);
    over_b = rsqrt0(bb);
    cost = ab * over_a * over_b;
    over_sin = rsqrt0(max(1.0 - cost*cost, 1e-6));

    *pcos = cost; *pover_sin = over_sin; *pover_a = over_a; *pover_b = over_b;
    return ANGLE_OK;
}

_S_ void dih_write(double3 a, double3 b, double3 c, double3 d) {
    PRINT("%g %g %g\n", a.x, a.y, a.z);
    PRINT("%g %g %g\n", b.x, b.y, b.z);
    PRINT("%g %g %g\n", c.x, c.y, c.z);
    PRINT("%g %g %g\n", d.x, d.y, d.z);
}

/* forces from one dihedral */
template <int update>
_S_ double3 dih0(double phi, double kb,
                 double3 a, double3 b, double3 c, double3 d) {
    int status;
    double over_k, over_n, cosTheta, sinTheta_1,
        beta, b11, b12, sint0kb, cost0kb;
    double3 ab, ac, cd, bc, bd, da, k, n, k_n;
    diff(&a, &b, /**/ &ab);
    diff(&a, &c, /**/ &ac);
    diff(&c, &d, /**/ &cd);
    diff(&b, &d, /**/ &bd);
    diff(&b, &c, /**/ &bc);
    diff(&d, &a, /**/ &da);

    cross(&ab, &ac, /**/ &k);
    cross(&bd, &bc, /**/ &n);

    status = angle(k, n, /**/
                   &cosTheta, &sinTheta_1,
                   &over_k, &over_n);
    if (status != ANGLE_OK) {
        PRINT("bad dihedral\n");
        dih_write(a, b, c, d);
        EXIT();
    }
    diff(&k, &n, /**/ &k_n);
    sinTheta_1 = copysign(sinTheta_1, dot<double>(&k_n, &da));

    sint0kb = sin(phi) * kb;
    cost0kb = cos(phi) * kb;
    beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

    b11 = -beta *  cosTheta * over_k * over_k;
    b12 =  beta * over_k * over_n;

    if (update == 1) {
        double3 r32, f1, f;
        diff(&c, &b, /**/ &r32);
        cross(&k, &r32, /**/ &f);
        cross(&n, &r32, /**/ &f1);
        scal(b11, /**/ &f);
        axpy(b12, &f1, /**/ &f);
        return f;
    }
    else if (update == 2) {
        double3 f, f1, f2, f3;
        double b22 = -beta * cosTheta * over_n * over_n;

        cross(&k, &ac, /**/ &f);
        cross(&k, &cd, /**/ &f1);
        cross(&n, &ac, /**/ &f2);
        cross(&n, &cd, /**/ &f3);

        scal(b11, /**/ &f);
        add(&f2, /**/ &f1);
        axpy(b12, &f1, /**/ &f);
        axpy(b22, &f3, /**/ &f);
        return f;
    }
    else {
        double3 f;
        f.x = f.y = f.z = 0;
        assert(0);
        return f;
    }
}

_I_ double3 dih_a(double phi, double kb, double3 a, double3 b, double3 c, double3 d) {
    return dih0<1>(phi, kb, a, b, c, d);
}

_I_ double3 dih_b(double phi, double kb, double3 a, double3 b, double3 c, double3 d) {
    return dih0<2>(phi, kb, a, b, c, d);
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
