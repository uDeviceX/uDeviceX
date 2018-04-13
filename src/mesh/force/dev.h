#ifdef FORCE_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace force_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace force_dev {
  #define END }
#endif

BEGIN

_S_ double rsqrt(double x) { return 1/sqrt(x); } /* TODO */
_S_ double max(double a, double b) { return a > b ? a : b; }

/* forces from one dihedral */
template <int update>
_S_ double3 dih0(double phi, double kb,
                 double3 r1, double3 r2, double3 r3, double3 r4) {
    double overIksiI, overIdzeI, cosTheta, IsinThetaI2, sinTheta_1,
        beta, b11, b12, sint0kb, cost0kb;
    double3 r12, r13, r34, r24, r41, ksi, dze, ksimdze;
    diff(&r1, &r2, /**/ &r12);
    diff(&r1, &r3, /**/ &r13);
    diff(&r3, &r4, /**/ &r34);
    diff(&r2, &r4, /**/ &r24);
    diff(&r4, &r1, /**/ &r41);

    cross(&r12, &r13, /**/ &ksi);
    cross(&r34, &r24, /**/ &dze);

    overIksiI = rsqrt(dot<double>(&ksi, &ksi));
    overIdzeI = rsqrt(dot<double>(&dze, &dze));

    cosTheta = dot<double>(&ksi, &dze) * overIksiI * overIdzeI;
    IsinThetaI2 = 1.0f - cosTheta * cosTheta;

    diff(&ksi, &dze, /**/ &ksimdze);

    sinTheta_1 = copysignf
        (rsqrt(max(IsinThetaI2, 1.0e-6)),
         dot<double>(&ksimdze, &r41)); // ">" because the normals look inside

    sint0kb = sin(phi) * kb;
    cost0kb = cos(phi) * kb;
    beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

    b11 = -beta *  cosTheta * overIksiI * overIksiI;
    b12 =  beta * overIksiI * overIdzeI;

    if (update == 1) {
        double3 r32, f1, f;
        diff(&r3, &r2, /**/ &r32);
        cross(&ksi, &r32, /**/ &f);
        cross(&dze, &r32, /**/ &f1);
        scal(b11, /**/ &f);
        axpy(b12, &f1, /**/ &f);
        return f;
    }
    else if (update == 2) {
        double3 f, f1, f2, f3;
        double b22 = -beta * cosTheta * overIdzeI * overIdzeI;

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
    else {
        double3 f;
        f.x = f.y = f.z = 0;
        assert(0);
        return f;
    }
}

_I_ double3 dih_a(double phi, double kb,
                                double3 a, double3 b, double3 c, double3 d) {
    return dih0<1>(phi, kb, a, b, c, d);
}

_I_ double3 dih_b(double phi, double kb,
                                double3 a, double3 b, double3 c, double3 d) {
    return dih0<2>(phi, kb, a, b, c, d);
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
