#ifdef TRI_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace tri_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace tri_dev {
  #define END }
#endif

BEGIN

_S_ void diff(const double *a, const double *b, /**/ double  *c) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
_S_ double vabs(double *a) {
    enum {X, Y, Z};
    double r;
    r = a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z];
    return sqrt(r);
}
_S_ void swap(double *a, double *b) {
    double t;
    t = *a; *a = *b; *b = t;
}
_S_ int less(double *a, double *b) { return (*a) < (*b); }
_S_ void  sort3(double *a, double *b, double *c) {
    if (less(c, b)) swap(c, b);
    if (less(b, a)) swap(b, a);
    if (less(c, b)) swap(c, b);
}

_I_ double kahan_area0(double a, double b, double c) {
    sort3(&c, &b, &a); /* make a > b > c */
    return sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))/4;
}

_I_ double kahan_area(const double r0[3], const double r1[3], const double r2[3]) {
    double r01[3], r12[3], r20[3], a, b, c;
    diff(r0, r1, /**/ r01);
    diff(r1, r2, /**/ r12);
    diff(r2, r0, /**/ r20);
    a = vabs(r01); b = vabs(r12); c = vabs(r20);
    return kahan_area0(a, b, c);
}

_S_ double orient2d_xy(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = X, Q = Y};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_S_ double orient2d_zx(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = Z, Q = X};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_S_ double orient2d_yz(const double a[3], const double b[3], const double c[3]) {
    enum {X, Y, Z};
    enum {P = Y, Q = Z};
    double acp, bcp, acq, bcq;
    acp = a[P] - c[P]; bcp = b[P] - c[P];
    acq = a[Q] - c[Q]; bcq = b[Q] - c[Q];
    return acp * bcq - acq * bcp;
}

_I_ void ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]) {
    enum {X, Y, Z};
    r[X] = orient2d_yz(a, b, c);
    r[Y] = orient2d_zx(a, b, c);
    r[Z] = orient2d_xy(a, b, c);
}

_I_ double shewchuk_area(const double a[3], const double b[3], const double c[3]) {
    double x, y, z;
    x = orient2d_yz(a, b, c);
    y = orient2d_zx(a, b, c);
    z = orient2d_xy(a, b, c);
    return sqrt(x*x + y*y + z*z)/2;
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
