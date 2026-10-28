#include "macros.cuh"
void error_signal(const char *file, int line, const char *fmt, ...);
void error_report();
void error_abort();


void linal_inv3x3(const float *m,      float *r) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    double eps = 1e-8;

    double xx, yy, zz, xy, xz, yz;
    double yz2, xz2, xy2;
    double mx, my, mz;
    double d, i;

    xx = m[XX]; yy = m[YY]; zz = m[ZZ];
    xy = m[XY]; xz = m[XZ];
    yz = m[YZ];
    yz2 = yz*yz; xz2 = xz*xz; xy2 = xy*xy;

    mx  = yy*zz-yz2;
    my  = xy*zz-xz*yz;
    mz  = xy*yz-xz*yy;
    d   = mz*xz-my*xy+mx*xx;
    if (d < eps && d > -eps)
        ERR("Matrix is singular, determinant %g", d);
    i   = 1/d;

    r[XX] =  mx*i;
    r[XY] = -my*i;
    r[XZ] =  mz*i;
    r[YY] = i*(xx*zz-xz2);
    r[YZ] = i*(xy*xz-xx*yz);
    r[ZZ] = i*(xx*yy-xy2);
}
