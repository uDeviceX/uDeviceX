#include <stdio.h>

void inv3x3(float *m, /**/ float *r) {
  enum {XX, XY, XZ, YY, YZ, ZZ};

  double xx, yy, zz, xy, xz, yz;
  double yz2, xz2, xy2;
  double mx, my, mz; /* minors */
  double d, i; /* determinant and its inverse */

  xx = m[XX]; yy = m[YY]; zz = m[ZZ];
  xy = m[XY]; xz = m[XZ];
  yz = m[YZ];
  yz2 = yz*yz; xz2 = xz*xz; xy2 = xy*xy;

  mx  = yy*zz-yz2;
  my  = xy*zz-xz*yz;
  mz  = xy*yz-xz*yy;
  d   = mz*xz-my*xy+mx*xx;
  i   = 1/d;

  r[XX] =  mx*i;
  r[XY] = -my*i;
  r[XZ] =  mz*i;
  r[YY] = i*(xx*zz-xz2);
  r[YZ] = i*(xy*xz-xx*yz);
  r[ZZ] = i*(xx*yy-xy2);
}

void dump(float *m) {
  enum {XX, XY, XZ, YY, YZ, ZZ};
  enum {YX = XY, ZX = XZ, ZY = YZ};
  int d[] = {XX, XY, XZ,
	     YX, YY, YZ,
	     ZX, ZY, ZZ};
  int i, j, k;
  k = 0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      printf("%8.1g%s", m[d[k++]], " ");
    }
    puts("");
  }
}

int main() {
  float r[6];
  float m[6] = {
    1, 2, 3,
       2, 3,
	  4
  };
  puts("m:"); dump(m);
  inv3x3(m, /**/ r);
  puts("r:"); dump(r);
}
