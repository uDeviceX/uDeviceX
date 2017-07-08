namespace l { namespace gsl {
enum {XX, XY, XZ, YY, YZ, ZZ};
enum {YX = XY, ZX = XZ, ZY = YZ};
#define DIM 3

static void inv3x3_s1(gsl_matrix *m, gsl_permutation *p, gsl_matrix *minv) {
	int s;
	gsl_linalg_LU_decomp(m, p, &s);
	gsl_linalg_LU_invert(m, p, minv);
}

static void inv3x3_s0(double *m0, double *minv0) {
	gsl_permutation *p = gsl_permutation_alloc(DIM);
	gsl_matrix_view m = gsl_matrix_view_array(m0, DIM, DIM);
	gsl_matrix_view minv = gsl_matrix_view_array(minv0, DIM, DIM);

	inv3x3_s1(&(m.matrix), p, &(minv.matrix));

	gsl_permutation_free(p);
}

/* inverts symmetric matrix m[6] (see poc/3x3) */
void inv3x32(float *m, /**/ float *r) {
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

void inv3x3(float *m0, float *minv0) {
	double m[DIM*DIM], minv[DIM*DIM];
	int i;

	i = 0;
	m[i++] = m0[XX]; m[i++] = m0[XY]; m[i++] = m0[XZ];
	m[i++] = m0[YX]; m[i++] = m0[YY]; m[i++] = m0[YZ];
	m[i++] = m0[ZX]; m[i++] = m0[ZY]; m[i++] = m0[ZZ];

	inv3x3_s0(m, minv);

	i = 0;
	minv0[XX] = minv[i++]; minv0[XY] = minv[i++]; minv0[XZ] = minv[i++];
	minv0[YX] = minv[i++]; minv0[YY] = minv[i++]; minv0[YZ] = minv[i++];
	minv0[ZX] = minv[i++]; minv0[ZY] = minv[i++]; minv0[ZZ] = minv[i++];
}	
}}
