namespace gsl {
void invert_matrix() {
	double y = gsl_sf_bessel_J0(0.5);
	printf("GSL TEST: %g\n", y);
}
}
