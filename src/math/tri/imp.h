namespace tri_hst {
// tag::interface[]
double kahan_area0(double a, double b, double c); // <1>
double kahan_area(const double a[3], const double b[3], const double c[3]); // <2>
void ac_bc_cross(const double a[3], const double b[3], const double c[3], /**/ double r[3]); // <3>
// end::interface[]
}
