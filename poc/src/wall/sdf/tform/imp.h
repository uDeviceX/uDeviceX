struct Tform;
struct Coords;

// tag::int[]
/* T: texture size, N: sdf file size, M: wall margin */
void tex2sdf_ini(const Coords*, const int T[3], const int N[3], const int M[3], /**/ Tform*); // <1>
void out2sdf_ini(const Coords*, const int N[3], /**/ Tform*); // <2>
void sub2tex_ini(const Coords*, const int T[3], const int M[3], /**/ Tform*); // <3>
// end::int[]

