struct Tform;
struct Coords;

/* T: texture size, N: sdf file size, M: wall margin */
void tex2sdf_ini(const Coords*, const int T[3], const int N[3], const int M[3], /**/ Tform*);
void sub2sdf_ini(const Coords*, const int N[3], /**/ Tform*);
void sub2tex_ini(/**/ Tform*);
