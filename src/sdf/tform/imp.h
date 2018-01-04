struct Tform;
struct Coords;

/* T: texture size, N: sdf file size, M: wall margin */
void tex2sdf_ini(const Coords*, int T[3], int N[3], int M[3], /**/ Tform*);
void ini_sub2tex(/**/ Tform*);
