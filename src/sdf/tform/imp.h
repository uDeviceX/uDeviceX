struct Tform;
struct Coords;
/* T: texture size, N: sdf size, M: wall margin */
void ini_tex2sdf(const Coords, const int T[3], const int N[3], const int M[3],
                 /**/ Tform*);
void ini_sub2tex(/**/ Tform*);
