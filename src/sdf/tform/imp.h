struct Tform;
struct Coords;
/* T: texture size, N: sdf size, M: wall margin */
void ini_tex2sdf(const Coords, int T[3], int N[3], int M[3],
                 /**/ Tform*);
void ini_sub2tex(/**/ Tform*);
