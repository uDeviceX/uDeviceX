struct Coords;
struct Particle;
struct Matrices;

// tag::int[]
/* vv : vertices of the mesh template : x y z x y z, ...   */
int rbc_gen_from_file(const Coords*, const float *vv, const char *ic, int nv, /**/ Particle*); // <1>
void rbc_gen_from_matrices(int nv, const float *vv, const Matrices*, /**/ int *pn, Particle*);    // <2>
void rbc_shift(const Coords*, int n, /**/ Particle*); // <3>
// end::int[]
