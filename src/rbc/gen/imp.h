struct Coords;
struct Particle;
struct Matrices;

// tag::int[]
/* vv : vertices of the mesh template : x y z x y z, ...   */
int rbc_gen(const Coords*, const float *vv, const char *ic, int nv, /**/ Particle*); // <1>
void rbc_gen0(int nv, const float *vv, const Matrices*, /**/ int *pn, Particle*);    // <2>
void rbc_shift(const Coords*, int n, /**/ Particle*); // <3>
// end::int[]
