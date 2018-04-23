struct Coords;
struct Particle;
struct Matrices;

// tag::int[]
/* vv : vertices of the mesh template : x y z x y z, ...   */
int  mesh_gen_from_file(const Coords*, const float *vv, const char *ic, int nv, /**/ Particle*); // <1>
void mesh_gen_from_matrices(int nv, const float *vv, const Matrices*, /**/ int *pn, Particle*);  // <2>
void mesh_shift(const Coords*, int n, /**/ Particle*); // <3>
// end::int[]
