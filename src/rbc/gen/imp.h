struct Coords;
struct Particle;
struct Matrices;

/* vv : vertices of a template : x y z x y z, ...   */
int rbc_gen(const Coords*, const float *vv, const char *ic, int nv, /**/ Particle*);

void rbc_gen0(int nv, const float *vv, const Matrices*, /**/ int *pn, Particle*);
void rbc_shift(const Coords*, int n, /**/ Particle*);
