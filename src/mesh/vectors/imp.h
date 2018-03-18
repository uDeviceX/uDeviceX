struct Particle;
struct Vectors;
struct Coords;

// tag::interface[]
void vectors_float_ini(int n, const float*, /**/ Vectors**);
void vectors_postions_ini(int n, const Particle*, /**/ Vectors**);
void vectors_postions_edge  (Coords*, int n, const Particle*, /**/ Vectors**);
void vectors_postions_center(Coords*, int n, const Particle*, /**/ Vectors**);
void vectors_velocities_ini(int n, const Particle*, /**/ Vectors**);
void vectors_zero_ini(int n, /**/ Vectors**);

void vectors_fin(Vectors*);
void vectors_get(Vectors*, int i, /**/ float r[3]);
// end::interface[]
