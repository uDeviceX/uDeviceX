struct Particle;
struct Vectors;

// tag::interface[]
void positions_float_ini(int n, const float*, /**/ Vectors**);
void positions_particle_ini(int n, const Particle*, /**/ Vectors**);
void positions_fin(Vectors*);
void positions_get(Vectors*, int i, /**/ float r[3]);
// end::interface[]
