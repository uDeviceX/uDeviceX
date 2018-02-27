struct Particle;
struct Positions;

// tag::interface[]
void positions_float_ini(int n, const float*, /**/ Positions**);
void positions_particle_ini(int n, const Particle*, /**/ Positions**);
void positions_fin(Positions*);
void positions_get(Positions*, int i, /**/ float r[3]);
// end::interface[]
