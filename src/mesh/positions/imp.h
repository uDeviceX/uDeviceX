struct Particle;
struct Positions;

// tag::interface[]
void Positions_float_ini(int n, const float*, /**/ Positions**);
void Positions_particle_ini(int n, const Particle*, /**/ Positions**);
void Positions_fin(Positions*);
void Positions_get(Positions*, int i, /**/ float r[3]);
// end::interface[]
