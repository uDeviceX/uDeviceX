struct Particle;
struct Positions;

void Positions_float_ini(int n, float*, /**/ Positions**);
void Positions_particle_ini(int n, Particle*, /**/ Positions**);
void Positions_fin(Positions*);

void Positions_get(Positions*, int i, /**/ float r[3]);
