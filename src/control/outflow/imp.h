struct Outflow;

void ini(int maxp, /**/ Outflow **o);
void ini_params_circle(Coords coords, float3 c, float R, Outflow *o);
void ini_params_plate(Coords c, int dir, float r0, Outflow *o);

void fin(/**/ Outflow *o);

void filter_particles(int n, const Particle *pp, /**/ Outflow *o);

void download_ndead(Outflow *o);

int* get_deathlist(Outflow *o);
int  get_ndead(Outflow *o);
