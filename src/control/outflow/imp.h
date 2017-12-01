struct Outflow;

void ini(int maxp, /**/ Outflow **o);
void ini_params_circle(float R, Outflow *o);
void ini_params_plane(int dir, float r0, Outflow *o);

void fin(/**/ Outflow *o);

void filter_particles_circle(float R, int n, const Particle *pp, Outflow *o);
void filter_particles_plane(float3 normal, float3 r, int n, const Particle *pp, Outflow *o);

void filter_particles(int n, const Particle *pp, /**/ Outflow *o);

void download_ndead(Outflow *o);

int* get_deathlist(Outflow *o);
int  get_ndead(Outflow *o);
