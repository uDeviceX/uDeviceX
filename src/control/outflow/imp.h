struct Outflow;

struct Config;
struct Coords;
struct Particle;
struct float3;

// tag::mem[]
void outflow_ini(int maxp, /**/ Outflow **o);
void outflow_fin(/**/ Outflow *o);
// end::mem[]

// tag::ini[]
void outflow_set_params_circle(const Coords*, float3 c, float R, Outflow *o);
void outflow_set_params_plate(const Coords*, int dir, float r0, Outflow *o);
// end::ini[]

void outflow_set_cfg(const Config *cfg, const Coords*, Outflow *);

// tag::int[]
void outflow_filter_particles(int n, const Particle *pp, /**/ Outflow *o); // <1>

void outflow_download_ndead(Outflow *o);                                   // <2>

int* outflow_get_deathlist(Outflow *o);                                    // <3>
int  outflow_get_ndead(Outflow *o);                                        // <4>
// tag::int[]

