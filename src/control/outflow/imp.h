struct Outflow;

// tag::mem[]
void ini(int maxp, /**/ Outflow **o);
void fin(/**/ Outflow *o);
// end::mem[]

// tag::ini[]
void ini_params_circle(const Coords *coords, float3 c, float R, Outflow *o);
void ini_params_plate(const Coords *c, int dir, float r0, Outflow *o);
// end::ini[]

// tag::int[]
void filter_particles(int n, const Particle *pp, /**/ Outflow *o); // <1>

void download_ndead(Outflow *o);                                   // <2>

int* get_deathlist(Outflow *o);                                    // <3>
int  get_ndead(Outflow *o);                                        // <4>
// tag::int[]

