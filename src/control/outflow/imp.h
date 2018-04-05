struct Outflow;

// tag::mem[]
void outflow_ini(int maxp, /**/ Outflow **o);
void outflow_fin(/**/ Outflow *o);
// end::mem[]

// tag::ini[]
void outflow_ini_params_circle(const Coords*, float3 c, float R, Outflow *o);
void outflow_ini_params_plate(const Coords*, int dir, float r0, Outflow *o);
// end::ini[]

// tag::int[]
void outflow_filter_particles(int n, const Particle *pp, /**/ Outflow *o); // <1>

void outflow_download_ndead(Outflow *o);                                   // <2>

int* outflow_get_deathlist(Outflow *o);                                    // <3>
int  outflow_get_ndead(Outflow *o);                                        // <4>
// tag::int[]

