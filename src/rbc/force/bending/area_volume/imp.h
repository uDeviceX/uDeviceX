struct AreaVolume;
struct Particle;
struct int4;

// tag::interface[]
void area_volume_ini(int nv, int nt, const int4 *tt, int max_cell, AreaVolume**); // <1>
void area_volume_fin(AreaVolume*);

void area_volume_compute(AreaVolume*, int nm, const Particle*, /**/ float **av); // <2>
void area_volume_host(AreaVolume*, /**/ float **av); // <3>
// end::interface[]
