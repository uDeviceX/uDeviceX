struct AreaVolume;
struct Particle;
struct int4;

void area_volume_ini(int nv, int nt, const int4 *tt, int max_cell, AreaVolume**);
void area_volume_fin(AreaVolume*);

/* TODO: remove */
const int4* area_volume_tri(AreaVolume*);

void area_volume_compute(AreaVolume*, int nc, const Particle*, /**/ float **av);
void area_volume_host(AreaVolume*, /**/ float **av);
