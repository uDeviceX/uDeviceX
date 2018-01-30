struct AreaVolume;
struct Particle;
struct int4;

void area_volume_ini(int nv, int nt, const int4 *tt, AreaVolume**);
void area_volume_setup(int nt, int nv, const int4 *tri, /**/ AreaVolume*);
void area_volume_fin(AreaVolume*);

int4* area_volume_tri(AreaVolume*);

void area_volume_compute(AreaVolume*, int nc, const Particle*, /**/ float *av);
