struct AreaVolume;
struct Particle;
struct int4;

void area_volume_ini(int nt, AreaVolume**);
void area_volume_setup(int nt, int nv, int4 *tri, /**/ AreaVolume*);
void area_volume_fin(AreaVolume*);

void area_volume_compute(int nt, int nv, int nc, const Particle*, const int4 *tri, /**/ float *av);
