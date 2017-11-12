namespace rbc { namespace com {
struct Helper {
    float3 *drr;  /* helper to compute centers of mass on device */
    float3 *hrr;  /* centers of mass on host                     */
};

void ini(int maxcells, /**/ Helper *com);
void fin(/**/ Helper *com);
void get_com(int nm, int nv, const Particle *pp, /**/ Helper *com);

}} /* namespace */
