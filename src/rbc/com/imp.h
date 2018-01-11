namespace rbc { namespace com {
struct ComProps {
    float3 *drr;  /* helper to compute centers of mass on device */
    float3 *hrr;  /* centers of mass on host                     */
};

void ini(int maxcells, /**/ ComProps *com);
void fin(/**/ ComProps *com);
void get(int nm, int nv, const Particle *pp, /**/ ComProps *com);

}} /* namespace */
