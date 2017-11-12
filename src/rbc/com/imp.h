namespace rbc { namespace com {
struct ComHelper {
    float3 *drr;  /* helper to compute centers of mass on device */
    float3 *hrr;  /* centers of mass on host                     */
};

void ini(int maxcells, /**/ ComHelper *com);
void fin(/**/ ComHelper *com);
void get_com(int nm, int nv, const Particle *pp, /**/ ComHelper *com);

}} /* namespace */
