namespace rbc { namespace com {
struct ComProps {
    float3 *drr, *dvv;  /* positions, velocities on device */
    float3 *hrr, *hvv;  /* positions, velocities on host   */
};

void ini(int maxcells, /**/ ComProps *com);
void fin(/**/ ComProps *com);
void get(int nm, int nv, const Particle *pp, /**/ ComProps *com);

}} /* namespace */
