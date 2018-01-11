namespace rbc { namespace com {
struct RbcComProps {
    float3 *drr, *dvv;  /* positions, velocities on device */
    float3 *hrr, *hvv;  /* positions, velocities on host   */
};

void rbc_com_ini(int maxcells, /**/ RbcComProps *com);
void rbc_com_fin(/**/ RbcComProps *com);
void rbc_com_get(int nm, int nv, const Particle *pp, /**/ RbcComProps *com);

}} /* namespace */
