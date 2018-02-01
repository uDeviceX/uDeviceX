struct RbcComProps {
    int nv;
    int max_cell;
    float3 *drr, *dvv;  /* positions, velocities on device */
    float3 *hrr, *hvv;  /* positions, velocities on host   */
};

void rbc_com_ini(int nv, int maxcells, /**/ RbcComProps *com);
void rbc_com_fin(/**/ RbcComProps *com);
void rbc_com_get(int nm, const Particle *pp, /**/ RbcComProps *com);
