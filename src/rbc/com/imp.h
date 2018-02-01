struct RbcCom {
    int nv;
    int max_cell;
    float3 *drr, *dvv;  /* positions, velocities on device */
    float3 *hrr, *hvv;  /* positions, velocities on host   */
};

void rbc_com_ini(int nv, int max_cell, /**/ RbcCom *com);
void rbc_com_fin(/**/ RbcCom*);
void rbc_com_compute(RbcCom*, int nm, const Particle*, /**/ float3 **rr, float3 **vv);
