struct RbcCom {
    int nv;
    int max_cell;
    float3 *drr, *dvv;  /* positions, velocities on device */
    float3 *hrr, *hvv;  /* positions, velocities on host   */
};
