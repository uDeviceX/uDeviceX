

/* helper functions for DPD MSD calculations */
class Particle;

__device__ float3 compute_dpd_force_traced(int type1, int type2,
                                           float3 pos1, float3 pos2, float3 vel1, float3 vel2, float myrandnr);
