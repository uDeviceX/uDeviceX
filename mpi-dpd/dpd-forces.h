#define  OUT_TYPE 0
#define   IN_TYPE 1
#define MEMB_TYPE 2
#define WALL_TYPE 3

/* helper functions for DPD calculations */
__device__ float3 compute_dpd_force_traced(int type1, int type2, float3 pos1,
                                           float3 pos2, float3 vel1,
                                           float3 vel2, float myrandnr);
