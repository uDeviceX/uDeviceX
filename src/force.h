enum
{
    SOLVENT_TYPE = 0,
    SOLID_TYPE   = 1,
    WALL_TYPE    = 2,
    RBC_TYPE     = 3,
};

/* helper functions for DPD calculations */
__device__ float3 force(int type1, int type2, float3 pos1,
			float3 pos2, float3 vel1,
			float3 vel2, float myrandnr);

__device__ void force0(int typed, int types,
                       float xd, float yd, float zd,
                       float xs, float ys, float zs,
                       float vxd, float vyd, float vzd,
                       float vxs, float vys, float vzs,
                       float rnd, float *fx, float *fy, float *fz);
