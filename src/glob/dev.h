__device__ void local2center(const Coords c, float3 rl, /**/ float3 *rc) {
    rc->x = rl.x + XS * (c.d[X] - 2.f * c.c[X] - 1) / 2;
    rc->y = rl.y + YS * (c.d[Y] - 2.f * c.c[Y] - 1) / 2;
    rc->z = rl.z + ZS * (c.d[Z] - 2.f * c.c[Z] - 1) / 2;
}

__device__ void center2local(const Coords c, float3 rc, /**/ float3 *rl) {
    rl->x = rc.x - XS * (c.d[X] - 2.f * c.c[X] - 1) / 2;
    rl->y = rc.y - YS * (c.d[Y] - 2.f * c.c[Y] - 1) / 2;
    rl->z = rc.z - ZS * (c.d[Z] - 2.f * c.c[Z] - 1) / 2;
}

__device__ void local2global(const Coords c, float3 rl, /**/ float3 *rg) {
    rg->x = (c.c[X] + 0.5f) * XS + rl.x;
    rg->y = (c.c[Y] + 0.5f) * YS + rl.y;
    rg->z = (c.c[Z] + 0.5f) * ZS + rl.z;    
}

__device__ void global2local(const Coords c, float3 rg, /**/ float3 *rl) {
    rl->x = rg.x - (c.c[X] + 0.5f) * XS;
    rl->y = rg.y - (c.c[Y] + 0.5f) * YS;
    rl->z = rg.z - (c.c[Z] + 0.5f) * ZS; 
}
