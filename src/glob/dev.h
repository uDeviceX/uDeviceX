__device__ void domain_center(const Coords co, /**/ float3 *rc) {
    const int *c, *d;
    c = co.c;
    d = co.d;
    
    rc->x = XS * (d[X] - 2.f * c[X] - 1) / 2;
    rc->y = YS * (d[Y] - 2.f * c[Y] - 1) / 2;
    rc->z = ZS * (d[Z] - 2.f * c[Z] - 1) / 2;
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
