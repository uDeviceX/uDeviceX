static __device__ float xl2xc(const Coords c, float xl) {
    return xl + XS * (c.xd - 2.f * c.xc - 1) / 2;
}

static __device__ float yl2yc(const Coords c, float yl) {
    return yl + YS * (c.yd - 2.f * c.yc - 1) / 2;
}

static __device__ float zl2zc(const Coords c, float zl) {
    return zl + ZS * (c.zd - 2.f * c.zc - 1) / 2;
}

static __device__ void local2center(const Coords c, float3 rl, /**/ float3 *rc) {
    rc->x = xl2xc(c, rl.x);
    rc->y = yl2yc(c, rl.y);
    rc->z = zl2zc(c, rl.z);
}

static __device__ void center2local(const Coords c, float3 rc, /**/ float3 *rl) {
    enum {X, Y, Z};
    rl->x = rc.x - XS * (c.xd - 2.f * c.xc - 1) / 2;
    rl->y = rc.y - YS * (c.yd - 2.f * c.yc - 1) / 2;
    rl->z = rc.z - ZS * (c.zd - 2.f * c.zc - 1) / 2;
}

static __device__ void local2global(const Coords c, float3 rl, /**/ float3 *rg) {
    enum {X, Y, Z};
    rg->x = (c.xc + 0.5f) * XS + rl.x;
    rg->y = (c.yc + 0.5f) * YS + rl.y;
    rg->z = (c.zc + 0.5f) * ZS + rl.z;    
}

static __device__ void global2local(const Coords c, float3 rg, /**/ float3 *rl) {
    enum {X, Y, Z};
    rl->x = rg.x - (c.xc + 0.5f) * XS;
    rl->y = rg.y - (c.yc + 0.5f) * YS;
    rl->z = rg.z - (c.zc + 0.5f) * ZS; 
}
