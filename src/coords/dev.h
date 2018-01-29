/* domain sizes */

static __device__ int xdomain(const Coords_v c) {
    return c.Lx * c.xd;
}

static __device__ int ydomain(const Coords_v c) {
    return c.Ly * c.yd;
}

static __device__ int zdomain(const Coords_v c) {
    return c.Lz * c.zd;
}

/* [l]ocal to [c]enter */

static __device__ float xl2xc(const Coords_v c, float xl) {
    return xl - 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
}

static __device__ float yl2yc(const Coords_v c, float yl) {
    return yl - 0.5f * c.Ly * (c.yd - 2 * c.yc - 1);
}

static __device__ float zl2zc(const Coords_v c, float zl) {
    return zl - 0.5f * c.Lz * (c.zd - 2 * c.zc - 1);
}

static __device__ void local2center(const Coords_v c, float3 rl, /**/ float3 *rc) {
    rc->x = xl2xc(c, rl.x);
    rc->y = yl2yc(c, rl.y);
    rc->z = zl2zc(c, rl.z);
}

/* [c]enter to [l]ocal  */

static __device__ float xc2xl(const Coords_v c, float xc) {
    return xc + 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
}

static __device__ float yc2yl(const Coords_v c, float yc) {
    return yc + 0.5f * c.Ly * (c.yd - 2.f * c.yc - 1);
}

static __device__ float zc2zl(const Coords_v c, float zc) {
    return zc + 0.5f * c.Lz * (c.zd - 2.f * c.zc - 1);
}

static __device__ void center2local(Coords_v c, float3 rc, /**/ float3 *rl) {
    rl->x = xc2xl(c, rc.x);
    rl->y = yc2yl(c, rc.y);
    rl->z = zc2zl(c, rc.z);
}

/* [l]ocal to [g]lobal */

static __device__ float xl2xg(const Coords_v c, float xl) {
    return (c.xc + 0.5f) * c.Lx + xl;
}

static __device__ float yl2yg(const Coords_v c, float yl) {
    return (c.yc + 0.5f) * c.Ly + yl;
}

static __device__ float zl2zg(const Coords_v c, float zl) {
    return (c.zc + 0.5f) * c.Lz + zl;
}

static __device__ void local2global(const Coords_v c, float3 rl, /**/ float3 *rg) {
    rg->x = xl2xg(c, rl.x);
    rg->y = yl2yg(c, rl.y);
    rg->z = zl2zg(c, rl.z);
}

/* [g]lobal to [l]ocal */

static __device__ float xg2xl(const Coords_v c, float xg) {
    return xg - (c.xc + 0.5f) * c.Lx;
}

static __device__ float yg2yl(const Coords_v c, float yg) {
    return yg - (c.yc + 0.5f) * c.Ly;
}

static __device__ float zg2zl(const Coords_v c, float zg) {
    return zg - (c.zc + 0.5f) * c.Lz;
}

static __device__ void global2local(const Coords_v c, float3 rg, /**/ float3 *rl) {
    rl->x = xg2xl(c, rg.x);
    rl->y = yg2yl(c, rg.y);
    rl->z = zg2zl(c, rg.z);
}

