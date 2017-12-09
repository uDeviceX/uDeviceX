enum {X, Y, Z, D};
    
void ini(MPI_Comm cart, Coords *c) {
    int dims[D], periods[D], coords[D];
    MC(m::Cart_get(cart, D, dims, periods, coords));

    c->xc = coords[X];
    c->yc = coords[Y];
    c->zc = coords[Z];

    c->xd = dims[X];
    c->yd = dims[Y];
    c->zd = dims[Z];
}

void fin(Coords *) {/*empty*/}

/* [l]ocal to [c]enter */

float xl2xc(const Coords c, float xl) {
    return xl + XS * (c.xd - 2.f * c.xc - 1) / 2;
}

float yl2yc(const Coords c, float yl) {
    return yl + YS * (c.yd - 2.f * c.yc - 1) / 2;
}

float zl2zc(const Coords c, float zl) {
    return zl + ZS * (c.zd - 2.f * c.zc - 1) / 2;
}

void local2center(Coords c, float3 rl, /**/ float3 *rc) {
    rc->x = xl2xc(c, rl.x);
    rc->y = yl2yc(c, rl.y);
    rc->z = zl2zc(c, rl.z);
}

/* [c]enter to [l]ocal  */

float xc2xl(const Coords c, float xc) {
    return xc - XS * (c.xd - 2.f * c.xc - 1) / 2;
}

float yc2yl(const Coords c, float yc) {
    return yc - YS * (c.yd - 2.f * c.yc - 1) / 2;
}

float zc2zl(const Coords c, float zc) {
    return zc - ZS * (c.zd - 2.f * c.zc - 1) / 2;
}

void center2local(Coords c, float3 rc, /**/ float3 *rl) {
    rl->x = xc2xl(c, rc.x);
    rl->y = yc2yl(c, rc.y);
    rl->z = zc2zl(c, rc.z);
}

/* [l]ocal to [g]lobal */

float xl2xg(const Coords c, float xl) {
    return (c.xc + 0.5f) * XS + xl;
}

float yl2yg(const Coords c, float yl) {
    return (c.yc + 0.5f) * YS + yl;
}

float zl2zg(const Coords c, float zl) {
    return (c.zc + 0.5f) * ZS + zl;
}

void local2global(const Coords c, float3 rl, /**/ float3 *rg) {
    rg->x = xl2xg(c, rl.x);
    rg->y = yl2yg(c, rl.y);
    rg->z = zl2zg(c, rl.z);
}

/* [g]lobal to [l]ocal */

float xg2xl(const Coords c, float xg) {
    return xg - (c.xc + 0.5f) * XS;
}

float yg2yl(const Coords c, float yg) {
    return yg - (c.yc + 0.5f) * YS;
}

float zg2zl(const Coords c, float zg) {
    return zg - (c.zc + 0.5f) * ZS;
}

void global2local(const Coords c, float3 rg, /**/ float3 *rl) {
    rl->x = xg2xl(c, rg.x);
    rl->y = yg2yl(c, rg.y);
    rl->z = zg2zl(c, rg.z);
}
