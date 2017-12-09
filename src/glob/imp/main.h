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

void local2global(const Coords *c, float3 rl, /**/ float3 *rg) {
    rg->x = (c->xc + 0.5f) * XS + rl.x;
    rg->y = (c->yc + 0.5f) * YS + rl.y;
    rg->z = (c->zc + 0.5f) * ZS + rl.z;    
}

void global2local(const Coords *c, float3 rg, /**/ float3 *rl) {
    rl->x = rg.x - (c->xc + 0.5f) * XS;
    rl->y = rg.y - (c->yc + 0.5f) * YS;
    rl->z = rg.z - (c->zc + 0.5f) * ZS; 
}
