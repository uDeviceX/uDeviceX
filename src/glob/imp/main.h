enum {X, Y, Z, D};
    
void ini(MPI_Comm cart, Coords *c) {
    int periods[D], coords[D];
    MC(m::Cart_get(cart, D, c->d, periods, coords));
    c->xc = coords[X];
    c->yc = coords[Y];
    c->zc = coords[Z];
}

void fin(Coords *) {/*empty*/}

static float xl2xc(const Coords c, float xl) {
    return xl + XS * (c.d[0] - 2.f * c.xc - 1) / 2;
}

static float yl2yc(const Coords c, float yl) {
    return yl + YS * (c.d[1] - 2.f * c.yc - 1) / 2;
}

static float zl2zc(const Coords c, float zl) {
    return zl + ZS * (c.d[2] - 2.f * c.zc - 1) / 2;
}

void local2center(Coords c, float3 rl, /**/ float3 *rc) {
    rc->x = xl2xc(c, rl.x);
    rc->y = yl2yc(c, rl.y);
    rc->z = zl2zc(c, rl.z);
}

void center2local(const Coords *c, float3 rc, /**/ float3 *rl) {
    rl->x = rc.x - XS * (c->d[X] - 2.f * c->xc - 1) / 2;
    rl->y = rc.y - YS * (c->d[Y] - 2.f * c->yc - 1) / 2;
    rl->z = rc.z - ZS * (c->d[Z] - 2.f * c->zc - 1) / 2;
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
