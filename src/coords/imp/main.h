enum {X, Y, Z, D};
    
void coords_ini(MPI_Comm cart, Coords *c) {
    int dims[D], periods[D], coords[D];
    MC(m::Cart_get(cart, D, dims, periods, coords));

    c->xc = coords[X];
    c->yc = coords[Y];
    c->zc = coords[Z];

    c->xd = dims[X];
    c->yd = dims[Y];
    c->zd = dims[Z];
}

void coords_fin(Coords *) {/*empty*/}

/* domain sizes */

int xdomain(const Coords *c) {
    return XS * c->xd;
}

int ydomain(const Coords *c) {
    return YS * c->yd;
}

int zdomain(const Coords *c) {
    return ZS * c->zd;
}

/* [l]ocal to [c]enter */

float xl2xc(const Coords *c, float xl) {
    return xl - 0.5f * XS * (c->xd - 2.f * c->xc - 1);
}

float yl2yc(const Coords *c, float yl) {
    return yl - 0.5f * YS * (c->yd - 2 * c->yc - 1);
}

float zl2zc(const Coords *c, float zl) {
    return zl - 0.5f * ZS * (c->zd - 2 * c->zc - 1);
}

void local2center(const Coords *c, float3 rl, /**/ float3 *rc) {
    rc->x = xl2xc(c, rl.x);
    rc->y = yl2yc(c, rl.y);
    rc->z = zl2zc(c, rl.z);
}

/* [c]enter to [l]ocal  */

float xc2xl(const Coords *c, float xc) {
    return xc + 0.5f * XS * (c->xd - 2 * c->xc - 1);
}

float yc2yl(const Coords *c, float yc) {
    return yc + 0.5f * YS * (c->yd - 2 * c->yc - 1);
}

float zc2zl(const Coords *c, float zc) {
    return zc + 0.5f * ZS * (c->zd - 2 * c->zc - 1);
}

void center2local(const Coords *c, float3 rc, /**/ float3 *rl) {
    rl->x = xc2xl(c, rc.x);
    rl->y = yc2yl(c, rc.y);
    rl->z = zc2zl(c, rc.z);
}

/* [l]ocal to [g]lobal */

float xl2xg(const Coords *c, float xl) {
    return (c->xc + 0.5f) * XS + xl;
}

float yl2yg(const Coords *c, float yl) {
    return (c->yc + 0.5f) * YS + yl;
}

float zl2zg(const Coords *c, float zl) {
    return (c->zc + 0.5f) * ZS + zl;
}

void local2global(const Coords *c, float3 rl, /**/ float3 *rg) {
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

/* edges of the sub-domain in global coordinates */

int xlo(const Coords c) { return xl2xg(&c, 0) - 0.5*XS; }
int ylo(const Coords c) { return yl2yg(&c, 0) - 0.5*YS; }
int zlo(const Coords c) { return zl2zg(&c, 0) - 0.5*ZS; }

int xhi(const Coords c) { return xl2xg(&c, 0) + 0.5*XS; }
int yhi(const Coords c) { return yl2yg(&c, 0) + 0.5*YS; }
int zhi(const Coords c) { return zl2zg(&c, 0) + 0.5*ZS; }

/* sizes of the sub-domain */

int xs(const Coords) { return XS; }
int ys(const Coords) { return YS; }
int zs(const Coords) { return ZS; }

/* rank predicates */

bool is_end(Coords c, int dir) {
    enum {X, Y, Z};
    switch (dir) {
    case X: return c.xc == c.xd - 1;
    case Y: return c.yc == c.yd - 1;
    case Z: return c.zc == c.zd - 1;
    }
    return false;
}

void coord_stamp(Coords c, /**/ char *s) {
    int r;
    r = sprintf(s, "%03d.%03d.%03d", c.xc, c.yc, c.zc);
    if (r < 0) ERR("sprintf failed");
}
