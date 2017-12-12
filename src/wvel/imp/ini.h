void ini(WvelCste p, Wvel *vw) {
    vw->type = WALL_VEL_CSTE;
    vw->p.cste = p;
}

void ini(WvelShear p, Wvel *vw) {
    vw->type = WALL_VEL_SHEAR;
    vw->p.shear = p;
}

void ini(WvelShearSin p, Wvel *vw) {
    vw->type = WALL_VEL_SHEAR_SIN;
    vw->p.shearsin = p;
}

void ini(WvelHS p, Wvel *vw) {
    vw->type = WALL_VEL_HS;
    vw->p.hs = p;
}
