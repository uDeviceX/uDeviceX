void ini(WvelCste p, Wvel *vw) {
    vw->type = WALL_VEL_CSTE;
    vw->p.cste = p;
    step2params(0, /**/ vw);
}

void ini(WvelShear p, Wvel *vw) {
    vw->type = WALL_VEL_SHEAR;
    vw->p.shear = p;
    step2params(0, /**/ vw);
}

void ini(WvelShearSin p, Wvel *vw) {
    vw->type = WALL_VEL_SHEAR_SIN;
    vw->p.shearsin = p;
    step2params(0, /**/ vw);
}

void ini(WvelHS p, Wvel *vw) {
    vw->type = WALL_VEL_HS;
    vw->p.hs = p;
    step2params(0, /**/ vw);
}
