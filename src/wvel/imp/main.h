void set_params(WvelCste_d p, Wvel_d *wv) {
    wv->type   = WALL_VEL_DEV_CSTE;
    wv->p.cste = p;
}

void set_params(WvelShear_d p, Wvel_d *wv) {
    wv->type    = WALL_VEL_DEV_SHEAR;
    wv->p.shear = p;
}

void step2params(long it, /**/ Wvel_d *wv) {
    
}
