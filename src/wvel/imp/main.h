static void set_dev(WvelCste p, Wvel_d *wv) {
    wv->type = WALL_VEL_DEV_CSTE;
    wv->p.cste.u = p.u;
}

static void set_dev(WvelShear p, Wvel_d *wv) {
    wv->type = WALL_VEL_DEV_SHEAR;
    wv->p.shear.gdot = p.g;
}

void step2params(long it, /**/ Wvel *wv) {
    switch (wv->type) {
    case WALL_VEL_CSTE:
        set_dev(wv->p.cste, /**/ &wv->dev);
        break;
    case WALL_VEL_SHEAR:
        set_dev(wv->p.shear, /**/ &wv->dev);
        break;
    case WALL_VEL_SHEAR_SIN:
        // TODO
        break;
    default:
        ERR("wrong type provided: <%d>", wv->type);
        break;
    };
}
