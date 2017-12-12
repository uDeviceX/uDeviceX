static void set_dev(WvelCste p, Wvel_d *wv) {
    wv->type = WALL_VEL_DEV_CSTE;
    wv->p.cste.u = p.u;
}

static void set_dev(WvelShear p, Wvel_d *wv) {
    wv->type = WALL_VEL_DEV_SHEAR;
    wv->p.shear.gdot = p.gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
    wv->p.shear.half = p.half;
}

static void set_dev(long it, WvelShearSin p, Wvel_d *wv) {
    float gdot;
    float t, w;
    bool cond;
    wv->type = WALL_VEL_DEV_SHEAR;

    w = p.w;
    t = it * dt;
    gdot = p.gdot * sin(w * t);
    
    wv->p.shear.gdot = gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
    wv->p.shear.half = p.half;
    
    cond = p.log_freq > 0 && it % p.log_freq == 0;
    if (cond)
        MSG("WVEL_SIN: gd = %6.3g", gdot);
}

static void set_dev(long it, WvelHS p, Wvel_d *wv) {
    wv->type = WALL_VEL_DEV_HS;    
    wv->p.hs.u = p.u;
    wv->p.hs.h = p.h;
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
        set_dev(it, wv->p.shearsin, /**/ &wv->dev);
        break;
    case WALL_VEL_HS:
        set_dev(it, wv->p.hs, /**/ &wv->dev);
        break;
    default:
        ERR("wrong type provided: <%d>", wv->type);
        break;
    };
}
