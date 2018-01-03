static void set_dev(WvelCste p, Wvel_v *wv) {
    wv->type = WALL_VEL_V_CSTE;
    wv->p.cste.u = p.u;
}

static void set_dev(WvelShear p, Wvel_v *wv) {
    wv->type = WALL_VEL_V_SHEAR;
    wv->p.shear.gdot = p.gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
    wv->p.shear.half = p.half;
}

static void set_dev(long it, WvelShearSin p, Wvel_v *wv) {
    float gdot;
    float t, w;
    bool cond;
    wv->type = WALL_VEL_V_SHEAR;

    w = p.w;
    t = it * dt;
    gdot = p.gdot * sin(w * t);
    
    wv->p.shear.gdot = gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
    wv->p.shear.half = p.half;
    
    cond = p.log_freq > 0 && it % p.log_freq == 0;
    if (cond)
        msg_print("WVEL_SIN: gd = %6.3g", gdot);
}

static void set_dev(long it, WvelHS p, Wvel_v *wv) {
    wv->type = WALL_VEL_V_HS;    
    wv->p.hs.u = p.u;
    wv->p.hs.h = p.h;
}

void step2params(long it, const Wvel *wv, /**/ Wvel_v *view) {
    switch (wv->type) {
    case WALL_VEL_CSTE:
        set_dev(wv->p.cste, /**/ view);
        break;
    case WALL_VEL_SHEAR:
        set_dev(wv->p.shear, /**/ view);
        break;
    case WALL_VEL_SHEAR_SIN:
        set_dev(it, wv->p.shearsin, /**/ view);
        break;
    case WALL_VEL_HS:
        set_dev(it, wv->p.hs, /**/ view);
        break;
    default:
        ERR("wrong type provided: <%d>", wv->type);
        break;
    };
}
