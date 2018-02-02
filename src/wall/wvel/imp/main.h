void wvel_ini(Wvel **wv) {
    UC(emalloc(sizeof(Wvel), (void**) wv));
}

void wvel_fin(Wvel *wv) {
    UC(efree(wv));
}

void wvel_set_cste(float3 u, Wvel *vw) {
    WvelCste p;
    p.u = u;
    vw->type = WALL_VEL_CSTE;
    vw->p.cste = p;
}

void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw) {
    WvelShear p;
    p.gdot = gdot;
    p.vdir = vdir;
    p.gdir = gdir;
    p.half = half;
    
    vw->type = WALL_VEL_SHEAR;
    vw->p.shear = p;
}

void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, int log_freq, Wvel *vw) {
    WvelShearSin p;
    p.gdot     = gdot;
    p.vdir     = vdir;
    p.gdir     = gdir;
    p.half     = half;
    p.w        = w;
    p.log_freq = log_freq;
    
    vw->type = WALL_VEL_SHEAR_SIN;
    vw->p.shearsin = p;
}

void wvel_set_hs(float u, float h, Wvel *vw) {
    WvelHS p;
    p.u = u;
    p.h = h;
    vw->type = WALL_VEL_HS;
    vw->p.hs = p;
}

void wvel_set_timestep(float dt0, Wvel *vw) {
    vw->dt0 = dt0;
    assert(dt0 >= 0.95*dt && dt0 <= 1.05*dt);
}


static void set_dev(float dt0, WvelCste p, Wvel_v *wv) {
    wv->type = WALL_VEL_V_CSTE;
    wv->p.cste.u = p.u;
    wv->dt0 = dt0;
}

static void set_dev(float dt0, WvelShear p, Wvel_v *wv) {
    wv->type = WALL_VEL_V_SHEAR;
    wv->p.shear.gdot = p.gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
    wv->p.shear.half = p.half;
    wv->dt0 = dt0;
}

static void set_dev(float dt0, long it, WvelShearSin p, Wvel_v *wv) {
    float gdot;
    float t, w;
    bool cond;
    wv->type = WALL_VEL_V_SHEAR;
    wv->dt0 = dt0;

    w = p.w;
    t = it * dt0;
    gdot = p.gdot * sin(w * t);
    
    wv->p.shear.gdot = gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
    wv->p.shear.half = p.half;
    
    cond = p.log_freq > 0 && it % p.log_freq == 0;
    if (cond)
        msg_print("WVEL_SIN: gd = %6.3g", gdot);
}

static void set_dev(float dt0, WvelHS p, Wvel_v *wv) {
    wv->type = WALL_VEL_V_HS;    
    wv->p.hs.u = p.u;
    wv->p.hs.h = p.h;
    wv->dt0 = dt0;
}

void wvel_get_view(long it, const Wvel *wv, /**/ Wvel_v *view) {
    switch (wv->type) {
    case WALL_VEL_CSTE:
        set_dev(wv->dt0, wv->p.cste, /**/ view);
        break;
    case WALL_VEL_SHEAR:
        set_dev(wv->dt0, wv->p.shear, /**/ view);
        break;
    case WALL_VEL_SHEAR_SIN:
        set_dev(wv->dt0, it, wv->p.shearsin, /**/ view);
        break;
    case WALL_VEL_HS:
        set_dev(wv->dt0, wv->p.hs, /**/ view);
        break;
    default:
        ERR("wrong type provided: <%d>", wv->type);
        break;
    };
}
