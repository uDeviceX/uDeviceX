void wvel_ini(Wvel **wv) {
    EMALLOC(1, wv);
}

void wvel_fin(Wvel *wv) {
    EFREE(wv);
}

void wvel_step_ini(WvelStep **wv) {
    EMALLOC(1, wv);
}

void wvel_step_fin(WvelStep *wv) {
    EFREE(wv);
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
    
    vw->type = WALL_VEL_SHEAR;
    vw->p.shear = p;
}

void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, int log_freq, Wvel *vw) {
    WvelShearSin p;
    p.gdot     = gdot;
    p.vdir     = vdir;
    p.gdir     = gdir;
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

static void set_step_cste(WvelCste p, WvelStep *wv) {
    wv->type = WALL_VEL_V_CSTE;
    wv->p.cste.u = p.u;
}

static void set_step_shear(WvelShear p, WvelStep *wv) {
    wv->type = WALL_VEL_V_SHEAR;
    wv->p.shear.gdot = p.gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
}

static void set_step_shear(float dt, long it, WvelShearSin p, WvelStep *wv) {
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
    
    cond = p.log_freq > 0 && it % p.log_freq == 0;
    if (cond)
        msg_print("WVEL_SIN: gd = %6.3g", gdot);
}

static void set_step_hs(WvelHS p, WvelStep *wv) {
    wv->type = WALL_VEL_V_HS;    
    wv->p.hs.u = p.u;
    wv->p.hs.h = p.h;
}

void wvel_get_step(float dt, long it, const Wvel *wv, /**/ WvelStep *view) {
    switch (wv->type) {
    case WALL_VEL_CSTE:
        set_step_cste(wv->p.cste, /**/ view);
        break;
    case WALL_VEL_SHEAR:
        set_step_shear(wv->p.shear, /**/ view);
        break;
    case WALL_VEL_SHEAR_SIN:
        set_step_shear(dt, it, wv->p.shearsin, /**/ view);
        break;
    case WALL_VEL_HS:
        set_step_hs(wv->p.hs, /**/ view);
        break;
    default:
        ERR("wrong type provided: <%d>", wv->type);
        break;
    };
}

int wvel_get_type(const WvelStep *w) {
    return w->type;
}

void wvel_get_view(const WvelStep *w, /**/ WvelCste_v *v) {
    *v = w->p.cste;
}

void wvel_get_view(const WvelStep *w, /**/ WvelShear_v *v) {    
    *v = w->p.shear;
}

void wvel_get_view(const WvelStep *w, /**/ WvelHS_v *v) {
    *v = w->p.hs;
}
