void run_eq(long te, Sim *s) { /* equilibrate */
    BForce *bforce;
    UC(bforce_ini(&bforce));
    s->equilibrating = true;
    
    bforce_ini_none(/**/ bforce);    
    bool wall0 = false;
    for (long it = 0; it < te; ++it) step(bforce, wall0, it, s);
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}

static void ini_bforce(const Config *cfg, BForce *bforce) {
    const char *type;
    UC(conf_lookup_string(cfg, "bforce.type", /**/ &type));

    if      (same_str(type, "none")) {
        bforce_ini_none(/**/ bforce);
    }
    else if (same_str(type, "constant")) {
        float3 f;
        UC(conf_lookup_float3(cfg, "bforce.f", /**/ &f));
        UC(bforce_ini_cste(f, /**/ bforce));
    }
    else if (same_str(type, "double_poiseuille")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_dp(a, /**/ bforce));
    }
    else if (same_str(type, "shear")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_shear(a, /**/ bforce));
    }
    else if (same_str(type, "four_roller")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_rol(a, /**/ bforce));
    }
    else if (same_str(type, "rad")) {
        float a;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &a));
        UC(bforce_ini_rad(a, /**/ bforce));
    }
    else {
        ERR("Unrecognized type <%s>", type);
    }
}

void run(long ts, long te, Sim *s) {
    long it; /* current timestep */
    Wall *wall = &s->wall;
    BForce *bforce;
    
    UC(bforce_ini(&bforce));
    UC(ini_bforce(s->cfg, bforce));

    dump_strt_templ(s->coords, wall, s); /* :TODO: is it the right place? */
    s->equilibrating = false;   
    
    /* ts, te: time start and end */
    for (it = ts; it < te; ++it) {
        step2params(it - ts, &wall->vel, /**/ &wall->vview);
        step(bforce, walls, it, s);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}
