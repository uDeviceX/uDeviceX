void run_eq(long te, Sim *s) { /* equilibrate */
    BForce bforce;
    s->equilibrating = true;
    
    ini_none(/**/ &bforce);    
    bool wall0 = false;
    for (long it = 0; it < te; ++it) step(&bforce, wall0, it, s);
    UC(distribute_flu(/**/ s));
}

static void ini_bforce(const Config *cfg, BForce *bforce) {
    const char *type;
    UC(conf_lookup_string(cfg, "bforce.type", /**/ &type));

    if      (same_str(type, "none")) {
        ini_none(/**/ bforce);
    }
    else if (same_str(type, "constant")) {
        BForce_cste par;
        UC(conf_lookup_float3(cfg, "bforce.f", /**/ &par.a));
        UC(ini(par, /**/ bforce));
    }
    else if (same_str(type, "double_poiseuille")) {
        BForce_dp par;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &par.a));
        UC(ini(par, /**/ bforce));
    }
    else if (same_str(type, "shear")) {
        BForce_shear par;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &par.a));
        UC(ini(par, /**/ bforce));
    }
    else if (same_str(type, "four_roller")) {
        BForce_rol par;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &par.a));
        UC(ini(par, /**/ bforce));
    }
    else if (same_str(type, "shear")) {
        BForce_rad par;
        UC(conf_lookup_float(cfg, "bforce.a", /**/ &par.a));
        UC(ini(par, /**/ bforce));
    }
    else {
        ERR("Unrecognized type <%s>", type);
    }
}

void run(long ts, long te, Sim *s) {
    long it; /* current timestep */
    Wall *wall = &s->wall;

    s->equilibrating = false;

    dump_strt_templ(s->coords, wall, s); /* :TODO: is it the right place? */

    BForce bforce;
    ini_bforce(s->cfg, &bforce);
    
    /* ts, te: time start and end */
    for (it = ts; it < te; ++it) {
        step2params(it - ts, &wall->vel, /**/ &wall->vview);
        step(&bforce, walls, it, s);
    }
    UC(distribute_flu(/**/ s));
}
