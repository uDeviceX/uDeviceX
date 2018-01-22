void run_eq(long te, Sim *s) { /* equilibrate */
    BForce *bforce;
    UC(bforce_ini(&bforce));
    s->equilibrating = true;
    
    bforce_ini_none(/**/ bforce);    
    bool wall0 = false;
    for (long it = 0; it < te; ++it) UC(step(bforce, wall0, it, s));
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}

void run(long ts, long te, Sim *s) {
    long it; /* current timestep */
    Wall *wall = &s->wall;
    BForce *bforce;
    
    UC(bforce_ini(&bforce));
    UC(bforce_ini_conf(s->cfg, bforce));

    dump_strt_templ(s->coords, wall, s); /* :TODO: is it the right place? */
    s->equilibrating = false;   
    
    /* ts, te: time start and end */
    for (it = ts; it < te; ++it) {
        UC(wvel_get_view(it - ts, wall->vel, /**/ &wall->vview));
        UC(step(bforce, walls, it, s));
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}
