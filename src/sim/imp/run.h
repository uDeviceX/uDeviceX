static void run_eq(Time *time, float te, Sim *s) { /* equilibrate */
    float dt;
    BForce *bforce;
    UC(bforce_ini(&bforce));
    s->equilibrating = true;
    bforce_ini_none(/**/ bforce);    
    bool wall0 = false;
    while (time_current(time) < te) {
        dt = get_dt(s);
        UC(step(time, dt, bforce, wall0, 0.0, s));
        time_next(time, dt);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}

static void run(const Config *cfg, Time *time, float ts, float te, Sim *s) {
    float dt;
    Wall *wall = &s->wall;
    BForce *bforce;

    UC(bforce_ini(&bforce));
    UC(bforce_ini_conf(cfg, bforce));

    dump_strt_templ(s->coords, wall, s); /* :TODO: is it the right place? */
    s->equilibrating = false;   

    /* ts, te: time start and end */
    while (time_current(time) < te) {
        dt = get_dt(s);
        UC(step(time, dt, bforce, walls, ts, s));
        time_next(time, dt);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}
