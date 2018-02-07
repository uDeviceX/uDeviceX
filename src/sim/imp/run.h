static void run_eq(Time *time, float te, Sim *s) { /* equilibrate */
    long it;
    float dt;
    BForce *bforce;
    UC(bforce_ini(&bforce));
    s->equilibrating = true;
    bforce_ini_none(/**/ bforce);    
    bool wall0 = false;
    for (it = 0; (time_current(time) < te); ++it) {
        dt = time_dt(time);
        UC(step(dt, bforce, wall0, 0, it, s));
        time_next(time, dt);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}

static void run(Config *cfg, Time *time, float ts, float te, Sim *s) {
    float dt;
    long start, it; /* current timestep */
    Wall *wall = &s->wall;
    BForce *bforce;

    UC(bforce_ini(&bforce));
    UC(bforce_ini_conf(cfg, bforce));

    dump_strt_templ(s->coords, wall, s); /* :TODO: is it the right place? */
    s->equilibrating = false;   

    /* ts, te: time start and end */
    start = (long)(ts/time_dt(time));
    for (it = start; (time_current(time) < te); ++it) {
        dt = time_dt(time);
        UC(step(dt, bforce, walls, ts, it, s));
        time_next(time, dt);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}
