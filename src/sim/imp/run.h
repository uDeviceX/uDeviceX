static void run_eq(Time *time, float te, Sim *s) { /* equilibrate */
    long it;
    float dt;
    BForce *bforce;
    UC(bforce_ini(&bforce));
    s->equilibrating = true;
    bforce_ini_none(/**/ bforce);    
    bool wall0 = false;
    for (it = 0; (time_current(time) < te); ++it) {
        UC(step(time, bforce, wall0, 0, it, s));
        dt = time_dt(time);
        time_next(time, dt);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}

static void run(const Config *cfg, Time *time, float ts, float te, Sim *s) {
    float dt;
    long start, it; /* current timestep */
    Wall *wall = &s->wall;
    BForce *bforce;

    UC(bforce_ini(&bforce));
    UC(bforce_ini_conf(cfg, bforce));

    dump_strt_templ(s->coords, wall, s); /* :TODO: is it the right place? */
    s->equilibrating = false;   

    /* ts, te: time start and end */
    start = (long)(ts/time_dt(time)); //assumes dt=const up to now.
    for (it = start; (time_current(time) < te); ++it) {
        UC(step(time, bforce, walls, start, it, s));
        dt = time_dt(time);
        time_next(time, dt);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}
