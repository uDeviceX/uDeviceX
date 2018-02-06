void run_eq(Time *time, float te, Sim *s) { /* equilibrate */
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
        msg_print("te/dt: %d", (int) te/dt);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}

void run(Time *time, long ts, float te, Sim *s) {
    float dt;
    long it; /* current timestep */
    Wall *wall = &s->wall;
    BForce *bforce;

    UC(bforce_ini(&bforce));
    UC(bforce_ini_conf(s->cfg, bforce));

    dump_strt_templ(s->coords, wall, s); /* :TODO: is it the right place? */
    s->equilibrating = false;   

    /* ts, te: time start and end */
    for (it = ts; (time_current(time) < te); ++it) {
        dt = time_dt(time);
        UC(step(dt, bforce, walls, ts, it, s));
        time_next(time, dt);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}
