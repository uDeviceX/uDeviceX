static float get_dt0(Sim *s) {
    TimeStep *time_step;    
    time_step = s->time_step;
    return time_step_dt0(time_step);
}

static float get_dt(Sim* s, Time* time) {
    /* Possibility to adapt dt only after equilibration */
    if (s->equilibrating)
        return time_step_dt0(s->time_step);
    else {
        const Flu *flu = &s->flu;
        const Rbc *rbc = &s->rbc;
        const Opt *opt = &s->opt;

        time_step_accel_reset(s->time_step_accel);
        if (flu->q.n)
            time_step_accel_push(s->time_step_accel, flu->mass, flu->q.n, flu->ff);
        if (opt->rbc && rbc->q.n)
            time_step_accel_push(s->time_step_accel, rbc->mass, rbc->q.n, rbc->ff);

        const float dt = time_step_dt(s->time_step, s->cart, s->time_step_accel);

        if (time_cross(time, opt->freq_parts))
            time_step_log(s->time_step);

        return dt;
    }
}

static void run_eq(Time *time, float te, Sim *s) { /* equilibrate */
    float dt;
    BForce *bforce;
    UC(bforce_ini(&bforce));
    s->equilibrating = true;
    bforce_ini_none(/**/ bforce);    
    bool wall0 = false;
    dt = get_dt0(s);    
    while (time_current(time) < te) {
        UC(step(time, dt, bforce, wall0, 0.0, s));
        time_next(time, dt);
        dt = get_dt(s, time);
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
    dt = get_dt0(s);
    while (time_current(time) < te) {
        UC(step(time, dt, bforce, walls, ts, s));
        time_next(time, dt);
        dt = get_dt(s, time);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}
