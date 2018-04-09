static void run_eq(float te, Sim *s) { /* equilibrate */
    float dt;
    s->equilibrating = true;
    bool wall0 = false;
    dt = utils_get_dt0(s);
    while (time_line_get_current(s->time.t) < te) {
        UC(step(s->time.t, dt, wall0, 0.0, s));
        time_line_advance(dt, s->time.t);
        dt = utils_get_dt(s, s->time.t);
    }
    UC(distribute_flu(/**/ s));
}

static void pre_run(const Config *cfg, Sim *s) {
    UC(bforce_set_conf(cfg, s->bforce));

    UC(utils_dump_history(cfg, "conf.history.cfg"));
    UC(dump_strt_templ(s));

    utils_compute_hematocrit(s);
    
    s->equilibrating = false;
}

static void run(float ts, float te, Sim *s) {
    float dt;

    /* ts, te: time start and end */
    dt = utils_get_dt0(s);
    while (time_line_get_current(s->time.t) < te) {
        UC(step(s->time.t, dt, s->opt.wall, ts, s));
        time_line_advance(dt, s->time.t);
        dt = utils_get_dt(s, s->time.t);
    }
    UC(distribute_flu(/**/ s));
}

