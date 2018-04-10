static void pre_run(const Config *cfg, Sim *s) {
    s->equilibrating = false;
    
    UC(bforce_set_conf(cfg, s->bforce));

    UC(utils_dump_history(cfg, "conf.history.cfg"));
    UC(dump_strt_templ(s));

    utils_compute_hematocrit(s);
}

static void run(float ts, float te, Sim *s) {
    float dt;

    /* ts, te: time start and end */
    dt = utils_get_dt0(s);
    while (time_line_get_current(s->time.t) < te) {
        UC(step(s->time.t, dt, ts, s));
        time_line_advance(dt, s->time.t);
        dt = utils_get_dt(s, s->time.t);
    }
    UC(distribute_flu(/**/ s));
}

