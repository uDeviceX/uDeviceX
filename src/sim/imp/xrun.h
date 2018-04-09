static void sim_run(Sim *s) {
    float dt, ts;
    ts = 0;

    dt = get_dt0(s);
    while (time_line_get_current(s->time.t) < s->time.end) {
        UC(step(s->time.t, dt, s->opt.wall, ts, s));
        time_line_advance(dt, s->time.t);
        dt = get_dt(s, s->time.t);
    }
    UC(distribute_flu(/**/ s));
}
