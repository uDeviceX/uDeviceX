static float get_dt0(Sim *s) {
    TimeStep *time_step;    
    time_step = s->time_step;
    return time_step_dt0(time_step);
}

static float get_dt(Sim *s, TimeLine *time) {
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

        if (time_line_cross(time, opt->freq_parts))
            time_step_log(s->time_step);

        return dt;
    }
}

static void run_eq(TimeLine *time, float te, Sim *s) { /* equilibrate */
    float dt;
    s->equilibrating = true;
    bool wall0 = false;
    dt = get_dt0(s);    
    while (time_line_current(time) < te) {
        UC(step(time, dt, wall0, 0.0, s));
        time_line_next(time, dt);
        dt = get_dt(s, time);
    }
    UC(distribute_flu(/**/ s));
}

static void dump_history(const Config *cfg, const char *fname) {
    FILE *f;
    UC(efopen(fname, "w", &f));
    UC(conf_write_history(cfg, f));
    UC(efclose(f));
}

static double compute_volume_rbc(MPI_Comm comm, const Rbc *r) {
    double loc, tot, V0;
    long nc;
    nc = r->q.nc;
    V0 = rbc_params_get_tot_volume(r->params);

    tot = 0;
    loc = nc * V0;
    MC(m::Allreduce(&loc, &tot, 1, MPI_DOUBLE, MPI_SUM, comm));
    
    return tot;
}

static void compute_hematocrit(const Sim *s) {
    const Opt *opt = &s->opt;
    double Vdomain, Vrbc, Ht;
    if (!opt->rbc) return;

    if (opt->wall) {
        enum {NSAMPLES = 100000};
        Vdomain = sdf_compute_volume(s->cart, s->params.L, s->wall.sdf, NSAMPLES);
    }
    else {
        const Coords *c = s->coords;
        Vdomain = xdomain(c) * ydomain(c) * zdomain(c);
    }

    Vrbc = compute_volume_rbc(s->cart, &s->rbc);
    
    Ht = Vrbc / Vdomain;

    msg_print("Geometry volume: %g", Vdomain);
    msg_print("Hematocrit: %g", Ht);
}

static void pre_run(const Config *cfg, Sim *s) {
    UC(bforce_set_conf(cfg, s->bforce));

    UC(dump_history(cfg, "conf.history.cfg"));
    UC(dump_strt_templ(s));

    compute_hematocrit(s);
    
    s->equilibrating = false;         
}

static void run(TimeLine *time, float ts, float te, Sim *s) {
    float dt;

    /* ts, te: time start and end */
    dt = get_dt0(s);
    while (time_line_current(time) < te) {
        UC(step(time, dt, s->opt.wall, ts, s));
        time_line_next(time, dt);
        dt = get_dt(s, time);
    }
    UC(distribute_flu(/**/ s));
}
