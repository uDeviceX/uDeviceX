static bool active_walls(const Sim *s) {
    return !s->equilibrating && s->opt.wall;
}

static bool active_rbc(const Sim *s) {
    return !s->equilibrating && s->opt.rbc.active;
}

static bool active_rig(const Sim *s) {
    return !s->equilibrating && s->opt.rig.active;
}

static bool is_sampling_time(const Sim *s) {
    const Opt *opt = &s->opt;
    const float freq = opt->freq_field / opt->sampler_npdump;
    const TimeLine *time = s->time.t;

    return time_line_cross(time, freq);
}

static void utils_compute_hematocrit(const Sim *s) {
    const Opt *opt = &s->opt;
    double Vdomain, Vrbc, Ht;
    if (!active_rbc(s)) return;

    if (opt->wall) {
        Vdomain = wall_compute_volume(s->wall, s->cart, s->opt.params.L);
    }
    else {
        const Coords *c = s->coords;
        Vdomain = xdomain(c) * ydomain(c) * zdomain(c);
    }

    Vrbc = objects_mbr_tot_volume(s->obj);
    
    Ht = Vrbc / Vdomain;

    msg_print("Geometry volume: %g", Vdomain);
    msg_print("Hematocrit: %g", Ht);
}

static void utils_dump_history(const Config *cfg, const char *fname) {
    FILE *f;
    UC(efopen(fname, "w", &f));
    UC(conf_write_history(cfg, f));
    UC(efclose(f));
}

static float utils_get_dt0(Sim *s) {
    TimeStep *time_step;
    time_step = s->time.step;
    return time_step_dt0(time_step);
}

static float utils_get_dt(Sim *s, TimeLine *time) {
    /* Possibility to adapt dt only after equilibration */
    if (s->equilibrating)
        return time_step_dt0(s->time.step);
    else {
        const Flu *flu = &s->flu;
        // TODO
        // const Rbc *rbc = &s->rbc;
        const Opt *opt = &s->opt;

        time_step_accel_reset(s->time.accel);
        if (flu->q.n)
            time_step_accel_push(s->time.accel, flu->mass, flu->q.n, flu->ff);
        // if (active_rbc(s) && rbc->q.n)
        //     time_step_accel_push(s->time.accel, rbc->mass, rbc->q.n, rbc->ff);

        const float dt = time_step_dt(s->time.step, s->cart, s->time.accel);

        if (time_line_cross(time, opt->freq_parts))
            time_step_log(s->time.step);

        return dt;
    }
}

static void utils_get_pf_flu(Sim *s, PFarray *flu) {
    Flu *f = &s->flu;
    flu->n = f->q.n;
    UC(parray_push_pp(f->q.pp, &flu->p));
    if (s->opt.flucolors)
        parray_push_cc(f->q.cc, &flu->p);

    UC(farray_push_ff(f->ff, &flu->f));
}

static InterWalInfos get_winfo(Sim *s) {
    InterWalInfos wi;
    wi.active = s->opt.wall;
    if (s->opt.wall)  wall_get_sdf_ptr(s->wall, &wi.sdf);
    else wi.sdf = NULL;
    return wi;
}

static InterFluInfos get_finfo(Sim *s) {
    InterFluInfos fi;
    fi.q = &s->flu.q;
    return fi;
}

