static void pre_run(const Config *cfg, Sim *s) {
    s->equilibrating = false;
    
    UC(utils_dump_history(cfg, "conf.history.cfg"));
    UC(dump_strt_templ(s));
    UC(utils_compute_hematocrit(s));
}

static void step(TimeLine *time, float dt, float tstart, Sim *s) {
    long it;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;
    BForce *bforce = s->bforce;
    const Opt *opt = &s->opt;
    if (active_walls(s))
        UC(wvel_get_step(time_line_get_current(time) - tstart, wall->vel, /**/ wall->velstep));

    UC(check_sizes(s));
    UC(check_pos_soft(s));

    UC(distribute_flu(s));
    if (active_rig(s)) UC(distribute_rig(/**/ rig));
    if (active_rbc(s)) UC(distribute_rbc(/**/ rbc));

    UC(check_sizes(s));
    UC(forces(dt, time, s));
    UC(check_forces(dt, s));

    it = time_line_get_iteration(time);
    dump_diag(time, s);
    dump_diag_after(time, active_rig(s), s);
    if (!s->equilibrating) UC(body_force(bforce, s));

    UC(restrain(it, /**/ s));
    UC(update_solvent(dt, /**/ flu));
    if (active_rig(s)) UC(update_solid(dt, /**/ rig));
    if (active_rbc(s)) UC(update_rbc(dt, it, rbc, s));

    UC(check_vel(dt, s));
    if (opt->vcon && !s->equilibrating) {
        sample(s->coords, it, flu, /**/ &s->vcon);
        adjust(it, /**/ &s->vcon, bforce);
        log(it, &s->vcon);
    }

    if (active_walls(s)) bounce_wall(dt, active_rbc(s), s->coords, wall, /**/ flu, rbc);

    if (active_rig(s) && opt->rig_bounce) bounce_solid(dt, s->params.L, /**/ &s->bb, rig, flu);

    UC(check_pos_soft(s));
    UC(check_vel(dt, s));

    if (! s->equilibrating) {
        if (opt->inflow)     UC(apply_inflow(s->params.kBT, s->params.numdensity, dt, s->inflow, /**/ flu));
        if (opt->outflow)    UC(mark_outflow(flu, /**/ s->outflow));
        if (opt->denoutflow) UC(mark_outflowden(s->params, flu, s->mapoutflow, /**/ s->denoutflow));
        if (opt->flucolors)  UC(recolor_flux(s->coords, &s->recolorer, flu));
    }
}

/* ts, te: time start and end */
static void run(float ts, float te, Sim *s) {
    float dt;

    dt = utils_get_dt0(s);
    while (time_line_get_current(s->time.t) < te) {
        UC(step(s->time.t, dt, ts, s));
        time_line_advance(dt, s->time.t);
        dt = utils_get_dt(s, s->time.t);
    }
    UC(distribute_flu(/**/ s));
}

