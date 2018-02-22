static void step(Time *time, float dt, BForce *bforce, bool wall0, float tstart, Sim *s) {
    long it;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;
    const Opt *opt = &s->opt;
    if (opt->wall && !s->equilibrating)
        UC(wvel_get_step(time_current(time) - tstart, wall->vel, /**/ wall->velstep));

    UC(check_sizes(s));
    UC(check_pos_soft(s));

    UC(distribute_flu(s));
    if (s->solids0) UC(distribute_rig(/**/ rig));
    if (opt->rbc) UC(distribute_rbc(/**/ rbc));

    UC(check_sizes(s));
    UC(forces(dt, time, wall0, s));
    UC(check_forces(dt, s));

    it = time_iteration(time);
    dump_diag(time, s);
    dump_diag_after(time, s->solids0, s);
    UC(body_force(bforce, s));

    UC(restrain(it, /**/ s));
    UC(update_solvent(dt, /**/ flu));
    if (s->solids0) update_solid(dt, /**/ rig);
    if (opt->rbc) update_rbc(dt, it, rbc, s);

    UC(check_vel(dt, s));
    if (opt->vcon && !s->equilibrating) {
        sample(s->coords, it, flu, /**/ &s->vcon);
        adjust(it, /**/ &s->vcon, bforce);
        log(it, &s->vcon);
    }

    if (wall0) bounce_wall(dt, opt->rbc, s->coords, wall, /**/ flu, rbc);

    if (opt->rig_bounce && s->solids0) bounce_solid(dt, s->params.L, /**/ &s->bb, rig, flu);

    UC(check_pos_soft(s));
    UC(check_vel(dt, s));

    if (! s->equilibrating) {
        if (opt->inflow)     UC(apply_inflow(s->params.kBT, numberdensity, dt, s->inflow, /**/ flu));
        if (opt->outflow)    UC(mark_outflow(flu, /**/ s->outflow));
        if (opt->denoutflow) UC(mark_outflowden(flu, s->mapoutflow, /**/ s->denoutflow));
        if (opt->flucolors)  UC(recolor_flux(s->coords, &s->recolorer, flu));
    }
}
