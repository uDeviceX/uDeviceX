static void step(TimeLine *time, float dt, bool wall0, float tstart, Sim *s) {
    long it;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;
    BForce *bforce = s->bforce;
    const Opt *opt = &s->opt;
    if (opt->wall && !s->equilibrating)
        UC(wvel_get_step(time_line_current(time) - tstart, wall->vel, /**/ wall->velstep));

    UC(check_sizes(s));
    UC(check_pos_soft(s));

    UC(distribute_flu(s));
    if (s->rigids) UC(distribute_rig(/**/ rig));
    if (opt->rbc)  UC(distribute_rbc(/**/ rbc));

    UC(check_sizes(s));
    UC(forces(dt, time, wall0, s));
    UC(check_forces(dt, s));

    it = time_line_iteration(time);
    dump_diag(time, s);
    dump_diag_after(time, s->rigids, s);
    UC(body_force(bforce, s));

    UC(restrain(it, /**/ s));
    UC(update_solvent(dt, /**/ flu));
    if (s->rigids) update_solid(dt, /**/ rig);
    if (opt->rbc)  update_rbc(dt, it, rbc, s);

    UC(check_vel(dt, s));
    if (opt->vcon && !s->equilibrating) {
        sample(s->coords, it, flu, /**/ &s->vcon);
        adjust(it, /**/ &s->vcon, bforce);
        log(it, &s->vcon);
    }

    if (wall0) bounce_wall(dt, opt->rbc, s->coords, wall, /**/ flu, rbc);

    if (opt->rig_bounce && s->rigids) bounce_solid(dt, s->params.L, /**/ &s->bb, rig, flu);

    UC(check_pos_soft(s));
    UC(check_vel(dt, s));

    if (! s->equilibrating) {
        if (opt->inflow)     UC(apply_inflow(s->params.kBT, s->params.numdensity, dt, s->inflow, /**/ flu));
        if (opt->outflow)    UC(mark_outflow(flu, /**/ s->outflow));
        if (opt->denoutflow) UC(mark_outflowden(s->params, flu, s->mapoutflow, /**/ s->denoutflow));
        if (opt->flucolors)  UC(recolor_flux(s->coords, &s->recolorer, flu));
    }
}
