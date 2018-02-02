void step(float dt0, BForce *bforce, bool wall0, int ts, int it, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;

    if (walls && !s->equilibrating)
        UC(wvel_get_view(it - ts, wall->vel, /**/ &wall->vview));
    
    UC(check_sizes(s));
    UC(check_pos_soft(s));
    
    UC(distribute_flu(s));
    if (s->solids0) UC(distribute_rig(/**/ rig));
    if (rbcs)       UC(distribute_rbc(/**/ rbc));

    UC(check_sizes(s));
    
    forces(wall0, s);

    UC(check_forces(s));
    
    dump_diag0(it, s);
    dump_diag_after(it, wall0, s->solids0, s);
    body_force(it, bforce, s);

    restrain(it, /**/ s);
    update_solvent(s->moveparams, it, /**/ flu);
    if (s->solids0) update_solid(dt0, /**/ rig);
    if (rbcs)       update_rbc(s->moveparams, it, rbc, s);

    UC(check_vel(s));
    
    if (s->opt.vcon && !s->equilibrating) {
        sample(s->coords, it, flu, /**/ &s->vcon);
        adjust(it, /**/ &s->vcon, bforce);
        log(it, &s->vcon);
    }

    if (wall0) bounce_wall(s->coords, wall, /**/ flu, rbc);
    
    if (sbounce_back && s->solids0) bounce_solid(it, s->L, /**/ &s->bb, rig, flu);

    UC(check_pos_soft(s));
    UC(check_vel(s));

    if (! s->equilibrating) {
        if (s->opt.inflow)     apply_inflow(s->inflow, /**/ flu);
        if (s->opt.outflow)    mark_outflow(flu, /**/ s->outflow);
        if (s->opt.denoutflow) mark_outflowden(flu, s->mapoutflow, /**/ s->denoutflow);
    }
}
