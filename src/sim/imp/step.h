static void check_size(long n, long nmax) {
    if (n < 0 || n > nmax)
        ERR("wrong size: %ld / %ld", n, nmax);
}

static void check_sizes(Sim *s) {
    UC(check_size(s->rbc.q.nc, MAX_CELL_NUM));
    UC(check_size(s->rbc.q.n , MAX_CELL_NUM * RBCnv));
    UC(check_size(s->flu.q.n , MAX_PART_NUM)); 
}

void step(BForce *bforce, bool wall0, int it, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    
    UC(check_sizes(s));
    
    UC(distribute_flu(flu));
    if (solids0) UC(distribute_rig(/**/ rig));
    if (rbcs)    UC(distribute_rbc(/**/ rbc));

    UC(check_sizes(s));
    
    forces(wall0, s);

    dump_diag0(coords, it, s);
    dump_diag_after(it, wall0, solids0, s);
    body_force(it, *bforce, s);

    restrain(it, /**/ flu, rbc);
    update_solvent(it, /**/ flu);
    if (solids0) update_solid(/**/ rig);
    if (rbcs)    update_rbc(it, rbc, s);

    if (VCON && wall0) {
        sample(it, flu, /**/ &vcont);
        adjust(it, /**/ &vcont, bforce);
        log(it, &vcont);
    }

    if (wall0) bounce_wall(coords, &wall, /**/ flu, rbc);

    if (sbounce_back && solids0) bounce_solid(it, /**/ &bb, rig, flu);

    if (wall0) {
        if (INFLOW)  apply_inflow(inflow, flu);
        if (OUTFLOW) mark_outflow(flu, /**/ outflow);
        if (OUTFLOW_DEN) mark_outflowden(flu, mapoutflow, /**/ denoutflow);
    }
}
