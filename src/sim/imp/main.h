enum {
    NCELLSWALL =
    (XS + 2*XWM) *
    (YS + 2*YWM) *
    (ZS + 2*ZWM)
};

enum {
    MAXNWALL = NCELLSWALL * numberdensity
};

static void gen(const Coords *coords, Wall *w, Sim *s) { /* generate */
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    
    run_eq(wall_creation, s);
    if (walls) {
        dSync();
        UC(sdf_gen(coords, s->cart, /**/ w->sdf));
        MC(m::Barrier(s->cart));
        inter::create_walls(s->cart, MAXNWALL, w->sdf, /*io*/ &flu->q, /**/ &w->q);
    }
    inter::freeze(coords, s->cart, w->sdf, /*io*/ &flu->q, /**/ &rig->q, &rbc->q);
    clear_vel(s);

    if (multi_solvent) {
        Particle *pp = flu->q.pp;
        int n = flu->q.n;
        int *cc = flu->q.cc;
        Particle *pp_hst = s->pp_dump;
        int *cc_hst = flu->q.cc_hst;
        inter::color_dev(*coords, pp, n, /*o*/ cc, /*w*/ pp_hst, cc_hst);
    }
}

void sim_gen(Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Wall *wall = &s->wall;
    
    flu_gen_quants(s->coords, &flu->q);
    flu_build_cells(&flu->q);
    if (global_ids)  flu_gen_ids  (s->cart, flu->q.n, &flu->q);
    if (rbcs) {
        rbc_gen_quants(&s->coords, s->cart, "rbc.off", "rbcs-ic.txt", /**/ &rbc->q);
        rbc_force_gen(rbc->q, &rbc->tt);

        if (multi_solvent) gen_colors(rbc, &s->colorer, /**/ flu);
    }
    MC(m::Barrier(s->cart));

    long nsteps = (long)(tend / dt);
    msg_print("will take %ld steps", nsteps);
    if (walls || solids) {
        s->solids0 = false;
        gen(&s->coords, /**/ wall, s);
        dSync();
        if (walls && wall->q.n) UC(wall_gen_ticket(&wall->q, wall->t));
        s->solids0 = solids;
        if (rbcs && multi_solvent) gen_colors(rbc, &s->colorer, /**/ flu);
        run(wall_creation, nsteps, s);
    } else {
        s->solids0 = solids;
        run(            0, nsteps, s);
    }
    /* final strt dump*/
    if (strt_dumps) dump_strt(restart::FINAL, s);
}

void sim_strt(Sim *s) {
    long nsteps = (long)(tend / dt);
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;
    
    /*Q*/
    flu_strt_quants(s->coords, restart::BEGIN, &flu->q);
    flu_build_cells(&flu->q);

    if (rbcs) rbc_strt_quants(s->coords, "rbc.off", restart::BEGIN, &rbc->q);
    dSync();

    if (solids) rig_strt_quants(&s->coords, restart::BEGIN, &rig->q);

    if (walls) wall_strt_quants(s->coords, MAXNWALL, &wall->q);

    /*T*/
    if (rbcs)               UC(rbc_force_gen(rbc->q, &rbc->tt));
    if (walls && wall->q.n) UC(wall_gen_ticket(&wall->q, wall->t));

    MC(m::Barrier(s->cart));
    if (walls) {
        dSync();
        UC(sdf_gen(&s->coords, s->cart, /**/ wall->sdf));
        MC(m::Barrier(s->cart));
    }

    s->solids0 = solids;

    msg_print("will take %ld steps", nsteps - wall_creation);
    run(wall_creation, nsteps, s);

    if (strt_dumps) dump_strt(restart::FINAL, s);
}
