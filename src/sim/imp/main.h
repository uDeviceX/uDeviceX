static long get_max_parts_wall(const Coords *c) {
    return numberdensity *
        (xs(c) + 2 * XWM) *
        (ys(c) + 2 * YWM) *
        (zs(c) + 2 * ZWM);
}

static void gen(const Coords *coords, Wall *w, Sim *s) { /* generate */
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    bool dump_sdf = s->opt.dump_field;
    long maxp_wall = get_max_parts_wall(coords);
    
    run_eq(wall_creation, s);
    if (walls) {
        dSync();
        UC(sdf_gen(coords, s->cart, dump_sdf, /**/ w->sdf));
        MC(m::Barrier(s->cart));
        inter_create_walls(s->cart, maxp_wall, w->sdf, /*io*/ &flu->q, /**/ &w->q);
    }
    inter_freeze(coords, s->cart, w->sdf, /*io*/ &flu->q, /**/ &rig->q, &rbc->q);
    clear_vel(s);

    if (multi_solvent) {
        Particle *pp = flu->q.pp;
        int n = flu->q.n;
        int *cc = flu->q.cc;
        inter_color_apply_dev(coords, s->gen_color, n, pp, /*o*/ cc);
    }
}

void sim_gen(Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Wall *wall = &s->wall;
    OffRead *cell = s->rbc.cell;
    
    UC(flu_gen_quants(s->coords, s->gen_color, &flu->q));
    UC(flu_build_cells(&flu->q));
    if (global_ids)  flu_gen_ids  (s->cart, flu->q.n, &flu->q);
    if (rbcs) {
        rbc_gen_quants(s->coords, s->cart, cell, "rbcs-ic.txt", /**/ &rbc->q);
        rbc_force_gen(rbc->q, &rbc->tt);

        if (multi_solvent) gen_colors(rbc, &s->colorer, /**/ flu);
    }
    MC(m::Barrier(s->cart));

    long nsteps = (long)(tend / dt);
    msg_print("will take %ld steps", nsteps);
    if (walls || solids) {
        s->solids0 = false;
        gen(s->coords, /**/ wall, s);
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
    if (strt_dumps) dump_strt(RESTART_FINAL, s);
}

void sim_strt(Sim *s) {
    long nsteps = (long)(tend / dt);
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;
    OffRead *cell = s->rbc.cell;
    bool dump_sdf = s->opt.dump_field;
    long maxp_wall = get_max_parts_wall(s->coords);
    
    /*Q*/
    flu_strt_quants(s->coords, RESTART_BEGIN, &flu->q);
    flu_build_cells(&flu->q);

    if (rbcs) rbc_strt_quants(s->coords, cell, RESTART_BEGIN, &rbc->q);
    dSync();

    if (solids) rig_strt_quants(s->coords, RESTART_BEGIN, &rig->q);

    if (walls) wall_strt_quants(s->coords, maxp_wall, &wall->q);

    /*T*/
    if (rbcs)               UC(rbc_force_gen(rbc->q, &rbc->tt));
    if (walls && wall->q.n) UC(wall_gen_ticket(&wall->q, wall->t));

    MC(m::Barrier(s->cart));
    if (walls) {
        dSync();
        UC(sdf_gen(s->coords, s->cart, dump_sdf, /**/ wall->sdf));
        MC(m::Barrier(s->cart));
    }

    s->solids0 = solids;

    msg_print("will take %ld steps", nsteps - wall_creation);
    run(wall_creation, nsteps, s);

    if (strt_dumps) dump_strt(RESTART_FINAL, s);
}
