static long get_max_parts_wall(const Coords *c) {
    return numberdensity *
        (xs(c) + 2 * XWM) *
        (ys(c) + 2 * YWM) *
        (zs(c) + 2 * ZWM);
}

static void gen(Time *time, float tw, const Coords *coords, Wall *w, Sim *s) { /* generate */
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    const Opt *opt = &s->opt;
    bool dump_sdf = opt->dump_field;
    long maxp_wall = get_max_parts_wall(coords);
    
    InterWalInfos winfo;
    InterFluInfos finfo;
    InterRbcInfos rinfo;
    InterRigInfos sinfo;

    winfo.active = opt->wall;
    winfo.sdf = w->sdf;
    finfo.q = &flu->q;
    rinfo.active = opt->rbc;
    rinfo.q = &rbc->q;
    sinfo.active = opt->rig;
    sinfo.q = &rig->q;
    sinfo.pi = rig->pininfo;
    sinfo.mass = rig->mass;
    sinfo.empty_pp = opt->rig_empty_pp;
    sinfo.numdensity = numberdensity;
    
    run_eq(time, tw, s);
    if (opt->wall) {
        dSync();
        UC(sdf_gen(coords, s->cart, dump_sdf, /**/ w->sdf));
        MC(m::Barrier(s->cart));
        inter_create_walls(s->cart, maxp_wall, w->sdf, /*io*/ &flu->q, /**/ &w->q);
    }
    inter_freeze(coords, s->cart, winfo, /*io*/ finfo, rinfo, sinfo);
    clear_vel(s);

    if (opt->flucolors) {
        Particle *pp = flu->q.pp;
        int n = flu->q.n;
        int *cc = flu->q.cc;
        inter_color_apply_dev(coords, s->gen_color, n, pp, /*o*/ cc);
    }
}

void sim_gen(Sim *s, const Config *cfg, Time *time, TimeSeg *time_seg) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Wall *wall = &s->wall;
    MeshRead *cell = s->rbc.cell;
    const Opt *opt = &s->opt;
    
    UC(flu_gen_quants(s->coords, numberdensity, s->gen_color, &flu->q));
    UC(flu_build_cells(&flu->q));
    if (opt->fluids)  flu_gen_ids  (s->cart, flu->q.n, &flu->q);
    if (opt->rbc) {
        rbc_gen_quants(s->coords, s->cart, cell, "rbcs-ic.txt", /**/ &rbc->q);
        if (opt->flucolors) gen_colors(rbc, &s->colorer, /**/ flu);
    }
    MC(m::Barrier(s->cart));
    if (opt->wall || opt->rig) {
        s->solids0 = false;
        gen(time, time_seg->wall, s->coords, /**/ wall, s);
        dSync();
        if (opt->wall && wall->q.n) UC(wall_gen_ticket(&wall->q, wall->t));
        s->solids0 = opt->rig;
        if (opt->rbc && opt->flucolors) gen_colors(rbc, &s->colorer, /**/ flu);
        run(cfg, time, time_seg->wall, time_seg->end, s);
    } else {
        s->solids0 = opt->rig;
        run(cfg, time, 0, time_seg->end, s);
    }
    /* final strt dump*/
    if (opt->dump_strt) dump_strt0(RESTART_FINAL, s);
}

void sim_strt(Sim *s, const Config *cfg, Time *time, TimeSeg *time_seg) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;
    MeshRead *cell = s->rbc.cell;
    const Opt *opt = &s->opt;
    bool dump_sdf = opt->dump_field;
    long maxp_wall = get_max_parts_wall(s->coords);

    /*Q*/
    flu_strt_quants(s->coords, RESTART_BEGIN, &flu->q);
    flu_build_cells(&flu->q);

    if (opt->rbc) rbc_strt_quants(s->coords, cell, RESTART_BEGIN, &rbc->q);
    dSync();

    if (opt->rig) rig_strt_quants(s->coords, RESTART_BEGIN, &rig->q);

    if (opt->wall) wall_strt_quants(s->coords, maxp_wall, &wall->q);

    /*T*/
    if (opt->wall && wall->q.n) UC(wall_gen_ticket(&wall->q, wall->t));

    MC(m::Barrier(s->cart));
    if (opt->wall) {
        dSync();
        UC(sdf_gen(s->coords, s->cart, dump_sdf, /**/ wall->sdf));
        MC(m::Barrier(s->cart));
    }

    s->solids0 = opt->rig;
    run(cfg, time, time_seg->wall, time_seg->end, s);
    if (opt->dump_strt) dump_strt0(RESTART_FINAL, s);
}
