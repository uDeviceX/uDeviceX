static long get_max_parts_wall(Params params) {
    int3 L = params.L;
    return params.numdensity *
        (L.x + 2 * XWM) *
        (L.y + 2 * YWM) *
        (L.z + 2 * ZWM);
}

static void gen_flu(Sim *s) {
    Flu *flu = &s->flu;
    UC(flu_gen_quants(s->coords, s->params.numdensity, s->gen_color, &flu->q));
    UC(flu_build_cells(&flu->q));
    if (s->opt.fluids) flu_gen_ids(s->cart, flu->q.n, &flu->q);
}

static void gen_rbc(Sim *s) {
    MeshRead *cell = s->rbc.cell;
    Rbc *rbc = &s->rbc;
    const Opt *opt = &s->opt;
    if (opt->rbc) {
        rbc_gen_quants(s->coords, s->cart, cell, "rbcs-ic.txt", /**/ &rbc->q);
    }
}

static void gen_wall(Sim *s) {
    Flu *flu = &s->flu;
    Wall *w = &s->wall;
    bool dump_sdf = s->opt.dump_field;
    long maxp_wall = get_max_parts_wall(s->params);
    
    if (s->opt.wall) {
        UC(sdf_gen(s->coords, s->cart, dump_sdf, /**/ w->sdf));
        MC(m::Barrier(s->cart));
        inter_freeze_walls(s->cart, maxp_wall, w->sdf, /*io*/ &flu->q, /**/ &w->q);
    }
}

static void freeze(Sim *s) { /* generate */
    Flu *flu = &s->flu;
    const Opt *opt = &s->opt;
    
    InterWalInfos winfo = get_winfo(s);
    InterFluInfos finfo = get_finfo(s);
    InterRbcInfos rinfo = get_rinfo(s);
    InterRigInfos sinfo = get_sinfo(s);

    UC(gen_rbc(s));
    UC(gen_wall(s));
    dSync();    
    UC(inter_freeze(s->coords, s->cart, winfo, /*io*/ finfo, rinfo, sinfo));
    clear_vel(s);

    if (opt->flucolors) {
        Particle *pp = flu->q.pp;
        int n = flu->q.n;
        int *cc = flu->q.cc;
        inter_color_apply_dev(s->coords, s->gen_color, n, pp, /*o*/ cc);
    }
}

void sim_gen(Sim *s) {
    Wall *wall = &s->wall;
    const Opt *opt = &s->opt;
    float tstart = 0;
    s->equilibrating = true;

    UC(gen_flu(s));

    MC(m::Barrier(s->cart));

    run(tstart, s->time.eq, s);

    freeze(/**/ s);
    dSync();

    if (opt->wall && wall->q.n)     UC(wall_gen_ticket(&wall->q, wall->t));
    if (opt->rbc && opt->flucolors) UC(gen_colors(s));

    tstart = s->time.eq;
    pre_run(s);
    run(tstart, s->time.end, s);

    /* final strt dump*/
    if (opt->dump_strt) dump_strt_final(s);
}

void sim_strt(Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;
    MeshRead *cell = s->rbc.cell;
    const Opt *opt = &s->opt;
    bool dump_sdf = opt->dump_field;
    long maxp_wall = get_max_parts_wall(s->params);
    const char *base_strt_read = opt->strt_base_read;

    /*Q*/
    flu_strt_quants(s->cart, base_strt_read, RESTART_BEGIN, &flu->q);
    if (opt->rbc)   rbc_strt_quants(s->cart, base_strt_read, cell, RESTART_BEGIN, &rbc->q);
    if (opt->rig)   rig_strt_quants(s->cart, base_strt_read,       RESTART_BEGIN, &rig->q);
    if (opt->wall) wall_strt_quants(s->cart, base_strt_read, maxp_wall,          &wall->q);

    /*T*/
    flu_build_cells(&flu->q);
    if (opt->wall && wall->q.n) UC(wall_gen_ticket(&wall->q, wall->t));

    MC(m::Barrier(s->cart));
    if (opt->wall) {
        dSync();
        UC(sdf_gen(s->coords, s->cart, dump_sdf, /**/ wall->sdf));
        MC(m::Barrier(s->cart));
    }

    pre_run(s);
    run(s->time.eq, s->time.end, s);
    if (opt->dump_strt) dump_strt_final(s);
}
