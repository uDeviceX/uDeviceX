static void gen_flu(Sim *s) {
    Flu *flu = &s->flu;
    UC(flu_gen_quants(s->coords, s->params.numdensity, s->gen_color, &flu->q));
    UC(flu_build_cells(&flu->q));
    if (s->opt.fluids) flu_gen_ids(s->cart, flu->q.n, &flu->q);
}

static void set_options_flu_only(Sim *s) {
    Opt *o = &s->opt;
    o->rbc = false;
    o->rig = false;
    o->wall = false;
}

void sim_gen_flu(Sim *s) {
    gen_flu(s);
    set_options_flu_only(s);
    dSync();
    MC(m::Barrier(s->cart));
}

static void freeze(const Coords *coords, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wa = &s->wall;
    const Opt *opt = &s->opt;
    bool dump_sdf = opt->dump_field;
    long maxp_wall = get_max_parts_wall(s->params);
    
    InterWalInfos winfo;
    InterFluInfos finfo;
    InterRbcInfos rinfo;
    InterRigInfos sinfo;

    winfo.active = opt->wall;
    winfo.sdf = wa->sdf;
    finfo.q = &flu->q;
    rinfo.active = opt->rbc;
    rinfo.q = &rbc->q;
    sinfo.active = opt->rig;
    sinfo.q = &rig->q;
    sinfo.pi = rig->pininfo;
    sinfo.mass = rig->mass;
    sinfo.empty_pp = opt->rig_empty_pp;
    sinfo.numdensity = s->params.numdensity;
    
    if (opt->wall) {
        dSync();
        UC(sdf_gen(coords, s->cart, dump_sdf, /**/ wa->sdf));
        MC(m::Barrier(s->cart));
        inter_freeze_walls(s->cart, maxp_wall, wa->sdf, /*io*/ &flu->q, /**/ &wa->q);
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

void sim_freeze(const Config *cfg, Sim *s) {
    const Opt *opt = &s->opt;
    MeshRead *cell = s->rbc.cell;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Wall *wall = &s->wall;
    
    if (opt->rbc) {
        rbc_gen_quants(s->coords, s->cart, cell, "rbcs-ic.txt", /**/ &rbc->q);
        if (opt->flucolors) UC(gen_colors(rbc, &s->colorer, /**/ flu));
    }
    if (opt->wall || opt->rig) {
        s->rigids = false;
        gen(s->time.wall, s->coords, /**/ wall, s);
        dSync();
    }
    if (opt->wall && wall->q.n) UC(wall_gen_ticket(&wall->q, wall->t));
    if (opt->rbc && opt->flucolors) UC(gen_colors(rbc, &s->colorer, /**/ flu));
}

void sim_restart(Sim *s){}
