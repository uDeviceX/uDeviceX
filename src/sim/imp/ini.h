_S_ void set_params(const Config *cfg, float kBT, float dt, const char *name_space, PairParams *p) {
    UC(pair_set_conf(cfg, name_space, p));
    UC(pair_compute_dpd_sigma(kBT, dt, /**/ p));
}

_S_ void ini_flu_exch(const Opt *opt, MPI_Comm comm, int3 L, /**/ FluExch *e) {
    int maxd = HSAFETY_FACTOR * opt->params.numdensity;

    UC(eflu_pack_ini(opt->flucolors, L, maxd, /**/ &e->p));
    UC(eflu_comm_ini(opt->flucolors, comm, /**/ &e->c));
    UC(eflu_unpack_ini(opt->flucolors, L, maxd, /**/ &e->u));
}

_S_ void ini_flu_distr(const Opt *opt, MPI_Comm comm, int3 L, /**/ FluDistr *d) {
    float maxdensity = ODSTR_FACTOR * opt->params.numdensity;
    UC(dflu_pack_ini(opt->flucolors, opt->fluids, L, maxdensity, /**/ &d->p));
    UC(dflu_comm_ini(opt->flucolors, opt->fluids, comm, /**/ &d->c));
    UC(dflu_unpack_ini(opt->flucolors, opt->fluids, L, maxdensity, /**/ &d->u));
    UC(dflu_status_ini(/**/ &d->s));
}

_S_ void ini_vcon(MPI_Comm comm, int3 L, const Config *cfg, /**/ Vcon *c) {
    UC(conf_lookup_int(cfg, "vcon.log_freq", &c->log_freq));
    UC(conf_lookup_int(cfg, "vcon.adjust_freq", &c->adjust_freq));
    UC(conf_lookup_int(cfg, "vcon.sample_freq", &c->sample_freq));

    UC(vcont_ini(comm, L, /**/ &c->vcont));
    UC(vcont_set_conf(cfg, /**/ c->vcont));
}

_S_ void ini_outflow(const Coords *coords, int maxp, const Config *cfg, Outflow **o) {
    UC(outflow_ini(maxp, /**/ o));
    UC(outflow_set_conf(cfg, coords, *o));
}

_S_ void ini_denoutflow(const Coords *c, int maxp, const Config *cfg, DCont **d, DContMap **m) {
    UC(den_ini(maxp, /**/ d));
    UC(den_map_ini(/**/ m));
    UC(den_map_set_conf(cfg, c, *m));
}

_S_ void ini_inflow(const Coords *coords, int3 L, const Config *cfg, Inflow **i) {
    /* number of cells */
    int2 nc;
    nc.x = L.y;
    nc.y = L.z/2;
    UC(inflow_ini(nc, /**/ i));
    UC(inflow_ini_params_conf(coords, cfg, *i));
    UC(inflow_ini_velocity(*i));
}

_S_ void ini_flu(const Config *cfg, const Opt *opt, MPI_Comm cart, int maxp, /**/ Flu *f) {
    int3 L = opt->params.L;
    
    UC(flu_ini(opt->flucolors, opt->fluids, L, maxp, &f->q));
    UC(fluforces_bulk_ini(L, maxp, /**/ &f->bulk));
    UC(fluforces_halo_ini(cart, L, /**/ &f->halo));

    UC(ini_flu_distr(opt, cart, L, /**/ &f->d));
    UC(ini_flu_exch(opt, cart, L, /**/ &f->e));

    UC(Dalloc(&f->ff, maxp));
    EMALLOC(maxp, /**/ &f->ff_hst);

    if (opt->fluss) {
        UC(Dalloc(&f->ss, 6*maxp));
        EMALLOC(6*maxp, /**/ &f->ss_hst);        
    }

    UC(conf_lookup_float(cfg, "flu.mass", &f->mass));
}

_S_ void read_recolor_opt(const Config *c, Recolorer *o) {
    int b;
    UC(conf_lookup_bool(c, "recolor.active", &b));
    o->flux_active = b;
    UC(conf_lookup_int(c, "recolor.dir", &o->flux_dir));
}

_S_ void coords_log(const Coords *c) {
    msg_print("domain: %d %d %d", xdomain(c), ydomain(c), zdomain(c));
    msg_print("subdomain: [%d:%d][%d:%d][%d:%d]",
              xlo(c), xhi(c), ylo(c), yhi(c), zlo(c), zhi(c));
}

_S_ void ini_pair_params(const Config *cfg, float kBT, float dt, Sim *s) {
    UC(pair_ini(&s->flu.params));
    UC(set_params(cfg, kBT, dt, "flu", s->flu.params));
}

_S_ int gsize(int L, int r) {
    return r >= 0 ? L * r : L / r;
}

_S_ int3 grid_size(int3 L, int3 r) {
    int3 N;
    N.x = gsize(L.x, r.x);
    N.y = gsize(L.y, r.y);
    N.z = gsize(L.z, r.z);
    return N;
}

_S_ void ini_sampler(const Coords *c, const Opt *opt, Sampler *s) {
    int3 N, L;
    bool stress, colors;
    stress = opt->fluss;
    colors = opt->flucolors;
    L = subdomain(c);
    N = grid_size(L, opt->sampler_grid_ref);
    
    UC(grid_sampler_data_ini(&s->d));
    UC(grid_sampler_ini(colors, stress, L, N, &s->s));
}

_S_ void ini_dump(int maxp, MPI_Comm cart, const Coords *c, const Opt *opt, Dump *d) {
    enum {NPARRAY = 3}; /* flu, rig and rbc */
    EMALLOC(NPARRAY * maxp, &d->pp);
    
    UC(io_bop_ini(cart, maxp, &d->bop));
    UC(diag_part_ini("diag.txt", &d->diagpart));
    if (opt->dump.field) {
        UC(ini_sampler(c, opt, &d->field_sampler));
        os_mkdir(DUMP_BASE "/h5");
    }

    d->id_bop = d->id_strt = 0;
}

_S_ void ini_time(const Config *cfg, /**/ Time *t) {
    const float t0 = 0;
    UC(conf_lookup_float(cfg, "time.end",  &t->end));
    UC(conf_lookup_float(cfg, "time.wall", &t->eq));
    UC(time_line_ini(t0, &t->t));
    UC(time_step_ini(cfg, &t->step));
    UC(time_step_accel_ini(&t->accel));
}

_S_ void ini_common(const Config *cfg, MPI_Comm cart, /**/ Sim *s) {
    MC(m::Comm_dup(cart, &s->cart));
    UC(coords_ini_conf(s->cart, cfg, /**/ &s->coords));
    UC(coords_log(s->coords));

    UC(dbg_ini(&s->dbg));
    UC(dbg_set_conf(cfg, s->dbg));

    UC(ini_time(cfg, &s->time));
}

_S_ void ini_optional_features(const Config *cfg, const Opt *opt, Sim *s) {
    int maxp = opt_estimate_maxp(opt);
    int3 L = opt->params.L;
    
    if (opt->vcon)       UC(ini_vcon(s->cart, L, cfg, /**/ &s->vcon));
    if (opt->outflow)    UC(ini_outflow(s->coords, maxp, cfg, /**/ &s->outflow));
    if (opt->inflow)     UC(ini_inflow (s->coords, L, cfg,  /**/ &s->inflow ));
    if (opt->denoutflow) UC(ini_denoutflow(s->coords, maxp, cfg, /**/ &s->denoutflow, &s->mapoutflow));    
}

void sim_ini(const Config *cfg, MPI_Comm cart, /**/ Sim **sim) {
    float dt;
    Sim *s;
    int maxp;
    Opt *opt;
    int3 L;
    
    EMALLOC(1, sim);
    s = *sim;
    opt = &s->opt;
    
    UC(ini_common(cfg, cart, /**/ s));
    
    dt = time_step_dt0(s->time.step);
    time_line_advance(dt, s->time.t);

    UC(opt_read(cfg, opt));
    UC(read_recolor_opt(cfg, &s->recolorer));
    UC(opt_check(opt));

    maxp = opt_estimate_maxp(opt);
    L = opt->params.L;

    UC(ini_pair_params(cfg, opt->params.kBT, dt, s));

    UC(ini_dump(maxp, s->cart, s->coords, opt, /**/ &s->dump));

    UC(bforce_ini(&s->bforce));
    UC(bforce_set_conf(cfg, s->bforce));

    UC(ini_flu(cfg, opt, s->cart, maxp, /**/ &s->flu));    
    if (opt->wall) UC(wall_ini(cfg, L, &s->wall));
    
    UC(objects_ini(cfg, opt, s->cart, s->coords, maxp, &s->obj));
    
    UC(obj_inter_ini(cfg, opt, s->cart, dt, maxp, /**/ &s->objinter));

    UC(ini_optional_features(cfg, opt, /**/ s));
    
    UC(scheme_restrain_ini(&s->restrain));
    UC(scheme_restrain_set_conf(cfg, s->restrain));

    UC(inter_color_ini(&s->gen_color));
    UC(inter_color_set_conf(cfg, s->gen_color));

    if (m::is_master(s->cart))
        UC(utils_dump_history(cfg, "conf.history.cfg"));
    
    MC(m::Barrier(s->cart));
}
