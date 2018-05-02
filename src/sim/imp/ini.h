static void set_params(const Config *cfg, float kBT, float dt, const char *name_space, PairParams *p) {
    UC(pair_set_conf(cfg, name_space, p));
    UC(pair_compute_dpd_sigma(kBT, dt, /**/ p));
}

static void ini_flu_exch(const Opt *opt, MPI_Comm comm, int3 L, /**/ FluExch *e) {
    int maxd = HSAFETY_FACTOR * opt->params.numdensity;

    UC(eflu_pack_ini(opt->flucolors, L, maxd, /**/ &e->p));
    UC(eflu_comm_ini(opt->flucolors, comm, /**/ &e->c));
    UC(eflu_unpack_ini(opt->flucolors, L, maxd, /**/ &e->u));
}

static void ini_mesh_exch(int nv, int max_m, MPI_Comm comm, int3 L, /**/ Mexch *e) {
    UC(emesh_pack_ini(L, nv, max_m, /**/ &e->p));
    UC(emesh_comm_ini(comm, /**/ &e->c));
    UC(emesh_unpack_ini(L, nv, max_m, /**/ &e->u));
}

static void ini_bb_exch(int nt, int nv, int max_m, MPI_Comm comm, int3 L, /**/ BBexch *e) {
    UC(ini_mesh_exch(nv, max_m, comm, L, /**/ e));

    UC(emesh_packm_ini(nt, max_m, /**/ &e->pm));
    UC(emesh_commm_ini(comm, /**/ &e->cm));
    UC(emesh_unpackm_ini(nt, max_m, /**/ &e->um));
}

static void ini_flu_distr(const Opt *opt, MPI_Comm comm, int3 L, /**/ FluDistr *d) {
    float maxdensity = ODSTR_FACTOR * opt->params.numdensity;
    UC(dflu_pack_ini(opt->flucolors, opt->fluids, L, maxdensity, /**/ &d->p));
    UC(dflu_comm_ini(opt->flucolors, opt->fluids, comm, /**/ &d->c));
    UC(dflu_unpack_ini(opt->flucolors, opt->fluids, L, maxdensity, /**/ &d->u));
    UC(dflu_status_ini(/**/ &d->s));
}

static void ini_rbc_distr(bool ids, int nv, MPI_Comm comm, int3 L, /**/ RbcDistr *d) {
    UC(drbc_pack_ini(ids, L, MAX_CELL_NUM, nv, /**/ &d->p));
    UC(drbc_comm_ini(ids, comm, /**/ &d->c));
    UC(drbc_unpack_ini(ids, L, MAX_CELL_NUM, nv, /**/ &d->u));
}

static void ini_rig_distr(int nv, MPI_Comm comm, int3 L, /**/ RigDistr *d) {
    UC(drig_pack_ini(L, MAX_SOLIDS, nv, /**/ &d->p));
    UC(drig_comm_ini(comm, /**/ &d->c));
    UC(drig_unpack_ini(L,MAX_SOLIDS, nv, /**/ &d->u));
}

static void ini_vcon(MPI_Comm comm, int3 L, const Config *cfg, /**/ Vcon *c) {
    UC(conf_lookup_int(cfg, "vcon.log_freq", &c->log_freq));
    UC(conf_lookup_int(cfg, "vcon.adjust_freq", &c->adjust_freq));
    UC(conf_lookup_int(cfg, "vcon.sample_freq", &c->sample_freq));

    UC(vcont_ini(comm, L, /**/ &c->vcont));
    UC(vcont_set_conf(cfg, /**/ c->vcont));
}

static void ini_outflow(const Coords *coords, int maxp, const Config *cfg, Outflow **o) {
    UC(outflow_ini(maxp, /**/ o));
    UC(outflow_set_conf(cfg, coords, *o));
}

static void ini_denoutflow(const Coords *c, int maxp, const Config *cfg, DCont **d, DContMap **m) {
    UC(den_ini(maxp, /**/ d));
    UC(den_map_ini(/**/ m));
    UC(den_map_set_conf(cfg, c, *m));
}

static void ini_inflow(const Coords *coords, int3 L, const Config *cfg, Inflow **i) {
    /* number of cells */
    int2 nc = make_int2(L.y, L.z/2);
    UC(inflow_ini(nc, /**/ i));
    UC(inflow_ini_params_conf(coords, cfg, *i));
    UC(inflow_ini_velocity(*i));
}

static void ini_colorer(int nv, MPI_Comm comm, int maxp, int3 L, /**/ Colorer *c) {
    UC(ini_mesh_exch(nv, MAX_CELL_NUM, comm, L, &c->e));
    Dalloc(&c->pp, maxp);
    Dalloc(&c->minext, maxp);
    Dalloc(&c->maxext, maxp);
}

static void ini_flu(const Config *cfg, const Opt *opt, MPI_Comm cart, int maxp, /**/ Flu *f) {
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

static void ini_rbc(const Config *cfg, const OptMbr *opt, MPI_Comm cart, int3 L, /**/ Rbc *r) {
    int nv;
    const char *directory = "r";
    UC(mesh_read_ini_off("rbc.off", &r->cell));
    UC(mesh_write_ini_from_mesh(cart, opt->shifttype, r->cell, directory, /**/ &r->mesh_write));

    nv = mesh_read_get_nv(r->cell);
    
    Dalloc(&r->ff, MAX_CELL_NUM * nv);
    UC(triangles_ini(r->cell, /**/ &r->tri));
    UC(rbc_ini(MAX_CELL_NUM, opt->ids, r->cell, &r->q));
    UC(ini_rbc_distr(opt->ids, nv, cart, L, /**/ &r->d));
    if (opt->dump_com) UC(rbc_com_ini(nv, MAX_CELL_NUM, /**/ &r->com));
    if (opt->stretch)   UC(rbc_stretch_ini("rbc.stretch", nv, /**/ &r->stretch));
    UC(rbc_params_ini(&r->params));
    UC(rbc_params_set_conf(cfg, r->params));

    UC(rbc_force_ini(r->cell, /**/ &r->force));
    UC(rbc_force_set_conf(r->cell, cfg, r->force));

    UC(conf_lookup_float(cfg, "rbc.mass", &r->mass));
}

static void ini_rig(const Config *cfg, MPI_Comm cart, const OptRig *opt, int maxp, int3 L, /**/ Rig *s) {
    UC(mesh_read_ini_ply("rig.ply", &s->mesh));
    UC(mesh_write_ini_from_mesh(cart, opt->shifttype, s->mesh, "s", /**/ &s->mesh_write));
    
    UC(rig_ini(MAX_SOLIDS, maxp, s->mesh, &s->q));


    EMALLOC(maxp, &s->ff_hst);
    Dalloc(&s->ff, maxp);

    UC(ini_rig_distr(s->q.nv, cart, L, /**/ &s->d));

    UC(rig_pininfo_ini(&s->pininfo));
    UC(rig_pininfo_set_conf(cfg, s->pininfo));

    UC(conf_lookup_float(cfg, "rig.mass", &s->mass));
}

static void ini_bounce_back(MPI_Comm cart, int maxp, int3 L, Rig *s, /**/ BounceBack *bb) {
    UC(meshbb_ini(maxp, /**/ &bb->d));
    Dalloc(&bb->mm, maxp);

    UC(ini_bb_exch(s->q.nt, s->q.nv, MAX_CELL_NUM, cart, L, /**/ &bb->e));
}

static void read_recolor_opt(const Config *c, Recolorer *o) {
    int b;
    UC(conf_lookup_bool(c, "recolor.active", &b));
    o->flux_active = b;
    UC(conf_lookup_int(c, "recolor.dir", &o->flux_dir));
}

static void coords_log(const Coords *c) {
    msg_print("domain: %d %d %d", xdomain(c), ydomain(c), zdomain(c));
    msg_print("subdomain: [%d:%d][%d:%d][%d:%d]",
              xlo(c), xhi(c), ylo(c), yhi(c), zlo(c), zhi(c));
}

static void ini_pair_params(const Config *cfg, float kBT, float dt, Sim *s) {
    UC(pair_ini(&s->flu.params));
    UC(set_params(cfg, kBT, dt, "flu", s->flu.params));
}

static int gsize(int L, int r) {
    return r >= 0 ? L * r : L / r;
}

static int3 grid_size(int3 L, int3 r) {
    int3 N;
    N.x = gsize(L.x, r.x);
    N.y = gsize(L.y, r.y);
    N.z = gsize(L.z, r.z);
    return N;
}

static void ini_sampler(const Coords *c, const Opt *opt, Sampler *s) {
    int3 N, L;
    bool stress, colors;
    stress = opt->fluss;
    colors = opt->flucolors;
    L = subdomain(c);
    N = grid_size(L, opt->sampler_grid_ref);
    
    UC(grid_sampler_data_ini(&s->d));
    UC(grid_sampler_ini(colors, stress, L, N, &s->s));
}

static void ini_dump(int maxp, MPI_Comm cart, const Coords *c, const Opt *opt, Dump *d) {
    enum {NPARRAY = 3}; /* flu, rig and rbc */
    EMALLOC(NPARRAY * maxp, &d->pp);
    
    if (opt->dump_parts) UC(io_rig_ini(&d->iorig));
    UC(io_bop_ini(cart, maxp, &d->bop));
    UC(diag_part_ini("diag.txt", &d->diagpart));
    if (opt->dump_field) {
        UC(ini_sampler(c, opt, &d->field_sampler));
        os_mkdir(DUMP_BASE "/h5");
    }

    d->id_bop = d->id_rbc = d->id_rbc_com = d->id_rig_mesh = d->id_strt = 0;
}

static void ini_time(const Config *cfg, /**/ Time *t) {
    const float t0 = 0;
    UC(conf_lookup_float(cfg, "time.end",  &t->end));
    UC(conf_lookup_float(cfg, "time.wall", &t->eq));
    UC(time_line_ini(t0, &t->t));
    UC(time_step_ini(cfg, &t->step));
    UC(time_step_accel_ini(&t->accel));
}

static void ini_common(const Config *cfg, MPI_Comm cart, /**/ Sim *s) {
    MC(m::Comm_dup(cart, &s->cart));
    UC(coords_ini_conf(s->cart, cfg, /**/ &s->coords));
    UC(coords_log(s->coords));

    UC(dbg_ini(&s->dbg));
    UC(dbg_set_conf(cfg, s->dbg));

    UC(ini_time(cfg, &s->time));
}

static void ini_optional_features(const Config *cfg, const Opt *opt, Sim *s) {
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
    if (opt->rbc.active)  UC(ini_rbc(cfg, &opt->rbc, s->cart, L, /**/ &s->rbc));
    if (opt->rig.active)  UC(ini_rig(cfg, s->cart, &opt->rig, maxp, L, /**/ &s->rig));
    if (opt->wall) UC(wall_ini(cfg, L, &s->wall));

    UC(objects_ini(cfg, opt, cart, s->coords, maxp, &s->obj));
    
    UC(obj_inter_ini(cfg, opt, s->cart, dt, maxp, /**/ &s->objinter));
    
    if (opt->flucolors && opt->rbc.active)
        UC(ini_colorer(s->rbc.q.nv, s->cart, maxp, L, /**/ &s->colorer));

    if (opt->rig.active && opt->rig.bounce)
        UC(ini_bounce_back(s->cart, maxp, L, &s->rig, /**/ &s->bb));

    UC(ini_optional_features(cfg, opt, /**/ s));
    
    UC(scheme_restrain_ini(&s->restrain));
    UC(scheme_restrain_set_conf(cfg, s->restrain));

    UC(inter_color_ini(&s->gen_color));
    UC(inter_color_set_conf(cfg, s->gen_color));

    if (m::is_master(s->cart))
        UC(utils_dump_history(cfg, "conf.history.cfg"));
    
    MC(m::Barrier(s->cart));
}
