
static void set_params(const Config *cfg, float kBT, float dt, const char *name_space, PairParams *p) {
    UC(pair_set_conf(cfg, name_space, p));
    UC(pair_compute_dpd_sigma(kBT, dt, /**/ p));
}

static void ini_flu_exch(Opt opt, Params par, MPI_Comm comm, int3 L, /**/ FluExch *e) {
    int maxd = HSAFETY_FACTOR * par.numdensity;

    UC(eflu_pack_ini(opt.flucolors, L, maxd, /**/ &e->p));
    UC(eflu_comm_ini(opt.flucolors, comm, /**/ &e->c));
    UC(eflu_unpack_ini(opt.flucolors, L, maxd, /**/ &e->u));
}

static void ini_obj_exch(MPI_Comm comm, int3 L, /**/ ObjExch *e) {
    int maxpsolid = MAX_PSOLID_NUM;

    UC(eobj_pack_ini(L, MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid, &e->p));
    UC(eobj_comm_ini(comm, /**/ &e->c));
    UC(eobj_unpack_ini(L, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->u));
    UC(eobj_packf_ini(L, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->pf));
    UC(eobj_unpackf_ini(L, MAX_OBJ_DENSITY, maxpsolid, /**/ &e->uf));
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

static void ini_flu_distr(Opt opt, Params par, MPI_Comm comm, int3 L, /**/ FluDistr *d) {
    float maxdensity = ODSTR_FACTOR * par.numdensity;
    UC(dflu_pack_ini(opt.flucolors, opt.fluids, L, maxdensity, /**/ &d->p));
    UC(dflu_comm_ini(opt.flucolors, opt.fluids, comm, /**/ &d->c));
    UC(dflu_unpack_ini(opt.flucolors, opt.fluids, L, maxdensity, /**/ &d->u));
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
    const char *type;
    UC(den_ini(maxp, /**/ d));

    UC(conf_lookup_string(cfg, "denoutflow.type", &type));
    if (same_str(type, "none")) {
        UC(den_ini_map_none(c, /**/ m));
    }
    else if (same_str(type, "circle")) {
        float R;
        UC(conf_lookup_float(cfg, "denoutflow.R", &R));
        UC(den_ini_map_circle(c, R, /**/ m));
    } else {
        ERR("Unrecognized type <%s>", type);
    }
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

static void ini_flu(const Config *cfg, Opt opt, Params par, MPI_Comm cart, int maxp, int3 L, /**/ Flu *f) {

    UC(flu_ini(opt.flucolors, opt.fluids, L, maxp, &f->q));
    UC(fluforces_bulk_ini(L, maxp, /**/ &f->bulk));
    UC(fluforces_halo_ini(cart, L, /**/ &f->halo));

    UC(ini_flu_distr(opt, par, cart, L, /**/ &f->d));
    UC(ini_flu_exch(opt, par, cart, L, /**/ &f->e));

    UC(Dalloc(&f->ff, maxp));
    EMALLOC(maxp, /**/ &f->ff_hst);

    if (opt.fluss) {
        UC(Dalloc(&f->ss, 6*maxp));
        EMALLOC(6*maxp, /**/ &f->ss_hst);        
    }

    UC(conf_lookup_float(cfg, "flu.mass", &f->mass));
}

static void ini_rbc(const Config *cfg, Opt opt, MPI_Comm cart, int3 L, /**/ Rbc *r) {
    int nv;
    const char *directory = "r";
    UC(mesh_read_ini_off("rbc.off", &r->cell));
    UC(mesh_write_ini_from_mesh(cart, opt.rbcshifttype, r->cell, directory, /**/ &r->mesh_write));

    nv = mesh_read_get_nv(r->cell);
    
    Dalloc(&r->ff, MAX_CELL_NUM * nv);
    UC(triangles_ini(r->cell, /**/ &r->tri));
    UC(rbc_ini(opt.rbcids, r->cell, &r->q));
    UC(ini_rbc_distr(opt.rbcids, nv, cart, L, /**/ &r->d));
    if (opt.dump_rbc_com) UC(rbc_com_ini(nv, MAX_CELL_NUM, /**/ &r->com));
    if (opt.rbcstretch)   UC(rbc_stretch_ini("rbc.stretch", nv, /**/ &r->stretch));
    UC(rbc_params_ini(&r->params));
    UC(rbc_params_set_conf(cfg, r->params));

    UC(rbc_force_ini(r->cell, /**/ &r->force));
    UC(rbc_force_set_conf(r->cell, cfg, r->force));

    UC(conf_lookup_float(cfg, "rbc.mass", &r->mass));
}

static void ini_rig(const Config *cfg, MPI_Comm cart, Opt opt, int maxp, int3 L, /**/ Rig *s) {
    const int4 *tt;
    int nv, nt;

    UC(rig_ini(maxp, &s->q));
    tt = s->q.htt; nv = s->q.nv; nt = s->q.nt;
    UC(mesh_write_ini(cart, opt.rigshifttype, tt, nv, nt, "s", /**/ &s->mesh_write));

    UC(scan_ini(L.x * L.y * L.z, /**/ &s->ws));
    EMALLOC(maxp, &s->ff_hst);
    Dalloc(&s->ff, maxp);

    UC(ini_rig_distr(s->q.nv, cart, L, /**/ &s->d));

    UC(rig_ini_pininfo(&s->pininfo));
    UC(rig_set_pininfo_conf(cfg, s->pininfo));

    UC(conf_lookup_float(cfg, "rig.mass", &s->mass));
}

static void ini_bounce_back(MPI_Comm cart, int maxp, int3 L, Rig *s, /**/ BounceBack *bb) {
    meshbb_ini(maxp, /**/ &bb->d);
    Dalloc(&bb->mm, maxp);

    UC(ini_bb_exch(s->q.nt, s->q.nv, MAX_CELL_NUM, cart, L, /**/ &bb->e));
}

static void ini_wall(const Config *cfg, int3 L, Wall *w) {
    UC(sdf_ini(L, &w->sdf));
    UC(wall_ini_quants(L, &w->q));
    UC(wall_ini_ticket(L, &w->t));
    UC(wvel_ini(&w->vel));
    UC(wvel_set_conf(cfg, w->vel));
    UC(wvel_step_ini(&w->velstep));
}

static void ini_objinter(MPI_Comm cart, int maxp, int3 L, const Opt *opt, /**/ ObjInter *o) {
    int rank;
    MC(m::Comm_rank(cart, &rank));
    UC(ini_obj_exch(cart, L, &o->e));
    if (opt->cnt) cnt_ini(maxp, rank, L, /**/ &o->cnt);
    if (opt->fsi) fsi_ini(rank, L, /**/ &o->fsi);
}

static void read_color_opt(const Config *c, Recolorer *o) {
    int b;
    UC(conf_lookup_bool(c, "recolor.active", &b));
    o->flux_active = b;
    UC(conf_lookup_int(c, "recolor.dir", &o->flux_dir));
}

static int get_shifttype(const Config *c, const char *desc) {
    const char *type;
    UC(conf_lookup_string(c, desc, &type));
    if      (same_str(type, "edge"  )) return EDGE;
    else if (same_str(type, "center")) return CENTER;
    else
        ERR("Unrecognised rbc shift type <%s>", type);
    return -1;
}

static void read_opt(const Config *c, Opt *o) {
    int b;
    const char *s;
    UC(conf_lookup_bool(c, "fsi.active", &b));
    o->fsi = b;
    UC(conf_lookup_bool(c, "cnt.active", &b));
    o->cnt = b;

    UC(conf_lookup_bool(c, "flu.colors", &b));
    o->flucolors = b;
    UC(conf_lookup_bool(c, "flu.ids", &b));
    o->fluids = b;
    UC(conf_lookup_bool(c, "flu.stresses", &b));
    o->fluss = b;

    UC(conf_lookup_bool(c, "rbc.active", &b));
    o->rbc = b;
    UC(conf_lookup_bool(c, "rbc.ids", &b));
    o->rbcids = b;
    UC(conf_lookup_bool(c, "rbc.stretch", &b));
    o->rbcstretch = b;
    o->rbcshifttype = get_shifttype(c, "rbc.shifttype");

    UC(conf_lookup_bool(c, "rig.active", &b));
    o->rig = b;
    UC(conf_lookup_bool(c, "rig.bounce", &b));
    o->rig_bounce = b;
    UC(conf_lookup_bool(c, "rig.empty_pp", &b));
    o->rig_empty_pp = b;
    o->rigshifttype = get_shifttype(c, "rig.shifttype");

    UC(conf_lookup_bool(c, "wall.active", &b));
    o->wall = b;
    
    UC(conf_lookup_bool(c, "outflow.active", &b));
    o->outflow = b;
    UC(conf_lookup_bool(c, "inflow.active", &b));
    o->inflow = b;
    UC(conf_lookup_bool(c, "denoutflow.active", &b));
    o->denoutflow = b;
    UC(conf_lookup_bool(c, "vcon.active", &b));
    o->vcon = b;

    UC(conf_lookup_bool(c, "dump.field", &b));
    o->dump_field = b;
    UC(conf_lookup_float(c, "dump.freq_field", &o->freq_field));
    
    UC(conf_lookup_bool(c, "dump.strt", &b));
    o->dump_strt = b;
    UC(conf_lookup_float(c, "dump.freq_strt", &o->freq_strt));

    UC(conf_lookup_string(c, "dump.strt_base_dump", &s));
    strcpy(o->strt_base_dump, s);

    UC(conf_lookup_string(c, "dump.strt_base_read", &s));
    strcpy(o->strt_base_read, s);

    UC(conf_lookup_bool(c, "dump.parts", &b));
    o->dump_parts = b;
    UC(conf_lookup_float(c, "dump.freq_parts", &o->freq_parts));

    UC(conf_lookup_bool(c, "dump.rbc_com", &b));
    o->dump_rbc_com = b;
    UC(conf_lookup_float(c, "dump.freq_rbc_com", &o->freq_rbc_com));

    UC(conf_lookup_bool(c, "dump.forces", &b));
    o->dump_forces = b;
    
    UC(conf_lookup_int(c, "flu.recolor_freq", &o->recolor_freq));

    UC(conf_lookup_bool(c, "flu.push", &b));
    o->push_flu = b;
    UC(conf_lookup_bool(c, "rbc.push", &b));
    o->push_rbc = b;
    UC(conf_lookup_bool(c, "rig.push", &b));
    o->push_rig = b;
}

static void check_opt(Opt opt) {
    if (opt.dump_rbc_com && !opt.rbcids)
        ERR("Need rbc.ids activated to dump rbc com!");
}

static void coords_log(const Coords *c) {
    msg_print("domain: %d %d %d", xdomain(c), ydomain(c), zdomain(c));
    msg_print("subdomain: [%d:%d][%d:%d][%d:%d]",
              xlo(c), xhi(c), ylo(c), yhi(c), zlo(c), zhi(c));
}

static void ini_pair_params(const Config *cfg, float kBT, float dt, Sim *s) {
    UC(pair_ini(&s->flu.params));
    UC(pair_ini(&s->objinter.cntparams));
    UC(pair_ini(&s->objinter.fsiparams));

    UC(set_params(cfg, kBT, dt, "flu", s->flu.params));
    if (s->opt.cnt) UC(set_params(cfg, kBT, dt, "cnt", s->objinter.cntparams));
    if (s->opt.fsi) UC(set_params(cfg, kBT, dt, "fsi", s->objinter.fsiparams));
}

static void ini_dump(int maxp, MPI_Comm cart, const Coords *c, Opt opt, Dump *d) {
    enum {NPARRAY = 3}; /* flu, rig and rbc */
    EMALLOC(NPARRAY * maxp, &d->pp);
    
    if (opt.dump_field) UC(io_field_ini(cart, c, &d->iofield));
    if (opt.dump_parts) UC(io_rig_ini(&d->iorig));
    UC(io_bop_ini(cart, maxp, &d->bop));
    UC(diag_part_ini("diag.txt", &d->diagpart));

    d->id_bop = d->id_rbc = d->id_rbc_com = d->id_rig_mesh = d->id_strt = 0;
}

static long num_pp_estimate(Params p) {
    int3 L = p.L;
    return L.x * L.y * L.z * p.numdensity;
}

void sim_ini(Config *cfg, MPI_Comm cart, /**/ Time *time, Sim **sim) {
    float dt;
    Sim *s;
    int maxp;
    Params params;
    Opt opt;
    
    EMALLOC(1, sim);
    s = *sim;

    MC(m::Comm_dup(cart, &s->cart));
    datatype::ini();
    UC(coords_ini_conf(s->cart, cfg, /**/ &s->coords));
    UC(coords_log(s->coords));

    params.L = subdomain(s->coords);
    UC(conf_lookup_float(cfg, "glb.kBT",        &params.kBT       ));
    UC(conf_lookup_int  (cfg, "glb.numdensity", &params.numdensity));
    
    maxp = SAFETY_FACTOR_MAXP * num_pp_estimate(params);
    UC(time_step_ini(cfg, &s->time_step));
    UC(time_step_accel_ini(&s->time_step_accel));
    dt = time_step_dt0(s->time_step);
    time_next(time, dt);
    UC(read_opt(cfg, &opt));
    s->opt = opt;
    UC(read_color_opt(cfg, &s->recolorer));
    UC(check_opt(opt));
    UC(ini_pair_params(cfg, params.kBT, dt, s));

    UC(ini_dump(maxp, s->cart, s->coords, opt, /**/ &s->dump));

    UC(bforce_ini(&s->bforce));
    UC(bforce_set_none(/**/ s->bforce));
        
    if (opt.rbc)        UC(ini_rbc(cfg, opt, s->cart, params.L, /**/ &s->rbc));

    if (opt.vcon)       UC(ini_vcon(s->cart, params.L, cfg, /**/ &s->vcon));
    if (opt.outflow)    UC(ini_outflow(s->coords, maxp, cfg, /**/ &s->outflow));
    if (opt.inflow)     UC(ini_inflow (s->coords, params.L, cfg,  /**/ &s->inflow ));
    if (opt.denoutflow) UC(ini_denoutflow(s->coords, maxp, cfg, /**/ &s->denoutflow, &s->mapoutflow));

    if (opt.rbc || opt.rig)
        UC(ini_objinter(s->cart, maxp, params.L, &opt, /**/ &s->objinter));

    if (opt.wall) ini_wall(cfg, params.L, &s->wall);

    UC(ini_flu(cfg, opt, params, s->cart, maxp, params.L, /**/ &s->flu));

    if (opt.flucolors && opt.rbc)
        UC(ini_colorer(s->rbc.q.nv, s->cart, maxp, params.L, /**/ &s->colorer));

    if (opt.rig) {
        UC(ini_rig(cfg, s->cart, opt, maxp, params.L, /**/ &s->rig));

        if (opt.rig_bounce)
            UC(ini_bounce_back(s->cart, maxp, params.L, &s->rig, /**/ &s->bb));
    }

    UC(scheme_restrain_ini(&s->restrain));
    UC(scheme_restrain_set_conf(cfg, s->restrain));

    UC(inter_color_ini(&s->gen_color));
    UC(inter_color_set_conf(cfg, s->gen_color));

    UC(dbg_ini(&s->dbg));
    UC(dbg_set_conf(cfg, s->dbg));

    s->params = params;
    
    MC(MPI_Barrier(s->cart));
}

void time_seg_ini(Config *cfg, /**/ TimeSeg **pq) {
    TimeSeg *q;
    EMALLOC(1, &q);
    UC(conf_lookup_float(cfg, "time.end",  &q->end));
    UC(conf_lookup_float(cfg, "time.wall", &q->wall));
    *pq = q;
}
