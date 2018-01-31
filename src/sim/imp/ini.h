static void ini_flu_exch(MPI_Comm comm, int3 L, /**/ FluExch *e) {
    int maxd = HSAFETY_FACTOR * numberdensity;

    UC(eflu_pack_ini(L, maxd, /**/ &e->p));
    UC(eflu_comm_ini(comm, /**/ &e->c));
    UC(eflu_unpack_ini(L, maxd, /**/ &e->u));
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

static void ini_flu_distr(MPI_Comm comm, int3 L, /**/ FluDistr *d) {
    float maxdensity = ODSTR_FACTOR * numberdensity;
    UC(dflu_pack_ini(L, maxdensity, /**/ &d->p));
    UC(dflu_comm_ini(comm, /**/ &d->c));
    UC(dflu_unpack_ini(L, maxdensity, /**/ &d->u));
    UC(dflu_status_ini(/**/ &d->s));
}

static void ini_rbc_distr(int nv, MPI_Comm comm, int3 L, /**/ RbcDistr *d) {
    UC(drbc_pack_ini(L, MAX_CELL_NUM, nv, /**/ &d->p));
    UC(drbc_comm_ini(comm, /**/ &d->c));
    UC(drbc_unpack_ini(L, MAX_CELL_NUM, nv, /**/ &d->u));
}

static void ini_rig_distr(int nv, MPI_Comm comm, int3 L, /**/ RigDistr *d) {
    UC(drig_pack_ini(L, MAX_SOLIDS, nv, /**/ &d->p));
    UC(drig_comm_ini(comm, /**/ &d->c));
    UC(drig_unpack_ini(L,MAX_SOLIDS, nv, /**/ &d->u));
}

static void ini_vcon(MPI_Comm comm, int3 L, const Config *cfg, /**/ Vcon *c) {
    const char *type;
    float3 U;
    float factor;
    PidVCont *vc;

    UC(conf_lookup_int(cfg, "vcon.log_freq", &c->log_freq));
    UC(conf_lookup_int(cfg, "vcon.adjust_freq", &c->adjust_freq));
    UC(conf_lookup_int(cfg, "vcon.sample_freq", &c->sample_freq));

    UC(conf_lookup_string(cfg, "vcon.type", &type));
    UC(conf_lookup_float3(cfg, "vcon.U", &U));
    UC(conf_lookup_float(cfg, "vcon.factor", &factor));

    UC(vcont_ini(comm, L, U, factor, /**/ &c->vcont));
    vc = c->vcont;

    if      (same_str(type, "cart"))
        UC(vcon_set_cart(/**/ vc));
    else if (same_str(type, "rad"))
        UC(vcon_set_radial(/**/ vc));
    else
        ERR("Unrecognised type <%s>", type);
}

static void ini_outflow(const Coords *coords, const Config *cfg, Outflow **o) {
    UC(ini(MAX_PART_NUM, /**/ o));
    const char *type;
    UC(conf_lookup_string(cfg, "outflow.type", &type));

    if      (same_str(type, "circle")) {
        float3 center;
        float R;
        UC(conf_lookup_float(cfg, "outflow.R", &R));
        UC(conf_lookup_float3(cfg, "outflow.center", &center));

        ini_params_circle(coords, center, R, /**/ *o);
    }
    else if (same_str(type, "plate")) {
        int dir;
        float r0;
        UC(conf_lookup_int(cfg, "outflow.direction", &dir));
        UC(conf_lookup_float(cfg, "outflow.position", &r0));
        ini_params_plate(coords, dir, r0, /**/ *o);
    }
    else {
        ERR("Unrecognized type <%s>", type);
    }
}

static void ini_denoutflow(const Coords *c, const Config *cfg, DCont **d, DContMap **m) {
    const char *type;
    UC(den_ini(MAX_PART_NUM, /**/ d));

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

static void ini_inflow(const Coords *coords, const Config *cfg, Inflow **i) {
    /* number of cells */
    int2 nc = make_int2(YS, ZS/2);
    UC(inflow_ini(nc, /**/ i));
    UC(inflow_ini_params_conf(coords, cfg, *i));
    UC(inflow_ini_velocity(*i));
}

static void ini_colorer(int nv, MPI_Comm comm, int3 L, /**/ Colorer *c) {
    UC(ini_mesh_exch(nv, MAX_CELL_NUM, comm, L, &c->e));
    Dalloc(&c->pp, MAX_PART_NUM);
    Dalloc(&c->minext, MAX_CELL_NUM);
    Dalloc(&c->maxext, MAX_CELL_NUM);
}

static void ini_flu(MPI_Comm cart, int3 L, /**/ Flu *f) {

    UC(flu_ini(L, MAX_PART_NUM, &f->q));
    UC(fluforces_bulk_ini(L, MAX_PART_NUM, /**/ &f->bulk));
    UC(fluforces_halo_ini(cart, L, /**/ &f->halo));

    UC(ini_flu_distr(cart, L, /**/ &f->d));
    UC(ini_flu_exch(cart, L, /**/ &f->e));

    UC(Dalloc(&f->ff, MAX_PART_NUM));
    UC(emalloc(MAX_PART_NUM * sizeof(Force), /**/ (void**) &f->ff_hst));
}

static void ini_rbc(const Config *cfg, MPI_Comm cart, int3 L, /**/ Rbc *r) {
    int nv;
    const char *directory = "r";
    UC(off_read("rbc.off", &r->cell));
    UC(mesh_write_ini_off(r->cell, directory, /**/ &r->mesh_write));

    nv = off_get_nv(r->cell);

    Dalloc(&r->ff, MAX_CELL_NUM * nv);
    UC(rbc_ini(r->cell, &r->q));

    UC(ini_rbc_distr(r->q.nv, cart, L, /**/ &r->d));
    if (rbc_com_dumps) UC(rbc_com_ini(MAX_CELL_NUM, /**/ &r->com));
    if (RBC_STRETCH)   UC(rbc_stretch_ini("rbc.stretch", r->q.nv, /**/ &r->stretch));

    UC(rbc_params_ini(&r->params));
    UC(rbc_params_set_conf(cfg, r->params));
}

static void ini_rig(MPI_Comm cart, int3 L, /**/ Rig *s) {
    const int4 *tt;
    int nv, nt;

    UC(rig_ini(&s->q));
    tt = s->q.htt; nv = s->q.nv; nt = s->q.nt;
    UC(mesh_write_ini(tt, nv, nt, "s", /**/ &s->mesh_write));

    UC(scan_work_ini(XS * YS * ZS, /**/ &s->ws));
    UC(emalloc(sizeof(&s->ff_hst)*MAX_PART_NUM, (void**) &s->ff_hst));
    Dalloc(&s->ff, MAX_PART_NUM);

    UC(ini_rig_distr(s->q.nv, cart, L, /**/ &s->d));
}

static void ini_bounce_back(MPI_Comm cart, int3 L, Rig *s, /**/ BounceBack *bb) {
    meshbb_ini(MAX_PART_NUM, /**/ &bb->d);
    Dalloc(&bb->mm, MAX_PART_NUM);

    UC(ini_bb_exch(s->q.nt, s->q.nv, MAX_CELL_NUM, cart, L, /**/ &bb->e));
}

static void ini_wall(const Config *cfg, int3 L, Wall *w) {
    UC(sdf_ini(L, &w->sdf));
    UC(wall_ini_quants(L, &w->q));
    UC(wall_ini_ticket(&w->t));
    UC(wvel_ini(&w->vel));
    UC(wvel_set_conf(cfg, w->vel));
}

static void ini_objinter(MPI_Comm cart, int3 L, /**/ ObjInter *o) {
    int rank;
    MC(m::Comm_rank(cart, &rank));
    UC(ini_obj_exch(cart, L, &o->e));
    if (contactforces) cnt_ini(MAX_PART_NUM, rank, L, /**/ &o->cnt);
    if (fsiforces)     fsi_ini(rank, L, /**/ &o->fsi);
}

static void read_opt(const Config *c, Opt *o) {
    int b;
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
    UC(conf_lookup_int(c, "dump.freq_field", &o->freq_field));

    UC(conf_lookup_bool(c, "dump.parts", &b));
    o->dump_parts = b;
    UC(conf_lookup_int(c, "dump.freq_parts", &o->freq_parts));

}

static void coords_log(const Coords *c) {
    msg_print("domain: %d %d %d", xdomain(c), ydomain(c), zdomain(c));
    msg_print("subdomain: [%d:%d][%d:%d][%d:%d]",
              xlo(c), xhi(c), ylo(c), yhi(c), zlo(c), zhi(c));
}

void sim_ini(int argc, char **argv, MPI_Comm cart, /**/ Sim **sim) {
    Sim *s;
    UC(emalloc(sizeof(Sim), (void**) sim));
    s = *sim;

    // TODO this will be runtime
    s->L = make_int3(XS, YS, ZS);
    
    MC(m::Comm_dup(cart, &s->cart));

    Config *cfg = s->cfg;
    datatype::ini();

    UC(conf_ini(&cfg)); s->cfg = cfg;
    UC(conf_read(argc, argv, /**/ cfg));

    UC(read_opt(s->cfg, &s->opt));

    UC(coords_ini(s->cart, XS, YS, ZS, /**/ &s->coords));
    UC(coords_log(s->coords));

    UC(emalloc(3 * MAX_PART_NUM * sizeof(Particle), (void**) &s->pp_dump));

    if (rbcs) UC(ini_rbc(cfg, s->cart, s->L, /**/ &s->rbc));

    if (s->opt.vcon)       UC(ini_vcon(s->cart, s->L, s->cfg, /**/ &s->vcon));
    if (s->opt.outflow)    UC(ini_outflow(s->coords, s->cfg, /**/ &s->outflow));
    if (s->opt.inflow)     UC(ini_inflow (s->coords, s->cfg, /**/ &s->inflow ));
    if (s->opt.denoutflow) UC(ini_denoutflow(s->coords, s->cfg, /**/ &s->denoutflow, &s->mapoutflow));

    if (rbcs || solids)
        UC(ini_objinter(s->cart, s->L, /**/ &s->objinter));

    UC(bop_ini(s->cart, MAX_PART_NUM, &s->dumpt));

    if (walls) ini_wall(cfg, s->L, &s->wall);

    UC(ini_flu(s->cart, s->L, /**/ &s->flu));

    if (multi_solvent && rbcs)
        UC(ini_colorer(s->rbc.q.nv, s->cart, s->L, /**/ &s->colorer));

    if (solids) {
        UC(ini_rig(s->cart, s->L, /**/ &s->rig));

        if (sbounce_back)
            UC(ini_bounce_back(s->cart, s->L, &s->rig, /**/ &s->bb));
    }

    UC(scheme_restrain_ini(&s->restrain));
    UC(scheme_restrain_set_conf(s->cfg, s->restrain));

    UC(inter_color_ini(&s->gen_color));
    UC(inter_color_set_conf(s->cfg, s->gen_color));

    UC(dbg_ini(&s->dbg));
    UC(dbg_set_conf(s->cfg, s->dbg));

    MC(MPI_Barrier(s->cart));
}
