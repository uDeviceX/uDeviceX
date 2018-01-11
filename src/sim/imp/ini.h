static void ini_flu_exch(MPI_Comm comm, /**/ FluExch *e) {
    using namespace exch::flu;
    int maxd = HSAFETY_FACTOR * numberdensity;
    
    UC(eflu_pack_ini(maxd, /**/ &e->p));
    UC(eflu_comm_ini(comm, /**/ &e->c));
    UC(eflu_unpack_ini(maxd, /**/ &e->u));
}

static void ini_obj_exch(MPI_Comm comm, /**/ ObjExch *e) {
    using namespace exch::obj;
    int maxpsolid = MAX_PSOLID_NUM;
    
    UC(eobj_pack_ini(MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid, &e->p));
    UC(eobj_comm_ini(comm, /**/ &e->c));
    UC(eobj_unpack_ini(MAX_OBJ_DENSITY, maxpsolid, /**/ &e->u));
    UC(eobj_packf_ini(MAX_OBJ_DENSITY, maxpsolid, /**/ &e->pf));
    UC(eobj_unpackf_ini(MAX_OBJ_DENSITY, maxpsolid, /**/ &e->uf));
}

static void ini_mesh_exch(int nv, int max_m, MPI_Comm comm, /**/ Mexch *e) {
    UC(emesh_pack_ini(nv, max_m, /**/ &e->p));
    UC(emesh_comm_ini(comm, /**/ &e->c));
    UC(emesh_unpack_ini(nv, max_m, /**/ &e->u));
}

static void ini_bb_exch(int nt, int nv, int max_m, MPI_Comm comm, /**/ BBexch *e) {
    UC(ini_mesh_exch(nv, max_m, comm, /**/ e));

    UC(emesh_packm_ini(nt, max_m, /**/ &e->pm));
    UC(emesh_commm_ini(comm, /**/ &e->cm));
    UC(emesh_unpackm_ini(nt, max_m, /**/ &e->um));
}

static void ini_flu_distr(MPI_Comm comm, /**/ FluDistr *d) {
    float maxdensity = ODSTR_FACTOR * numberdensity;
    UC(dflu_pack_ini(maxdensity, /**/ &d->p));
    UC(dflu_comm_ini(comm, /**/ &d->c));
    UC(dflu_unpack_ini(maxdensity, /**/ &d->u));
}

static void ini_rbc_distr(int nv, MPI_Comm comm, /**/ RbcDistr *d) {
    UC(drbc_pack_ini(MAX_CELL_NUM, nv, /**/ &d->p));
    UC(drbc_comm_ini(comm, /**/ &d->c));
    UC(drbc_unpack_ini(MAX_CELL_NUM, nv, /**/ &d->u));
}

static void ini_rig_distr(int nv, MPI_Comm comm, /**/ RigDistr *d) {
    UC(drig_pack_ini(MAX_SOLIDS, nv, /**/ &d->p));
    UC(drig_comm_ini(comm, /**/ &d->c));
    UC(drig_unpack_ini(MAX_SOLIDS, nv, /**/ &d->u));
}

static void ini_vcon(MPI_Comm comm, const Config *cfg, /**/ Vcon *c) {
    int3 L = {XS, YS, ZS};
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

static void ini_outflow(Coords coords, const Config *cfg, Outflow **o) {
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

static void ini_denoutflow(Coords coords, const Config *cfg, DCont **d, DContMap **m) {
    const char *type;
    UC(den_ini(MAX_PART_NUM, /**/ d));

    UC(conf_lookup_string(cfg, "denoutflow.type", &type));
    if (same_str(type, "none")) {
        UC(den_ini_map_none(coords, /**/ m));
    }
    else if (same_str(type, "circle")) {
        float R;
        UC(conf_lookup_float(cfg, "denoutflow.R", &R));
        UC(den_ini_map_circle(coords, R, /**/ m));
    } else {
        ERR("Unrecognized type <%s>", type);
    }
}

static void ini_inflow(Coords coords, const Config *cfg, Inflow **i) {
    /* number of cells */
    int2 nc = make_int2(YS, ZS/2);
    UC(inflow_ini(nc, /**/ i));
    UC(inflow_ini_params_conf(coords, cfg, *i));    
    UC(inflow_ini_velocity(*i));
}

static void ini_colorer(int nv, MPI_Comm comm, /**/ Colorer *c) {
    UC(ini_mesh_exch(nv, MAX_CELL_NUM, comm, &c->e));
    Dalloc(&c->pp, MAX_PART_NUM);
    Dalloc(&c->minext, MAX_CELL_NUM);
    Dalloc(&c->maxext, MAX_CELL_NUM);
}

static void ini_flu(MPI_Comm cart, /**/ Flu *f) {

    flu::ini(&f->q);
    ini(MAX_PART_NUM, /**/ &f->bulkdata);
    ini(cart, /**/ &f->halodata);
    
    UC(ini_flu_distr(cart, /**/ &f->d));
    UC(ini_flu_exch(cart, /**/ &f->e));
    
    UC(Dalloc(&f->ff, MAX_PART_NUM));
    UC(emalloc(MAX_PART_NUM * sizeof(Force), /**/ (void**) &f->ff_hst));
}

static void ini_rbc(MPI_Comm cart, /**/ Rbc *r) {
    Dalloc(&r->ff, MAX_CELL_NUM * RBCnv);
    UC(rbc::main::ini(&r->q));

    UC(ini_rbc_distr(r->q.nv, cart, /**/ &r->d));
    if (rbc_com_dumps) UC(rbc::com::ini(MAX_CELL_NUM, /**/ &r->com));
    if (RBC_STRETCH)   UC(rbc::stretch::ini("rbc.stretch", r->q.nv, /**/ &r->stretch));
}

static void ini_rig(MPI_Comm cart, /**/ Rig *s) {
    rig::ini(&s->q);
    scan::alloc_work(XS * YS * ZS, /**/ &s->ws);
    UC(emalloc(sizeof(&s->ff_hst)*MAX_PART_NUM, (void**) &s->ff_hst));
    Dalloc(&s->ff, MAX_PART_NUM);

    UC(ini_rig_distr(s->q.nv, cart, /**/ &s->d));    
}

static void ini_bounce_back(MPI_Comm cart, Rig *s, /**/ BounceBack *bb) {
    meshbb::ini(MAX_PART_NUM, /**/ &bb->d);
    Dalloc(&bb->mm, MAX_PART_NUM);

    UC(ini_bb_exch(s->q.nt, s->q.nv, MAX_CELL_NUM, cart, /**/ &bb->e));
}

static void ini_wall(Wall *w) {
    ini(&w->sdf);
    wall::alloc_quants(&w->q);
    wall::alloc_ticket(&w->t);
    Wvel *vw = &w->vel;

#if defined(WVEL_HS)
    WvelHS p;
    p.u = WVEL_PAR_U;
    p.h = WVEL_PAR_H;
#else
#if   defined(WVEL_SIN)
    WvelShearSin p;
    p.log_freq = WVEL_LOG_FREQ;
    p.w = WVEL_PAR_W;
    p.half = 1;
#else
    WvelShear p;
    p.half = 0;
#endif
    p.gdot = WVEL_PAR_A;
    p.vdir = 0;

    p.gdir = 0;
    if (WVEL_PAR_Y)
        p.gdir = 1;
    else if (WVEL_PAR_Z)
        p.gdir = 2;
#endif
    ini(p, vw);
}

static void ini_objinter(MPI_Comm cart, /**/ ObjInter *o) {
    int rank;
    MC(m::Comm_rank(cart, &rank));
    UC(ini_obj_exch(cart, &o->e));
    if (contactforces) cnt_ini(rank, /**/ &o->cnt);
    if (fsiforces)     fsi_ini(rank, /**/ &o->fsi);
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
}

void sim_ini(int argc, char **argv, MPI_Comm cart, /**/ Sim **sim) {
    Sim *s;
    UC(emalloc(sizeof(Sim), (void**) sim));
    s = *sim;

    MC(m::Comm_dup(cart, &s->cart));
    
    Config *cfg = s->cfg;    
    datatype::ini();

    UC(conf_ini(&cfg)); s->cfg = cfg;
    UC(conf_read(argc, argv, /**/ cfg));

    UC(read_opt(s->cfg, &s->opt));
    
    UC(coords_ini(s->cart, /**/ &s->coords));
    
    UC(emalloc(3 * MAX_PART_NUM * sizeof(Particle), (void**) &s->pp_dump));
    
    if (rbcs) UC(ini_rbc(s->cart, /**/ &s->rbc));

    if (s->opt.vcon)       UC(ini_vcon(s->cart, s->cfg, /**/ &s->vcon));
    if (s->opt.outflow)    UC(ini_outflow(s->coords, s->cfg, /**/ &s->outflow));
    if (s->opt.inflow)     UC(ini_inflow (s->coords, s->cfg, /**/ &s->inflow ));
    if (s->opt.denoutflow) UC(ini_denoutflow(s->coords, s->cfg, /**/ &s->denoutflow, &s->mapoutflow));
    
    if (rbcs || solids)
        UC(ini_objinter(s->cart, /**/ &s->objinter));        
    
    UC(bop::ini(s->cart, &s->dumpt));

    if (walls) ini_wall(&s->wall);
    
    UC(ini_flu(s->cart, /**/ &s->flu));
   
    if (multi_solvent && rbcs)
        UC(ini_colorer(s->rbc.q.nv, s->cart, /**/ &s->colorer));
    
    if (solids) {
        UC(ini_rig(s->cart, /**/ &s->rig));

        if (sbounce_back)
            UC(ini_bounce_back(s->cart, &s->rig, /**/ &s->bb));
    }

    MC(MPI_Barrier(s->cart));
}
