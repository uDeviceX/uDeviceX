static void ini_flu_exch(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ FluExch *e) {
    using namespace exch::flu;
    int maxd = HSAFETY_FACTOR * numberdensity;
    
    UC(ini(maxd, /**/ &e->p));
    UC(ini(comm, /*io*/ tg, /**/ &e->c));
    UC(ini(maxd, /**/ &e->u));
}

static void ini_obj_exch(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ ObjExch *e) {
    using namespace exch::obj;
    int maxpsolid = MAX_PSOLID_NUM;
    
    UC(ini(MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid, &e->p));
    UC(ini(comm, /*io*/ tg, /**/ &e->c));
    UC(ini(MAX_OBJ_DENSITY, maxpsolid, /**/ &e->u));
    UC(ini(MAX_OBJ_DENSITY, maxpsolid, /**/ &e->pf));
    UC(ini(MAX_OBJ_DENSITY, maxpsolid, /**/ &e->uf));
}

static void ini_mesh_exch(int nv, int max_m, MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Mexch *e) {
    using namespace exch::mesh;
    UC(ini(nv, max_m, /**/ &e->p));
    UC(ini(comm, /*io*/ &tag_gen, /**/ &e->c));
    UC(ini(nv, max_m, /**/ &e->u));
}

static void ini_bb_exch(int nt, int nv, int max_m, MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ BBexch *e) {
    UC(ini_mesh_exch(nv, max_m, comm, /*io*/ tg, /**/ e));

    using namespace exch::mesh;
    UC(ini(nt, max_m, /**/ &e->pm));
    UC(ini(comm, /*io*/ tg, /**/ &e->cm));
    UC(ini(nt, max_m, /**/ &e->um));
}

static void ini_flu_distr(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ FluDistr *d) {
    using namespace distr::flu;
    float maxdensity = ODSTR_FACTOR * numberdensity;
    UC(ini(maxdensity, /**/ &d->p));
    UC(ini(comm, /**/ tg, /**/ &d->c));
    UC(ini(maxdensity, /**/ &d->u));
}

static void ini_rbc_distr(int nv, MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ RbcDistr *d) {
    using namespace distr::rbc;
    UC(ini(MAX_CELL_NUM, nv, /**/ &d->p));
    UC(ini(comm, /**/ tg, /**/ &d->c));
    UC(ini(MAX_CELL_NUM, nv, /**/ &d->u));
}

static void ini_rig_distr(int nv, MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ RigDistr *d) {
    using namespace distr::rig;
    UC(ini(MAX_SOLIDS, nv, /**/ &d->p));
    UC(ini(comm, /*io*/ tg, /**/ &d->c));
    UC(ini(MAX_SOLIDS, nv, /**/ &d->u));
}

static void ini_vcont(MPI_Comm comm, /**/ PidVCont *c) {
    int3 L = {XS, YS, ZS};
    float3 V = {VCON_VX, VCON_VY, VCON_VZ};
    UC(ini(comm, L, V, VCON_FACTOR, /**/ c));
}

static void ini_colorer(int nv, MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Colorer *c) {
    UC(ini_mesh_exch(nv, MAX_CELL_NUM, comm, /*io*/ tg, &c->e));
    Dalloc(&c->pp, MAX_PART_NUM);
    Dalloc(&c->minext, MAX_CELL_NUM);
    Dalloc(&c->maxext, MAX_CELL_NUM);
}

static void ini_flu(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ Flu *f) {

    flu::ini(&f->q);
    ini(MAX_PART_NUM, /**/ &f->bulkdata);
    ini(cart, /**/ &f->halodata);
    
    UC(ini_flu_distr(cart, /*io*/ tg, /**/ &f->d));
    UC(ini_flu_exch(cart, /*io*/ tg, /**/ &f->e));
    
    UC(Dalloc(&f->ff, MAX_PART_NUM));
    UC(emalloc(MAX_PART_NUM * sizeof(Force), /**/ (void**) &f->ff_hst));
}

static void ini_rbc(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ Rbc *r) {
    Dalloc(&r->ff, MAX_PART_NUM);
    rbc::main::ini(&r->q);

    UC(ini_rbc_distr(r->q.nv, cart, /*io*/ tg, /**/ &r->d));
    if (rbc_com_dumps) rbc::com::ini(MAX_CELL_NUM, /**/ &r->com);
    if (RBC_STRETCH)   UC(rbc::stretch::ini("rbc.stretch", r->q.nv, /**/ &r->stretch));
}

static void ini_rig(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ Rig *s) {
    rig::ini(&s->q);
    scan::alloc_work(XS * YS * ZS, /**/ &s->ws);
    UC(emalloc(sizeof(&s->ff_hst)*MAX_PART_NUM, (void**) &s->ff_hst));
    Dalloc(&s->ff, MAX_PART_NUM);

    UC(ini_rig_distr(s->q.nv, cart, /*io*/ tg, /**/ &s->d));    
}

static void ini_bounce_back(MPI_Comm cart, Rig *s, /*io*/ basetags::TagGen *tg, /**/ BounceBack *bb) {
    meshbb::ini(MAX_PART_NUM, /**/ &bb->d);
    Dalloc(&bb->mm, MAX_PART_NUM);

    UC(ini_bb_exch(s->q.nt, s->q.nv, MAX_CELL_NUM, cart, /*io*/ tg, /**/ &bb->e));
}

static void ini_wall(Wall *w) {
    sdf::alloc_quants(&w->qsdf);
    wall::alloc_quants(&w->q);
    wall::alloc_ticket(&w->t);
}

static void ini_objhelper(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ ObjHelper *o) {
    if (contactforces) cnt::ini(&o->cnt);
    UC(ini_obj_exch(cart, tg, &o->e));    
}

void ini() {
    basetags::ini(&tag_gen);
    datatype::ini();

    UC(emalloc(3 * MAX_PART_NUM * sizeof(Particle), (void**) &a::pp_hst));
    
    if (rbcs) ini_rbc(m::cart, /* io */ &tag_gen, /**/ &rbc);

    if (VCON) ini_vcont(m::cart, /**/ &vcont);
    if (fsiforces) fsi::ini();

    cnt::ini(&rs::c);

    if (rbcs || solids)
        UC(ini_obj_exch(m::cart, &tag_gen, &rs::e));
    
    bop::ini(&dumpt);

    if (walls) ini_wall(&wall);
    
    ini_flu(m::cart, /*io*/ &tag_gen, /**/ &flu);
   
    if (multi_solvent && rbcs)
        UC(ini_colorer(rbc.q.nv, m::cart, /*io*/ &tag_gen, /**/ &colorer));
    
    if (solids) {
        ini_rig(m::cart, /* io */ &tag_gen, /**/ &rig);

        if (sbounce_back)
            ini_bounce_back(m::cart, &rig, /*io*/ &tag_gen, /**/ &bb);
    }

    MC(MPI_Barrier(m::cart));
}
