static void ini_flu_exch(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Fluexch *e) {
    using namespace exch::flu;
    int maxd = HSAFETY_FACTOR * numberdensity;
    
    UC(ini(maxd, /**/ &e->p));
    UC(ini(comm, /*io*/ tg, /**/ &e->c));
    UC(ini(maxd, /**/ &e->u));
}

static void ini_obj_exch(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Objexch *e) {
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

void ini() {
    basetags::ini(&tag_gen);
    datatype::ini();
    if (rbcs) {
        Dalloc(&r::ff, MAX_PART_NUM);
        rbc::main::ini(&r::q);

        UC(ini_rbc_distr(r::q.nv, m::cart, /*io*/ &tag_gen, /**/ &r::d));
        if (rbc_com_dumps) rbc::com::ini(MAX_CELL_NUM, /**/ &r::com);
        if (RBC_STRETCH)   rbc::stretch::ini("rbc.stretch", r::q.nv, /**/ &r::stretch);
    }

    if (VCON) ini_vcont(m::cart, /**/ &o::vcont);
    if (fsiforces) fsi::ini();

    cnt::ini(&rs::c);

    if (rbcs || solids)
        UC(ini_obj_exch(m::cart, &tag_gen, &rs::e));
    
    bop::ini(&dumpt);

    if (walls) {
        sdf::alloc_quants(&w::qsdf);
        wall::alloc_quants(&w::q);
        wall::alloc_ticket(&w::t);
    }

    flu::ini(&o::q);
    ini(MAX_PART_NUM, /**/ &o::bulkdata);
    
    UC(ini_flu_distr(m::cart, /*io*/ &tag_gen, /**/ &o::d));

    dpdr::ini_ticketcom(m::cart, &tag_gen, &o::h.tc);
    dpdr::ini_ticketrnd(o::h.tc, /**/ &o::h.trnd);
    dpdr::alloc_ticketSh(/**/ &o::h.ts);
    dpdr::alloc_ticketRh(/**/ &o::h.tr);

    Dalloc(&o::ff, MAX_PART_NUM);
    
    if (multi_solvent) {
        dpdr::ini_ticketIcom(/*io*/ &tag_gen, /**/ &o::h.tic);
        dpdr::alloc_ticketSIh(/**/ &o::h.tsi);
        dpdr::alloc_ticketRIh(/**/ &o::h.tri);
    }

    if (multi_solvent && rbcs)
        UC(ini_colorer(r::q.nv, m::cart, /*io*/ &tag_gen, /**/ &colorer));
    
    if (solids) {
        rig::ini(&s::q);
        scan::alloc_work(XS*YS*ZS, /**/ &s::ws);
        UC(emalloc(sizeof(&s::ff_hst)*MAX_PART_NUM, (void**) &s::ff_hst));
        Dalloc(&s::ff, MAX_PART_NUM);

        meshbb::ini(MAX_PART_NUM, /**/ &bb::bbd);
        Dalloc(&bb::mm, MAX_PART_NUM);

        UC(ini_rig_distr(s::q.nv, m::cart, /*io*/ &tag_gen, /**/ &s::d));

        if (sbounce_back)
            UC(ini_bb_exch(s::q.nt, s::q.nv, MAX_CELL_NUM, m::cart, /*io*/ &tag_gen, /**/ &s::e));
    }

    MC(MPI_Barrier(m::cart));
}
