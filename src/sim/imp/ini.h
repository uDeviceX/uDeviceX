static void ini_obj_exch(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Objexch *e) {
    using namespace exch::obj;
    int maxpsolid = MAX_PSOLID_NUM;
    
    ini(MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid, &e->p);
    ini(comm, /*io*/ tg, /**/ &e->c);
    ini(MAX_OBJ_DENSITY, maxpsolid, &e->u);
    ini(MAX_OBJ_DENSITY, maxpsolid, &e->pf);
    ini(MAX_OBJ_DENSITY, maxpsolid, &e->uf);    
}

static void ini_mesh_exch(int nv, int max_m, MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Mexch *e) {
    using namespace exch::mesh;
    ini(nv, max_m, &e->p);
    ini(comm, /*io*/ &tag_gen, /**/ &e->c);
    ini(nv, max_m, &e->u);
}

static void ini_bb_exch(int nt, int nv, int max_m, MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ BBexch *e) {
    ini_mesh_exch(nv, max_m, comm, /*io*/ tg, /**/ e);

    using namespace exch::mesh;
    ini(nt, max_m, &e->pm);
    ini(comm, /*io*/ tg, /**/ &e->cm);
    ini(nt, max_m, &e->um);
}

static void ini_flu_distr(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ FluDistr *d) {
    using namespace distr::flu;
    float maxdensity = ODSTR_FACTOR * numberdensity;
    ini(maxdensity, /**/ &d->p);
    ini(comm, /**/ tg, /**/ &d->c);
    ini(maxdensity, /**/ &d->u);
}

static void ini_rbc_distr(int nv, MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ RbcDistr *d) {
    using namespace distr::rbc;
    ini(MAX_CELL_NUM, nv, /**/ &d->p);
    ini(comm, /**/ tg, /**/ &d->c);
    ini(MAX_CELL_NUM, nv, /**/ &d->u);
}

static void ini_rig_distr(int nv, MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ RigDistr *d) {
    using namespace distr::rig;
    ini(MAX_SOLIDS, nv, /**/ &d->p);
    ini(comm, /*io*/ tg, /**/ &d->c);
    ini(MAX_SOLIDS, nv, /**/ &d->u);
}

static void ini_vcont(MPI_Comm comm, /**/ PidVCont *c) {
    int3 L = {XS, YS, ZS};
    float3 V = {VCON_VX, VCON_VY, VCON_VZ};
    ini(comm, L, V, VCON_FACTOR, /**/ c);
}

void ini() {
    basetags::ini(&tag_gen);
    datatype::ini();
    if (rbcs) {
        Dalloc(&r::ff, MAX_PART_NUM);
        rbc::ini(&r::q);

        ini_rbc_distr(r::q.nv, m::cart, /*io*/ &tag_gen, /**/ &r::d);

        if (rbc_com_dumps)
            rbc::ini(MAX_CELL_NUM, /**/ &r::com);
    }

    if (VCON) ini_vcont(m::cart, /**/ &o::vcont);
    if (fsiforces) fsi::ini();

    cnt::ini(&rs::c);

    if (rbcs || solids)
        ini_obj_exch(m::cart, &tag_gen, &rs::e);
    
    bop::ini(&dumpt);

    if (walls) {
        sdf::alloc_quants(&w::qsdf);
        wall::alloc_quants(&w::q);
        wall::alloc_ticket(&w::t);
    }

    flu::ini(&o::q);
    flu::ini(&o::tz);
    flu::ini(&o::trnd);
    
    ini_flu_distr(m::cart, /*io*/ &tag_gen, /**/ &o::d);

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

    if (multi_solvent && rbcs) {
        ini_mesh_exch(r::q.nv, MAX_CELL_NUM, m::cart, /*io*/ &tag_gen, &mc::e);
        Dalloc(&mc::pp, MAX_PART_NUM);
    }
    
    if (solids) {
        rig::ini(&s::q);
        scan::alloc_work(XS*YS*ZS, /**/ &s::ws);
        s::ff_hst = (Force*)malloc(sizeof(&s::ff_hst)*MAX_PART_NUM);
        Dalloc(&s::ff, MAX_PART_NUM);

        ini(MAX_PART_NUM, /**/ &bb::bbd);
        Dalloc(&bb::mm, MAX_PART_NUM);

        ini_rig_distr(s::q.nv, m::cart, /*io*/ &tag_gen, /**/ &s::d);

        if (sbounce_back)
            ini_bb_exch(s::q.nt, s::q.nv, MAX_CELL_NUM, m::cart, /*io*/ &tag_gen, /**/ &s::e);
    }

    MC(MPI_Barrier(m::cart));
}
