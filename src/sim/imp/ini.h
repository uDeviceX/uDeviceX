static void ini_obj_exch(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Objexch *e) {
    using namespace exch::obj;
    ini(MAX_OBJ_TYPES, MAX_OBJ_DENSITY, &e->p);
    ini(comm, /*io*/ tg, /**/ &e->c);
    ini(MAX_OBJ_DENSITY, &e->u);
    ini(MAX_OBJ_DENSITY, &e->pf);
    ini(MAX_OBJ_DENSITY, &e->uf);    
}

static void ini_mesh_exch(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Mexch *e) {
    using namespace exch::mesh;
    ini(MAX_VERT_NUM, MAX_CELL_NUM, &e->p);
    ini(m::cart, /*io*/ &tag_gen, /**/ &e->c);
    ini(MAX_VERT_NUM, MAX_CELL_NUM, &e->u);
}

static void ini_bb_exch(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ BBexch *e) {
    ini_mesh_exch(comm, /*io*/ tg, /**/ e);

    using namespace exch::mesh;
    ini(MAX_FACE_NUM, MAX_CELL_NUM, &e->pm);
    ini(comm, /*io*/ tg, /**/ &e->cm);
    ini(MAX_FACE_NUM, MAX_CELL_NUM, &e->um);
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
    ini(nv, /**/ &d->p);
    ini(comm, /**/ tg, /**/ &d->c);
    ini(nv, /**/ &d->u);
}

static void ini_rig_distr(int nv, MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ RigDistr *d) {
    using namespace distr::rig;
    ini(nv, /**/ &d->p);
    ini(comm, /*io*/ tg, /**/ &d->c);
    ini(nv, /**/ &d->u);
}

void ini() {
    basetags::ini(&tag_gen);
    datatype::ini();
    if (rbcs) {
        Dalloc(&r::ff, MAX_PART_NUM);
        rbc::alloc_quants(&r::q);

        ini_rbc_distr(r::q.nv, m::cart, /*io*/ &tag_gen, /**/ &r::d);
    }
    
    if (fsiforces) fsi::ini();

    bbhalo::ini(&tag_gen);
    cnt::ini();

    if (rbcs || solids)
        ini_obj_exch(m::cart, &tag_gen, &rs::e);
    
    bop::ini(&dumpt);

    if (walls) {
        sdf::alloc_quants(&w::qsdf);
        wall::alloc_quants(&w::q);
        wall::alloc_ticket(&w::t);
    }

    flu::alloc_quants(&o::q);
    flu::alloc_ticketZ(&o::tz);

    ini_flu_distr(m::cart, /*io*/ &tag_gen, /**/ &o::d);

    dpdr::ini_ticketcom(m::cart, &tag_gen, &o::h.tc);
    dpdr::ini_ticketrnd(o::h.tc, /**/ &o::h.trnd);
    dpdr::alloc_ticketSh(/**/ &o::h.ts);
    dpdr::alloc_ticketRh(/**/ &o::h.tr);

    Dalloc(&o::ff, MAX_PART_NUM);

    if (global_ids) {
        flu::alloc_quantsI(&o::qi);
    }
    
    if (multi_solvent) {
        flu::alloc_quantsI(&o::qc);

        dpdr::ini_ticketIcom(/*io*/ &tag_gen, /**/ &o::h.tic);
        dpdr::alloc_ticketSIh(/**/ &o::h.tsi);
        dpdr::alloc_ticketRIh(/**/ &o::h.tri);
    }

    if (multi_solvent && rbcs) {
        ini_mesh_exch(m::cart, /*io*/ &tag_gen, &mc::e);
        Dalloc(&mc::pp, MAX_PART_NUM);
    }
    
    if (solids) {
        mrescue::ini(MAX_PART_NUM);
        rig::alloc_quants(&s::q);
        rig::alloc_ticket(&s::t);
        scan::alloc_work(XS*YS*ZS, /**/ &s::ws);
        s::ff_hst = (Force*)malloc(sizeof(&s::ff_hst)*MAX_PART_NUM);
        Dalloc(&s::ff, MAX_PART_NUM);

        tcells::alloc_quants(MAX_SOLIDS, &bb::qtc);
        mbounce::alloc_ticketM(&bb::tm);

        ini(MAX_PART_NUM, /**/ &bb::bbd);
        Dalloc(&bb::mm, MAX_PART_NUM);

        ini_rig_distr(s::q.nv, m::cart, /*io*/ &tag_gen, /**/ &s::d);

        if (sbounce_back)
            ini_bb_exch(m::cart, /*io*/ &tag_gen, /**/ &s::e);
    }

    MC(MPI_Barrier(m::cart));
}
