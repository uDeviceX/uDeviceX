static void fin_obj_exch(/**/ Objexch *e) {
    using namespace exch::obj;
    fin(&e->p);
    fin(&e->c);
    fin(&e->u);
    fin(&e->pf);
    fin(&e->uf);
}

static void fin_bb_exch(/**/ BBexch *e) {
    using namespace exch::mesh;
    fin(&e->p);
    fin(&e->c);
    fin(&e->u);

    fin(&e->pm);
    fin(&e->cm);
    fin(&e->um);
}

static void fin_flu_distr(/**/ FluDistr *d) {
    using namespace distr::flu;
    fin(/**/ &d->p);
    fin(/**/ &d->c);
    fin(/**/ &d->u);
}

static void fin_rbc_distr(/**/ RbcDistr *d) {
    using namespace distr::rbc;
    fin(/**/ &d->p);
    fin(/**/ &d->c);
    fin(/**/ &d->u);
}

static void fin_rig_distr(/**/ RigDistr *d) {
    using namespace distr::rig;
    fin(/**/ &d->p);
    fin(/**/ &d->c);
    fin(/**/ &d->u);
}

void fin() {
    bbhalo::fin();
    cnt::fin();
    bop::fin(&dumpt);
    if (rbcs || solids)
        fin_obj_exch(&rs::e);

    if (fsiforces)  fsi::fin();
    if (solids) mrescue::fin();

    if (walls) {
        sdf::free_quants(&w::qsdf);
        wall::free_quants(&w::q);
        wall::free_ticket(&w::t);
    }

    flu::free_quants(&o::q);
    flu::free_ticketZ(&o::tz);
    flu::free_ticketRND(&o::trnd);

    dpdr::free_ticketcom(&o::h.tc);
    dpdr::free_ticketrnd(&o::h.trnd);
    dpdr::free_ticketSh(&o::h.ts);
    dpdr::free_ticketRh(&o::h.tr);

    fin_flu_distr(/**/ &o::d);
    
    if (global_ids) {
        flu::free_quantsI(&o::qi);
    }

    if (multi_solvent) {
        flu::free_quantsI(&o::qc);
    
        dpdr::free_ticketIcom(&o::h.tic);
        dpdr::free_ticketSIh(&o::h.tsi);
        dpdr::free_ticketRIh(&o::h.tri);
    }

    if (multi_solvent && rbcs) {
        exch::mesh::fin(/**/ &mc::e.p);
        exch::mesh::fin(/**/ &mc::e.c);
        exch::mesh::fin(/**/ &mc::e.u);
        Dfree(mc::pp);
    }

    if (solids) {
        rig::free_quants(&s::q);
        rig::free_ticket(&s::t);
        scan::free_work(/**/ &s::ws);
        Dfree(s::ff);
        free(s::ff_hst);

        tcells::free_quants(&bb::qtc);
        mbounce::free_ticketM(&bb::tm);

        fin(/**/ &bb::bbd);
        Dfree(bb::mm);

        fin_rig_distr(/**/ &s::d);
        
        if (sbounce_back)
            fin_bb_exch(/**/ &s::e);
    }

    if (rbcs) {
        rbc::free_quants(&r::q);
        rbc::destroy_textures(&r::tt);

        fin_rbc_distr(/**/ &r::d);
        
        Dfree(r::ff);
    }
    datatype::fin();
}
