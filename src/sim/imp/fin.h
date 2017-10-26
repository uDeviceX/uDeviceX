static void fin_obj_exch(/**/ Objexch *e) {
    using namespace exch::obj;
    fin(&e->p);
    fin(&e->c);
    fin(&e->u);
    fin(&e->pf);
    fin(&e->uf);
}

static void fin_mesh_exch(/**/ Mexch *e) {
    using namespace exch::mesh;
    fin(&e->p);
    fin(&e->c);
    fin(&e->u);
}

static void fin_bb_exch(/**/ BBexch *e) {
    fin_mesh_exch(/**/ e);
    
    using namespace exch::mesh;
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
    cnt::fin();
    cnt::fin(&rs::c);
    bop::fin(&dumpt);
    if (rbcs || solids)
        fin_obj_exch(&rs::e);

    if (VCON) fin(/**/ &o::vcont);
    if (fsiforces)  fsi::fin();
    
    if (walls) {
        sdf::free_quants(&w::qsdf);
        wall::free_quants(&w::q);
        wall::free_ticket(&w::t);
    }

    flu::fin(&o::q);
    flu::fin(&o::tz);
    flu::fin(&o::trnd);

    dpdr::free_ticketcom(&o::h.tc);
    dpdr::free_ticketrnd(&o::h.trnd);
    dpdr::free_ticketSh(&o::h.ts);
    dpdr::free_ticketRh(&o::h.tr);

    fin_flu_distr(/**/ &o::d);
    
    if (multi_solvent) {
        dpdr::free_ticketIcom(&o::h.tic);
        dpdr::free_ticketSIh(&o::h.tsi);
        dpdr::free_ticketRIh(&o::h.tri);
    }

    if (multi_solvent && rbcs) {
        fin_mesh_exch(/**/ &mc::e);
        Dfree(mc::pp);
    }

    if (solids) {
        rig::fin(&s::q);
        scan::free_work(/**/ &s::ws);
        Dfree(s::ff);
        free(s::ff_hst);

        fin(/**/ &bb::bbd);
        Dfree(bb::mm);

        fin_rig_distr(/**/ &s::d);
        
        if (sbounce_back)
            fin_bb_exch(/**/ &s::e);
    }

    if (rbcs) {
        rbc::fin(&r::q);
        rbc::fin_ticket(&r::tt);

        fin_rbc_distr(/**/ &r::d);
        
        Dfree(r::ff);

        if (rbc_com_dumps)
            rbc::fin(/**/ &r::com);
    }
    datatype::fin();
}
