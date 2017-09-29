static void fin_obj_exch(Sexch *e) {
    using namespace exch::obj;
    fin(&e->p);
    fin(&e->c);
    fin(&e->u);
    fin(&e->pf);
    fin(&e->uf);
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
    
    distr::flu::fin(/**/ &o::d.p);
    distr::flu::fin(/**/ &o::d.c);
    distr::flu::fin(/**/ &o::d.u);
    
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

        distr::rig::fin(&s::d.p);
        distr::rig::fin(&s::d.c);
        distr::rig::fin(&s::d.u);
    }

    if (rbcs) {
        rbc::free_quants(&r::q);
        rbc::destroy_textures(&r::tt);

        distr::rbc::fin(/**/ &r::d.p);
        distr::rbc::fin(/**/ &r::d.c);
        distr::rbc::fin(/**/ &r::d.u);
        
        Dfree(r::ff);
    }
    datatype::fin();
}
