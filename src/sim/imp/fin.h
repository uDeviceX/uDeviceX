void fin() {
    sdstr::fin();
    bbhalo::fin();
    cnt::fin();
    bop::fin(&dumpt);
    if (rbcs || solids) rex::fin(); /* rex:: */
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
    
    odstr::free_ticketD(&o::td);
    odstr::free_ticketU(&o::tu);
    odstr::free_work(&o::w);

    distr::flu::fin(/**/ &o::d.p);
    distr::flu::fin(/**/ &o::d.p);
    distr::flu::fin(/**/ &o::d.u);
    
    if (global_ids) {
        flu::free_quantsI(&o::qi);
        odstr::free_ticketI(&o::ti);
        odstr::free_ticketUI(&o::tui);
    }

    if (multi_solvent) {
        flu::free_quantsI(&o::qc);
        odstr::free_ticketI(&o::tc);
        odstr::free_ticketUI(&o::tuc);

        dpdr::free_ticketIcom(&o::h.tic);
        dpdr::free_ticketSIh(&o::h.tsi);
        dpdr::free_ticketRIh(&o::h.tri);
        
        mcomm::free_ticketcom(&mc::tc);
        mcomm::free_ticketS(&mc::ts);
        mcomm::free_ticketR(&mc::tr);
    }
    
    if (solids) {
        rig::free_quants(&s::q);
        rig::free_ticket(&s::t);
        scan::free_work(/**/ &s::ws);
        Dfree(s::ff);
        free(s::ff_hst);

        tcells::free_quants(&bb::qtc);
        mbounce::free_ticketM(&bb::tm);
    }

    if (rbcs) {
        rbc::free_quants(&r::q);
        rbc::destroy_textures(&r::tt);

        rdstr::free_ticketC(&r::tdc);
        rdstr::free_ticketP(&r::tdp);
        rdstr::free_ticketS(&r::tds);
        rdstr::free_ticketR(&r::tdr);
        rdstr::free_ticketE(&r::tde);
        
        Dfree(r::ff);
    }
    datatype::fin();
}
