void fin() {
    sdstr::fin();
    rdstr::fin();
    bbhalo::fin();
    cnt::fin();
    dump::fin(&dumpt);
    x::fin(); /* rex:: */
    if (fsiforces)  fsi::fin();
    if (solids) mrescue::fin();

    sdf::free_quants(&w::qsdf);
    wall::free_quants(&w::q);
    wall::free_ticket(&w::t);

    delete dump_field;

    flu::free_quants(&o::q);
    flu::free_ticketZ(&o::tz);
    flu::free_ticketRND(&o::trnd);

    dpdr::free_ticketcom(&o::h::tc);
    dpdr::free_ticketrnd(&o::h::trnd);
    dpdr::free_ticketSh(&o::h::ts);
    dpdr::free_ticketRh(&o::h::tr);
    
    odstr::free_ticketD(&o::td);
    odstr::free_ticketU(&o::tu);
    odstr::free_work(&o::w);

    if (global_ids) {
        flu::free_quantsI(&o::qi);
        odstr::free_ticketI(&o::ti);
        odstr::free_ticketUI(&o::tui);
    }

    if (multi_solvent) {
        flu::free_quantsI(&o::qt);
        odstr::free_ticketI(&o::tt);
        odstr::free_ticketUI(&o::tut);

        dpdr::free_ticketIcom(&o::h::tic);
        dpdr::free_ticketSIh(&o::h::tsi);
        dpdr::free_ticketRIh(&o::h::tri);
        
        mcomm::free_ticketcom(&mc::tc);
        mcomm::free_ticketS(&mc::ts);
        mcomm::free_ticketR(&mc::tr);
    }
    
    if (solids) {
        rig::free_quants(&s::q);
        rig::free_ticket(&s::t);
        scan::free_work(/**/ &s::ws);
        CC(cudaFree(s::ff)); delete[] s::ff_hst;
    }

    if (rbcs) {
        rbc::free_quants(&r::q);
        rbc::destroy_textures(&r::tt);
        CC(cudaFree(r::ff));
    }
    datatype::fin();
}
