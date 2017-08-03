void ini() {
    datatype::ini();
    if (rbcs) {
        CC(cudaMalloc(&r::ff, MAX_PART_NUM));
        rbc::alloc_quants(&r::q);

        rdstr::ini_ticketC(&tag_gen, &r::tdc);
        rdstr::ini_ticketP(MAX_PART_NUM, &r::tdp);
        rdstr::ini_ticketS(&tag_gen, &r::tds);
        rdstr::ini_ticketR(&r::tds, &r::tdr);
        rdstr::alloc_ticketE(&r::tde);
    }

    basetags::ini(&tag_gen);
    
    if (fsiforces) fsi::ini();
    sdstr::ini(&tag_gen);
    bbhalo::ini(&tag_gen);
    cnt::ini();
    x::ini(&tag_gen); /* rex:: */
    dump::ini(&dumpt);

    sdf::alloc_quants(&w::qsdf);
    wall::alloc_quants(&w::q);
    wall::alloc_ticket(&w::t);

    flu::alloc_quants(&o::q);
    flu::alloc_ticketZ(&o::tz);

    odstr::alloc_ticketD(&tag_gen, &o::td);
    odstr::alloc_ticketU(&o::tu);

    dpdr::ini_ticketcom(m::cart, &tag_gen, &o::h::tc);
    dpdr::ini_ticketrnd(o::h::tc, /**/ &o::h::trnd);
    dpdr::alloc_ticketSh(/**/ &o::h::ts);
    dpdr::alloc_ticketRh(/**/ &o::h::tr);

    odstr::alloc_work(&o::w);

    mpDeviceMalloc(&o::ff);

    if (global_ids) {
        flu::alloc_quantsI(&o::qi);
        odstr::alloc_ticketI(&tag_gen, &o::ti);
        odstr::alloc_ticketUI(&o::tui);
    }
    
    if (multi_solvent) {
        flu::alloc_quantsI(&o::qt);
        odstr::alloc_ticketI(&tag_gen, &o::tt);
        odstr::alloc_ticketUI(&o::tut);

        dpdr::ini_ticketIcom(/*io*/ &tag_gen, /**/ &o::h::tic);
        dpdr::alloc_ticketSIh(/**/ &o::h::tsi);
        dpdr::alloc_ticketRIh(/**/ &o::h::tri);
        
        mcomm::ini_ticketcom(m::cart, /*io*/ &tag_gen, /**/ &mc::tc);
        mcomm::alloc_ticketS(/**/ &mc::ts);
        mcomm::alloc_ticketR(&mc::ts, /**/ &mc::tr);
    }
    
    if (solids) {
        mrescue::ini(MAX_PART_NUM);
        rig::alloc_quants(&s::q);
        rig::alloc_ticket(&s::t);
        scan::alloc_work(XS*YS*ZS, /**/ &s::ws);
        s::ff_hst = new Force[MAX_PART_NUM];
        CC(cudaMalloc(&s::ff, MAX_PART_NUM * sizeof(Force)));

        mbounce::alloc_ticketM(&bb::tm);
    }

    dump_field = new H5FieldDump;
    MC(MPI_Barrier(m::cart));
}
