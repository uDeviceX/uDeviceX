void ini() {
    datatype::ini();
    if (rbcs) {
        CC(cudaMalloc(&r::ff, MAX_PART_NUM));
        rbc::alloc_quants(&r::q);
    }

    basetags::ini(&tag_gen);
    
    rdstr::ini(&tag_gen);
    if (fsiforces) fsi::ini();
    sdstr::ini(&tag_gen);
    bbhalo::ini(&tag_gen);
    cnt::ini();
    rex::ini(&tag_gen);
    dump::ini(&dumpt);

    sdf::alloc_quants(&w::qsdf);
    wall::alloc_quants(&w::q);
    wall::alloc_ticket(&w::t);

    flu::alloc_quants(&o::q);
    flu::alloc_ticketZ(&o::tz);

    odstr::alloc_ticketD(&tag_gen, &o::td);
    odstr::alloc_ticketU(&o::tu);

    if (global_ids) {
        flu::alloc_quantsI(&o::qi);
        odstr::alloc_ticketI(&tag_gen, &o::ti);
        odstr::alloc_ticketUI(&o::tui);
    }
    
    if (multi_solvent) {
        flu::alloc_quantsI(&o::qt);
        odstr::alloc_ticketI(&tag_gen, &o::tt);
        odstr::alloc_ticketUI(&o::tut);

        mcomm::ini_ticketcom(m::cart, /*io*/ &tag_gen, /**/ &mc::tc);
        mcomm::alloc_ticketS(/**/ &mc::ts);
        mcomm::alloc_ticketR(&mc::ts, /**/ &mc::tr);
    }
    
    odstr::alloc_work(&o::w);

    dpdr::ini_ticketcom(m::cart, &tag_gen, &o::h::tc);
    dpdr::ini_ticketrnd(o::h::tc, /**/ &o::h::trnd);
    dpdr::alloc_ticketSh(/**/ &o::h::ts);
    dpdr::alloc_ticketRh(/**/ &o::h::tr);

    mpDeviceMalloc(&o::ff);

    if (solids) {
        mrescue::ini(MAX_PART_NUM);
        rig::alloc_quants(&s::q);
        rig::alloc_ticket(&s::t);
        s::ff_hst = new Force[MAX_PART_NUM];
        CC(cudaMalloc(&s::ff, MAX_PART_NUM * sizeof(Force)));
    }

    dump_field = new H5FieldDump;
    MC(MPI_Barrier(m::cart));
}
