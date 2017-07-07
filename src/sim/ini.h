void ini() {
    if (rbcs) {
        CC(cudaMalloc(&r::ff, MAX_PART_NUM));
        rbc::alloc_quants(&r::q);
    }
    rdstr::ini();
    fsi::ini();
    sdstr::ini();
    bbhalo::ini();
    cnt::ini();
    rex::ini();
    dump::ini();

    sdf::alloc_quants(&w::qsdf);
    wall::alloc_quants(&w::q);
    wall::alloc_ticket(&w::t);

    flu::alloc_quants(&o::q);
    flu::alloc_ticketZ(&o::tz);

    odstr::alloc_ticketD(&o::td);
    odstr::alloc_work(&o::w);

    dpdr::ini_ticketcom(m::cart, &o::h::tc);
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
