namespace sim {
void ini() {
    if (rbcs) {
        CC(cudaMalloc(&r::ff, MAX_PART_NUM));
        rbc::alloc_quants(&r::q);
        rbc::setup("rbc.off", &r::q);
        rbc::setup_textures(r::q, &r::tt);
    }
    rdstr::ini();
    dpd::ini();
    fsi::ini();
    sdstr::ini();
    bbhalo::ini();
    cnt::ini();
    rex::ini();
    dump::ini();

    wall::alloc_quants(&w::q);
    wall::alloc_ticket(&w::t);

    o::cells   = new Clist(XS, YS, ZS);
    flu::alloc_ticketD(&o::td);
    flu::alloc_ticketZ(&o::tz);
    flu::alloc_work(&o::w);

    mpDeviceMalloc(&o::pp);
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
}
