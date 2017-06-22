void fin() {
    sdstr::fin();
    rdstr::fin();
    bbhalo::fin();
    cnt::fin();
    dpd::fin();
    dump::fin();
    rex::fin();
    fsi::fin();

    if (solids) mrescue::fin();

    sdf::free_quants(&w::qsdf);
    wall::free_quants(&w::q);
    wall::free_ticket(&w::t);

    delete dump_field;

    flu::free_quants(&o::q);
    flu::free_ticketZ(&o::tz);

    odstr::free_ticketD(&o::td);
    odstr::free_work(&o::w);

    if (solids) {
        rig::free_quants(&s::q);
        rig::free_ticket(&s::t);
        CC(cudaFree(s::ff)); delete[] s::ff_hst;
    }

    if (rbcs) {
        rbc::free_quants(&r::q);
        rbc::destroy_textures(&r::tt);
        CC(cudaFree(r::ff));
    }
}
