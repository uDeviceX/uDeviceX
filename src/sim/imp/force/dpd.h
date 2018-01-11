void forces_dpd(Flu *f) {
    const int *count = f->q.cells.counts;
    const int *start = f->q.cells.starts;
    Cloud cloud;
    flu::LFrag26 lfrags;
    flu::RFrag26 rfrags;

    FluExch *e = &f->e;
    
    ini_cloud(f->q.pp, /**/ &cloud);
    if (multi_solvent)
        ini_cloud_color(f->q.cc, /**/ &cloud);

    eflu_compute_map(start, count, /**/ &e->p);
    eflu_download_cell_starts(/**/ &e->p);
    eflu_pack(&cloud, /**/ &e->p);
    eflu_download_data(/**/ &e->p);

    UC(eflu_post_recv(&e->c, &e->u));
    UC(eflu_post_send(&e->p, &e->c));
    
     prepare(f->q.n, &cloud, /**/ f->bulkdata);
    bulk_forces(f->q.n, f->bulkdata, start, count, /**/ f->ff);

    dSync();
    UC(eflu_wait_recv(&e->c, &e->u));
    UC(eflu_wait_send(&e->c));

    eflu_unpack(&e->u);

    eflu_get_local_frags(&e->p, /**/ &lfrags);
    eflu_get_remote_frags(&e->u, /**/ &rfrags);

    prepare(lfrags, rfrags, /**/ f->halodata);
    halo_forces(f->halodata, /**/ f->ff);
}
