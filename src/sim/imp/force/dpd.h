void forces_dpd(Flu *f) {
    const int *count = f->q.cells.counts;
    const int *start = f->q.cells.starts;
    Cloud cloud;
    flu::LFrag26 lfrags;
    flu::RFrag26 rfrags;

    Fluexch *e = &f->e;
    
    ini_cloud(f->q.pp, /**/ &cloud);
    if (multi_solvent)
        ini_cloud_color(f->q.cc, /**/ &cloud);

    compute_map(start, count, /**/ &e->p);
    download_cell_starts(/**/ &e->p);
    pack(&cloud, /**/ &e->p);
    download_data(/**/ &e->p);

    UC(post_recv(&e->c, &e->u));
    UC(post_send(&e->p, &e->c));
    
    prepare(f->q.n, &cloud, /**/ f->bulkdata);
    bulk_forces(f->q.n, f->bulkdata, start, count, /**/ f->ff);
    dSync();
    UC(wait_recv(&e->c, &e->u));
    UC(wait_send(&e->c));

    unpack(&e->u);

    get_local_frags(&e->p, /**/ &lfrags);
    get_remote_frags(&e->u, /**/ &rfrags);

    prepare(lfrags, rfrags, /**/ f->halodata);
    halo_forces(f->halodata, /**/ f->ff);
}
