void forces_dpd() {
    using namespace o;
    const int *count = q.cells.counts;
    const int *start = q.cells.starts;
    Cloud cloud;
    flu::LFrag26 lfrags;
    flu::RFrag26 rfrags;

    ini_cloud(q.pp, /**/ &cloud);
    if (multi_solvent)
        ini_cloud_color(q.cc, /**/ &cloud);

    compute_map(start, count, /**/ &e.p);
    download_cell_starts(/**/ &e.p);
    pack(&cloud, /**/ &e.p);
    download_data(/**/ &e.p);

    UC(post_recv(&e.c, &e.u));
    UC(post_send(&e.p, &e.c));
    
    prepare(q.n, &cloud, /**/ bulkdata);
    bulk_forces(q.n, bulkdata, start, count, /**/ ff);
    dSync();
    UC(wait_recv(&e.c, &e.u));
    UC(wait_send(&e.c));

    unpack(&e.u);

    get_local_frags(&e.p, /**/ &lfrags);
    get_remote_frags(&e.u, /**/ &rfrags);

    prepare(lfrags, rfrags, /**/ halodata);
    halo_forces(halodata, /**/ ff);
}
