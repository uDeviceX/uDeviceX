void forces_dpd(bool stress, Flu *f) {
    const int *count = f->q.cells.counts;
    const int *start = f->q.cells.starts;
    PaArray parray;
    FoArray farray;
    flu::LFrag26 lfrags;
    flu::RFrag26 rfrags;

    FluExch *e = &f->e;
    
    parray_push_pp(f->q.pp, /**/ &parray);
    if (f->q.colors)
        parray_push_cc(f->q.cc, /**/ &parray);

    farray_push_ff(f->ff, /**/ &farray);
    if (stress)
        farray_push_ss(f->ss, /**/ &farray);
    
    UC(eflu_compute_map(start, count, /**/ e->p));
    UC(eflu_download_cell_starts(/**/ e->p));
    UC(eflu_pack(&parray, /**/ e->p));
    UC(eflu_download_data(/**/ e->p));

    UC(eflu_post_recv(e->c, e->u));
    UC(eflu_post_send(e->p, e->c));
    
    UC(fluforces_bulk_prepare(f->q.n, &parray, /**/ f->bulk));
    UC(fluforces_bulk_apply(f->params, f->q.n, f->bulk, start, count, /**/ &farray));

    UC(eflu_wait_recv(e->c, e->u));
    UC(eflu_wait_send(e->c));

    UC(eflu_unpack(e->u));

    UC(eflu_get_local_frags(e->p, /**/ &lfrags));
    UC(eflu_get_remote_frags(e->u, /**/ &rfrags));

    UC(fluforces_halo_prepare(lfrags, rfrags, /**/ f->halo));
    UC(fluforces_halo_apply(f->params, f->halo, /**/ &farray));
}
