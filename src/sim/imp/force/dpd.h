void forces_dpd() {
    using namespace dpdr;
    using namespace o;
    int *count = q.cells.counts;
    int *start = q.cells.starts;
    gather_cells(start, count, /**/ &h.ts);
    if (h.tc.first) post_expected_recv(&h.tc, &h.tr);
    if (multi_solvent && h.tic.first)
        post_expected_recv_ii(&h.tc, &h.tr, /**/ &h.tic, &h.tri);

    copy_cells(&h.ts);

    pack(q.pp, /**/ &h.ts);
    if (multi_solvent)
        pack_ii(qc.ii, &h.ts, /**/ &h.tsi);

    post_send(&h.tc, &h.ts);
    if (multi_solvent)
        post_send_ii(&h.tc, &h.ts, /**/ &h.tic, &h.tsi);
 
    if (multi_solvent) flocal_color(tz.zip0, tz.zip1, qc.ii, q.n, start, count, trnd.rnd, /**/ ff);
    else flocal(tz.zip0, tz.zip1, q.n, start, count, trnd.rnd, /**/ ff);

    wait_recv(&h.tc);
    if (multi_solvent)
        wait_recv_ii(&h.tic);
    
    recv(&h.tr);
    if (multi_solvent)
        recv_ii(&h.tr, /**/ &h.tri);

    post_expected_recv(&h.tc, &h.tr);
    if (multi_solvent)
        post_expected_recv_ii(&h.tc, &h.tr, /**/ &h.tic, &h.tri);
    
    if (multi_solvent) fremote_color(h.trnd, h.ts, h.tr, h.tsi, h.tri, /**/ ff);
    else               fremote(h.trnd, h.ts, h.tr, /**/ ff);

    h.tc.first = false;
    h.tic.first = false;
}
