void forces_dpd() {
    using namespace dpdr;
    using namespace o;
    int *count = q.cells->count;
    int *start = q.cells->start;
    gather_cells(start, count, /**/ &h.ts);
    if (h.tc.first) post_expected_recv(&h.tc, &h.tr);
    copy_cells(&h.ts);
    pack(q.pp, /**/ &h.ts);
    post_send(&h.tc, &h.ts);

    if (multi_solvent) flocal_color(tz.zip0, tz.zip1, qc.ii, q.n, start, count, trnd.rnd, /**/ ff);
    else flocal(tz.zip0, tz.zip1, q.n, start, count, trnd.rnd, /**/ ff);

    wait_recv(&h.tc);
    recv(&h.tr);
    post_expected_recv(&h.tc, &h.tr);
    if (multi_solvent) fremote_color(h.trnd, h.ts, h.tr, h.tsi, h.tri, /**/ ff);
    else               fremote(h.trnd, h.ts, h.tr, /**/ ff);
    h.tc.first = false;
}
