void forces_dpd() {
    using namespace dpdr;
    using namespace o;
    int *count = q.cells->count;
    int *start = q.cells->start;
    CC(cudaPeekAtLastError());
    gather_cells(start, count, /**/ &h::ts);
    CC(cudaPeekAtLastError());
    if (h::tc.first) post_expected_recv(&h::tc, &h::tr);
    CC(cudaPeekAtLastError());
    copy_cells(&h::ts);
    CC(cudaPeekAtLastError());
    pack(q.pp, /**/ &h::ts);
    CC(cudaPeekAtLastError());
    post_send(&h::tc, &h::ts);
    CC(cudaPeekAtLastError());
    flocal(tz.zip0, tz.zip1, q.n, start, count, trnd.rnd, /**/ ff);

    wait_recv(&h::tc);
    recv(&h::tr);
    post_expected_recv(&h::tc, &h::tr);
    fremote(h::trnd, h::ts, h::tr, /**/ ff);
    h::tc.first = false;
}
