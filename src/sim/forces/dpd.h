void forces_dpd() {
    dpdr::gather_cells(o::q.cells->start, o::q.cells->count, /**/ &o::h::ts);
    if (o::h::tc.first) dpdr::post_expected_recv(&o::h::tc, &o::h::tr);
    dpdr::copy_cells(&o::h::ts);
    dpdr::pack(o::q.pp, /**/ &o::h::ts);
    dpdr::post_send(&o::h::tc, &o::h::ts);

    flocal(o::tz.zip0, o::tz.zip1, o::q.n, o::q.cells->start, o::q.cells->count, o::trnd.rnd, /**/ o::ff);

    dpdr::wait_recv(&o::h::tc);
    dpdr::recv(&o::h::tr);
    dpdr::post_expected_recv(&o::h::tc, &o::h::tr);
    dpdr::fremote(o::h::trnd, o::h::ts, o::h::tr, /**/ o::ff);
    o::h::tc.first = false;
}
