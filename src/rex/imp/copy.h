namespace sub {
void copy_starts(rex::TicketPack tp, /**/ rex::TicketPinned ti) {
    aD2H(ti.starts, tp.tstarts, 27);
}

void copy_offset(int nw, rex::TicketPack tp, rex::TicketPinned ti) {
    aD2H(ti.counts, tp.offsets + 26 * nw, 26);
}
}
