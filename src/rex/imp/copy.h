namespace sub {
void copy_starts(rex::TicketPack tp, /**/ rex::TicketPinned ti) {
    CC(cudaMemcpyAsync(ti.starts, tp.tstarts, sizeof(int) * 27, D2H));
}

void copy_offset(int nw, rex::TicketPack tp, rex::TicketPinned ti) {
    aD2H(ti.counts, tp.offsets + 26 * nw, 26);
}
}
