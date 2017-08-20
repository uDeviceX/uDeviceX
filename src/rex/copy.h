namespace rex {
void copy_starts(x::TicketPack tp, x::TicketPinned ti) {
    CC(cudaMemcpyAsync(ti.starts, tp.tstarts, sizeof(int) * 27, D2H));
}

void copy_offset(int nw, x::TicketPack tp, x::TicketPinned ti) {
    CC(cudaMemcpyAsync(ti.counts, tp.offsets + 26 * nw, sizeof(int) * 26, D2H));
}

}
