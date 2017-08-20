namespace rex {
void copy_starts(x::TicketPack tp, x::TicketPinned ti) {
    CC(cudaMemcpyAsync(ti.starts, tp.tstarts, sizeof(int) * 27, D2H));
}

void copy_ff() {
    int i;
    float *ff[26];
    for (i = 0; i < 26; ++i) ff[i] = (float*)local[i].ff;
    CC(cudaMemcpyToSymbolAsync(k_rex::g::ff, ff, sizeof(ff), 0, H2D));
}

void copy_offset(int nw, x::TicketPack tp, x::TicketPinned ti) {
    CC(cudaMemcpyAsync(ti.counts, tp.offsets + 26 * nw, sizeof(int) * 26, D2H));
}

}
