namespace rex {
void copy_tstarts(x::TicketPack tp, x::TicketPinned ti) {
    CC(cudaMemcpyAsync(ti.tstarts, tp.tstarts, sizeof(int) * 27, H2H));
}

void copy_pack(x::TicketPinned t) {
    if (t.tstarts[26]) CC(cudaMemcpyAsync(host_packbuf, packbuf, sizeof(Particle) * t.tstarts[26], H2H));
}

void copy_state() {
    int i;
    for (i = 0; i < 26; ++i)
        CC(cudaMemcpyAsync(remote[i]->dstate.D, remote[i]->hstate.D, sizeof(Particle) * remote[i]->hstate.S, H2D));
}

void copy_bags() {
    float *recvbags[26];
    for (int i = 0; i < 26; ++i) recvbags[i] = (float *)local[i]->result->DP;
    CC(cudaMemcpyToSymbolAsync(k_rex::g::recvbags, recvbags, sizeof(recvbags), 0, H2D));
}

void copy_count(x::TicketPinned ti) {
    int i;
    for (i = 0; i < 26; ++i) send_counts[i] = ti.offsets[i];
}

void copy_offset(int nw, x::TicketPack tp, x::TicketPinned ti) {
    CC(cudaMemcpyAsync(ti.offsets, tp.offsets + 26 * nw, sizeof(int) * 26, H2H));
}

}
