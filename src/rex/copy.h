namespace rex {
void copy_tstarts(x::TicketPack tp, x::TicketPinned ti) {
    CC(cudaMemcpyAsync(ti.tstarts, tp.tstarts, sizeof(int) * 27, H2H));
}

void copy_pack(x::TicketPinned t, Particle *buf, Particle *buf_pinned) {
    if (t.tstarts[26]) CC(cudaMemcpyAsync(buf_pinned, buf, sizeof(Particle) * t.tstarts[26], H2H));
}

void copy_state() {
    int i;
    for (i = 0; i < 26; ++i)
        CC(cudaMemcpyAsync(remote[i]->dstate, remote[i]->hstate, sizeof(Particle) * remote[i]->n, H2D));
}

void copy_ff() {
    int i;
    float *ff[26];
    for (i = 0; i < 26; ++i) ff[i] = (float*)local[i]->ff;
    CC(cudaMemcpyToSymbolAsync(k_rex::g::ff, ff, sizeof(ff), 0, H2D));
}

void copy_count(x::TicketPinned ti) {
    int i;
    for (i = 0; i < 26; ++i) send_counts[i] = ti.offsets[i];
}

void copy_offset(int nw, x::TicketPack tp, x::TicketPinned ti) {
    CC(cudaMemcpyAsync(ti.offsets, tp.offsets + 26 * nw, sizeof(int) * 26, H2H));
}

void copy_hstate() {
    int i, count;
    for (i = 0; i < 26; ++i) {
        count = recv_counts[i];
        memcpy(remote[i]->hstate, remote[i]->pp, sizeof(Particle) * count);
    }
}

}
