namespace dpd {
void pack_first0(SendHalo* sendhalos[]) {
    cellpackstarts.d[0] = 0;
    for (int i = 0, s = 0; i < 26; ++i)
    cellpackstarts.d[i + 1] =
        (s += sendhalos[i]->dcellstarts->S * (sendhalos[i]->expected > 0));
    ncells = cellpackstarts.d[26];

    for (int i = 0; i < 26; ++i) {
        frag::str.d[i] = sendhalos[i]->tmpstart->D;
        frag::cnt.d[i] = sendhalos[i]->tmpcount->D;
        frag::cum.d[i] = sendhalos[i]->dcellstarts->D;
        frag::nc.d[i]  = sendhalos[i]->dcellstarts->S;
    }
}

void pack_first1(SendHalo* sendhalos[]) {
    for (int i = 0; i < 26; ++i) srccells.d[i] = sendhalos[i]->dcellstarts->D;
    for (int i = 0; i < 26; ++i) dstcells.d[i] = sendhalos[i]->hcellstarts->DP;
}

void gather_cells(const int *start, const int *count) {
    if (ncells) k_halo::count<<<k_cnf(ncells)>>>(cellpackstarts, start, count, frag::str, frag::cnt);
    k_halo::scan<32><<<26, 32 * 32>>>(frag::nc, frag::cnt, /**/ frag::cum);
}

void copycells() {
    if (ncells) k_halo::copycells<<<k_cnf(ncells)>>>(cellpackstarts, srccells, /**/ dstcells);
}
  
void pack(Particle *p, int n) {
    if (ncells)
    k_halo::fill_all<<<(ncells + 1) / 2, 32>>>(cellpackstarts, p, required_send_bag_size,
                                               frag::str, frag::cnt, frag::cum, frag::nc,
                                               frag::capacity, frag::ii, frag::pp);
    CC(cudaEventRecord(evfillall));
}
}
