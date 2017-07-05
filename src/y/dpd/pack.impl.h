namespace dpd {
void pack_first0(SendHalo* sendhalos[]) {
  {
    cellpackstarts.d[0] = 0;
    for (int i = 0, s = 0; i < 26; ++i)
      cellpackstarts.d[i + 1] =
	(s += sendhalos[i]->dcellstarts->S * (sendhalos[i]->expected > 0));
    ncells = cellpackstarts.d[26];
  }

  {
    static k_halo::CellPackSOA cellpacks[26];
    for (int i = 0; i < 26; ++i) {
      cellpacks[i].start = sendhalos[i]->tmpstart->D;
      cellpacks[i].count = sendhalos[i]->tmpcount->D;
      cellpacks[i].scan = sendhalos[i]->dcellstarts->D;
      cellpacks[i].size = sendhalos[i]->dcellstarts->S;
    }
    CC(cudaMemcpyToSymbol(k_halo::cellpacks, cellpacks,
			  sizeof(cellpacks), 0, H2D));
  }
}

void pack_first1(SendHalo* sendhalos[]) {
    for (int i = 0; i < 26; ++i) srccells.d[i] = sendhalos[i]->dcellstarts->D;
    for (int i = 0; i < 26; ++i) dstcells.d[i] = sendhalos[i]->hcellstarts->DP;
}

void scan(int *start, int *count) {
  if (ncells) k_halo::count<<<k_cnf(ncells)>>>(cellpackstarts, start, count);
  k_halo::scan_diego<32><<<26, 32 * 32>>>();
}

void copycells() {
    if (ncells) k_halo::copycells<<<k_cnf(ncells)>>>(cellpackstarts, srccells, /**/ dstcells);
}
  
void pack(Particle *p, int n) {
    if (ncells)
      k_halo::fill_all<<<(ncells + 1) / 2, 32>>>(cellpackstarts, p, required_send_bag_size);
    CC(cudaEventRecord(evfillall));
}
}
