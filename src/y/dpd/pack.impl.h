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
    for (int i = 0; i < 26; ++i) {
        cellpacks::start.d[i] = sendhalos[i]->tmpstart->D;
        cellpacks::count.d[i] = sendhalos[i]->tmpcount->D;
        cellpacks::scan.d[i] = sendhalos[i]->dcellstarts->D;
        cellpacks::size.d[i] = sendhalos[i]->dcellstarts->S;
    }
  }
}

void pack_first1(SendHalo* sendhalos[]) {
    for (int i = 0; i < 26; ++i) srccells.d[i] = sendhalos[i]->dcellstarts->D;
    for (int i = 0; i < 26; ++i) dstcells.d[i] = sendhalos[i]->hcellstarts->DP;
}

void wait_send() {
  MPI_Status ss[26 * 2];
  MC(l::m::Waitall(26, sendcellsreq, ss));
  MC(l::m::Waitall(nsendreq, sendreq, ss));
  MC(l::m::Waitall(26, sendcountreq, ss));
}

void scan(int *start, int *count) {
    if (ncells) k_halo::count<<<k_cnf(ncells)>>>(cellpackstarts, start, count, cellpacks::start, cellpacks::count);
    k_halo::scan_diego<32><<<26, 32 * 32>>>(cellpacks::size, cellpacks::count, /**/ cellpacks::scan);
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
