namespace dpd {
void pack_first0() {
  cellpackstarts[0] = 0;
  for (int i = 0, s = 0; i < 26; ++i)
    cellpackstarts[i + 1] =
      (s += sendhalos[i]->dcellstarts->S * (sendhalos[i]->expected > 0));
  ncells = cellpackstarts[26];
  CC(cudaMemcpyToSymbol(phalo::cellpackstarts, cellpackstarts,
			sizeof(cellpackstarts), 0, H2D));

  for (int i = 0; i < 26; ++i) {
    cellpacks[i].start = sendhalos[i]->tmpstart->D;
    cellpacks[i].count = sendhalos[i]->tmpcount->D;
    cellpacks[i].enabled = sendhalos[i]->expected > 0;
    cellpacks[i].scan = sendhalos[i]->dcellstarts->D;
    cellpacks[i].size = sendhalos[i]->dcellstarts->S;
  }
  CC(cudaMemcpyToSymbol(phalo::cellpacks, cellpacks,
			sizeof(cellpacks), 0, H2D));
}

void pack_first1() {
  int i;
  for (i = 0; i < 26; ++i) srccells0[i] = sendhalos[i]->dcellstarts->D;
  CC(cudaMemcpyToSymbol(phalo::srccells, srccells0, sizeof(srccells0), 0, H2D));
  for (i = 0; i < 26; ++i) dstcells0[i] = sendhalos[i]->hcellstarts->DP;
  CC(cudaMemcpyToSymbol(phalo::dstcells, dstcells0, sizeof(dstcells0), 0, H2D));
  for (i = 0; i < 26; ++i) srccells1[i] = recvhalos[i]->hcellstarts->DP;
  CC(cudaMemcpyToSymbol(phalo::srccells, srccells1, sizeof(srccells1), sizeof(srccells1), H2D));
  for (i = 0; i < 26; ++i) dstcells1[i] = recvhalos[i]->dcellstarts->D;
  CC(cudaMemcpyToSymbol(phalo::dstcells, dstcells1, sizeof(dstcells1), sizeof(dstcells1), H2D));
}

void pack(Particle *p, int n, int *cellsstart, int *cellscount) {
  if (firstpost) pack_first0();

  if (ncells)
    phalo::count_all<<<k_cnf(ncells)>>>(cellsstart, cellscount, ncells);
  phalo::scan_diego<32><<<26, 32 * 32>>>();

  if (firstpost)
    post_expected_recv();
  else {
    MPI_Status statuses[26 * 2];
    MC(l::m::Waitall(26, sendcellsreq, statuses));
    MC(l::m::Waitall(nsendreq, sendreq, statuses));
    MC(l::m::Waitall(26, sendcountreq, statuses));
  }
  if (firstpost) pack_first1();

  if (ncells) phalo::copycells<0><<<k_cnf(ncells)>>>(ncells);
  _pack_all(p, n, firstpost);
}
}
