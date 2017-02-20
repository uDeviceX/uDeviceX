namespace Contact {
  void init(MPI_Comm comm) {
  cellsstart = new SimpleDeviceBuffer<int>(KernelsContact::NCELLS + 16);
  cellscount = new SimpleDeviceBuffer<int>(KernelsContact::NCELLS + 16);
  compressed_cellscount = new SimpleDeviceBuffer<unsigned char>(KernelsContact::NCELLS + 16);

  cellsentries = new SimpleDeviceBuffer<int>;
  subindices = new SimpleDeviceBuffer<uchar4>;
  local_trunk = new Logistic::KISS;
  
  int myrank;
  MPI_CHECK(MPI_Comm_rank(comm, &myrank));

  *local_trunk = Logistic::KISS(7119 - myrank, 187 + myrank, 18278, 15674);

  CC(cudaPeekAtLastError());
}

  void build_cells(std::vector<ParticlesWrap> wsolutes,
    cudaStream_t stream) {
  nsolutes = wsolutes.size();

  int ntotal = 0;

  for (int i = 0; i < wsolutes.size(); ++i) ntotal += wsolutes[i].n;

  subindices->resize(ntotal);
  cellsentries->resize(ntotal);

  CC(cudaMemsetAsync(cellscount->D, 0, sizeof(int) * cellscount->S, stream));

  CC(cudaPeekAtLastError());

  int ctr = 0;
  for (int i = 0; i < wsolutes.size(); ++i) {
  ParticlesWrap it = wsolutes[i];

  if (it.n)
    subindex_local<true><<<(it.n + 127) / 128, 128, 0, stream>>>(
    it.n, (float2 *)it.p, cellscount->D, subindices->D + ctr);

  ctr += it.n;
}

  compress_counts<<<(compressed_cellscount->S + 127) / 128, 128, 0, stream>>>(
  compressed_cellscount->S, (int4 *)cellscount->D,
    (uchar4 *)compressed_cellscount->D);

  scan(compressed_cellscount->D, compressed_cellscount->S, stream,
    (uint *)cellsstart->D);

  ctr = 0;
  for (int i = 0; i < wsolutes.size(); ++i) {
  ParticlesWrap it = wsolutes[i];

  if (it.n)
    KernelsContact::populate<<<(it.n + 127) / 128, 128, 0, stream>>>(
    subindices->D + ctr, cellsstart->D, it.n, i, ntotal,
      (KernelsContact::CellEntry *)cellsentries->D);

  ctr += it.n;
}
  CC(cudaPeekAtLastError());
  KernelsContact::bind(cellsstart->D, cellsentries->D, ntotal, wsolutes, stream,
    cellscount->D);
}

  void bulk(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream) {
  if (wsolutes.size() == 0) return;

  for (int i = 0; i < wsolutes.size(); ++i) {
  ParticlesWrap it = wsolutes[i];

  if (it.n)
    KernelsContact::bulk_3tpp<<<(3 * it.n + 127) / 128, 128, 0, stream>>>(
    (float2 *)it.p, it.n, cellsentries->S, wsolutes.size(), (float *)it.a,
      local_trunk->get_float(), i);

  CC(cudaPeekAtLastError());
}
}

  void halo(ParticlesWrap halos[26], cudaStream_t stream) {
  int nremote_padded = 0;

  {
  int recvpackcount[26], recvpackstarts_padded[27];

  for (int i = 0; i < 26; ++i) recvpackcount[i] = halos[i].n;

  CC(cudaMemcpyToSymbolAsync(KernelsContact::packcount, recvpackcount,
    sizeof(recvpackcount), 0, cudaMemcpyHostToDevice,
    stream));

  recvpackstarts_padded[0] = 0;
  for (int i = 0, s = 0; i < 26; ++i)
    recvpackstarts_padded[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));

  nremote_padded = recvpackstarts_padded[26];

  CC(cudaMemcpyToSymbolAsync(
  KernelsContact::packstarts_padded, recvpackstarts_padded,
    sizeof(recvpackstarts_padded), 0, cudaMemcpyHostToDevice, stream));

  const Particle *recvpackstates[26];

  for (int i = 0; i < 26; ++i) recvpackstates[i] = halos[i].p;

  CC(cudaMemcpyToSymbolAsync(KernelsContact::packstates, recvpackstates,
    sizeof(recvpackstates), 0,
    cudaMemcpyHostToDevice, stream));
  Acceleration *packresults[26];
  for (int i = 0; i < 26; ++i) packresults[i] = halos[i].a;
  CC(cudaMemcpyToSymbolAsync(KernelsContact::packresults, packresults,
    sizeof(packresults), 0, cudaMemcpyHostToDevice,
    stream));
}

  if (nremote_padded)
    KernelsContact::halo<<<(nremote_padded + 127) / 128, 128, 0, stream>>>(
    nremote_padded, cellsentries->S, nsolutes, local_trunk->get_float());
  CC(cudaPeekAtLastError());
}

  void close() {
    delete subindices;
    delete compressed_cellscount;
    delete cellsentries;
    delete cellsstart;
    delete  cellscount;
    delete local_trunk;
  }
}
