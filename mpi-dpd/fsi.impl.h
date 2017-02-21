namespace FSI {
  void bind_solvent(SolventWrap wrap) {*wsolvent = wrap;}
  void init(MPI_Comm comm) {
    int myrank;
    MC(MPI_Comm_rank(comm, &myrank));
    
    local_trunk = new Logistic::KISS;
    wsolvent    = new SolventWrap;
    
    *local_trunk = Logistic::KISS(1908 - myrank, 1409 + myrank, 290, 12968);
    CC(cudaPeekAtLastError());
  }

  void close() {
    delete local_trunk;
    delete wsolvent;
  }

void bulk(std::vector<ParticlesWrap> wsolutes,
                      cudaStream_t stream) {
  if (wsolutes.size() == 0) return;

  KernelsFSI::setup(wsolvent->p, wsolvent->n, wsolvent->cellsstart,
                    wsolvent->cellscount);

  CC(cudaPeekAtLastError());

  for (std::vector<ParticlesWrap>::iterator it = wsolutes.begin();
       it != wsolutes.end(); ++it)
    if (it->n)
      KernelsFSI::
	interactions_3tpp<<<(3 * it->n + 127) / 128, 128, 0, stream>>>
	((float2 *)it->p, it->n, wsolvent->n, (float *)it->a,
	 (float *)wsolvent->a, local_trunk->get_float());

  CC(cudaPeekAtLastError());
}

void halo(ParticlesWrap halos[26], cudaStream_t stream) {
  KernelsFSI::setup(wsolvent->p, wsolvent->n, wsolvent->cellsstart,
                    wsolvent->cellscount);

  CC(cudaPeekAtLastError());

  int nremote_padded = 0;

  {
    int recvpackcount[26], recvpackstarts_padded[27];

    for (int i = 0; i < 26; ++i) recvpackcount[i] = halos[i].n;

    CC(cudaMemcpyToSymbolAsync(KernelsFSI::packcount, recvpackcount,
                               sizeof(recvpackcount), 0, cudaMemcpyHostToDevice,
                               stream));

    recvpackstarts_padded[0] = 0;
    for (int i = 0, s = 0; i < 26; ++i)
      recvpackstarts_padded[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));

    nremote_padded = recvpackstarts_padded[26];

    CC(cudaMemcpyToSymbolAsync(
			       KernelsFSI::packstarts_padded, recvpackstarts_padded,
			       sizeof(recvpackstarts_padded), 0, cudaMemcpyHostToDevice, stream));
  }

  {
    const Particle *recvpackstates[26];

    for (int i = 0; i < 26; ++i) recvpackstates[i] = halos[i].p;

    CC(cudaMemcpyToSymbolAsync(KernelsFSI::packstates, recvpackstates,
                               sizeof(recvpackstates), 0,
                               cudaMemcpyHostToDevice, stream));
  }

  {
    Acceleration *packresults[26];

    for (int i = 0; i < 26; ++i) packresults[i] = halos[i].a;

    CC(cudaMemcpyToSymbolAsync(KernelsFSI::packresults, packresults,
                               sizeof(packresults), 0, cudaMemcpyHostToDevice,
                               stream));
  }

  if (nremote_padded)
    KernelsFSI::
      interactions_halo<<<(nremote_padded + 127) / 128, 128, 0, stream>>>
      (nremote_padded, wsolvent->n, (float *)wsolvent->a,
       local_trunk->get_float());

  CC(cudaPeekAtLastError());
}
}
