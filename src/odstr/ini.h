void Fluid::ini(MPI_Comm cart, int rank[])  {
  enum {X, Y, Z};
  s.size_pin = new PinnedHostBuffer4<int>(27);

  for(int i = 0; i < 27; ++i) {
    int d[3] = { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 };
    r.tags[i] = (3 - d[0]) % 3 + 3 * ((3 - d[1]) % 3 + 3 * ((3 - d[2]) % 3));
    int send_coor[3], ranks[3] = {m::coords[X], m::coords[Y], m::coords[Z]};
    for(int c = 0; c < 3; ++c) send_coor[c] = ranks[c] + d[c];
    l::m::Cart_rank(cart, send_coor, rank + i) ;

    int nhalodir[3] =  {
      d[0] != 0 ? 1 : XS,
      d[1] != 0 ? 1 : YS,
      d[2] != 0 ? 1 : ZS
    };

    int nhalocells = nhalodir[0] * nhalodir[1] * nhalodir[2];
    int safety_factor = 2;
    int estimate = numberdensity * safety_factor * nhalocells;
    CC(cudaMalloc(&s.iidx_[i], sizeof(int) * estimate));

    if (i && estimate) {
      CC(cudaHostAlloc(&s.hst[i], sizeof(float) * 6 * estimate, cudaHostAllocMapped));
      CC(cudaHostGetDevicePointer(&s.hst_[i], s.hst[i], 0));
      CC(cudaHostAlloc(&r.hst[i], sizeof(float) * 6 * estimate, cudaHostAllocMapped));
      CC(cudaHostGetDevicePointer(&r.hst_[i], r.hst[i], 0));
    } else {
      CC(cudaMalloc(&s.hst_[i], sizeof(float) * 6 * estimate));
      r.hst_[i] = s.hst_[i];
      s.hst[i] = NULL;
      r.hst[i] = NULL;
    }
  }

  CC(cudaMalloc(&s.size_dev, 27*sizeof(s.size_dev[0])));
  CC(cudaMalloc(&s.strt,     28*sizeof(s.strt[0])));
  CC(cudaMalloc(&r.strt, 28*sizeof(r.strt[0])));
  CC(cudaMalloc(&r.strt_pa,     28*sizeof(r.strt_pa[0])));

  CC(cudaMalloc(&s.iidx, SZ_PTR_ARR(s.iidx_)));
  CC(cudaMemcpy(s.iidx, s.iidx_, sizeof(s.iidx_), H2D));

  CC(cudaMalloc(&s.dev, SZ_PTR_ARR(s.hst_)));
  CC(cudaMemcpy(s.dev, s.hst_, sizeof(s.hst_), H2D));

  CC(cudaMalloc(&r.dev, SZ_PTR_ARR(r.hst_)));
  CC(cudaMemcpy(r.dev, r.hst_, sizeof(r.hst_), H2D));
}

