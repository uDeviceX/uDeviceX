namespace rdstr {
void _post_recvcount() {
  recv_counts[0] = 0;
  for (int i = 1; i < 27; ++i) {
    MPI_Request req;
    MC(MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, anti_rankneighbors[i],
			i + 1024, cartcomm, &req));
    recvcountreq.push_back(req);
  }
}

void redistribute_rbcs_init(MPI_Comm _cartcomm) {
  bulk = new DeviceBuffer<Particle>;
  for (int i = 0; i < HALO_BUF_SIZE; i++) halo_recvbufs[i] = new PinnedHostBuffer<Particle>;
  for (int i = 0; i < HALO_BUF_SIZE; i++) halo_sendbufs[i] = new PinnedHostBuffer<Particle>;
  minextents = new PinnedHostBuffer<float3>;
  maxextents = new PinnedHostBuffer<float3>;
  _ddestinations = new DeviceBuffer<float *>;
  _dsources = new DeviceBuffer<const float *>;

  nvertices = rbc::get_nvertices();
  rbc::setup();
  /* TODO: move it to a better place; [xyz]lo, [xyz]hi pbc[xyz] (9
     arguments for iotags_domain, pbc: 1: for periods boundary
     conditions) */
  iotags_init_file("rbc.dat");
  iotags_domain(0, 0, 0,
		XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN,
		1, 1, 1);
  MC(MPI_Comm_dup(_cartcomm, &cartcomm));
  MC(MPI_Comm_rank(cartcomm, &myrank));
  int dims[3];
  MC(MPI_Cart_get(cartcomm, 3, dims, periods, coords));

  rankneighbors[0] = myrank;
  for (int i = 1; i < 27; ++i) {
    int d[3] = {(i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1};
    int coordsneighbor[3];
    for (int c = 0; c < 3; ++c) coordsneighbor[c] = coords[c] + d[c];
    MC(MPI_Cart_rank(cartcomm, coordsneighbor, rankneighbors + i));
    for (int c = 0; c < 3; ++c) coordsneighbor[c] = coords[c] - d[c];
    MC(MPI_Cart_rank(cartcomm, coordsneighbor, anti_rankneighbors + i));
  }

  CC(cudaEventCreate(&evextents, cudaEventDisableTiming));
  _post_recvcount();
}

void _compute_extents(Particle *xyzuvw,
					int nrbcs) {
  if (nrbcs)
    minmax(xyzuvw, nvertices, nrbcs, minextents->DP, maxextents->DP);
}

void pack_all(const int nrbcs, const int nvertices,
	      const float **const sources, float **const destinations) {
  if (nrbcs == 0) return;
  int nthreads = nrbcs * nvertices * 6;

  if (nrbcs < k_rdstr::cmaxnrbcs) {
    CC(cudaMemcpyToSymbolAsync(k_rdstr::cdestinations, destinations,
			       sizeof(float *) * nrbcs, 0,
			       H2D));
    CC(cudaMemcpyToSymbolAsync(k_rdstr::csources,
			       sources, sizeof(float *) * nrbcs, 0,
			       H2D));
    k_rdstr::pack_all_kernel<true><<<(nthreads + 127) / 128, 128, 0>>>(
	nrbcs, nvertices, NULL, NULL);
  } else {
    _ddestinations->resize(nrbcs);
    _dsources->resize(nrbcs);
    CC(cudaMemcpyAsync(_ddestinations->D, destinations, sizeof(float *) * nrbcs,
		       H2D));
    CC(cudaMemcpyAsync(_dsources->D, sources, sizeof(float *) * nrbcs,
		       H2D));
    k_rdstr::pack_all_kernel<false><<<(nthreads + 127) / 128, 128, 0>>>(
	nrbcs, nvertices, _dsources->D, _ddestinations->D);
  }

}

void extent(Particle *xyzuvw, int nrbcs) {
  minextents->resize(nrbcs);
  maxextents->resize(nrbcs);

  _compute_extents(xyzuvw, nrbcs);

  CC(cudaEventRecord(evextents));
}


void pack_sendcount(Particle *xyzuvw,
				      int nrbcs) {
  CC(cudaEventSynchronize(evextents));
  std::vector<int> reordering_indices[27];

  for (int i = 0; i < nrbcs; ++i) {
    float3 minext = minextents->D[i];
    float3 maxext = maxextents->D[i];
    float p[3] = {0.5 * (minext.x + maxext.x), 0.5 * (minext.y + maxext.y),
		  0.5 * (minext.z + maxext.z)};
    int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
    int vcode[3];
    for (int c = 0; c < 3; ++c)
      vcode[c] = (2 + (p[c] >= -L[c] / 2) + (p[c] >= L[c] / 2)) % 3;
    int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);
    reordering_indices[code].push_back(i);
  }

  bulk->resize(reordering_indices[0].size() * nvertices);
  for (int i = 1; i < 27; ++i)
    halo_sendbufs[i]->resize(reordering_indices[i].size() * nvertices);
  {
    static std::vector<const float *> src;
    static std::vector<float *> dst;
    src.clear();
    dst.clear();
    for (int i = 0; i < 27; ++i)
      for (int j = 0; j < reordering_indices[i].size(); ++j) {
	src.push_back((float *)(xyzuvw + nvertices * reordering_indices[i][j]));

	if (i)
	  dst.push_back((float *)(halo_sendbufs[i]->DP + nvertices * j));
	else
	  dst.push_back((float *)(bulk->D + nvertices * j));
      }
    pack_all(src.size(), nvertices, &src.front(),
	     &dst.front());

  }
  CC(cudaDeviceSynchronize()); /* was CC(cudaStreamSynchronize(stream)); */
  for (int i = 1; i < 27; ++i)
    MC(MPI_Isend(&halo_sendbufs[i]->S, 1, MPI_INTEGER,
			rankneighbors[i], i + 1024, cartcomm,
			&sendcountreq[i - 1]));
}

int post() {
  {
    MPI_Status statuses[recvcountreq.size()];
    MC(MPI_Waitall(recvcountreq.size(), &recvcountreq.front(), statuses));
    recvcountreq.clear();
  }

  arriving = 0;
  for (int i = 1; i < 27; ++i) {
    int count = recv_counts[i];
    arriving += count;
    halo_recvbufs[i]->resize(count);
  }

  arriving /= nvertices;
  notleaving = bulk->S / nvertices;

  MPI_Status statuses[26];
  MC(MPI_Waitall(26, sendcountreq, statuses));

  for (int i = 1; i < 27; ++i)
    if (halo_recvbufs[i]->S > 0) {
      MPI_Request request;
      MC(MPI_Irecv(halo_recvbufs[i]->D, halo_recvbufs[i]->S,
			  Particle::datatype(), anti_rankneighbors[i], i + 1155,
			  cartcomm, &request));
      recvreq.push_back(request);
    }

  for (int i = 1; i < 27; ++i)
    if (halo_sendbufs[i]->S > 0) {
      MPI_Request request;
      MC(MPI_Isend(halo_sendbufs[i]->D, halo_sendbufs[i]->S,
			  Particle::datatype(), rankneighbors[i], i + 1155,
			  cartcomm, &request));

      sendreq.push_back(request);
    }
  return notleaving + arriving;
}

void unpack(Particle *xyzuvw, int nrbcs) {
  MPI_Status statuses[26];
  MC(MPI_Waitall(recvreq.size(), &recvreq.front(), statuses));
  MC(MPI_Waitall(sendreq.size(), &sendreq.front(), statuses));
  recvreq.clear();
  sendreq.clear();
  CC(cudaMemcpyAsync(xyzuvw, bulk->D, notleaving * nvertices * sizeof(Particle),
		     D2D));

  for (int i = 1, s = notleaving * nvertices; i < 27; ++i) {
    int count = halo_recvbufs[i]->S;
    if (count > 0)
      k_rdstr::shift<<<(count + 127) / 128, 128, 0>>>(
	  halo_recvbufs[i]->DP, count, i, myrank, false, xyzuvw + s);
    s += halo_recvbufs[i]->S;
  }
  _post_recvcount();
}

void redistribute_rbcs_close() {
  MC(MPI_Comm_free(&cartcomm));
  delete bulk;
  for (int i = 0; i < HALO_BUF_SIZE; i++) delete halo_recvbufs[i];
  for (int i = 0; i < HALO_BUF_SIZE; i++) delete halo_sendbufs[i];
  delete minextents;
  delete maxextents;

  delete _ddestinations;
  delete _dsources;
}
}

