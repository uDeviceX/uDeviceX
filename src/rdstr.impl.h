namespace rdstr {

/* decode neighbors linear index to "delta"
     0 -> { 0, 0, 0}
     1 -> { 1, 0, 0}
     ...
    20 -> {-1, 0, -1}
     ...
    26 -> {-1, -1, -1}
*/
#define i2del(i) {((i) + 1) % 3 - 1, \
		  ((i) / 3 + 1) % 3 - 1, \
		  ((i) / 9 + 1) % 3 - 1}
#define pb push_back

void _post_recvcnt() {
  recv_cnts[0] = 0;
  for (int i = 1; i < 27; ++i) {
    MPI_Request req;
    MC(MPI_Irecv(&recv_cnts[i], 1, MPI_INTEGER, ank_ne[i], i + 1024, cart, &req));
    recvcntreq.pb(req);
  }
}

/* generate ranks and anti-ranks of the neighbors */
void gen_ne(MPI_Comm cart, /* */ int* rnk_ne, int* ank_ne) {
  rnk_ne[0] = m::rank;
  for (int i = 1; i < 27; ++i) {
    int d[3] = i2del(i); /* index to delta */
    int co_ne[3];
    for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] + d[c];
    MC(MPI_Cart_rank(cart, co_ne, &rnk_ne[i]));
    for (int c = 0; c < 3; ++c) co_ne[c] = m::coords[c] - d[c];
    MC(MPI_Cart_rank(cart, co_ne, &ank_ne[i]));
  }
}

void init() {
  mpDeviceMalloc(&bulk);

  for (int i = 0; i < 27; i++) rbuf[i] = new PinnedHostBuffer1<Particle>;
  for (int i = 0; i < 27; i++) sbuf[i] = new PinnedHostBuffer1<Particle>;

  llo = new PinnedHostBuffer2<float3>;
  hhi = new PinnedHostBuffer2<float3>;
  _ddst = new DeviceBuffer<float *>;
  _dsrc = new DeviceBuffer<const float *>;

  MC(MPI_Comm_dup(m::cart, &cart));
  gen_ne(cart,   rnk_ne, ank_ne); /* generate ranks and anti-ranks */

  _post_recvcnt();
}

void pack_all(int nc, int nv,
	      const float **const src, float **const dst) {
  if (nc == 0) return;
  int nth = nc * nv * 6; /* number of threads */

  if (nc < k_rdstr::cmaxnc) {
    CC(cudaMemcpyToSymbolAsync(k_rdstr::cdst, dst, sizeof(float*) * nc, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(k_rdstr::csrc, src, sizeof(float*) * nc, 0, H2D));
    k_rdstr::pack_all_kernel<true><<<k_cnf(nth)>>>
      (nc, nv, NULL, NULL);
  } else {
    _ddst->resize(nc);
    _dsrc->resize(nc);
    CC(cudaMemcpyAsync(_ddst->D, dst, sizeof(float*) * nc, H2D));
    CC(cudaMemcpyAsync(_dsrc->D, src, sizeof(float*) * nc, H2D));
    k_rdstr::pack_all_kernel<false><<<k_cnf(nth)>>>
      (nc, nv, _dsrc->D, _ddst->D);
  }
}

void extent(Particle *pp, int nc, int nv) {
  llo->resize(nc); hhi->resize(nc);
  if (nc) minmax(pp, nv, nc, llo->DP, hhi->DP);
}

/* build `ord' structure --- who goes where */
void reord(float3* llo, float3* hhi, int nrbcs, /* */ std::vector<int>* ord) {
  int i, vcode[3], c, code,
    L[3] = {XS, YS, ZS};
  for (i = 0; i < nrbcs; ++i) {
    float3 lo = llo[i], hi = hhi[i];
    float p[3] = {0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y), 0.5 * (lo.z + hi.z)};
    for (c = 0; c < 3; ++c)
      vcode[c] = (2 + (p[c] >= -L[c] / 2) + (p[c] >= L[c] / 2)) % 3;
    code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);
    ord[code].pb(i);
  }
}

void pack_sendcnt(Particle *pp, int nc, int nv) {
  std::vector<int> ord[27];
  reord(llo->D, hhi->D, nc, ord); /* build `ord' */
  n_bulk  = ord[0].size() * nv;

  static std::vector<const float*> src;
  static std::vector<      float*> dst;
  src.clear(); dst.clear();
  for (int i = 1; i < 27; ++i) sbuf[i]->resize(ord[i].size() * nv);

  for (int i = 0; i < 27; ++i)
    for (int j = 0; j < ord[i].size(); ++j) {
      src.pb((float*)(pp + nv * ord[i][j]));
      if (i) dst.pb((float*)(sbuf[i]->DP + nv * j));
      else   dst.pb((float*)(       bulk + nv * j));
    }
  pack_all(src.size(), nv, &src.front(), &dst.front());
  dSync(); /* was CC(cudaStreamSynchronize(stream)); */
  for (int i = 1; i < 27; ++i)
    MC(MPI_Isend(&sbuf[i]->S, 1, MPI_INTEGER,
		 rnk_ne[i], i + 1024, cart,
		 &sendcntreq[i - 1]));
}

int post(int nv) {
  {
    MPI_Status statuses[recvcntreq.size()];
    MC(MPI_Waitall(recvcntreq.size(), &recvcntreq.front(), statuses));
    recvcntreq.clear();
  }

  int ncome = 0;
  for (int i = 1; i < 27; ++i) {
    int cnt = recv_cnts[i];
    ncome += cnt;
    rbuf[i]->resize(cnt);
  }
  ncome /= nv;
  nstay = n_bulk / nv;

  MPI_Status statuses[26];
  MC(MPI_Waitall(26, sendcntreq, statuses));

  for (int i = 1; i < 27; ++i)
    if (rbuf[i]->S > 0) {
      MPI_Request request;
      MC(MPI_Irecv(rbuf[i]->D, rbuf[i]->S,
		   Particle::datatype(), ank_ne[i], i + 1155,
		   cart, &request));
      recvreq.pb(request);
    }

  for (int i = 1; i < 27; ++i)
    if (sbuf[i]->S > 0) {
      MPI_Request request;
      MC(MPI_Isend(sbuf[i]->D, sbuf[i]->S,
		   Particle::datatype(), rnk_ne[i], i + 1155,
		   cart, &request));
      sendreq.pb(request);
    }
  return nstay + ncome;
}

void unpack(Particle *pp, int nrbcs, int nv) {
  MPI_Status statuses[26];
  MC(MPI_Waitall(recvreq.size(), &recvreq.front(), statuses));
  MC(MPI_Waitall(sendreq.size(), &sendreq.front(), statuses));
  recvreq.clear();
  sendreq.clear();
  CC(cudaMemcpyAsync(pp, bulk, nstay * nv * sizeof(Particle), D2D));

  for (int i = 1, s = nstay * nv; i < 27; ++i) {
    int cnt = rbuf[i]->S;
    if (cnt > 0)
      k_rdstr::shift<<<k_cnf(cnt)>>>
	(rbuf[i]->DP, cnt, i, m::rank, false, &pp[s]);
    s += rbuf[i]->S;
  }
  _post_recvcnt();
}

void close() {
  MC(MPI_Comm_free(&cart));
  for (int i = 0; i < 27; i++) delete rbuf[i];
  for (int i = 0; i < 27; i++) delete sbuf[i];
  delete   llo; delete   hhi;
  delete _ddst; delete _dsrc;
  CC(cudaFree(bulk));
}
#undef pb
}
