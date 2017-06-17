static void exch(Particle *pp, int *n) { /* excchange pp(hst) between processors */
  assert(sizeof(Particle) == 6 * sizeof(float)); /* :TODO: dependencies */
  enum {X, Y, Z};
  int i, j, c;
  int dstranks[26], remsizes[26], recv_tags[26];
  for (i = 0; i < 26; ++i) {
    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
    recv_tags[i] =
      (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
    int co_ne[3], ranks[3] = {m::coords[X], m::coords[Y], m::coords[Z]};
    for (c = 0; c < 3; ++c) co_ne[c] = ranks[c] + d[c];
    l::m::Cart_rank(m::cart, co_ne, dstranks + i);
  }

  // send local counts - receive remote counts
  {
    for (i = 0; i < 26; ++i) remsizes[i] = -1;
    MPI_Request reqrecv[26], reqsend[26];
    MPI_Status  statuses[26];
    for (i = 0; i < 26; ++i)
      l::m::Irecv(remsizes + i, 1, MPI_INTEGER, dstranks[i],
	       123 + recv_tags[i], m::cart, reqrecv + i);
    for (i = 0; i < 26; ++i)
      l::m::Isend(n, 1, MPI_INTEGER, dstranks[i], 123 + i, m::cart, reqsend + i);
    l::m::Waitall(26, reqrecv, statuses);
    l::m::Waitall(26, reqsend, statuses);
  }

  std::vector<Particle> remote[26];
  // send local data - receive remote data
  {
    for (i = 0; i < 26; ++i) remote[i].resize(remsizes[i]);
    MPI_Request reqrecv[26], reqsend[26];
    MPI_Status  statuses[26];
    for (i = 0; i < 26; ++i)
      l::m::Irecv(remote[i].data(), remote[i].size() * 6, MPI_FLOAT,
	       dstranks[i], 321 + recv_tags[i], m::cart,
	       reqrecv + i);
    for (i = 0; i < 26; ++i)
      l::m::Isend(pp, (*n) * 6, MPI_FLOAT,
	       dstranks[i], 321 + i, m::cart, reqsend + i);
    l::m::Waitall(26, reqrecv, statuses);
    l::m::Waitall(26, reqsend, statuses);
  }
  l::m::Barrier(m::cart);

  int L[3] = {XS, YS, ZS}, WM[3] = {XWM, YWM, ZWM};
  float lo[3], hi[3];
  for (c = 0; c < 3; c ++) {
    lo[c] = -0.5*L[c] - WM[c];
    hi[c] =  0.5*L[c] + WM[c];
  }

  for (i = 0; i < 26; ++i) {
    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
    for (j = 0; j < remote[i].size(); ++j) {
      Particle p = remote[i][j];
      for (c = 0; c < 3; ++c) {
	p.r[c] += d[c] * L[c];
	if (p.r[c] < lo[c] || p.r[c] >= hi[c]) goto next;
      }
      pp[(*n)++] = p;
    next: ;
    }
  }
}

int init(Particle *pp, int n, Particle *frozen, int *w_n) {
    thrust::device_vector<int> keys(n);
    k_sdf::fill_keys<<<k_cnf(n)>>>(pp, n, thrust::raw_pointer_cast(&keys[0]));
    thrust::sort_by_key(keys.begin(), keys.end(), thrust::device_ptr<Particle>(pp));

    int nsurvived = thrust::count(keys.begin(), keys.end(), 0);
    int nbelt = thrust::count(keys.begin() + nsurvived, keys.end(), 1);
    thrust::device_vector<Particle> solid_local(pp + nsurvived, pp + nsurvived + nbelt);
    MSG("nsurvived/nbelt : %d/%d", nsurvived, nbelt);
    dSync();

    DeviceBuffer<Particle> solid_remote;
    {
	thrust::host_vector<Particle> local = solid_local;

	int dstranks[26], remsizes[26], recv_tags[26];
	for (int i = 0; i < 26; ++i) {
	    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

	    recv_tags[i] =
		(2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	    int coordsneighbor[3];
	    for (int c = 0; c < 3; ++c) coordsneighbor[c] = m::coords[c] + d[c];
	    MC(l::m::Cart_rank(m::cart, coordsneighbor, dstranks + i));
	}

	// send local counts - receive remote counts
	{
	    for (int i = 0; i < 26; ++i) remsizes[i] = -1;

	    MPI_Request reqrecv[26];
	    for (int i = 0; i < 26; ++i)
	    MC(l::m::Irecv(remsizes + i, 1, MPI_INTEGER, dstranks[i],
			 BT_C_WALL + recv_tags[i], m::cart, reqrecv + i));

	    int localsize = local.size();
	    MPI_Request reqsend[26];
	    for (int i = 0; i < 26; ++i)
	    MC(l::m::Isend(&localsize, 1, MPI_INTEGER, dstranks[i], BT_C_WALL + i,
			 m::cart, reqsend + i));
	    MPI_Status statuses[26];
	    MC(l::m::Waitall(26, reqrecv, statuses));
	    MC(l::m::Waitall(26, reqsend, statuses));
	}

	std::vector<Particle> remote[26];
	// send local data - receive remote data
	{
	    for (int i = 0; i < 26; ++i) remote[i].resize(remsizes[i]);

	    MPI_Request reqrecv[26];
	    for (int i = 0; i < 26; ++i)
	    MC(l::m::Irecv(remote[i].data(), remote[i].size() * 6, MPI_FLOAT,
			 dstranks[i], BT_P_WALL + recv_tags[i], m::cart,
			 reqrecv + i));
	    MPI_Request reqsend[26];
	    for (int i = 0; i < 26; ++i)
	    MC(l::m::Isend(local.data(), local.size() * 6, MPI_FLOAT,
			 dstranks[i], BT_P_WALL + i, m::cart, reqsend + i));

	    MPI_Status statuses[26];
	    MC(l::m::Waitall(26, reqrecv, statuses));
	    MC(l::m::Waitall(26, reqsend, statuses));
	}

	// select particles within my region [-L / 2 - MARGIN, +L / 2 + MARGIN]
	std::vector<Particle> selected;
	int L[3] = {XS, YS, ZS};
	int MARGIN[3] = {XWM, YWM, ZWM};

	for (int i = 0; i < 26; ++i) {
	    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
	    for (int j = 0; j < (int) remote[i].size(); ++j) {
		Particle p = remote[i][j];
		for (int c = 0; c < 3; ++c) p.r[c] += d[c] * L[c];
		bool inside = true;
		for (int c = 0; c < 3; ++c)
		inside &=
		    p.r[c] >= -L[c] / 2 - MARGIN[c] && p.r[c] < L[c] / 2 + MARGIN[c];
		if (inside) selected.push_back(p);
	    }
	}

	solid_remote.resize(selected.size());
	cH2D(solid_remote.D, selected.data(), solid_remote.S);
    }

    *w_n = solid_local.size() + solid_remote.S;

    cD2D(frozen, thrust::raw_pointer_cast(&solid_local[0]), solid_local.size());
    cD2D(frozen + solid_local.size(), solid_remote.D, solid_remote.S);

    return nsurvived;
} /* end of ini */

void build_cells(const int n, Particle *pp, Clist *cells) {if (n) cells->build(pp, n);}

void create(int *o_n, Particle *o_pp, int *w_n, float4 **w_pp, Clist *cells,
            Texo<int> *texstart, Texo<float4> *texpp) {
    Particle *frozen;
    CC(cudaMalloc(&frozen, sizeof(Particle) * MAX_PART_NUM));

    *o_n = init(o_pp, *o_n, frozen, w_n);

    build_cells(*w_n, /**/ frozen, cells);

    MSG0("consolidating wall particles");

    CC(cudaMalloc(w_pp, *w_n * sizeof(float4)));

    if (*w_n > 0)
    dev::strip_solid4 <<<k_cnf(*w_n)>>> (frozen, *w_n, /**/ *w_pp);

    texstart->setup(cells->start, cells->ncells);
    texpp->setup(*w_pp, *w_n);
    
    CC(cudaFree(frozen));
}

void interactions(const int type, const Particle *const pp, const int n, const Texo<int> texstart,
                  const Texo<float4> texpp, const int w_n, /**/ l::rnd::d::KISS *rnd, Force *ff) {
    if (n > 0 && w_n > 0) {
        dev::interactions_3tpp <<<k_cnf(3 * n)>>>
            ((float2 *)pp, n, w_n, (float *)ff, rnd->get_float(), type, texstart, texpp);
    }
}
