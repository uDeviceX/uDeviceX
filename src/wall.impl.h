namespace wall {
int init(Particle *pp, int n) {
    thrust::device_vector<int> keys(n);
    k_sdf::fill_keys<<<k_cnf(n)>>>(pp, n, thrust::raw_pointer_cast(&keys[0]));
    thrust::sort_by_key(keys.begin(), keys.end(), thrust::device_ptr<Particle>(pp));

    int nsurvived = thrust::count(keys.begin(), keys.end(), 0);
    int nbelt = thrust::count(keys.begin() + nsurvived, keys.end(), 1);
    thrust::device_vector<Particle> solid_local
	(thrust::device_ptr<Particle>(pp + nsurvived),
	 thrust::device_ptr<Particle>(pp + nsurvived + nbelt));
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
	    MC(MPI_Cart_rank(m::cart, coordsneighbor, dstranks + i));
	}

	// send local counts - receive remote counts
	{
	    for (int i = 0; i < 26; ++i) remsizes[i] = -1;

	    MPI_Request reqrecv[26];
	    for (int i = 0; i < 26; ++i)
	    MC(MPI_Irecv(remsizes + i, 1, MPI_INTEGER, dstranks[i],
			 BT_C_WALL + recv_tags[i], m::cart, reqrecv + i));

	    int localsize = local.size();
	    MPI_Request reqsend[26];
	    for (int i = 0; i < 26; ++i)
	    MC(MPI_Isend(&localsize, 1, MPI_INTEGER, dstranks[i], BT_C_WALL + i,
			 m::cart, reqsend + i));
	    MPI_Status statuses[26];
	    MC(MPI_Waitall(26, reqrecv, statuses));
	    MC(MPI_Waitall(26, reqsend, statuses));
	}

	std::vector<Particle> remote[26];
	// send local data - receive remote data
	{
	    for (int i = 0; i < 26; ++i) remote[i].resize(remsizes[i]);

	    MPI_Request reqrecv[26];
	    for (int i = 0; i < 26; ++i)
	    MC(MPI_Irecv(remote[i].data(), remote[i].size() * 6, MPI_FLOAT,
			 dstranks[i], BT_P_WALL + recv_tags[i], m::cart,
			 reqrecv + i));
	    MPI_Request reqsend[26];
	    for (int i = 0; i < 26; ++i)
	    MC(MPI_Isend(local.data(), local.size() * 6, MPI_FLOAT,
			 dstranks[i], BT_P_WALL + i, m::cart, reqsend + i));

	    MPI_Status statuses[26];
	    MC(MPI_Waitall(26, reqrecv, statuses));
	    MC(MPI_Waitall(26, reqsend, statuses));
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

    w_n = solid_local.size() + solid_remote.S;

    CC(cudaMalloc(&w_pp000, sizeof(Particle) * w_n));
    cD2D(w_pp000, thrust::raw_pointer_cast(&solid_local[0]), solid_local.size());
    cD2D(w_pp000 + solid_local.size(), solid_remote.D, solid_remote.S);

    cells = new x::Clist(XS + 2 * XWM, YS + 2 * YWM, ZS + 2 * ZWM);
    if (w_n > 0) cells->build(w_pp000, w_n);

    CC(cudaMalloc(&w_pp, sizeof(float4) * w_n));

    MSG0("consolidating wall particles");
    if (w_n > 0) k_sdf::strip_solid4<<<k_cnf(w_n)>>>(w_pp000, w_n, w_pp);
    return nsurvived;
} /* end of ini */

void interactions(const int type, const Particle *const p, const int n,
		  Force *const acc) {
    if (n > 0 && w_n > 0) {
	k_wall::interactions_3tpp <<<k_cnf(3 * n)>>>
	  ((float2 *)p, n, w_n, (float *)acc, trunk->get_float(), type, cells->start,
	   (float4*)w_pp, w_pp000);
    }
}

void close () { delete cells; }
}
