
int init(Particle *pp, int n, Particle *frozen, int *w_n) {
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

    *w_n = solid_local.size() + solid_remote.S;

    cD2D(frozen, thrust::raw_pointer_cast(&solid_local[0]), solid_local.size());
    cD2D(frozen + solid_local.size(), solid_remote.D, solid_remote.S);

    return nsurvived;
} /* end of ini */

void build_cells(const int n, Particle *pp, Clist *cells) {if (n) cells->build(pp, n);}

void make_texstart(int *start, int n, cudaTextureObject_t *texstart) {
    cudaResourceDesc resD;
    cudaTextureDesc  texD;

    memset(&resD, 0, sizeof(resD));
    resD.resType = cudaResourceTypeLinear;
    resD.res.linear.devPtr  = start;
    resD.res.linear.sizeInBytes = n * sizeof(int);
    resD.res.linear.desc = cudaCreateChannelDesc<int>();

    memset(&texD, 0, sizeof(texD));
    texD.normalizedCoords = 0;
    texD.readMode = cudaReadModeElementType;

    CC(cudaCreateTextureObject(texstart, &resD, &texD, NULL));
}

void make_texpp(float4 *pp, int n, cudaTextureObject_t *texpp) {
    cudaResourceDesc resD;
    cudaTextureDesc  texD;

    memset(&resD, 0, sizeof(resD));
    resD.resType = cudaResourceTypeLinear;
    resD.res.linear.devPtr  = pp;
    resD.res.linear.sizeInBytes = n * sizeof(float4);
    resD.res.linear.desc = cudaCreateChannelDesc<float4>();

    memset(&texD, 0, sizeof(texD));
    texD.normalizedCoords = 0;
    texD.readMode = cudaReadModeElementType;

    CC(cudaCreateTextureObject(texpp, &resD, &texD, NULL));
}

void interactions(const int type, const Particle *const pp, const int n, const float rnd,
                  const cudaTextureObject_t texstart, const cudaTextureObject_t texpp,
                  const int w_n, Force *ff) {
    if (n > 0 && w_n > 0) {
    dev::interactions_3tpp <<<k_cnf(3 * n)>>>
        ((float2 *)pp, n, w_n, (float *)ff, rnd, type, texstart, texpp);
    }
}
