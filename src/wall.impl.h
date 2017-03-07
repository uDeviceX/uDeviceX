namespace wall {
  void setup() {
    k_wall::texSDF.normalized = 0;
    k_wall::texSDF.filterMode = cudaFilterModePoint;
    k_wall::texSDF.mipmapFilterMode = cudaFilterModePoint;
    k_wall::texSDF.addressMode[0] = cudaAddressModeWrap;
    k_wall::texSDF.addressMode[1] = cudaAddressModeWrap;
    k_wall::texSDF.addressMode[2] = cudaAddressModeWrap;

    setup_texture(k_wall::texWallParticles, float4);
    setup_texture(k_wall::texWallCellStart, int);
    setup_texture(k_wall::texWallCellCount, int);

    CC(cudaFuncSetCacheConfig(k_wall::interactions_3tpp, cudaFuncCachePreferL1));
  }
  
  int init(Particle *pp, int n) {
    float grid_data[MAX_SUBDOMAIN_VOLUME];
    float *field = new float[XTEXTURESIZE * YTEXTURESIZE * ZTEXTURESIZE];
    int N[3];
    float extent[3];
    field::ini("sdf.dat", N, extent, grid_data);
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL};
    int TEXTURESIZE[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};
    if (m::rank == 0) printf("sampling the geometry file...\n");
    {
      float start[3], spacing[3];
      for (int c = 0; c < 3; ++c) {
	start[c] = N[c] * (m::coords[c] * L[c] - MARGIN[c]) /
	  (float)(m::dims[c] * L[c]);
	spacing[c] = N[c] * (L[c] + 2 * MARGIN[c]) /
	  (float)(m::dims[c] * L[c]) / (float)TEXTURESIZE[c];
      }
      float amplitude_rescaling = (XS /*+ 2 * XMARGIN_WALL*/) /
	(extent[0] / m::dims[0]);
      field::sample(start, spacing, TEXTURESIZE, N, amplitude_rescaling, grid_data,
		    field);
    }

    if (hdf5field_dumps) field::dump(N, extent, grid_data);
      
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CC(cudaMalloc3DArray
       (&arrSDF, &fmt, make_cudaExtent(XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE)));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr
      ((void *)field, XTEXTURESIZE * sizeof(float), XTEXTURESIZE, YTEXTURESIZE);

    copyParams.dstArray = arrSDF;
    copyParams.extent = make_cudaExtent(XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE);
    copyParams.kind = H2D;
    CC(cudaMemcpy3D(&copyParams));
    delete[] field;

    setup();

    CC(cudaBindTextureToArray(k_wall::texSDF, arrSDF, fmt));

    if (m::rank == 0) printf("carving out wall particles...\n");

    thrust::device_vector<int> keys(n);

    k_wall::fill_keys<<<k_cnf(n)>>>
      (pp, n, thrust::raw_pointer_cast(&keys[0]));

    thrust::sort_by_key(keys.begin(), keys.end(),
			thrust::device_ptr<Particle>(pp));

    int nsurvived = thrust::count(keys.begin(), keys.end(), 0);
    int nbelt = thrust::count(keys.begin() + nsurvived, keys.end(), 1);

    thrust::device_vector<Particle> solid_local
      (thrust::device_ptr<Particle>(pp + nsurvived),
       thrust::device_ptr<Particle>(pp + nsurvived + nbelt));

    /* can't use halo-exchanger class because of MARGIN */
    if (m::rank == 0) printf("fetching remote wall particles...\n");

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
		       123 + recv_tags[i], m::cart, reqrecv + i));

	int localsize = local.size();

	MPI_Request reqsend[26];
	for (int i = 0; i < 26; ++i)
	  MC(MPI_Isend(&localsize, 1, MPI_INTEGER, dstranks[i], 123 + i,
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
		       dstranks[i], 321 + recv_tags[i], m::cart,
		       reqrecv + i));

	MPI_Request reqsend[26];
	for (int i = 0; i < 26; ++i)
	  MC(MPI_Isend(local.data(), local.size() * 6, MPI_FLOAT,
		       dstranks[i], 321 + i, m::cart, reqsend + i));

	MPI_Status statuses[26];
	MC(MPI_Waitall(26, reqrecv, statuses));
	MC(MPI_Waitall(26, reqsend, statuses));
      }

      // select particles within my region [-L / 2 - MARGIN, +L / 2 + MARGIN]
      std::vector<Particle> selected;
      for (int i = 0; i < 26; ++i) {
	int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

	for (int j = 0; j < remote[i].size(); ++j) {
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
      CC(cudaMemcpy(solid_remote.D, selected.data(),
		    sizeof(Particle) * solid_remote.S,
		    H2D));
    }

    solid_size = solid_local.size() + solid_remote.S;

    Particle *solid;
    CC(cudaMalloc(&solid, sizeof(Particle) * solid_size));
    CC(cudaMemcpy(solid, thrust::raw_pointer_cast(&solid_local[0]),
		  sizeof(Particle) * solid_local.size(),
		  D2D));
    CC(cudaMemcpy(solid + solid_local.size(), solid_remote.D,
		  sizeof(Particle) * solid_remote.S,
		  D2D));

    wall_cells = new CellLists(XS + 2 * XMARGIN_WALL,
			       YS + 2 * YMARGIN_WALL,
			       ZS + 2 * ZMARGIN_WALL);
    if (solid_size > 0) wall_cells->build(solid, solid_size, 0);

    CC(cudaMalloc(&solid4, sizeof(float4) * solid_size));

    if (m::rank == 0) printf("consolidating wall particles...\n");

    if (solid_size > 0)
      k_wall::strip_solid4<<<k_cnf(solid_size)>>>
	(solid, solid_size, solid4);

    CC(cudaFree(solid));

    return nsurvived;
  } /* end of ini */

  void bounce(Particle *const p, const int n) {
    if (n > 0)
      k_wall::bounce<<<k_cnf(n)>>> ((float2 *)p, n);
  }

  void interactions(const Particle *const p, const int n,
		    Force *const acc) {
    // cellsstart and cellscount IGNORED for now
    if (n > 0 && solid_size > 0) {
      size_t textureoffset;
      CC(cudaBindTexture(&textureoffset,
			 &k_wall::texWallParticles, solid4,
			 &k_wall::texWallParticles.channelDesc,
			 sizeof(float4) * solid_size));

      CC(cudaBindTexture(&textureoffset,
			 &k_wall::texWallCellStart, wall_cells->start,
			 &k_wall::texWallCellStart.channelDesc,
			 sizeof(int) * wall_cells->ncells));

      CC(cudaBindTexture(&textureoffset,
			 &k_wall::texWallCellCount, wall_cells->count,
			 &k_wall::texWallCellCount.channelDesc,
			 sizeof(int) * wall_cells->ncells));

      k_wall::
	interactions_3tpp<<<k_cnf(3 * n)>>>
	((float2 *)p, n, solid_size, (float *)acc, trunk->get_float());

      CC(cudaUnbindTexture(k_wall::texWallParticles));
      CC(cudaUnbindTexture(k_wall::texWallCellStart));
      CC(cudaUnbindTexture(k_wall::texWallCellCount));
    }


  }

  void close () {
    CC(cudaUnbindTexture(k_wall::texSDF));
    CC(cudaFreeArray(arrSDF));

    delete wall_cells;
  }

}
