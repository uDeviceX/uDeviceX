template <int k> struct Bspline {
  template <int i> static float eval(float x) {
    return (x - i) / (k - 1) * Bspline<k - 1>::template eval<i>(x) +
      (i + k - x) / (k - 1) * Bspline<k - 1>::template eval<i + 1>(x);
  }
};

template <> struct Bspline<1> {
  template <int i> static float eval(float x) {
    return (float)(i) <= x && x < (float)(i + 1);
  }
};

struct FieldSampler {
  float *data,  extent[3];
  int N[3];
  FieldSampler(const char *path, MPI_Comm comm) { /* read sdf file */
    size_t CHUNKSIZE = 1 << 25; int rank;
    MC(MPI_Comm_rank(comm, &rank));
    if (rank == 0) {
      FILE *fh = fopen(path, "r");
      char line[2048];
      fgets(line, sizeof(line), fh);
      sscanf(line, "%f %f %f", &extent[0], &extent[1], &extent[2]);
      fgets(line, sizeof(line), fh);
      sscanf(line, "%d %d %d", &N[0], &N[1], &N[2]);

      MC(MPI_Bcast(N, 3, MPI_INT, 0, comm));
      MC(MPI_Bcast(extent, 3, MPI_FLOAT, 0, comm));

      int nvoxels = N[0] * N[1] * N[2];
      data = new float[nvoxels];
      fread(data, sizeof(float), nvoxels, fh);
      fclose(fh);
      for (size_t i = 0; i < nvoxels; i += CHUNKSIZE) {
	size_t s = (i + CHUNKSIZE <= nvoxels) ? CHUNKSIZE : (nvoxels - i);
	MC(MPI_Bcast(data + i, s, MPI_FLOAT, 0, comm));
      }
    } else {
      MC(MPI_Bcast(N, 3, MPI_INT, 0, comm));
      MC(MPI_Bcast(extent, 3, MPI_FLOAT, 0, comm));
      int nvoxels = N[0] * N[1] * N[2];
      data = new float[nvoxels];
      for (size_t i = 0; i < nvoxels; i += CHUNKSIZE) {
	size_t s = (i + CHUNKSIZE <= nvoxels) ? CHUNKSIZE : (nvoxels - i);
	MC(MPI_Bcast(data + i, s, MPI_FLOAT, 0, comm));
      }
    }
  }

  void sample(float start[3], float spacing[3], int nsize[3], float amplitude_rescaling,
	      float *const output) {
#define X 0
#define Y 1
#define Z 2
#define OOO(ix, iy, iz) (output[ix + nsize[X] * (iy + nsize[Y] * iz)])
#define DDD(ix, iy, iz) (data  [ix +     N[X] * (iy +     N[Y] * iz)])
#define i2r(i, d) (start[d] + (i + 0.5f) * spacing[d] - 0.5f)
#define i2x(i)    i2r(i,X)
#define i2y(i)    i2r(i,Y)
#define i2z(i)    i2r(i,Z)
    Bspline<4> bsp;
    for (int iz = 0; iz < nsize[Z]; ++iz)
      for (int iy = 0; iy < nsize[Y]; ++iy)
	for (int ix = 0; ix < nsize[X]; ++ix) {
	  float x[3] = {i2x(ix), i2y(iy), i2z(iz)};

	  int anchor[3];
	  for (int c = 0; c < 3; ++c) anchor[c] = (int)floor(x[c]);

	  float w[3][4];
	  for (int c = 0; c < 3; ++c)
	    for (int i = 0; i < 4; ++i)
	      w[c][i] = bsp.eval<0>(x[c] - (anchor[c] - 1 + i) + 2);

	  float tmp[4][4];
	  for (int sz = 0; sz < 4; ++sz)
	    for (int sy = 0; sy < 4; ++sy) {
	      float s = 0;
	      for (int sx = 0; sx < 4; ++sx) {
		int l[3] = {sx, sy, sz};
		int g[3];
		for (int c = 0; c < 3; ++c)
		  g[c] = (l[c] - 1 + anchor[c] + N[c]) % N[c];

		s += w[0][sx] * DDD(g[X], g[Y], g[Z]);
	      }
	      tmp[sz][sy] = s;
	    }
	  float partial[4];
	  for (int sz = 0; sz < 4; ++sz) {
	    float s = 0;
	    for (int sy = 0; sy < 4; ++sy) s += w[1][sy] * tmp[sz][sy];
	    partial[sz] = s;
	  }
	  float val = 0;
	  for (int sz = 0; sz < 4; ++sz) val += w[2][sz] * partial[sz];
	  OOO(ix, iy, iz) = val * amplitude_rescaling;
	}
#undef DDD
#undef OOO
#undef X
#undef Y
#undef Z
  }
  ~FieldSampler() { delete[] data; }
};

namespace wall {
  void init(Particle *const p, const int n,
		 int &nsurvived) {
    wall_cells = new CellLists(XSIZE_SUBDOMAIN + 2 * XMARGIN_WALL,
			       YSIZE_SUBDOMAIN + 2 * YMARGIN_WALL,
			       ZSIZE_SUBDOMAIN + 2 * ZMARGIN_WALL);
    int myrank, dims[3], periods[3];
    MC(MPI_Comm_rank(Cont::cartcomm, &myrank));
    MC(MPI_Cart_get(Cont::cartcomm, 3, dims, periods, Cont::coords));
    float *field = new float[XTEXTURESIZE * YTEXTURESIZE * ZTEXTURESIZE];
    FieldSampler sampler("sdf.dat", Cont::cartcomm);
    int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
    int MARGIN[3] = {XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL};
    int TEXTURESIZE[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};
    if (myrank == 0) printf("sampling the geometry file...\n");
    {
      float start[3], spacing[3];
      for (int c = 0; c < 3; ++c) {
	start[c] = sampler.N[c] * (Cont::coords[c] * L[c] - MARGIN[c]) /
	  (float)(dims[c] * L[c]);
	spacing[c] = sampler.N[c] * (L[c] + 2 * MARGIN[c]) /
	  (float)(dims[c] * L[c]) / (float)TEXTURESIZE[c];
      }
      float amplitude_rescaling = (XSIZE_SUBDOMAIN /*+ 2 * XMARGIN_WALL*/) /
	(sampler.extent[0] / dims[0]);
      sampler.sample(start, spacing, TEXTURESIZE, amplitude_rescaling, field);
    }

    if (myrank == 0) printf("estimating geometry-based message sizes...\n");
    {
      for (int dz = -1; dz <= 1; ++dz)
	for (int dy = -1; dy <= 1; ++dy)
	  for (int dx = -1; dx <= 1; ++dx) {
	    int d[3] = {dx, dy, dz};
	    int local_start[3] = {d[0] + (d[0] == 1) * (XSIZE_SUBDOMAIN - 2),
				  d[1] + (d[1] == 1) * (YSIZE_SUBDOMAIN - 2),
				  d[2] + (d[2] == 1) * (ZSIZE_SUBDOMAIN - 2)};
	    int local_extent[3] = {1 * (d[0] != 0 ? 2 : XSIZE_SUBDOMAIN),
				   1 * (d[1] != 0 ? 2 : YSIZE_SUBDOMAIN),
				   1 * (d[2] != 0 ? 2 : ZSIZE_SUBDOMAIN)};

	    float start[3], spacing[3];
	    for (int c = 0; c < 3; ++c) {
	      start[c] = (Cont::coords[c] * L[c] + local_start[c]) /
		(float)(dims[c] * L[c]) * sampler.N[c];
	      spacing[c] = sampler.N[c] / (float)(dims[c] * L[c]);
	    }
	    int nextent = local_extent[0] * local_extent[1] * local_extent[2];
	    float *data = new float[nextent];
	    sampler.sample(start, spacing, local_extent, 1, data);
	    int s = 0;
	    for (int i = 0; i < nextent; ++i) s += (data[i] < 0);

	    delete[] data;
	    double avgsize =
	      ceil(s * numberdensity /
		   (double)pow(2, abs(d[0]) + abs(d[1]) + abs(d[2])));
	  }
    }

    if (hdf5field_dumps) {
      if (myrank == 0) printf("H5 data dump of the geometry...\n");

      float *walldata =
	new float[XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN];

      float start[3], spacing[3];
      for (int c = 0; c < 3; ++c) {
	start[c] = Cont::coords[c] * L[c] / (float)(dims[c] * L[c]) * sampler.N[c];
	spacing[c] = sampler.N[c] / (float)(dims[c] * L[c]);
      }

      int size[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
      float amplitude_rescaling = L[0] / (sampler.extent[0] / dims[0]);
      sampler.sample(start, spacing, size, amplitude_rescaling, walldata);
      H5FieldDump dump(Cont::cartcomm);
      dump.dump_scalarfield(Cont::cartcomm, walldata, "wall");
      delete[] walldata;
    }


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

    k_wall::setup();

    CC(cudaBindTextureToArray(k_wall::texSDF, arrSDF, fmt));

    if (myrank == 0) printf("carving out wall particles...\n");

    thrust::device_vector<int> keys(n);

    k_wall::fill_keys<<<(n + 127) / 128, 128>>>
      (p, n, thrust::raw_pointer_cast(&keys[0]));



    thrust::sort_by_key(keys.begin(), keys.end(),
			thrust::device_ptr<Particle>(p));

    nsurvived = thrust::count(keys.begin(), keys.end(), 0);

    int nbelt = thrust::count(keys.begin() + nsurvived, keys.end(), 1);

    thrust::device_vector<Particle> solid_local
      (thrust::device_ptr<Particle>(p + nsurvived),
       thrust::device_ptr<Particle>(p + nsurvived + nbelt));

    /*
      can't use halo-exchanger class because of MARGIN HaloExchanger
      halo(cartcomm, L, 666); DeviceBuffer<Particle> solid_remote;
      halo.exchange(thrust::raw_pointer_cast(&solid_local[0]),
      solid_local.size(), solid_remote);
    */
    if (myrank == 0) printf("fetching remote wall particles...\n");

    DeviceBuffer<Particle> solid_remote;

    {
      thrust::host_vector<Particle> local = solid_local;

      int dstranks[26], remsizes[26], recv_tags[26];
      for (int i = 0; i < 26; ++i) {
	int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

	recv_tags[i] =
	  (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	int coordsneighbor[3];
	for (int c = 0; c < 3; ++c) coordsneighbor[c] = Cont::coords[c] + d[c];

	MC(MPI_Cart_rank(Cont::cartcomm, coordsneighbor, dstranks + i));
      }

      // send local counts - receive remote counts
      {
	for (int i = 0; i < 26; ++i) remsizes[i] = -1;

	MPI_Request reqrecv[26];
	for (int i = 0; i < 26; ++i)
	  MC(MPI_Irecv(remsizes + i, 1, MPI_INTEGER, dstranks[i],
		       123 + recv_tags[i], Cont::cartcomm, reqrecv + i));

	int localsize = local.size();

	MPI_Request reqsend[26];
	for (int i = 0; i < 26; ++i)
	  MC(MPI_Isend(&localsize, 1, MPI_INTEGER, dstranks[i], 123 + i,
		       Cont::cartcomm, reqsend + i));

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
		       dstranks[i], 321 + recv_tags[i], Cont::cartcomm,
		       reqrecv + i));

	MPI_Request reqsend[26];
	for (int i = 0; i < 26; ++i)
	  MC(MPI_Isend(local.data(), local.size() * 6, MPI_FLOAT,
		       dstranks[i], 321 + i, Cont::cartcomm, reqsend + i));

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

    if (solid_size > 0) wall_cells->build(solid, solid_size, 0);

    CC(cudaMalloc(&solid4, sizeof(float4) * solid_size));

    if (myrank == 0) printf("consolidating wall particles...\n");

    if (solid_size > 0)
      k_wall::strip_solid4<<<(solid_size + 127) / 128, 128>>>
	(solid, solid_size, solid4);

    CC(cudaFree(solid));


  }

  void bounce(Particle *const p, const int n) {
    if (n > 0)
      k_wall::bounce<<<(n + 127) / 128, 128, 0>>>
	((float2 *)p, n, dt);


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
	interactions_3tpp<<<(3 * n + 127) / 128, 128, 0>>>
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
