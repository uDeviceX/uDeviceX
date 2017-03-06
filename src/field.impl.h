namespace field {
  void ini(const char *path, int N[3], float extent[3]) { /* read sdf file */
    size_t CHUNKSIZE = 1 << 25;
    if (m::rank == 0) {
      FILE *fh = fopen(path, "r");
      char line[2048];
      fgets(line, sizeof(line), fh);
      sscanf(line, "%f %f %f", &extent[0], &extent[1], &extent[2]);
      fgets(line, sizeof(line), fh);
      sscanf(line, "%d %d %d", &N[0], &N[1], &N[2]);

      MC(MPI_Bcast(N, 3, MPI_INT, 0, m::cart));
      MC(MPI_Bcast(extent, 3, MPI_FLOAT, 0, m::cart));

      int np = N[0] * N[1] * N[2];
      data = new float[np];
      fread(data, sizeof(float), np, fh);
      fclose(fh);
      for (size_t i = 0; i < np; i += CHUNKSIZE) {
	size_t s = (i + CHUNKSIZE <= np) ? CHUNKSIZE : (np - i);
	MC(MPI_Bcast(data + i, s, MPI_FLOAT, 0, m::cart));
      }
    } else {
      MC(MPI_Bcast(N, 3, MPI_INT, 0, m::cart));
      MC(MPI_Bcast(extent, 3, MPI_FLOAT, 0, m::cart));
      int np = N[0] * N[1] * N[2];
      data = new float[np];
      for (size_t i = 0; i < np; i += CHUNKSIZE) {
	size_t s = (i + CHUNKSIZE <= np) ? CHUNKSIZE : (np - i);
	MC(MPI_Bcast(data + i, s, MPI_FLOAT, 0, m::cart));
      }
    }
  }

  void sample(float start[3], float spacing[3], int nsize[3], float amplitude_rescaling, int N[3],
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
  
  void fin() { delete[] data; }

} /* namespace field */
