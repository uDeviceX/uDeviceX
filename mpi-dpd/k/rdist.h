namespace k_rdist {
static const int cmaxnrbcs = 64 * 4;
__constant__ float *csources[cmaxnrbcs], *cdestinations[cmaxnrbcs];

template <bool from_cmem>
__global__ void pack_all_kernel(const int nrbcs, const int nvertices,
				const float **const dsources,
				float **const ddestinations) {
  if (nrbcs == 0) return;
  const int nfloats_per_rbc = 6 * nvertices;
  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= nfloats_per_rbc * nrbcs) return;
  const int idrbc = gid / nfloats_per_rbc;
  const int offset = gid % nfloats_per_rbc;

  float val;
  if (from_cmem) val = csources[idrbc][offset];
  else           val = dsources[idrbc][offset];

  if (from_cmem) cdestinations[idrbc][offset] = val;
  else           ddestinations[idrbc][offset] = val;
}

__global__ void shift(const Particle *const psrc, const int np, const int code,
		      const int rank, const bool check, Particle *const pdst) {
  int pid = threadIdx.x + blockDim.x * blockIdx.x;
  int d[3] = {(code + 1) % 3 - 1, (code / 3 + 1) % 3 - 1,
	      (code / 9 + 1) % 3 - 1};
  if (pid >= np) return;
  Particle pnew = psrc[pid];
  int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
  for (int c = 0; c < 3; ++c) pnew.x[c] -= d[c] * L[c];
  pdst[pid] = pnew;
}
}
