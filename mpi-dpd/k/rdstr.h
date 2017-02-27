namespace k_rdstr {
static const int cmaxnc = 64 * 4;
__constant__ float *csrc[cmaxnc], *cdst[cmaxnc];

template <bool from_cmem>
__global__ void pack_all_kernel(int nc, int nv,
				const float **const dsrc,
				float **const ddst) {
  if (nc == 0) return;
  int nfloats_per_rbc = 6 * nv;
  int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= nfloats_per_rbc * nc) return;
  int idrbc = gid / nfloats_per_rbc;
  int offset = gid % nfloats_per_rbc;

  float val;
  if (from_cmem) val = csrc[idrbc][offset];
  else           val = dsrc[idrbc][offset];

  if (from_cmem) cdst[idrbc][offset] = val;
  else           ddst[idrbc][offset] = val;
}

__global__ void shift(const Particle *const psrc, const int np, const int code,
		      const int rank, const bool check, Particle *const pdst) {
  int pid = threadIdx.x + blockDim.x * blockIdx.x;
  int d[3] = {(code + 1) % 3 - 1, (code / 3 + 1) % 3 - 1,
	      (code / 9 + 1) % 3 - 1};
  if (pid >= np) return;
  Particle pnew = psrc[pid];
  int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
  for (int c = 0; c < 3; ++c) pnew.r[c] -= d[c] * L[c];
  pdst[pid] = pnew;
}
}
