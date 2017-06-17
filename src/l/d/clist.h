namespace l { namespace clist { namespace d {
__device__ int encode(int ix, int iy, int iz, int3 ncells) {
  return ix + ncells.x * (iy + iz * ncells.y);
}

__device__ int3 decode(int code, int3 ncells) {
  int ix = code % ncells.x;
  int iy = (code / ncells.x) % ncells.y;
  int iz = (code / ncells.x/ ncells.y);
  return make_int3(ix, iy, iz);
}

__global__ void pid2code(int *codes, int *pids, const int np, const float *pp, const int3 ncells, const float3 domainstart) {
  int pid = threadIdx.x + blockDim.x * blockIdx.x;
  if (pid >= np) return;

  float x = (pp[0 + 6 * pid] - domainstart.x);
  float y = (pp[1 + 6 * pid] - domainstart.y);
  float z = (pp[2 + 6 * pid] - domainstart.z);

  int ix = (int)floor(x);
  int iy = (int)floor(y);
  int iz = (int)floor(z);

  ix = max(0, min(ncells.x - 1, ix));
  iy = max(0, min(ncells.y - 1, iy));
  iz = max(0, min(ncells.z - 1, iz));

  codes[pid] = encode(ix, iy, iz, ncells);
  pids[pid] = pid;
};

__global__ void gather(const float *input, const int *indices, float *output, const int n) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < n)
    output[tid] = input[(tid % 6) + 6 * indices[tid / 6]];
}

__global__ void cids(int *cids, const int ntotcells, const int offset, const int3 ncells) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < ntotcells)
    {
      int xcid = tid % ncells.x;
      int ycid = (tid / ncells.x) % ncells.y;
      int zcid = (tid / ncells.x / ncells.y) % ncells.z;

      cids[tid] = encode(xcid, ycid, zcid, ncells) + offset;
    }
}

__global__ void count(const int * const start, int * const cnt, const int ncells) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < ncells)
    cnt[tid] -= start[tid];
}
}}} /* namespace l { namespace clist { namespace d */
