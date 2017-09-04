__device__ void
pack0(const float2 *pp, int fid,
      int count, int offset, int tstart, int *indexes,
      int i, /**/ float2 *buf)
{
    int entry, pid;
    Pa p;
    
    entry = offset + i;
    pid = __ldg(indexes + entry);
    p = pp2p(pp, pid);
    shift(fid, &p); /* shift coordinates */
    p2pp(p, tstart + entry, /**/ buf);
}

__device__ void pack1(const float2 *pp, int pid, /**/ float2 *buf) {
    int i;
    int fid;

    fid = k_common::fid(g::starts, pid);
    i = pid - g::starts[fid];

    pack0(pp, fid,
          g::counts[fid], g::offsets[fid], g::tstarts[fid], g::indexes[fid], i, /**/ buf);
}

__global__ void pack(const float2 *pp, /**/ float2 *buf) {
    int warp, pid;
    int lo, hi, step;
    int ws; /* warp start in global coordinates */
    int dw; /* shift relative to `ws' (lane) */

    warp = threadIdx.x / warpSize;
    dw   = threadIdx.x % warpSize;
    lo = warpSize * warp + blockDim.x * blockIdx.x;
    hi = g::starts[26];
    step = gridDim.x * blockDim.x;

    for (ws = lo; ws < hi; ws += step) {
        pid = ws + dw;
        pack1(pp, pid, /**/ buf);
    }
}
