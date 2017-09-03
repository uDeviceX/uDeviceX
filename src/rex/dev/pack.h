namespace dev {
__device__ void
pack0(const float2 *pp, int fid,
      int count, int offset, int tstart, int *indexes,
      int wsf, int dw, /**/ float2 *buf)
{
    int dwe;  /* warp or buffer end relative to `ws' */
    int entry, pid;
    Pa p;
    
    dwe = min(warpSize, count - wsf);
    if (dw < dwe) {
        entry = offset + wsf + dw;
        pid = __ldg(indexes + entry);
        p = pp2p(pp, pid);
        shift(fid, &p); /* shift coordinates */
    }
    p2pp(p, dwe, tstart + offset + wsf, /**/ buf);
}

__device__ void pack1(const float2 *pp, int ws, int dw, /**/ float2 *buf) {
    int wsf;  /* warp start in fragment coordinates */
    int fid;

    fid = k_common::fid(g::starts, ws);
    wsf = ws - g::starts[fid];

    pack0(pp, fid,
          g::counts[fid], g::offsets[fid], g::tstarts[fid], g::indexes[fid],
          wsf, dw, /**/ buf);
}

__global__ void pack(const float2 *pp, /**/ float2 *buf) {
    int warp;
    int lo, hi, step;
    int ws; /* warp start in global coordinates */
    int dw; /* shift relative to `ws' (lane) */

    warp = threadIdx.x / warpSize;
    dw   = threadIdx.x % warpSize;

    lo = warpSize * warp + blockDim.x * blockIdx.x;
    hi = g::starts[26];
    step = gridDim.x * blockDim.x;

    for (ws = lo; ws < hi; ws += step)
        pack1(pp, ws, dw, /**/ buf);
}

}
