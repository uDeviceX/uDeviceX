namespace k_rex {
__device__ void fid2dr(int fid, /**/ float *d) {
    /* fragment id to coordinate shift */
    enum {X, Y, Z};
    d[X] = - ((fid +     2) % 3 - 1) * XS;
    d[Y] = - ((fid / 3 + 2) % 3 - 1) * YS;
    d[Z] = - ((fid / 9 + 2) % 3 - 1) * ZS;
}

__device__ void pack0(const float2 *pp, int ws, int dw, /**/ float2 *buf) {
    enum {X, Y, Z};
    int wsf;  /* wraps start in fragment coordinates */
    int dwe;  /* end of wrap relative to `ws' */
    int fid;
    float2 s0, s1, s2;
    int entry, pid, entry2;
    float d[3]; /* coordinate shift */

    fid = k_common::fid(g::starts, ws);
    wsf = ws - g::starts[fid];
    dwe = min(warpSize, g::counts[fid] - wsf);
    if (dw < dwe) {
        entry = g::offsets[fid] + wsf + dw;
        pid = __ldg(g::scattered_indices[fid] + entry);

        entry2 = 3 * pid;

        s0 = __ldg(pp + entry2);
        s1 = __ldg(pp + entry2 + 1);
        s2 = __ldg(pp + entry2 + 2);

        fid2dr(fid, d);
        s0.x += d[X];
        s0.y += d[Y];
        s1.x += d[Z];
    }
    k_write::AOS6f(buf + 3 * (g::tstarts[fid] + g::offsets[fid] + wsf), dwe, s0, s1, s2);
}

__device__ void pack1(const float2 *pp, /**/ float2 *buf) {
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
        pack0(pp, ws, dw, /**/ buf);
}

__global__ void pack(const float2 *pp, /**/ float2 *buf) {
    if (g::failed) return;
    pack1(pp, buf);
}
}
