namespace k_rex {
struct Pa { float2 s0, s1, s2; };

__device__ Pa pp2p(const float2 *pp, int i) {
    Pa p;
    float2 s0, s1, s2;
    pp += 3*i;
    s0 = __ldg(pp++); s1 = __ldg(pp++); s2 = __ldg(pp++);
    p.s0 = s0; p.s1 = s1; p.s2 = s2;
    return p;
}

__device__ void p2pp(Pa p, int n, int i, /**/ float2 *pp) {
    /* collective write : p to buffer pp */
    k_write::AOS6f(pp + 3 * i, n, p.s0, p.s1, p.s2);
}

__device__ void fid2dr(int fid, /**/ float *d) {
    /* fragment id to coordinate shift */
    enum {X, Y, Z};
    d[X] = - ((fid +     2) % 3 - 1) * XS;
    d[Y] = - ((fid / 3 + 2) % 3 - 1) * YS;
    d[Z] = - ((fid / 9 + 2) % 3 - 1) * ZS;
}

__device__ void shift(int fid, Pa *p) {
    enum {X, Y, Z};
    float d[3]; /* coordinate shift */
    fid2dr(fid, d);
    p->s0.x += d[X];
    p->s0.y += d[Y];
    p->s1.x += d[Z];
}

__device__ void
pack0(const float2 *pp, int fid,
      int count, int offset, int tstart, int *scattered_indices,
      int wsf, int dw, /**/ float2 *buf)
{
    int dwe;  /* wrap or buffer end relative to `ws' */
    int entry, pid;
    Pa p;
    
    dwe = min(warpSize, count - wsf);
    if (dw < dwe) {
        entry = offset + wsf + dw;
        pid = __ldg(scattered_indices + entry);
        p = pp2p(pp, pid);
        shift(fid, &p); /* shift coordinates */
    }
    p2pp(p, dwe, tstart + offset + wsf, /**/ buf);
}

__device__ void pack1(const float2 *pp, int ws, int dw, /**/ float2 *buf) {
    int wsf;  /* wraps start in fragment coordinates */
    int fid;

    fid = k_common::fid(g::starts, ws);
    wsf = ws - g::starts[fid];

    pack0(pp, fid,
          g::counts[fid], g::offsets[fid], g::tstarts[fid], g::scattered_indices[fid], /**/
          wsf, dw, /**/ buf);
}

__device__ void pack2(const float2 *pp, /**/ float2 *buf) {
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

__global__ void pack(const float2 *pp, /**/ float2 *buf) {
    if (g::failed) return;
    pack2(pp, buf);
}
}
