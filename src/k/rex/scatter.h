namespace k_rex {
__device__ void pp2xyz_col(const float2 *pp, int n, int i, /**/ float *x, float *y, float *z) {

}

__global__ void scatter(const float2 *pp, const int n, /**/ int *counts) {
    int warpid, lane, base, nsrc, pid;
    int d;
    int xterm, yterm, zterm, fid;
    int myid;
    Pa p;
    
    warpid = threadIdx.x / warpSize;
    lane   = threadIdx.x % warpSize;
    base   = 32 * (warpid + 4 * blockIdx.x);
    nsrc   = min(32, n - base);
    p = pp2p_coll(pp, nsrc, base);
    //    k_read::AOS6f(pp + 3 * base, nsrc, s0, s1, s2);

    pid = base + lane;
    if (lane >= nsrc) return;
    enum {
        HXSIZE = XS / 2,
        HYSIZE = YS / 2,
        HZSIZE = ZS / 2
    };
    int halocode[3] = {
        -1 + (int)(p.s0.x >= -HXSIZE + 1) + (int)(p.s0.x >= HXSIZE - 1),
        -1 + (int)(p.s0.y >= -HYSIZE + 1) + (int)(p.s0.y >= HYSIZE - 1),
        -1 + (int)(p.s1.x >= -HZSIZE + 1) + (int)(p.s1.x >= HZSIZE - 1)};
    if (halocode[0] == 0 && halocode[1] == 0 && halocode[2] == 0) return;
    // faces
    for (d = 0; d < 3; ++d)
        if (halocode[d]) {
            xterm = (halocode[0] * (d == 0) + 2) % 3;
            yterm = (halocode[1] * (d == 1) + 2) % 3;
            zterm = (halocode[2] * (d == 2) + 2) % 3;

            fid = xterm + 3 * (yterm + 3 * zterm);
            myid = g::offsets[fid] + atomicAdd(counts + fid, 1);

            if (myid < g::capacities[fid]) g::scattered_indices[fid][myid] = pid;
        }
    // edges
    for (d = 0; d < 3; ++d)
        if (halocode[(d + 1) % 3] && halocode[(d + 2) % 3]) {
            xterm = (halocode[0] * (d != 0) + 2) % 3;
            yterm = (halocode[1] * (d != 1) + 2) % 3;
            zterm = (halocode[2] * (d != 2) + 2) % 3;

            fid = xterm + 3 * (yterm + 3 * zterm);
            myid = g::offsets[fid] + atomicAdd(counts + fid, 1);

            if (myid < g::capacities[fid]) g::scattered_indices[fid][myid] = pid;
        }
    // one corner
    if (halocode[0] && halocode[1] && halocode[2]) {
        xterm = (halocode[0] + 2) % 3;
        yterm = (halocode[1] + 2) % 3;
        zterm = (halocode[2] + 2) % 3;

        fid = xterm + 3 * (yterm + 3 * zterm);
        myid = g::offsets[fid] + atomicAdd(counts + fid, 1);

        if (myid < g::capacities[fid]) g::scattered_indices[fid][myid] = pid;
    }
}
}
