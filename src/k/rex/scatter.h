namespace k_rex {
__device__ void pp2xyz_col(const float2 *pp, int n, int i, /**/ float *x, float *y, float *z) {
    Pa p;
    p = pp2p_col(pp, n, i);
    p2xyz(p, /**/ x, y, z);
}

__device__ void xyz2fdir(float x, float y, float z, /**/ int fdir[]) {
    enum {X, Y, Z};
    enum { HXSIZE = XS / 2, HYSIZE = YS / 2, HZSIZE = ZS / 2 };
    fdir[X] = -1 + (int)(x >= -HXSIZE + 1) + (int)(x >= HXSIZE - 1);
    fdir[Y] = -1 + (int)(y >= -HYSIZE + 1) + (int)(y >= HYSIZE - 1);
    fdir[Z] = -1 + (int)(z >= -HZSIZE + 1) + (int)(z >= HZSIZE - 1);
}

__device__ void scatter0(const float2 *pp, int pid, float x, float y, float z, /**/ int *counts) {
    int d;
    int xterm, yterm, zterm, fid;
    int myid;
    int fdir[3]; /* [f]ragment [dir]ection */
    xyz2fdir(x, y, z, fdir);
    
    if (fdir[0] == 0 && fdir[1] == 0 && fdir[2] == 0) return;
    // faces
    for (d = 0; d < 3; ++d)
        if (fdir[d]) {
            xterm = (fdir[0] * (d == 0) + 2) % 3;
            yterm = (fdir[1] * (d == 1) + 2) % 3;
            zterm = (fdir[2] * (d == 2) + 2) % 3;

            fid = xterm + 3 * (yterm + 3 * zterm);
            myid = g::offsets[fid] + atomicAdd(counts + fid, 1);

            if (myid < g::capacities[fid]) g::scattered_indices[fid][myid] = pid;
        }
    // edges
    for (d = 0; d < 3; ++d)
        if (fdir[(d + 1) % 3] && fdir[(d + 2) % 3]) {
            xterm = (fdir[0] * (d != 0) + 2) % 3;
            yterm = (fdir[1] * (d != 1) + 2) % 3;
            zterm = (fdir[2] * (d != 2) + 2) % 3;

            fid = xterm + 3 * (yterm + 3 * zterm);
            myid = g::offsets[fid] + atomicAdd(counts + fid, 1);

            if (myid < g::capacities[fid]) g::scattered_indices[fid][myid] = pid;
        }
    // one corner
    if (fdir[0] && fdir[1] && fdir[2]) {
        xterm = (fdir[0] + 2) % 3;
        yterm = (fdir[1] + 2) % 3;
        zterm = (fdir[2] + 2) % 3;

        fid = xterm + 3 * (yterm + 3 * zterm);
        myid = g::offsets[fid] + atomicAdd(counts + fid, 1);

        if (myid < g::capacities[fid]) g::scattered_indices[fid][myid] = pid;
    }
}

__global__ void scatter(const float2 *pp, int n, /**/ int *counts) {
    int warp;
    float x, y, z;
    int ws;  /* warp start in global coordinates */
    int dw;  /* shift relative to `ws' (lane) */
    int dwe; /* wrap or buffer end relative to `ws' */
    int pid; /* particle id */

    warp = threadIdx.x / warpSize;
    dw   = threadIdx.x % warpSize;
    ws   = warpSize * warp + blockDim.x * blockIdx.x;
    dwe  = min(32, n - ws);
    pp2xyz_col(pp, dwe, ws, /**/ &x, &y, &z);
    if (dw < dwe) {
        pid = ws + dw;
        scatter0(pp, pid, x, y, z, /**/ counts);
    }
}
}
