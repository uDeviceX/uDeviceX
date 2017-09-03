namespace dev {
static __device__ void pp2xyz_col(const float2 *pp, int n, int i, /**/ float *x, float *y, float *z) {
    /* [col]collective (wrap) */
    Pa p;
    p = pp2p_col(pp, n, i);
    p2xyz(p, /**/ x, y, z);
}

static __device__ void xyz2fdir(float x, float y, float z, /**/ int fdir[]) {
    enum {X, Y, Z};
    enum { HXSIZE = XS / 2, HYSIZE = YS / 2, HZSIZE = ZS / 2 };
    fdir[X] = -1 + (int)(x >= -HXSIZE + 1) + (int)(x >= HXSIZE - 1);
    fdir[Y] = -1 + (int)(y >= -HYSIZE + 1) + (int)(y >= HYSIZE - 1);
    fdir[Z] = -1 + (int)(z >= -HZSIZE + 1) + (int)(z >= HZSIZE - 1);
}

static __device__ void reg_p(int pid, int dx, int dy, int dz, int *offsets, /**/ int *counts) {
    /* register the particle */
    int fid;
    int i; /* particle in fragment coordinates */
    fid = dx + 3 * (dy + 3 * dz);
    i = offsets[fid] + atomicAdd(counts + fid, 1);
    g::indexes[fid][i] = pid;
}

static __device__ void scatter0(int pid, float x, float y, float z, int *offsets, /**/ int *counts) {
    enum {X, Y, Z};
    int d;
    int dx, dy, dz;
    int fdir[3]; /* [f]ragment [dir]ection */
    xyz2fdir(x, y, z, fdir);
    
    if (fdir[X] == 0 && fdir[Y] == 0 && fdir[Z] == 0) return;
    // faces
    for (d = 0; d < 3; ++d)
        if (fdir[d]) {
            dx = (fdir[X] * (d == X) + 2) % 3;
            dy = (fdir[Y] * (d == Y) + 2) % 3;
            dz = (fdir[Z] * (d == Z) + 2) % 3;
            reg_p(pid, dx, dy, dz, offsets, /**/ counts);
        }
    // edges
    for (d = 0; d < 3; ++d)
        if (fdir[(d + 1) % 3] && fdir[(d + 2) % 3]) {
            dx = (fdir[X] * (d != X) + 2) % 3;
            dy = (fdir[Y] * (d != Y) + 2) % 3;
            dz = (fdir[Z] * (d != Z) + 2) % 3;
            reg_p(pid, dx, dy, dz, offsets, /**/ counts);
        }
    // corner
    if (fdir[X] && fdir[Y] && fdir[Z]) {
        dx = (fdir[X] + 2) % 3;
        dy = (fdir[Y] + 2) % 3;
        dz = (fdir[Z] + 2) % 3;
        reg_p(pid, dx, dy, dz, offsets, /**/ counts);
    }
}

__global__ void scatter(const float2 *pp, int *offsets, int n, /**/ int *counts) {
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
        scatter0(pid, x, y, z, offsets, /**/ counts);
    }
}
}
