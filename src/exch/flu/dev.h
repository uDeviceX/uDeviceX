namespace dev {

__global__ void build_map(int3 L, int solventid, int n, const Particle *pp, /**/ Map map) {
    int pid, fid, fids[MAX_DSTS], ndsts, j;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= n) return;
    const Particle p = pp[pid];

    fid = map_code(L, p.r);
    ndsts = map_decode(fid, /**/ fids);

    for (j = 0; j < ndsts; ++j)
        if (fid < 13) /* select only half the fragments */
            add_to_map(solventid, pid, fids[j], /**/ map);
}

} // dev
