__global__ void populate(uchar4 *subindices,
                         int *cellstart, int nparticles,
                         int soluteid, int ntotalparticles,
                         int *entrycells) {
    int warpid = threadIdx.x / warpSize;
    int tid = threadIdx.x % warpSize;

    int base = 32 * (warpid + 4 * blockIdx.x);
    int pid = base + tid;

    if (pid >= nparticles) return;

    uchar4 subindex = subindices[pid];

    if (subindex.x == 0xff && subindex.y == 0xff && subindex.z == 0xff) return;

    int cellid = subindex.x + XCELLS * (subindex.y + YCELLS * subindex.z);
    int mystart = __ldg(cellstart + cellid);
    int slot = mystart + subindex.w;

    set(soluteid, pid, slot, /**/ entrycells);
}
