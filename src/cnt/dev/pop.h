__global__ void populate(uchar4 *subindices,
                         int *cellstart, int nparticles,
                         int soluteid, int ntotalparticles,
                         int *entrycells) {
    int pid, cellid, mystart, slot;
    uchar4 subindex;

    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= nparticles) return;

    subindex = subindices[pid];

    if (subindex.x == 0xff && subindex.y == 0xff && subindex.z == 0xff) return;

    cellid = subindex.x + XCELLS * (subindex.y + YCELLS * subindex.z);
    mystart = __ldg(cellstart + cellid);
    slot = mystart + subindex.w;
    set(soluteid, pid, slot, /**/ entrycells);
}
