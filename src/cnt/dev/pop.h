__global__ void populate(uchar4 *subindices,
                         int *cellstart, int nparticles,
                         int soluteid, 
                         int *entrycells) {
    int pid, cellid, mystart, slot;
    uchar4 subindex;

    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= nparticles) return;
    subindex = subindices[pid];
    if (subindex.x == 255 && subindex.y == 255 && subindex.z == 255) return;
    cellid = subindex.x + XCELLS * (subindex.y + YCELLS * subindex.z);
    mystart = __ldg(cellstart + cellid);
    slot = mystart + subindex.w;
    set(soluteid, pid, slot, /**/ entrycells);
}
