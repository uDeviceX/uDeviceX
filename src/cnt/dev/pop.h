__global__ void populate(uchar4 *subindices, int *starts, int n, int objid, /**/ int *entry) {
    int pid, id, start, slot;
    uchar4 subindex;

    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    subindex = subindices[pid];
    if (subindex.x == 255 && subindex.y == 255 && subindex.z == 255) return;
    id = subindex.x + XCELLS * (subindex.y + YCELLS * subindex.z);

    start = __ldg(starts + id);
    slot = start + subindex.w;
    set(objid, pid, slot, /**/ entry);
}
