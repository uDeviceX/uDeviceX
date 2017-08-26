namespace dev {
template <typename T, int STRIDE>
__global__ void pack(const T *data, int *const iidx[], const int start[], /**/ T *buf[]) {
    int gid, slot;
    int fid; /* [f]ragment [id] */
    int offset, pid, c, d, s;
    
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    slot = gid / STRIDE;
    fid = k_common::fid(start, slot);
    if (slot >= start[27]) return;
    c = gid % STRIDE;

    offset = slot - start[fid];
    pid = __ldg(iidx[fid] + offset);

    d = c + STRIDE * offset;
    s = c + STRIDE * pid;
    
    buf[fid][d] = data[s];
}

} /* namespace */
