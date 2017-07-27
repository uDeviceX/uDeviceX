namespace k_rex {
__device__ void unpack0(int pid, int dim, /**/ float *ff) {
    int fid, lpid, entry, dpid;
    float myval;
    fid = k_common::fid(g::starts, pid);
    lpid = pid - g::starts[fid];
    if (lpid >= g::counts[fid]) return;
    entry = g::offsets[fid] + lpid;
    myval = __ldg(g::recvbags[fid] + dim + 3 * entry);
    dpid = __ldg(g::scattered_indices[fid] + entry);
    atomicAdd(ff + 3 * dpid + dim, myval);
}

__global__ void unpack(/**/ float *ff) {
    int n;
    int gid; /* global id */
    int pid; /* particle id */
    int dim; /* dimenshion of the force (x, y, z) */
    int lo, hi, step;
    n = g::starts[26];
    lo = threadIdx.x + blockDim.x * blockIdx.x;
    step = blockDim.x * gridDim.x;
    hi = 3*n;
    for (gid = lo; gid < hi; gid += step) {
        pid = gid / 3;
        if (pid >= n) return;
        dim = gid % 3;
        unpack0(pid, dim, /**/ ff);
    }
}
}
