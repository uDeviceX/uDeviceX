namespace k_rex {
__global__ void unpack(/**/ float *ff) {
    int n;
    int gid, pid, fid, lpid, component, entry, dpid;
    float myval;
    n = g::starts[26];
    for (gid = threadIdx.x + blockDim.x * blockIdx.x; gid < 3 * n; gid += blockDim.x * gridDim.x) {
        pid = gid / 3;
        if (pid >= n) return;
        fid = k_common::fid(g::starts, pid);
        lpid = pid - g::starts[fid];
        if (lpid >= g::counts[fid]) continue;
        component = gid % 3;
        entry = g::offsets[fid] + lpid;
        myval = __ldg(g::recvbags[fid] + component + 3 * entry);
        dpid = __ldg(g::scattered_indices[fid] + entry);
        atomicAdd(ff + 3 * dpid + component, myval);
    }
}
}
