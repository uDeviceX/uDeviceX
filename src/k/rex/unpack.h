namespace k_rex {
__global__ void unpack(/**/ float *ff) {
    int n;
    n = g::starts[26];
    for (int gid = threadIdx.x + blockDim.x * blockIdx.x; gid < 3 * n;
         gid += blockDim.x * gridDim.x) {
        int pid = gid / 3;

        if (pid >= n) return;

        int code = k_common::fid(g::starts, pid);
        int lpid = pid - g::starts[code];

        if (lpid >= g::counts[code]) continue;

        int component = gid % 3;

        int entry = g::offsets[code] + lpid;
        float myval = __ldg(g::recvbags[code] + component + 3 * entry);
        int dpid = __ldg(g::scattered_indices[code] + entry);

        atomicAdd(ff + 3 * dpid + component, myval);
    }
}
}
