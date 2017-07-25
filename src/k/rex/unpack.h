namespace k_rex {
__global__ void unpack(/**/ float *forces) {
    int npack_padded = g::cpaddedstarts[26];

    for (int gid = threadIdx.x + blockDim.x * blockIdx.x; gid < 3 * npack_padded;
         gid += blockDim.x * gridDim.x) {
        int pid = gid / 3;

        if (pid >= npack_padded) return;

        int code = k_common::fid(g::cpaddedstarts, pid);
        int lpid = pid - g::cpaddedstarts[code];

        if (lpid >= g::counts[code]) continue;

        int component = gid % 3;

        int entry = g::offsets[code] + lpid;
        float myval = __ldg(g::recvbags[code] + component + 3 * entry);
        int dpid = __ldg(g::scattered_indices[code] + entry);

        atomicAdd(forces + 3 * dpid + component, myval);
    }
}
}
