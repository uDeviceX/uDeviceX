namespace k_rex {
__global__ void unpack(/**/ float *ff) {
    int npack_padded, gid, pid, code, lpid, component, entry, myval, dpid;

    npack_padded = g::starts[26];
    for (gid = threadIdx.x + blockDim.x * blockIdx.x; gid < 3 * npack_padded;
         gid += blockDim.x * gridDim.x) {
        pid = gid / 3;

        if (pid >= npack_padded) return;

        code = k_common::fid(g::starts, pid);
        lpid = pid - g::starts[code];

        if (lpid >= g::counts[code]) continue;

        component = gid % 3;

        entry = g::offsets[code] + lpid;
        myval = __ldg(g::recvbags[code] + component + 3 * entry);
        dpid = __ldg(g::scattered_indices[code] + entry);

        atomicAdd(ff + 3 * dpid + component, myval);
    }
}
}
