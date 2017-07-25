namespace k_rex {
__global__ void unpack(int nparticles, /**/ float *forces) {
    int npack_padded = cpaddedstarts[26];

    for (int gid = threadIdx.x + blockDim.x * blockIdx.x; gid < 3 * npack_padded;
         gid += blockDim.x * gridDim.x) {
        int pid = gid / 3;

        if (pid >= npack_padded) return;

        int key9 =
            9 * ((int)(pid >= cpaddedstarts[9]) + (int)(pid >= cpaddedstarts[18]));

        int key3 = 3 * ((int)(pid >= cpaddedstarts[key9 + 3]) +
                        (int)(pid >= cpaddedstarts[key9 + 6]));

        int key1 = (int)(pid >= cpaddedstarts[key9 + key3 + 1]) +
            (int)(pid >= cpaddedstarts[key9 + key3 + 2]);

        int code = key9 + key3 + key1;
        int lpid = pid - cpaddedstarts[code];

        if (lpid >= ccounts[code]) continue;

        int component = gid % 3;

        int entry = coffsets[code] + lpid;
        float myval = __ldg(recvbags[code] + component + 3 * entry);
        int dpid = __ldg(scattered_indices[code] + entry);

        atomicAdd(forces + 3 * dpid + component, myval);
    }
}
}
