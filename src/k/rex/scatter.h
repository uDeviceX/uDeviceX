namespace k_rex {
__global__ void scatter(const float2 *particles,
                        const int nparticles, /**/ int *counts) {
    int warpid = threadIdx.x >> 5;
    int base = 32 * (warpid + 4 * blockIdx.x);
    int nsrc = min(32, nparticles - base);
    float2 s0, s1, s2;
    k_common::read_AOS6f(particles + 3 * base, nsrc, s0, s1, s2);
    int lane = threadIdx.x & 0x1f;
    int pid = base + lane;
    if (lane >= nsrc) return;
    enum {
        HXSIZE = XS / 2,
        HYSIZE = YS / 2,
        HZSIZE = ZS / 2
    };
    int halocode[3] = {
        -1 + (int)(s0.x >= -HXSIZE + 1) + (int)(s0.x >= HXSIZE - 1),
        -1 + (int)(s0.y >= -HYSIZE + 1) + (int)(s0.y >= HYSIZE - 1),
        -1 + (int)(s1.x >= -HZSIZE + 1) + (int)(s1.x >= HZSIZE - 1)};
    if (halocode[0] == 0 && halocode[1] == 0 && halocode[2] == 0) return;
    // faces
#pragma unroll 3
    for (int d = 0; d < 3; ++d)
    if (halocode[d]) {
        int xterm = (halocode[0] * (d == 0) + 2) % 3;
        int yterm = (halocode[1] * (d == 1) + 2) % 3;
        int zterm = (halocode[2] * (d == 2) + 2) % 3;

        int bagid = xterm + 3 * (yterm + 3 * zterm);
        int myid = coffsets[bagid] + atomicAdd(counts + bagid, 1);

        if (myid < ccapacities[bagid]) scattered_indices[bagid][myid] = pid;
    }
    // edges
#pragma unroll 3
    for (int d = 0; d < 3; ++d)
    if (halocode[(d + 1) % 3] && halocode[(d + 2) % 3]) {
        int xterm = (halocode[0] * (d != 0) + 2) % 3;
        int yterm = (halocode[1] * (d != 1) + 2) % 3;
        int zterm = (halocode[2] * (d != 2) + 2) % 3;

        int bagid = xterm + 3 * (yterm + 3 * zterm);
        int myid = coffsets[bagid] + atomicAdd(counts + bagid, 1);

        if (myid < ccapacities[bagid]) scattered_indices[bagid][myid] = pid;
    }
    // one corner
    if (halocode[0] && halocode[1] && halocode[2]) {
        int xterm = (halocode[0] + 2) % 3;
        int yterm = (halocode[1] + 2) % 3;
        int zterm = (halocode[2] + 2) % 3;

        int bagid = xterm + 3 * (yterm + 3 * zterm);
        int myid = coffsets[bagid] + atomicAdd(counts + bagid, 1);

        if (myid < ccapacities[bagid]) scattered_indices[bagid][myid] = pid;
    }
}
}
