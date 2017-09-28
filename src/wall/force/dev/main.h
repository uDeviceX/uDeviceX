static __global__ void force(hforces::Cloud cloud, int np, float seed, Wa wa, /**/ float *ff) {
    forces::Pa a; /* bulk particle */
    int gid, pid, zplane;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    pid = gid / 3;
    zplane = gid % 3;

    if (pid >= np) return;
    fetch(cloud, pid, /**/ &a);

    /* call generic function from polymorphic */
    wall::dev::force0(a, pid, zplane, seed, wa, /**/ ff);
}
