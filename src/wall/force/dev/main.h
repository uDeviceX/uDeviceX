static __global__ void force(hforces::Cloud cloud, int np, float seed, Wa wa, /**/ float *ff) {
    forces::Pa a; /* bulk particle */
    int gid, aid, zplane;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    aid    = gid / 3;
    zplane = gid % 3;

    if (aid >= np) return;
    fetch(cloud, aid, /**/ &a);

    /* call generic function from polymorphic */
    wall::dev::force0(a, aid, zplane, seed, wa, /**/ ff);
}
