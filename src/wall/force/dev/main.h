__global__ void force(Wvel_v wv, Coords c, Cloud cloud, int np, float seed, WallForce wa, /**/ float *ff) {
    forces::Pa a; /* bulk particle */
    int gid, aid, zplane;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    aid    = gid / 3;
    zplane = gid % 3;

    if (aid >= np) return;
    fetch(cloud, aid, /**/ &a);

    /* call generic function from polymorphic */
    ::wf_dev::force0(wv, c, a, aid, zplane, seed, wa, /**/ ff);
}
