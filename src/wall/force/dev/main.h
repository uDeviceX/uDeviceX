static __global__ void force(sdf::Tex_t texsdf, hforces::Cloud cloud, int np, int w_n,
                             float seed, const Texo<int> texstart, const Texo<float4> texwpp, /**/
                             float *ff) {
    forces::Pa a; /* bulk particle */
    int gid, pid, zplane;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    pid = gid / 3;
    zplane = gid % 3;

    if (pid >= np) return;
    fetch(cloud, pid, /**/ &a);

    /* call from polymorphic */
    wall::dev::force0(a, pid, zplane,
                     texsdf, w_n, seed,
                     texstart, texwpp, ff);
}
