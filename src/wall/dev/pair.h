namespace sdfdev = sdf::sub::dev;
typedef const sdf::tex3Dca<float> TexSDF_t;
__global__ void pair(TexSDF_t texsdf, hforces::Cloud cloud, const int np, const int w_n,
                     float *const ff, const float seed,
                     const Texo<int> texstart, const Texo<float4> texwpp) {
#define   wpp_fetch(i) (texwpp.fetch(i))
    map::Map m;
    forces::Pa a, b;  /* bulk and wall particles */
    float vx, vy, vz; /* wall velocity */
    float fx, fy, fz, rnd;
    float x, y, z;
    int gid, pid, zplane;
    float threshold;
    int i, spid;

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    pid = gid / 3;
    zplane = gid % 3;

    if (pid >= np) return;
    fetch(cloud, pid, /**/ &a);
    forces::p2r3(&a, /**/ &x, &y, &z);
    threshold =
        -1 - 1.7320f * ((float)XSIZE_WALLCELLS / (float)XTE);
    if (sdfdev::cheap_sdf(texsdf, x, y, z) <= threshold) return;

    map::ini(zplane, texstart, w_n, x, y, z, /**/ &m);
    float xforce = 0, yforce = 0, zforce = 0;
    for (i = 0; !map::endp(m, i); ++i) {
        spid = map::m2id(m, i);
        const float4 r = wpp_fetch(spid);
        k_wvel::vell(r.x, r.y, r.z, /**/ &vx, &vy, &vz);
        forces::r3v3k2p(r.x, r.y, r.z, vx, vy, vz, WALL_KIND, /**/ &b);
        rnd = rnd::mean0var1ii(seed, pid, spid);
        forces::gen(a, b, rnd, /**/ &fx, &fy, &fz);
        xforce += fx; yforce += fy; zforce += fz;
    }
    atomicAdd(ff + 3 * pid + 0, xforce);
    atomicAdd(ff + 3 * pid + 1, yforce);
    atomicAdd(ff + 3 * pid + 2, zforce);
#undef wpp_fetch
}
