namespace sdfdev = sdf::sub::dev;
static __device__ void force0(forces::Pa a, int aid, int zplane,
                              float seed, Wa wa, /**/ float *ff) {
    map::Map m;
    forces::Pa b;  /* wall particles */
    float vx, vy, vz; /* wall velocity */
    float fx, fy, fz, rnd;
    float x, y, z;
    float threshold;
    int i, bid;

    forces::p2r3(&a, /**/ &x, &y, &z);
    threshold =
        -1 - 1.7320f * ((float)XSIZE_WALLCELLS / (float)XTE);
    if (sdfdev::cheap_sdf(wa.sdf, x, y, z) <= threshold) return;

    map::ini(zplane, wa.start, wa.n, x, y, z, /**/ &m);
    float xforce = 0, yforce = 0, zforce = 0;
    for (i = 0; !map::endp(m, i); ++i) {
        bid = map::m2id(m, i);
        const float4 r = wa.pp.fetch(bid);
        k_wvel::vell(r.x, r.y, r.z, /**/ &vx, &vy, &vz);
        forces::r3v3k2p(r.x, r.y, r.z, vx, vy, vz, WALL_KIND, /**/ &b);
        rnd = rnd::mean0var1ii(seed, aid, bid);
        forces::gen(a, b, rnd, /**/ &fx, &fy, &fz);
        xforce += fx; yforce += fy; zforce += fz;
    }
    atomicAdd(ff + 3 * aid + 0, xforce);
    atomicAdd(ff + 3 * aid + 1, yforce);
    atomicAdd(ff + 3 * aid + 2, zforce);
}
