static __device__ void fetch_wall(Wvel_v wv, Coords c, Texo<float4> pp, int i, /**/ forces::Pa *a) {
    float3 r, v; /* wall velocity */
    float4 r0;
    r0 = fetch(pp, i);
    r = make_float3(r0.x, r0.y, r0.z);
    wvel(wv, c, r, /**/ &v);
    forces::r3v3k2p(r.x, r.y, r.z, v.x, v.y, v.z, WALL_KIND, /**/ a);
}

static __device__ void force0(Wvel_v wv, Coords c, forces::Pa a, int aid, int zplane,
                              float seed, WallForce wa, /**/ float *ff) {
    map::Map m;
    forces::Pa b;  /* wall particles */
    float rnd;
    forces::Fo f;
    float x, y, z;
    float threshold;
    int i, bid;

    forces::p2r3(&a, /**/ &x, &y, &z);
    threshold =
        -1 - 1.7320f * ((float)XSIZE_WALLCELLS / (float)XTE);
    if (cheap_sdf(&wa.sdf_v, x, y, z) <= threshold) return;

    map::ini(zplane, wa.start, wa.n, x, y, z, /**/ &m);
    float xforce = 0, yforce = 0, zforce = 0;
    for (i = 0; !map::endp(m, i); ++i) {
        bid = map::m2id(m, i);
        fetch_wall(wv, c, wa.pp, bid, /**/ &b);
        rnd = rnd::mean0var1ii(seed, aid, bid);
        forces::gen(a, b, rnd, /**/ &f);
        xforce += f.x; yforce += f.y; zforce += f.z;
    }
    atomicAdd(ff + 3 * aid + 0, xforce);
    atomicAdd(ff + 3 * aid + 1, yforce);
    atomicAdd(ff + 3 * aid + 2, zforce);
}
