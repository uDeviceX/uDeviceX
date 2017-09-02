namespace dev {
static __device__ void warp2rv(const Particle *p, int n, int i, /**/
                               float  *x, float  *y, float  *z,
                               float *vx, float *vy, float *vz) {
    float2 s0, s1, s2;
    p += i;
    k_read::AOS6f((float2*)p, n, s0, s1, s2);
     *x = fst(s0);  *y = scn(s0);  *z = fst(s1);
    *vx = scn(s1); *vy = fst(s2); *vz = scn(s2);
}

static __device__ Pa warp2p(const Particle *pp, int n, int i) {
    /* NOTE: collective read */
    Pa p;
    warp2rv(pp, n, i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    return p;
}

static __device__ void halo0(const float *ppB, int n1, float seed, int lid, int dw, int unpackbase, int nunpack,
                             Particle *pp, Force *ff,
                             /**/ float *ff1) {
    Pa l, r; /* local and remote particles */
    Fo f;
    float *dst = NULL;

    Map m;
    int nzplanes;
    int zplane;
    int i;
    int rid; /* remote particle id */
    float xforce, yforce, zforce;

    l = warp2p(pp, nunpack, unpackbase);
    dst = (float *)(ff + unpackbase);

    xforce = yforce = zforce = 0;
    nzplanes = dw < nunpack ? 3 : 0;
    for (zplane = 0; zplane < nzplanes; ++zplane) {
        if (!tex2map(zplane, n1, l.x, l.y, l.z, /**/ &m)) continue;
        for (i = 0; !endp(m, i); ++i) {
            rid = m2id(m, i);
            pp2p(ppB, rid, /**/ &r);
            f = ff2f(ff1, rid);
            pair(l, r, random(lid, rid, seed), /**/ &xforce, &yforce, &zforce,   f);
        }
    }

    k_write::AOS3f(dst, nunpack, xforce, yforce, zforce);
}

static __device__ void halo1(const float *ppB, int n1, float seed, int lid, int base, int dw, /**/ float *ff1) {
    int fid; /* fragment id */
    int start, count;
    Particle *pp;
    Force *ff;
    int nunpack, unpackbase;
    fid = k_common::fid(g::starts, base);
    start = g::starts[fid];
    count = g::counts[fid];
    pp = g::pp[fid];
    ff = g::ff[fid];
    unpackbase = base - start;
    nunpack = min(32, count - unpackbase);
    if (nunpack == 0) return;

    halo0(ppB, n1, seed, lid, dw, unpackbase, nunpack, pp, ff, /**/ ff1);
}

__global__ void halo(const float *ppB, int n0, int n1, float seed, float *ff1) {
    int dw, warp, base;
    int i; /* particle id */
    warp = threadIdx.x / warpSize;
    dw = threadIdx.x % warpSize;
    base = warpSize * warp + blockDim.x * blockIdx.x;
    if (base >= n0) return;
    i = base + dw;
    halo1(ppB, n1, seed, i, base, dw, /**/ ff1);
}
}
