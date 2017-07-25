namespace k_fsi {
static __device__ unsigned int get_fid(const int a[], const int i) {
    /* where is `i' in sorted a[27]? */
    int k1, k3, k9;
    k9 = 9 * ((i >= a[9])           + (i >= a[18]));
    k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
    k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}

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

__device__ void halo0(int n1, float seed, int lid, int lane, int unpackbase, int nunpack,
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
    nzplanes = lane < nunpack ? 3 : 0;
    for (zplane = 0; zplane < nzplanes; ++zplane) {
        if (!tex2map(zplane, n1, l.x, l.y, l.z, /**/ &m)) continue;
        for (i = 0; !endp(m, i); ++i) {
            rid = m2id(m, i);
            r = tex2p(rid);
            f = ff2f(ff1, rid);
            pair(l, r, random(lid, rid, seed), /**/ &xforce, &yforce, &zforce,   f);
        }
    }

    k_write::AOS3f(dst, nunpack, xforce, yforce, zforce);
}

__device__ void halo1(int n1, float seed, int lid, int base, int lane, /**/ float *ff1) {
    int fid; /* fragment id */
    int start, count;
    Particle *pp;
    Force *ff;
    int nunpack, unpackbase;
    fid = get_fid(packstarts_padded, base);
    start = packstarts_padded[fid];
    count = packcount[fid];
    pp = packstates[fid];
    ff = packresults[fid];
    unpackbase = base - start;
    nunpack = min(32, count - unpackbase);
    if (nunpack == 0) return;

    halo0(n1, seed, lid, lane, unpackbase, nunpack, pp, ff, /**/ ff1);
}

__global__ void halo(int n0, int n1, float seed, float *ff1) {
    int lane, warp, base;
    int i; /* particle id */
    warp = threadIdx.x / 32;
    lane = threadIdx.x % 32;
    base = 32 * warp + blockDim.x * blockIdx.x;
    if (base >= n0) return;
    i = base + lane;
    halo1(n1, seed, i, base, lane, /**/ ff1);
}

}
