namespace k_fsi {
static __device__ unsigned int get_hid(const int a[], const int i) {
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
    k_common::read_AOS6f((float2*)p, n, s0, s1, s2);
     *x = fst(s0);  *y = scn(s0);  *z = fst(s1);
    *vx = scn(s1); *vy = fst(s2); *vz = scn(s2);
}

static __device__ Pa warp2p(const Particle *pp, int n, int i) {
    Pa p;
    warp2rv(pp, n, i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    return p;
}

__device__ void halo0(int n1, float seed, int lid, int lane, int unpackbase, int nunpack,
                      Particle *pp, Force *ff,
                      /**/ float *ff1) {
    Pa l; /* local particle */
    Fo f;
    float2 dst0, dst1, dst2;
    float x, y, z;
    float *dst = NULL;

    Map m;
    int nzplanes;
    int zplane;
    int i;
    int rid; /* remote particle id */
    float myrandnr;

    float3 pos1, pos2, vel1, vel2;
    float3 strength;
    float xinteraction, yinteraction, zinteraction;
    float xforce, yforce, zforce;
    k_common::read_AOS6f((float2 *)(pp + unpackbase), nunpack, dst0, dst1, dst2);
    x = fst(dst0); y = scn(dst0); z = fst(dst1);
    dst = (float *)(ff + unpackbase);

    xforce = yforce = zforce = 0;
    nzplanes = lane < nunpack ? 3 : 0;
    for (zplane = 0; zplane < nzplanes; ++zplane) {
        if (!tex2map(zplane, n1, x, y, z, /**/ &m)) continue;
        for (i = 0; !endp(m, i); ++i) {
            rid = m2id(m, i);
            l = tex2p(rid);
            f = ff2f(ff1, rid);
            myrandnr = l::rnd::d::mean0var1ii(seed, lid, rid);

            pos1 = make_float3(dst0.x, dst0.y, dst1.x);
            pos2 = make_float3(l.x,    l.y,    l.z);
            vel1 = make_float3(dst1.y, dst2.x, dst2.y);
            vel2 = make_float3(l.vx,   l.vy,   l.vz);
            strength = force(SOLID_TYPE, SOLVENT_TYPE, pos1, pos2, vel1, vel2, myrandnr);

            xinteraction = strength.x;
            yinteraction = strength.y;
            zinteraction = strength.z;

            xforce += xinteraction;
            yforce += yinteraction;
            zforce += zinteraction;

            atomicAdd(f.x, -xinteraction);
            atomicAdd(f.y, -yinteraction);
            atomicAdd(f.z, -zinteraction);
        }
    }

    k_common::write_AOS3f(dst, nunpack, xforce, yforce, zforce);
}

__device__ void halo1(int n1, float seed, int lid, int base, int lane, /**/ float *ff1) {
    int fid; /* fragment id */
    int start, count;
    Particle *pp;
    Force *ff;
    int nunpack, unpackbase;
    fid = get_hid(packstarts_padded, base);
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
