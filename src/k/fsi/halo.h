namespace k_fsi {
static __device__ unsigned int get_hid(const int a[], const int i) {
    /* where is `i' in sorted a[27]? */
    int k1, k3, k9;
    k9 = 9 * ((i >= a[9])           + (i >= a[18]));
    k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
    k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}

__device__ void halo0(int n1, float seed,
                      int pid,
                      int localbase, int laneid, /**/
                      float *ff1) {
    Pa r;
    int nunpack;
    float2 dst0, dst1, dst2;
    float x, y, z;
    float *dst = NULL;
    int fid; /* fragment id */
    int unpackbase;

    Map m;
    int nzplanes;
    int zplane;
    int i, spid;
    int sentry;
    float myrandnr;

    float3 pos1, pos2, vel1, vel2;
    float3 strength;
    float xinteraction, yinteraction, zinteraction;

    float xforce, yforce, zforce;
    fid = get_hid(packstarts_padded, localbase);
    unpackbase = localbase - packstarts_padded[fid];

    nunpack = min(32, packcount[fid] - unpackbase);
    if (nunpack == 0) return;

    k_common::read_AOS6f((float2 *)(packstates[fid] + unpackbase), nunpack, dst0, dst1, dst2);
    x = fst(dst0); y = scn(dst0); z = fst(dst1);

    dst = (float *)(packresults[fid] + unpackbase);

    xforce = yforce = zforce = 0;
    nzplanes = laneid < nunpack ? 3 : 0;
    for (zplane = 0; zplane < nzplanes; ++zplane) {
        if (!tex2map(zplane, n1, x, y, z, /**/ &m)) continue;
        for (i = 0; !endp(m, i); ++i) {
            spid = m2id(m, i);
            r = tex2p(spid);
            sentry = 3 * spid;
            myrandnr = l::rnd::d::mean0var1ii(seed, pid, spid);

            pos1 = make_float3(dst0.x, dst0.y, dst1.x);
            pos2 = make_float3(r.x,    r.y,    r.z);
            vel1 = make_float3(dst1.y, dst2.x, dst2.y);
            vel2 = make_float3(r.vx,   r.vy,   r.vz);
            strength = force(SOLID_TYPE, SOLVENT_TYPE, pos1, pos2, vel1, vel2, myrandnr);

            xinteraction = strength.x;
            yinteraction = strength.y;
            zinteraction = strength.z;

            xforce += xinteraction;
            yforce += yinteraction;
            zforce += zinteraction;

            atomicAdd(ff1 + sentry, -xinteraction);
            atomicAdd(ff1 + sentry + 1, -yinteraction);
            atomicAdd(ff1 + sentry + 2, -zinteraction);
        }
    }

    k_common::write_AOS3f(dst, nunpack, xforce, yforce, zforce);
}

__global__ void halo(int n0, int n1, float seed, float *ff1) {
    int laneid, warpid, localbase, pid;
    laneid = threadIdx.x & 0x1f;
    warpid = threadIdx.x >> 5;
    localbase = 32 * (warpid + 4 * blockIdx.x);
    pid = localbase + laneid;
    if (localbase >= n0) return;
    halo0(n1, seed, pid, localbase, laneid, /**/ ff1);
}

}
