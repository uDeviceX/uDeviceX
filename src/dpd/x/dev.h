enum {X, Y, Z};

/* particle - float2 union */
union Part {
    float2 f2[3];
    struct { float r[3], v[3]; };
};

__device__ Part tex2Part(const Texo<float2> texpp, const int id) {
    Part p;
    p.f2[0] = texpp.fetch(3 * id + 0);
    p.f2[1] = texpp.fetch(3 * id + 1);
    p.f2[2] = texpp.fetch(3 * id + 2);
    return p;
}

__global__ void flocal(const Texo<float2> texpp, const Texo<int> texstart, int n, const float seed, /**/ float *ff) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int pid = gid / 3;
    int zplane = gid % 3;
    
#define start_fetch(id) (id < XS*YS*ZS ? texstart.fetch(id) : n)
    
    if (pid >= n) return;
    
    Part p = tex2Part(texpp, pid);
    /* TODO check from here */
    uint scan1, scan2, ncandidates;
    int start0, start1, start2 = 0;

    int count0, count1 = 0, count2 = 0;
        
    int xcid = (int)(p.r[X] + XS / 2);
    int ycid = (int)(p.r[Y] + YS / 2);
    int zcid = (int)(p.r[Z] + ZS / 2);

    xcid = min(XS-1, max(0, xcid));
    ycid = min(YS-1, max(0, ycid));
    zcid = min(ZS-1, max(0, zcid));
        
    int xcid0 = max(xcid - 1, 0);
    int ycid0 = max(ycid - 1, 0);
    int zcid0 = zcid - 1 + zplane;

    int dx = (xcid == 0 || xcid == (XS-1)) ? 2 : 3;
    int dy = (ycid == 0 || ycid == (YS-1)) ? 2 : 3;

    if (zcid0 < 0 || zcid0 >= ZS) return;
        
    int cid0 = xcid0 + XS * (ycid0 + YS * zcid0);
    start0 = start_fetch(cid0);
    count0 = start_fetch(cid0 + dx) - start0;

    int cid1 = cid0 + XS;
    start1 = start_fetch(cid1);
    count1 = start_fetch(cid1 + dx) - start1;
        
    if (dy > 2) {
        int cid2 = cid0 + XS * 2;
        start2 = start_fetch(cid2);
        count2 = start_fetch(cid2 + dx) - start2;
    }

    scan1 = count0;
    scan2 = count0 + count1;
    ncandidates = scan2 + count2;

    start1 -= scan1;
    start2 -= scan2;
    
    float xforce = 0, yforce = 0, zforce = 0;

    for (int i = 0; i < ncandidates; ++i) {
        int m1 = (int)(i >= scan1);
        int m2 = (int)(i >= scan2);
        int spid = i + (m2 ? start2 : (m1 ? start1 : start0));

        if (spid == pid) continue;

        const Part ps = tex2Part(texpp, spid); /* source particle */

        float rnd = rnd::mean0var1ii(seed, pid, spid);

        float fx, fy, fz;
        forces::dpd0(SOLVENT_TYPE, SOLVENT_TYPE,
                     p.r[X], p.r[Y], p.r[Z],
                     ps.r[X], ps.r[Y], ps.r[Z],
                     p.v[X], p.v[Y], p.v[Z],
                     ps.v[X], ps.v[Y], ps.v[Z], rnd, &fx, &fy, &fz);
        
        xforce += fx; yforce += fy; zforce += fz;
    }
    atomicAdd(ff + 3 * pid + 0, xforce);
    atomicAdd(ff + 3 * pid + 1, yforce);
    atomicAdd(ff + 3 * pid + 2, zforce);
#undef start_fetch
}
