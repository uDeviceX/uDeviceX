__global__ void halo(int n, float seed) {
    enum {X, Y, Z};
    Map m;
    int mapstatus;
    float x, y, z;
    forces::Pa a, b;
    float fx, fy, fz;
    int aid, start;
    float2 dst0, dst1, dst2;
    int fid;
    float xforce, yforce, zforce;
    int zplane;
    int i;
    int slot;
    int objid, spid;
    int sentry;
    float2 stmp0, stmp1, stmp2;
    float rnd;

    aid = threadIdx.x + blockDim.x * blockIdx.x;
    if (aid >= n) return;

    fid = k_common::fid(g::starts, aid);
    start = g::starts[fid];
    float2 *pp0 = (float2*)g::pp[fid];
    dst0 = pp0[aid - start];
    dst1 = pp0[aid - start + 1];
    dst2 = pp0[aid - start + 2];

    float *fA;
    fA =g::ff[fid][aid - start].f;
    for (zplane = 0; zplane < 3; ++zplane) {
        x = dst0.x;
        y = dst0.y;
        z = dst1.x;
        mapstatus = tex2map(zplane, x, y, z, /**/ &m);
        if (mapstatus == EMPTY) continue;
        for (i = 0; !endp(m, i); ++i) {
            slot = m2id(m, i);
            get(slot, &objid, &spid);

            sentry = 3 * spid;
            stmp0 = __ldg(g::csolutes[objid] + sentry);
            stmp1 = __ldg(g::csolutes[objid] + sentry + 1);
            stmp2 = __ldg(g::csolutes[objid] + sentry + 2);

            rnd = rnd::mean0var1ii(seed, aid, spid);
            forces::f2k2p(dst0,   dst1,  dst2, SOLID_TYPE, /**/ &a);
            forces::f2k2p(stmp0, stmp1, stmp2, SOLID_TYPE, /**/ &b);
            forces::gen(a, b, rnd, &fx, &fy, &fz);
            xforce += fx;
            yforce += fy;
            zforce += fz;
            atomicAdd(g::csolutesacc[objid] + sentry,     -fx);
            atomicAdd(g::csolutesacc[objid] + sentry + 1, -fy);
            atomicAdd(g::csolutesacc[objid] + sentry + 2, -fz);
        }
    }

    fA[X] = xforce; fA[Y] = yforce; fA[Z] = zforce;
}
