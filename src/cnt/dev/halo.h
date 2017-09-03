__global__ void halo(int n, float seed) {
    enum {X, Y, Z};
    /* n: padded */
    Map m;
    int mapstatus;
    float x, y, z;
    forces::Pa a, b;
    float fx, fy, fz;
    int dw, warp, ws, aid, start;
    int nunpack;
    float2 dst0, dst1, dst2;
    int fid, dwe;
    float xforce, yforce, zforce;
    int nzplanes, zplane;
    int i;
    int slot;
    int objid, spid;
    int sentry;
    float2 stmp0, stmp1, stmp2;
    float rnd;

    dw =   threadIdx.x % warpSize;
    warp = threadIdx.x / warpSize;
    ws = warpSize * (warp + 4 * blockIdx.x);
    aid = ws + dw;
    if (ws >= n) return;

    fid = k_common::fid(g::starts, ws);
    start = g::starts[fid];
    dwe = ws - start;
    nunpack = min(32, g::counts[fid] - dwe);

    if (nunpack == 0) return;

    float2 *pp0 = (float2*)g::pp[fid];
    dst0 = pp0[aid - start];
    dst1 = pp0[aid - start + 1];
    dst2 = pp0[aid - start + 2];

    float *fA;
    fA =g::ff[fid][aid - start].f;

    nzplanes = dw < nunpack ? 3 : 0;
    for (zplane = 0; zplane < nzplanes; ++zplane) {
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
