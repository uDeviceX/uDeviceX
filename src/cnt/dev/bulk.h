__global__ void bulk(float2 *pp, int n,
                     int ncellentries, int nsolutes,
                     float seed, int mysoluteid, float *ff) {
    Map m; /* see map/ */
    float x, y, z;

    float fx, fy, fz, rnd;
    forces::Pa a, b;
    int gid, pid, zplane;
    float2 dst0, dst1, dst2;
    float xforce, yforce, zforce;
    int i, slot;

    int soluteid;
    int spid;
    int sentry;
    float2 stmp0, stmp1, stmp2;
    int mapstatus;

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    pid = gid / 3;
    zplane = gid % 3;

    if (pid >= n) return;

    dst0 = __ldg(pp + 3 * pid + 0);
    dst1 = __ldg(pp + 3 * pid + 1);
    dst2 = __ldg(pp + 3 * pid + 2);
    x = dst0.x;
    y = dst0.y;
    z = dst1.x;
    mapstatus = r2map(zplane, n, x, y, z, /**/ &m);
    
    if (mapstatus == EMPTY) return;
    xforce = yforce = zforce = 0;
    for (i = 0; !endp(m, i); ++i) {
        slot = m2id(m, i);
        get(slot, &soluteid, &spid);
        if (mysoluteid < soluteid || mysoluteid == soluteid && pid <= spid)
            continue;

        sentry = 3 * spid;
        stmp0 = __ldg(g::csolutes[soluteid] + sentry);
        stmp1 = __ldg(g::csolutes[soluteid] + sentry + 1);
        stmp2 = __ldg(g::csolutes[soluteid] + sentry + 2);

        rnd = rnd::mean0var1ii(seed, pid, spid);
        forces::f2k2p(dst0,   dst1,  dst2, SOLID_TYPE, /**/ &a);
        forces::f2k2p(stmp0, stmp1, stmp2, SOLID_TYPE, /**/ &b);
        forces::gen(a, b, rnd, &fx, &fy, &fz);
        xforce += fx;
        yforce += fy;
        zforce += fz;
        atomicAdd(g::csolutesacc[soluteid] + sentry,     -fx);
        atomicAdd(g::csolutesacc[soluteid] + sentry + 1, -fy);
        atomicAdd(g::csolutesacc[soluteid] + sentry + 2, -fz);
    }

    atomicAdd(ff + 3 * pid + 0, xforce);
    atomicAdd(ff + 3 * pid + 1, yforce);
    atomicAdd(ff + 3 * pid + 2, zforce);
}
