__global__ void bulk(float2 *pp, int n,
                     int ncellentries, int nsolutes,
                     float seed, int mysoluteid, float *ff) {
    Map m; /* see map/ */
    float fx, fy, fz, rnd;
    forces::Pa a, b;
    int cnt0, cnt1, cnt2, org0;
    int org1, org2;
    int gid, pid, zplane;
    float2 dst0, dst1, dst2;
    int xcenter, xstart, xcount;
    int ycenter, zcenter, zmy;
    bool zvalid;
    int count0, count1, count2;
    int cid0, cid1, cid2;
    float xforce, yforce, zforce;
    int i, m1, m2, slot;

    int soluteid;
    int spid;
    int sentry;
    float2 stmp0, stmp1, stmp2;

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    pid = gid / 3;
    zplane = gid % 3;

    if (pid >= n) return;

    dst0 = __ldg(pp + 3 * pid + 0);
    dst1 = __ldg(pp + 3 * pid + 1);
    dst2 = __ldg(pp + 3 * pid + 2);


    {
        xcenter = min(XCELLS - 1, max(0, XOFFSET + (int)floorf(dst0.x)));
        xstart = max(0, xcenter - 1);
        xcount = min(XCELLS, xcenter + 2) - xstart;

        if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) return;

        ycenter = min(YCELLS - 1, max(0, YOFFSET + (int)floorf(dst0.y)));

        zcenter = min(ZCELLS - 1, max(0, ZOFFSET + (int)floorf(dst1.x)));
        zmy = zcenter - 1 + zplane;
        zvalid = zmy >= 0 && zmy < ZCELLS;

        count0 = count1 = count2 = 0;
        

        if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
            cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
            org0 = fetchS(cid0);
            count0 = fetchS(cid0 + xcount) - org0;
        }

        if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
            cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
            org1 = fetchS(cid1);
            count1 = fetchS(cid1 + xcount) - org1;
        }

        if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
            cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
            org2 = fetchS(cid2);
            count2 = fetchS(cid2 + xcount) - org2;
        }

        cnt0 = count0;
        cnt1 = count0 + count1;
        cnt2 = cnt1 + count2;

        org1 -= cnt0;
        org2 -= cnt1;
    }
    
    xforce = yforce = zforce = 0;
    for (i = 0; i < cnt2; ++i) {
        m1 = (int)(i >= cnt0);
        m2 = (int)(i >= cnt1);
        slot = i + (m2 ? org2 : m1 ? org1 : org0);

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
