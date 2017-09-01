__global__ void bulk(float2 *particles, int np,
                     int ncellentries, int nsolutes,
                     float *acc, float seed,
                     int mysoluteid) {
    float fx, fy, fz, rnd;
    forces::Pa a, b;
    int scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;
    int gid, pid, zplane;
    float2 dst0, dst1, dst2;
    int xcenter, xstart, xcount;
    int ycenter, zcenter, zmy, zvalid;
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

    if (pid >= np) return;

    dst0 = __ldg(particles + 3 * pid + 0);
    dst1 = __ldg(particles + 3 * pid + 1);
    dst2 = __ldg(particles + 3 * pid + 2);


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
            spidbase = fetchS(cid0);
            count0 = fetchS(cid0 + xcount) - spidbase;
        }

        if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
            cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
            deltaspid1 = fetchS(cid1);
            count1 = fetchS(cid1 + xcount) - deltaspid1;
        }

        if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
            cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
            deltaspid2 = fetchS(cid2);
            count2 = fetchS(cid2 + xcount) - deltaspid2;
        }

        scan1 = count0;
        scan2 = count0 + count1;
        ncandidates = scan2 + count2;

        deltaspid1 -= scan1;
        deltaspid2 -= scan2;
    }

    xforce = yforce = zforce = 0;
    for (i = 0; i < ncandidates; ++i) {
        m1 = (int)(i >= scan1);
        m2 = (int)(i >= scan2);
        slot = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

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

    atomicAdd(acc + 3 * pid + 0, xforce);
    atomicAdd(acc + 3 * pid + 1, yforce);
    atomicAdd(acc + 3 * pid + 2, zforce);
}
