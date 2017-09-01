__global__ void halo(int nparticles_padded, int ncellentries,
                     int nsolutes, float seed) {
    forces::Pa a, b;
    float fx, fy, fz;
    int laneid, warpid, base, pid;
    int nunpack;
    float2 dst0, dst1, dst2;
    int code, unpackbase;
    float xforce, yforce, zforce;
    int nzplanes, zplane;

    int scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;
    int xcenter, xstart, xcount;
    int ycenter, zcenter, zmy;
    bool zvalid;
    int count0, count1, count2;
    float *dst = NULL;
    int cid0, cid1, cid2;
    int i;
    int m1, m2, slot;
    int soluteid, spid;
    int sentry;
    float2 stmp0, stmp1, stmp2;
    float rnd;

    laneid = threadIdx.x % warpSize;
    warpid = threadIdx.x / warpSize;
    base = 32 * (warpid + 4 * blockIdx.x);
    pid = base + laneid;
    if (base >= nparticles_padded) return;

    {
        code = k_common::fid(g::starts, base);
        unpackbase = base - g::starts[code];
        nunpack = min(32, g::counts[code] - unpackbase);

        if (nunpack == 0) return;

        k_read::AOS6f((float2*)(g::pp[code] + unpackbase),
                      nunpack, dst0, dst1, dst2);

        dst = (float *)(g::ff[code] + unpackbase);
    }

    k_read::AOS3f(dst, nunpack, xforce, yforce, zforce);

    nzplanes = laneid < nunpack ? 3 : 0;

    for (zplane = 0; zplane < nzplanes; ++zplane) {

        {
            xcenter = XOFFSET + (int)floorf(dst0.x);
            xstart = max(0, xcenter - 1);
            xcount = min(XCELLS, xcenter + 2) - xstart;

            if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) continue;

            ycenter = YOFFSET + (int)floorf(dst0.y);

            zcenter = ZOFFSET + (int)floorf(dst1.x);
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

        for (i = 0; i < ncandidates; ++i) {
            m1 = (int)(i >= scan1);
            m2 = (int)(i >= scan2);
            slot = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);
            get(slot, &soluteid, &spid);

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
    }

    k_write::AOS3f(dst, nunpack, xforce, yforce, zforce);
}
