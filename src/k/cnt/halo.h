namespace k_cnt {
__global__ void halo(int nparticles_padded, int ncellentries,
                     int nsolutes, float seed) {
    forces::Pa a, b;
    float fx, fy, fz;

    int laneid = threadIdx.x % warpSize;
    int warpid = threadIdx.x / warpSize;
    int base = 32 * (warpid + 4 * blockIdx.x);
    int pid = base + laneid;
    if (base >= nparticles_padded) return;

    int nunpack;
    float2 dst0, dst1, dst2;
    float *dst = NULL;

    {
        int code = k_common::fid(g::starts, base);
        int unpackbase = base - g::starts[code];
        nunpack = min(32, g::counts[code] - unpackbase);

        if (nunpack == 0) return;

        k_read::AOS6f((float2*)(g::pp[code] + unpackbase),
                      nunpack, dst0, dst1, dst2);

        dst = (float *)(g::ff[code] + unpackbase);
    }

    float xforce, yforce, zforce;
    k_read::AOS3f(dst, nunpack, xforce, yforce, zforce);

    int nzplanes = laneid < nunpack ? 3 : 0;

    for (int zplane = 0; zplane < nzplanes; ++zplane) {
        int scan1, scan2, ncandidates, spidbase;
        int deltaspid1, deltaspid2;

        {
            int xcenter = XOFFSET + (int)floorf(dst0.x);
            int xstart = max(0, xcenter - 1);
            int xcount = min(XCELLS, xcenter + 2) - xstart;

            if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) continue;

            int ycenter = YOFFSET + (int)floorf(dst0.y);

            int zcenter = ZOFFSET + (int)floorf(dst1.x);
            int zmy = zcenter - 1 + zplane;
            bool zvalid = zmy >= 0 && zmy < ZCELLS;

            int count0 = 0, count1 = 0, count2 = 0;

            if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
                int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
                spidbase = fetchS(cid0);
                count0 = fetchS(cid0 + xcount) - spidbase;
            }

            if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
                int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
                deltaspid1 = fetchS(cid1);
                count1 = fetchS(cid1 + xcount) - deltaspid1;
            }

            if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
                int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
                deltaspid2 = fetchS(cid2);
                count2 = fetchS(cid2 + xcount) - deltaspid2;
            }

            scan1 = count0;
            scan2 = count0 + count1;
            ncandidates = scan2 + count2;

            deltaspid1 -= scan1;
            deltaspid2 -= scan2;
        }

        for (int i = 0; i < ncandidates; ++i) {
            int m1 = (int)(i >= scan1);
            int m2 = (int)(i >= scan2);
            int slot = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

            int soluteid, spid;
            get(slot, &soluteid, &spid);

            int sentry = 3 * spid;
            float2 stmp0 = __ldg(g::csolutes[soluteid] + sentry);
            float2 stmp1 = __ldg(g::csolutes[soluteid] + sentry + 1);
            float2 stmp2 = __ldg(g::csolutes[soluteid] + sentry + 2);

            float rnd = rnd::mean0var1ii(seed, pid, spid);
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
}
