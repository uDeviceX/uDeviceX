namespace k_cnt {
__global__ void halo(int nparticles_padded, int ncellentries,
                     int nsolutes, float seed) {
    int laneid = threadIdx.x % warpSize;
    int warpid = threadIdx.x / warpSize;
    int localbase = 32 * (warpid + 4 * blockIdx.x);
    int pid = localbase + laneid;

    if (localbase >= nparticles_padded) return;

    int nunpack;
    float2 dst0, dst1, dst2;
    float *dst = NULL;

    {
        uint key9 = 9 * (localbase >= packstarts_padded[9]) +
            9 * (localbase >= packstarts_padded[18]);
        uint key3 = 3 * (localbase >= packstarts_padded[key9 + 3]) +
            3 * (localbase >= packstarts_padded[key9 + 6]);
        uint key1 = (localbase >= packstarts_padded[key9 + key3 + 1]) +
            (localbase >= packstarts_padded[key9 + key3 + 2]);
        int code = key9 + key3 + key1;
        int unpackbase = localbase - packstarts_padded[code];

        nunpack = min(32, packcount[code] - unpackbase);

        if (nunpack == 0) return;

        k_read::AOS6f((float2 *)(packstates[code] + unpackbase), nunpack, dst0, dst1, dst2);

        dst = (float *)(packresults[code] + unpackbase);
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
                spidbase = tex1Dfetch(texCellsStart, cid0);
                count0 = tex1Dfetch(texCellsStart, cid0 + xcount) - spidbase;
            }

            if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
                int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
                deltaspid1 = tex1Dfetch(texCellsStart, cid1);
                count1 = tex1Dfetch(texCellsStart, cid1 + xcount) - deltaspid1;
            }

            if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
                int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
                deltaspid2 = tex1Dfetch(texCellsStart, cid2);
                count2 = tex1Dfetch(texCellsStart, cid2 + xcount) - deltaspid2;
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

            CellEntry ce;
            ce.pid = tex1Dfetch(texCellEntries, slot);
            int soluteid = ce.code.w;
            ce.code.w = 0;

            int spid = ce.pid;

            int sentry = 3 * spid;
            float2 stmp0 = __ldg(csolutes[soluteid] + sentry);
            float2 stmp1 = __ldg(csolutes[soluteid] + sentry + 1);
            float2 stmp2 = __ldg(csolutes[soluteid] + sentry + 2);

            float myrandnr = rnd::mean0var1ii(seed, pid, spid);

            // check for particle types and compute the DPD force
            float3 pos1 = make_float3(dst0.x, dst0.y, dst1.x),
                pos2 = make_float3(stmp0.x, stmp0.y, stmp1.x);
            float3 vel1 = make_float3(dst1.y, dst2.x, dst2.y),
                vel2 = make_float3(stmp1.y, stmp2.x, stmp2.y);
            int type1 = SOLID_TYPE;
            int type2 = SOLID_TYPE;
            float3 strength = forces::dpd(type1, type2, pos1, pos2, vel1, vel2, myrandnr);

            float xinteraction = strength.x;
            float yinteraction = strength.y;
            float zinteraction = strength.z;

            xforce += xinteraction;
            yforce += yinteraction;
            zforce += zinteraction;

            atomicAdd(csolutesacc[soluteid] + sentry, -xinteraction);
            atomicAdd(csolutesacc[soluteid] + sentry + 1, -yinteraction);
            atomicAdd(csolutesacc[soluteid] + sentry + 2, -zinteraction);
        }
    }

    k_write::AOS3f(dst, nunpack, xforce, yforce, zforce);
}
}
