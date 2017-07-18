namespace k_fsi {
__global__ void halo(const int n0,
                     const int n1, float *const ff1,
                     const float seed) {
    const int laneid = threadIdx.x & 0x1f;
    const int warpid = threadIdx.x >> 5;
    const int localbase = 32 * (warpid + 4 * blockIdx.x);
    const int pid = localbase + laneid;

    if (localbase >= n0) return;

    int nunpack;
    float2 dst0, dst1, dst2;
    float *dst = NULL;

    {
        const uint key9 = 9 * (localbase >= packstarts_padded[9]) +
            9 * (localbase >= packstarts_padded[18]);
        const uint key3 = 3 * (localbase >= packstarts_padded[key9 + 3]) +
            3 * (localbase >= packstarts_padded[key9 + 6]);
        const uint key1 = (localbase >= packstarts_padded[key9 + key3 + 1]) +
            (localbase >= packstarts_padded[key9 + key3 + 2]);
        const int code = key9 + key3 + key1;
        const int unpackbase = localbase - packstarts_padded[code];

        nunpack = min(32, packcount[code] - unpackbase);

        if (nunpack == 0) return;

        k_common::read_AOS6f((float2 *)(packstates[code] + unpackbase), nunpack, dst0, dst1,
                             dst2);

        dst = (float *)(packresults[code] + unpackbase);
    }

    float xforce = 0, yforce = 0, zforce = 0;

    const int nzplanes = laneid < nunpack ? 3 : 0;

    for (int zplane = 0; zplane < nzplanes; ++zplane) {
        int scan1, scan2, ncandidates, spidbase;
        int deltaspid1, deltaspid2;

        {
            enum {
                XCELLS = XS,
                YCELLS = YS,
                ZCELLS = ZS,
                XOFFSET = XCELLS / 2,
                YOFFSET = YCELLS / 2,
                ZOFFSET = ZCELLS / 2
            };

            const int NCELLS = XS * YS * ZS;
            const int xcenter = XOFFSET + (int)floorf(dst0.x);
            const int xstart = max(0, xcenter - 1);
            const int xcount = min(XCELLS, xcenter + 2) - xstart;

            if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) continue;

            const int ycenter = YOFFSET + (int)floorf(dst0.y);

            const int zcenter = ZOFFSET + (int)floorf(dst1.x);
            const int zmy = zcenter - 1 + zplane;
            const bool zvalid = zmy >= 0 && zmy < ZCELLS;

            int count0 = 0, count1 = 0, count2 = 0;

            if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
                const int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
                spidbase = tex1Dfetch(texCellsStart, cid0);
                count0 = ((cid0 + xcount == NCELLS)
                          ? n1
                          : tex1Dfetch(texCellsStart, cid0 + xcount)) -
                    spidbase;
            }

            if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
                const int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
                deltaspid1 = tex1Dfetch(texCellsStart, cid1);
                count1 = ((cid1 + xcount == NCELLS)
                          ? n1
                          : tex1Dfetch(texCellsStart, cid1 + xcount)) -
                    deltaspid1;
            }

            if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
                const int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
                deltaspid2 = tex1Dfetch(texCellsStart, cid2);
                count2 = ((cid2 + xcount == NCELLS)
                          ? n1
                          : tex1Dfetch(texCellsStart, cid2 + xcount)) -
                    deltaspid2;
            }

            scan1 = count0;
            scan2 = count0 + count1;
            ncandidates = scan2 + count2;

            deltaspid1 -= scan1;
            deltaspid2 -= scan2;
        }

        for (int i = 0; i < ncandidates; ++i) {
            const int m1 = (int)(i >= scan1);
            const int m2 = (int)(i >= scan2);
            const int spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

            const int sentry = 3 * spid;
            const float2 stmp0 = tex1Dfetch(texSolventParticles, sentry);
            const float2 stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);
            const float2 stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);

            const float myrandnr = l::rnd::d::mean0var1ii(seed, pid, spid);

            // check for particle types and compute the DPD force
            float3 pos1 = make_float3(dst0.x, dst0.y, dst1.x),
                pos2 = make_float3(stmp0.x, stmp0.y, stmp1.x);
            float3 vel1 = make_float3(dst1.y, dst2.x, dst2.y),
                vel2 = make_float3(stmp1.y, stmp2.x, stmp2.y);
	
            const float3 strength = force(SOLID_TYPE, SOLVENT_TYPE, pos1, pos2,
                                                             vel1, vel2, myrandnr);

            const float xinteraction = strength.x;
            const float yinteraction = strength.y;
            const float zinteraction = strength.z;

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
}
