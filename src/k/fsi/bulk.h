namespace k_fsi {
__global__ void bulk(const float2 *const particles, const int np,
                     const int nsolvent, float *const acc,
                     float *const accsolvent, const float seed) {
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;
    const int pid = gid / 3;
    const int zplane = gid % 3;

    if (pid >= np) return;

    const float2 dst0 = __ldg(particles + 3 * pid + 0);
    const float2 dst1 = __ldg(particles + 3 * pid + 1);
    const float2 dst2 = __ldg(particles + 3 * pid + 2);
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

        const int xcenter = XOFFSET + (int)floorf(dst0.x);
        const int xstart = max(0, xcenter - 1);
        const int xcount = min(XCELLS, xcenter + 2) - xstart;

        if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) return;

        const int ycenter = YOFFSET + (int)floorf(dst0.y);

        const int zcenter = ZOFFSET + (int)floorf(dst1.x);
        const int zmy = zcenter - 1 + zplane;
        const bool zvalid = zmy >= 0 && zmy < ZCELLS;

        int count0 = 0, count1 = 0, count2 = 0;

        if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
            const int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
            spidbase = tex1Dfetch(texCellsStart, cid0);
            count0 = ((cid0 + xcount == NCELLS)
                      ? nsolvent
                      : tex1Dfetch(texCellsStart, cid0 + xcount)) -
                spidbase;
        }

        if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
            const int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
            deltaspid1 = tex1Dfetch(texCellsStart, cid1);
            count1 = ((cid1 + xcount == NCELLS)
                      ? nsolvent
                      : tex1Dfetch(texCellsStart, cid1 + xcount)) -
                deltaspid1;
        }

        if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
            const int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
            deltaspid2 = tex1Dfetch(texCellsStart, cid2);
            count2 = ((cid2 + xcount == NCELLS)
                      ? nsolvent
                      : tex1Dfetch(texCellsStart, cid2 + xcount)) -
                deltaspid2;
        }

        scan1 = count0;
        scan2 = count0 + count1;
        ncandidates = scan2 + count2;

        deltaspid1 -= scan1;
        deltaspid2 -= scan2;
    }

    float xforce = 0, yforce = 0, zforce = 0;

#pragma unroll 3
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

        atomicAdd(accsolvent + sentry, -xinteraction);
        atomicAdd(accsolvent + sentry + 1, -yinteraction);
        atomicAdd(accsolvent + sentry + 2, -zinteraction);
    }

    atomicAdd(acc + 3 * pid + 0, xforce);
    atomicAdd(acc + 3 * pid + 1, yforce);
    atomicAdd(acc + 3 * pid + 2, zforce);
}
}
