namespace k_cnt {
__global__ void bulk(float2 *particles, int np,
                     int ncellentries, int nsolutes,
                     float *acc, float seed,
                     int mysoluteid) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int pid = gid / 3;
    int zplane = gid % 3;

    if (pid >= np) return;

    float2 dst0 = __ldg(particles + 3 * pid + 0);
    float2 dst1 = __ldg(particles + 3 * pid + 1);
    float2 dst2 = __ldg(particles + 3 * pid + 2);

    int scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;

    {
        int xcenter = min(XCELLS - 1, max(0, XOFFSET + (int)floorf(dst0.x)));
        int xstart = max(0, xcenter - 1);
        int xcount = min(XCELLS, xcenter + 2) - xstart;

        if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) return;

        int ycenter = min(YCELLS - 1, max(0, YOFFSET + (int)floorf(dst0.y)));

        int zcenter = min(ZCELLS - 1, max(0, ZOFFSET + (int)floorf(dst1.x)));
        int zmy = zcenter - 1 + zplane;
        bool zvalid = zmy >= 0 && zmy < ZCELLS;

        int count0 = 0, count1 = 0, count2 = 0;

        if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
            int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
            spidbase = tex1Dfetch(t::start, cid0);
            count0 = tex1Dfetch(t::start, cid0 + xcount) - spidbase;
        }

        if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
            int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
            deltaspid1 = tex1Dfetch(t::start, cid1);
            count1 = tex1Dfetch(t::start, cid1 + xcount) - deltaspid1;
        }

        if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
            int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
            deltaspid2 = tex1Dfetch(t::start, cid2);
            count2 = tex1Dfetch(t::start, cid2 + xcount) - deltaspid2;
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
        int m1 = (int)(i >= scan1);
        int m2 = (int)(i >= scan2);
        int slot = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

        CellEntry ce;
        ce.pid = tex1Dfetch(t::id, slot);
        int soluteid = ce.code.w;

        ce.code.w = 0;

        int spid = ce.pid;

        if (mysoluteid < soluteid || mysoluteid == soluteid && pid <= spid)
        continue;

        int sentry = 3 * spid;
        float2 stmp0 = __ldg(g::csolutes[soluteid] + sentry);
        float2 stmp1 = __ldg(g::csolutes[soluteid] + sentry + 1);
        float2 stmp2 = __ldg(g::csolutes[soluteid] + sentry + 2);

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

        atomicAdd(g::csolutesacc[soluteid] + sentry, -xinteraction);
        atomicAdd(g::csolutesacc[soluteid] + sentry + 1, -yinteraction);
        atomicAdd(g::csolutesacc[soluteid] + sentry + 2, -zinteraction);
    }

    atomicAdd(acc + 3 * pid + 0, xforce);
    atomicAdd(acc + 3 * pid + 1, yforce);
    atomicAdd(acc + 3 * pid + 2, zforce);
}
}
