namespace k_fsi {
static __device__ unsigned int get_hid(const int a[], const int i) {
    /* where is `i' in sorted a[27]? */
    int k1, k3, k9;
    k9 = 9 * ((i >= a[9])           + (i >= a[18]));
    k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
    k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}

__global__ void halo(int n0, int n1, float seed, float *ff1) {
    int laneid, warpid, localbase, pid;
    int nunpack;
    float2 dst0, dst1, dst2;
    float *dst = NULL;
    int fid; /* fragment id */
    int unpackbase;

    int scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;
    int nzplanes;
    int zplane;
    int NCELLS;

    int xstart, xcount;
    int zmy;
    int xcenter, ycenter, zcenter;
    bool zvalid;
    int count0, count1, count2;
    int cid0, cid1, cid2;
    int i, m1, m2, spid;
    int sentry;
    float2 stmp0, stmp1, stmp2;
    float myrandnr;

    float3 pos1, pos2, vel1, vel2;
    float3 strength;
    float xinteraction, yinteraction, zinteraction;

    float xforce, yforce, zforce;
    
    laneid = threadIdx.x & 0x1f;
    warpid = threadIdx.x >> 5;
    localbase = 32 * (warpid + 4 * blockIdx.x);
    pid = localbase + laneid;
    if (localbase >= n0) return;

    fid = get_hid(packstarts_padded, localbase);
    unpackbase = localbase - packstarts_padded[fid];

    nunpack = min(32, packcount[fid] - unpackbase);
    if (nunpack == 0) return;

    k_common::read_AOS6f((float2 *)(packstates[fid] + unpackbase), nunpack, dst0, dst1, dst2);
    dst = (float *)(packresults[fid] + unpackbase);

    xforce = yforce = zforce = 0;
    nzplanes = laneid < nunpack ? 3 : 0;
    for (zplane = 0; zplane < nzplanes; ++zplane) {
        {
            enum {
                XCELLS = XS,
                YCELLS = YS,
                ZCELLS = ZS,
                XOFFSET = XCELLS / 2,
                YOFFSET = YCELLS / 2,
                ZOFFSET = ZCELLS / 2
            };

            NCELLS = XS * YS * ZS;
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
                spidbase = tex1Dfetch(texCellsStart, cid0);
                count0 = ((cid0 + xcount == NCELLS)
                          ? n1
                          : tex1Dfetch(texCellsStart, cid0 + xcount)) -
                    spidbase;
            }

            if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
                cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
                deltaspid1 = tex1Dfetch(texCellsStart, cid1);
                count1 = ((cid1 + xcount == NCELLS)
                          ? n1
                          : tex1Dfetch(texCellsStart, cid1 + xcount)) -
                    deltaspid1;
            }

            if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
                cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
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

        for (i = 0; i < ncandidates; ++i) {
            m1 = (int)(i >= scan1);
            m2 = (int)(i >= scan2);
            spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

            sentry = 3 * spid;
            stmp0 = tex1Dfetch(texSolventParticles, sentry);
            stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);
            stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);

            myrandnr = l::rnd::d::mean0var1ii(seed, pid, spid);

            // check for particle types and compute the DPD force
            pos1 = make_float3(dst0.x, dst0.y, dst1.x);
            pos2 = make_float3(stmp0.x, stmp0.y, stmp1.x);
            vel1 = make_float3(dst1.y, dst2.x, dst2.y);
            vel2 = make_float3(stmp1.y, stmp2.x, stmp2.y);
            strength = force(SOLID_TYPE, SOLVENT_TYPE, pos1, pos2, vel1, vel2, myrandnr);

            xinteraction = strength.x;
            yinteraction = strength.y;
            zinteraction = strength.z;

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
