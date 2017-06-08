namespace k_wall {
texture<float4, 1, cudaReadModeElementType> texWallParticles;
texture<int, 1, cudaReadModeElementType> texWallCellStart, texWallCellCount;

__global__ void interactions_3tpp(const float2 *const pp, const int np,
                                  const int nsolid, float *const acc,
                                  const float seed) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int pid = gid / 3;
    int zplane = gid % 3;

    if (pid >= np) return;

    float2 dst0 = pp[3 * pid + 0];
    float2 dst1 = pp[3 * pid + 1];

    float interacting_threshold =
        -1 - 1.7320f * ((float)XSIZE_WALLCELLS / (float)XTEXTURESIZE);

    if (k_sdf::cheap_sdf(dst0.x, dst0.y, dst1.x) <= interacting_threshold) return;

    float2 dst2 = pp[3 * pid + 2];

    uint scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;

    {
        int xbase = (int)(dst0.x - (-XS / 2 - XWM));
        int ybase = (int)(dst0.y - (-YS / 2 - YMARGIN_WALL));
        int zbase = (int)(dst1.x - (-ZS / 2 - ZMARGIN_WALL));

        enum {
            XCELLS = XS + 2 * XWM,
            YCELLS = YS + 2 * YMARGIN_WALL,
            ZCELLS = ZS + 2 * ZMARGIN_WALL,
            NCELLS = XCELLS * YCELLS * ZCELLS
        };

        int cid0 = xbase - 1 + XCELLS * (ybase - 1 + YCELLS * (zbase - 1 + zplane));

        spidbase = tex1Dfetch(texWallCellStart, cid0);
        int count0 = tex1Dfetch(texWallCellStart, cid0 + 3) - spidbase;

        int cid1 = cid0 + XCELLS;
        deltaspid1 = tex1Dfetch(texWallCellStart, cid1);
        int count1 = tex1Dfetch(texWallCellStart, cid1 + 3) - deltaspid1;

        int cid2 = cid0 + XCELLS * 2;
        deltaspid2 = tex1Dfetch(texWallCellStart, cid2);
        int count2 = cid2 + 3 == NCELLS
            ? nsolid
            : tex1Dfetch(texWallCellStart, cid2 + 3) - deltaspid2;

        scan1 = count0;
        scan2 = count0 + count1;
        ncandidates = scan2 + count2;

        deltaspid1 -= scan1;
        deltaspid2 -= scan2;
    }

    float xforce = 0, yforce = 0, zforce = 0;

#define zig x
#define zag y

#define uno x
#define due y
#define tre z

#define mf3 make_float3
    float  x = dst0.zig,  y = dst0.zag,  z = dst1.zig; /* bulk particle  */
    float vx = dst1.zag, vy = dst2.zig, vz = dst2.zag;

    for (int i = 0; i < ncandidates; ++i) {
        int m1 = (int)(i >= scan1);
        int m2 = (int)(i >= scan2);
        int spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);
        float4 stmp0 = tex1Dfetch(texWallParticles, spid);

        float  xw = stmp0.uno, yw = stmp0.due, zw = stmp0.tre; /* wall particle */
        float vxw, vyw, vzw;
        k_wvel::vell(xw, yw, zw, &vxw, &vyw, &vzw);
        float rnd = Logistic::mean0var1(seed, pid, spid);

        // check for particle types and compute the DPD force

        float3 strength = compute_dpd_force_traced(SOLVENT_TYPE, WALL_TYPE,
                                                   mf3(x ,  y,  z), mf3( xw,  yw,  zw),
                                                   mf3(vx, vy, vz), mf3(vxw, vyw, vzw), rnd);
        xforce += strength.x; yforce += strength.y; zforce += strength.z;
    }
#undef zig
#undef zag

#undef uno
#undef due
#undef tre
#undef mf3
    atomicAdd(acc + 3 * pid + 0, xforce);
    atomicAdd(acc + 3 * pid + 1, yforce);
    atomicAdd(acc + 3 * pid + 2, zforce);
}
} /* namespace k_wall */
