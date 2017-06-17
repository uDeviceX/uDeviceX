namespace dev {
  enum {X, Y, Z};

__global__ void strip_solid4(Particle *const src, const int n, float4 *dst) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle p = src[pid];
    dst[pid] = make_float4(p.r[0], p.r[1], p.r[2], 0);
}

__device__ int minmax(int lo, int hi, int a) { return min(hi, max(lo, a)); }

__global__ void interactions_3tpp(const float2 *const pp, const int np, const int w_n, float *const acc,
                                  const float seed, const int type, const Texo<int> texstart, const Texo<float4> texwpp) {
                                  
#define start_fetch(i) (texstart.fetch(i))
#define   wpp_fetch(i) (texwpp.fetch(i))

    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int pid = gid / 3;
    int zplane = gid % 3;

    if (pid >= np) return;

    float2 dst0 = pp[3 * pid + 0];
    float2 dst1 = pp[3 * pid + 1];

    float interacting_threshold =
        -1 - 1.7320f * ((float)XSIZE_WALLCELLS / (float)XTE);

    if (k_sdf::cheap_sdf(dst0.x, dst0.y, dst1.x) <= interacting_threshold) return;

    float2 dst2 = pp[3 * pid + 2];

    uint scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;

    {
        int xbase = (int)(dst0.x - (-XS / 2 - XWM));
        int ybase = (int)(dst0.y - (-YS / 2 - YWM));
        int zbase = (int)(dst1.x - (-ZS / 2 - ZWM));

        xbase = minmax(-XWM+1, XS + XWM - 2, xbase);
        ybase = minmax(-YWM+1, YS + YWM - 2, ybase);
        zbase = minmax(-ZWM+1, ZS + ZWM - 2, zbase);
                
        enum {
            XCELLS = XS + 2 * XWM,
            YCELLS = YS + 2 * YWM,
            ZCELLS = ZS + 2 * ZWM,
            NCELLS = XCELLS * YCELLS * ZCELLS
        };

        int cid0 = xbase - 1 + XCELLS * (ybase - 1 + YCELLS * (zbase - 1 + zplane));

        spidbase = start_fetch(cid0);
        int count0 = start_fetch(cid0 + 3) - spidbase;

        int cid1 = cid0 + XCELLS;
        deltaspid1 = start_fetch(cid1);
        int count1 = start_fetch(cid1 + 3) - deltaspid1;

        int cid2 = cid0 + XCELLS * 2;
        deltaspid2 = start_fetch(cid2);
        int count2 = cid2 + 3 == NCELLS
            ? w_n
            : start_fetch(cid2 + 3) - deltaspid2;

        scan1 = count0;
        scan2 = count0 + count1;
        ncandidates = scan2 + count2;

        deltaspid1 -= scan1;
        deltaspid2 -= scan2;
    }

    float xforce = 0, yforce = 0, zforce = 0;

#define zig x
#define zag y
#define mf3 make_float3
    float  x = dst0.zig,  y = dst0.zag,  z = dst1.zig; /* bulk particle  */
    float vx = dst1.zag, vy = dst2.zig, vz = dst2.zag;

    for (int i = 0; i < ncandidates; ++i) {
        int m1 = (int)(i >= scan1);
        int m2 = (int)(i >= scan2);
        int spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

        const float4 rw = wpp_fetch(spid); /* wall particle */

        float vxw, vyw, vzw;
        k_wvel::vell(rw.x, rw.y, rw.z, &vxw, &vyw, &vzw);
        float rnd = l::rnd::d::mean0var1(seed, pid, spid);

        // check for particle types and compute the DPD force

        float3 strength = force(type, WALL_TYPE,
				mf3(x ,  y,  z), mf3(rw.x, rw.y, rw.z),
				mf3(vx, vy, vz), mf3( vxw,  vyw,  vzw), rnd);
        xforce += strength.x; yforce += strength.y; zforce += strength.z;
    }
#undef zig
#undef zag
    atomicAdd(acc + 3 * pid + 0, xforce);
    atomicAdd(acc + 3 * pid + 1, yforce);
    atomicAdd(acc + 3 * pid + 2, zforce);
#undef start_fetch
#undef wpp_fetch
}
} /* namespace k_wall */
