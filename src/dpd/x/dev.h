enum {X, Y, Z};

/* particle - float2 union */
union Part {
    float2 f2[3];
    struct { float r[3], v[3]; };
};

__device__ Part tex2Part(const Texo<float2> texpp, const int id) {
    Part p;
    p.f2[0] = texvert.fetch(3 * id + 0);
    p.f2[1] = texvert.fetch(3 * id + 1);
    p.f2[2] = texvert.fetch(3 * id + 2);
    return p;
}

__global__ void interactions(const Texo<float2> texpp, const int *sstart, int n, /**/ Force *ff) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int pid = gid / 3;
    int zplane = gid % 3;

#define start_fetch(id) sstart[id]
    
    if (pid >= n) return;
    
    Part p = tex2Part(texpp, pid);
    /* TODO check from here */
    uint scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;

    {
        int xcid = (int)(p.r[X] + XS / 2);
        int ycid = (int)(p.r[Y] + YS / 2));
        int zcid = (int)(p.r[Z] + ZS / 2));
                
        enum {
            XCELLS = XS,
            YCELLS = YS,
            ZCELLS = ZS,
            NCELLS = XCELLS * YCELLS * ZCELLS
        };
/* needs check on boundaries */
        int cid0 = xcid - 1 + XCELLS * (ycid - 1 + YCELLS * (zcid - 1 + zplane));

        spidbase = start_fetch(cid0);
        int count0 = start_fetch(cid0 + 3) - spidbase;

        int cid1 = cid0 + XCELLS;
        deltaspid1 = start_fetch(cid1);
        int count1 = start_fetch(cid1 + 3) - deltaspid1;

        int cid2 = cid0 + XCELLS * 2;
        deltaspid2 = start_fetch(cid2);
        int count2 = cid2 + 3 == NCELLS
            ? n
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
        float rnd = l::rnd::d::mean0var1ii(seed, pid, spid);
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
