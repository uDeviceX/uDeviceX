namespace k_fsi {

struct Pa { /* local particle */
    float x, y, z;
    float vx, vy, vz;
};

static __device__ float fst(float2 p) { return p.x; }
static __device__ float scn(float2 p) { return p.y; }
static __device__ void p2rv(const float2 *p, uint i, /**/
                            float  *x, float  *y, float  *z,
                            float *vx, float *vy, float *vz) {
    float2 s0, s1, s2;
    p += 3*i;
    s0 = __ldg(p++); s1 = __ldg(p++); s2 = __ldg(p++);
     *x = fst(s0);  *y = scn(s0);  *z = fst(s1);
    *vx = scn(s1); *vy = fst(s2); *vz = scn(s2);
}

static __device__ Pa pp2p(float2 *pp, int i) {
    Pa p;
    p2rv(pp, i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    return p;
}

static __device__ void bulk0(float2 *pp, int pid, int zplane, int n, float seed, float *ff0, float *ff1) {
    Pa p;
    float x, y, z;
    p = pp2p(pp, pid);
    x = p.x; y = p.y; z = p.z;
    int cnt0, cnt1, cnt2, org0;
    int org1, org2;

    {
        enum {
            XCELLS = XS,
            YCELLS = YS,
            ZCELLS = ZS,
            XOFFSET = XCELLS / 2,
            YOFFSET = YCELLS / 2,
            ZOFFSET = ZCELLS / 2
        };

        const int xcenter = XOFFSET + (int)floorf(x);
        const int xstart = max(0, xcenter - 1);
        const int xcount = min(XCELLS, xcenter + 2) - xstart;

        if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) return;

        const int ycenter = YOFFSET + (int)floorf(y);

        const int zcenter = ZOFFSET + (int)floorf(z);
        const int zmy = zcenter - 1 + zplane;
        const bool zvalid = zmy >= 0 && zmy < ZCELLS;

        int count0 = 0, count1 = 0, count2 = 0;

        if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
            const int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
            org0 = tex1Dfetch(texCellsStart, cid0);
            count0 = ((cid0 + xcount == NCELLS)
                      ? n
                      : tex1Dfetch(texCellsStart, cid0 + xcount)) -
                org0;
        }

        if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
            const int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
            org1 = tex1Dfetch(texCellsStart, cid1);
            count1 = ((cid1 + xcount == NCELLS)
                      ? n
                      : tex1Dfetch(texCellsStart, cid1 + xcount)) -
                org1;
        }

        if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
            const int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
            org2 = tex1Dfetch(texCellsStart, cid2);
            count2 = ((cid2 + xcount == NCELLS)
                      ? n
                      : tex1Dfetch(texCellsStart, cid2 + xcount)) -
                org2;
        }

        cnt0 = count0;
        cnt1 = count0 + count1;
        cnt2 = cnt1 + count2;

        org1 -= cnt0;
        org2 -= cnt1;
    }

    float xforce = 0, yforce = 0, zforce = 0;
    for (int i = 0; i < cnt2; ++i) {
        const int m1 = (int)(i >= cnt0);
        const int m2 = (int)(i >= cnt1);
        const int spid = i + (m2 ? org2 : m1 ? org1 : org0);

        const int sentry = 3 * spid;
        const float2 stmp0 = tex1Dfetch(texSolventParticles, sentry);
        const float2 stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);
        const float2 stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);

        const float myrandnr = l::rnd::d::mean0var1ii(seed, pid, spid);

        // check for particle types and compute the DPD force
        float3 pos1 = make_float3(x, y, z), pos2 = make_float3(stmp0.x, stmp0.y, stmp1.x);
        float3 vel1 = make_float3(p.vx, p.vy, p.vz), vel2 = make_float3(stmp1.y, stmp2.x, stmp2.y);

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

    atomicAdd(ff0 + 3 * pid + 0, xforce);
    atomicAdd(ff0 + 3 * pid + 1, yforce);
    atomicAdd(ff0 + 3 * pid + 2, zforce);
}

__global__ void bulk(float2 *pp, int n0, int n1, float seed, float *ff0, float *ff1) {
    int gid, pid, zplane;
    gid    = threadIdx.x + blockDim.x * blockIdx.x;
    pid    = gid / 3;
    zplane = gid % 3;
    if (pid >= n0) return;
    bulk0(pp, pid, zplane, n1, seed, ff0, ff1);
}
}
