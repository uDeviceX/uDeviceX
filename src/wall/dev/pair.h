namespace sdfdev = sdf::sub::dev;
__device__ int minmax(int lo, int hi, int a) { return min(hi, max(lo, a)); }
typedef const sdf::tex3Dca<float> TexSDF_t;
__global__ void pair(TexSDF_t texsdf, hforces::Cloud cloud, const int np, const int w_n,
                     float *const ff, const float seed,
                     const Texo<int> texstart, const Texo<float4> texwpp) {
#define start_fetch(i) (texstart.fetch(i))
#define   wpp_fetch(i) (texwpp.fetch(i))
    map::Map m;

    forces::Pa a, b;  /* bulk and wall particles */
    float vx, vy, vz; /* wall velocity */
    float fx, fy, fz, rnd;
    float x, y, z;
    int gid, pid, zplane;
    uint cnt0, cnt1, ncandidates, org0;
    int org1, org2;
    float threshold;

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    pid = gid / 3;
    zplane = gid % 3;

    if (pid >= np) return;

    fetch(cloud, pid, /**/ &a);
    forces::p2r3(&a, /**/ &x, &y, &z);

    threshold =
        -1 - 1.7320f * ((float)XSIZE_WALLCELLS / (float)XTE);
    if (sdfdev::cheap_sdf(texsdf, x, y, z) <= threshold) return;

    {
        int xbase = (int)(x - (-XS / 2 - XWM));
        int ybase = (int)(y - (-YS / 2 - YWM));
        int zbase = (int)(z - (-ZS / 2 - ZWM));

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

        org0 = start_fetch(cid0);
        int count0 = start_fetch(cid0 + 3) - org0;

        int cid1 = cid0 + XCELLS;
        org1 = start_fetch(cid1);
        int count1 = start_fetch(cid1 + 3) - org1;

        int cid2 = cid0 + XCELLS * 2;
        org2 = start_fetch(cid2);
        int count2 = cid2 + 3 == NCELLS
            ? w_n
            : start_fetch(cid2 + 3) - org2;

        cnt0 = count0;
        cnt1 = count0 + count1;
        ncandidates = cnt1 + count2;

        org1 -= cnt0;
        org2 -= cnt1;
    }

    float xforce = 0, yforce = 0, zforce = 0;

    for (int i = 0; i < ncandidates; ++i) {
        int m1 = (int)(i >= cnt0);
        int m2 = (int)(i >= cnt1);
        int spid = i + (m2 ? org2 : m1 ? org1 : org0);
        const float4 r = wpp_fetch(spid);
        k_wvel::vell(r.x, r.y, r.z, /**/ &vx, &vy, &vz);
        forces::r3v3k2p(r.x, r.y, r.z, vx, vy, vz, WALL_KIND, /**/ &b);
        rnd = rnd::mean0var1ii(seed, pid, spid);
        forces::gen(a, b, rnd, /**/ &fx, &fy, &fz);
        xforce += fx; yforce += fy; zforce += fz;
    }
    atomicAdd(ff + 3 * pid + 0, xforce);
    atomicAdd(ff + 3 * pid + 1, yforce);
    atomicAdd(ff + 3 * pid + 2, zforce);
#undef start_fetch
#undef wpp_fetch
}
