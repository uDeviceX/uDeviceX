namespace k_read { /* collective (wrap) read */
__device__ __inline__
void AOS6f(const float2 * const data, const int nparticles, float2& s0, float2& s1, float2& s2)
{
    if (nparticles == 0)
    return;

    int laneid;
    laneid = d::lane();

    const int nfloat2 = 3 * nparticles;

    if (laneid < nfloat2)
    s0 = data[laneid];

    if (laneid + 32 < nfloat2)
    s1 = data[laneid + 32];

    if (laneid + 64 < nfloat2)
    s2 = data[laneid + 64];

    const int srclane0 = (3 * laneid + 0) % warpSize;
    const int srclane1 = (srclane0 + 1)   % warpSize;
    const int srclane2 = (srclane0 + 2)   % warpSize;

    const int start = laneid % 3;

    {
        const float t0 = __shfl(start == 0 ? s0.x : start == 1 ? s1.x : s2.x, srclane0);
        const float t1 = __shfl(start == 0 ? s2.x : start == 1 ? s0.x : s1.x, srclane1);
        const float t2 = __shfl(start == 0 ? s1.x : start == 1 ? s2.x : s0.x, srclane2);

        s0.x = t0;
        s1.x = t1;
        s2.x = t2;
    }

    {
        const float t0 = __shfl(start == 0 ? s0.y : start == 1 ? s1.y : s2.y, srclane0);
        const float t1 = __shfl(start == 0 ? s2.y : start == 1 ? s0.y : s1.y, srclane1);
        const float t2 = __shfl(start == 0 ? s1.y : start == 1 ? s2.y : s0.y, srclane2);

        s0.y = t0;
        s1.y = t1;
        s2.y = t2;
    }
}
__device__ __inline__
void AOS3f(const float * const data, const int nparticles, float& s0, float& s1, float& s2)
{
    if (nparticles == 0)
    return;

    int laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

    const int nfloat2 = 3 * nparticles;

    if (laneid < nfloat2)
    s0 = data[laneid];

    if (laneid + 32 < nfloat2)
    s1 = data[laneid + 32];

    if (laneid + 64 < nfloat2)
    s2 = data[laneid + 64];

    const int srclane0 = (3 * laneid + 0) % warpSize;
    const int srclane1 = (srclane0 + 1)   % warpSize;
    const int srclane2 = (srclane0 + 2)   % warpSize;

    const int start = laneid % 3;

    {
        const float t0 = __shfl(start == 0 ? s0 : start == 1 ? s1 : s2, srclane0);
        const float t1 = __shfl(start == 0 ? s2 : start == 1 ? s0 : s1, srclane1);
        const float t2 = __shfl(start == 0 ? s1 : start == 1 ? s2 : s0, srclane2);

        s0 = t0;
        s1 = t1;
        s2 = t2;
    }
}
}
