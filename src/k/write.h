namespace k_write { /* collective (wrap) write */
__device__ __forceinline__
void AOS6f(float2 * const data, const int nparticles, float2& s0, float2& s1, float2& s2)
{
    if (nparticles == 0)
    return;

    int laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

    const int srclane0 = (32 * ((laneid) % 3) + laneid) / 3;
    const int srclane1 = (32 * ((laneid + 1) % 3) + laneid) / 3;
    const int srclane2 = (32 * ((laneid + 2) % 3) + laneid) / 3;

    const int start = laneid % 3;

    {
        const float t0 = __shfl(s0.x, srclane0);
        const float t1 = __shfl(s2.x, srclane1);
        const float t2 = __shfl(s1.x, srclane2);

        s0.x = start == 0 ? t0 : start == 1 ? t2 : t1;
        s1.x = start == 0 ? t1 : start == 1 ? t0 : t2;
        s2.x = start == 0 ? t2 : start == 1 ? t1 : t0;
    }

    {
        const float t0 = __shfl(s0.y, srclane0);
        const float t1 = __shfl(s2.y, srclane1);
        const float t2 = __shfl(s1.y, srclane2);

        s0.y = start == 0 ? t0 : start == 1 ? t2 : t1;
        s1.y = start == 0 ? t1 : start == 1 ? t0 : t2;
        s2.y = start == 0 ? t2 : start == 1 ? t1 : t0;
    }

    const int nfloat2 = 3 * nparticles;

    if (laneid < nfloat2)
    data[laneid] = s0;

    if (laneid + 32 < nfloat2)
    data[laneid + 32] = s1;

    if (laneid + 64 < nfloat2)
    data[laneid + 64] = s2;
}
}
