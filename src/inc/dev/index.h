namespace k_index {
template<bool project>
__global__  void local(const int nparticles, const float2 * particles, int * const partials,
                       uchar4 * const subindices)
{
    const int lane = threadIdx.x % warpSize;
    const int warpid = threadIdx.x / warpSize;
    const int base = 32 * (warpid + 4 * blockIdx.x);
    const int nsrc = min(32, nparticles - base);

    if (nsrc == 0)
    return;

    int cid = -1;

    //LOAD PARTICLES, COMPUTE CELL ID
    {
        float2 data0, data1, data2;

        k_read::AOS6f(particles + 3 * base, nsrc, data0, data1, data2);

        const bool inside = project ||
            (data0.x >= -XS / 2 && data0.x < XS / 2 &&
             data0.y >= -YS / 2 && data0.y < YS / 2 &&
             data1.x >= -ZS / 2 && data1.x < ZS / 2 );

        if (lane < nsrc && inside)
        {
            if (project)
            {
                const int xcid = min(XS - 1, max(0, (int)floor((double)data0.x + XS / 2)));
                const int ycid = min(YS - 1, max(0, (int)floor((double)data0.y + YS / 2)));
                const int zcid = min(ZS - 1, max(0, (int)floor((double)data1.x + ZS / 2)));

                cid = xcid + XS * (ycid + YS * zcid);
            }
            else
            {
                const int xcid = (int)floor((double)data0.x + XS / 2);
                const int ycid = (int)floor((double)data0.y + YS / 2);
                const int zcid = (int)floor((double)data1.x + ZS / 2);

                cid = xcid + XS * (ycid + YS * zcid);
            }
        }
    }

    int pid = lane + base;

    //BITONIC SORT
    {
#pragma unroll
        for(int D = 1; D <= 16; D <<= 1)
#pragma unroll
        for(int L = D; L >= 1; L >>= 1)
        {
            const int mask = L == D ? 2 * D - 1 : L;

            const int othercid = __shfl_xor(cid, mask);
            const int otherpid = __shfl_xor(pid, mask);

            const bool exchange =  (2 * (int)(lane < (lane ^ mask)) - 1) * (cid - othercid) > 0;

            if (exchange)
            {
                cid = othercid;
                pid = otherpid;
            }
        }
    }

    int start, pcount;

    //FIND PARTICLES SHARING SAME CELL IDS
    {
        __shared__ volatile int keys[4][32];

        keys[warpid][lane] = cid;

        const bool ishead = cid != __shfl(cid, lane - 1) || lane == 0;

        if (cid >= 0)
        {
            const int searchval = ishead ? cid + 1 : cid;

            int first = ishead ? lane : 0;
            int last = ishead ? 32 : (lane + 1);
            int count = last - first;

            while (count > 0)
            {
                const int step = count / 2;
                const int it = first + step;

                if (keys[warpid][it] < searchval)
                {
                    first = it + 1;
                    count -= step + 1;
                }
                else
                count = step;
            }

            start = ishead ? lane : first;

            if (ishead)
            pcount = first - lane;
        }
    }

    //ADD COUNT TO PARTIALS, WRITE SUBINDEX
    {
        int globalstart;

        if (cid >= 0 && lane == start)
        globalstart = atomicAdd(partials + cid, pcount);

        const int subindex = __shfl(globalstart, start) + (lane - start);

        uchar4 entry = make_uchar4(0xff, 0xff, 0xff, 0xff);

        if (cid >= 0)
        {
            const int xcid = cid % XS;
            const int ycid = (cid / XS) % YS;
            const int zcid = cid / (XS * YS);

            entry = make_uchar4(xcid, ycid, zcid, subindex);
        }

        if (pid < nparticles)
        subindices[pid] = entry;
    }
}
} /* namespace */

