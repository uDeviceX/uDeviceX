static __device__ float sqdist(float x, float y, float z,   float x0, float y0, float z0) {
    x -= x0; y -= y0; z -= z0;
    return x*x + y*y + z*z;
}

static __device__ void merged1(uint dststart, uint lastdst, uint nsrc, uint spidext,
                               uint tid, uint pshare) {
    float xs, ys, zs;
    float xd, yd, zd;
    float d2;
    uint p, spid, pid, dpid, nb = 0;
    for (p = 0; p < nsrc; p += 32) {
        pid = p + tid;
        spid = asmb::id(pid, nsrc, tid, pshare);
        cloud_pos(xmin(spid, lastdst), &xs, &ys, &zs);
        for (dpid = dststart; dpid < lastdst; dpid++) {
            cloud_pos(dpid, /**/ &xd, &yd, &zd);
            d2 = sqdist(xd, yd, zd,   xs, ys, zs);
            asmb::inc(d2, spid, dpid, dststart, lastdst, pshare, /*io*/ &nb);
            if (nb >= 32u) {
                core(dststart, pshare, tid, spidext );
                nb = xsub( nb, 32u );
                asmb::write(tid, pshare);
            }
        }
    }
    if (tid < nb) {
        core(dststart, pshare, tid, spidext);
    }
}

static __device__ void merged2(uint tid, uint pshare) {
    uint dststart, lastdst, nsrc, spidext;
    uint x13, y13, y14;
    asm volatile( "ld.volatile.shared.v2.u32 {%0,%1}, [%3+104];" // 104 = 13 x 8-byte uint2
                  "ld.volatile.shared.u32     %2,     [%3+116];" // 116 = 14 x 8-bute uint2 + .y
                  : "=r"(x13), "=r"(y13), "=r"(y14) : "r"(pshare) : "memory" );
    dststart = x13;
    lastdst  = xsub(xadd(dststart, y14), y13);
    nsrc     = y14;
    spidext  = x13;
    merged1(dststart, lastdst, nsrc, spidext, tid, pshare);
}

static __device__ void merged3(uint mystart, uint mycount, uint tid, uint pshare) {
    uint myscan;
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::
                  "r"(xmad(tid, 8.f, pshare)),
                  "r"(mystart) :
                  "memory");
    myscan  = mycount;
    asmb::scan(&myscan);
    asm volatile("{    .reg .pred lt15;"
                 "      setp.lt.f32 lt15, %0, %1;"
                 "@lt15 st.volatile.shared.u32 [%2+4], %3;"
                 "}":: "f"(u2f(tid)), "f"(u2f(15u)), "r"(xmad(tid, 8.f, pshare)), "r"(xsub(myscan, mycount)) : "memory");
    merged2(tid, pshare);
}

static __device__ void merged4(int cid, uint tid, uint pshare) {
    uint mystart, mycount;
    asmb::c2loc(cid, tid, /**/ &mystart, &mycount);
    merged3(mystart, mycount, tid, pshare);
}

static __device__ void merged5(uint it, int cbase, uint tid, uint pshare) {
    int cid;
    cid = asmb::get_cid(it, cbase);
    merged4(cid, tid, pshare);
}

static __global__ void merged() {
    uint tid, wid, pshare, it;
    int cbase;
    char4 offs;
    int3 nc;
    asm volatile( ".shared .u32 smem[512];" ::: "memory" );
    tid = threadIdx.x;
    wid = threadIdx.y;
    pshare = 256*wid;
    offs = tid2ind[tid];
    nc   = info.ncells;
    cbase = blockIdx.z * MYCPBZ * nc.x * nc.y + blockIdx.y * MYCPBY * nc.x + blockIdx.x * MYCPBX + wid +
            offs.z *              nc.x * nc.y + offs.y *              nc.x + offs.x;
    for (it = 0; it < 4; it++)
        merged5(it, cbase, tid, pshare);
}
