static __device__ void core0(uint dpid, uint spid, uint spidext) {
    forces::FoFo f;
    dpd(dpid, spid, /**/ &f);
    uint off  = dpid & 0x0000001FU;
    uint base = xdiv(dpid, 1 / 32.f);
    float* acc = info.ff + xmad(base, 96.f, off);
    atomicAdd(acc     , f.x);
    atomicAdd(acc + 32, f.y);
    atomicAdd(acc + 64, f.z);

    if (spid < spidext) {
        uint off  = spid & 0x0000001FU;
        uint base = xdiv(spid, 1 / 32.f);
        float* acc = info.ff + xmad(base, 96.f, off);
        atomicAdd(acc     , -f.x);
        atomicAdd(acc + 32, -f.y);
        atomicAdd(acc + 64, -f.z);
    }
}

static __device__ void core(const uint dststart, const uint pshare, const uint tid, const uint spidext) {
    uint item, offset, dpid, spid;
    uint2 pid;
    offset = xmad( tid, 4.f, pshare );
    asm volatile( "ld.volatile.shared.u32 %0, [%1+1024];" : "=r"( item ) : "r"( offset ) : "memory" );
    pid = __unpack_8_24( item );
    dpid = xadd( dststart, pid.x );
    spid = pid.y;
    core0(dpid, spid, spidext);
}
