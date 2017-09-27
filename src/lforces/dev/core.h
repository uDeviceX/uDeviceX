static __device__ void core0(uint dpid, uint spid, uint spidext) {
    float fx, fy, fz;
    forces::Fo f;
    forces::f32f(&fx, &fy, &fz, /**/ &f);
    dpd(dpid, spid, /**/ f);
    uint off  = dpid & 0x0000001FU;
    uint base = xdiv( dpid, 1 / 32.f );
    float* acc = info.ff + xmad( base, 96.f, off );
    atomicAdd( acc     , fx );
    atomicAdd( acc + 32, fy );
    atomicAdd( acc + 64, fz );

    if( spid < spidext ) {
        uint off  = spid & 0x0000001FU;
        uint base = xdiv( spid, 1 / 32.f );
        float* acc = info.ff + xmad( base, 96.f, off );
        atomicAdd( acc     , -fx );
        atomicAdd( acc + 32, -fy );
        atomicAdd( acc + 64, -fz );
    }
}

static __device__ void core(const uint dststart, const uint pshare, const uint tid, const uint spidext) {
    uint item;
    uint offset = xmad( tid, 4.f, pshare );
    asm volatile( "ld.volatile.shared.u32 %0, [%1+1024];" : "=r"( item ) : "r"( offset ) : "memory" );
    uint2 pid = __unpack_8_24( item );
    uint dpid = xadd( dststart, pid.x );
    uint spid = pid.y;
    core0(dpid, spid, spidext);
}

