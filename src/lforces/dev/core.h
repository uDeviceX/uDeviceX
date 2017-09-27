static __device__ void core( const uint dststart, const uint pshare, const uint tid, const uint spidext )
{
    float fx, fy, fz;
    forces::Fo f;
    forces::f32f(&fx, &fy, &fz, /**/ &f);

    float3 f0;

    uint item;
    uint offset = xmad( tid, 4.f, pshare );
    asm volatile( "ld.volatile.shared.u32 %0, [%1+1024];" : "=r"( item ) : "r"( offset ) : "memory" );
    uint2 pid = __unpack_8_24( item );
    uint dpid = xadd( dststart, pid.x );
    uint spid = pid.y;

    f0 = dpd(dpid, spid, /**/ f);

    // the overhead of transposition acc back
    // can be completely killed by changing the integration kernel
    uint off  = dpid & 0x0000001FU;
    uint base = xdiv( dpid, 1 / 32.f );
    float* acc = info.ff + xmad( base, 96.f, off );
    atomicAdd( acc     , f0.x );
    atomicAdd( acc + 32, f0.y );
    atomicAdd( acc + 64, f0.z );

    if( spid < spidext ) {
        uint off  = spid & 0x0000001FU;
        uint base = xdiv( spid, 1 / 32.f );
        float* acc = info.ff + xmad( base, 96.f, off );
        atomicAdd( acc     , -f0.x );
        atomicAdd( acc + 32, -f0.y );
        atomicAdd( acc + 64, -f0.z );
    }
}
