static __device__ void core( const uint dststart, const uint pshare, const uint tid, const uint spidext )
{
    float4 xdest, xsrc, udest, usrc;
    float3 f;

    uint item;
    uint offset = xmad( tid, 4.f, pshare );
    asm volatile( "ld.volatile.shared.u32 %0, [%1+1024];" : "=r"( item ) : "r"( offset ) : "memory" );
    uint2 pid = __unpack_8_24( item );
    uint dpid = xadd( dststart, pid.x );
    uint spid = pid.y;

    uint dentry = xscale( dpid, 2.f );
    uint sentry = xscale( spid, 2.f );

    xdest = fetchF4(dentry);
    xsrc  = fetchF4(sentry);
    udest = fetchF4(xadd(dentry, 1u));
    usrc  = fetchF4(xadd(sentry, 1u));

    f = dpd( dpid, xdest, udest, xsrc, usrc, spid );

    // the overhead of transposition acc back
    // can be completely killed by changing the integration kernel
    uint off  = dpid & 0x0000001FU;
    uint base = xdiv( dpid, 1 / 32.f );
    float* acc = info.ff + xmad( base, 96.f, off );
    atomicAdd( acc   , f.x );
    atomicAdd( acc + 32, f.y );
    atomicAdd( acc + 64, f.z );

    if( spid < spidext ) {
        uint off  = spid & 0x0000001FU;
        uint base = xdiv( spid, 1 / 32.f );
        float* acc = info.ff + xmad( base, 96.f, off );
        atomicAdd( acc   , -f.x );
        atomicAdd( acc + 32, -f.y );
        atomicAdd( acc + 64, -f.z );
    }
}
