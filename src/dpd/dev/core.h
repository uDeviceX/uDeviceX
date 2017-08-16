static __device__ void core( const uint dststart, const uint pshare, const uint tid, const uint spidext )
{
    uint item;
    uint offset = xmad( tid, 4.f, pshare );
    asm volatile( "ld.volatile.shared.u32 %0, [%1+1024];" : "=r"( item ) : "r"( offset ) : "memory" );
    uint2 pid = __unpack_8_24( item );
    uint dpid = xadd( dststart, pid.x );
    uint spid = pid.y;

    uint dentry = xscale( dpid, 2.f );
    uint sentry = xscale( spid, 2.f );
    float4 xdest = tex1Dfetch( texParticlesF4,       dentry );
    float4 xsrc  = tex1Dfetch( texParticlesF4,       sentry );
    float4 udest = tex1Dfetch( texParticlesF4, xadd( dentry, 1u ) );
    float4 usrc  = tex1Dfetch( texParticlesF4, xadd( sentry, 1u ) );
    float3 f = dpd( dpid, xdest, udest, xsrc, usrc, spid );

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
