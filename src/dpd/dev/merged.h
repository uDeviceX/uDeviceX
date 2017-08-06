__global__
void merged()
{

    asm volatile( ".shared .u32 smem[512];" ::: "memory" );

    const uint tid = threadIdx.x;
    const uint wid = threadIdx.y;
    const uint pshare = xscale( threadIdx.y, 256.f );

#if !(defined(__CUDA_ARCH__)) || __CUDA_ARCH__>= 350
    const char4 offs = __ldg( tid2ind + tid );
#else
    const char4 offs = tid2ind[tid];
#endif

    const int cbase = blockIdx.z * MYCPBZ * info.ncells.x * info.ncells.y +
        blockIdx.y * MYCPBY * info.ncells.x +
        blockIdx.x * MYCPBX + wid +
        offs.z * info.ncells.x * info.ncells.y +
        offs.y * info.ncells.x +
        offs.x;

    for( uint it = 0; it < 4 ; it = xadd( it, 1u ) ) {
        int cid;
        asm( "{  .reg .pred    p;"
             "   .reg .f32     incf;"
             "   .reg .s32     inc;"
             "    setp.lt.f32  p, %2, %3;"
             "    selp.f32     incf, %4, 0.0, p;"
             "    add.f32      incf, incf, %5;"
             "    mov.b32      inc, incf;"
             "    mul.lo.u32   inc, inc, %6;"
             "    add.s32 %0,  %1, inc;"
             "}" :
             "=r"( cid ) : "r"( cbase ), "f"( u2f( it ) ), "f"( u2f( 2u ) ), "f"( i2f( info.ncells.y ) ), "f"( u2f( ( it & 1u ) ^ ( it >> 1 ) ) ),
             "r"( info.ncells.x ) );

        uint mystart = 0, mycount = 0, myscan;
        asm( "{  .reg .pred vc;"
             "   .reg .u32  foo, bar;"
             "    setp.lt.f32     vc, %2, %3;"
             "    setp.ge.and.f32 vc, %5, 0.0, vc;"
             "    setp.lt.and.s32 vc, %4, %6, vc;"
             "    selp.s32 %0, 1, 0, vc;"
             "@vc tex.1d.v4.s32.s32 {%0, %1, foo, bar}, [texStartAndCount, %4];"
             "}" :
             "+r"( mystart ), "+r"( mycount )  :
             "f"( u2f( tid ) ), "f"( u2f( 14u ) ), "r"( cid ), "f"( i2f( cid ) ),
             "r"( info.nxyz ) );
        myscan  = mycount;
        asm volatile( "st.volatile.shared.u32 [%0], %1;" ::
                      "r"( xmad( tid, 8.f, pshare ) ),
                      "r"( mystart ) :
                      "memory" );

        asm( "{ .reg .pred   p;"
             "  .reg .f32    myscan, theirscan;"
             "   mov.b32     myscan, %0;"
             "   shfl.up.b32 theirscan|p, myscan, 0x1, 0x0;"
             "@p add.f32     myscan, theirscan, myscan;"
             "   shfl.up.b32 theirscan|p, myscan, 0x2, 0x0;"
             "@p add.f32     myscan, theirscan, myscan;"
             "   shfl.up.b32 theirscan|p, myscan, 0x4, 0x0;"
             "@p add.f32     myscan, theirscan, myscan;"
             "   shfl.up.b32 theirscan|p, myscan, 0x8, 0x0;"
             "@p add.f32     myscan, theirscan, myscan;"
             "   shfl.up.b32 theirscan|p, myscan, 0x10, 0x0;"
             "@p add.f32     myscan, theirscan, myscan;"
             "   mov.b32     %0, myscan;"
             "}" : "+r"( myscan ) );


        asm volatile( "{    .reg .pred lt15;"
                      "      setp.lt.f32 lt15, %0, %1;"
                      "@lt15 st.volatile.shared.u32 [%2+4], %3;"
                      "}":: "f"( u2f( tid ) ), "f"( u2f( 15u ) ), "r"( xmad( tid, 8.f, pshare ) ), "r"( xsub( myscan, mycount ) ) : "memory" );

        uint x13, y13, y14; // TODO: LDS.128
        asm volatile( "ld.volatile.shared.v2.u32 {%0,%1}, [%3+104];" // 104 = 13 x 8-byte uint2
                      "ld.volatile.shared.u32     %2,     [%3+116];" // 116 = 14 x 8-bute uint2 + .y
                      : "=r"( x13 ), "=r"( y13 ), "=r"( y14 ) : "r"( pshare ) : "memory" );
        const uint dststart = x13;
        const uint lastdst  = xsub( xadd( dststart, y14 ), y13 );
        const uint nsrc     = y14;
        const uint spidext  = x13;

        uint nb = 0;

        for( uint p = 0; p < nsrc; p = xadd( p, 32u ) ) {

            const uint pid = p + tid;
            uint spid;
            asm volatile( "{ .reg .pred p, q, r;" // TODO: HOW TO USE LDS.128
                          "  .reg .f32  key;"
                          "  .reg .f32  scan3, scan6, scan9;"
                          "  .reg .f32  mystart, myscan;"
                          "  .reg .s32  array;"
                          "  .reg .f32  array_f;"
                          "   mov.b32           array, %4;"
                          "   ld.shared.f32     scan9,  [array +  9*8 + 4];"
                          "   setp.ge.f32       p, %1, scan9;"
                          "   selp.f32          key, %2, 0.0, p;"
                          "   mov.b32           array_f, array;"
                          "   fma.f32.rm        array_f, key, 8.0, array_f;"
                          "   mov.b32 array,    array_f;"
                          "   ld.shared.f32     scan3, [array + 3*8 + 4];"
                          "   setp.ge.f32       p, %1, scan3;"
                          "@p add.f32           key, key, %3;"
                          "   setp.lt.f32       p, key, %2;"
                          "   setp.lt.and.f32   p, %5, %6, p;"
                          "   ld.shared.f32     scan6, [array + 6*8 + 4];"
                          "   setp.ge.and.f32   q, %1, scan6, p;"
                          "@q add.f32           key, key, %3;"
                          "   fma.f32.rm        array_f, key, 8.0, %4;"
                          "   mov.b32           array, array_f;"
                          "   ld.shared.v2.f32 {mystart, myscan}, [array];"
                          "   add.f32           mystart, mystart, %1;"
                          "   sub.f32           mystart, mystart, myscan;"
                          "   mov.b32           %0, mystart;"
                          "}" : "=r"( spid ) : "f"( u2f( pid ) ), "f"( u2f( 9u ) ), "f"( u2f( 3u ) ), "f"( u2f( pshare ) ), "f"( u2f( pid ) ), "f"( u2f( nsrc ) ) );

            const float4 xsrc = tex1Dfetch( texParticlesH4, xmin( spid, lastdst ) );

            for( uint dpid = dststart; dpid < lastdst; dpid = xadd( dpid, 1u ) ) {

                const float4 xdest = tex1Dfetch( texParticlesH4, dpid );
                const float dx = xdest.x - xsrc.x;
                const float dy = xdest.y - xsrc.y;
                const float dz = xdest.z - xsrc.z;
                const float d2 = dx * dx + dy * dy + dz * dz;

                asm volatile( ".reg .pred interacting;" );
                uint overview;
                asm( "   setp.lt.ftz.f32  interacting, %3, 1.0;"
                     "   setp.ne.and.f32  interacting, %1, %2, interacting;"
                     "   setp.lt.and.f32  interacting, %2, %5, interacting;"
                     "   vote.ballot.b32  %0, interacting;" :
                     "=r"( overview ) : "f"( u2f( dpid ) ), "f"( u2f( spid ) ), "f"( d2 ), "f"( u2f( 1u ) ), "f"( u2f( lastdst ) ) );

                const uint insert = xadd( nb, i2u( __popc( overview & __lanemask_lt() ) ) );

                asm volatile( "@interacting st.volatile.shared.u32 [%0+1024], %1;" : :
                              "r"( xmad( insert, 4.f, pshare ) ),
                              "r"( __pack_8_24( xsub( dpid, dststart ), spid ) ) :
                              "memory" );

                nb = xadd( nb, i2u( __popc( overview ) ) );
                if( nb >= 32u ) {
                    core( dststart, pshare, tid, spidext );
                    nb = xsub( nb, 32u );

                    asm volatile( "{ .reg .u32 tmp;"
                                  "   ld.volatile.shared.u32 tmp, [%0+1024+128];"
                                  "   st.volatile.shared.u32 [%0+1024], tmp;"
                                  "}" :: "r"( xmad( tid, 4.f, pshare ) ) : "memory" );
                }

            }
        }

        if( tid < nb ) {
            core( dststart, pshare, tid, spidext );
        }
        nb = 0;
    }
}
