static __device__ int get_cid(uint it, uint cbase) {
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
    return cid;
}

/* cell id to location? */
static __device__ void c2loc(int cid, uint tid, /**/ uint *pstart, uint *pcout) {
    uint mystart, mycount;
    mystart = mycount = 0;
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

    *pstart = mystart; *pcout  = mycount;
}

static __device__ void scan(uint *pscan) {
    uint myscan;
    myscan = *pscan;
    asm("{ .reg .pred   p;"
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
        "}" : "+r"(myscan));
    *pscan = myscan;
}

/* source particle id? */
static __device__ uint id(uint pid, uint nsrc, uint tid, uint pshare) {
    uint spid;
    asm volatile("{ .reg .pred p, q, r;"
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
                 "}" : "=r"(spid) : "f"(u2f(pid)),    "f"(u2f(9u)),  "f"(u2f(3u)),
                                    "f"(u2f(pshare)), "f"(u2f(pid)), "f"(u2f(nsrc)));
    return spid;
}

/* increase `nb' counter? */
static __device__ void inc(float d2, uint spid, uint dpid,
                           uint dststart, uint lastdst, uint pshare,
                           /**/ uint *pnb) {
    uint nb;
    uint overview, insert;
    nb = *pnb;

    asm volatile( ".reg .pred interacting;" );
    asm( "   setp.lt.ftz.f32  interacting, %3, 1.0;"
         "   setp.ne.and.f32  interacting, %1, %2, interacting;"
         "   setp.lt.and.f32  interacting, %2, %5, interacting;"
         "   vote.ballot.b32  %0, interacting;" :
         "=r"( overview ) : "f"( u2f( dpid ) ), "f"( u2f( spid ) ), "f"( d2 ), "f"( u2f( 1u ) ), "f"( u2f( lastdst ) ) );
    insert = xadd( nb, i2u( __popc( overview & __lanemask_lt() ) ) );
    asm volatile( "@interacting st.volatile.shared.u32 [%0+1024], %1;" : :
                  "r"( xmad( insert, 4.f, pshare ) ),
                  "r"( __pack_8_24( xsub( dpid, dststart ), spid ) ) :
                  "memory" );
    nb = xadd( nb, i2u( __popc( overview ) ) );

    *pnb = nb;
}

static __device__ void write(uint tid, uint pshare) {
    asm volatile( "{ .reg .u32 tmp;"
                  "   ld.volatile.shared.u32 tmp, [%0+1024+128];"
                  "   st.volatile.shared.u32 [%0+1024], tmp;"
                  "}" :: "r"(xmad(tid, 4.f, pshare)) : "memory");
}

static __device__ void get_pair(uint dststart, uint pshare, uint tid, /**/ uint *dpid, uint *spid) {
    uint item, offset;
    uint2 pid;
    offset = xmad(tid, 4.f, pshare);
    asm volatile( "ld.volatile.shared.u32 %0, [%1+1024];" : "=r"( item ) : "r"( offset ) : "memory" );
    pid = __unpack_8_24(item);

    *dpid = xadd(dststart, pid.x);
    *spid = pid.y;
}
