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

