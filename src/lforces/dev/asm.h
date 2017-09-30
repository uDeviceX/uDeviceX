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
