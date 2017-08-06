__device__ void f2tof3(float4 r, /**/ float3 *l) { /* lhs = rhs */
    l->x = r.x; l->y = r.y; l->z = r.z;
}

__inline__ __device__ uint __lanemask_lt()
{
    uint mask;
    asm( "mov.u32 %0, %lanemask_lt;" : "=r"( mask ) );
    return mask;
}

__inline__ __device__ uint __pack_8_24( uint a, uint b )
{
    uint d;
    asm( "bfi.b32  %0, %1, %2, 24, 8;" : "=r"( d ) : "r"( a ), "r"( b ) );
    return d;
}

__inline__ __device__ uint2 __unpack_8_24( uint d )
{
    uint a;
    asm( "bfe.u32  %0, %1, 24, 8;" : "=r"( a ) : "r"( d ) );
    return make_uint2( a, d & 0x00FFFFFFU );
}
