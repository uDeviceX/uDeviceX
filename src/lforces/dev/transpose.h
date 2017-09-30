#define __IMOD(x,y) ((x)-((x)/(y))*(y))
static __global__ void transpose(const int np, float *ff) {
    __shared__ volatile float  smem[32][96];
    const uint lane = threadIdx.x % warpSize;
    const uint warpid = threadIdx.x / warpSize;

    for( uint i = ( blockIdx.x * blockDim.x + threadIdx.x ) & 0xFFFFFFE0U; i < np; i += blockDim.x * gridDim.x ) {
        const uint base = xmad( i, 3.f, lane );
        smem[warpid][lane   ] = ff[ base      ];
        smem[warpid][lane + 32] = ff[ base + 32 ];
        smem[warpid][lane + 64] = ff[ base + 64 ];
        ff[ base      ] = smem[warpid][ xmad( __IMOD( lane + 0, 3 ), 32.f, ( lane + 0 ) / 3 ) ];
        ff[ base + 32 ] = smem[warpid][ xmad( __IMOD( lane + 32, 3 ), 32.f, ( lane + 32 ) / 3 ) ];
        ff[ base + 64 ] = smem[warpid][ xmad( __IMOD( lane + 64, 3 ), 32.f, ( lane + 64 ) / 3 ) ];
    }
}
#undef __IMOD
