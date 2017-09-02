static __global__ void transpose( const int np ) {
    __shared__ volatile float  smem[32][96];
    const uint lane = threadIdx.x % warpSize;
    const uint warpid = threadIdx.x / warpSize;

    for( uint i = ( blockIdx.x * blockDim.x + threadIdx.x ) & 0xFFFFFFE0U; i < np; i += blockDim.x * gridDim.x ) {
        const uint base = xmad( i, 3.f, lane );
        smem[warpid][lane   ] = info.ff[ base      ];
        smem[warpid][lane + 32] = info.ff[ base + 32 ];
        smem[warpid][lane + 64] = info.ff[ base + 64 ];
        info.ff[ base      ] = smem[warpid][ xmad( __IMOD( lane + 0, 3 ), 32.f, ( lane + 0 ) / 3 ) ];
        info.ff[ base + 32 ] = smem[warpid][ xmad( __IMOD( lane + 32, 3 ), 32.f, ( lane + 32 ) / 3 ) ];
        info.ff[ base + 64 ] = smem[warpid][ xmad( __IMOD( lane + 64, 3 ), 32.f, ( lane + 64 ) / 3 ) ];
    }
}
