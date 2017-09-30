#define __IMOD(x,y) ((x)-((x)/(y))*(y))

static __device__ uint idx(int a, int b) {
    uint r = 32 * a  + b;
    assert(r == xmad( a, 32.f, b));
    return r;
}

static __global__ void transpose(const int np, float *ff) {
    uint lane, warpid, base;
    __shared__ volatile float  smem[32][96];
    lane = threadIdx.x % warpSize;
    warpid = threadIdx.x / warpSize;

    for( uint i = ( blockIdx.x * blockDim.x + threadIdx.x ) & 0xFFFFFFE0U; i < np; i += blockDim.x * gridDim.x ) {
        base = 3*i + lane;
        smem[warpid][lane   ] =   ff[ base      ];
        smem[warpid][lane + 32] = ff[ base + 32 ];
        smem[warpid][lane + 64] = ff[ base + 64 ];
        ff[ base      ] = smem[warpid][idx(__IMOD( lane + 0,  3), (lane + 0 )/3)];
        ff[ base + 32 ] = smem[warpid][idx(__IMOD( lane + 32, 3), (lane + 32)/3)];
        ff[ base + 64 ] = smem[warpid][idx(__IMOD( lane + 64, 3), (lane + 64)/3)];
    }
}
#undef __IMOD
