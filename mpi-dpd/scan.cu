template <int NWARP>
__global__ void breduce(uint4 *vin, unsigned int *vout, int n) {

    const int wid = threadIdx.x/32;
    const int lid = threadIdx.x%32;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    uint4 val = make_uint4(0,0,0,0);

    __shared__ unsigned int shtmp[NWARP];

    if (tid < n) val = vin[tid];

    val.x = ((val.x & 0xFF000000) >> 24) + ((val.x & 0xFF0000) >> 16) + ((val.x & 0xFF00) >> 8) + (val.x & 0xFF);
    val.y = ((val.y & 0xFF000000) >> 24) + ((val.y & 0xFF0000) >> 16) + ((val.y & 0xFF00) >> 8) + (val.y & 0xFF);
    val.z = ((val.z & 0xFF000000) >> 24) + ((val.z & 0xFF0000) >> 16) + ((val.z & 0xFF00) >> 8) + (val.z & 0xFF);
    val.w = ((val.w & 0xFF000000) >> 24) + ((val.w & 0xFF0000) >> 16) + ((val.w & 0xFF00) >> 8) + (val.w & 0xFF);
    val.x += val.y + val.z + val.w;

#pragma unroll
    for(int i = 16; i > 0; i >>= 1)
        val.x += __shfl_down((int)val.x, i);

    if (0 == lid)
        shtmp[wid] = val.x;

    __syncthreads();
    if (0 == wid) {
        val.x = (lid < NWARP) ? shtmp[lid] : 0;

#pragma unroll
        for(int i = 16; i > 0; i >>= 1)
            val.x += __shfl_down((int)val.x, i);
    }
    if (0 == threadIdx.x) vout[blockIdx.x] = val.x;
    return;
}

template <int BDIM>
__global__ void bexscan(unsigned int *v, int n) {

    extern __shared__ unsigned int shtmp[];

    for(int i = threadIdx.x; i < n; i += BDIM) shtmp[i] = v[i];

    int i = threadIdx.x;
    __syncthreads();
    for(; n >= BDIM; i += BDIM, n -= BDIM) {

        __syncthreads();
        if (i > 0 && 0 == threadIdx.x)
            shtmp[i] += shtmp[i-1];

        unsigned int a = 0;

#pragma unroll
        for(int j = 1; j < BDIM; j <<= 1) {
            a = 0;

            __syncthreads();
            if (threadIdx.x >= j) a = shtmp[i] + shtmp[i-j];

            __syncthreads();
            if (threadIdx.x >= j) shtmp[i] = a;
        }
        v[i] = shtmp[i];
    }
    if (threadIdx.x < n) {

        __syncthreads();
        if (i > 0 && 0 == threadIdx.x)
            shtmp[i] += shtmp[i-1];

        unsigned int a = 0;
        for(int j = 1; j < BDIM; j <<= 1) {
            a = 0;

            __syncthreads();
            if (threadIdx.x >= j) a = shtmp[i] + shtmp[i-j];

            __syncthreads();
            if (threadIdx.x >= j) shtmp[i] = a;
        }
        v[i] = shtmp[i];
    }
    return;
}

template <int NWARP>
__global__ void gexscan(uint4 *vin, unsigned int *offs, uint4 *vout, int n) {

    const int wid = threadIdx.x/32;
    const int lid = threadIdx.x%32;
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    uint4 val[4];

    __shared__ unsigned int woff[NWARP];

    if (tid < n) val[0] = vin[tid];
    else         val[0] = make_uint4(0,0,0,0);

    // read bytes B0 B1 B2 B3 into VAL[*].y VAL[*].z VAL[*].w VAL[*].x
    val[3].y = (val[0].w & 0x000000FF);
    val[3].z = (val[0].w & 0x0000FF00) >>  8;
    val[3].w = (val[0].w & 0x00FF0000) >> 16;
    val[3].x = (val[0].w & 0xFF000000) >> 24;

    val[2].y = (val[0].z & 0x000000FF);
    val[2].z = (val[0].z & 0x0000FF00) >>  8;
    val[2].w = (val[0].z & 0x00FF0000) >> 16;
    val[2].x = (val[0].z & 0xFF000000) >> 24;

    val[1].y = (val[0].y & 0x000000FF);
    val[1].z = (val[0].y & 0x0000FF00) >>  8;
    val[1].w = (val[0].y & 0x00FF0000) >> 16;
    val[1].x = (val[0].y & 0xFF000000) >> 24;

    val[0].y = (val[0].x & 0x000000FF);
    val[0].z = (val[0].x & 0x0000FF00) >>  8;
    val[0].w = (val[0].x & 0x00FF0000) >> 16;
    val[0].x = (val[0].x & 0xFF000000) >> 24;

    uint4 tu4;
    tu4.x = val[0].x + val[0].y + val[0].z + val[0].w;
    tu4.y = val[1].x + val[1].y + val[1].z + val[1].w;
    tu4.z = val[2].x + val[2].y + val[2].z + val[2].w;
    tu4.w = val[3].x + val[3].y + val[3].z + val[3].w;

    unsigned int tmp = tu4.x + tu4.y + tu4.z + tu4.w;

    tu4.w = tmp;
#pragma unroll
    for(int i = 1; i < 32; i <<= 1)
        tu4.w += (lid >= i)*__shfl_up((int)tu4.w, i);

    if (lid == 31)
        if (wid < NWARP-1) woff[wid+1] = tu4.w;
        else               woff[0]     = (blockIdx.x > 0) ? offs[blockIdx.x-1] : 0;

    tu4.w -= tmp;

    __syncthreads();
    if (0 == wid) {
        tmp = (lid < NWARP) ? woff[lid] : 0;
#pragma unroll
        for(int i = 1; i < NWARP; i <<= 1)
            tmp += (lid >= i)*__shfl_up((int)tmp, i);

        if (lid < NWARP) woff[lid] = tmp;
    }
    __syncthreads();

    if (tid >= n) return;

    uint4 lps;
    lps.x = woff[wid] + tu4.w;
    lps.y = lps.x + tu4.x;
    lps.z = lps.y + tu4.y;
    lps.w = lps.z + tu4.z;

    val[0].x = lps.x;
    val[1].x = lps.y;
    val[2].x = lps.z;
    val[3].x = lps.w;

    val[0].y += val[0].x;
    val[1].y += val[1].x;
    val[2].y += val[2].x;
    val[3].y += val[3].x;

    val[0].z += val[0].y;
    val[1].z += val[1].y;
    val[2].z += val[2].y;
    val[3].z += val[3].y;

    val[0].w += val[0].z;
    val[1].w += val[1].z;
    val[2].w += val[2].z;
    val[3].w += val[3].z;

    vout += tid*4;

#pragma unroll
    for(int i = 0; i < 4; i++)
        vout[i] = val[i];
    return;
}

void scan(unsigned char *input, int size, uint *output)
{
    enum { THREADS = 128 } ;

    static uint * tmp = NULL;

    if (tmp == NULL)
        cudaMalloc(&tmp, sizeof(uint) * (64 * 64 * 64 / THREADS));

    int nblocks = ((size / 16) + THREADS - 1 ) / THREADS;

    breduce< THREADS / 32 ><<<nblocks, THREADS, 0>>>((uint4 *)input, tmp, size / 16);

    bexscan< THREADS ><<<1, THREADS, nblocks*sizeof(uint)>>>(tmp, nblocks);

    gexscan< THREADS / 32 ><<<nblocks, THREADS, 0>>>((uint4 *)input, tmp, (uint4 *)output, size / 16);
}
