static __global__ void tex( uint2 *start_and_count, const int *start, const int *count, const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x ) {
        start_and_count[i] = make_uint2( start[i], count[i] );
    }
}
