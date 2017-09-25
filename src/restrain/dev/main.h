/* restrain drop kernels */
static __device__ float3 sumv;
static __device__ int indrop;

static __device__ float3 warpReduceSumf3(float3 v) {
    for (int offset = warpSize>>1; offset > 0; offset >>= 1) {
        v.x += __shfl_down(v.x, offset);
        v.x += __shfl_down(v.y, offset);
        v.z += __shfl_down(v.z, offset);
    }
    return v;
}

static __device__ int warpReduceSum(int v) {
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
    return v;
}

static __global__ void sum_vel(int color, int n, const Particle *pp, const int *cc) {
    int i, valid, nvalid;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    
    float3 v = make_float3(0, 0, 0);

    valid = (i < n) && (cc[i] == color);

    if (valid) {
        enum {X, Y, Z};
        const Particle p = pp[i]; 
        v.x = p.v[X];
        v.y = p.v[Y];
        v.z = p.v[Z];
    }

    v  = warpReduceSumf3(v);
    nvalid = warpReduceSum(valid);

    if (threadIdx.x == 0 && nvalid > 0) {
        atomicAdd(&sumv.x, v.x);
        atomicAdd(&sumv.y, v.y);
        atomicAdd(&sumv.z, v.z);
        atomicAdd(&indrop, nvalid);
    }
}

static __global__ void shift_vel(float3 v, int n, Particle *pp) {
    enum {X, Y, Z};
    int i;
    i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= n) return;

    Particle p = pp[i];
    
    p.v[X] -= v.x;
    p.v[Y] -= v.y;
    p.v[Z] -= v.z;

    pp[i] = p;
}
