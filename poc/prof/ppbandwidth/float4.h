__global__ void inif4(int n, float4 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float4 r, v;
    r = make_float4(0.f, 1.f, 2.f, -1.f);
    v = make_float4(3.f, 4.f, 5.f, -1.f);
    
    pp[2*i + 0] = r;
    pp[2*i + 1] = v;
}

__global__ void updf4(int n, float4 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float4 r, v;
    r = pp[2*i + 0];
    v = pp[2*i + 1];

    r.x += dt * v.x;
    r.y += dt * v.y;
    r.z += dt * v.z;
        
    pp[2*i+0] = r;
    pp[2*i+1] = v;
}

__device__ bool position(int slot) {return slot == 0;}

__global__ void inif4_2tpp(int n, float4 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int slot = i % 2;
    int pid  = i / 2;
    if (pid >= n) return;
    
    float4 rv;
    
    rv = position(slot) ?
        make_float4(0.f, 1.f, 2.f, -1.f) :
        make_float4(3.f, 4.f, 5.f, -1.f) ;
    
    pp[i] = rv;
}

__device__ float4 shfl_down_v(float4 v) {
    float4 a;
    a.x = __shfl_down(v.x, 1);
    a.y = __shfl_down(v.y, 1);
    a.z = __shfl_down(v.z, 1);
    return a;
}

__global__ void updf4_2tpp(int n, float4 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int slot = i % 2;
    int pid  = i / 2;
    if (pid >= n) return;
    
    float4 rv, vr;

    rv = pp[i];
    vr = shfl_down_v(rv);

    rv.x += dt * vr.x;
    rv.y += dt * vr.y;
    rv.z += dt * vr.z;

    if (position(slot))
        pp[i] = rv;
}

