__global__ void inif3(int n, float3 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float3 r, v;
    r = make_float3(0.f, 1.f, 2.f);
    v = make_float3(3.f, 4.f, 5.f);
    
    pp[2*i + 0] = r;
    pp[2*i + 1] = v;
}

__global__ void updf3(int n, float3 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float3 r, v;
    r = pp[2*i + 0];
    v = pp[2*i + 1];

    r.x += dt * v.x;
    r.y += dt * v.y;
    r.z += dt * v.z;
        
    pp[2*i+0] = r;
    pp[2*i+1] = v;
}


__global__ void inif3_2tpp(int n, float3 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int slot = i % 2;
    int pid  = i / 2;
    if (pid >= n) return;
    
    float3 rv;
    
    rv = position(slot) ?
        make_float3(0.f, 1.f, 2.f):
        make_float3(3.f, 4.f, 5.f);
    
    pp[i] = rv;
}

__device__ float3 shfl_down_v(float3 v) {
    float3 a;
    a.x = __shfl_down(v.x, 1);
    a.y = __shfl_down(v.y, 1);
    a.z = __shfl_down(v.z, 1);
    return a;
}

__global__ void updf3_2tpp(int n, float3 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int slot = i % 2;
    int pid  = i / 2;
    if (pid >= n) return;
    
    float3 rv, vr;

    rv = pp[i];
    vr = shfl_down_v(rv);

    rv.x += dt * vr.x;
    rv.y += dt * vr.y;
    rv.z += dt * vr.z;

    if (position(slot))
        pp[i] = rv;
}

