__global__ void inif2(int n, float2 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float2 s0, s1, s2;
    s0 = make_float2(0.f, 1.f);
    s1 = make_float2(2.f, 3.f);
    s2 = make_float2(4.f, 5.f);
    
    pp[3*i + 0] = s0;
    pp[3*i + 1] = s1;
    pp[3*i + 2] = s2;
}

__global__ void updf2(int n, float2 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float2 s0, s1, s2;
    s0 = pp[3*i + 0];
    s1 = pp[3*i + 1];
    s2 = pp[3*i + 2];

    s0.x += dt * s1.y;
    s0.y += dt * s2.x;
    s1.x += dt * s2.y;
        
    pp[3*i+0] = s0;
    pp[3*i+1] = s1;
    pp[3*i+2] = s2;
}


// __global__ void inif2_3tpp(int n, float3 *pp) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int slot = i % 2;
//     int pid  = i / 2;
//     if (pid >= n) return;
    
//     float3 rv;
    
//     rv = position(slot) ?
//         make_float3(0.f, 1.f, 2.f):
//         make_float3(3.f, 4.f, 5.f);
    
//     pp[i] = rv;
// }

// __device__ float3 shfl_down_v(float3 v) {
//     float3 a;
//     a.x = __shfl_down(v.x, 1);
//     a.y = __shfl_down(v.y, 1);
//     a.z = __shfl_down(v.z, 1);
//     return a;
// }

// __global__ void updf3_2tpp(int n, float3 *pp) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int slot = i % 2;
//     int pid  = i / 2;
//     if (pid >= n) return;
    
//     float3 rv, vr;

//     rv = pp[i];
//     vr = shfl_down_v(rv);

//     rv.x += dt * vr.x;
//     rv.y += dt * vr.y;
//     rv.z += dt * vr.z;

//     if (position(slot))
//         pp[i] = rv;
// }

