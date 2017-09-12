
__global__ void iniP4(int n, Particle4 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    Particle4 p;
    p.r = make_float4(0.f, 1.f, 2.f, -1.f);
    p.v = make_float4(3.f, 4.f, 5.f, -1.f);
    
    pp[i] = p;
}

__global__ void updP4(int n, Particle4 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    Particle4 p = pp[i];

    p.r.x += dt * p.v.x;
    p.r.y += dt * p.v.y;
    p.r.z += dt * p.v.z;
        
    pp[i] = p;
}

// __device__ bool position(int slot) {return slot == 0;}

// __global__ void inif4_2tpp(int n, float4 *pp) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int slot = i % 2;
//     int pid  = i / 2;
//     if (pid >= n) return;
    
//     float4 rv;
    
//     rv = position(slot) ?
//         make_float4(0.f, 1.f, 2.f, -1.f) :
//         make_float4(3.f, 4.f, 5.f, -1.f) ;
    
//     pp[i] = rv;
// }

// __device__ float4 shfl_down_v(float4 v) {
//     float4 a;
//     a.x = __shfl_down(v.x, 1);
//     a.y = __shfl_down(v.y, 1);
//     a.z = __shfl_down(v.z, 1);
//     return a;
// }

// __global__ void updf4_2tpp(int n, float4 *pp) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int slot = i % 2;
//     int pid  = i / 2;
//     if (pid >= n) return;
    
//     float4 rv, vr;

//     rv = pp[i];
//     vr = shfl_down_v(rv);

//     rv.x += dt * vr.x;
//     rv.y += dt * vr.y;
//     rv.z += dt * vr.z;

//     if (position(slot))
//         pp[i] = rv;
// }

