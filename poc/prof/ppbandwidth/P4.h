
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
