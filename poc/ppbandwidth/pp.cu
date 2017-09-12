#include <stdio.h>

#include "u.h" /* utils */

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )
#define k_cnf(n) ceiln((n), 128), 128

#define dt (1e-1)

struct Particle {
    float r[3], v[3];
};


__global__ void ini(int n, Particle *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    Particle p = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    pp[i] = p;
}

__global__ void upd(int n, Particle *pp) {
    enum {X, Y, Z};
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    Particle p = pp[i];
    p.r[X] += dt * p.v[X];
    p.r[Y] += dt * p.v[Y];
    p.r[Z] += dt * p.v[Z];
    pp[i] = p;
}


__global__ void ini(int n, float4 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float4 r, v;
    r = make_float4(0.f, 1.f, 2.f, -1.f);
    v = make_float4(3.f, 4.f, 5.f, -1.f);
    
    pp[2*i + 0] = r;
    pp[2*i + 1] = v;
}

__global__ void upd(int n, float4 *pp) {
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

__global__ void ini_2tpp(int n, float4 *pp) {
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

__global__ void upd_2tpp(int n, float4 *pp) {
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


int main() {
    int n = 10000, ntrials = 1000;
    Particle *pp;
    float4 *pp4;

    CC(cudaMalloc(&pp,  n*sizeof(Particle)));
    CC(cudaMalloc(&pp4, n*2*sizeof(float4)));

    for (int i = 0; i < ntrials; ++i) {
        ini <<<k_cnf(n)>>> (n, pp);
        upd <<<k_cnf(n)>>> (n, pp);

        ini <<<k_cnf(n)>>> (n, pp4);
        upd <<<k_cnf(n)>>> (n, pp4);

        ini_2tpp <<<k_cnf(2*n)>>> (n, pp4);
        upd_2tpp <<<k_cnf(2*n)>>> (n, pp4);
    }
    
    CC(cudaDeviceSynchronize());

    CC(cudaFree(pp));
    CC(cudaFree(pp4));
}
