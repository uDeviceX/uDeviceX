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
    if (i >= 2*n) return;
    
    int slot = i % 2;
    float4 rv = slot == 0 ?
        make_float4(0.f, 1.f, 2.f, -1.f) :
        make_float4(3.f, 4.f, 5.f, -1.f) ;
    
    pp[i] = rv;
}

__global__ void upd(int n, float4 *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 2*n) return;
    
    int slot = i % 2;
    float4 rv = pp[i];
    float4 up;
    up.x = __shfl_down(rv.x, 1);
    up.y = __shfl_down(rv.y, 1);
    up.z = __shfl_down(rv.z, 1);

    if (slot == 0) { // pos
        rv.x += dt * up.x;
        rv.y += dt * up.y;
        rv.z += dt * up.z;
    }
    
    pp[i] = rv;
}

int main() {
    int n = 10000, ntrials = 100;
    Particle *pp;
    float4 *pp4;

    CC(cudaMalloc(&pp,  n*sizeof(Particle)));
    CC(cudaMalloc(&pp4, n*2*sizeof(float4)));

    for (int i = 0; i < ntrials; ++i) {
        ini <<<k_cnf(n)>>> (n, pp);
        upd <<<k_cnf(n)>>> (n, pp);

        ini <<<k_cnf(2*n)>>> (n, pp4);
        upd <<<k_cnf(2*n)>>> (n, pp4);
    }
    
    CC(cudaDeviceSynchronize());

    CC(cudaFree(pp));
    CC(cudaFree(pp4));
}
