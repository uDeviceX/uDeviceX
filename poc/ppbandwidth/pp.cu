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

int main() {
    int n = 100000, ntrials = 1000;
    Particle *pp;
    float4 *pp4;

    CC(cudaMalloc(&pp,  n*sizeof(Particle)));
    CC(cudaMalloc(&pp4, n*2*sizeof(float4)));

    for (int i = 0; i < ntrials; ++i) {
        ini <<<k_cnf(n)>>> (n, pp);
        upd <<<k_cnf(n)>>> (n, pp);

        ini <<<k_cnf(n)>>> (n, pp4);
        upd <<<k_cnf(n)>>> (n, pp4);
    }
    
    CC(cudaDeviceSynchronize());

    CC(cudaFree(pp));
    CC(cudaFree(pp4));
}
