#include <stdio.h>

#include "u.h" /* utils */

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )
#define k_cnf(n) ceiln((n), 128), 128

#define dt (1e-1)

struct Particle {
    float r[3], v[3];
};

__global__ void upd(int n, Particle *pp) {
    enum {X, Y, Z};
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p = pp[i];
    p.r[X] += dt * p.v[X];
    p.r[Y] += dt * p.v[Y];
    p.r[Z] += dt * p.v[Z];
    pp[i] = p;
}

__global__ void ini(int n, Particle *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    pp[i] = p;
}

int main() {
    int n = 10000;
    Particle *pp;

    CC(cudaMalloc(&pp, n*sizeof(Particle)));

    ini <<<k_cnf(n)>>> (n, pp);
    upd <<<k_cnf(n)>>> (n, pp);
    
    CC(cudaDeviceSynchronize());
}
