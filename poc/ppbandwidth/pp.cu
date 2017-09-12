#include <stdio.h>

#include "u.h" /* utils */

#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )
#define k_cnf(n) ceiln((n), 128), 128

#define dt (1e-1)

struct Particle {
    float r[3], v[3];
};

struct Particle4 {
    float4 r, v;
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

#define ESC(...) __VA_ARGS__
#define measure(F, C, A, t) do {                                        \
        CC(cudaEventRecord(start));                                     \
        for (int i = 0; i < ntrials; ++i) F <<<ESC C>>> A;              \
        CC(cudaEventRecord(stop));                                      \
        CC(cudaEventSynchronize(stop));                                 \
        CC(cudaEventElapsedTime(&t, start, stop));                      \
    } while (0)

void print_bw(const char *fun, float t, size_t nbytes, int neval, int rw) {
    double tav = t / neval;
    double bw = (nbytes * rw / tav) * 1e-6;
    printf("%20s : t = %6e [ms], %6e [Gb/s]\n", fun, tav, bw);
}

int main() {
    int n = 100000, ntrials = 10000;
    Particle *pp;
    float4 *pp4;

    cudaEvent_t start, stop;
    float tiniP, tupdP, tiniP4, tupdP4, tini2P4, tupd2P4;
    tiniP = tupdP = tiniP4 = tupdP4 = tini2P4 = tupd2P4 = 0;

    CC(cudaSetDevice(2));
    
    CC(cudaMalloc(&pp,  n*sizeof(Particle)));
    CC(cudaMalloc(&pp4, n*2*sizeof(float4)));

    CC(cudaEventCreate(&start));
    CC(cudaEventCreate(&stop));

    measure(ini, (k_cnf(n)), (n, pp), /**/ tiniP);
    measure(upd, (k_cnf(n)), (n, pp), /**/ tupdP);

    measure(ini, (k_cnf(n)), (n, pp4), /**/ tiniP4); 
    measure(upd, (k_cnf(n)), (n, pp4), /**/ tupdP4);

    measure(ini_2tpp, (k_cnf(2*n)), (2*n, pp4), /**/ tini2P4);
    measure(upd_2tpp, (k_cnf(2*n)), (2*n, pp4), /**/ tupd2P4);

    print_bw("ini", tiniP, n*sizeof(Particle), ntrials, 6);
    print_bw("upd", tupdP, n*sizeof(Particle), ntrials, 9);

    print_bw("ini4", tiniP4, 2*n*sizeof(float4), ntrials, 8);
    print_bw("upd4", tupdP4, 2*n*sizeof(float4), ntrials, 12);

    print_bw("ini4 2tpp", tini2P4, 2*n*sizeof(float4), ntrials, 8);
    print_bw("upd4 2tpp", tupd2P4, 2*n*sizeof(float4), ntrials, 12);

    CC(cudaFree(pp));
    CC(cudaFree(pp4));
}
