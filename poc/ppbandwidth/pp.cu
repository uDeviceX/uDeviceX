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

#include "vanilla.h"
#include "float4.h"
#include "P4.h"

void print_bw(const char *fun, float t, size_t nbytes, int neval, float rw) {
    double tav = t / neval;
    double bw = (nbytes * rw / tav) * 1e-6;
    printf("%20s : t = %6e [ms], %6e [GB/s]\n", fun, tav, bw);
}

#define ESC(...) __VA_ARGS__
#define measure(F, C, A, nbytes, rw) do {                               \
        CC(cudaEventRecord(start));                                     \
        for (int i = 0; i < ntrials; ++i) F <<<ESC C>>> A;              \
        CC(cudaEventRecord(stop));                                      \
        CC(cudaEventSynchronize(stop));                                 \
        CC(cudaEventElapsedTime(&t, start, stop));                      \
        print_bw(#F, t, nbytes, ntrials, rw);                           \
    } while (0)

int main() {
    int n = 100000, ntrials = 10000;

    Particle *pp;
    Particle4 *pp4;

    cudaEvent_t start, stop;
    float t;

    CC(cudaSetDevice(2));
    
    CC(cudaMalloc(&pp,  n*sizeof(Particle)));
    CC(cudaMalloc(&pp4, n*sizeof(Particle4)));

    CC(cudaEventCreate(&start));
    CC(cudaEventCreate(&stop));

    float rwini = 2.0/3.0, rwupd = 1.5;
    
    measure(iniP, (k_cnf(n)), (n, pp), n*sizeof(float), rwini);
    measure(updP, (k_cnf(n)), (n, pp), n*sizeof(float), rwupd);

    measure(inif, (k_cnf(n)), (n, (float*)pp), n*sizeof(Particle), rwini);
    measure(updf, (k_cnf(n)), (n, (float*)pp), n*sizeof(Particle), rwupd);

    measure(inif4, (k_cnf(n)), (n, (float4*) pp4), 2*n*sizeof(float4), rwini);
    measure(updf4, (k_cnf(n)), (n, (float4*) pp4), 2*n*sizeof(float4), rwupd);

    measure(inif4_2tpp, (k_cnf(2*n)), (n, (float4*) pp4), 2*n*sizeof(float4), rwini);
    measure(updf4_2tpp, (k_cnf(2*n)), (n, (float4*) pp4), 2*n*sizeof(float4), rwupd);

    measure(iniP4, (k_cnf(n)), (n, pp4), n*sizeof(Particle4), rwini);
    measure(updP4, (k_cnf(n)), (n, pp4), n*sizeof(Particle4), rwupd);

    CC(cudaFree(pp));
    CC(cudaFree(pp4));
}
