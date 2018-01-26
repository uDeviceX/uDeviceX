#include <stdio.h>
#include <stdlib.h>

#include "common/type.h"
#include "common/dump.h"


void usg() {
    printf("usage: ./gen_random <X> <Y> <Z>  <nparticles> > file.txt\n\n");
}

void srnd() {
    srand(12345);
}

float rnd(int L) {
    return L * (-0.5f + 0.9999f * drand48());
}

void gen(int n, int LX, int LY, int LZ, Particle *pp) {
    int i;
    Particle p;
    for (i = 0; i < n; ++i) {
        p.r[0] = rnd(LX);
        p.r[1] = rnd(LY);
        p.r[2] = rnd(LZ);
        p.v[0] = rnd(2);
        p.v[1] = rnd(2);
        p.v[2] = rnd(2);
        pp[i] = p;
    }
}

int main(int argc, char **argv) {
    if (argc != 5) {
        usg();
        return 1;
    }
    int X, Y, Z, n, iarg;

    srnd();

    iarg = 1;
    X = atoi(argv[iarg++]);
    Y = atoi(argv[iarg++]);
    Z = atoi(argv[iarg++]);
    n = atoi(argv[iarg++]);

    Particle *pp = new Particle[n];

    gen(n, X, Y, Z, pp);
    write_pp(n, pp, stdout);
    
    delete[] pp;    
    return 0;
}
     
