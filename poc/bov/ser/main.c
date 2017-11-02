#include "bov.h"

#define NX 5
#define NY 5
#define NZ 5
#define NCOMP 1

int main() {
    /* float data[NZ][NY][NX][NCOMP]; */
    float D[NX*NY*NZ*NCOMP];
    write("main", D,
          0,   0,  0,
          NX, NY, NZ,
          NCOMP);
}
