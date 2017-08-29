/*
  hello world : add two vectors

 */

#include <stdio.h>
#include "u.h"

/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less
   than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

const long n = 1<<15; /* number of elements */
char *A, *C;
int  *B;

__global__ void f (int s, char *A, int *B, char *C) {
    int i, cnt;
    A[s + 2] = ((char*)B)[s + 0];
    printf("A[2] = %c\n", A[2]);
}

__global__ void func (char* stringInput, int stringSize, int* integerInput, char* dummySpace) {
    int counter = 0;
    for (int i=0;i<stringSize;i++)
        dummySpace[counter++] = stringInput[i];
 
    for (int i=0;i<sizeof(int);i++)
        dummySpace[counter++] = ((char*)integerInput)[i];
}

void ini() {
    cudaMalloc(&A, n*sizeof(A[0]));
    cudaMalloc(&B, n*sizeof(B[0]));
    cudaMalloc(&C, n*sizeof(C[0]));    
}

union Part {
    struct {float2 d0, d1, d2; };
    struct {float  r[3], v[3]; };
};

int main() {
  ini();
  // f<<<k_cnf(n)>>>(A, B, C);
  // f<<<1,1>>>(5, A, B, C);

  func <<<k_cnf(100)>>>(A, 15001, B, C);
  CC(cudaDeviceSynchronize());

  printf("Size of union = %d\n", sizeof(Part));
  printf("Size of float2 = %d\n", sizeof(float2));

  Part p;
  p.d0 = make_float2(0, 1);
  p.d1 = make_float2(2, 3);
  p.d2 = make_float2(4, 5);

  printf("Position = %g %g %g\n", p.r[0], p.r[1], p.r[2]);
  printf("Velocity = %g %g %g\n", p.v[0], p.v[1], p.v[2]);

  p.r[0] = 6;

  printf("Position = %g %g %g\n", p.d0.x, p.d0.y, p.d1.x);
  
}
