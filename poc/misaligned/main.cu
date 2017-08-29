/*
  hello world : add two vectors

 */

#include <stdio.h>

/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less
   than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

#define n 32 /* number of elements */
char *A, *C;
int  *B;

__global__ void f (char *A, int *B, char *C) {
    int i, cnt;
    for (cnt = i = 0; i < n / 2; i++)
        C[cnt++] = B[i];

    for (i = 0; i < n / 2; i++)
        C[cnt++] = ((char*)B)[i];
}

void ini() {
    cudaMalloc(&A, n*sizeof(A[0]));
    cudaMalloc(&B, n*sizeof(B[0]));
    cudaMalloc(&C, n*sizeof(C[0]));    
}

int main() {
  ini();
  f<<<k_cnf(n)>>>(A, B, C);
}
