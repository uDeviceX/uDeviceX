/*
  hello world : add two vectors

 */

#include <stdio.h>

/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less
   than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

#define n 5 /* number of elements */
int  *A;
char *B;

/*
__global__ void func (char* stringInput, int stringSize, int* integerInput, char* dummySpace) {
    int counter = 0;
    for(int i=0;i<stringSize;i++)
       dummySpace[counter++] = stringInput[i];
    for(int i=0;i<sizeof(int);i++)
       dummySpace[counter++] = ((char*)integerInput)[i];
    }
}
*/

void ini() {
    cudaMalloc(&A, n*sizeof(A[0]));
    cudaMalloc(&B, n*sizeof(B[0]));
}

int main() {
  ini();
  //  misaligned<<<k_cnf(n)>>>(d_A, d_B, d_C);
}
