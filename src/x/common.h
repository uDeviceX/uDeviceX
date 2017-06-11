/* part of common from master */

/* size of array of pointers to every element in array (used to allocate
   pointer-to-pointer on device) */
#define SZ_PTR_ARR(A) (   sizeof(&A[0])*sizeof(A)/sizeof(A[0])   )

/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less
   than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )
