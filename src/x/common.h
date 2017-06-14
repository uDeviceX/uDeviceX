/* part of common from master */

/* size of array of pointers to every element in array (used to allocate
   pointer-to-pointer on device) */
#define SZ_PTR_ARR(A) (   sizeof(&A[0])*sizeof(A)/sizeof(A[0])   )
