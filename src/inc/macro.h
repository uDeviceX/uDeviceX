/* wired constants from wall.impl */
enum {
    XSIZE_WALLCELLS = 2 * XWM + XS,
    YSIZE_WALLCELLS = 2 * YWM + YS,
    ZSIZE_WALLCELLS = 2 * ZWM + ZS,
    XE = XS + 2*XWM, YE = YS + 2*YWM, ZE = ZS + 2*ZWM,
    XTE  = 16*16,/* texture sizes */
    _YTE = ceiln(YE*XTE, XE), YTE  = 16*ceiln(_YTE, 16),
    _ZTE = ceiln(ZE*XTE, XE), ZTE  = 16*ceiln(_ZTE, 16)
};

enum {  /* used in sorting of bulk particle when wall is created */
    W_BULK,  /* remains in bulk */
    W_WALL,  /* becomes wall particle */
    W_DEEP   /* deep inside the wall */
};

/* size of array of pointers to every element in array (used to allocate
   pointer-to-pointer on device) */
#define SZ_PTR_ARR(A) (   sizeof(&A[0])*sizeof(A)/sizeof(A[0])   )

/*
  #define allsync() do {                                                  \
  CC(cudaDeviceSynchronize()); MC(l::m::Barrier(m::cart));          \
  if (m::rank == 0)                                               \
  fprintf(stderr, "%s : %d\n", __FILE__, __LINE__);               \
  } while (0)
*/
