/* DPD kernel envelop parameter: random and dissipative kernels (wd = wr^2)
   0: wr = 1 - r
   1: wr = (1 - r)^(1/2)
   2: wr = (1 - r)^(1/4) */
#ifndef S_LEVEL
  #define S_LEVEL (2)
#endif

/* [k]ernel [l]aunch macros */
#if !defined(KL_RELEASE)    && !defined(KL_NONE)   &&   \
    !defined(KL_SYNC)
#define KL_RELEASE
#endif

/* [c]uda [c]heck macro */
#if !defined(CC_RELEASE) && !defined(CC_SYNC)
  #define CC_RELEASE
#endif

/* who plays as device? */
#if !defined(DEV_CUDA) && !defined(DEV_CPU)
  #define DEV_CUDA
#endif

/* compute rbc force in double or float */
#if !defined(RBC_DOUBLE) && !defined(RBC_FLOAT)
  #define RBC_DOUBLE
#endif

#ifndef DUMP_BASE
#define DUMP_BASE "."
#endif

#ifdef BASE_STRT_DUMP
#error BASE_STRT_DUMP is runtime : dump.base_strt_dump
#endif

#ifdef BASE_STRT_READ
#error BASE_STRT_READ  is runtime : dump.base_strt_read
#endif
