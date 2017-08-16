/* function and variable type [q]ualifiers */

#if   defined(DEV_CUDA)
  #include "d/cuda/ker.h"
#elif defined(DEV_CPU)
  #include "d/cpu/ker.h"
#else
  #error DEV_* is undefined
#endif
