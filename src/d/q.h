/* function and variable type [q]ualifiers */

#if   defined(DEV_CUDA)
  #include "d/cuda/q.h"
#elif defined(DEV_CPU)
  #include "d/cpu/q.h"
#else
  #error DEV_* is undefined
#endif
