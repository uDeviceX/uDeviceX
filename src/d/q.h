/* function and variable type [q]ualifiers */

#if !defined(DEV_CUDA) && !defined(DEV_CPU)
  #include "d/cuda/q.h"
#elif
  #include "d/cpu/q.h"
#else
  #error DEV_* is undefined
#endif
