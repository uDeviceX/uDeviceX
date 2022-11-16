#if   defined(DEV_CUDA)
  #include "d/cuda/type.h"
#elif defined(DEV_CPU)
  #include "d/cpu/type.h"
#else
  #error DEV_* is undefined
#endif
