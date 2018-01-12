#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <curand.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "utils/imp.h"

#include "type.h"
#include "imp.h"

namespace rnd_api {

#if   defined(DEV_CUDA)
  #include "imp/cuda.h"
#elif defined(DEV_CPU)
  #include "imp/gaussrand.h"
  #include "imp/cpu.h"
#else
  #error DEV_* is undefined
#endif

} /* namespace */
