#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/halloc.h"

#include "type.h"
#include "imp.h"

namespace rbc { namespace rnd { namespace api {

#if   defined(DEV_CUDA)
  #include "imp/cuda.h"
#elif defined(DEV_CPU)
  #include "imp/gaussrand.h"
  #include "imp/cpu.h"
#else
  #error DEV_* is undefined
#endif

}}} /* namespace */
