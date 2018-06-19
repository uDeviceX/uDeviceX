#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/imp.h"
#include "utils/error.h"
#include "d/api.h"

namespace d {
#include "d/common.h"

#if   defined(DEV_CUDA)
  #include "cuda/imp.h"
#elif defined(DEV_CPU)
  #include "cpu/imp.h"
  #include "cpu/release/alloc.h"
#else
  #error DEV_* is undefined
#endif
}
