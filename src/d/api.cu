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
#  if  defined(CUDA_ALLOC_RELEASE)
#include "cuda/release/alloc.h"
#  elif defined(CUDA_ALLOC_DEBUG)
#include "cuda/debug/alloc.h"
#  else
#    error CUDA_ALLOC_* is undefined
#  endif
#elif defined(DEV_CPU)
#include "cpu/imp.h"
#else
#  error DEV_* is undefined
#endif

}
