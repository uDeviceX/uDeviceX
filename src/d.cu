#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "cc.h"
#include "d.h"

namespace d {
#if   defined(DEV_CUDA)
  #include "d/cuda/imp.h"
#elif defined(DEV_HST)
  #include "d/hst/imp.h"
#else
  #error DEV_* is undefined
#endif
}
