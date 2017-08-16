#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "common.h"
#include "inc/dev.h"
#include "msg.h"
#include "cc.h"
#include "kl.h"
#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "scan/int.h"
#include "scan/dev.h"

namespace scan {
#if   defined(DEV_CUDA)
  #include "scan/cuda/imp.h"
#elif defined(DEV_CPU)
  #include "scan/cpu/imp.h"
#else
  #error DEV_* is undefined
#endif
}
