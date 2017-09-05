#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "msg.h"
#include "utils/cc.h"
#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "mpi/wrapper.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "utils/mc.h"
#include "io/restart.h"

#include "imp.h"

#if   defined(DEV_CUDA)
  #include "dev1.h"
#elif defined(DEV_CPU)
  #include "dev0.h"
#else
  #error DEV_* is undefined
#endif

#include "utils/kl.h"
namespace sub {
#include "imp/main.h"
}
