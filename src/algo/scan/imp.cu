#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "inc/dev.h"
#include "utils/imp.h"
#include "utils/error.h"
#include "utils/cc.h"
#include "utils/kl.h"
#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "imp.h"
#include "dev.h"

#if   defined(DEV_CUDA)
  #include "cuda/type.h"
  #include "cuda/imp.h"
#elif defined(DEV_CPU)
  #include "cpu/type.h"
  #include "cpu/imp.h"
#else
  #error DEV_* is undefined
#endif
