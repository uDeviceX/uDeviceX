#include <stdint.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "msg.h"
#include "utils/cc.h"
#include "d/q.h"
#include "d/ker.h"
#include "d/api.h"

#include "rnd/imp.h"
#include "rnd/dev.h"
#include "inc/dev.h"
#include "inc/type.h"
#include "local.h"

#if   defined(DEV_CUDA)
  #include "utils/kl.h"
  #include "forces/type.h"
  #include "forces/pack.h"
  #include "forces/hook.h"
  #include "forces/imp.h"

  #include "imp/type.h"

  #include "dev/float.h"
  #include "dev/decl.h"
  #include "dev/fetch.h"

  #define __IMOD(x,y) ((x)-((x)/(y))*(y))

  #include "dev/pack.h"
  #include "cloud/lforces/get.h"
  #include "dev/dpd.h"
  #include "dev/core.h"

  #define MYCPBX  (4)
  #define MYCPBY  (2)
  #define MYCPBZ  (2)
  #define MYWPB   (4)

  #include "dev/merged.h"
  #include "dev/tex.h"
  #include "dev/transpose.h"

  #include "cloud/lforces/int.h"

  #include "imp/decl.h"
  #include "imp/setup.h"
  #include "imp/tex.h"
  #include "imp/info.h"
  #include "imp/flocal.h"
#elif defined(DEV_CPU)
  #include "local0.h"
#else
  #error DEV_* is undefined
#endif
