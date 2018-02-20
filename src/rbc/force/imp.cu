#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "math/dev.h"

#include "io/off/imp.h"

#include "utils/error.h"
#include "utils/msg.h"
#include "utils/imp.h"

#include "d/q.h"
#include "d/api.h"
#include "d/ker.h"
#include "utils/cc.h"
#include "utils/kl.h"

#include "rbc/params/imp.h"
#include "rbc/adj/imp.h"
#include "rbc/shape/imp.h"

#include "rbc/rnd/imp.h"
#include "rbc/type.h"

#include "rbc/adj/type/common.h"
#include "rbc/adj/type/dev.h"
#include "rbc/adj/dev.h"
#include "area_volume/imp.h"

#include "imp.h"
#include "imp/type.h"

namespace rbc_force_dev {
#if   defined(RBC_DOUBLE)
  #include "dev/double.h"
#elif defined(RBC_FLOAT)
  #include "dev/float.h"
#else
  #error RBC_DOUBLE or RBC_FLOAT must be defined
#endif
#include "dev/common.h"

#if   RBC_STRESS_FREE
  #include "dev/stress_free1/shape.h"
  #include "dev/stress_free1/force.h"
#else
  #include "dev/stress_free0/shape.h"
  #include "dev/stress_free0/force.h"
#endif

#if   RBC_RND
  #include "dev/rnd1/main.h"
#else
  #include "dev/rnd0/main.h"
#endif
#include "dev/main.h"
}

#include "imp/main.h"
#include "imp/forces.h"
#include "imp/stat.h"
