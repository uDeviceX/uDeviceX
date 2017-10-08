#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "inc/def.h"
#include "msg.h"
#include "utils/mc.h"
#include "utils/cc.h"

#include "utils/kl.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "glb.h"

#include "mpi/wrapper.h"
#include "restrain/imp.h"

#include "imp.h"

namespace scheme {
namespace dev {
#ifdef FORWARD_EULER
  #include "imp/euler.h"
#else
  #include "imp/vv.h"
#endif
#if   defined(FORCE_NONE)
  #include "dev/force/none.h"
#elif defined(FORCE_CONSTANT)
  #include "dev/force/constant.h"
#elif defined(FORCE_DOUBLE_POISEUILLE)
  #include "dev/force/double_poiseuille.h"
#elif defined(FORCE_4ROLLER)
  #include "dev/force/4roller.h"
#else
  #error FORCE_* is undefined
#endif
#include "dev/main.h"
} /* namespace */

#include "imp/main.h"

#if   defined(RESTRAIN_NONE)
  #include "imp/restrain/none.h"
#elif defined(RESTRAIN_RED_VEL)
  #include "imp/restrain/red_vel.h"
#elif defined(RESTRAIN_RBC_VEL)
  #include "imp/restrain/rbc_vel.h"
#else
  #error RESTRAIN_* is undefined
#endif

} /* namespace */
