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
#include "glb/get.h"

#include "mpi/wrapper.h"
#include "restrain/imp.h"

#include "imp.h"

namespace scheme { namespace restrain {
#if   defined(RESTRAIN_NONE)
  #include "imp/none.h"
#elif defined(RESTRAIN_RED_VEL)
  #include "imp/red_vel.h"
#elif defined(RESTRAIN_RBC_VEL)
  #include "imp/rbc_vel.h"
#else
  #error RESTRAIN_* is undefined
#endif
}} /* namespace */
