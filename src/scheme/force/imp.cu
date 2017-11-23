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

#include "imp.h"

namespace scheme { namespace force {
namespace dev {
#if   defined(FORCE_NONE)
  #include "dev/none.h"
#elif defined(FORCE_CONSTANT)
  #include "dev/constant.h"
#elif defined(FORCE_DOUBLE_POISEUILLE)
  #include "dev/double_poiseuille.h"
#elif defined(FORCE_4ROLLER)
  #include "dev/4roller.h"
#else
  #error FORCE_* is undefined
#endif
} /* namespace */

#include "imp/main.h"
}} /* namespace */
