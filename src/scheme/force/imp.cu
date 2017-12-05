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
#if   defined(FORCE_NONE)
  #include "imp/none.h"
#elif defined(FORCE_CONSTANT)
  namespace dev {
  #include "dev/constant.h"
  }
  #include "imp/constant.h"
#elif defined(FORCE_DOUBLE_POISEUILLE)
  namespace dev {
  #include "dev/double_poiseuille.h"
  }
  #include "imp/double_poiseuille.h"
#elif defined(FORCE_SHEAR)
  namespace dev {
  #include "dev/shear.h"
  }
  #include "imp/shear.h"
#elif defined(FORCE_4ROLLER)
  namespace dev {
  #include "dev/4roller.h"
  }
  #include "imp/4roller.h"
#elif defined(FORCE_RADIAL)
  namespace dev {
  #include "dev/radial.h"
  }
  #include "imp/radial.h"
#else
  #error FORCE_* is undefined
#endif
}} /* namespace */
