#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "inc/def.h"
#include "msg.h"
#include "utils/cc.h"

#include "utils/kl.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "glb.h"

#include "int.h"
namespace scheme {
namespace dev {
#ifdef FORWARD_EULER
  #include "scheme/imp/euler.h"
#else
  #include "scheme/imp/vv.h"
#endif
#include "scheme/dev/main.h"
}

/*
#if   defined(FORCE_NONE)
  #include "dev/force/none.h"
#elif defined(FORCE_DOUBLE_POISEUILLE)
  #include "dev/force/double_poiseuille.h"
#elif defined(FORCE_4ROLLER)
  #include "dev/force/4roller.h"
#else
  #error DEV_* is undefined
#endif
*/
}
