#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "imp.h"

namespace vwall {
#if   defined(WALL_FLAT)
  #include "imp/flat.h"
#elif defined(WVEL_DUPIRE_UP)
  #include "imp/dupire/up.h"
  #include "imp/dupire/common.h"
#elif defined(WVEL_DUPIRE_DOWN)
  #include "imp/dupire/down.h"
  #include "imp/dupire/common.h"
#elif defined(WVEL_SIN)
  #include "imp/sin.h"
#else
  #error WVEL_* is not defined
#endif
}
