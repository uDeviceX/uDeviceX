#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "imp.h"

namespace vwall {
#if   defined(WALL_FLAT)
  #include "imp/flat.h"
#elif defined(VWALL_DUPIRE_UP)
  #include "imp/dupire/up.h"
  #include "imp/dupire/common.h"
#elif defined(VWALL_DUPIRE_DOWN)
  #include "imp/dupire/down.h"
  #include "imp/dupire/common.h"
#elif defined(VWALL_SIN)
  #include "imp/sin.h"
#else
  #error VWALL_* is not defined
#endif
}
