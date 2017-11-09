#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "imp.h"

namespace gdot {
#if   defined(WALL_FLAT)
  #include "imp/flat.h"
#elif defined(GDOT_DUPIRE_UP)
  #include "imp/dupire/up.h"
  #include "imp/dupire/common.h"
#elif defined(GDOT_DUPIRE_DOWN)
  #include "imp/dupire/down.h"
  #include "imp/dupire/common.h"
#endif
}
