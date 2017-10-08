#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "imp.h"

namespace gdot {
#if   defined(GDOT_FLAT)
  #include "imp/flat.h"
#elif defined(GDOT_DUPIRE_UP)
  #include "imp/dupire/up.h"
#elif defined(GDOT_DUPIRE_DOWN)
  #include "imp/dupire/down.h"
#endif
}
