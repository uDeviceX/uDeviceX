#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "imp.h"

namespace vwall {
#if   defined(WVEL_FLAT)
  #include "imp/flat.h"
#elif defined(WVEL_SIN)
  #include "imp/sin.h"
#else
  #error WVEL_* is not defined
#endif
}
