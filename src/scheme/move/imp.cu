#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "inc/def.h"
#include "utils/msg.h"
#include "utils/cc.h"

#include "utils/kl.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "scheme/move/params/imp.h"

#include "imp.h"

namespace scheme { namespace move {
namespace dev {
#ifdef FORWARD_EULER
  #include "dev/euler.h"
#else
  #include "dev/vv.h"
#endif
#include "dev/main.h"
} /* namespace */

#include "imp/main.h"
}} /* namespace */
