#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/def.h"
#include "msg.h"
#include "utils/error.h"

#include "glb/get.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "utils/cc.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "utils/texo.h"
#include "utils/texo.dev.h"
#include "utils/te.h"

#include "inc/macro.h"

#include "rnd/imp.h"
#include "rnd/dev.h"

#include "glob/type.h"
#include "wvel/type.h"

#include "sdf/type.h"
#include "sdf/dev.h"
#include "sdf/imp.h"

#include "inc/dev/wvel.h"
#include "forces/type.h"
#include "forces/use.h"
#include "forces/pack.h"
#include "forces/imp.h"

#include "cloud/imp.h"
#include "cloud/dev.h"

#include "utils/kl.h"
#include "imp.h"

namespace wall {
/*** generic ***/
namespace dev {
  namespace map {
    #include "dev/map/type.h"
    #include "dev/map/ini.h"
    #include "dev/map/use.h"
  }
  #include "dev/main0.h"
}

/*** polymorphic ***/
namespace grey {
  namespace dev {
    #include "dev/fetch/grey.h"
    #include "dev/main.h"
  }
  #include "imp/main.h"
}

namespace color {
  namespace dev {
    #include "dev/fetch/color.h"
    #include "dev/main.h"
  }
  #include "imp/main.h"
}
}
