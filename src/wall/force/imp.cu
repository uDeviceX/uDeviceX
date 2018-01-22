#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "utils/msg.h"
#include "utils/error.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "utils/cc.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "utils/texo.h"
#include "utils/texo.dev.h"

#include "math/rnd/imp.h"
#include "math/rnd/dev.h"

#include "coords/type.h"
#include "coords/dev.h"
#include "wvel/type.h"
#include "wvel/dev.h"

#include "math/tform/type.h"
#include "math/tform/dev.h"

#include "sdf/def.h"
#include "sdf/tex3d/type.h"
#include "sdf/type.h"
#include "sdf/dev.h"

#include "forces/type.h"
#include "forces/use.h"
#include "forces/pack.h"
#include "forces/imp.h"

#include "cloud/imp.h"
#include "cloud/dev.h"

#include "utils/kl.h"
#include "imp.h"

/*** generic ***/
namespace wf_dev {
  namespace map {
    #include "dev/map/type.h"
    #include "dev/map/ini.h"
    #include "dev/map/use.h"
  }
  #include "dev/main0.h"
}

/*** polymorphic ***/
namespace grey {
  namespace wf_dev {
    #include "dev/fetch/grey.h"
    #include "dev/main.h"
  }
  #include "imp/main.h"
}

namespace color {
  namespace wf_dev {
    #include "dev/fetch/color.h"
    #include "dev/main.h"
  }
  #include "imp/main.h"
}
