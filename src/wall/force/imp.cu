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
#include "math/dev.h"

#include "coords/type.h"
#include "coords/dev.h"
#include "coords/imp.h"

#include "wall/wvel/type.h"
#include "wall/wvel/dev.h"

#include "math/tform/type.h"
#include "math/tform/dev.h"

#include "wall/sdf/tex3d/type.h"
#include "wall/sdf/type.h"
#include "wall/sdf/dev.h"

#include "pair/type.h"
#include "pair/dev.h"
#include "pair/imp.h"

#include "parray/type.h"
#include "parray/imp.h"
#include "parray/dev.h"

#include "utils/kl.h"
#include "imp.h"

namespace wf_dev {
namespace map {
#include "dev/map/type.h"
#include "dev/map/ini.h"
#include "dev/map/use.h"
}
#include "dev/main.h"
}

#include "imp/main.h"
