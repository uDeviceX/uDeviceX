#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "algo/scan/int.h"

#include "rnd/imp.h"
#include "rnd/dev.h"

#include "inc/def.h"
#include "msg.h"
#include "mpi/glb.h"
#include "utils/cc.h"

#include "frag/imp.h"

#include "utils/kl.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "dbg/imp.h"

#include "inc/dev/read.h"
#include "inc/dev/common.h"
#include "inc/dev/index.h"

#include "forces/type.h"
#include "forces/use.h"
#include "forces/pack.h"
#include "forces/imp.h"

namespace cnt {

typedef Sarray<const float2*, MAX_OBJ_TYPES> float2pWraps;
typedef Sarray<      float *, MAX_OBJ_TYPES>  ForcepWraps;

namespace dev {
#include "dev/decl.h"
#include "dev/fetch.h"
#include "dev/pair.h"
#include "dev/map/common.h"
#include "dev/map/halo.h"
#include "dev/map/bulk.h"
#include "dev/code.h"
#include "dev/bulk.h"
#include "dev/halo.h"
#include "dev/pop.h"
}


#include "imp/decl.h"
#include "imp/bind.h"
#include "imp/bulk.h"
#include "imp/fin.h"
#include "imp/halo.h"
#include "imp/ini.h"
#include "imp/setup.h"

} /* namespace */
