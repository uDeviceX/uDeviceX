#include <assert.h>
#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "utils/error.h"
#include "utils/imp.h"

#include "math/rnd/imp.h"
#include "math/rnd/dev.h"
#include "math/dev.h"

#include "inc/def.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "utils/cc.h"

#include "frag/imp.h"

#include "utils/kl.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "frag/dev.h"

#include "dbg/imp.h"

#include "pair/type.h"
#include "pair/dev.h"
#include "pair/imp.h"

#include "partlist/type.h"

#include "clist/imp.h"
#include "clist/dev.h"

#include "imp.h"

typedef Sarray<const float2*, MAX_OBJ_TYPES> float2pWraps;
typedef Sarray<      float *, MAX_OBJ_TYPES>  ForcepWraps;

namespace dev {
#include "dev/pair.h"
#include "dev/map/common.h"
#include "dev/map/halo.h"
#include "dev/map/bulk.h"
#include "dev/bulk.h"
#include "dev/halo.h"
}

#include "imp/type.h"
#include "imp/bulk.h"
#include "imp/halo.h"
#include "imp/main.h"
