#include <stdio.h>
#include <assert.h>


#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "math/rnd/imp.h"
#include "math/rnd/dev.h"
#include "math/dev.h"

#include "inc/def.h"
#include "utils/msg.h"
#include "utils/imp.h"

#include "mpi/glb.h"
#include "utils/cc.h"
#include "utils/mc.h"
#include "frag/imp.h"
#include "frag/dev.h"

#include "utils/kl.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "dbg/imp.h"

#include "pair/type.h"
#include "pair/dev.h"
#include "pair/imp.h"
#include "cloud/imp.h"
#include "cloud/dev.h"

/* local */

#include "imp.h"

/* body */
namespace dev {
#include "dev/type.h"
#include "dev/common.h"
#include "dev/map.common.h"
#include "dev/map/bulk.h"
#include "dev/map/halo.h"
#include "dev/pair.h"
#include "dev/bulk.h"
#include "dev/halo.h"
}

#include "imp/type.h"
#include "imp/main.h"
#include "imp/bulk.h"
#include "imp/halo.h"
