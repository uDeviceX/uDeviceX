#include <assert.h>
#include <mpi.h>
#include <stdint.h>
#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "rnd/imp.h"
#include "rnd/dev.h"

#include "inc/def.h"
#include "msg.h"
#include "mpi/glb.h"
#include "utils/cc.h"
#include "utils/mc.h"
#include "frag/imp.h"

#include "utils/kl.h"
#include "inc/type.h"
#include "mpi/type.h"
#include "inc/dev.h"

#include "utils/texo.h"
#include "utils/te.h"

#include "sim/imp.h"
#include "dbg/imp.h"

#include "inc/dev/common.h"

#include "forces/type.h"
#include "forces/pack.h"
#include "forces/imp.h"
#include "cloud/imp.h"
#include "cloud/dev.h"

/* local */
#include "type.h"
#include "imp.h"

/* body */
namespace fsi {
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

#include "imp/main.h"
#include "imp/bulk.h"
#include "imp/halo.h"
}
