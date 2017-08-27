#include <assert.h>
#include <vector>
#include <mpi.h>
#include <stdint.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "l/m.h"
#include "l/off.h"
#include "scan/int.h"

#include "rnd/imp.h"
#include "rnd/dev.h"

#include "inc/def.h"
#include "msg.h"
#include "m.h"
#include "cc.h"
#include "mc.h"
#include "frag.h"

#include "kl.h"
#include "basetags.h"
#include "inc/type.h"
#include "inc/mpi.h"
#include "inc/dev.h"

#include "dual/type.h"
#include "dual/int.h"
#include "texo.h"
#include "te.h"

#include "inc/tmp/wrap.h"
#include "inc/tmp/pinned.h"
#include "io/field.h"
#include "io/rbc.h"
#include "diag.h"

#include "dbg.h"

#include "restart.h"

#include "glb.h"

#include "k/read.h"
#include "k/write.h"
#include "k/common.h"
#include "k/index.h"

#include "clist/int.h"

#include "mcomm/type.h"
#include "mcomm/int.h"

#include "rbc/int.h"

#include "mdstr/buf.h"
#include "mdstr/tic.h"
#include "mdstr/int.h"
#include "rdstr/int.h"

#include "field.h"

#include "forces/imp.h"

#include "sdf/type.h"
#include "sdf/int.h"

#include "wall/int.h"

#include "flu/int.h"

#include "odstr/type.h"
#include "odstr/int.h"
#include "cnt/int.h"
#include "fsi/int.h"

#include "int.h"
#include "x/int.h"
#include "dpd/local.h"

#include "sdstr/dec.h"
#include "sdstr/dev.h"
#include "sdstr/imp.h"
#include "sdstr/templ.h"

namespace dpdx {
namespace dev {
#include "dpd/x/dev.h"
}
#include "dpd/x/imp.h"
}

#include "dpdr/type.h"
#include "dpdr/int.h"

#include "mesh/collision.h"
#include "mesh/bbox.h"

#include "solid.h"
#include "tcells/int.h"

#include "mbounce/imp.h"
#include "mrescue.h"

#include "bbhalo.decl.h"
#include "bbhalo.impl.h"

#include "dump/int.h"
#include "rig/int.h"

namespace sim {
namespace dev {
#ifdef FORWARD_EULER
  #include "sch/euler.h"
#else
  #include "sch/vv.h"
#endif
#include "dev.h"
#include "force/dev.h"
}

/* local */
#include "type.h"
#include "dec.h"
#include "force/common.h"
#include "force/dpd.h"
#include "force/imp.h"

#include "ini.h"
#include "fin.h"
#include "generic.h"
#include "dump.h"
#include "tag.h"


#define HST (true)
#define DEV (false)
#define DEVICE_SOLID
#ifdef DEVICE_SOLID
  #include "0dev/sim.impl.h"
#else
  #include "0hst/sim.impl.h"
#endif
#undef HST
#undef DEV

#if   defined(UPDATE1)
  #include "update/release.h"
#elif defined(UPDATE_SAFE)
  #include "update/safe.h"
#else
  #error UPDATE* is undefined
#endif

#if   defined(ODSTR1)
  #include "odstr/release.h"
#elif defined(ODSTR0)
  #include "odstr/none.h"
#elif defined(ODSTR_SAFE)
  namespace sub {
    #include "odstr/release.h"
  }
  #include "odstr/safe.h"
#else
  #error ODSTR* is undefined
#endif

#include "step.h"
#include "run.h"
#include "imp.h"
}
