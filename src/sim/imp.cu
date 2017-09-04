#include <assert.h>
#include <vector>
#include <mpi.h>
#include <stdint.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "mpi/wrapper.h"
#include "io/off.h"
#include "algo/scan/int.h"

#include "rnd/imp.h"
#include "rnd/dev.h"

#include "inc/def.h"
#include "msg.h"
#include "mpi/glb.h"
#include "utils/cc.h"
#include "utils/mc.h"
#include "frag.h"

#include "utils/kl.h"
#include "mpi/basetags.h"
#include "inc/type.h"
#include "mpi/type.h"
#include "inc/dev.h"

#include "dual/type.h"
#include "dual/int.h"
#include "utils/texo.h"
#include "utils/te.h"

#include "cloud/hforces/type.h"
#include "cloud/hforces/int.h"

#include "inc/tmp/wrap.h"
#include "inc/tmp/pinned.h"
#include "io/field.h"
#include "io/fields_grid.h"
#include "io/rbc.h"
#include "io/rig.h"
#include "io/diag.h"

#include "dbg/imp.h"

#include "io/restart.h"

#include "glb.h"

#include "inc/dev/read.h"
#include "inc/dev/write.h"
#include "inc/dev/common.h"
#include "inc/dev/index.h"

#include "clist/int.h"

#include "mcomm/type.h"
#include "mcomm/int.h"

#include "rbc/int.h"

#include "mdstr/buf.h"
#include "mdstr/tic.h"
#include "mdstr/int.h"
#include "rdstr/int.h"

#include "field.h"

#include "forces/type.h"
#include "forces/imp.h"

#include "sdf/type.h"
#include "sdf/int.h"

#include "wall/int.h"

#include "flu/int.h"

#include "odstr/type.h"
#include "odstr/int.h"
#include "cnt/int.h"

#include "fsi/type.h"
#include "fsi/int.h"

#include "int.h"
#include "rex/int.h"
#include "lforces/local.h"

#include "sdstr/dec.h"
#include "sdstr/dev.h"
#include "sdstr/imp.h"
#include "sdstr/templ.h"

namespace dpdx {
namespace dev {
#include "lforces/x/dev.h"
}
#include "lforces/x/imp.h"
}

#include "dpdr/type.h"
#include "dpdr/int.h"

#include "mesh/collision.h"
#include "mesh/bbox.h"

#include "rigid/int.h"
#include "tcells/int.h"

#include "mbounce/imp.h"
#include "mrescue.h"

#include "bbhalo.decl.h"
#include "bbhalo.impl.h"

#include "io/bop/imp.h"
#include "rig/int.h"
#include "inter/int.h"

namespace sim {
namespace dev {
#ifdef FORWARD_EULER
  #include "scheme/euler.h"
#else
  #include "scheme/vv.h"
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
#include "dump.h"
#include "colors.h"

#define HST (true)
#define DEV (false)
#define DEVICE_SOLID
#ifdef DEVICE_SOLID
  #include "0dev/bounce.h"
  #include "0dev/update.h"
  #include "0dev/distr.h"
#else
  #include "0hst/bounce.h"
  #include "0hst/update.h"
  #include "0hst/distr.h"
#endif
#undef HST
#undef DEV

#include "update.h"

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

#include "rbc.h"
#include "step.h"
#include "run.h"
#include "imp.h"
}
