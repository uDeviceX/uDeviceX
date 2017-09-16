#include <assert.h>
#include <vector>
#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "mpi/wrapper.h"
#include "algo/scan/int.h"

#include "rnd/imp.h"
#include "rnd/dev.h"

#include "inc/def.h"
#include "msg.h"
#include "mpi/glb.h"
#include "utils/cc.h"
#include "utils/mc.h"

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
#include "io/fields_grid.h"
#include "io/rbc.h"
#include "io/rig.h"
#include "io/diag.h"

#include "dbg/imp.h"
#include "io/restart.h"
#include "glb.h"

#include "inc/dev/read.h"
#include "clist/imp.h"

#include "mcomm/type.h"
#include "mcomm/int.h"

#include "rbc/int.h"

#include "mdstr/buf.h"
#include "mdstr/tic.h"
#include "mdstr/int.h"
#include "rdstr/int.h"

#include "sdf/type.h"
#include "sdf/int.h"
#include "wall/int.h"

#include "flu/int.h"


#include "comm/imp.h"
#include "distr/flu/type.h"
#include "distr/flu/imp.h"
#include "cnt/imp.h"

#include "fsi/type.h"
#include "fsi/int.h"
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

#include "bbhalo/imp/dec.h"
#include "bbhalo/imp/main.h"

#include "io/bop/imp.h"
#include "rig/int.h"
#include "inter/int.h"
#include "scheme/int.h"

#include "imp.h"
namespace sim {
#include "imp/type.h"
#include "imp/dec.h"
#include "imp/force/common.h"
#include "imp/force/dpd.h"
#include "imp/force/imp.h"

#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/dump.h"
#include "imp/colors.h"

#define HST (true)
#define DEV (false)
#define DEVICE_SOLID
#ifdef DEVICE_SOLID
  #include "imp/0dev/bounce.h"
  #include "imp/0dev/update.h"
  #include "imp/0dev/distr.h"
#else
  #include "imp/0hst/bounce.h"
  #include "imp/0hst/update.h"
  #include "imp/0hst/distr.h"
#endif
#undef HST
#undef DEV

#include "imp/update.h"
#include "imp/distr.h"
#include "imp/rbc.h"
#include "imp/step.h"
#include "imp/run.h"
#include "imp/main.h"

}
