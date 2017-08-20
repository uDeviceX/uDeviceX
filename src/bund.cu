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

#include "common.h"
#include "msg.h"
#include "m.h"
#include "cc.h"

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
#include "bund.h"
#include "diag.h"

#include "restart.h"

#include "glb.h"

#include "l/float3.h"
#include "k/read.h"
#include "k/write.h"
#include "k/common.h"

#include "clist/int.h"

#include "mcomm/type.h"
#include "mcomm/int.h"

#include "rbc/imp.h"
namespace rbc {
#include "params/rbc.inc0.h"
#include "rbc/int/imp.h"
}

#include "mdstr/buf.h"
#include "mdstr/tic.h"
#include "mdstr/int.h"
#include "rdstr/int.h"

#include "sdstr.decl.h"
#include "sdstr.impl.h"
#include "field.h"

#include "forces.h"

#include "k/cnt/type.h"
#include "k/cnt/decl.h"
#include "k/cnt/fetch.h"
#include "k/cnt/bulk.h"
#include "k/cnt/halo.h"
#include "k/cnt/pop.h"
#include "cnt/decl.h"
#include "cnt/bind.h"
#include "cnt/build.h"
#include "cnt/bulk.h"
#include "cnt/fin.h"
#include "cnt/halo.h"
#include "cnt/ini.h"
#include "cnt/setup.h"

#include "sdf/type.h"
#include "sdf/int.h"

#include "wall/int.h"

#include "flu/int.h"

#include "odstr/type.h"
#include "odstr/int.h"

#include "k/fsi/type.h"
#include "k/fsi/decl.h"
#include "k/fsi/fetch.h"
#include "k/fsi/common.h"
#include "k/fsi/map.common.h"
#include "k/fsi/map.bulk.h"
#include "k/fsi/map.halo.h"
#include "k/fsi/bulk.h"
#include "k/fsi/halo.h"

#include "fsi/decl.h"
#include "fsi/setup.h"
#include "fsi/bind.h"
#include "fsi/bulk.h"
#include "fsi/fin.h"
#include "fsi/halo.h"
#include "fsi/ini.h"

#include "x/type.h"
  #include "rex/type/remote.h"
  #include "rex/type/local.h"
  #include "rex/decl.h"

  #include "k/rex/type.h"
  #include "k/rex/decl.h"
  #include "k/rex/common.h"
  #include "k/rex/x.h" /* TODO */
  #include "k/rex/pack.h"
  #include "k/rex/scan.h"
  #include "k/rex/scatter.h"
  #include "k/rex/unpack.h"

  #include "rex/ini.h"
  #include "rex/copy.h"
  #include "rex/wait.h"
  #include "rex/halo.h"
  #include "rex/scan.h"
  #include "rex/pack.h"
  #include "rex/send.h"
  #include "rex/recv.h"
  #include "rex/unpack.h"
  #include "rex/fin.h"

#include "x/decl.h"
#include "x/common.h"
#include "x/ticketcom.h"
#include "x/ticketr.h"
#include "x/tickettags.h"
#include "x/ticketpack.h"
#include "x/ticketpinned.h"
#include "x/ini.h"
#include "x/fin.h"
#include "x/impl.h"

#include "dpd/local.h"

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

#include "dump.h"

#include "rig/int.h"

namespace sim {
namespace dev {
#ifdef FORWARD_EULER
  #include "sim/sch/euler.h"
#else
  #include "sim/sch/vv.h"
#endif

#include "sim/dev.h"
}
#include "sim/dec.h"
#include "sim/ini.h"
#include "sim/fin.h"
#include "sim/generic.h"
#include "sim/dump.h"
#include "sim/tag.h"
#include "sim/forces/dpd.h"
#include "sim/forces.h"

#if   defined(FORCE1)
  #include "sim/force1.h"
#elif defined(FORCE0)
  #include "sim/force0.h"
#else
  #error FORCE[01] is undefined
#endif


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
#include "sim/update.h"

#if   defined(ODSTR1)
  #include "sim/odstr1.h"
#elif defined(ODSTR0)
  #include "sim/odstr0.h"
#else
  #error ODSTR[01] is undefined
#endif

#include "sim/step.h"
#include "sim/run.h"
#include "sim/imp.h"
}
