#include <sys/stat.h> /* for dump.impl */

#include <assert.h>
#include <vector>
#include <mpi.h>


#include "l/clist.h"
#include "l/m.h"
#include "l/off.h"
#include "scan/int.h"

#include <limits> /* for rnd */
#include <stdint.h>
#include "rnd.h"

#include <conf.h>
#include "conf.common.h"
#include "m.h"     /* MPI */
#include "basetags.h"
#include "common.h"
#include "common.mpi.h"
#include "common.cuda.h"
#include "common.tmp.h"
#include "common.macro.h"
#include "io.h"
#include "bund.h"
#include "diag.h"

#include "restart.h"

#include "glb.h"

#include "l/float3.h"
#include "k/common.h"

#include "clist.face.h"
#include "minmax.h"

namespace mcomm {
namespace sub {
namespace dev {
#include "mcomm/dev.h"
}
#include "mcomm/ini.h"
#include "mcomm/imp.h"
#include "mcomm/fin.h"
}
#include "mcomm/int.h"
}

namespace rbc {
namespace sub {
#define __DF__ __device__ __forceinline__
/* physical part of RBC calculations : see also params/rbc.inc0.h */
#include "rbc/dev0.h"
#include "rbc/dev.h"
#undef __DF__
#include "rbc/imp.h"
}
namespace ic {
#include "rbc/ic.h"
}
#include "rbc/int.h"
}

#include "rdstr.decl.h"
#include "k/rdstr.h"
#include "rdstr.impl.h"

#include "sdstr.decl.h"
#include "sdstr.impl.h"
#include "field.h"

#include "k/wvel.h" /* wall velocity used by sdf and wall */
#include "forces.h"

#include "cnt.decl.h"
#include "k/cnt.h"
#include "cnt.impl.h"

namespace sdf {
namespace sub {
namespace dev {
#include "sdf/dev.h"
}
#include "sdf/imp.h"
}
#include "sdf/int.h"
}

namespace wall {
namespace sub {
namespace dev {
#include "wall/dev.h"
}
namespace strt {
#include "wall/strt.h"
}
#include "wall/imp.h"
}
#include "wall/int.h"
}

namespace flu {
namespace sub {
namespace dev {
#include "flu/dev.h"
}
#include "flu/imp.h"
}
#include "flu/int.h"
}

namespace odstr {
namespace sub {
namespace dev {
#include "odstr/dev.h"
}
#include "odstr/buf.h"
#include "odstr/ini.h"
#include "odstr/imp.h"
#include "odstr/fin.h"
}
#include "odstr/int.h"
}

#include "k/fsi/type.h"
#include "k/fsi/decl.h"
#include "k/fsi/common.h"
#include "k/fsi/map.common.h"
#include "k/fsi/map.bulk.h"
#include "k/fsi/map.halo.h"
#include "k/fsi/bulk.h"
#include "k/fsi/halo.h"

#include "fsi/decl.h"
#include "fsi/setup.h"
#include "fsi/impl.h"

#include "x/type.h"

  #include "rex/decl.h"
  #include "k/rex.h"
  #include "rex/mpi.h"
  #include "rex/ini.h"
  #include "rex/pack.h"
  #include "rex/impl.h"
  #include "rex/post.h"
  #include "rex/fin.h"

#include "x/decl.h"
#include "x/common.h"
#include "x/ticketcom.h"
#include "x/ticketr.h"
#include "x/tickettags.h"
#include "x/impl.h"

#include "bipsbatch.type.h"
#include "k/bipsbatch/map.h"
#include "k/bipsbatch/common.h"
#include "bipsbatch.impl.h"

#include "dpd/local.h"
#include "dpd/flocal.h"

namespace dpdx {
namespace dev {
#include "dpd/x/dev.h"
}
#include "dpd/x/imp.h"
}

namespace dpdr {
namespace sub {
namespace dev {
#include "dpdr/dev.h"
}
#include "dpdr/buf.h"
#include "dpdr/ini.h"
#include "dpdr/imp.h"
#include "dpdr/fin.h"
}
#include "dpdr/int.h"
}

#include "mesh/collision.h"
#include "mesh/dist.h"
#include "mesh/bbox.h"

#include "solid.h"
#include "tcells.h"

#include "mbounce.h"
#include "mrescue.h"

#include "bbhalo.decl.h"
#include "bbhalo.impl.h"

#include "dump.h"
#include "l/ply.h"

namespace rig {
namespace sub {
namespace ic {
#include "rig/ic.h"
}
#include "rig/imp.h"
}
#include "rig/int.h"
}

namespace sim {
namespace dev {
#include "sim/dev.h"
}
#include "sim/dec.h"
#include "sim/ini.h"
#include "sim/fin.h"
#include "sim/generic.h"
#include "sim/dump.h"
#include "sim/tag.h"
#include "sim/forces.h"
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
#include "sim/step.h"
#include "sim/run.h"
#include "sim/imp.h"
}
