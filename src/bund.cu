#include <sys/stat.h>

#include <assert.h>
#include <string>
#include <vector>
#include <cstdio>
#include <mpi.h>

#include <limits> /* for rnd */
#include <stdint.h>

#include "l/clist.h"
#include "l/m.h"
#include "l/off.h"
#include "l/scan.h"

#include "l/rnd.h"

#include <conf.h>
#include "conf.common.h"
#include "m.h"     /* MPI */
#include "basetags.h"
#include "common.h"
#include "common.tmp.h"
#include "common.macro.h"
#include "io.h"
#include "bund.h"
#include "force.h"

#include "restart.h"

#include "glb.h"

#include "l/float3.h"
#include "k/common.h"

#include "clist.face.h"
#include "minmax.h"

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

#include "containers.impl.h"

#include "field.h"

#include "k/wvel.h" /* wall velocity used by sdf and wall */

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
#include "odstr/hdr.h" /* Distr class decl */
#include "odstr/ini.h"
#include "odstr/imp.h" /* Distr class impl */
#include "odstr/fin.h"
}
#include "odstr/int.h"
}

#include "k/fsi.h"

#include "fsi.decl.h"
#include "fsi.impl.h"

#include "rex.decl.h"
#include "k/rex.h"
#include "rex.impl.h"

#include "bipsbatch.decl.h"
#include "bipsbatch.impl.h"

#include "dpd/local.h"

#ifdef  YDPD
  #include "y/k/halo.h"
  #include "y/dpd/remote.decl.h"
  #include "y/dpd/buf.decl.h"
  #include "dpd/forces.h"        /* common */
  #include "y/dpd/ini.h"
  #include "y/dpd/remote.impl.h"
  #include "y/dpd/pack.impl.h"
  #include "y/dpd/post.impl.h"
  #include "y/dpd/recv.impl.h"
#else
  /* old dpd */
  #include "x/phalo.decl.h"
  #include "x/phalo.impl.h"
  #include "x/dpd/remote.decl.h"
  #include "dpd/forces.h"          /* common */
  #include "x/dpd/ini.h"
  #include "x/dpd/remote.impl.h"
#endif

#include "collision.h"

#include "solid.h"
#include "tcells.h"

#include "mbounce.h"
#include "mrescue.h"

#include "bbhalo.decl.h"
#include "bbhalo.impl.h"

#include "dump.decl.h"
#include "dump.impl.h"

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
