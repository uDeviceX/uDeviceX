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

#include "dpd/cuda-dpd.h"
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

#include "glb.h"

#include "l/float3.h"
#include "k/common.h"

#include "clist.face.h"

#include "k/odstr.h"
#include "odstr.decl.h"
#include "odstr/ini.h"
#include "odstr.impl.h"
#include "odstr/fin.h"
#include "minmax.h"

#define __DF__ __device__ __forceinline__
/* physical part of RBC calculations : see also params/rbc.inc0.h */
#include "k/rbc0.h"
#include "k/rbc.h"
#undef __DF__
#include "rbc.decl.h"
#include "rbc.impl.h"

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

#include "k/sdf.h"
#include "sdf.decl.h"
#include "sdf.impl.h"

namespace wall {
namespace sub {
#include "wall/dev.h"
#include "wall/imp.h"
}
#include "wall/int.h"
}

namespace sol {
namespace sub {
#include "sol/dev.h"
#include "sol/imp.h"
}
#include "sol/int.h"
}


#include "k/fsi.h"

#include "fsi.decl.h"
#include "fsi.impl.h"

#include "rex.decl.h"
#include "k/rex.h"
#include "rex.impl.h"

#include "packinghalo.decl.h"
#include "packinghalo.impl.h"

#include "bipsbatch.decl.h"
#include "bipsbatch.impl.h"

#include "dpd.decl.h"
#include "dpd.impl.h"

#include "collision.h"

#include "solid.h"
#include "tcells.h"

#include "mbounce.h"
#include "mrescue.h"

#include "bbhalo.decl.h"
#include "bbhalo.impl.h"

#include "ic.impl.h"

#include "dump.decl.h"
#include "dump.impl.h"

#include "ply.h"

#include "s/decl.h"
#include "s/ic.h"
#include "s/impl.h"

#include "safety.impl.h"

#include "restart.h"

#include "k/sim.h"
#include "sim.decl.h"

#define HST (true)
#define DEV (false)
#define DEVICE_SOLID
#ifdef DEVICE_SOLID
  #include "0dev/sim.impl.h"
#else
  #include "0hst/sim.impl.h"
#endif
#include "sim.impl.h"
#undef HST
#undef DEV
