#include <sys/stat.h>

#include <string>
#include <vector>
#include <cstdio>
#include <mpi.h>

#include "dpd/cuda-dpd.h"
#include "dpd/dpd-rng.h"

#include <conf.h>
#include "conf.common.h"
#include "m.h"     /* MPI */
#include "basetags.h"
#include "common.h"
#include "common.tmp.h"
#include "common.macro.h"
#include "io.h"
#include "bund.h"
#include "dpd-forces.h"

#include <thrust/device_vector.h> /* for clist.impl.h */
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include "glb.h"

#include "helper-math/helper_math.h"

#include "k/scan.h"
#include "k/common.h"

namespace x {
  #include "x/k/clist.h"
  #include "x/clist.impl.h"
  #include "x/clist.face.h"

  #include "x/k/sdstr.h"
  #include "x/common.h"
  #include "x/common.tmp.h"

  namespace m {
    #include "x/m.impl.h"
  }

  #include "x/sdstr.decl.h"
  #include "x/sdstr/ini.h"
  #include "x/sdstr.impl.h"
  #include "x/sdstr/fin.h"

  #include "x/x.decl.h" /* an interface */
  #include "x/x.impl.h"
}

#include "rminmax.h"
#include "off.impl.h"

#include "k/rbc.h"
#include "rbc.impl.h"

#include "rdstr.decl.h"
#include "k/rdstr.h"
#include "rdstr.impl.h"

#include "sminmax.h"

#include "sdstr.decl.h"
#include "sdstr.impl.h"

#include "containers.impl.h"

#include "field.h"

#include "wall.decl.h"
#include "k/wvel.h" /* wall velocity used by sdf and wall */

#include "cnt.decl.h"
#include "k/cnt.h"
#include "cnt.impl.h"

#include "k/sdf.h"
#include "sdf.decl.h"
#include "sdf.impl.h"

#include "k/wall.h"
#include "wall.impl.h"

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
  #include "dev/sim.impl.h"
#else
  #include "hst/sim.impl.h"
#endif
#include "sim.impl.h"
#undef HST
#undef DEV
