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

#include "inc/tmp/wrap.h"
#include "inc/tmp/pinned.h"
#include "sim/int.h"

#include "dbg/imp.h"

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

#include "cloud/hforces/type.h"
#include "fsi/type.h"
#include "fsi/int.h"

#include "int/type.h" /* TODO */
#include "imp/type.h"
#include "imp/decl.h"

namespace dev {
  #include "dev/type.h"
  #include "dev/decl.h"
  #include "dev/common.h"
  #include "dev/x.h" /* TODO */
  #include "dev/pack.h"
  #include "dev/scan.h"
  #include "dev/scatter.h"
  #include "dev/unpack.h"
} /* namespace */

#include "imp/ini.h"
#include "imp/copy.h"
#include "imp/wait.h"
#include "imp/scan.h"
#include "imp/pack.h"
#include "imp/send.h"
#include "imp/recv.h"
#include "imp/unpack.h"
