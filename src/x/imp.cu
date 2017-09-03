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

#include "forces/type.h"
#include "forces/imp.h"

#include "sdf/type.h"
#include "sdf/int.h"

#include "wall/int.h"

#include "flu/int.h"

#include "odstr/type.h"
#include "odstr/int.h"
#include "cnt/int.h"
#include "fsi/int.h"

#include "x/type.h"
  #include "rex/type/remote.h"
  #include "rex/type/local.h"
  #include "rex/decl.h"

  #include "rex/dev/type.h"
  #include "rex/dev/decl.h"
  #include "rex/dev/common.h"
  #include "rex/dev/x.h" /* TODO */
  #include "rex/dev/pack.h"
  #include "rex/dev/scan.h"
  #include "rex/dev/scatter.h"
  #include "rex/dev/unpack.h"

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
#include "x/imp.h"
