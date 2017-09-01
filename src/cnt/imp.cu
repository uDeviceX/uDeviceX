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
#include "frag.h"

#include "utils/kl.h"
#include "mpi/basetags.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "utils/texo.h"
#include "utils/te.h"

#include "inc/tmp/wrap.h"
#include "inc/tmp/pinned.h"
#include "dbg/imp.h"
#include "glb.h"

#include "k/read.h"
#include "k/write.h"
#include "k/common.h"
#include "k/index.h"

#include "clist/int.h"
#include "forces/type.h"
#include "forces/pack.h"
#include "forces/imp.h"

namespace cnt {
namespace dev {
#include "dev/decl.h"
#include "dev/fetch.h"
#include "dev/code.h"
#include "dev/bulk.h"
#include "dev/halo.h"
#include "dev/pop.h"
}


#include "cnt/decl.h"
#include "cnt/bind.h"
#include "cnt/build.h"
#include "cnt/bulk.h"
#include "cnt/fin.h"
#include "cnt/halo.h"
#include "cnt/ini.h"
#include "cnt/setup.h"

} /* namespace */
