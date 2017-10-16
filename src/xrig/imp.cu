#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "inc/dev.h"
#include "inc/def.h"
#include "inc/type.h"
#include "utils/cc.h"

#include "msg.h"
#include "io/ply.h"
#include "io/restart.h"

#include "imp.h"

/* TODO clear this mess (includes belong to ic) */
#include <mpi.h>
#include <vector>
#include "mpi/wrapper.h"
#include "mpi/glb.h"
#include "mpi/type.h"
#include "utils/mc.h"

#include "d/ker.h"
#include "utils/texo.h"
#include "mesh/collision.h"
#include "mesh/dist.h"
#include "mesh/bbox.h"

#include "rigid/int.h"

namespace rig {

/* TODO clear this mess */
#include "imp/ic2.h"
#include "imp/ic1.h"
#include "imp/ic0.h"


#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/common.h"
#include "imp/generate.h"
#include "imp/start.h"

} // rig
