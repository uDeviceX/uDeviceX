#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include <conf.h>

#include "inc/conf.h"
#include "inc/def.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "math/dev.h"

#include "msg.h"

#include "d/api.h"
#include "d/ker.h"
#include "utils/cc.h"
#include "utils/kl.h"
#include "utils/te.h"
#include "utils/texo.h"

#include "mpi/wrapper.h"
#include "mpi/glb.h"

#include "io/off.h"
#include "io/restart.h"

#include "imp.h"

namespace rbc {

#include "params/rbc.inc0.h"

#include "dev/forces.h"
#include "dev/main.h"

#include "imp/ini.h"
#include "imp/fin.h"

#include "imp/setup.h"
#include "imp/generate.h"
#include "imp/start.h"

#include "imp/main.h"

} // rbc
