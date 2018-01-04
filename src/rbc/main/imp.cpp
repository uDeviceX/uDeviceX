#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector_types.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "glob/type.h"

#include "utils/msg.h"

#include "d/api.h"
#include "utils/error.h"
#include "utils/imp.h"
#include "utils/cc.h"

#include "utils/mc.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"

#include "io/off/imp.h"
#include "io/restart.h"

#include "rbc/type.h"
#include "rbc/adj/type/common.h"
#include "rbc/adj/type/hst.h"
#include "rbc/adj/imp.h"
#include "rbc/gen/imp.h"

#include "anti/imp.h"

#include "imp.h"

namespace rbc { namespace main {
#include "imp/util.h"
#include "imp/ini.h"
#include "imp/fin.h"

#include "imp/setup.h"
#include "imp/generate.h"
#include "imp/start.h"

}} /* namespace */
