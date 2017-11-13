#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector_types.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "msg.h"

#include "d/api.h"
#include "utils/cc.h"

#include "utils/mc.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"

#include "io/off.h"
#include "io/restart.h"

#include "rbc/type.h"
#include "rbc/adj/type.h"
#include "rbc/adj/imp.h"

#include "imp.h"

namespace rbc { namespace main {
#include "imp/ini.h"
#include "imp/fin.h"

#include "imp/setup.h"
#include "imp/generate.h"
#include "imp/start.h"

}} /* namespace */
