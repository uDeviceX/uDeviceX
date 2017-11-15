#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <curand.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/os.h"
#include "msg.h"
#include "mpi/glb.h"
#include "d/api.h"
#include "inc/dev.h"

#include "utils/cc.h"

#include "imp.h"
#include "type.h"

namespace rbc { namespace rnd {
#include "imp/macro.h"
#include "imp/seed.h"
#include "imp/main.h"
}} /* namespace */
