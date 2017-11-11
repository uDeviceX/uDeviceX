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
#include "utils/texo.dev.h"

#include "utils/mc.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"

#include "io/off.h"
#include "io/restart.h"

#include "rbc/type.h"
#include "imp.h"

namespace rbc { namespace force {

namespace dev {
#if   defined(RBC_PARAMS_TEST)
   #include "params/test.h"
#elif defined(RBC_PARAMS_LINA)
   #include "params/lina.h"
#else
   #error RBC_PARAMS_* is undefined
#endif
#include "dev/forces.h"
#include "dev/main.h"
}

#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/forces.h"

}} /* namespace */
