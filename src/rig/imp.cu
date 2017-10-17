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

#include "generate/rig/imp.h"

#include "rigid/imp.h"

namespace rig {

#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/common.h"
#include "imp/generate.h"
#include "imp/start.h"

} // rig
