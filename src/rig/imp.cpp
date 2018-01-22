#include <mpi.h>
#include <stdio.h>
#include <vector_types.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"

#include "coords/type.h"
#include "coords/imp.h"

#include "d/api.h"
#include "inc/dev.h"
#include "inc/def.h"
#include "inc/type.h"
#include "utils/cc.h"

#include "utils/msg.h"
#include "io/ply/imp.h"
#include "io/restart/imp.h"

#include "imp.h"

#include "generate/rig/imp.h"

#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/common.h"
#include "imp/generate.h"
#include "imp/start.h"
