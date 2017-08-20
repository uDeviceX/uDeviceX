#include <assert.h>
#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/ker.h"
#include "common.h"
#include "msg.h"
#include "cc.h"

#include "basetags.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "texo.h"

#include "inc/tmp/pinned.h"

#include "glb.h"
#include "rbc/imp.h"
#include "rbc/int.h"

namespace rbc {
#include "params/rbc.inc0.h"
#include "rbc/int/imp.h"
}
