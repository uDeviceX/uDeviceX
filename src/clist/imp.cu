#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"
#include "inc/def.h"
#include "msg.h"
#include "utils/cc.h"
#include "utils/kl.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "algo/scan/int.h"
#include "clist/imp.h"

namespace clist {
namespace dev {
#include "dev.h"
}
#include "imp/main.h"
}
