#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"
#include "inc/def.h"
#include "utils/msg.h"
#include "utils/cc.h"
#include "utils/error.h"
#include "utils/imp.h"

#include "d/q.h"
#include "d/ker.h"
#include "d/api.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "sdf/def.h"

#include "utils/kl.h"

#include "sdf/type.h"
#include "sdf/imp.h"
#include "sdf/dev.h"

#include "imp.h"

namespace label {
namespace dev0 {
#include "dev/main.h"
}
#include "imp/main.h"
}
