#include <assert.h>
#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "utils/msg.h"
#include "utils/cc.h"

#include "utils/kl.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "sum/imp.h"
#include "stat/imp.h"

#include "imp.h"

/* common part */
namespace dev {
#include "dev/dec.h"
#include "dev/util.h"
#include "dev/main0.h"
}
#include "imp/common.h"


/* polymorphic part */
namespace color {
namespace dev {
#include "dev/color/map.h"
#include "dev/main.h"
}
#include "imp/main0.h"
#include "imp/color/main.h"
}

namespace grey {
namespace dev {
#include "dev/grey/map.h"
#include "dev/main.h"
}
#include "imp/main0.h"
#include "imp/grey/main.h"
}


