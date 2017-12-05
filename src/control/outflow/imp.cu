#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "utils/halloc.h"
#include "utils/cc.h"
#include "utils/kl.h"
#include "d/api.h"

#include "math/dev.h"
#include "mpi/glb.h"

#include "inc/dev.h"
#include "inc/type.h"

#include "imp.h"

namespace circle {
#include "circle/dev.h"
#include "dev/filter.h"
}

namespace plane {
#include "plane/dev.h"
#include "dev/filter.h"
}

#include "imp/type.h"
#include "imp/main.h"

#include "circle/imp.h"
#include "plane/imp.h"
