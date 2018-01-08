#include <stdio.h>
#include <curand_kernel.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"

#include "utils/error.h"
#include "utils/imp.h"
#include "utils/cc.h"
#include "utils/kl.h"
#include "d/api.h"

#include "inc/dev.h"
#include "inc/type.h"

#include "glob/type.h"
#include "glob/imp.h"
#include "math/dev.h"

#include "cloud/imp.h"

#include "imp.h"

namespace dev {
#include "dev/common.h"
}

namespace plate {
#include "plate/type.h"
#include "plate/dev.h"
}

namespace circle {
#include "circle/type.h"
#include "circle/dev.h"
}

#include "dev/main.h"

#include "imp/type.h"

#include "plate/imp.h"
#include "circle/imp.h"

#include "imp/main.h"
