#include <stdio.h>
#include <curand_kernel.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "utils/imp.h"
#include "utils/cc.h"
#include "utils/kl.h"
#include "d/api.h"

#include "inc/dev.h"
#include "inc/type.h"

#include "coords/type.h"
#include "coords/imp.h"
#include "math/dev.h"

#include "imp.h"

#include "plate/type.h"
#include "plate/dev.h"

#include "circle/type.h"
#include "circle/dev.h"


#include "imp/type.h"

namespace inflow_dev {
#include "dev/ini.h"
#include "dev/main.h"
}

#include "plate/imp.h"
#include "circle/imp.h"

#include "imp/main.h"
