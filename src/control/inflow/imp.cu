#include <stdio.h>
#include <curand_kernel.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "utils/halloc.h"
#include "utils/cc.h"
#include "utils/kl.h"
#include "d/api.h"

#include "inc/dev.h"
#include "inc/type.h"

#include "math/dev.h"

#include "cloud/imp.h"

#include "imp.h"

#include "dev/common.h"

namespace plate {
#include "plate/type.h"
#include "plate/dev.h"
#include "dev/main.h"
}

#include "imp/type.h"

#include "plate/imp.h"

#include "imp/main.h"
