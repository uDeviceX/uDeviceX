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

#include "algo/dev.h"

#include "imp.h"

namespace den_dev {
#include "dev/main.h"
}

#include "imp/type.h"
#include "imp/main.h"
#include "imp/map.h"

