#include <stdio.h>
#include <mpi.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"
#include "utils/msg.h"

#include "utils/error.h"
#include "utils/imp.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "d/api.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"
#include "utils/cc.h"
#include "utils/kl.h"
#include "utils/mc.h"

#include "math/dev.h"
#include "glob/type.h"
#include "glob/dev.h"

#include "imp.h"

namespace dev {
#include "dev/common.h"

#if   defined(VCON_CART)
#include "dev/cart.h"
#elif defined(VCON_RAD)
#include "dev/radial.h"
#else
#error VCON_* transformation undefined
#endif

#include "dev/sample.h"
} // dev

#include "imp/type.h"
#include "imp/main.h"
