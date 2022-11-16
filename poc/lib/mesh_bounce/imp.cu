#include <assert.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "utils/imp.h"
#include "utils/error.h"
#include "inc/dev.h"
#include "utils/kl.h"
#include "utils/cc.h"
#include "inc/type.h"
#include "d/api.h"

#include "math/dev.h"

#include "imp.h"
#include "imp/type.h"

/* conf */
enum {MAX_COL = 4};

#include "bbstates.h"

#define _I_ static __device__
#define _S_ static __device__

namespace mesh_bounce_dev {
#include "dev/type.h"
#include "dev/roots.h"
#include "dev/utils.h"
#include "dev/intersection.h"
#include "dev/collect.h"
#include "dev/main.h"
}

#include "imp/main.h"

#undef _I_
#undef _S_
