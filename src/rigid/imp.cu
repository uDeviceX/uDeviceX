#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"
#include "utils/msg.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "utils/cc.h"
#include "d/api.h"
#include "utils/kl.h"

#include "imp.h"

enum {XX, XY, XZ, YY, YZ, ZZ};
enum {YX = XY, ZX = XZ, ZY = YZ};

namespace dev {
#include "dev/utils.h"
#include "dev/main.h"
}

#include "imp/main.h"
