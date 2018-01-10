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

#include "plate/type.h"
#include "plate/dev.h"

#include "circle/type.h"
#include "circle/dev.h"


#include "imp/type.h"

namespace dev {
#include "dev/ini.h"
#include "dev/main.h"
}

#include "plate/imp.h"
#include "circle/imp.h"

#include "imp/main.h"
