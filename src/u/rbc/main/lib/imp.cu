#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "utils/imp.h"

#include "d/api.h"
#include "utils/msg.h"

#include "inc/def.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "utils/te.h"
#include "utils/texo.h"
#include "utils/cc.h"

#include "glob/type.h"

#include "rbc/type.h"
#include "rbc/force/area_volume/imp.h"
#include "rbc/main/imp.h"
#include "rbc/rnd/imp.h"
#include "rbc/force/imp.h"
#include "rbc/stretch/imp.h"

#include "scheme/move/imp.h"
#include "scheme/force/imp.h"

#include "io/mesh/imp.h"
#include "io/diag/imp.h"

#include "mpi/wrapper.h"
#include "mpi/glb.h"

#include "imp.h"

namespace stretch {
#if RBC_STRETCH==true
  #include "imp/stretch1.h"
#else
  #include "imp/stretch0.h"
#endif
}

#include "imp/main.h"
