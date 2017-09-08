#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"
#include "inc/def.h"
#include "msg.h"
#include "utils/cc.h"

#include "d/ker.h"
#include "d/api.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "utils/texo.h"

#include <stdint.h>
#include "rnd/imp.h"

#include "sdf/type.h"
#include "sdf/int.h"

#include "clist/int.h"

#include "forces/type.h"
#include "cloud/hforces/type.h"

#include "imp.h"
#include "int.h"

namespace wall {
#include "int/main.h"

/*** polymorphic ***/
namespace grey {
  #include "int/pair.h"
}
namespace color {
  #include "int/pair.h"
}

} /* wall */
