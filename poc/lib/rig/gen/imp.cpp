#include <mpi.h>
#include <string.h>
#include <vector_types.h>

#include "inc/conf.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "inc/def.h"

#include "utils/error.h"
#include "utils/msg.h"
#include "utils/imp.h"
#include "utils/mc.h"
#include "utils/cc.h"

#include "d/api.h"
#include "mpi/wrapper.h"

#include "math/linal/imp.h"
#include "io/mesh_read/imp.h"
#include "coords/imp.h"
#include "comm/imp.h"
#include "exch/mesh/imp.h"
#include "algo/minmax/imp.h"
#include "mesh/triangles/type.h"
#include "mesh/collision/imp.h"
#include "mesh/props/imp.h"
#include "mesh/dist/imp.h"
#include "frag/imp.h"

#include "rigid/imp.h"

#include "imp.h"

#define _I_ static
#define _S_ static

#include "imp/common.h"
#include "imp/template.h"
#include "imp/kill.h"
#include "imp/props.h"
#include "imp/main.h"

#undef _I_
#undef _S_
