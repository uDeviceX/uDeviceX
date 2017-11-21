#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"
#include "utils/os.h"
#include "utils/error.h"
#include "utils/halloc.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "inc/type.h"
#include "utils/mc.h"

#include "imp.h"

#if   defined(MESH_SHIFT_EDGE)
   #include "imp/shift/edge.h"
#elif defined(MESH_SHIFT_CENTER)
   #include "imp/shift/center.h"
#else
   #error     MESH_SHIFT_* is undefined
#endif
namespace write {
#include "imp/write.h"
}
#include "imp/main.h"
