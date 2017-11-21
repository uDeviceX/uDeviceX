#include <vector_types.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include "inc/conf.h"
#include "utils/os.h"
#include "utils/error.h"
#include "utils/halloc.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "inc/type.h"
#include "io/mesh.h"

#include "utils/mc.h"

#if   defined(MESH_SHIFT_EDGE)
   #include "mesh/shift/edge.h"
#elif defined(MESH_SHIFT_CENTER)
   #include "mesh/shift/center.h"
#else
   #error     MESH_SHIFT_* is undefined
#endif
#include "mesh/main.h"
