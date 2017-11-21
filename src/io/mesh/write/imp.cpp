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

namespace write {
#include "imp/main.h"
}
