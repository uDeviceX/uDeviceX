#include <conf.h>
#include "inc/conf.h"
#include "mpi/glb.h"
#include "d/api.h"

#include "imp.h"

/* globals for all kernels */
namespace glb {
#include "imp/dec.h"
#include "imp/main.h"
}
