#include <mpi.h>

#include <conf.h> 
#include "inc/conf.h"

#include "frag.h"

#include "utils/mc.h"
#include "utils/cc.h"
#include "d/api.h"
#include "mpi/wrapper.h"

#include "imp.h"

// imp
namespace comm {
#include "imp/ini.h"
#include "imp/fin.h"
#include "imp/main.h"
} // comm
