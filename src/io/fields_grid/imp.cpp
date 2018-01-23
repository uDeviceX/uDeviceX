#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"

#include "coords/imp.h"

#include "inc/type.h"
#include "utils/msg.h"
#include "utils/error.h"

#include "d/api.h"

#include "io/field/imp.h"
#include "utils/cc.h"
#include "inc/dev.h"
#include "mpi/wrapper.h"

#include "imp.h"

/* body */
#if dump_all_fields
  #include "imp/all.h"
#else
  #include "imp/solvent.h"
#endif
