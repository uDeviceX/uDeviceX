#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"

#include "glob/type.h"
#include "glob/imp.h"

#include "inc/type.h"
#include "utils/msg.h"
#include "utils/error.h"

#include "d/api.h"

#include "field/imp.h"
#include "utils/cc.h"
#include "inc/dev.h"
#include "mpi/wrapper.h"

#include "fields_grid.h"

/* body */
#if dump_all_fields
  #include "fields_grid/all.h"
#else
  #include "fields_grid/solvent.h"
#endif
