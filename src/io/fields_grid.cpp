#include <stdlib.h>
#include <math.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/type.h"

#include "d/api.h"

#include "field.h"
#include "utils/cc.h"
#include "inc/dev.h"

#include "fields_grid.h"

/* body */
#if DUMP_ALL_FIELDS
  #include "fields_grid/all.h"
#else
  #include "fields_grid/solvent.h"
#endif
