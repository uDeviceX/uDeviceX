#include <mpi.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "utils/msg.h"

#include "inc/type.h"
#include "sub/imp.h"
#include "imp.h"

#if   defined(RESTRAIN_NONE)
  #include "imp/none.h"
#elif defined(RESTRAIN_RED_VEL)
  #include "imp/red_vel.h"
#elif defined(RESTRAIN_RBC_VEL)
  #include "imp/rbc_vel.h"
#else
  #error RESTRAIN_* is undefined
#endif
