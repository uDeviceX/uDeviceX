#include <assert.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "utils/imp.h"
#include "utils/error.h"
#include "inc/dev.h"
#include "utils/kl.h"
#include "utils/cc.h"
#include "inc/type.h"
#include "d/api.h"

#include "math/dev.h"

#include "imp.h"
#include "imp/type.h"

/* conf */
enum {MAX_COL = 4};

#include "dev/type.h"
#include "bbstates.h"
#include "dev/roots.h"
#include "dev/utils.h"
#include "dev/cubic_root/main.h"
#ifdef MESHBB_LOG_ROOTS
  #include "dev/cubic_root/log_root1.h"
#else
  #include "dev/cubic_root/log_root0.h"
#endif
#include "dev/intersection.h"
#include "dev/collect.h"
#include "dev/main.h"

#ifdef MESHBB_LOG_ROOTS
  #include "imp/find_collisions/log_root1.h"
#else
  #include "imp/find_collisions/log_root0.h"
#endif
#include "imp/main.h"

