/* [c]uda [c]heck macros */

#ifdef CC_SYNC
  #include "cc/sync.h"
#else
  #include "cc/release.h"
#endif
#include "cc/common.h"

/* TODO: header includes header */
