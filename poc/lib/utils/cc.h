/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
  #include "cc/sync.h"
#elif defined(CC_RELEASE)
  #include "cc/release.h"
#else
  #error CC_* is undefined
#endif

#include "cc/common.h"

/* TODO: header includes header */
