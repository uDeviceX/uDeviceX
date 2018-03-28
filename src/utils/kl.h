/* [k]ernel [l]aunch macros */

#if    defined(KL_RELEASE)
  #include "kl/release.h"
#elif  defined(KL_NONE)
  #include "kl/none.h"
#elif  defined(KL_SYNC)
  #include "kl/sync.h"
#else
  #error KL_* is undefined
#endif

#include "kl/common.h"
#include "kl/macro.h"

/* TODO: header includes header */
