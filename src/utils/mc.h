/* [m]pi [c]heck macros */

#if    defined(MC_RELEASE)
  #include "mc/release.h"
#else
  #error MC_* is undefined
#endif

#include "mc/common.h"

/* TODO: header includes header */
