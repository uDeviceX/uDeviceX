/* [te]xture macros */

#if   defined(TE_TRACE)
  #include "te/trace.h"
#elif defined(TE_RELEASE)
  #include "te/release.h"
#else
  #error TE_* is undefined
#endif
#include "te/common.h"

/* TODO: header includes header */
