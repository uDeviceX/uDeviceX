/* TODO: header includes header */
#include "kl/common.h"
#include "kl/macro.h"

#if    defined(KL_RELEASE)
  #include "kl/release.h"
#elif  defined(KL_TRACE)
  #include "kl/trace.h"
#elif  defined(KL_PEEK)
  #include "kl/peek.h"
#elif  defined(KL_SAFE)
  #include "kl/safe.h"
#endif
